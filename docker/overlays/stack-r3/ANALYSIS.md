# stack-r3: composition analysis

Written 2026-09-24 by the stack-r3 Opus sub-agent; saved by the operator from its final report (the harness refused the agent's .md writes). Base is `tabbyapi:stack-r2`, the daily since R701 (slotfix-r1 + hcfast-r1 + moefast-r1, 27 EXTRA_ENV keys). Every check below ran on a stack-r2 tree rebuilt locally: the pristine served package plus hcfast-r1.patch and moefast-r1.patch, applied at fuzz 0. Nothing here ran on a GPU.

## 1. What goes into the image

| # | patch | class | source | what it carries |
| --- | --- | --- | --- | --- |
| 01 | 01-hcfast-r2-on-stack-r2.patch | core | derived | hcfast r1 → r2 (EXL3_HC_MIX_V3=2, per-R tables, hc_mix_v3_revision 2) |
| 02 | 02-latchain-r1b.patch | core | derived | latchain r1 (RR, QF, GF, QT). One line re-anchored, code identical |
| 03 | 03-moefast-r3.patch | opt mf3 | verbatim R713 | moefast r2 kernels + r3 shared-expert overlap levers |
| 04 | 04-densegemm-r1.patch | opt dg1 | verbatim R710b | V2 twins; mode 3 = gemv twin only |
| 05 | 05-densegemm-r2.patch | opt dg2 | verbatim R714 | r2 twins (fold after the main loop, no spills), needs DGV2_NVCC_DEFS |
| 06 | 06-dense-lcguard.patch | auto with dg1/dg2 | new | densegemm scratch for dense calls on latchain's QSA side branch |

series/SERIES pins every patch's sha256. apply-series.sh applies the core patches plus the INCLUDE set at --fuzz=0 and fails on any .rej. There is one rebuild.

## 2. Composition audit (pairwise, fuzz 0, on the reconstructed stack-r2 tree)

series/subset-matrix.txt is the full output of check-subsets.sh. All six subsets of {mf3} × {none, dg1, dg2} apply. For each subset, applying the same patches in a permuted order (optional patches first, core last) gives a byte-identical tree, so no patch depends on another's context lines. Each tree has exactly one bindings line per revision marker, with the expected value.

Blockers found and fixed:

1. **latchain-r1 × moefast-r3, bindings.cpp.** Both patches anchor on the `m.attr("moe_coop_v3_revision") = 1;` line. moefast-r3 rewrites it to 3, so latchain-r1 rejects one hunk in either order (reproduced both ways; the matrix keeps both as negative controls). latchain-r1b moves its `m.attr("latchain_revision") = 1;` below `m.def("bighead_attn_workspace_size", ...)`, whose context no other patch touches. The non-bindings hunks are byte-identical to latchain-r1. The standalone install relaxes its moefast assert from `== 1` to `>= 1`. latchain-r1b also applies alone on stack-r2 (dry run, fuzz 0).
2. **hcfast-r2 is cumulative over slotfix-r1.** Its install refuses a base that already has hc_mix_v3.cu, and stack-r2 has one. Patch 01 is diff(stack-r2, src + hcfast-r2 + moefast-r1). That tree does not depend on application order, and it is exactly R702's measured source.
3. **HC2 env vs the served HC1 env.** In r2, EXL3_HC_MIX_V3_DOTS_B and _UP_B are per-R tables. The served DOTS_B=2 UP_B=8 would override R702's tables if they stayed in EXTRA_ENV. Every stack-r3 env is therefore built as "served env minus the families the config sets, plus the config". The launcher diff does the same by replacing the whole EXTRA_ENV default, not by using EXTRA_ENV_ADD.
4. **QF × densegemm (the R712 / R710b race).** See §3.
5. **moefast-r3's shared-expert side stream vs the QF fork window.** This was R713's condition. start_shared forks the shared expert before the router. The router is an fp16 Linear (qmap=None), not a dense EXL3 call. shared_done is waited on before routed stage B, inside exl3_moe_coop_run. The shared stream is therefore idle before the next layer's attention fork, and the QF fork and the shared stream never overlap three ways. On the shared stream, shared.down runs with lock offset 0, so the gemv twin counts in the lower hctr half. No main-stream dense twin runs in that window: the router is fp16, and routed stage A is the MoE coop kernel, not densegemm. Part C of test_dense_lcguard_cpu.py models that window for modes 1-3 at 4 and 16 rows and finds no shared array.
6. **Not carried:** GF (R712: marginal inside ALL ≈ 0); rows32-r3 and pdl-l2-r1, which are not in the series; EXL3_DENSE_ROWS32 (the gate aborts if it is on).

## 3. The densegemm side-branch guard (06-dense-lcguard.patch)

**Race.** QF runs the QSA indexer (index_qk, an exl3_gemm_gr) as a parallel CUDA-graph branch. latchain gives that call a separate lock region: lc_gemm_locks_offset = LC_LOCKS_ALT_OFFSET (MAX_TILES_C/2), set through exl3_gemm_set_locks_offset and reset by RAII. densegemm's V2 twins have one scratch per device: g_ws.slots for the gemm/mgemm twins and g_ws.hctr (16384 counters) for the gemv twin. Neither has a per-stream copy, so a twin on the side branch and a twin on the main branch could share scratch in the same graph window.

**Detection.** The locks pointer is kernel argument 6 in both exl3_gemm and the gemv path ("same kernel arguments"). It equals DevCtx::get_locks(device) + lc_gemm_locks_offset. dense_v2_on_side_branch() compares that argument with get_locks(device). A non-zero offset is exactly latchain's side-branch call. The lints check that it is the only non-zero set_locks_offset call in the tree.

This is equivalent to the getter the R710b review proposed (locks_arg − base == lc_gemm_locks_offset), with two advantages:
- the densegemm files need no latchain symbol, so the guard compiles on a densegemm-only image;
- latchain-r1b stays code-identical to the measured latchain-r1.

**Action.** A side-branch call uses its own scratch:
- The gemv twin counts in the upper half of hctr (hctr + DENSE_V2_GEMV_MAX_HBLOCKS/2). The main branch keeps the lower half. The block bound becomes H/2. The largest served gemv call is 20 blocks (n 2560) and index_qk is 5 blocks, so every real shape fits.
- The gemm twin writes a separate side slot region of 2 MiB. It is allocated only when a gemm twin can run (mode 1/2 or rows32) and EXL3_LC_QSA_FORK=1. A side call that does not fit takes the served kernel and is counted.
- dense_v2_launch_mgemm is unchanged, because exl3_mgemm_gr never runs on the side branch.
- The first side call on each device writes one line to stderr ("densegemm lcguard: device N: ..."), and dense_v2_lc_side_counts() exposes the counters. The gate uses both to show that the side path was exercised.

**Why it is bitwise-neutral.** The same kernels run with the same grids, tiles and arithmetic. Only the scratch addresses change, and scratch is zeroed and consumed within one call. On the main branch, the one textual change is the gemv bound (H → H/2), which no served shape comes near.

test_dense_lcguard_cpu.py passes on dg1 and dg2 trees:
- **Part A** compiles code extracted from the tree. The workspace byte ranges are pairwise disjoint in all 16 mode × rows32 × fork configurations.
- **Part B** is source lints. It includes a diff against the unguarded tree: only the scratch-selection lines change, and mgemm is byte-identical.
- **Part C** is the resource model:
  - Without the guard, modes 1/2 at 16 rows share slots in the QF window, and an unfused q/k/v configuration shares hctr. The model therefore does detect races.
  - With the guard, no configuration shares an array.

**The other direction (main branch during the fork window).** In the served config the main branch runs the fused qkv exl3_mgemm_gr in that window (plus deinterleave, rope and cache append). The mgemm twin only exists in modes 1/2 and uses slots.
- Mode 3 (dg1): no main-branch twin runs in the window, so the side call has hctr to itself even without the guard. That is an accident of today's kernel inventory. The guard makes it a property of the code.
- Modes 1/2 at 16 rows: the mgemm on main and the side gemm would share slots without the guard. With it, they use separate regions.

## 4. Headroom

- densegemm allocates its workspace lazily, only when a dense flag is on: slots 4 MiB (8 MiB with rows32), plus 64 KiB of hctr, plus 2 MiB of side slots only in modes 1/2 with QF. R710b measured UP-line free at −6 / −8 MiB for mode 3.
- latchain: RR and QT change kernels, not buffers. QF adds graph branch nodes.
- HC2 is tables only.
- moefast-r3 (if included) adds event objects.
- Expected UP-line free: about −6 to −10 MiB against the stack-r2 boot, inside the REF − 32 MiB bar. With flags off, image C should match A.

## 5. Expected gain (sum of the per-component results, stall-excluded where available; ms/iterate, arm − OFF)

| component | c1d3 | c4d3 | c8d1 | source |
| --- | --- | --- | --- | --- |
| HC2 | −0.15 (5/5) | −0.64 mixed | +0.06 mixed | R702 (raw; no stall-excluded read) |
| RR | −0.14 | −0.50 | −0.21 | R712 |
| QT | −0.12 | −0.13 | −0.24 | R712 |
| QF | −0.30 | −0.15 mixed | −0.10 mixed | R712 |
| GV (dg1 mode 3) | −0.37 | −0.17 (noise: no engaged call) | +0.04 flat | R710b |
| sum | −1.08 | −1.59 (−0.78 without the two mixed/noise cells) | −0.45 | |

Caveats:
- The parts do not add up. R712 ALL (RR+QF+GF+QT) measured −0.56 / −0.76 / −0.64, which is about 80 % of its parts' sum at c1d3/c4d3 and about 93 % at c8d1.
- QF and GV may also overlap. If index_qk engages the gemv twin, QF moves it off the critical path, so part of GV's c1 saving on that call is already counted in QF.
- Realistic range: c1d3 −0.8 to −1.1 ms (−6 to −9 % of ~12.8 ms), c4d3 −0.8 to −1.4 ms (−4 to −7 % of ~20 ms), c8d1 −0.4 to −0.7 ms (−2 to −4 % of ~18.5 ms).
- Served decode moves by roughly 1/(1 − x). c1 is noisy boot to boot (±20 % single-stream), so read the fn_gate it_ms column next to the t/s ratio.
- GV is a c1/c2 lever only: the 16-row verify at c4d3/c8d1 never takes the gemv.
- moefast-r3 (R713) and densegemm-r2 (R714) have no STACK rows yet. r2 kernels were flat under lost shared-expert overlap (R703b), and r3 exists to recover that overlap.

## 6. Which UNION to run

The pre-registered default is INCLUDE=dg1, UNION = HC2 RR QT QF GV (EXL3_DENSE_V2=3). This was the configuration in the launcher diff that came with the series. The served configuration is INCLUDE `mf3 dg2` with `EXL3_DENSE_V2=1` (R714, R716b). It is safe only because of the guard, and the gate aborts QF + a dense mode on an image without it.

| R713 (moefast-r3) | R714 (densegemm-r2) | INCLUDE | DG_ENV | MF3_ENV |
| --- | --- | --- | --- | --- |
| not accepted | not accepted | dg1 | EXL3_DENSE_V2=3 (default) | empty (default) |
| ACCEPT | not accepted | mf3 dg1 | default | R713's R3 line (default EXL3_MOE_COOP_V3=3 EXL3_MOE_COOP_V3_MAP=2-4:2 EXL3_SHARED_EXPERT_EARLY=1) |
| any | ON (mode 1) ACCEPT | dg2 (+ mf3 if R713 ACCEPT) | EXL3_DENSE_V2=1 (default for dg2) | as above |
| any | GM (mode 2) ACCEPT, ON not | dg2 only if GM's c4d3+c8d1 gain > GV's 0.37 ms at c1d3, then DG_ENV=EXL3_DENSE_V2=2; otherwise stay on dg1 | | |

With dg2, also set DGV2_NVCC_DEFS to R714's BISECT-CHOICE defines (the image label local.densegemm.defs). Always set DG_ENV from the arm R714 accepted, not from the default.

Fallback without QF: UNION_PRESET=noqf. Use it if the image cannot carry the guard, if the served LOO boots name QF, or if the operator wants a QF-free batch. The gate pads its P1 arms to 6 with UnoRR so it still runs 6 rounds. If P1 marks exactly one optional component DROP, the gate itself serves the union without it.

## 7. GPU time

- Build: 15-25 min CPU inside the lock, while the daily keeps serving.
- Kernel parity: 20-30 min.
- Model parity: about 15 min (9 runs).
- P1: 108 runs, about 50-55 min (7 arms × 7 rounds = 147 runs, about 70 min, with mf3).
- Served: 7 boots × 17-20 min, about 2 h.
- Total about 3.5-4.5 h. A B1 identity divergence adds one A/A boot and up to 5 LOO greedy boots (about 8 min each, no fn_gate). The unit then stops before the ABAB.
