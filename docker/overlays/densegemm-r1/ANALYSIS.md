# densegemm r1: dense EXL3 K=4 decode GEMMs, bitwise-identical V2 twins

Written 2026-09-24 by the dense-GEMM Opus sub-agent; saved by the operator from its final report (the harness refused the agent's .md writes). Nothing here was compiled or run on a GPU: the workstation has no CUDA toolchain. Every µs figure is either sourced or labelled *estimate*. The operator's build is the first compile; P0 and P1 replace the estimates. Queued as `flan/r710-densegemm-gate.sh`.

## 1. What runs today (served inventory, stack-r2)

Source: R680 nsys traces read with `out-roofline/tools/kgrid.py`, and R682 d0. Calls per step count the target's 36 GDN layers and 12 attention layers.

| projection | c1d3 (4 rows) | c4d3 / c8d1 (16 rows) | calls/step |
|---|---|---|---|
| GDN in_proj (qkv+z, sliced) | mgemm shape 4 (TK16 TN512, 256 thr), (20,1,8), 24.8 µs | mgemm shape 3 (TK32 TN256, 512 thr), (20,1,8), 26 µs | 36 |
| attention q/k/v (sliced) | mgemm shape 3, (6,1,26), ~22 µs | (6,1,26), ~23 µs | 12 |
| GDN out_proj | `exl3_gemv_kernel<4,1,2,1,0,0>` (80,1), 17.2 µs med | gemm shape 2 (TK32 TN128), (120,1,1), 18.75 µs | 36 |
| attention o_proj | gemv (80,1), ~17 µs | gemm (120,1,1), 18.75 µs | 12 |
| index_qk_proj | gemv (20,1), ~10.5 µs | gemm (20,1,1), 12.1 µs | 12 |
| shared gate+up | mgemm shape 2, (30,1,2), ~13.6 µs | (28,1,2), 14.6 µs | 48, overlap stream |
| shared down | gemv (80,1), 6.3 µs min | gemm (40,1,1), 8.8 µs | 48, overlap stream |

Corrections to the brief:
- GDN out_proj at c1d3 is the QTIP `exl3_gemv_kernel`, not `exl3_gemv_int8_sq<4>`. The int8 GEMV (`EXL3_INT8_GEMV` mode 2, K ≤ 6 on Blackwell) takes rows ≤ 2 only: the MTP draft calls, which r1 does not touch.
- index_qk at 16 rows has one k-segment per column (grid 20 over 80×20 tiles), so there is no chain.
- in_proj and qkv at 16 rows have 2 handoffs, not 4-5.
- The shared-expert calls run on the overlap stream beside the routed MoE (R703b), so they are mostly off the critical path.

Critical-path dense K4 per step: c1d3 ≈ 36·24.8 + 12·22 + 48·17.2 + 12·10.5 ≈ 2.1 ms; c4d3 / c8d1 ≈ 36·26 + 12·23 + 48·18.75 + 12·12.1 ≈ 2.3 ms. That is 11-17 % of the 12.53 / 19.84 / 20.75 ms steps (c1 / c4 / c8, R653 via R657).

## 2. Why they reach 36-50 % of DRAM bandwidth

R682: **fixed per-call latency.** mgemm K4 intercept is 13.4-14.7 µs; the marginal rate is 85-88 % of DRAM. The ncu top stall is "barrier", with 16-33 % of warps active. gemm K4 at 1-4 rows is mixed (long_scoreboard, and that cell is the gemv kernel).

Where the fixed time goes (source):
1. **The input stage runs before any weight byte is requested:** suh + Hadamard of A into A_had, then `grid.sync()` (mgemm: a group barrier), then the first B `cp.async`. The first DRAM round trip sits behind a grid-wide barrier.
2. **Serialized stream-K fixup** (`exl3_gemm_inner.cuh` `reduce()`). Each segment spins until `lock == tiles_k - k_end - 1`, reads the running sum from C (rounded to C's dtype), adds its partial, writes C, then `__threadfence()` and lock bump: s−1 serialized L2 round trips after the compute. Served handoffs (`test_densegemm_cpu.py` §1): in_proj r4: 4; out/o_proj r16: 5; shared gate/up: 5-6; in_proj r16 and qkv: 2; shared down r16: 1; index_qk: 0.
3. **mgemm barriers the dense case doesn't need:** a group barrier after the sliced grid.sync, a per-chunk barrier, and a separate output-Hadamard pass behind another barrier.
4. **gemm's trailing `grid.sync()`.**
5. **gemv:** a register prefetch ring (`ld.global.cs` into `pf[PF][LOADS]`) issued with the activation loads; by analogy with moefast R700b, the hypothesis is a shared scoreboard, so the MMA waits on a same-iteration DRAM load (**hypothesis, not measured**; R682's long_scoreboard is consistent with it). Plus grid.sync and a full pass over C for the output Hadamard.

With one wave and 16-33 % of warps active, nothing hides items 1-4.

## 3. Levers and why each is bit-exact

Bit identity requires, per output element, the same MMA products, in-segment order, cross-segment order and rounding, and output transform. The partition (grid, TILESIZE_K/N) fixes the segments. **r1 reuses the served autotuned config for every call** (`CoopKernelAutotuner::peek` reads the served cache entry; V2 runs only on a hit; a shape's first call runs and tunes the served kernel).

Rejected as not bit-exact: per-shape tile tables or more warps per tile for gemm/mgemm (partition change), a different gemv k-split (reduction order), atomic split-K (non-deterministic). Not pursued: PDL / L2 prefetch (R705 flat), chain fusion (R684 ~0).

- **A. Gathered fixup.** A non-head segment always starts its CTA's range, so one slot per CTA suffices (CPU model, 308 configurations). Each such segment stores its fp32 partial (`__stcg`) and does `barrier_release(lock,1)`. The head waits once for `nprod`, then folds S = P_top; S = P_b + rnd(S) … own + rnd(S) (rnd = fp16 round trip for fp16 C, identity for fp32), then resets the lock. The CPU model matches the chain's order and bits for the 8 served configs plus 300 random ones.
- **B. Weight prologue.** B's pipeline stages are issued before the Hadamard + barrier (the `pre_a` hook); A comes after. The inner loop is copied verbatim.
- **C. Barrier trimming.** gemm: no trailing grid.sync. mgemm (dense only: no indices/weights/filter, one token): no redundant group barrier, no per-chunk barrier; the per-matrix barrier stays because A_had is reused. The finalizing CTA applies `had_ff` / `had_fh` / `had_hf` from shared memory; with fp16 C it rounds to half first, as the served pass reads back fp16 C.
- **D. gemv V2.** Warp-private `cp.async` ring, RING 6, evict-first; `wait_group` tracks completion. Each lane reads `tp[lane]` and `tp[(lane+31)&31]` (the served shuffle mapping). The A fragment is loaded one slice ahead; the first group's ring fills before the Hadamard + grid.sync. Last-arriver output Hadamard per 128-column block (counter + threadfence).
- **E. ROWS32.** TILESIZE_M = 32 twins of shapes 2 and 3 reuse the 16-row tuned config (rows hash as min(pow2, 16)); same grid and tiles, per-16-row-block sequence unchanged. Independent of `EXL3_DENSE_V2`; inert at ≤ 16 rows. Piece 4a of the MTP >16-row redo.

Flags: `EXL3_DENSE_V2` = 0 (unset) / 1 (all twins) / 2 (gemm + mgemm only) / 3 (gemv only); `EXL3_DENSE_ROWS32` = 0/1; single digit, anything else fails loudly.

## 4. Expected gains (all *estimates*; P0 measures)

| mechanism | estimate |
|---|---|
| one chain handoff | 1.0-1.5 µs |
| batched gather | ~1.5 µs total |
| B prologue | 0.7-1.0 µs |
| mgemm barriers + output pass | 1-2 µs |
| gemm trailing sync | ~0.5 µs |
| gemv sync + pass | 1.5-2.5 µs |
| gemv ring | 0-2 µs |

| projection | c1d3 µs/call | c1d3 µs/step | c4d3·c8d1 µs/call | c4d3·c8d1 µs/step |
|---|---|---|---|---|
| GDN in_proj ×36 | 4.7-7.5 | 170-270 | 2.7-4.5 | 97-162 |
| attn qkv ×12 | 2.7-4.5 | 32-54 | 2.7-4.5 | 32-54 |
| out_proj + o_proj ×48 | 1.5-4.5 (gemv) | 72-216 | 3-5 | 144-240 |
| index_qk ×12 | 1.5-2.5 (gemv) | 18-30 | 0.7-1.5 | 8-18 |
| **critical path** | | **0.29-0.57 ms (2.3-4.5 % of 12.53)** | | **0.28-0.47 ms (1.4-2.4 % of 19.84; 1.4-2.3 % of 20.75)** |
| shared (overlap stream) | 4-7 + 1.5-2.5 | mostly hidden | 5-8 + 1.5-2 | mostly hidden |

The upper ends assume R682's barrier stall is mostly items 1-4 of §2; the lower ends assume half of it is something r1 doesn't touch. P1 expectation: c4/c8 about −0.3 to −0.45 ms, inside R702's ±5 % per-round spread; the draft-0 cells and the batched stack gate give the answer. c1d3 risk: the gemv ring (if the scoreboard hypothesis is wrong and 40 KB of static shared memory lowers residency); GM (`=2`) and `=3` isolate it.

## 5. Safety

- Default off: relaxed atomic loads only. The install requires unchanged served SASS plus exactly 12 + 12 + 8 twins.
- Graphs: served argument list plus one trailing pointer (graph parameter indices unchanged). The workspace is allocated at the first dense gemm/mgemm call once a flag is on, never during capture (`BC_Attention` / `BC_GatedDeltaNet` run each slot eagerly before capture). Counter zeroing is synchronous; locks and counters return to 0 after every call.
- VRAM: 4 MiB slots (8 with rows32) + 64 KiB counters per device. The largest served need is 2.5 MiB; a call that doesn't fit falls back.
- Concurrency: the same hazard class as the served `locks` (one dense GEMM at a time per device). **If later work (e.g. MoE r3 overlap) lets a dense GEMM overlap another, both the served locks and these slots need per-stream copies.**
- Registers/spills: unknown until compiled; the install dumps `cuobjdump -res-usage` for the twins; LOCAL > 0 is a finding.

## 6. Open items

1. First compile: `build.log`, twin counts.
2. Parity at every dense K4 projection, with engagement at the served rows.
3. P0: which mechanism pays; m1 − m2 at rows 4 is the gemv ring's net effect.
4. For r2: RING 4 or a hybrid ring if the gemv is slower; overlap the Hadamard with every B stage if in_proj r16 stays slow.
