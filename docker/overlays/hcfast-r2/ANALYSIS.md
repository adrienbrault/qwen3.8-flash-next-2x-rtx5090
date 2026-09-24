# hcfast r2: HC mix chain, round 2 (bitwise, flag-gated, stacked on r1)

2026-09-24. r2 adds `EXL3_HC_MIX_V3=2` on top of hcfast r1. `EXL3_HC_MIX_V3=1` is r1 unchanged: its 24 kernels compile to the same SASS in the r2 build (`r1-sass-identity.txt`). Unset means the served path. Every output is bitwise-identical by construction. Nothing in r2 has run on a GPU. The operator's steps are in HOW-TO-VERIFY.md, and the gate unit is `hcfast-gate.sh`.

## 0. Verdict

- **What R698 changed:** at the served 16-row shapes (c4d3, c8d1) the win came from tiling, not from r1's instruction changes.
  - V3 dots at the served B=8 is a regression: `v3-dots` 32.00 vs reference 29.03 µs/chain, 1.10.
  - V3 up at B=8 is faster: `v3-up` 25.60, 0.88.
  - Cutting dots to 2 rows per CTA gives the best chain: `v3-d2` 23.97, 0.83 (`flan:/srv/qwen5090/results/2026-09-24-r698-hcfast-gate/p0.log`).
  - r2 therefore works on the shapes that will actually run: dots at B ≤ 2 and up at B = 8.
  - This supersedes the R697 review's lever-7 statement ("more warps works at R=4 only; at R=16 no headroom"). The headroom at 16 rows comes from `grid.y` (more row tiles), not from registers.
- **Item 1, dots stream rounds.** Verified on the post-change SASS with a scoreboard-level counter (`sass_rounds.py` → `sass-rounds-r2.txt`). Served dots at B=8 has 8 serial load rounds per column iteration (24 per warp) and 2 per iteration at B=2 (6).
  - r1 had already collapsed B=2 to 3 rounds per warp.
  - r2 gets B=2 to **2** at `DOTS_PF=1`, the level-2 default. The alternatives are worse: PF=0 gives 4 and PF=2 gives 4, because ptxas re-splits the batches.
  - B=8 stays at r1's 2 rounds per iteration, because the unrolled prefetch does not fit 128 registers without spills.
  - Static rounds are a proxy; P0 measures.
- **Item 2, up prelude.** The brief's literal lever, issuing up_q before the prelude barrier, is already r1 U1 (`hc_mix_v3.cu` r1 kernel, step 1).
  - What remained at B=8 was a second serial round: the d4 loads were issued only after the rsqrt had consumed the sum-of-squares load. r2 issues them first and makes the per-element code branch-free.
  - Up rounds: served up<8> 16, r1 up<8> 4, r2 up<8,Q=4> **2**.
  - r2 also adds Q=4: 16 warps per CTA, so the same state rows are re-derived by half as many CTAs.
- **Item 3, defaults.** With the flag at 2, the defaults are DOTS_B=2, DOTS_PF=1, DOTS_J=4, UP_B=8, UP_Q=4 and PDL off. These are the tiling r1 served with (DOTS_B=2, UP_B=8) with r2's load changes. UP_Q=4 is the default because at UP_B=8 it is the only r2 up form (2 rounds against r1's 4); UP_Q=2 at UP_B=8 falls back to r1's own up kernel (§3). So bare `EXL3_HC_MIX_V3=2` runs both r2 kernels and reports mask 15 (31 with PDL=1).
  - Every knob accepts a per-R table (`"4:4,32:2"`).
  - The gate does not measure the defaults. It measures the per-R tables that P0's sweep plus interleaved confirmation recommend (bench_hcfast_p0.py stages A, B and C).
- **New bitwise knobs, not in the brief:**
  - `DOTS_J=8`: 8 fn rows per CTA. Each stream row is then re-read from L2 by 42 CTAs instead of 82.
  - `UP_Q=4`: 4 column quads per CTA. The state prelude's `dots` rows are then re-read by 160 CTAs instead of 320.
  - `PDL=1`: programmatic dependent launch, which hides the kernel-boundary launch gap.
  - J=8 and Q=4 **halve the CTA count**. That inverts r1's "CTA count ≥ served" rule. They are sweep options, off by default, and only the P0 sweep can enable them.
- **Item 4 (report only):** no bitwise column assignment fixes the dots EXIT-barrier imbalance (§5).
- **Estimate (not measured):** r2 vs r1-d2 at R=16 is −1.5 to −6 µs/chain, which is −0.15 to −0.65 ms/iterate at c4d3/c8d1 (§6). The lower end is below the harness floor, so FLAT against HC1 is possible. Against OFF, r2 should clear the floor wherever r1-d2 did.

## 1. Inputs, and what the R697 review corrected in r1's §7

Inputs:
- R698/R699 (`STACK.md` rows 1-2; `p0.log` above);
- the independent review of R697 ([`r697-hc-source-ncu.md`](../../../bench/results/r697-hc-source-ncu.md) carries its corrections);
- OPERATIONS §15/§16;
- [`scripts/r701-stack-gate.sh`](../../../scripts/r701-stack-gate.sh) (the P1 pattern);
- the moefast-r1 patch (composition).

The six corrections, accepted as stated. r1's own ANALYSIS is left as landed; the corrected statements live here.
1. Served dots repeats its stream loads **per row b**, not per j.
2. Served dots has **B rounds** per column iteration, not 1+B: the weights share round 0.
   - Per 3-iteration warp that is 12 at B=4 and 24 at B=8.
   - The control-code count in `sass-rounds-r2.txt` agrees: the served dots<8> loop body has 8 rounds plus the post-barrier `fn_s` round, and served dots<4> has 4 per iteration.
3. "up R=4 barrier ~9.5 %" is two barriers: the rmr barrier (3.9 %, one warp's `dots` load plus rsqrt) and the prelude barrier (5.4 %, warp 0's extra trip).
4. The "prelude share is inflated by the flush" argument was overstated. The serial round count is served-real; only the per-round latency is inflated.
5. The "attacked?" column was inverted for dots at R=4. D3's batching fully attacks the stream-round bucket (20.2 %) at B ≤ 4. The I2F/weight bucket (26.1 %) is only partly attacked: one stream round per iteration remained.
6. r1's load counts came from an op-class summary without registers. They are replaced by the scoreboard-level count here. The r1 numbers that change: V3 dots<2> is 3 rounds per warp and V3 dots<8> is 2 per iteration. r1 §7 had "~9" at B=8 as an estimate; the count is 6 plus the `fs` round.

**R698 P0 against the review's lever ranking.** The review had ranked "more warps" last, as "R=4 only". At R=16, going from B=8 to B=2 (grid.y 2 → 8) is the largest measured gain: 0.98 → 0.83 of reference. Some of that is the extra warps. Some is that V3 dots<8> itself is slower than served (1.10). Registers do not explain it: V3 dots<8> uses 128 registers (4 CTAs/SM) against served 94 (5/SM), but R=16's 656 CTAs fit one wave either way (680 slots). The cause is unmeasured. The rolled loop and SASS length are hypotheses; the optional ncu pass in HOW-TO-VERIFY §3 would show it.

## 2. Where r2 looks next, and why: L2 traffic at 16 rows (model, labelled estimates)

The model (estimate, arithmetic only):
- Stream bytes per dots call = R·H·D·4 B, re-read by every fn-row group: (M/J + 1) groups at M = 324.
- Weight bytes = M·H·D·1 B, re-read by every row tile: ⌈R/B⌉ tiles.

| R=16 dots tiling | streams from L2 | weights from L2 | total | source |
|---|---:|---:|---:|---|
| served B=8, J=4 | 0.66 MB × 82 = 53.7 | 3.3 MB × 2 = 6.6 | 60.3 MB | R687 ncu measured **60.4 MB** L2→L1 (hcfast r1 ANALYSIS §0): the model holds |
| B=2, J=4 (r1-d2) | 53.7 | 3.3 × 8 = 26.5 | 80.2 MB | estimate |
| B=2, J=8 | 0.66 × 42 = 27.5 | 26.5 | 54.0 MB | estimate |
| B=4, J=8 | 27.5 | 13.3 | 40.8 MB | estimate |

A check at R=4 (served B=4): 0.16 × 82 + 3.3 = 16.7 MB, against the 16.8 MB measured (same source).

- At 16 rows, activation re-reads are two thirds of dots' L2 traffic, and the review's largest dots bucket is the stream rounds (37.5 %).
- J=8 halves the re-reads without touching any per-output arithmetic. Each (row, j) accumulator is independent of every other j, and J only decides which outputs share a CTA.
- The cost of J=8: 2J weight registers per iteration and half the CTAs. At B=2, J=8 PF=0 uses 80 registers (6 CTAs/SM) and 42 × 8 × 4 = 1,344 CTAs, against 2,624 at J=4.

**Waves, not one-wave tiling, at 16 rows.** Dots grid = (⌈M/J⌉ + 1 sum-of-squares CTA) × ⌈R/B⌉ × 4 column blocks, M = 324 fn rows (`launch_dots_v4` / r1 `launch_dots_v3`); slots = 170 SMs × CTAs/SM from the ptxas register counts in `ptxas-r2.log`. R699's accepted tile `v3-d2` launches 82 × 8 × 4 = 2,624 dots CTAs at 64 registers (8/SM, 1,360 slots), which is **1.93 waves**, and it beat every one-wave form, including served B=8 at 656 CTAs.
- r1's auto-tiler rule ("smallest B whose grid fits one wave, CTA count ≥ served") was never the binding constraint at 16 rows.
- The r2 default (B=2, PF=1, 80 registers, 6/SM, 1,020 slots) is 2.6 waves.
- J=8, PF=0 is 1,344 CTAs in 1,020 slots, 1.3 waves.
- Wave count trades against per-CTA latency and L2 bytes, which is why the fewer-CTA forms (J=8, Q=4) are legitimate sweep options and why P0, not occupancy arithmetic, picks.

up at R=16, B=8 (estimate):
- Each of the 640 CTAs derives the full state for its 8 rows from `dots`: 8 × 325 × 16 B = 41.6 KB per CTA, 26.6 MB in total.
- Q=4 halves this to 13.3 MB. It also halves each warp's prelude work, because WPR goes 1 → 2 warps per row at B=8.
- The prelude plus its barrier is 28.8 % + 3.7 % of up samples at R=16 (review).

## 3. The change (`hc_mix_v3.cu`, namespace `hcv4`, mode bits 4/8/16)

| # | kernel | change | attacks (R697 bucket) | bitwise because |
|---|---|---|---|---|
| E1 | dots | streams of the next PF column iterations outstanding (PF 0/1/2, template); streams issued **before** the weight words in program order; stream row addresses formed up front | stream rounds (20.2 % R=4, 37.5 % R=16) | the same loads into the same accumulators; order of FMAs unchanged |
| E2 | dots | weight words double-buffered (WD=1), or all three iterations when B=1/J=4 or B=2/J=4/PF=2; the 0x80808080 bias applied at use (raw words held) | weight round (26.1 % / 12.5 %) | same operand to every `fmaf` |
| E3 | dots | J = 4 or 8 fn rows per CTA (template) | L2 re-reads of streams (§2) | per-(row, j) accumulator independent of j |
| E4 | dots | sum-of-squares CTA: all rows of SQPF+1 iterations' float4s issued together (all 5 at B ≤ 2) | the one CTA per (row tile, h) with 5 iterations | served fmaf nest per float4, iterations ascending |
| E5 | up | d4 prelude loads issued before the rsqrt consumes the sum-of-squares load; per-element code branch-free (served expressions evaluated on zeros for absent elements, results discarded) | prelude rounds (28.1 % / 28.8 %) | the same expressions for every existing element |
| E6 | up | Q = 2 or 4 column quads per CTA (H·Q warps; 8/B or 16/B prelude warps per row) | prelude re-derivation (§2), prelude + barrier | warp → (h, quad) map and each lane's rank-index chain unchanged; which quads share a CTA only |
| E7 | both | PDL: `cudaLaunchKernelEx` + programmatic stream serialization. `griddepcontrol.wait` precedes every read of earlier stream work; dots issues `launch_dependents` right after its wait; nothing is written before the wait | kernel-boundary launch gap | no arithmetic touched |
| — | up | B=8, Q=2 runs r1's V3 up kernel, without PDL | — | it is r1's kernel |

Why that last row exists: every r2 form of up<8, Q=2> spilled 4-16 bytes at the 64-register budget (ptxas, four variants tried). The mask reports 2 for it, not 8. With that fallback, PDL=1 only changes the dots launch (bit 16 still set; `launch_dependents` in dots then has no programmatic consumer), which is why the level-2 default is UP_Q=4.

**PDL pre-wait safety (static check, by reading the source; parity cannot rule out a race).** When dots triggers at CTA start, up's pre-`griddepcontrol.wait` region can run while dots is still writing. What up reads before its wait (`up_i8_state_v4_kernel`, hc_mix_v3.cu): at every B, `up_q` into wq[NI] (module weights); at B ≤ 4 only (EARLY), `load_consts`: writer/wb/wk/eb/eq/ek/ed/e_on are thread and block index arithmetic, `scale` reads `up_s` and `ew` reads `w` (the epilogue's H × D half weight), both module parameters. None of these is the dots workspace, the streams, headT/state or `gates`. `load_es` (streams) runs after the wait at every B; at B=8 `load_consts` runs after it too. Before its own wait, dots reads only `fs` / its weights; the only thing dots writes is the dots workspace, which up reads after its wait. The same-workspace write-after-read (the next mixer's dots vs this mixer's up) is ordered by that next dots' own wait.

**PDL scope, from the SASS.** ptxas places `ACQBULK` (griddepcontrol.wait) ahead of the constant loads too: fn_q, up_q, up_s and w are all issued after it in every hcv4 kernel. No load overlaps the previous kernel. PDL here buys the launch gap only: the up grid is resident when dots retires.

A possible follow-up, not done and not measured: the `.nc` (`__restrict__ const`) loads may be why ptxas orders them after the barrier. Plain `LDG` (`__ldcg`) would be bitwise and might let them issue early.

**Registers** (`ptxas-r2.log`, sm_120, the extension's flags): 47 of 47 kernels have 0 spills. The 24 r1 kernels keep their r1 SASS (`r1-sass-identity.txt`).

| r2 dots <B, J, PF> | regs | CTAs/SM | r2 up <B, Q> | regs | CTAs/SM |
|---|---:|---:|---|---:|---:|
| <1,4,0/1/2> | 63 / 63 / 59 | 8 | <1,2> / <1,4> | 60 / 62 | 4 / 2 |
| <1,8,0/1/2> | 80 / 96 / 94 | 6 / 5 / 5 | <2,2> / <2,4> | 61 / 64 | 4 / 2 |
| <2,4,0/1/2> | 64 / 80 / 94 | 8 / 6 / 5 | <4,2> / <4,4> | 64 / 62 | 4 / 2 |
| <2,8,0/1/2> | 80 / 96 / 128 | 6 / 5 / 4 | <8,4> | 64 | 2 (512 threads) |
| <4,4,0/1>, <4,8,0> | 96 / 128 / 128 | 5 / 4 / 4 | <8,2> = r1 V3 up<8> | 64 | 4 |
| <8,4,0> (rolled, = r1 form) | 128 | 4 | | | |

**Serial load rounds**, whole kernel per warp, static, from `sass-rounds-r2.txt`. A round is the loads issued together before the first wait on any of them. Rolled loops (served dots, and dots<8>) are counted per iteration × 3. The trailing post-barrier `fn_s` load is 1 more for served and r1 dots<8>.

| kernel | served | r1 | r2 |
|---|---:|---:|---:|
| dots B=2 (main CTAs) | 2 / iteration = 6, + fn_s | 3 | PF0 4, **PF1 2**, PF2 4; J=8: 3 |
| dots B=4 | 4 / iteration = 12, + fn_s | 4 | PF0 3, PF1 3; J=8: 5 |
| dots B=8 | 8 / iteration = 24, + fn_s | 2 / iteration = 6, + fs | same as r1 |
| dots B=1 | 1 / iteration = 3, + fn_s | 3 | J=4: 3-4; J=8: 4-5 |
| up B=8 | 16 | 4 | Q=4: **2**; Q=2 = r1 |
| up B=2 | 10 | 2 | 2 |

These are ptxas's schedule, not the source's intent. Twice, the source order that should have produced one round did not: PF=2 at B=2, and J=8. P0 decides among the forms; the static count only explains them.

## 4. What each lever the brief named maps to

1. **Dots stream rounds (brief item 1).**
   - E1/E2 cover it: B=2 goes to 2 rounds per warp at PF=1 (r1: 3, served: 6 + 1). E3 attacks the same bucket's bandwidth side.
   - At B=8 nothing changes: the second stream buffer does not fit 128 registers.
   - The brief asked for round counts at B=2 and B=8 on post-change SASS. They are in the table above; they come from control codes, not an op summary.
2. **Up prelude (brief item 2).**
   - E5/E6. The up_q-before-barrier part was already r1 U1.
   - The rmr barrier the review names no longer exists in V3/r2: rmr comes by shuffle. The prelude barrier stays, because headT has to be complete before the rank loop.
3. **Defaults (brief item 3).**
   - The level-2 defaults are the r1-d2 tiling plus r2's load changes (PF=1).
   - A per-R table is supported and chosen by P0. R698 had R=4 at d4 10.43 against d2 10.62, 0.19 µs × 107 chains = 0.02 ms per iterate, below any P1 floor. So one flat DOTS_B=2 is a reasonable default. Stage C re-measures it.
4. **EXIT imbalance (brief item 4):** §5.

## 5. Item 4 (report only): the dots EXIT-barrier imbalance

**No bitwise assignment exists.** Each output (row, j, h) is fixed by three steps:
- (a) thread t's serial `fmaf` chain over its column groups {t, t+128, t+256} in that order, 8 FMAs per group;
- (b) `__shfl_down_sync` over offsets 16..1 within each warp;
- (c) the sum of red[v] over v = 0..3 from 0.0f.

Groups 256..319 exist only for t < 64 (warps 0-1). The chain in (a) is serial: fmaf is not associative, so a thread's 24-FMA chain cannot be split between two threads. Any assignment that gives every warp at most 2 iterations must move groups 256..319 to other threads. That changes the partial each thread contributes to (b) and (c).

A 160-thread CTA (5 warps × 2 iterations) or any column split with ≤ 2 iterations per warp is therefore numerics-changing. That is the user's call, under the fidelity regime.

Bitwise options that do not help:
- `bar.arrive`-style early exit for warps 2-3 is bitwise, but a no-op for time: CTA resources are released only when the whole CTA exits, so it only removes barrier samples.
- Moving work into warps 2-3 while they wait (for example, loading warps 0-1's third iteration into shared memory) is bitwise, but it adds a barrier or smem round trip for loads that E1 already prefetches.

What r2 does instead: when PF ≥ 1 is scheduled as intended (B=2, PF=1: 2 rounds), warps 0-1's third iteration is FMA-only. Its load latency is off their critical path, which is the bitwise part of the review's "third of the loop" argument.

## 6. Estimates and P1 resolution (labelled; P0 and P1 decide)

**R=16 chain, r2 vs r1-d2** (23.97 µs/chain in R698 P0). The estimate for V3 dots<2> (13.5 µs) comes from R698 P0 differences:
- `v3` − `v3-up` = V3 dots<8> − served dots<8> = +2.78 µs;
- `v3-d2` − `v3-up` = V3 dots<2> − served dots<8> = −1.63 µs;
- served dots<8> ≈ 15.1 µs (GAP-TABLE §4), so V3 dots<2> ≈ 13.5 µs and V3 up<8> ≈ 25.60 − 15.1 − 1.3 = 9.2 µs.

| part | estimate | basis |
|---|---:|---|
| dots, J=8 and/or PF=1 | −1 to −3 µs | L2 bytes −33 % (§2); rounds 3 → 2 |
| up, Q=4 + single-round prelude | −0.5 to −2 µs | prelude re-reads −50 % (§2); prelude 28.8 % + barrier 3.7 % of up samples |
| PDL (launch gap, 1 boundary: dots → up) | 0 to −1 µs | not measured on this box; launch-gap only (§3). hc_apply → dots is not a programmatic boundary: hc_apply issues no `launch_dependents`, so a PDL dots starts only when hc_apply completes |
| **chain** | **−1.5 to −6 µs** | sum |

**Per iterate:**
- at 101-108 chains per step (GAP-TABLE §4: 108 at c4d3, 101 at c8d1, 107 at c1d3), that is **−0.15 to −0.65 ms/iterate vs HC1** at c4d3 and c8d1;
- against OFF, add R699's HC1 − OFF (c8d1 −0.51 / −0.84 ms): about −0.65 to −1.5 ms;
- at c1d3 (R=4), r2 vs r1-d2 is −0.3 to −1.5 µs/chain (estimate: prelude round, PF, PDL; J matters less at 4 rows, where streams are 13 of 17 MB), so −0.03 to −0.16 ms/iterate.

**Resolution, pre-registered.** §16 puts the in-process harness floor at about 0.5 % at c8d1: 0.1 ms at an OFF of 19.5 ms (R698). A P0 difference clears that floor at c8d1 when Δµs/chain × 101 ≥ 0.1 ms, that is **|Δ| ≥ 1 µs/chain at R=16**.
- Read stage C's `r2 / r1-d2` medians against that. At R=16, below 1 µs means expect FLAT for HC2 − HC1 in P1.
- c1d3 (R=4) will very likely be FLAT against HC1, since c1 noise is larger. That is not a rejection under §16.
- No P1 sign is promised.

## 7. Risks and what would discriminate

- **PDL under graph capture.** Parity section 6 runs every PDL configuration in a separate process. A launch or capture error there is reported as `pdl: unsupported` and the gate runs P0 with `--no-pdl`, so the non-PDL r2 still gets measured. A PDL output mismatch fails parity.
  - The write-after-read case is covered by the parity section 4 graphs. Two mixers share one `dots` workspace, so B's PDL dots writes it while A's up is the kernel it follows. A producer kernel writes the streams right before A's dots.
- **ptxas scheduling.** The source-level prefetch is not what runs; `sass-rounds-r2.txt` is. If P0 prefers forms whose static rounds look worse, the static count is not the bound there, and L2 bytes or occupancy are.
  - The discriminating probe is ncu on stage C's pick vs r1-d2: `bench_hcfast_p0.py --ncu --ncu-variants r1-d2 <pick>`, with `lts__t_sectors_srcunit_tex_op_read.sum` and the long-scoreboard share.
- **J=8 / Q=4 fewer CTAs.** At R=4, dots J=8 means 42 × 1 × 4 = 168 CTAs at B=4, an under-filled grid. The sweep includes it and will reject it there if it is slower.
- **Not done, next levers:**
  - plain-`LDG` constants so PDL overlaps the weight fetch with the previous kernel (bitwise);
  - cp.async stream prefetch into smem, to force one round at B=2 regardless of ptxas (bitwise; costs occupancy);
  - the 160-thread dots CTA (numerics-changing, §5).
