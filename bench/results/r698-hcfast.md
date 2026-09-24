# R698, R699: hcfast r1 is bitwise-identical; the DOTS_B=2 tile is accepted, the 8/8 default regresses at 8 streams

Results directories on the serving host: `results/2026-09-24-r698-hcfast-gate` and `results/2026-09-24-r699-hcfast-d2`. Raw records: [`2026-09-24-r698-hcfast-gate/`](2026-09-24-r698-hcfast-gate/) and [`2026-09-24-r699-hcfast-d2/`](2026-09-24-r699-hcfast-d2/). Drivers: `r698-hcfast-gate.sh` and `r699-hcfast-d2.sh` (queued GPU-exclusive units; the steps are in [`HOW-TO-VERIFY.md`](../../docker/overlays/hcfast-r1/HOW-TO-VERIFY.md)). Image `tabbyapi:hcfast-r1` = `tabbyapi:slotfix-r1` plus [`docker/overlays/hcfast-r1`](../../docker/overlays/hcfast-r1/). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

## The change

`EXL3_HC_MIX_V3=1` routes the decode-time int8 hyper-connection mixer (the `gr_v2_dots_i8` and `gr_v2_up_i8_state` kernels behind `hc_apply`) through new kernels in `hc_mix_v3.cu`. The dots kernel converts each int8 weight once per iteration instead of once per row it owns, issues the iteration's loads before the first multiply, and prefetches the next iteration's weights. The up kernel reads the state values with vector shared-memory loads, reduces with a reduce-scatter of 35 shuffles instead of a 160-shuffle butterfly, and issues its weight words and epilogue operands first. Launch count and grids are unchanged. Each output keeps its operand order, so the results are bitwise-identical by construction. `EXL3_HC_MIX_V3_DOTS_B` and `EXL3_HC_MIX_V3_UP_B` cap the rows per CTA of each kernel; 8 is the served tiling and the default. The design and the ncu evidence behind it are in [`ANALYSIS.md`](../../docker/overlays/hcfast-r1/ANALYSIS.md).

## What was measured

R698 ran 2026-09-24 07:15 to 07:31 UTC with the default caps (8/8). R699 ran 07:31 to 07:39 UTC with `EXL3_HC_MIX_V3_DOTS_B=2` (the `v3-d2` column of R698's P0).

**Parity** (both runs, real weights from 5 layers): 2,080 of 2,080 kernel comparisons identical over rows, sites, mixed dtypes and tile configurations; 80 of 80 on the module's `_mix` with the flag on against off; 200 of 200 fuzz seeds; 80 of 80 after graph capture and 20 replays; the auto tile captured cold in a fresh process 10 of 10; a misaligned workspace declines to V3 with exact outputs, 3 of 3 ([`parity.txt`](2026-09-24-r698-hcfast-gate/parity.txt)).

**P0, kernel chain** (R698, CUDA graph, µs per chain, 24 weight sites cycled so every chain reads its weights from DRAM, 504 chains per variant). Ratios against the served kernels:

| rows | reference µs | V3, caps 8/8 | V3, DOTS_B=2 |
| ---: | ---: | ---: | ---: |
| 1 | 9.92 | 0.86 | 0.86 |
| 4 | 15.21 | 0.69 | 0.70 |
| 8 | 21.84 | 0.69 | 0.68 |
| 16 | 29.03 | 0.98 | 0.83 |

A served step runs this chain at 4 rows at 1 stream with depth 3, and at 16 rows at 4 streams with depth 3 and at 8 streams with depth 1. At 16 rows the default tile keeps almost none of the gain and the DOTS_B=2 tile keeps 17 %. Full table with every tile variant: [`p0.txt`](2026-09-24-r698-hcfast-gate/p0.txt). The L2-warm control ([`p0-hot.txt`](2026-09-24-r698-hcfast-gate/p0-hot.txt)) failed its drift check at 1 row (the repeated reference read 5 % slower) and is not used.

**P1, in-process harness** (4,096 tokens of context, untraced, ms per iterate, OFF → ON, two pairs per shape):

| shape | R698, caps 8/8 | R699, DOTS_B=2 |
| --- | --- | --- |
| 1 stream, depth 3 | 13.24 → 12.85, 13.40 → 12.85 | 13.27 → 12.38, 12.81 → 12.49 |
| 4 streams, depth 3 | 22.00 → 21.18, 21.37 → 21.92 | void (see below) |
| 8 streams, depth 1 | 19.53 → 19.70, 19.59 → 19.69 | pair 1 void; pair 2 19.23 → 18.72 |

The generated-sequence hashes were identical in all 12 pairs. Greedy output on the served launcher, with the flag on and with it off, was identical to the reference on 6 of 6 prompts in both runs.

**Void cells.** From 07:30 to 07:35 UTC a build of the moefast image ran on the same host while R699 held the GPU lock. R699's 4-stream cells (07:31 to 07:34) and its first 8-stream pair (07:34 to 07:35) overlap it and are void. The clean R699 cells are the 1-stream pairs and the second 8-stream pair.

## Reading

- Caps 8/8: at 8 streams both pairs are slower (+0.17 and +0.10 ms, +0.7 %), a same-sign regression, and 4 streams is mixed. Rejected.
- DOTS_B=2: at 1 stream −0.61 ms per iterate (−4.6 %), both pairs the same sign; at 8 streams −0.51 ms (−2.6 %) on the clean pair. No clean cell regresses. Accepted into the stack, with the 4-stream shape left to the stack gate's five rounds.

The stack gate ([R701](r701-stack-r2.md)) measured this flag set (`EXL3_HC_MIX_V3=1 EXL3_HC_MIX_V3_DOTS_B=2 EXL3_HC_MIX_V3_UP_B=8`) over five rounds: −5.2 % at 4 streams (5 of 5 rounds faster), −1.7 % at 8 streams and −3.9 % at 1 stream (mixed signs), hashes identical in every round.
