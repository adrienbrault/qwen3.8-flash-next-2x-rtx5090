# R702: hcfast r2 is bitwise-identical and 1.2 % faster at 1 stream in 5 of 5 rounds; flat at 4 and 8 streams

Results directory on the serving host: `results/2026-09-24-r702-hcfast-r2` (raw records not copied here). Driver [`scripts/r702-hcfast-r2.sh`](../../scripts/r702-hcfast-r2.sh), run with `HC_BASE=tabbyapi:moefast-r1`. Image `tabbyapi:hcfast-r2-moe` = `tabbyapi:moefast-r1` plus [`docker/overlays/hcfast-r2`](../../docker/overlays/hcfast-r2/). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

## The change

hcfast r2 is cumulative over hcfast r1 and adds `EXL3_HC_MIX_V3=2`: the dots kernel batches its per-row stream loads, the up kernel issues its state loads before the prelude's reduction and runs it branch-free, every tile knob accepts a table indexed by row count, and a programmatic-dependent-launch switch for dots → up is added (off in the measured arm). `EXL3_HC_MIX_V3=1` keeps r1's kernels, which compile to the same SASS in the r2 build. The design is in [`ANALYSIS.md`](../../docker/overlays/hcfast-r2/ANALYSIS.md).

The measured arm, HC2, is the line the P0 microbenchmark recommended: `EXL3_HC_MIX_V3=2 EXL3_HC_MIX_V3_DOTS_B=1:1,4:2,32:4 EXL3_HC_MIX_V3_UP_B=1:1,8:4,32:8 EXL3_HC_MIX_V3_DOTS_J=1:4,32:8 EXL3_HC_MIX_V3_DOTS_PF=1:1,32:0 EXL3_HC_MIX_V3_UP_Q=1:4,8:2,32:4 EXL3_HC_MIX_V3_PDL=0`.

## What was measured

2026-09-24 08:47 to 09:18 UTC. stack-r2 ([R701](r701-stack-r2.md)) became the served configuration before the unit ran, so the served environment already carried hcfast r1 at `DOTS_B=2 UP_B=8`. The OFF arm (served environment) and the HC1 arm were therefore the same kernels, and HC1 − OFF is an A/A control.

**Identity.** Kernel parity passed. The generated-sequence hashes of HC1 and HC2 equalled the round's OFF in every round and shape. Greedy output on the served launcher with HC2 on and with it off was identical to the reference on 6 of 6 prompts.

**P1, in-process harness.** 4,096 tokens of context, untraced, ms per iterate, 5 rounds per shape in rotated order. Values are the arm minus the same round's OFF.

| shape | HC1 − OFF (A/A) | HC2 − OFF (HC2 against the served HC1) |
| --- | --- | --- |
| 4 streams, depth 3 | −1.20, −0.08, +0.29, +0.12, −0.97 | −1.04, +0.01, −0.39, +0.13, −1.91; mean −0.64 (−3.2 %), mixed signs |
| 8 streams, depth 1 | −0.60, +0.03, +0.71, −0.45, +0.87 | −1.47, +0.58, +0.42, −1.11, +1.90; mean +0.06, mixed signs |
| 1 stream, depth 3 | −0.38, −0.05, −0.09, +0.79, −1.08 | −0.20, −0.15, −0.14, −0.15, −0.10; mean −0.15 (−1.2 %), 5 of 5 negative |

## Reading

- The A/A control puts the per-round spread of the in-process harness at ±1.2 ms at 4 streams (±5 %), ±0.9 ms at 8 streams (±4.5 %) and ±1.1 ms at 1 stream. The sign rule reads the A/A as mixed at every shape. At that spread a 1-2 % effect at 4 or 8 streams cannot show as 5 of 5 same-sign rounds; it is read from the batched served gate. [`docs/PROMOTION.md`](../../docs/PROMOTION.md) carries the corrected resolution; the earlier statement there that the harness separates about 0.5 % came from one pair of R698 cells.
- HC2 gains at 1 stream in 5 of 5 rounds and has no same-sign regression at any shape. It was accepted as a stack entry and served from stack-r3 on ([R716b](r716b-stack-r3.md)), where it replaces hcfast r1's `DOTS_B=2 UP_B=8` environment.
