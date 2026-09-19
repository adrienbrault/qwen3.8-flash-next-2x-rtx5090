# R531 and R533: deterministic E3 prefill is bit-identical run to run; it costs 1.9 % prefill at the 60k target and 0.2 % at 120k

Results directories on the serving host: `results/2026-09-19-r531-e3-det` (validation, try 5) and `results/2026-09-19-r533-e3-det-precise` (8-boot A/B). Raw records: [`2026-09-19-r531-e3-det/`](2026-09-19-r531-e3-det/), [`2026-09-19-r533-e3-det-precise/`](2026-09-19-r533-e3-det-precise/). Drivers: [`scripts/r531-e3-det.sh`](../../scripts/r531-e3-det.sh), [`scripts/r533-e3-det-precise.sh`](../../scripts/r533-e3-det-precise.sh). Date: 2026-09-19.

The E3 grouped MoE prefill ([R517](r517-promote-stack.md)) sums expert outputs with `atomicAdd`, so the order of floating-point additions changes from run to run. Two cold runs of one 8,017-token prompt diverged at generated token 128 ([R524](r524-recurrent-tip.md)). That made byte-identity gates unusable on long prompts. The overlay [`docker/overlays/e3-det-r1/`](../../docker/overlays/e3-det-r1/) (written for this repository by an Opus agent round) adds `EXL3_MOE_PREFILL_E3_DET=1`: each expert assignment writes its own slot, and a second kernel reduces the slots in a fixed order. Without the flag the image runs the atomic path unchanged.

## Validation (R531, try 5)

Kernel, one K=2 layer (12) and one K=3 layer (0) of this pack, 512, 1,024 and 2,048 rows, 20 repeats each:

| | result |
| --- | --- |
| deterministic path | bit-identical over repeats in all 6 cases, also with a concurrent GEMM on another stream, and across processes |
| atomic path | 20 distinct outputs in 20 runs, in all 6 cases |
| deterministic vs atomic | normalised RMSE 8.1e-8 to 8.5e-8 (both differ from E3 off by 3.5e-4 to 1.2e-3, the existing E3 accumulation-order gap) |

Cold prefill in process, three runs per prompt:

| prompt | deterministic | atomic |
| --- | --- | --- |
| 8,020 tokens | identical ×3; 2.46–2.97 s (the first run includes warm-up) | first divergence at generated tokens 47 and 172; 2.42–2.55 s |
| 30,020 tokens | identical ×3; 4.65–5.26 s | first divergence at tokens 58 and 66; 4.76–4.88 s |

Served boots with the flag: c1 `e7fb377c987d685c` and 30k `4a255910dee2d9c5`, the daily's fingerprints. Free VRAM during a 120k prefill is the same as without the flag; the per-assignment slot buffer is 200 MiB at a 2,048-token chunk. Two unpaired prefill runs gave 0.968× at the 60k target and 0.990× at 120k, too noisy to decide.

Tries 1 to 4 stopped on the harness: two out-of-memory errors in the in-process probe (the loader fills cuda:0 to within one transient of full whatever the split; `EXL3_AUTOSPLIT_MARGIN_MB=2048` fixed it) and a check that required a K=4 layer, which this pack does not have.

## Precise A/B (R533)

One image (the served image plus the overlay), 8 boots in the order A B B A B A A B; A: the daily's environment, B: the same plus `EXL3_MOE_PREFILL_E3_DET=1`. Per boot: two salted cold prefills per size and `fn_bench` code decode at 1 stream (warm-up round, 2 runs).

| | atomic (A) | deterministic (B) | B/A, 95 % interval of the difference |
| --- | --- | --- | --- |
| prefill, `fn_bench --ctx` 60k target (about 45k tokens) | 9,658 t/s | 9,478 t/s | 0.981, [−4.63, +0.88] % |
| prefill, 120k target (about 90k tokens) | 10,084 t/s | 10,065 t/s | 0.998, [−1.91, +1.54] % |
| decode, code, 1 stream | 222.8 t/s | 223.4 t/s | 1.003, [−0.47, +1.04] % |

Every boot produced the canonical c1 fingerprint.
