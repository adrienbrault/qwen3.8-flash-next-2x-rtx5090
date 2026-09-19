# R511: the 2.50bpw pack at a 786,432-token pool is the served configuration

Results directory on the serving host: `results/2026-09-18-r511-promote-2p50`. Raw records: [`2026-09-18-r511-promote-2p50/`](2026-09-18-r511-promote-2p50/). Driver: [`scripts/r511-promote-2p50.sh`](../../scripts/r511-promote-2p50.sh). Date: 2026-09-18.

Promoted 2026-09-18 21:21 UTC. Launcher: [`scripts/launchers/launch-flashnext-r511.sh`](../../scripts/launchers/launch-flashnext-r511.sh), which is the previous launcher with the checkpoint name and `CACHE=786432`; image, flags, 4 slots, [30, 30], chunk 2048, vision and draft policy unchanged. Rollback: the 3.05bpw pack at 360,448.

| gate | result |
| --- | --- |
| G1 boot | 786,432, 2,015 / 1,219 MiB free; new canonical fingerprints (c1 `ae890c45d1000582`, 30k `2aa8d1024daece5c`) |
| G2 agentic-edit, 6 files | 6/6 in four modes; aggregate greedy c1 213.8, greedy c4 419.9, sampled c1 228.9, sampled c4 440.1 t/s |
| G3 needles 131k / 240k | 5/5 / 5/5 |
| G4 tool-eval 69×4 | 86.0 ± 2.6 (trials 120 / 123 / 117 / 115) |
| GSM8K (R509, no stop strings) | 0.978 against 0.980 on 3.05bpw |

TabbyAPI ignores the request's `model` field, so clients configured with the 3.05bpw model id keep working; `/v1/model` reports the 2.50bpw id.
