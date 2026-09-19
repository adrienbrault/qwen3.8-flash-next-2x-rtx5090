# R514: decode round 4 group I is served

Results directory on the serving host: `results/2026-09-19-r514-promote-r4i`. Raw records: [`2026-09-19-r514-promote-r4i/`](2026-09-19-r514-promote-r4i/). Driver: [`scripts/r514-promote-r4i.sh`](../../scripts/r514-promote-r4i.sh). Date: 2026-09-19.

Promoted 2026-09-18 23:10 UTC: image `tabbyapi:decode-kernels-r4` with `EXL3_DRAFT_PINNED_STAGING=1 EXL3_BATCH_VERIFY=1 EXL3_MTP_HEAD_N=65536` on top of [R511](r511-promote-2p50.md). Evidence: [R499](r499-decode-r4.md). Launcher: [`scripts/launchers/launch-flashnext-r514.sh`](../../scripts/launchers/launch-flashnext-r514.sh).

| gate | result |
| --- | --- |
| G0 sampler fallbacks (presence penalty, repetition penalty, stop strings) | byte-identical to the served image |
| G1 boot | image, environment and config as intended; fingerprints canonical |
| G2 agentic-edit | 6/6 in four modes; greedy c1 219.1, greedy c4 412.7, sampled c1 230.2, sampled c4 388.8 t/s |
| G3 needles 131k / 240k | 5/5 / 5/5 |
| G4 tool-eval 69×4 | 84.5 ± 1.9 (trials 114 / 117 / 114 / 120) |

Free VRAM after boot: 2,015 / 1,099 MiB.
