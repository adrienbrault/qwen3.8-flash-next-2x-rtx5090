# R483: chunk 1024 with split [30, 31] boots 409,600 tokens, but prefill halves

Results directory on the serving host: `results/2026-09-18-r483-exl3-pool-chunk`. Raw records: [`2026-09-18-r483-exl3-pool-chunk/`](2026-09-18-r483-exl3-pool-chunk/). Driver: [`scripts/r483-exl3-pool-chunk.sh`](../../scripts/r483-exl3-pool-chunk.sh). Date: 2026-09-18.

The layer-split loader reserves headroom for the largest transient of a dummy prefill chunk (`model/model_ls.py:230-270` in exllamav3), so the chunk size and the split were traded for page pool. 3.05bpw pack, 4 slots, 8-bit KV. Launcher: [`scripts/launchers/launch-flashnext-r483.sh`](../../scripts/launchers/launch-flashnext-r483.sh).

Boot ladder: [30, 31] with chunk 2048 boots 393,216 (one layer moves to cuda:1); [30, 31] with chunk 1024 boots **409,600**; no tried combination boots 425,984.

| | served ([30, 30], chunk 2048, 360,448) | D ([30, 31], chunk 1024, 409,600) |
| --- | --- | --- |
| c1 greedy / 30k greedy | canonical / canonical | differs (near-tie flip at about 160 tokens) / differs |
| code c1 / c4, 2 runs | 211.9–215.6 / 427.5–441.7 t/s | 166.3–175.6 / 391.2–435.9 t/s |
| prose c1 / c4 | 165.8–171.4 / 418.0–435.2 t/s | 159.7–169.5 / 376.2–420.3 t/s |
| cold prefill 22.6k / 90.1k prompt tokens | 7,228 t/s (3.13 s) / 11,135 t/s (8.10 s) | 4,033 t/s (5.61 s) / 5,361 t/s (16.81 s) |
| 4 × 81k-token prompts at once | — | 4/4 complete, TTFT 80 s |
| needle 131k / 240k | 5/5 / 5/5 | 5/5 / 5/5 |
| GSM8K n=200, c4 | 0.925 | 0.930 |

Not promoted. [R485 and R486](r485-pool-frontier.md) isolate the chunk as the cause of the prefill loss.
