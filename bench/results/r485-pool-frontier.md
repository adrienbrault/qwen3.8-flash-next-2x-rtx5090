# R485 and R486: the pool frontier at chunk 2048 is 393,216 under every split, and chunk 1024 switches the prefill pipeline off

Results directory on the serving host: `results/2026-09-18-r485-pool-frontier`. Raw records: [`2026-09-18-r485-pool-frontier/`](2026-09-18-r485-pool-frontier/). Driver: [`scripts/r485-pool-frontier.sh`](../../scripts/r485-pool-frontier.sh), [`scripts/r486-fp-diag.sh`](../../scripts/r486-fp-diag.sh). Date: 2026-09-18.

R485 booted the largest pool in 8,192-token steps for six splits ([30.5, 31], [30.75, 31], [31, 31], [30.25, 31.25], [30.5, 31.25], [31, 31.25]) at three chunk sizes. 3.05bpw pack, 4 slots, 8-bit KV. The frontier does not depend on the split: chunk 2048 → **393,216**, chunk 1024 → 409,600, chunk 512 → 417,792. At 393,216 a layer has already moved to cuda:1, which becomes the binding card (721 MiB free while cuda:0 keeps 2,075 MiB).

R486 at the served 360,448 (results `2026-09-18-r486-fp-diag`, raw records [`2026-09-18-r486-fp-diag/`](2026-09-18-r486-fp-diag/)):

| arm | split / chunk | c1 greedy (×2) / 30k | code c1 (t/s) | prefill 22.6k / 90.1k (t/s) | prefill pipeline |
| --- | --- | --- | --- | --- | --- |
| CTL | [30, 30] / 2048 | canonical / canonical | 212.3 / 215.1 | 7,315 / 11,154 | engaged |
| S | [30, 31] / 2048 | canonical / canonical | 215.9 / 216.9 | 7,770 / 11,212 | engaged |
| K | [30, 30] / 1024 | canonical / canonical | 198.7 / 211.9 | 4,010 / 5,450 | absent from the boot log |
| SK | [30, 31] / 1024 | canonical / canonical | 202.6 / 210.6 | 4,251 / 5,581 | absent |

At chunk 1024 the two-card prefill pipeline does not engage, so the cards prefill serially again. Split and chunk alone leave greedy output canonical; R483's changed fingerprint comes from the larger pool moving a layer to cuda:1.
