# R495b: the 2.50bpw pack boots a 786,432-token pool (2.18×) and decodes 3–8 % faster except code c1

Results directory on the serving host: `results/2026-09-18-r495b-2p50-audition`. Raw records: [`2026-09-18-r495b-2p50-audition/`](2026-09-18-r495b-2p50-audition/). Driver: [`scripts/r495-fetch-2p50.sh`](../../scripts/r495-fetch-2p50.sh), [`scripts/r495b-2p50-audition.sh`](../../scripts/r495b-2p50-audition.sh). Date: 2026-09-18.

[r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw][r0b0tlab] (routed experts at mixed K = 2 / 3 / 4) against the served 3.05bpw pack, same launcher, image, flags, 4 slots, [30, 30], chunk 2048, vision on.

Pool ladder at 8-bit KV: 1,048,576 / 983,040 / 917,504 / 851,968 / 819,200 do not boot; **786,432 boots** (2,075 / 1,281 MiB free). Whole-pool test: four 141.6k-token prose prompts at once (566k tokens in flight) complete 4/4, TTFT 90 s, then 115 t/s per stream.

| load | 3.05bpw → 2.50bpw |
| --- | --- |
| code c1 (`fn_bench`, mean of 2) | 216.7 → 210.2 t/s |
| code c4 aggregate | 434.7 → 470.9 |
| prose c1 | 170.7 → 181.0 |
| prose c4 aggregate | 424.9 → 438.6 |
| multiprompt code c1, per-request median | 177.5 → 183.9 |
| multiprompt code c4 aggregate | 397.4 → 418.8 |
| multiprompt prose c1 median | 176.5 → 188.6 |
| multiprompt prose c4 aggregate | 349.2 → 347.4 |
| prefill 22.6k tokens | 7,235 → 7,144 t/s |
| needles 131k / 240k | 5/5 / 5/5 |
| tool-eval 69×4 | 85.5 ± 1.3 (controls the same day 83.2 and 85.0) |
| GSM8K n=200, c4 | 0.815 against 0.925 |

The GSM8K gap held the promotion. [R503 and R509](r509-gsm8k-nostop.md) traced it to lm-eval's stop strings, not to the weights, and the pack was promoted in [R511](r511-promote-2p50.md).

[r0b0tlab]: https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw
