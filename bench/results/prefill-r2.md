# Cold prefill on ExLlamaV3 v1.6.7 r2

Measured 2026-11-26, source `20261126T021547Z-578452-prefill-r2-b`, on image `96ab0d09` and launcher `e7ffd0a6`. [Numeric records and validation receipt](2026-11-26-prefill-r2/)

The c1 prose run used one calibration request and three fresh salted prompts per target, T=0, 64 forced output tokens and template-default thinking. The packet validated all 22 server requests as zero-cache. Each rate is the arithmetic mean of per-request prompt tokens / TTFT, including scheduling and first decode. Prompt length and TTFT are arithmetic means.

| target | mean prompt tokens | mean tok/s | mean TTFT (s) | requests |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 2,012 | 6,914 | 0.291 | 3 |
| 8,000 | 7,972 | 10,719 | 0.744 | 3 |
| 30,000 | 30,039 | 12,736 | 2.359 | 3 |
| 60,000 | 60,079 | 13,146 | 4.570 | 3 |
| 120,000 | 120,133 | 13,091 | 9.177 | 3 |
| 200,000 | 200,319 | 12,917 | 15.508 | 3 |
| 240,000 | 240,329 | 12,775 | 18.813 | 3 |

Peak rate was 13,146 tok/s at 60,079 mean prompt tokens on the 2026-11-26 r2 boot. The [2026-11-16 names-fast r1 boot](prefill-r1.md) recorded 12,537 tok/s at 120,133 prompt tokens. Each of the seven targets recorded a higher rate on 2026-11-26 under the same protocol.
