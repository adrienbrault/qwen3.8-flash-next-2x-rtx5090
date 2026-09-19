# R477: Code and prose from one boot of the served configuration

Results directory on the serving host: `results/2026-09-18-r477-daily-prose-code`. Raw records: [`2026-09-18-r477-daily-prose-code/`](2026-09-18-r477-daily-prose-code/). Date: 2026-09-18.


`fn_bench`, 2,048 forced tokens, greedy, two runs per cell, code first, prose, then code again to bracket drift, on the served image of 2026-09-17 12:45 CEST. Every request finished on `length`. Aggregate over the streams; per stream in parentheses.

| kind | c1 | c4 aggregate | c8 aggregate |
| --- | --- | --- | --- |
| code | 209.8–215.0 (drift bracket 215.7) | 422.7–442.1 (111.6–116.7) | 540.6–573.4 (68.7–72.9) |
| prose | 163.9–171.6 | 417.6–433.2 (104.9–108.8) | 539.0–558.4 (67.7–70.1) |

Prose is 0.78× code at c1 and within 3 % of it at c4 and c8. The draft policy is depth 3 up to four streams and depth 1 above, so at c8 both kinds run one draft per step and the batch, not acceptance, sets the rate; at c4 the acceptance difference is absorbed by the batched step. The 2026-09-16 prose figure on the pre-promotion image was 160.6–165.3 at c1 (`2026-09-16-r339-gates`).
