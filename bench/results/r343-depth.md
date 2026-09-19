# R343: Decode and TTFT against prompt depth

Results directory on the serving host: `results/2026-09-16-r343-depth`. Driver: [`scripts/r343-depth.sh`](../../scripts/r343-depth.sh). Date: 2026-09-16.


Code, 1,024 forced tokens, c1, greedy. The rungs are labels: the filler's token estimate runs ~28 % high, so the
`prompt tokens seen` column is what the server actually received.

| requested depth | prompt tokens seen | decode (t/s) | TTFT (s) |
| --- | --- | --- | --- |
| 0 | 101 | 183.4 / 189.0 | 0.15 |
| 30,000 | 38,266 | 155.5 / 158.9 | 0.74 cold, **0.24 warm** |
| 120,000 | 152,761 | 150.1 / 155.6 | 24.0 cold, **0.43 warm** |

Decode falls 18 % from a 101-token prompt to a 152,761-token one. The second TTFT column is the larger effect: a
152k prompt costs 24 s cold and **0.43 s on a repeat**, a 56× reduction from the paged prefix cache. An agent that
resends a long conversation every step pays the warm figure, not the cold one.

At c4 the same rungs read 50.3–53.7 t/s per stream at 38,283 prompt tokens (TTFT 0.73–1.01 s) and 46.9–49.9 t/s at
152,778 (TTFT 1.62–1.67 s). **Those c4 rows share their filler prefix** — only a short suffix differs between the
four requests — so they measure shared-prefix concurrency, which is what a fan-out of agents on one harness
actually sends, not four independent contexts. Independent contexts are measured in `2026-09-16-r345-pool`.
