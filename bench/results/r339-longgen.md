# R339: Decode, code, forced length

Results directory on the serving host: `results/2026-09-16-r339-longgen`. Raw records: [`2026-09-16-r339-longgen.jsonl`](2026-09-16-r339-longgen.jsonl). Driver: [`scripts/r339-gates.sh`](../../scripts/r339-gates.sh). Date: 2026-09-16.


4,096 tokens forced per request with `min_tokens`, greedy, chat endpoint, `finish_reason: length` on every row.
`per stream` is the median steady-state rate; `aggregate` is total tokens over the round's wall.

| concurrency | per stream (t/s) | aggregate (t/s) | TTFT (s) |
| --- | --- | --- | --- |
| 1 | **217.6** | 217.6 | 0.21 |
| 2 | 121.9 | 244 | 0.26 |
| 4 | 65.5–70.3 | ~265 | 0.49–0.57 |
| 8 | *see below* | | |

The c1 figure is the one interactive use sees: **217.6 t/s steady-state on code**, from a 3.05 bpw checkpoint that
fits two cards with a 262k window.

Aggregate scaling is the weakness: 1→4 buys 1.22×, and the whole ladder buys 1.60× from c1 to c8 (256-token
requests, below). That is the cost of layer splitting: with `tensor_parallel: false` the two cards take turns, so
each card is busy only while its own layers run (measured 44–47 % utilisation, 236/225 W against 600/575 W limits,
during a request).
