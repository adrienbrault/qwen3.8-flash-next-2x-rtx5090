# R339: Concurrency, short generations

Results directory on the serving host: `results/2026-09-16-r339-honest-conc`. Raw records: [`2026-09-16-r339-honest-conc-256.json`](2026-09-16-r339-honest-conc-256.json). Driver: [`scripts/r339-gates.sh`](../../scripts/r339-gates.sh). Date: 2026-09-16.


256 forced-by-prompt-length tokens, greedy, `temperature 0`, streaming, 2 runs. This is the same instrument R219
and R331 used, but the `usage` figures are now correct: the engine under-reported any generation past ~2048 tokens
by ~5× until the R338 image patch, and these rows never crossed that boundary, so they are unaffected either way.

| concurrency | aggregate (t/s) | per stream (t/s) | TTFT (s) |
| --- | --- | --- | --- |
| 1 | 177.6 | 177.6 | 0.103 |
| 2 | 232.8 | 129.0 | 0.18 |
| 4 | 252.7 | 70.2 | 0.33 |
| 8 | 283.5 | 42.9 | 0.51 |
