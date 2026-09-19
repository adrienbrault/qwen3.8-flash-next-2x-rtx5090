# R554: decode at 0, 100k and 200k prompt tokens on the 2.50 bpw configuration

Results directory on the serving host: `results/2026-09-19-r554-depth-decode`. Raw records: [`2026-09-19-r554-depth-decode/`](2026-09-19-r554-depth-decode/). Driver: [`scripts/r554-depth-decode.sh`](../../scripts/r554-depth-decode.sh). Date: 2026-09-19.

Configuration: the served configuration of [R548](r548-promote-gdnbf16-ring.md) (4 slots, 1,032,192-token pool, NVMe tier off). `fn_bench`, 2,048 forced tokens, greedy; the second run of each depth is reported (its prefix is cached, so it measures decode).

| kind | prompt tokens | decode, 1 stream (t/s) |
| --- | --- | --- |
| prose | 89 | 196.9 |
| prose | 99,839 | 195.6 |
| prose | 199,451 | 193.3 |
| code | 101 | 197.8 |
| code | 179,575 | 212.0 |

- 4 prose streams of ~100k distinct prompt tokens each (399k resident): 455.4 t/s aggregate, 118–169 per stream.
- Cold prefill from the first run: 99,839 tokens in 9.86 s (10,130 tok/s), 199,451 in 19.52 s (10,218 tok/s), 179,575 code tokens in 18.25 s; 4 × ~100k sent together in 51.98 s (7,690 tok/s).
- The code filler counts about 1.35× its target in tokens; the 266,000 code target exceeded the 262,144-token window and was rejected (HTTP 400).

Prose decode at 1 stream drops about 2 % between 89 and 199,451 prompt tokens.
