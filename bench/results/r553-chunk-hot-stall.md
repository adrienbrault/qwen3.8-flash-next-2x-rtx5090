# R553: prefill chunk 2,048 / 1,024 / 512 against the decode stalls of running streams; 2,048 stays

Results directory on the serving host: `results/2026-09-19-r553-chunk-hot-stall`. Raw records: [`2026-09-19-r553-chunk-hot-stall/`](2026-09-19-r553-chunk-hot-stall/). Driver: [`scripts/r553-chunk-hot-stall.sh`](../../scripts/r553-chunk-hot-stall.sh). Probe: [`bench/hot_slots.py`](../hot_slots.py). Date: 2026-09-19.

Configuration: the served configuration of [R548](r548-promote-gdnbf16-ring.md) (4 slots, 1,032,192-token pool, NVMe tier off), one boot per chunk size. Each boot runs phase C of [R549](r549-hot-slots.md) (0 to 3 slots decoding ~112k-token contexts, then a short request and a ~22.5k-token request arrive) and a salted cold prefill at the 60k and 120k `fn_bench` targets.

| chunk | running streams: longest gap during the new request's prefill | running streams: frames/s during (46–73 before) | new ~22.5k request: TTFT with 0 / 3 slots decoding | cold prefill, 60k / 120k target (tok/s) |
| --- | --- | --- | --- | --- |
| 2,048 | 0.52–0.53 s | 4.1–4.7 | 2.42 / 3.21 s | 9,051 / 10,192 |
| 1,024 | 0.20–0.25 s | 6.2–6.9 | 4.22 / 4.64 s | 5,481 / 5,274 |
| 512 | 0.16–0.49 s | 8.0–8.5 | 5.54 / 6.50 s | 4,137 / 4,093 |

At 1,024 cold prefill runs at about half the rate of 2,048, the running streams get about 1.5× the frames during another request's prefill, and the new request waits 1.4–1.7× longer for its first token. Chunk 512 also moved layer 23 to cuda:0 at boot (free VRAM 189 / 2,701 MiB). The served chunk stays 2,048.
