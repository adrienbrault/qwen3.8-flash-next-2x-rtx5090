# R559: the n-gram embedding rows prefetched after the draft readback give byte-identical output; one boot per arm reads +0.7 to +2.0 % decode

Results directory on the serving host: `results/2026-09-19-r559-ngram-prefetch`. Raw records: [`2026-09-19-r559-ngram-prefetch/`](2026-09-19-r559-ngram-prefetch/). Driver: [`scripts/r559-ngram-prefetch.sh`](../../scripts/r559-ngram-prefetch.sh). Overlay: [`docker/overlays/ngram-prefetch-r1/`](../../docker/overlays/ngram-prefetch-r1/). Date: 2026-09-19, 17:42–17:52 UTC.

## What changes

The checkpoint's per-layer n-gram embedding table (18.5 GiB) stays on NVMe and is read row by row for every token. Today a decode step stages its rows inline, when the forward reaches the embedding. With `EXL3_NGRAM_PREFETCH2=1`, each verify forward's row gather starts right after the MTP draft token ids reach the host, while the GPU still runs, and prefill staging moves onto the worker thread. The rows and the order of reads are unchanged, so the output is byte-identical by construction. The flag also turns on a wall-clock instrument (`EXL3_NGRAM_TIMING`) that prints the exposed wait every 30 s.

Configuration: 8 slots, 966,656-token pool, NVMe tier off, the served image of [R561](r561-promote-slots8.md) with the overlay installed.

## Results

- GPU harness on both cards: 9 of 9 checks pass.
- Flag off and flag on: c1 fingerprint `f4add302e176d78e` and 30k fingerprint `4a255910dee2d9c5` on both boots; free VRAM equal.
- Exposed wait with the flag off: prefill 81 calls, mean 13.1 ms (p90 35 ms); decode and verify 806 calls, mean 0.46 ms (p90 0.78 ms).
- Cold prefill, flag on / flag off: 60k target 9,054 / 9,326 t/s (0.971×), 120k target 10,213 / 10,089 t/s (1.012×).

`mp_decode.py`, 24 code and 24 prose prompts, 512 tokens, one boot per arm, paired geometric mean on / off with 95 % bootstrap interval:

| shape | off (t/s) | on (t/s) | on / off |
| --- | --- | --- | --- |
| code, 1 stream | 198.6 | 200.8 | +1.13 % [+0.58, +1.69] |
| code, 4 streams | 125.3 | 127.7 | +2.01 % [−0.02, +4.19] |
| prose, 1 stream | 198.1 | 199.5 | +0.74 % [+0.38, +1.12] |
| prose, 4 streams | 117.6 | 119.4 | +1.58 % [+0.57, +2.63] |

One boot per arm sits inside the ~1.7 % spread between boots of one configuration ([R548](r548-promote-gdnbf16-ring.md)), so [R565](r565-promote-ngram-prefetch.md) measured the change over four boots before serving it.
