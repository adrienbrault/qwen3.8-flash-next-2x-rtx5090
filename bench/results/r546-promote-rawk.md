# R546: the QSA raw-key ring is served; the page pool grows from 819,200 to 983,040 tokens with the same output and speed

Results directory on the serving host: `results/2026-09-19-r546-promote-rawk`. Raw records: [`2026-09-19-r546-promote-rawk/`](2026-09-19-r546-promote-rawk/). Driver: [`scripts/r546-promote-rawk.sh`](../../scripts/r546-promote-rawk.sh). Overlay: [`docker/overlays/qsa-rawk-ring-r1/`](../../docker/overlays/qsa-rawk-ring-r1/). Date: 2026-09-19, 13:32–14:30 UTC.

## What changes

The QSA sparse-attention indexer keeps two planes per cached token next to the K/V pages: `raw_k`, the indexer key of every token (256 B per token and layer), and `pooled`, one mean key per 4-token block (64 B per token and layer). Only `pooled` is read by the selection; `raw_k` is read by the pool-update kernel, and only for the rows of the block an append touches. With `EXL3_QSA_RAWK_RING=1` the `raw_k` plane becomes a ring of 20 rows per 256-token page: 20 B per token and layer instead of 256. The pooled plane is computed from the same fp16 inputs in the same order, so it is bit-identical; the argument and its limits (verify windows up to 17 tokens, rewinds from page boundaries) are in the overlay's [`impl-status.md`](../../docker/overlays/qsa-rawk-ring-r1/impl-status.md). No CUDA or C++ file changes and the extension is not rebuilt.

Per token, over the 13 attention layers (12 in the checkpoint plus the MTP block's), the cache costs 18,304 B with the full plane and 15,236 B with the ring. At equal bytes that is 984,064 tokens instead of 819,200, +20.1 % (arithmetic from the allocation code).

The launcher changes four lines: the image (`tabbyapi:nvme-tier-r4-e3det-r6-rawk`, the [R540](r540-promote-r6.md) image plus the overlay), the same name in the NVMe tier default so the tier stays on, `EXL3_QSA_RAWK_RING=1`, and `cache_size` 983,040.

## Kernel harness and placement

The GPU harness ([`tests/gpu_rawk_ring.py`](../../docker/overlays/qsa-rawk-ring-r1/tests/gpu_rawk_ring.py)) compiles the ring kernels ahead of time as a CUDA graph slot does, runs 9 generator-shaped call schedules and 2 negative controls, and compares the pooled plane after every call against the served kernels: 0 failures on both cards, Triton 3.5.0 (`harness-gpu0.json`, `harness-gpu1.json`).

Greedy output depends on which card holds each layer. With the split `[30, 30]`, a boot that leaves more memory free can move layer 23 from cuda:1 to cuda:0, and the text then changes although no arithmetic changed. Results `2026-09-19-r544b-placement-control` showed it on 2026-09-19: the served image without the ring at 753,664 tokens moves layer 23 and gives c1 fingerprint `4fad2dbf`, the ring at 851,968 to 917,504 tokens moves it too and gives the same `4fad2dbf`, and the ring at normal placement gives the served fingerprints. The ladder below therefore only accepts pool sizes at normal placement (layer 23 on cuda:1).

## Ladder (NVMe tier off)

The reference boot of the served launcher at 819,200 gives c1 `e7fb377c987d685c` and 30k `4a255910dee2d9c5`, with 1,257 / 289 MiB free on cuda:0 / cuda:1 after the fingerprint prompts. Each ladder step adds 16,384 tokens; a step is accepted when the layer placement is normal, both fingerprints equal the reference, a 120k prompt and a 4-request round complete, and each card's free VRAM stays within 32 MiB of the reference.

| pool | placement | c1 fingerprint | free VRAM MiB (cuda:0 / cuda:1) |
| --- | --- | --- | --- |
| 835,584 | layer 23 moved | `e3e01aa9` | 601 / 4,337 |
| 851,968 to 933,888 | layer 23 moved | `4fad2dbf` | 1,183 / 3,473 down to 563 / 2,953 |
| 950,272 | normal | reference, 30k reference | 1,423 / 497 |
| 966,656 | normal | reference, 30k reference | 1,323 / 375 |
| 983,040 | normal | reference, 30k reference | 1,223 / 257 |
| 999,424 | does not boot | `RuntimeError: Insufficient VRAM in split for model and cache` | |

983,040 is served: +163,840 tokens, +20.0 %. Right after boot, free VRAM reads 1,973 / 737 MiB at 983,040 with the ring and at 819,200 without it.

## A/B (8 boots, NVMe tier off)

A is the served launcher at 819,200, B the candidate at 983,040, boots in the order A B B A B A A B. Every boot gave the c1 reference fingerprint. `fn_bench`, 2,048 forced tokens, greedy, a warm-up round before each shape; code c1 3 runs per boot, the others 2. Means of boot means, B against A, 95 % interval over boots:

| shape | A (t/s) | B (t/s) | B / A |
| --- | --- | --- | --- |
| code, 1 stream | 224.5 | 224.6 | +0.06 % [−0.48, +0.60] |
| code, 4 streams, aggregate | 508.4 | 510.1 | +0.34 % [−0.14, +0.82] |
| prose, 1 stream | 199.8 | 199.3 | −0.22 % [−0.53, +0.09] |
| prose, 4 streams, aggregate | 482.4 | 483.7 | +0.27 % [−1.96, +2.50] |
| cold prefill, 60k target, salted | 9,670 (n 8, 9,536–9,832) | 9,607 (n 8, 9,293–9,876) | 0.993× |
| cold prefill, 120k target, salted | 10,061 (n 4, 9,972–10,200) | 10,038 (n 4, 9,975–10,142) | 0.998× |

In every code c4 run 2 of the 4 requests stop at 1,861 tokens on both arms, and the aggregate includes them.

## Gates (NVMe tier on)

| gate | result |
| --- | --- |
| G1 boot | 983,040 tokens, 8-bit KV, 4 slots, vision on, normal placement, fingerprints equal to the reference; the tier opened a new namespace (the engine sources and the flag list changed) |
| G1b cold prefill, salted | 60k target 9,626 t/s, 120k target 9,834 t/s; floors 9,186 and 9,557 (0.95 × the A arm above) |
| GT restart restore | a 30,000-token prompt after a container restart: 29,952 tokens served from the tier in 0.55 s, output identical to the warm run (0.30 s) and the cold run (3.81 s) |
| G2 agentic-edit | 6/6 in four modes; greedy c1 227.5 t/s aggregate (first full wave 252.2), greedy c4 419.1 aggregate (first full wave 492.3, 145.4 per stream) |
| G3 needles | 5/5 at ~131k and 5/5 at ~240k prompt tokens |
| G4 tool-eval 69×4 | 86.5 ± 2.4 (95 % interval 85.0 to 88.8) |
| G5 GSM8K 5-shot n=500, no stop strings | 0.976 |

Promoted 2026-09-19 14:30 UTC (16:30 CEST); the served launcher then booted with the reference fingerprints and 1,973 / 737 MiB free. Rollback: the R540 launcher, or the ring image with `EXL3_QSA_RAWK_RING` removed and `CACHE=819200`.
