# R538: decode kernels round 6 are bit-exact and +1.0 % code / +1.1 % prose at 1 stream; c4 unchanged (8 boots)

Results directory on the serving host: `results/2026-09-19-r538-decode-r6`. Raw records: [`2026-09-19-r538-decode-r6/`](2026-09-19-r538-decode-r6/). Driver: [`scripts/r538-decode-r6.sh`](../../scripts/r538-decode-r6.sh). Date: 2026-09-19.

Round 6 ([`docker/overlays/decode-kernels-r6/`](../../docker/overlays/decode-kernels-r6/)) rebases round 5 onto the served chain `tabbyapi:nvme-tier-r4-e3det` ([R535](r535-promote-e3det.md)). It adds three selectors, each off by default:

- `EXL3_GDN_BA_WARP1`: the GDN B/A projection GEMV runs one warp per block instead of eight when the served grid has fewer blocks than the card has SMs (the idea of [exllamav3#369](https://github.com/turboderp-org/exllamav3/pull/369)). At 1 stream with draft depth 3 the call has 4 rows, and the grid goes from 12 × 4 blocks of 256 threads to 96 × 4 blocks of 32. At 4 streams the served grid already fills the card and the served kernel runs.
- `EXL3_HC_APPLY_WARP1`: the same one-warp re-grid for the hyper-connection `hc_apply` kernel.
- `EXL3_GR_STATE_REGRID`: the V2 mixer's state kernel spread over more blocks, ported to the int8 mixer path that the daily runs.

Each output keeps its arithmetic order in all three, so the output is the same bit for bit. A fourth round-5 selector, `EXL3_GR_FREE_UP_H`, frees nothing under the int8 mixer and was dropped.

## Kernel equality

`tests/test_r6_kernels.py` on each card, rows 1 to 16, `torch.equal` against the served kernel: 224 cases per card (32 GDN B/A, 64 `hc_apply`, 128 V2 state), 0 mismatches. A profiler pass confirmed that each candidate kernel launched (`kernels-gpu0.json`, `kernels-gpu1.json`).

## Serving A/B

One image, 8 boots in the order A B B A B A A B. A: the daily's environment. B: the same plus the three selectors. The NVMe tier is off in both arms. Per boot: c1 and 30k greedy fingerprints, then `bench/probe.py`, 2,048 forced tokens, greedy, one unrecorded warm-up round, then code c1 × 3, code c4 × 2 and prose c1 × 2.

| shape | A boot means (t/s) | B boot means (t/s) | B − A | 95 % CI |
| --- | --- | --- | --- | --- |
| code c1 | 224.1, 222.6, 223.5, 222.7 (mean 223.2) | 225.0, 225.8, 225.0, 225.8 (mean 225.4) | +0.99 % | +0.39 to +1.59 % |
| prose c1 | 198.0, 198.0, 197.6, 198.5 (mean 198.0) | 200.2, 200.3, 199.8, 200.4 (mean 200.2) | +1.10 % | +0.74 to +1.45 % |
| code c4, aggregate | 513.8, 510.4, 511.1, 508.4 (mean 510.9) | 510.9, 510.2, 508.6, 510.3 (mean 510.0) | −0.17 % | −0.93 to +0.58 % |

Per stream at c4: A 140.7, B 140.3 t/s. In every c4 run of both arms, 2 of the 4 requests stopped at 1,861 tokens with `finish_reason: stop` and the other 2 reached 2,048; the aggregate includes both. The composition is the same in all 16 c4 runs, so the comparison is between like runs.

Every boot of both arms produced c1 `e7fb377c987d685c` and 30k `4a255910dee2d9c5`, the daily's fingerprints. Free VRAM after boot was 1,257 / 289 MiB on every boot of both arms.

The attribution run in the spec (the B/A selector alone) was not run, so the split of the c1 gain between the three selectors is not measured. The bundle was promoted in [R540](r540-promote-r6.md).
