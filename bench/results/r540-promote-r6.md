# R540: decode kernels round 6 are served; every gate passes

Results directory on the serving host: `results/2026-09-19-r540-promote-r6`. Raw records: [`2026-09-19-r540-promote-r6/`](2026-09-19-r540-promote-r6/). Driver: [`scripts/r540-promote-r6.sh`](../../scripts/r540-promote-r6.sh). Date: 2026-09-19.

Image `tabbyapi:nvme-tier-r4-e3det-r6`: the [R535](r535-promote-e3det.md) image plus [`docker/overlays/decode-kernels-r6/`](../../docker/overlays/decode-kernels-r6/), served with `EXL3_GDN_BA_WARP1=1 EXL3_HC_APPLY_WARP1=1 EXL3_GR_STATE_REGRID=1`. Measured in [R538](r538-decode-r6.md): kernel equality 224 of 224 cases on each card, fingerprints unchanged on 8 boots, code c1 +0.99 % and prose c1 +1.10 % with 95 % intervals above zero, code c4 −0.17 % (interval −0.93 to +0.58 %). The launcher changes three lines: the image, the image name in the tier default (so the NVMe tier stays on), and the three flags. The tier's namespace follows a hash of the engine sources, so the first boot opened a new, empty namespace.

| gate | result |
| --- | --- |
| fingerprints | c1 `e7fb377c987d685c`, 30k `4a255910dee2d9c5`: equal to the daily before it |
| cold prefill, salted | `fn_bench --ctx` 60k target (45,037 prompt tokens): 9,131 t/s (floor 9,000); 120k target (90,097 prompt tokens): 9,834 t/s (floor 9,500) |
| decode (one boot, 2 runs after a warm-up round, against the [R522](r522-mtp-pruned.md) figures) | code c1 220.1 vs 221.8 t/s, c4 513.2 vs 508.4; prose c1 200.1 vs 194.5, c4 481.3 vs 484.4. In both code c4 runs 2 of the 4 requests stopped at 1,861 tokens and the aggregate includes them |
| agentic-edit | 6/6 in four modes (greedy and sampled, c1 and c4) |
| needles | 5/5 at 131k and 5/5 at 240k |
| tool-eval 69×4 | 87.2 ± 1.5 (95 % interval 85.8 to 88.0) |
| GSM8K 5-shot n=500, no stop strings | 0.974 |

Free VRAM after the promotion boot: 1,969 / 729 MiB. Promoted 2026-09-19 09:11 UTC (11:11 CEST). Rollback: the R535 launcher, or the r6 image with the three flags removed, which launches the R535 kernels.
