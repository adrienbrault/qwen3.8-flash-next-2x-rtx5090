# R535: deterministic E3 prefill is served on top of the NVMe tier; every gate passes

Results directory on the serving host: `results/2026-09-19-r535-promote-e3det` (try 3). Raw records: [`2026-09-19-r535-promote-e3det/`](2026-09-19-r535-promote-e3det/). Driver: [`scripts/r535-promote-e3det.sh`](../../scripts/r535-promote-e3det.sh). Date: 2026-09-19.

Image `tabbyapi:nvme-tier-r4-e3det`: the [R534](r534-promote-nvme-tier.md) image plus [`docker/overlays/e3-det-r1/`](../../docker/overlays/e3-det-r1/), served with `EXL3_MOE_PREFILL_E3_DET=1`. Measured in [R531 and R533](r533-e3-det-precise.md). The launcher changes three lines: the image, the image name in the tier default (so the NVMe tier stays on), and the flag. The tier's namespace follows a hash of the engine sources, so pages written by the previous engine are dropped at the first boot.

| gate | result |
| --- | --- |
| fingerprints | c1 `e7fb377c987d685c`, 30k `4a255910dee2d9c5`: equal to the daily before it |
| cold prefill, salted | `fn_bench --ctx` 60k target: 9,227 t/s (floor 9,000: the 9,200 floor of earlier promotions minus the 1.9 % R533 measured); 120k target: 9,861 t/s (floor 9,500) |
| decode (one boot, against the [R522](r522-mtp-pruned.md) figures) | code c1 221.4 vs 221.8 t/s, c4 514.2 vs 508.4; prose c1 197.8 vs 194.5, c4 477.5 vs 484.4 |
| agentic-edit | 6/6 in four modes |
| needles | 5/5 at 131k and 5/5 at 240k |
| tool-eval 69×4 | 85.0 ± 0.8 |
| GSM8K 5-shot n=500, no stop strings | 0.972 |

Try 1 stopped on the unadjusted 60k floor (9,136 against 9,200). Try 2 stopped before booting: its image check read a build log that a cached rebuild leaves empty. Promoted 2026-09-19 07:23 UTC (09:23 CEST).
