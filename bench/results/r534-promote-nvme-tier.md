# R534: the NVMe prefix tier is served; every gate passes

Results directory on the serving host: `results/2026-09-19-r534-promote-nvme-tier`. Raw records: [`2026-09-19-r534-promote-nvme-tier/`](2026-09-19-r534-promote-nvme-tier/). Driver: [`scripts/r534-promote-nvme-tier.sh`](../../scripts/r534-promote-nvme-tier.sh). Date: 2026-09-19.

Image `tabbyapi:nvme-tier-r4`, validated in [R532](r532-nvme-tier-r4.md). The launcher mounts `/srv/qwen5090/fast/exl3-nvme-daily` as the tier with a 64 GiB cap (about 2.7 million tokens at the 24 KB per token measured in R526). It does so only when it launches the tier image with `NVME_TIER` unset. Another image, or `NVME_TIER=` (empty), gets no tier, so experiments that reuse the launcher never open the daily's directory: the tier removes stale namespaces when it opens. The unit refuses to promote if the filesystem cannot hold the cap and stay above 15 % free.

| gate | result |
| --- | --- |
| fingerprints | c1 `e7fb377c987d685c`, 30k `4a255910dee2d9c5`: equal to the daily before it |
| cold prefill, salted | `fn_bench --ctx` 60k target: 9,330 t/s; 120k target: 9,864 t/s |
| decode (one boot, against the [R522](r522-mtp-pruned.md) figures) | code c1 219.6 vs 221.8 t/s, c4 511.2 vs 508.4; prose c1 198.2 vs 194.5, c4 486.0 vs 484.4 |
| restore on the served configuration | 30k prompt, drain, `docker rm -f`, relaunch: 29,952 tokens from disk, 0.69 s against 3.96 s cold; output identical to warm and cold; 14 of 14 checkpoints intact |
| agentic-edit | 6/6 in four modes |
| needles | 5/5 at 131k and 5/5 at 240k |
| tool-eval 69×4 | 87.0 ± 1.2 |
| GSM8K 5-shot n=500, no stop strings | 0.976 |

Promoted 2026-09-19 06:22 UTC (08:22 CEST).
