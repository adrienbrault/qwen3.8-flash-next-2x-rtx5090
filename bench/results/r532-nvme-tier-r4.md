# R532: the NVMe tier on the served chain passes every step; with the tier idle, decode is unchanged (8 boots)

Results directory on the serving host: `results/2026-09-19-r532-nvme-tier-r4`. Raw records: [`2026-09-19-r532-nvme-tier-r4/`](2026-09-19-r532-nvme-tier-r4/). Driver: [`scripts/r532-nvme-tier-r4.sh`](../../scripts/r532-nvme-tier-r4.sh). Date: 2026-09-19.

Round 4 of the tier ([`docker/overlays/nvme-tier-r4/`](../../docker/overlays/nvme-tier-r4/)) installs on the served image `tabbyapi:mtp-pruned-r1-tc1-plefix`. The tier and the pruned-draft overlay both replace `generator.py`; the round-4 file is the pruned-draft generator plus the tier's five blocks, and a CPU test shows that removing the tier's statements leaves exactly the pruned-draft file. The installer refuses a base without the PLE checkpoint fix. Every boot here uses the daily's environment unchanged, so the device draft chain is on.

| step | result |
| --- | --- |
| tier off on the new image | c1 `e7fb377c987d685c`, 30k `4a255910dee2d9c5`: the daily's fingerprints |
| tier on, cold | same fingerprints |
| crash restart, then the same prompts | 30k: 29,952 tokens from disk, 0.69 s (cold 4.09 s). 120k: 119,808, 0.99 s (cold 12.33 s). Restored = warm = cold outputs |
| reopen | 12 of 12 checkpoints intact in 1.7 s; 0 refused |
| decode while a 120k prefix drains | −3.8 % at 1 stream, −3.0 % at 4, for the 26 s the drain lasts |
| cap 2 GiB, four 40k prompts | at most 1.98 GiB after each drain |

Idle-tier decode against the daily, 8 boots in the order A B B A B A A B (A: the daily as served; B: the tier image with the tier on and populated), `fn_bench` code, warm-up round then 3 runs at 1 stream and 2 at 4:

| | daily | tier on, idle | difference, 95 % interval |
| --- | --- | --- | --- |
| 1 stream | 222.1 t/s | 221.7 t/s | −0.21 % [−1.15, +0.74] |
| 4 streams | 509.0 t/s | 509.6 t/s | +0.11 % [−1.04, +1.26] |

All 8 boots produced the canonical c1 fingerprint.
