# R428: mixer V2 and the GDN host-gap rewind are identical at MIN_R 1 and +12 % at c4, +8 % at c8

Results directory on the serving host: `results/2026-09-17-r428-hcmix2-stack-ab`. Driver: [`scripts/r428-hcmix2-stack-ab.sh`](../../scripts/r428-hcmix2-stack-ab.sh), [`scripts/r428-promote-hcmix2.sh`](../../scripts/r428-promote-hcmix2.sh). Date: 2026-09-17.

[`docker/hc-mix-v2-r2.patch`](../../docker/hc-mix-v2-r2.patch) is a bit-exact rewrite of the hyper-connection mixer kernels (`EXL3_HC_MIX_V2=1`, `EXL3_HC_MIX_V2_MIN_R=1`); [`docker/hostgap-gated_delta_net.py`](../../docker/hostgap-gated_delta_net.py) removes host-side gaps from the GDN decode path (`EXL3_HOST_GAP_REWIND=1`). Greedy output identical at MIN_R 1. Code decode c1 / c4 / c8 aggregate 213–218 / 422–434 / 537–548 t/s: c4 +12 %, c8 +8 %. Promoted 2026-09-17 04:32 CEST.
