# R421: the wide stage-B MoE tile at 128 slots or more is byte-identical and +7.5 % at c8

Results directory on the serving host: `results/2026-09-17-r421-coopwide-ab`. Driver: [`scripts/r421-coopwide-ab.sh`](../../scripts/r421-coopwide-ab.sh). Date: 2026-09-17.

[`docker/coopwide.patch`](../../docker/coopwide.patch) switches the MoE coop kernel's stage B to a wide tile when 128 or more expert slots are active. c1 byte-identical; code decode c1 / c4 / c8 aggregate 206–209 / 369–376 / 469–478 t/s, c8 +7.5 %. Promoted 2026-09-17 03:15 CEST.
