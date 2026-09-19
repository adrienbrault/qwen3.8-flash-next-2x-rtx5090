# R492: decode is nearly flat with prompt depth; cold prefill 8.2–8.4k t/s at 100–180k tokens

Results directory on the serving host: `results/2026-09-18-r492-depth`. Raw records: [`2026-09-18-r492-depth/`](2026-09-18-r492-depth/). Driver: [`scripts/r492-depth.sh`](../../scripts/r492-depth.sh). Date: 2026-09-18.

Served configuration of the day (3.05bpw pack, 360,448, 4 slots), one run per arm, 1,024 forced tokens, unique prompts.

| arm | prompt tokens | decode (t/s) |
| --- | --- | --- |
| prose c1 | 0 / 99,919 / 199,457 | 172.7 / 152.8 / 169.2 |
| code c1 | 0 / 179,258 | 192.4 / 176.5 |
| prose c4, per stream | 4 × 60k | 100.4 |
| code c4, per stream | 4 × 108k | 76.7 (433k tokens overflow the pool, so requests queued) |

Cold prefill: 99,919 prose tokens in 11.95 s = 8,359 t/s; 179,258 code tokens in 21.96 s = 8,164 t/s. The 199,457-token prose prompt shared its first 100k tokens with the 99,919-token one, so its 12.57 s TTFT is a prefix hit and not a cold figure. Code filler tokenizes about 1.35× denser than prose.
