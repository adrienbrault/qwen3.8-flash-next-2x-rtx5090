# R516: int8 mixer weights buy a 819,200-token pool (+4 %) at unchanged quality and 0 to −2 % decode

Results directory on the serving host: `results/2026-09-19-r516-int8-mixer-pool`. Raw records: [`2026-09-19-r516-int8-mixer-pool/`](2026-09-19-r516-int8-mixer-pool/). Driver: [`scripts/r516-int8-mixer-pool.sh`](../../scripts/r516-int8-mixer-pool.sh). Date: 2026-09-19.

`EXL3_HC_MIX_V2_INT8=1` stores the hyper-connection mixer weights in int8 ([R499](r499-decode-r4.md) group M), which frees 218 MiB on cuda:0 and 258 MiB on cuda:1. R516 measured what that VRAM buys as page pool and whether quality holds. Image `tabbyapi:decode-kernels-r4`, 2.50bpw pack, 4 slots, decode round 4 group I and E3 off in every arm.

Pool ladder: 835,584 does not boot ("Insufficient VRAM in split for model and cache"); **819,200 boots and survives a c4 stress pass** (1,927 / 1,135 MiB free). Arms OFF at 786,432 / ON at 819,200 / OFF2 / ON2. Fingerprints: OFF canonical, ON `e7fb377c987d685c` / `4a255910dee2d9c5` (the int8 weights change the numerics).

| load (`fn_bench` 2,048 × 2, mean aggregate) | OFF (t/s) | ON (t/s) | change |
| --- | --- | --- | --- |
| code c1 | 210.7 | 211.7 | +0.5 % |
| code c4 | 499.5 | 489.4 | −2.0 % (ON code outputs stop at a median 1,954 of 2,048 tokens) |
| prose c1 | 186.8 | 185.2 | −0.8 % |
| prose c4 | 466.1 | 461.1 | −1.1 % |
| multiprompt c4 code, OFF / OFF2 against ON / ON2 | 421.4 / 433.9 | 434.3 / 437.0 | flat |
| multiprompt c4 prose | 390.2 / 398.0 | 404.8 / 380.5 | flat |

| quality gate | OFF | ON |
| --- | --- | --- |
| GSM8K 5-shot n=500 without stop strings | 0.980 | 0.978; paired: only OFF right 3, only ON right 2 (z 0.45) |
| needles 131k / 240k | — | 5/5 / 5/5 |
| tool-eval 69×4 | — | 85.5 ± 1.3 (trials 116 / 117 / 120 / 119) |

The served image already carries the int8 kernels, so serving it is a flag and a pool size on top of whatever [R518](../../scripts/r518-slots6.sh) leaves as the served slot count.
