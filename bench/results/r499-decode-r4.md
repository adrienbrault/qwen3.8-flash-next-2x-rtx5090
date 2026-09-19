# R499: decode round 4 — pinned draft staging, batched verify and a 64K-token draft head give +2 to +3 % with identical output; int8 mixer weights give no speed

Results directory on the serving host: `results/2026-09-18-r499-decode-r4`. Raw records: [`2026-09-18-r499-decode-r4/`](2026-09-18-r499-decode-r4/). Driver: [`scripts/r499-decode-r4.sh`](../../scripts/r499-decode-r4.sh). Date: 2026-09-18.

Image `tabbyapi:decode-kernels-r4` (overlay: [`docker/overlays/decode-kernels-r4/`](../../docker/overlays/decode-kernels-r4/)), 2.50bpw pack at 786,432, arms OFF / ON / OFF2 / ON2 per group. Figures are the mean of the two OFF boots against the two ON boots.

**Group I**: `EXL3_DRAFT_PINNED_STAGING=1` (the draft chain's staging buffers in pinned memory, so the per-position copies are asynchronous), `EXL3_BATCH_VERIFY=1` (one readback for the verify step of all jobs), `EXL3_MTP_HEAD_N=65536` (the draft head scores the 65,536 most frequent tokens instead of the full 248,320-token vocabulary, an idea from [vcruz305's DGX Spark recipe][vcruz]). Fingerprints canonical on all four arms.

| load | OFF (t/s) | ON (t/s) | change |
| --- | --- | --- | --- |
| code c1 | 210.8 | 217.7 | +3.3 % (every ON run above every OFF run) |
| prose c1 | 186.9 | 190.7 | +2.0 % |
| code c4 | 497.9 | 510.5 | +2.5 % |
| prose c4 | 464.6 | 473.6 | +1.9 % |
| code c4, multiprompt 12 | 428.3 | 440.2 | +2.8 % |
| prose c4, multiprompt 12 | 363.8 | 367.6 | +1 % (within noise) |

VRAM: +120 MiB on cuda:1. Promoted in [R514](r514-promote-r4i.md).

**Group M**: `EXL3_HC_MIX_V2_INT8=1`, the hyper-connection mixer weights in int8 (the Spark recipe gains 4–7 % from it on a bandwidth-bound GB10). Output changes (c1 fingerprint differs). The isolated kernel runs at fp16 speed on both cards (10.3 against 10.3 µs at 1 row, 14.4 against 14.4 at 4, 28.7 against 27.7 at 16). Served: code c1 0 %, prose c1 −1.4 %, code c4 synthetic −2.9 % (the ON outputs stop earlier, median 1,954 of 2,048 tokens), multiprompt flat. It frees 218 MiB on cuda:0 and 258 MiB on cuda:1. Not a speed lever; its pool value is measured in R516.

[vcruz]: https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe
