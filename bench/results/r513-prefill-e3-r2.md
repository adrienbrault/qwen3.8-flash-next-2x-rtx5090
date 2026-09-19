# R513: grouped MoE prefill round 2 (all K) is +16 % at 30k, +21 % at 60k and +19 % at 120k cold prefill, decode unchanged

Results directory on the serving host: `results/2026-09-18-r513-prefill-e3-r2`. Raw records: [`2026-09-18-r513-prefill-e3-r2/`](2026-09-18-r513-prefill-e3-r2/). Driver: [`scripts/r513-prefill-e3-r2.sh`](../../scripts/r513-prefill-e3-r2.sh). Date: 2026-09-18.

Round 2 extends E3 to K = 2, 3 and 4, so every MoE layer of the 2.50bpw pack is eligible. Image `tabbyapi:prefill-e3-r2` (overlay: [`docker/overlays/prefill-e3-r2/`](../../docker/overlays/prefill-e3-r2/)).

Per layer, real weights, OFF against ON on identical input:

| layer (card) | K | 512 rows | 2,048 rows |
| --- | --- | --- | --- |
| layers.12 (cuda:0) | 2 | ×1.13 | ×1.56 |
| layers.0 (cuda:0) | 3 | ×1.20 | ×1.57 |
| layers.36 (cuda:1) | 2 | ×1.19 | ×1.60 |
| layers.37 (cuda:1) | 3 | ×1.12 | ×1.50 |

Maximum absolute difference 1.6e-4 (NRMSE 4e-4), all outputs finite.

Cold prefill, salted, one invocation per context, mean of OFF + OFF2 against ON + ON2 (4 runs each):

| prompt tokens | OFF (t/s) | ON (t/s) | change |
| --- | --- | --- | --- |
| 30k | 7,558 | 8,740 | +15.6 % |
| 60k | 8,115 | 9,814 | **+20.9 %** |
| 120k | 8,543 | 10,163 | **+19.0 %** |

Decode flat: code c1 210.5 / 215.1 against 208.2 / 213.3 t/s, code c4 491.6 / 510.7 against 485.6 / 506.4. c1 fingerprint canonical on every arm; the 30k fingerprint moves to `4a255910dee2d9c5` under E3 (prefill accumulation order). GSM8K n=200 without stop strings: 0.985 OFF2 and 0.985 ON2. Needles 5/5. Stacked with decode round 4 and promoted in [R517](r517-promote-stack.md).
