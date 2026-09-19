# R528: the MTP draft chain on the GPU with a pruned embedding copy is served; every gate passes

Results directory on the serving host: `results/2026-09-19-r528-promote-mtp-pruned`. Raw records: [`2026-09-19-r528-promote-mtp-pruned/`](2026-09-19-r528-promote-mtp-pruned/). Driver: [`scripts/r528-promote-mtp-pruned.sh`](../../scripts/r528-promote-mtp-pruned.sh). Date: 2026-09-19.

Promoted 2026-09-19 03:16 UTC. The candidate is the [R525](r525-promote-int8mix.md) launcher with image `tabbyapi:mtp-pruned-r1` ([`docker/overlays/mtp-pruned-r1/`](../../docker/overlays/mtp-pruned-r1/)) and `EXL3_MTP_DEVICE_DRAFT=1 EXL3_EMBED_GPU=1 EXL3_EMBED_GPU_PRUNED=1`. [R522 and R522b](r522-mtp-pruned.md) showed the output byte-identical and the speed-up resolved over 8 counterbalanced boots: +1.99 % at c1, +1.49 % at c4.

| gate | result |
| --- | --- |
| G1 boot | 819,200 tokens at 8-bit KV, 4 slots; the 320 MiB embedding copy on cuda:1; 1,973 / 737 MiB free on cuda:0 / cuda:1 |
| fingerprints | c1 `e7fb377c987d685c`, 30k `4a255910dee2d9c5`: equal to the daily before it |
| G1v whole pool | 4 × 60k and 4 × 190k concurrent, 8/8 ok, 0 error lines, 1,173 / 225 MiB free under load |
| G1b cold prefill, salted | requested 60k (45,110 prompt tokens): 9,841 t/s; requested 120k (90,135): 10,278 t/s |
| G1b decode (`fn_bench` 2,048 × 2 after a warm-up round, one boot, against R522b arm A) | code c1 228.9 vs 217.4 t/s (+5.3 %), code c4 507.0 vs 500.9 (+1.2 %), prose c1 198.4 vs 194.5 (+2.0 %), prose c4 481.8 vs 484.4 (−0.5 %). One boot; R522b's 8-boot figures are the measurement |
| G2 agentic-edit | 6/6 in four modes. Greedy: 229.2 t/s at 1 stream over the whole run; 498.6 aggregate / 133.8 per stream on the first wave of 4 streams |
| G3 needles | 5/5 at 131k and 5/5 at 240k |
| G4 tool-eval 69×4 | 84.8 ± 1.5 (trials 114 / 118 / 119 / 116) |
| G5 GSM8K 5-shot n=500, no stop strings | 0.970 (R525 0.974 with byte-identical greedy decode; the difference is E3 prefill run-to-run variation on long prompts and concurrency, 2 of 500 questions) |

cuda:1 keeps 225 MiB free with the whole pool in flight, the least headroom of any served configuration so far.
