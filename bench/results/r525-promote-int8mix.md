# R525: int8 mixer weights and a 819,200-token pool are served; every gate passes

Results directory on the serving host: `results/2026-09-19-r525-promote-int8mix`. Raw records: [`2026-09-19-r525-promote-int8mix/`](2026-09-19-r525-promote-int8mix/). Driver: [`scripts/r525-promote-int8mix.sh`](../../scripts/r525-promote-int8mix.sh). Date: 2026-09-19.

Promoted 2026-09-19 02:15 UTC. The candidate is the [R517](r517-promote-stack.md) launcher with `EXL3_HC_MIX_V2_INT8=1` added and `CACHE=819200`. [R516](r516-int8-mixer-pool.md) measured the int8 mixer weights: they free 218 MiB on cuda:0 and 258 MiB on cuda:1, which is what lets the 819,200-token pool boot (+4 % over 786,432). Image unchanged: `tabbyapi:stack-r4-e3r2`.

| gate | result |
| --- | --- |
| G1 boot | 819,200 tokens at 8-bit KV, 4 slots; 1,973 / 1,057 MiB free on cuda:0 / cuda:1 |
| fingerprints | c1 `e7fb377c987d685c`, 30k `4a255910dee2d9c5`. The int8 weights change the c1 greedy output, as in R516; the 30k fingerprint is unchanged |
| G1v whole pool | 8/8 long requests filling the pool, 0 error lines, 1,215 / 569 MiB free under load |
| G1b decode (`fn_bench` 2,048 × 2 after a warm-up round, one boot, against the R517 reference) | code c1 224.4 vs 219.1 t/s (+2.4 %), code c4 503.2 vs 510.4 (−1.4 %), prose c1 194.5 vs 192.2 (+1.2 %), prose c4 484.4 vs 492.0 (−1.5 %). One boot per arm; [R520b](r520b-int8gemv-precise.md) puts the between-boot spread at 0.4–1.4 % |
| G2 agentic-edit | 6/6 in four modes. Greedy: 223.7 t/s at 1 stream; 502.2 aggregate / 143.1 per stream on the first wave of 4 streams. Sampled: 233.6 at 1 stream; 464.0 / 137.4 on the first wave of 4 |
| G3 needles | 5/5 at 131k and 5/5 at 240k |
| G4 tool-eval 69×4 | 85.0 ± 1.4 (trials 117 / 119 / 115 / 119) |
| G5 GSM8K 5-shot n=500, no stop strings | 0.974 (R509 0.978; R516 0.980 off / 0.978 on) |

The G1b cold-prefill figures (45,085 tokens in 2.20 s, 90,135 in 7.96 s) are not a measurement of cold prefill. `fn_bench --unique` seeds its filler with 1000 + the request index, the same on every run, so the prompt most likely shared cached pages with the whole-pool gate's requests earlier in the same boot: R517 took 4.72 s for the same 45k-token prompt. They passed the floor, but they are not published as prefill rates. The int8 mixer weights do not touch the MoE or attention prefill kernels.
