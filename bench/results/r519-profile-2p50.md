# R519: the 2.50 bpw decode step is 81–82 % kernel time; MoE kernels are the largest share at every shape

Results directory on the serving host: `results/2026-09-19-r519-profile-2p50`. Raw records: [`2026-09-19-r519-profile-2p50/`](2026-09-19-r519-profile-2p50/) (per-kernel tables, no traces). Driver: [`scripts/r519-profile-2p50.sh`](../../scripts/r519-profile-2p50.sh). Date: 2026-09-19.

The first decode profile of the 2.50 bpw pack; every earlier kernel budget was taken on the 3.05 bpw pack. Served image `tabbyapi:stack-r4-e3r2` with the served environment flags, split `[30, 30]`, 8-bit KV, a 1,024-token prompt, 64 captured decode steps after 64 warmup and 8 settle steps. Profiler mode adds about 8 % to the step; the wall-mode step is in brackets.

| shape | ms per step | tokens per step | kernel ms per step | non-kernel | MoE (coop) | fp16 GEMV / GEMM | hyper-connection mixer | GDN | attention | int8 GEMV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 stream, draft 3 | 14.99 (13.91) | 2.95 | 12.34 | 2.65 | 4.07 | 3.76 | 1.65 | 0.80 | 0.60 | 0.38 |
| 4 streams, draft 3 | 23.62 (21.54) | 11.45 | 19.19 | 4.43 | 7.58 | 4.64 | 2.85 | 1.71 | 0.93 | 0.03 |
| 6 streams, draft 1 | 21.80 (20.11) | 11.03 | 18.49 | 3.31 | 8.01 | 4.19 | 2.66 | 1.38 | 0.92 | 0.09 |

Kernel time is summed over both cards, which run their own layers in turn, so the sum is comparable to the step time.

- The step is kernel-bound: kernels fill 81–82 % of it. Launch overhead and host gaps are the other 2.6–4.4 ms, which is the most a whole-step CUDA graph could recover.
- The fused MoE kernels take 33 % of the step at 1 stream and 41–43 % at 4 and 6.
- The int8-activation single-row GEMV (`EXL3_INT8_GEMV`, on by default) costs 0.38 ms per step at 1 stream and almost nothing at 4, so switching it off can move 1-stream decode by a few percent at most.
