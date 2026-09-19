# R562: decode profile at 4 to 8 streams; above 16 verify rows the MoE layers run a generic kernel at 2.4–2.9× the time

Results directory on the serving host: `results/2026-09-19-r562-profile-c8`. Raw records (summary, per-kernel tables): [`2026-09-19-r562-profile-c8/`](2026-09-19-r562-profile-c8/). Driver: [`scripts/r562-profile-c8.sh`](../../scripts/r562-profile-c8.sh). Date: 2026-09-19.

Configuration: the image and flags of [R548](r548-promote-gdnbf16-ring.md), layer split `[30, 30]`, 8-bit KV, 1,024-token context, 64 captured decode steps per shape at a fixed draft depth; one wall-clock pass and one profiler pass. Kernel time is summed over both cards, which run their layers in turn. Rows per step = streams × (draft depth + 1).

| shape | rows | wall ms/step | tokens/step | t/s | MoE kernels | non-kernel | EXL3 GEMM | hyper-connection mixer | GDN | attention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 streams, depth 3 | 16 | 19.50 | 11.91 | 611 | cooperative 7.37 | 3.06 | 4.50 | 2.73 | 1.46 | 0.94 |
| 6 streams, depth 1 | 12 | 19.05 | 10.92 | 573 | cooperative 7.85 | 3.93 | 4.04 | 2.53 | 1.02 | 0.93 |
| 8 streams, depth 1 | 16 | 20.05 | 14.59 | 728 | cooperative 7.25 | 4.52 | 4.18 | 2.76 | 1.36 | 1.13 |
| 6 streams, depth 2 | 18 | 39.96 | 14.61 | 366 | generic 17.15 | 10.44 | 6.69 | 3.43 | 1.36 | 1.19 |
| 8 streams, depth 2 | 24 | 44.68 | 19.25 | 431 | generic 19.78 | 11.44 | 6.85 | 4.14 | 1.74 | 1.45 |
| 8 streams, depth 3 | 32 | 53.24 | 23.69 | 445 | generic 21.10 | 14.10 | 7.42 | 5.01 | 2.19 | 1.49 |

Times in ms per step; non-kernel time from profiler mode. Up to 16 rows the MoE layers run the cooperative v2 kernels inside a graph-captured block (`BC_BlockSparseMLP::run_bszN`, `MAX_BSZN = 16`); above it they run `exl3_moe_kernel<K, 128, 2, 16>` and the step also leaves the captured path (+6 to +10 ms outside kernels). At 8 streams and depth 1 a step makes about 1,570 kernel launches averaging 11.4 µs; the hyper-connection mixer contributes about 400 of them (4 kernels at each of ~100 sites) and the 36 GDN layers about 180. In every shape cuda:0 runs 2.5× the kernel time of cuda:1 (it holds more layers and the MTP draft); in a sequential layer split the imbalance does not add time.
