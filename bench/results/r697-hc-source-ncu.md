# R697: the hyper-connection mixer chain is latency-bound on serial per-row activation loads (dots) and on the state prelude's reads of the previous kernel's output (up)

Results directory on the serving host: `results/2026-09-24-r697-hc-source-ncu`. Raw records: [`2026-09-24-r697-hc-source-ncu/`](2026-09-24-r697-hc-source-ncu/). Driver: `r697-hc-source-ncu.sh` (queued GPU-exclusive unit). Measurement only; nothing in the served configuration changed.

## What was measured

2026-09-24 07:01 to 07:02 UTC. The hyper-connection chain microbenchmark (`bench_hcfuse_p0.py --ncu --rows 4 16 --sites 3`, in the `tabbyapi:hcfuse-r1` image, whose reference variant is the served chain) ran under Nsight Compute with `--cache-control all --clock-control none` and the SourceCounters and WarpStateStats sections, on GPU 0. A kernel-name filter kept only the served kernels: `hc_apply`, `gr_v2_dots_i8` and `gr_v2_up_i8_state`. Per row count the run issues three unlabelled warm-up chains and one NVTX-labelled reference chain: 24 kernels. The labelled chain is raw IDs 9-11 at 4 rows and 21-23 at 16 rows ([`ncu-raw-nvtx.csv`](2026-09-24-r697-hc-source-ncu/ncu-raw-nvtx.csv)). The warm-up chains have the same SASS and stall profiles within sampling noise.

The first run, at 07:00, failed at `docker run` on an unset variable in the driver before any kernel was profiled ([`failed-run1-audit.txt`](2026-09-24-r697-hc-source-ncu/failed-run1-audit.txt)); the second run is the one reported.

| kernel | grid × threads, 4 rows | µs at 4 rows | warps active per SM | µs at 16 rows | warps active per SM |
| --- | --- | ---: | ---: | ---: | ---: |
| hc_apply | 80 × 32 | 2.1 | 2 % | 2.5 | 2 % |
| gr_v2_dots_i8 | 82×1×4 × 128 | 8.2 | 15 % | 16.0 | 31 % |
| gr_v2_up_i8_state | 320 × 256 | 9.3 | 30 % | 15.8 | 57 % |

Times are ncu durations with flushed caches and unlocked clocks. DRAM throughput is 14-24 % in all four mixer kernels, so none of them is bandwidth-bound.

## dots: B serial load rounds per iteration

dots is 48.9 % long_scoreboard at 4 rows and 51.5 % at 16 rows, and most of it falls on cycles where no other warp issues (47.7 % and 45.2 % of samples). The loop body issues the weight loads, then one stream-load pair per row, each behind its own row-bound branch and reusing the same registers, so row b+1's loads cannot issue before row b's multiplies have read them. Each column iteration therefore runs B serial load rounds (B = 4 at 4 rows, 8 at 16 rows).

| bucket | 4 rows | 16 rows |
| --- | ---: | ---: |
| round 0: four weight `LDG.E.64` plus row 0's stream pair, first consumed by an `I2F.S8` | 26.1 % | 12.5 % |
| rounds 1 to B−1: the per-row activation stream `LDG.E.128`, consumed by the next FMA | 20.2 % | 37.5 % |
| barrier before the cross-warp reduction | 12.4 % | 14.1 % |

The barrier share comes from warp imbalance: 320 column groups over 128 threads give warps 0-1 three iterations and warps 2-3 two, so a third of the loop is on the critical path of the slower warps.

## up: the state prelude waits on dots' output

up is 58.2 % long_scoreboard at 4 rows and 43.1 % at 16 rows. The largest bucket is the state prelude's serial reads of `dots`, the previous kernel's output: 28.1 % at 4 rows and 28.8 % at 16 rows. The int8 weight wait in the rank loop is 21.1 % at 4 rows and 6.8 % at 16 rows, because the weight words are first loaded only after the prelude barrier. At 16 rows 12.7 % is MIO pressure from the shuffle butterfly and the per-row shared-memory loads.

At 4 rows the warp count is set by the grid, not by registers: dots runs 7.4 warps per SM where its registers allow 28. At 16 rows up is already at its register limit of 4 CTAs per SM.

## Correction

The first reading of this run put the dots stall on the int8 weight conversion and its multiplies, and proposed wider int8 weight loads and split-K as levers. An independent review of the raw exports kept the percentages and the kernel mapping and corrected the producing loads: the hot multiplies wait on the per-row activation stream loads, not on the weight, and the weight loads are already four independent loads per iteration. In up, the largest stall is the prelude's reads of `dots`, not the weight. The review also narrowed the cache caveat: the weights come from DRAM in serving too, and the serial stream rounds are a served cost whose per-round latency the cache flush inflates. The sections above are the corrected reading.

## Levers, as ranked by the review

1. Issue all rows' stream loads of an iteration before the first FMA in dots (bitwise).
2. Shorten up's state prelude (bitwise).
3. Hide the weight's first use; worth at most about 0.9 µs per chain at 4 rows, the saving R687 measured with L2-warm weights (results `2026-09-24-r687-hcfuse-warm-ncu`), and nothing at 16 rows.
4. Even out the dots warps (not bitwise).
5. The int8 conversion.
6. up's MIO pressure at 16 rows.
7. More warps, at 4 rows only.

[hcfast r1](r698-hcfast.md) implements the row batching of lever 1 (fully at B ≤ 4), a restructured prelude with vector loads for lever 2, the early weight issue of lever 3 in both kernels, the single conversion per iteration of lever 5, and the transposed state with a reduce-scatter for lever 6. It leaves lever 4 alone, because every rebalancing it tried changes a thread's partial sums.

The 12 MB source-level CSV, the 10 MB `.ncu-rep` report and the 0.6 MB details page are not copied here; they stay in the results directory on the serving host.
