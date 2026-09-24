# R696: the routed-expert MoE decode kernels wait on unprefetched activation loads (K2), a collapsed weight prefetch (K3) and a per-item handoff; there is no grid barrier

Results directory on the serving host: `results/2026-09-24-r696-moe-source-ncu`. Raw records: [`2026-09-24-r696-moe-source-ncu/`](2026-09-24-r696-moe-source-ncu/). Driver: `r696-moe-source-ncu.sh` (queued GPU-exclusive unit). Image `tabbyapi:slotfix-r1` with the served 23 environment keys, GPU 0. Measurement only; nothing in the served configuration changed.

## What was measured

2026-09-24 06:54 to 06:55 UTC. The d0 kernel microbenchmark (`--parts moe`, synthetic experts at K = 2, 3, 4 and 6, 1, 4 and 16 rows, 10 distinct experts per row) ran under Nsight Compute 2025.1.1 with `--cache-control all --clock-control none` and the SourceCounters and WarpStateStats sections, which sample warp stall reasons per SASS instruction. 32 kernels were profiled: an a kernel and a b kernel per cell, plus a rot kernel at 4 and 16 rows. The cells are listed in [`ncu-cells.jsonl`](2026-09-24-r696-moe-source-ncu/ncu-cells.jsonl) and the per-kernel metrics in [`ncu-raw-nvtx.csv`](2026-09-24-r696-moe-source-ncu/ncu-raw-nvtx.csv).

The served cells are K2 and K3 (the checkpoint's routed experts are K=3 in 23 of the 48 layers and K=2 in the rest). At 4 rows the synthetic cell has 40 distinct experts where a served step has 28; both give 40 slots, so the host launches the same kernels with the same grid and only the item count differs.

The source page lists each a and b kernel twice as consecutive identical blocks and each rot kernel once, in launch order, without NVTX labels. The mapping from blocks to cells was checked by matching every block's sample total to the raw page's `smsp__pcsamp_sample_count`: all 32 kernels match. The served-shape blocks are 5/7 (K2, 4 rows, a/b), 19/21 (K3, 4 rows, a/b) and 24/26 (K3, 16 rows, a/b).

## a kernel: 43-46 % long_scoreboard, on two different loads

| cell | long_scoreboard | activation `LD.E` | weight `LDG.E.EF` | metadata (runs, sel, rw, expert pointer) |
| --- | ---: | ---: | ---: | ---: |
| K2, 4 rows | 43.5 % | 37.0 % | 2.6 % | 1.8 % |
| K2, 16 rows | 44.1 % | 39.6 % | 1.3 % | 1.6 % |
| K3, 4 rows | 45.6 % | 8.8 % | 27.9 % | 5.8 % |
| K3, 16 rows | 46.1 % | 9.3 % | 30.5 % | 4.8 % |

Shares are of each kernel's warp-stall samples; the producing load of each stalled instruction is found by walking back through the SASS register dataflow.

- **K2:** the stalled `HMMA` instructions read their A operand from generic loads of the routed rows' activations (`A2[a_row0 + a_col]`), issued 40-59 instructions earlier in the same k-slice and never prefetched. The 4-deep weight prefetch works at K2.
- **K3:** the weight prefetch is collapsed by the compiler. The conditional K3 load compiles to a branch into a temporary register, and the `MOV` that copies it into the ring register waits on the load 20-25 instructions after it was issued, so the effective prefetch distance is about zero.
- Most of these samples coincide with another warp issuing. The long_scoreboard samples on cycles where no warp issued are 15.7-22.5 %, and that bounds what removing the stall can recover.
- DRAM throughput in this run: a kernel 35.7-45.1 % at 4 rows and 52.6-65.0 % at 16 rows.

## b kernel: the per-item handoff, not a grid barrier

The b kernel is not a cooperative launch. Per work item it runs one atomic and no spin loop. The samples at the `MEMBAR.SC.GPU` instruction carry the barrier stall reason: they are warps waiting at the `__syncthreads` before the fence, sampled on the next instruction, while one warp scans the token's 10 slots from global memory and waits on the returning atomic.

| cell | handoff (2 barriers + 2 `__threadfence`) | of which the two fences | k-split reduction barrier |
| --- | ---: | ---: | ---: |
| K2, 4 rows | 26.6 % | 8.8 % | 8.7 % |
| K3, 4 rows | 23.3 % | 7.7 % | 10.3 % |
| K2, 16 rows | 14.4 % | 5.4 % | 2.0 % |
| K3, 16 rows | 12.8 % | 4.8 % | 2.3 % |

The drop from 4 to 16 rows comes from the tile: at 16 rows the b kernel runs the 128-column wide tile, which moves four times the weight bytes per handoff over the same 3,200 handoffs. The reduction barrier at 4 rows is a structural imbalance: 40 k-slices over 16 k-warps gives 13 warps three slices, one warp one slice and two warps none.

The rot kernel is 2.9-4.0 % of rot + a + b duration.

## Correction

The first reading of this run attributed the a kernel's stalls to weight loads issued too close to their use, with deeper weight prefetch as the lever, and the b kernel's to a grid barrier and its GPU-scope fence, with a counter handoff as the lever. An independent review of the raw exports corrected both: the K2 a kernel waits on activation loads, the K3 a kernel on a weight prefetch the compiler collapsed, and the b kernel already uses a counter handoff and has no grid barrier. It also found that the first pass had mapped four source blocks to the wrong cells, and that the analysis script attributed barrier stalls to the instruction after the barrier. The tables above are the corrected reading.

## Levers, as passed to the moefast round

1. Stage each item's activation rows in shared memory once, or carry the A fragment in the prefetch ring; this also makes the separate rot kernel removable.
2. Take the handoff off the 16-warp critical path in the b kernel at 4 rows: precompute each row's active-slot count, fence only in the thread that issues the atomic, and move the second fence into the last-arrival branch.
3. Make the K3 prefetch load write its ring register directly, or stage the weights through cp.async.
4. Shorten the per-item metadata chain (runs → slot table → expert pointer → weights), up to 18.9 % of samples in the K3 b kernel at 4 rows, an upper bound because the routing tables are cold under the cache flush.
5. Remove the k-split imbalance at 4 rows.

[moefast r1](r700b-moefast.md) implemented the cp.async weight ring (lever 3), a one-slice activation prefetch (part of lever 1) and the single counter arrival (part of lever 2).

The 45 MB source-level CSV, the 44 MB `.ncu-rep` report and the 0.9 MB details page are not copied here; they stay in the results directory on the serving host.
