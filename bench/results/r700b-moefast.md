# R700b: moefast r1 is bitwise-identical; mode 2 is 5 % faster at 4 streams and 3 % at 8

Results directory on the serving host: `results/2026-09-24-r700b-moefast-gate`. Raw records: [`2026-09-24-r700b-moefast-gate/`](2026-09-24-r700b-moefast-gate/). Driver: `r700-moefast-gate.sh` (queued GPU-exclusive unit; the steps are in [`HOW-TO-VERIFY.md`](../../docker/overlays/moefast-r1/HOW-TO-VERIFY.md)). Image `tabbyapi:moefast-r1` = `tabbyapi:slotfix-r1` plus [`docker/overlays/moefast-r1`](../../docker/overlays/moefast-r1/). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

The first queued unit, R700, was stopped before it took the GPU lock: both of its landing checks imported `exllamav3_ext` without importing torch first and failed on `libc10.so`. R700b is the same unit with the checks fixed.

## The change

`EXL3_MOE_COOP_V3=1|2`, read per launch on top of the served `EXL3_MOE_COOP_V2=1`, for batches above one row at K = 2 to 4, routes the routed-expert MoE decode kernels through `exl3_moe_coop_v3_kernel.cuh`:

- the weight prefetch is a cp.async ring with a counted wait, so the K=3 prefetch no longer collapses into a register copy right after issue ([R696](r696-moe-source-ncu.md));
- the activation fragment for the next k-slice is loaded one slice ahead;
- the per-item completion is one acquire/release counter arrival instead of the block-wide `__threadfence` pairs;
- mode 2 also merges the narrow down-projection stage per 128-column chunk.

Unset, the served V2 kernels run, and their SASS is unchanged by the patch. The design is in [`ANALYSIS.md`](../../docker/overlays/moefast-r1/ANALYSIS.md).

## What was measured

2026-09-24 07:39 to 07:55 UTC.

**Parity** ([`parity.txt`](2026-09-24-r700b-moefast-gate/parity.txt)): 2,352 of 2,352 comparisons storage-bit equal, over rows 1 to 16, K = 2, 3 and 4, four routings, the shared expert on and off, real weights from a K=2 and a K=3 layer plus synthetic K2/K3/K4, and 200 back-to-back launches per mode at 4, 12 and 16 rows. The dispatch check confirmed the V3 kernels run for modes 1 and 2 and the V2 kernels for mode 0.

**P0, kernel microbenchmark** (real weights, CUDA-event medians over 5 rotating rounds, µs for rot + a + b; ratios against the served kernels; D is the number of distinct experts in the call). The served cells are 4 rows with 28 experts (1 stream, depth 3) and 16 rows with 77 experts (4 streams, depth 3, and 8 streams, depth 1):

| cell | reference µs | mode 1 | mode 2 |
| --- | ---: | ---: | ---: |
| K2, 4 rows, D28 | 66.66 | 0.978 | 0.845 |
| K3, 4 rows, D28 | 78.98 | 0.903 | 0.793 |
| K2, 16 rows, D77 | 129.73 | 0.904 | 0.903 |
| K3, 16 rows, D77 | 156.77 | 0.852 | 0.848 |
| K2, 1 row, D10 (control) | 34.14 | 1.009 | 1.000 |
| K3, 1 row, D10 (control) | 40.38 | 0.998 | 1.000 |

The largest reductions are mode 2's merged down stage at 4 to 12 rows (K3, 8 rows, D40: b kernel 54.6 → 31.2 µs, total 0.697) and the a kernel at 16 rows (K3, 16 rows, D77: 90.2 → 73.5 µs). From these cells the unit predicted a step saving for mode 2 of 0.63 ms at 1 stream and 0.87 ms at 4 and 8 streams. All 20 cells are in [`p0.txt`](2026-09-24-r700b-moefast-gate/p0.txt) and [`p0-summary.json`](2026-09-24-r700b-moefast-gate/p0-summary.json).

**P1, in-process harness** (4,096 tokens of context, untraced, `EXL3_MOE_COOP_V3=2`, ms per iterate, OFF → ON):

| shape | pair 1 | pair 2 | mean |
| --- | --- | --- | --- |
| 4 streams, depth 3 | 22.24 → 21.40 | 21.21 → 19.84 | −1.10 ms (−5.1 %) |
| 8 streams, depth 1 | 19.41 → 18.91 | 19.47 → 18.73 | −0.62 ms (−3.2 %) |
| 1 stream, depth 3 | 13.12 → 13.19 | 12.85 → 13.17 | +0.20 ms (+1.5 %) |

The generated-sequence hashes were identical in all 6 pairs. Greedy output on the served launcher, with the flag on and with it off, was identical to the reference on 6 of 6 prompts.

The source-level ncu pass of the V3 kernels (step 2b) did not run: its `docker run` lacked `--cap-add SYS_ADMIN` and ncu stopped on `ERR_NVGPUCTRPERM` ([`p0-ncu.txt`](2026-09-24-r700b-moefast-gate/p0-ncu.txt)). It was diagnostic only.

## Reading

Mode 2 is faster at 4 and 8 streams with both pairs the same sign. At 1 stream it reads 1.5 % slower on both pairs, against P0's predicted 0.63 ms saving. Two pairs are too few to separate a regression from harness noise at 1 stream, so mode 2 went to the stack gate, whose five rounds apply the no-regression rule to every shape. There ([R701](r701-stack-r2.md)) the 1-stream reading did not reproduce: the stack with mode 2 was faster than OFF at 1 stream in all 5 rounds (−11.6 %).
