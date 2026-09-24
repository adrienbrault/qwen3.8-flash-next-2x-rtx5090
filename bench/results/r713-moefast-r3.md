# R713: moefast r3 is bitwise-identical; forking the shared expert before the router is 1.2 % faster per iterate at 8 streams in 6 of 6 rounds and 3.2 % at 1 stream in 5 of 6

Results directory on the serving host: `results/2026-09-24-r713-moefast-r3`. Raw records: [`2026-09-24-r713-moefast-r3/`](2026-09-24-r713-moefast-r3/). Driver [`scripts/r713-moefast-r3.sh`](../../scripts/r713-moefast-r3.sh). Image `tabbyapi:stack-moefast-r3` = `tabbyapi:stack-r2` plus [`docker/overlays/moefast-r3`](../../docker/overlays/moefast-r3/), built inside the GPU lock; served SASS identical, 36 r2 kernels and 1 head kernel added. Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

## The change

moefast r2 (`EXL3_MOE_COOP_V3=3`) rewrote the routed-expert MoE decode kernels. Its gate, R703b (results `2026-09-24-r703b-moefast-r2`, not written up here), measured it bitwise-identical and flat against the served mode 2 at every stack shape: the r2 kernels are faster alone, but the side-stream shared expert, which the served path starts before kernel A, then starts after it, and kernel B waits for it (nsys, 1 stream depth 3: A → B gap 1.9 → 14.1 µs).

moefast r3 keeps r2's kernels and restores that overlap ([`ANALYSIS.md`](../../docker/overlays/moefast-r3/ANALYSIS.md)). The measured arm R3 is `EXL3_MOE_COOP_V3=3 EXL3_MOE_COOP_V3_MAP=2-4:2 EXL3_SHARED_EXPERT_EARLY=1`: r2 kernels at 1 and 5 to 16 rows, the served mode 2 at 2 to 4 rows, and the shared expert forked onto its side stream before the router instead of after it.

## What was measured

2026-09-24 15:27 to 16:20 UTC. Arms OFF (the served environment with `EXL3_MOE_COOP_V3=0`), M2 (the served 27 keys exactly) and R3, 6 rotated rounds after a discarded warm-up, at 1 stream depth 3, 4 streams depth 3 and 8 streams depth 1, one container per run.

**Identity.** Kernel parity 10,208 of 10,208; side-stream arms 231 of 231; join stress 200 of 200 ([`parity.txt`](2026-09-24-r713-moefast-r3/parity.txt)). The P1 hashes of R3 and M2 equal the round's OFF in 36 of 36 cells each (drafting and depth-0 cells), and each cell has one hash across all rounds. Greedy output on the served launcher with the three R3 keys on and off: identical to the reference on 6 of 6 prompts, including the 100k-token prompt. That greedy set runs 1 stream (target rows 1 and 4), and the MAP sends 4 rows to mode 2, so the mode-3 kernels (5 to 16 rows) have no served greedy test in this round; their identity rests on parity and the 16-row P1 hashes.

**P1.** R3 − M2, ms per iterate; the stall-excluded read drops iterates slower than the run median plus 8 ms from both runs of a pair (2 to 4 of 192 per cell).

| shape | stall-excluded mean | raw median | per-iterate median | draft depth 0 cell |
| --- | --- | --- | --- | --- |
| 1 stream, depth 3 | −0.38 (−3.2 %), 5 of 6 | −0.20, 4 of 6 | −0.42, 5 of 6 | 1 row: −0.24 (−3.0 %), 6 of 6 |
| 4 streams, depth 3 | −0.21 (−1.1 %), 4 of 6 | −0.37, 4 of 6 | −0.15, 4 of 6 | 4 rows: −0.42 (−4.2 %), 6 of 6 |
| 8 streams, depth 1 | −0.22 (−1.2 %), 6 of 6 | −0.27 (−1.46 %), 6 of 6 | −0.18, 5 of 6 | 8 rows: −0.41 (−3.4 %), 6 of 6 |

The depth-0 cell at 4 rows is 6 of 6 once its iterate-27 stalls are dropped. M2 − OFF reproduces R703b (stall-excluded −0.83, −0.58 and −0.48 ms at the three shapes; R703b read −0.76, −0.56 and −0.60).

**Mechanism** (nsys, one run per arm, [`timeline-c1d3.txt`](2026-09-24-r713-moefast-r3/timeline-c1d3.txt), [`timeline-c4d3.txt`](2026-09-24-r713-moefast-r3/timeline-c4d3.txt)). At 1 stream depth 3 (4 rows, mode 2 under the MAP) the shared expert starts 14.6 µs before kernel A instead of 3.6, A drops from 38.2 to 31.0 µs and the MoE layer from 70.3 to 62.9 µs; −7.5 µs × 48 layers is −0.36 ms per step, against P1's −0.34 to −0.38. At 1 row (the depth-0 cell at batch 1) no cooperative mode runs, so its −0.24 ms is the early fork alone. At 16 rows mode 3 does not pay: MoE time 7.32 ms per step against M2's 7.11 at 4 streams depth 3.

**Headroom.** The launcher's UP line read 1,041 / 2,431 MiB free on the reference, on and off boots; 0 out-of-memory lines, 0 tracebacks.

## Reading

- Accepted as a stack entry on 8 streams depth 1 (6 of 6) and the three depth-0 cells (6 of 6 each), with no same-sign regression. The gain is the early fork, not the r2 kernels.
- The same flags against OFF read −9.1 % at 1 stream depth 3 and −11 % in the depth-0 cells, but that comparison includes moefast r1, which the served configuration already had.
- Served from stack-r3 on ([R716b](r716b-stack-r3.md)), whose multi-shape identity cells cover the mode-3 row counts this round did not greedy-test. Open: an arm with mode 2 plus the early fork alone, and a `13-16:2` MAP entry, since mode 3 is slower than mode 2 at 16 rows in nsys.
