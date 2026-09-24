# R712: latchain r1 is bitwise-identical; the register-resident GDN recurrence is accepted, and all four levers together are 3.5 to 4.8 % faster per iterate once one harness stall is excluded

Results directory on the serving host: `results/2026-09-24-r712-latchain-r1`. Raw records: [`2026-09-24-r712-latchain-r1/`](2026-09-24-r712-latchain-r1/). Driver [`scripts/r712-latchain-r1.sh`](../../scripts/r712-latchain-r1.sh). Image `tabbyapi:stack-latchain-r1` = `tabbyapi:stack-r2` plus [`docker/overlays/latchain-r1`](../../docker/overlays/latchain-r1/). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

## The change

latchain r1 shortens three latency chains of the decode step, each behind its own flag and off by default ([`ANALYSIS.md`](../../docker/overlays/latchain-r1/ANALYSIS.md)):

- RR, `EXL3_LC_GDN_RR=1`: the Gated-DeltaNet recurrent kernel carries its state in registers across a job's tokens and prefetches the next token's inputs.
- QF, `EXL3_LC_QSA_FORK=1`: the QSA indexer chain is captured as a parallel CUDA-graph branch next to q/k/v, rope and cache append, and joined before the sparse split.
- GF, `EXL3_LC_GDN_FORK=1`: the GDN b/a GEMV is captured as a parallel branch next to the qkvz projection.
- QT, `EXL3_LC_QSA_SPLIT_STAGES=2 EXL3_LC_QSA_COMBINE_STAGES=1 EXL3_LC_QSA_DIV16=1`: compile options of the QSA sparse split and combine kernels, chosen by P0 among the variants that passed parity.

## What was measured

2026-09-24 13:49 to 14:57 UTC. Arms OFF (the served environment), RR, QF, GF, QT and ALL (the four together), 6 rounds in rotated order at 1 stream depth 3, 4 streams depth 3 and 8 streams depth 1, one container per run; each run also records the draft-depth-0 cell at the same batch.

**Identity.** Kernel parity 272 of 272 ([`parity.txt`](2026-09-24-r712-latchain-r1/parity.txt)). Logits-level model parity at batch 1 depth 3, batch 4 depth 3 and batch 8 depth 1: forward digests and tokens identical for OFF, OFF2 and ALL (`mp-*-compare.txt`). The P1 sequence hashes equal the round's OFF for every arm, round and cell (180 of 180), and OFF's hash is the same in all six rounds. Greedy output on the served launcher with RR on and off: identical to the reference on 6 of 6 prompts. ALL, QF, GF and QT had no served greedy test in this round.

**P0** ([`p0.txt`](2026-09-24-r712-latchain-r1/p0.txt)), µs per call, served → latchain: the GDN recurrent kernel 14.93 → 9.92 at 1 stream depth 3, 27.85 → 19.96 at 4 streams depth 3, 27.24 → 24.39 at 8 streams depth 1 (× 36 layers per step); the QSA split and combine 29.10 → 20.51, 53.66 → 37.68 and 54.89 → 40.21 at the same shapes with the QT options (× 12 layers).

**Harness stall.** About a third of the P1 runs contain one iterate about 34 ms slower than the rest, at a fixed index per shape (iterate 19 at 1 stream depth 3 in 16 runs; iterate 27 and 23 in the depth-0 cells at 4 and 8 rows). It adds about 1.07 ms per iterate to the run mean, more than any single lever. QT's runs caught it in 6 of 6 runs at 1 stream depth 3 (OFF 2 of 6, ALL 1 of 6). The table drops the stalled iterates from both runs of each pair.

| arm − OFF, ms per iterate, stall-excluded | 1 stream, depth 3 | 4 streams, depth 3 | 8 streams, depth 1 | depth-0 cells, 1 / 4 / 8 rows |
| --- | --- | --- | --- | --- |
| RR | −0.14, 4 of 6 | −0.50, 5 of 6 | −0.21, 6 of 6 | −0.09 at 8 rows |
| QF | −0.30, 6 of 6 | −0.15 | −0.10 | −0.22 / −0.24 / −0.22 |
| GF | −0.11, 6 of 6 | −0.24 | −0.14 | −0.06 at 4 rows |
| QT | −0.12, 5 of 6 | −0.13 | −0.24, 6 of 6 | −0.11 / −0.08 / −0.11 |
| ALL | −0.56 (−4.8 %), 6 of 6 | −0.76 (−3.8 %), 6 of 6 | −0.64 (−3.5 %), 6 of 6 | −0.36 / −0.38 / −0.39 |

The depth-0 cells are 6 of 6 wherever a value is given. The raw means, which include OFF's stalls, read ALL at −6.4 / −3.9 / −5.1 % and QT at +4.9 % at 1 stream depth 3; QT's figure is the stall. OFF rotated through positions 1, 6, 5, 4, 3 and 2 of its rounds, and the independent review found no position effect. nsys, one run per arm: 1 stream depth 3 13.05 → 12.37 ms per step and 8 streams depth 1 19.34 → 18.26 ms (OFF → ALL).

**Headroom.** The launcher's UP line read 1,041 / 2,431 MiB free on all three boots.

## Reading

- The gate's pre-registered verdict: RR accepted (8 streams, raw, 6 of 6); QF, GF and QT flat under the raw gain clause; ALL passes on speed but is a composition, not an entry, and had no served greedy test.
- On the stall-excluded read, QF gains at 1 stream in 6 of 6 and QT at 8 streams in 6 of 6; both went to the next batch gate as candidates, which admitted them only on its own identity tests and served A/B ([R716b](r716b-stack-r3.md)). GF was dropped: its marginal inside ALL is about 0.
- Composition blockers found here and fixed in stack-r3 ([`ANALYSIS.md`](../../docker/overlays/stack-r3/ANALYSIS.md)): latchain r1 and moefast r3 anchor on the same `bindings.cpp` line, so they cannot be applied to one tree (latchain r1b re-anchors that one line); and QF's side-branch GEMM can reach densegemm's V2 twins, whose scratch has no per-stream copy (the dense-lcguard patch gives the side branch its own).
