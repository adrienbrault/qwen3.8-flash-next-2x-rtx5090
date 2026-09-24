# R710, R710b: densegemm r1's gemv twin is bitwise-identical and 0.37 ms faster per iterate at 1 stream; its gemm and mgemm twins are 2.6 to 6.6 times slower

Results directories on the serving host: `results/2026-09-24-r710-densegemm-gate` and `results/2026-09-24-r710b-densegemm-gv`. Raw records of R710b: [`2026-09-24-r710b-densegemm-gv/`](2026-09-24-r710b-densegemm-gv/). Driver of R710b: [`scripts/r710b-densegemm-gv.sh`](../../scripts/r710b-densegemm-gv.sh). Image `tabbyapi:densegemm-r1` = `tabbyapi:stack-r2` plus [`docker/overlays/densegemm-r1`](../../docker/overlays/densegemm-r1/). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

## The change

densegemm r1 adds V2 twins of the dense EXL3 K=4 decode kernels (the GDN in and out projections, attention q/k/v and o, the QSA indexer's `index_qk`, the shared expert), selected by `EXL3_DENSE_V2`: mode 1 runs every twin, mode 2 the gemm and mgemm twins, mode 3 the gemv twin only. Each twin keeps its served kernel's arithmetic order. The levers are in [`ANALYSIS.md`](../../docker/overlays/densegemm-r1/ANALYSIS.md).

## R710 (2026-09-24 12:38 to 12:50 UTC)

The build landed with the served SASS unchanged and 32 twins added. Kernel parity: 4,240 of 4,240 comparisons equal (3,824 engaged a twin); stress and graph replay 0 mismatches.

P0, µs per call, served → V2: in_proj at 4 rows 26.3 → 104.1, at 16 rows 28.3 → 174.9; qkv at 4 rows 23.6 → 141.0; out_proj at 16 rows 20.0 → 77.5; index_qk at 16 rows 13.9 → 36.7; gate_up at 4 rows 13.7 → 38.5; shared.down at 16 rows 10.1 → 20.4. The gemv twin alone: out_proj at 4 rows 20.0 → 14.2, o_proj 19.1 → 14.0, index_qk 11.5 → 8.9, shared.down 8.9 → 5.8.

Both P1 arms (modes 1 and 2) carried the slow twins, so the unit was stopped after P0 and R710b gated mode 3 alone. The cause of the slow twins, from the densegemm r2 diagnosis ([`DIAGNOSIS.md`](../../docker/overlays/densegemm-r2/DIAGNOSIS.md)): each gemm and mgemm twin has a 544 to 1,832 B per-thread stack against 8 to 80 B served, because the in-loop gathered fixup is outlined and pulls the accumulator fragment into local memory. r1's install check read `LOCAL:0` from `cuobjdump -res-usage` as "no spills", but these frames are reported under `STACK`. The densegemm r2 and stack-r3 install checks bound `STACK` at the served kernel's value plus 64 B.

## R710b (2026-09-24 14:57 to 15:27 UTC)

Arms OFF (the served environment), GV (`EXL3_DENSE_V2=3`) and OFF2 (the served environment again, an A/A control), 6 rounds in rotated order, one container per run, at 4 streams depth 3, 8 streams depth 1 and 1 stream depth 3.

**Identity.** Parity 4,240 of 4,240, graph replay and stress 0 mismatches. The P1 sequence hashes of GV and OFF2 equal the round's OFF in 36 of 36 cells each, and OFF's hash is the same in every round. Greedy output on the served launcher with mode 3 on and off: identical to the reference on 6 of 6 prompts, including the 100k-token prompt; 0 out-of-memory lines. Free VRAM at the launcher's UP line: 1,041 / 2,431 MiB (reference and off) against 1,035 / 2,423 MiB (on).

**P1.** Values are GV − OFF in ms per iterate. The stall-excluded read drops every iterate slower than the run's median plus 8 ms from both runs of a pair ([`docs/PROMOTION.md`](../../docs/PROMOTION.md#the-stack-track-since-2026-09-24)).

| shape | raw mean | stall-excluded mean | per-iterate median | draft depth 0 cell |
| --- | --- | --- | --- | --- |
| 1 stream, depth 3 (4 verify rows) | −0.65, 5 of 6 (round 4 is a 48 ms stall in GV) | −0.365 (−2.9 %), 6 of 6 | −0.352, 6 of 6 | 1 row: −0.048, 6 of 6 |
| 4 streams, depth 3 (16 rows) | −0.30, 4 of 6 | −0.165, 5 of 6 | −0.145 | 4 rows: −0.381, 6 of 6 |
| 8 streams, depth 1 (16 rows) | +0.39, 2 of 6 | +0.038, 3 of 6 | +0.015 | 8 rows: −0.339, 6 of 6 |

OFF2 − OFF has mixed signs at every cell, with a mean of at most 0.06 ms in absolute value.

**Harness stall.** 27 of 54 drafting runs contain one iterate about 34 to 53 ms slower than the rest (OFF 9, GV 8, OFF2 10, so no arm effect). Position 2 of a round catches it in 14 of 18 runs against 6 and 7 at positions 1 and 3. The gate script predates the stall-excluded rule, so its own verdict read raw means and printed FLAT.

## Reading

- The gemv twin engages only on K=4 target calls at up to 8 rows: the served verify at 1 stream (4 rows) and 2 streams (8 rows). The MTP block is K=5 and K=6, so draft passes never engage it, and the verify at 4 and 8 streams is 16 rows. P0 at 4 rows predicts −301 µs per step on the critical path plus −138 µs of shared.down on the overlap stream; 1 stream depth 3 (−365 µs) and the depth-0 cell at 4 rows (−381 µs) land between the two.
- The 4-stream −0.17 ms is noise: no call engages the twin at 16 rows, the OFF2 pairs span ±0.38 ms, and without round 1 the mean is −0.09.
- Mode 3 was accepted as a stack entry on the stall-excluded read and then superseded by densegemm r2's mode 1, which contains the same gemv twin and is faster at every shape ([R714](r714-densegemm-r2.md)).
- Mode 3 uses one per-device counter array (`hctr`) with no per-stream copy, so it could not share an image with latchain's QSA graph fork ([R712](r712-latchain-r1.md)) until the side-branch guard of [stack-r3](../../docker/overlays/stack-r3/ANALYSIS.md) gave the fork its own scratch.
