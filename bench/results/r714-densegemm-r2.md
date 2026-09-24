# R714: densegemm r2's twins no longer run out of local memory; mode 1 is bitwise-identical and 5.2 / 2.2 / 2.1 % faster per iterate at 1 / 4 / 8 streams, 6 of 6 rounds each

Results directory on the serving host: `results/2026-09-24-r714-densegemm-r2`. Raw records: [`2026-09-24-r714-densegemm-r2/`](2026-09-24-r714-densegemm-r2/). Driver [`scripts/r714-densegemm-r2.sh`](../../scripts/r714-densegemm-r2.sh). Image `tabbyapi:densegemm-r2` = `tabbyapi:stack-r2` plus [`docker/overlays/densegemm-r2`](../../docker/overlays/densegemm-r2/) (`densegemm-r2.patch`, SHA-256 `2bc1580c…`), built with no extra defines. Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

## The change

densegemm r2 keeps r1's gemv twin and rebuilds the gemm and mgemm twins that [R710](r710b-densegemm-gv.md) measured 2.6 to 6.6 times slower than served: the gathered fixup moves after the main loop to one call site, the partition arithmetic is 32-bit, and the helpers are force-inlined. The diagnosis and the design are in [`DIAGNOSIS.md`](../../docker/overlays/densegemm-r2/DIAGNOSIS.md). Before any GPU time the build runs a compile-time bisect over the candidate defines and checks every twin's `STACK` from `cuobjdump -res-usage` against its served kernel plus 64 B ([`bisect.txt`](2026-09-24-r714-densegemm-r2/bisect.txt), [`compile-bar.txt`](2026-09-24-r714-densegemm-r2/compile-bar.txt)).

Modes: GM, `EXL3_DENSE_V2=2`, runs the gemm and mgemm twins; ON, `EXL3_DENSE_V2=1`, runs every twin including the gemv. `EXL3_DENSE_ROWS32=1` (17 to 32 rows) was off in P1.

## What was measured

Queued 13:33 UTC, GPU lock 2026-09-24 16:20 to 17:03 UTC. Arms OFF (the served environment), GM and ON, 6 rotated rounds at 4 streams depth 3, 8 streams depth 1 and 1 stream depth 3.

**Build and identity.** The r1 probe fails the stack bar (1,072 to 1,832 B); every r2 variant passes, and the choice is plain r2 with no defines (`BISECT-CHOICE r2 none`). 22 of 24 gemm and mgemm twins have an 8 B stack and no spill; the two TN256 rows32 mgemm twins have a 64 B stack against 56 to 80 B served at 16 rows. Served SASS identical, 32 twins added. Kernel parity 4,240 of 4,240, including 17, 24 and 32 rows against the served two-pass kernel. P1 hashes 36 of 36 per arm, with one hash per cell across all 18 runs. Greedy output on the served launcher, off, GM and ON: identical to the reference on 6 of 6 prompts including the 120k-token prompt, 0 out-of-memory lines ([`greedy-compare.txt`](2026-09-24-r714-densegemm-r2/greedy-compare.txt)). That greedy set runs 1 stream, so the 16-row twins that carry the 4- and 8-stream gains have no served greedy test in this round.

**P1.** Arm − OFF, ms per iterate, stall-excluded (iterates slower than the run median plus 8 ms dropped from both runs of a pair, 4 to 10 of 192 per cell); every cell 6 of 6 rounds unless noted.

| shape | GM − OFF | ON − OFF | ON − OFF, per-iterate median |
| --- | --- | --- | --- |
| 1 stream, depth 3 (4 verify rows) | −0.31 (−2.6 %) | −0.62 (−5.2 %) | −0.59 |
| 4 streams, depth 3 (16 rows) | −0.41 (−2.0 %) | −0.44 (−2.2 %) | −0.49 |
| 8 streams, depth 1 (16 rows) | −0.44 (−2.4 %) | −0.39 (−2.1 %) | −0.36 |
| depth 0, 1 row | −0.14, 5 of 6 | −0.23 (−2.9 %) | |
| depth 0, 4 rows | −0.22 | −0.54 (−5.3 %) | |
| depth 0, 8 rows | −0.19 | −0.52 (−4.2 %) | |

The harness stall moved to iterates 7 to 9 (+34 ms) at 1 stream depth 3 on this image and hit OFF in 2 of 6 runs, GM in 4 and ON in 4, with no position pattern. The gate's own summary read raw means (GM +0.18 and ON −0.13 at 1 stream depth 3), so it called GM a gain only at 8 streams and ON only at 4 streams. On the stall-excluded read ON − GM is −0.04 ms at 4 streams and +0.01 at 8 streams: the 16-row verify never takes the gemv, so modes 1 and 2 run the same kernels there. At 4 rows ON − GM, the gemv increment, is −0.31 ms (5 of 6) and −0.32 / −0.33 ms in the depth-0 cells at 4 and 8 rows, which reproduces R710b's mode 3.

**Headroom.** Free VRAM at the launcher's UP line: 1,041 / 2,431 MiB (reference and off) against 1,035 / 2,423 MiB (GM and ON).

**ROWS32** (P0 only, `EXL3_DENSE_ROWS32=1` alone): −23 to −45 % per call against the served two-pass kernel at 18 to 32 rows; critical path 2.62 to 2.64 ms at 18, 21 and 24 rows against 3.87 to 3.91 ms served.

## Reading

- ON is accepted as a stack entry in place of R710b's mode 3: mode 1 contains the gemv twin and is as fast or faster at every shape (1 stream depth 3: −0.62 against −0.37 ms). GM is accepted and not carried, since mode 1 contains it.
- Served from stack-r3 on as `EXL3_DENSE_V2=1` with no defines ([R716b](r716b-stack-r3.md)). In mode 1 the QSA indexer's `index_qk` takes the gemm twin at 16 rows, so latchain's QSA fork reaches densegemm scratch at 4 and 8 streams; stack-r3's dense-lcguard patch gives that side branch its own scratch.
- ROWS32 is not served; it belongs to the verify-rows work beyond 16 rows.
