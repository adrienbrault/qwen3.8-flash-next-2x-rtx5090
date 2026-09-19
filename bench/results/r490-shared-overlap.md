# R490, R491, R491b: the shared expert on a side CUDA stream is byte-identical and +3 to +7 % decode; promoted

Results directory on the serving host: `results/2026-09-18-r490-shared-overlap`. Raw records: [`2026-09-18-r490-shared-overlap/`](2026-09-18-r490-shared-overlap/). Driver: [`scripts/r490-shared-overlap.sh`](../../scripts/r490-shared-overlap.sh), [`scripts/r491-shared-promote.sh`](../../scripts/r491-shared-promote.sh), [`scripts/r491b-tooleval-paired.sh`](../../scripts/r491b-tooleval-paired.sh). Date: 2026-09-18.

Each of the 48 MoE layers has one shared expert (5-bit, 147 MB per decode step in total) that ran in 144 launches of about 7 µs on the main stream. The patch (decode-kernels round 2, [`docker/overlays/decode-kernels-r2/`](../../docker/overlays/decode-kernels-r2/)) runs the unchanged shared-expert graph on a non-blocking side stream and joins it before routed stage B, behind `EXL3_SHARED_EXPERT_OVERLAP=1`. Image `tabbyapi:decode-kernels-r2`. `torch.equal` against the default path on real weights, one layer per card, rows 1 to 16: pass. One layer, OFF → ON: 71.3 → 61.9 µs at 1 row, 170.1 → 142.8 at 4, 263.6–310.4 → 241.0–279.9 at 16.

Served, 3.05bpw pack, 360,448, 4 slots:

| arm | fingerprints | code c1 / c4 (t/s, 2 runs) | prose c1 / c4 | multiprompt code c1 / c4 | multiprompt prose c1 / c4 | prefill 22.6k / 90.1k |
| --- | --- | --- | --- | --- | --- | --- |
| OFF | canonical | 212.6, 217.0 / 428.4, 440.6 | 166.1, 172.4 / 420.2, 433.8 | 162.6 / 375.0 | 163.0 / 347.7 | 7,130 / 10,991 |
| ON | canonical | 218.4, 220.8 / 448.2, 460.9 | 179.9, 180.1 / 446.7, 453.5 | 171.3 / 383.0 | 165.4 / 355.4 | 7,751 / 11,077 |
| OFF2 | canonical | 215.1, 216.7 / 432.2, 444.2 | 167.4, 171.9 / 420.6, 437.2 | 166.8 / 366.4 | 159.5 / 350.7 | 7,662 / 11,077 |
| ON2 | canonical | 220.1, 218.6 / 459.8, 460.4 | 179.7, 178.9 / 450.3, 453.4 | 178.6 / 388.5 | 164.9 / 364.3 | 7,804 / 11,144 |

Paired by seed on the multi-prompt probe (n = 24 per pair): code c1 +6.3 % (median +7.1 %), code c4 +5.5 %, prose c1 +2.7 %, prose c4 +6.9 %.

R491 (results `2026-09-18-r491-shared-promote`) passed fingerprints, needles and c8, and read tool-eval 69×4 at 79.5 ± 6.4 with one trial at 96 points. A byte-identical kernel change moves tool-eval only through sampling and batch noise, so R491b (results `2026-09-18-r491b-tooleval-paired`) ran four tool-eval passes in one session on fresh boots: CTL 83.2 ± 1.7, CAND 84.8 ± 2.6, CTL2 85.0 ± 0.8, CAND2 82.2 ± 2.5; means 84.1 against 83.5, inside the promotion rule (candidate mean ≥ control mean − 2, no run below 75). Promoted 2026-09-18 18:08 CEST.
