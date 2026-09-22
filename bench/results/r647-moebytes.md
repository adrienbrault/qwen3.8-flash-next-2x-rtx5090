# R647: MoE reads sit at the dedup floor; the residual host gap is real, ~2.1–2.4 ms/iterate

Results directory on the serving host: `results/2026-09-22-r647-gap-and-moebytes`. Raw records: [`2026-09-22-r647-gap-and-moebytes/`](2026-09-22-r647-gap-and-moebytes/). Driver: `r647-gap-and-moebytes.sh` (queued GPU-exclusive unit). Image `tabbyapi:bverify-r1` (the promoted daily). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

## Phase A — residual host gap on the promoted image

Same cells and harness as R644 (standalone profiler, forced length, warmed shapes, wall vs kernel-busy per iterate):

| cell | wall ms/iter | kernel ms/iter | gap | R644 gap (pre-`bverify`) |
|---|---:|---:|---:|---:|
| c8/26k d0 | 12.78 | 12.35 | 0.42 | 0.50 |
| c8/26k d1 | 19.55 | 17.11 | 2.44 | 2.03 |
| c4/26k d0 | 10.40 | 10.22 | 0.17 | 0.21 |
| c4/26k d3 | 20.20 | 18.10 | 2.11 | 1.70 |

The gap did not shrink on `bverify-r1`, and the trace shows why: `aten::argmax` = 1/iterate, i.e. the harness's `GreedySampler` has always run the batched verify path — the R645/646 veto only poisoned TabbyAPI's `CustomSampler`. The harness was already measuring the post-fix path; the ~2.1–2.4 ms/iterate residual is real for harness and server alike (R648 attributes it).

## Phase B — per-call MoE bytes vs slots/runs/distinct

The R630–R633 debate (whether expert-weight DRAM reads are slot-proportional or deduplicated) was unresolvable by its own data: `bytes_exp` is bimodal across K=2/K=3 layer bands and counts came from a different run than the ncu bytes. This run fixed both: `EXL3_MOE_DISTINCT_PROBE=1` at `INTERVAL=1` inside the ncu-profiled process, `--launch-skip 0 --launch-count 192`, pairing ncu launch rows `(2k, 2k+1)` with probe call `k` in program order. Guards all clean: 48 probe calls/step/subprocess, a/b alternation held on all 192 rows, the d1→d0 worker boundary landed at call 2450 (outside the profiled window).

`implied_reads = (a+b bytes)/bytes_exp` per call, 96 calls per shape:

| shape | band | slots | runs | distinct | implied | implied/distinct | slope | intercept |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| c4d0-26k | K2 | 40 | 24.0 | 24.0 | 24.3 | 1.013 | 0.992 | 0.5 |
| c4d0-26k | K3 | 40 | 13.5 | 13.5 | 13.9 | 1.031 | 1.006 | 0.3 |
| c8d1-26k | K2 | 160 | 59.0 | 57.5 | 58.8 | 1.023 | 0.992 | 1.8 |
| c8d1-26k | K3 | 160 | 32.5 | 30.0 | 31.4 | 1.046 | 0.999 | 1.4 |
| c8d1-26k | K4 (mtp layer) | 80 | 11.5 | 11.5 | 11.9 | 1.037 | 0.996 | 0.5 |

DRAM traffic is at the dedup floor on every band: implied ≈ runs ≈ distinct, regression slope ≈ 1.0, intercept ≈ 0. A grouped-decode MoE kernel would save ≈ (runs − distinct)/slots ≈ 0–2 % of MoE bytes against new on-device dedup machinery — permanently withdrawn, and this time the per-call matching removes R633's band-confound objection.

Two side facts: the MTP draft carries one MoE layer (`mtp.layers.0.mlp`, slots 80, same dedup behaviour), and coop a+b ≈ 6.2 ms/step at c8d1-26k for ~58–73 MB of reads per call ≈ 500–600 GB/s effective — the kernel is latency/issue-bound at ≤160 rows, not bandwidth-bound. "Faster coop at small rows" is a new experiment needing its own estimate, not the withdrawn dedup project.
