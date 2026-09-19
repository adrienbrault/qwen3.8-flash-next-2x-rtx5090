# R561: the daily serves 8 slots with a 966,656-token page pool

Results directory on the serving host: `results/2026-09-19-r561-promote-slots8`. Raw records: [`2026-09-19-r561-promote-slots8/`](2026-09-19-r561-promote-slots8/). Driver: [`scripts/r561-promote-slots8.sh`](../../scripts/r561-promote-slots8.sh). Date: 2026-09-19, 17:05–17:31 UTC.

## What changes

Two launcher lines: `MAXBS` 4 → 8 and `cache_size` 1,032,192 → 966,656 (−6.3 %), the top of the 8-slot ladder in [R558](r558-slots8.md). Image, `EXTRA_ENV`, draft policy `[[4, 3], [8, 1]]`, 8-bit KV, layer split `[30, 30]` and vision are unchanged from [R548](r548-promote-gdnbf16-ring.md). The unit promotes after the boot checks and runs the quality gates on the live daily; a failed gate restores the R548 launcher.

## Boot checks (NVMe tier off except G1)

| boot | free VRAM at boot, MiB | fingerprints c1 / 30k | cold prefill 60k / 120k, t/s |
| --- | --- | --- | --- |
| S0 served, 4 slots @ 1,032,192 | 2,013 / 737 | `f4add302e176d78e` / `4a255910dee2d9c5` | 9,285 / 10,221 |
| S1 candidate, 8 slots @ 966,656 | 2,085 / 867 (floors 1,981 / 705) | same | 9,480 / 10,239 (1.021 / 1.002×) |

G1 booted the candidate with the NVMe tier on. S1 also ran 8 streams of 2,048 forced tokens: 631.1 t/s aggregate, 85.3 t/s per stream decode.

## Gates (NVMe tier on, live daily)

| gate | result |
| --- | --- |
| GT restart restore | a 30,001-token prompt after a container restart: 29,952 tokens served from the tier in 0.61 s, output identical to the warm run (0.37 s) and the cold run (4.02 s) |
| G2 agentic-edit | 6/6 in four modes; greedy c4 first full wave 518.9 t/s aggregate, 158.0 per stream |
| G3 needles | 5/5 at ~131k and 5/5 at ~240k prompt tokens |
| G4 tool-eval 69×4 | 88.2 ± 1.0 (95 % interval 87.5 to 89.0) |
| G5 GSM8K 5-shot n=500, no stop strings | 0.974 |

Promoted 2026-09-19 17:07 UTC; gates finished 17:31 UTC. Rollback: the R548 launcher (4 slots at 1,032,192).

## Throughput at 8 slots

From [R558](r558-slots8.md) and [R560](r560-c8-policy.md) on the same configuration, aggregate t/s, code / prose: 1 stream 200.9 / 199.6, 4 streams 566 / 523, 6 streams 513–544 / 513–523, 8 streams 640–674 / 628–639. The 8-agent SWE-bench replay finishes in the same wall time as at 4 slots, with queue wait p50 0.12 s instead of 1.4 s.
