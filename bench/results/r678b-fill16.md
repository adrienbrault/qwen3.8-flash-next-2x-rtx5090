# R678b: mixed draft depth per job to fill 16 verify rows (fill16) is 16-21 % slower at 5 and 7 streams, rejected

Results directory on the serving host: `results/2026-09-24-r678b-fill16-gate`. Raw records: [`2026-09-24-r678b-fill16-gate/`](2026-09-24-r678b-fill16-gate/). Driver: `r678-fill16-gate.sh` (queued GPU-exclusive unit). Image `tabbyapi:fill16-r1` = `tabbyapi:slotfix-r1` plus the fill16 patch, behind `EXL3_DRAFT_FILL16=1`. Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, 8 slots, 999,424-token page pool, draft policy `[[4, 3], [5, 2], [8, 1]]`.

## The change

The served draft policy gives every job the same depth: 2 draft tokens at 5 concurrent jobs and 1 above, because a uniform depth 2 at 6 jobs would need 18 verify rows and the cooperative MoE decode kernels take at most 16 ([R560](r560-c8-policy.md), [R562](r562-profile-c8.md)). At 5, 6 and 7 jobs the uniform policy leaves 1, 4 and 2 of the 16 rows unused. fill16 gives the spare rows to individual jobs, up to 2 above the policy depth, ranked by a per-job moving average of draft acceptance, so a verify batch at 5 to 7 jobs carries 16 rows with mixed depths. The verify window is the largest depth in the batch.

## What was measured

2026-09-23 22:26 to 22:30 UTC, one boot per arm.

| check | result |
| --- | --- |
| CPU tests of the depth assignment and the row index | 25 of 25 ([`cpu-tests.txt`](2026-09-24-r678b-fill16-gate/cpu-tests.txt)) |
| greedy output, 6 prompts including one of about 100,000 tokens, flag off and flag on | identical to the reference, 6 of 6 each |
| verify shapes seen with the flag on | 29 distinct, all at 16 rows, mixed depths at 5 to 7 jobs, e.g. `[3, 2, 2, 2, 2]` at 5 and `[2, 2, 2, 1, 2, 1]` at 6 ([`verify-shapes.txt`](2026-09-24-r678b-fill16-gate/verify-shapes.txt)) |
| free VRAM on GPU 0 after a 1-to-8-stream ramp, off / on | 121 / 117 MiB |

Decode at 5, 6 and 7 streams: code, 256 forced tokens, greedy, one warm-up round and 2 recorded rounds per cell, median per-request decode rate in tokens/s per stream ([`midbench-off.txt`](2026-09-24-r678b-fill16-gate/midbench-off.txt), [`midbench-on.txt`](2026-09-24-r678b-fill16-gate/midbench-on.txt), records in [`midbench.jsonl`](2026-09-24-r678b-fill16-gate/midbench.jsonl)):

| streams | off, run 1 / run 2 | on, run 1 / run 2 | on/off |
| ---: | --- | --- | ---: |
| 5 | 132.2 / 133.2 | 102.8 / 103.1 | 0.79 |
| 6 | 96.1 / 110.6 | 92.8 / 94.0 | not cited |
| 7 | 95.4 / 97.4 | 82.0 / 81.4 | 0.84 |

The two off runs agree within 1 % at 5 streams and 2 % at 7, so the 16-21 % losses there are outside the run-to-run spread. At 6 streams the off runs differ by 14 %, the on/off ratio sits inside that spread, and it is not cited.

Per streamed frame (medians over the requests in `midbench.jsonl`), the flag raises tokens per frame at 6 and 7 streams (1.73 → 2.19 and 1.75 → 1.91, none at 5: 2.33 both) and raises the time per frame at every count: 17.5 → 22.8 ms at 5 streams, 16.6 → 22.7 at 6, 18.2 → 23.8 at 7. The window is the largest depth in the batch, so one job at a higher depth adds a sequential draft level for the whole batch: a third level at 5 streams and a second at 6 and 7. That added level costs more time per step than the extra accepted tokens return.

## Verdict

Rejected: the rule written before the run required at least 1.03× over 5 to 7 streams, and the total was 0.844×. The served configuration did not change.

The first attempt, R678 on 2026-09-23, did not produce a reading. Its first build failed at container start because the overlay's Dockerfile switched to a `tabby` user that the base image does not have. The rebuilt image passed the flag-off checks, but the patch logged through a logger with no handler, so its initialisation line and the verify-shape lines never reached the container log and the flag-on arm failed its activity check. The patch now prints those lines like the other `EXL3_*` flags, and R678b is the rerun.
