# R558: 8 slots fit 966,656 tokens; c8 674 / 639 t/s, +23 % / +16 % over 4 slots; same replay wall

Results directory on the serving host: `results/2026-09-19-r558-slots8`. Raw records: [`2026-09-19-r558-slots8/`](2026-09-19-r558-slots8/). Driver: [`scripts/r558-slots8.sh`](../../scripts/r558-slots8.sh). Date: 2026-09-19.

Configuration: the image and flags of [R548](r548-promote-gdnbf16-ring.md), NVMe tier off. Reference (4 slots, 1,032,192 tokens): free VRAM at boot 2,013 / 737 MiB, c1 greedy fingerprint `f4add302e176d78e`, minimum free VRAM during a cold 120k prefill plus 8 code streams 1,243 / 195 MiB. Ladder at 8 slots, 16,384-token steps down from 1,015,808: a pool passes when its c1 fingerprint equals the reference, its free VRAM at boot is at most 32 MiB below the reference on each card, and its minimum under the same load is at most 32 MiB below the reference minimum.

- 983,040: c1 equal, minimum under load 1,175 / 229 MiB, cuda:0 36 MiB under its floor of 1,211.
- **966,656: passes** (free at boot 2,085 / 867 MiB, minimum under load 1,299 / 353 MiB).

`fn_bench` at 966,656 × 8 slots, 2,048 forced tokens, 2 runs after a warm-up round, aggregate (per-stream decode) in t/s:

| streams | code | prose |
| --- | --- | --- |
| 1 | 200.9 (203.5) | 199.6 (202.0) |
| 4 | 566.3 (154.6) | 522.8 (135.3) |
| 6 | 544.2 (93.3) | 522.6 (89.9) |
| 8 | 674.2 (87.5) | 639.4 (83.0) |
| 8 requests on 4 slots (reference boot) | 549.4 (TTFT 7.3–8.0 s) | 549.4 (TTFT 7.0–7.4 s) |

The 6-stream row sits below the 4-stream row: the draft policy `[[4, 3], [8, 1]]` drafts 3 tokens up to 4 jobs and 1 token above, so 4 streams verify 16 rows at depth 3 and 6 streams 12 rows at depth 1 ([R560](r560-c8-policy.md)).

The 8-agent replay of [R557](r557-agent-replay-daily.md) at 8 slots: wall 408.6 s (4 slots: 413.2 / 395.1 s), 366 calls, 0 errors, latency p50 3.76 s (4.23 / 4.11), p90 11.04 s (10.87 / 10.95), mean 5.57 s (5.67 / 5.54); queue wait p50 / p90 0.12 / 0.27 s (1.39–1.52 / 3.28–3.43), draft acceptance 80.3 %. The run had 5.02 calls open on average and 8 open for 20 % of the wall time. With 5 to 7 jobs decoding at depth 1, the 8-slot engine produces no more tokens per second than the 4-slot engine running 4 jobs at depth 3 (6 streams 513–544 t/s, 4 streams 566), so the time saved in the queue returns as slower per-stream decode and the wall does not change.
