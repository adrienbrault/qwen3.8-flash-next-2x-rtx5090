# R557: 8-agent SWE-bench replay on the R548 configuration, 413 / 395 s

Results directory on the serving host: `results/2026-09-19-r557-agent-replay-daily`. Raw records: [`2026-09-19-r557-agent-replay-daily/`](2026-09-19-r557-agent-replay-daily/). Driver: [`scripts/r557-agent-replay-daily.sh`](../../scripts/r557-agent-replay-daily.sh). Replayer: [`bench/agent_replay.py`](../agent_replay.py). Date: 2026-09-19.

Configuration: the served configuration of [R548](r548-promote-gdnbf16-ring.md) (4 slots, 1,032,192-token pool), NVMe tier off. The replay of [R518](r518-slots6.md): recorded mini-SWE-agent trajectories, 8 agents, 16 conversations × 24 calls, 2 s tool gap, greedy, recorded history re-sent. Two runs on one boot.

| run | wall | calls | errors | latency p50 / p90 / mean |
| --- | --- | --- | --- | --- |
| D1 | 413.2 s | 366 | 0 | 4.23 / 10.87 / 5.67 s |
| D2 | 395.1 s | 366 | 0 | 4.11 / 10.95 / 5.54 s |

R518's 4-slot arms read 422.2 / 444.8 s with p50 4.5 s and p90 12.4 s on the 786,432-token configuration of 2026-09-19 00:13 UTC. Server log: 91.7 % of prompt tokens served from cache in both runs, queue wait p50 / p90 1.39–1.52 / 3.28–3.43 s, TTFT p50 / p90 1.81–1.87 / 3.80–3.97 s. Time-weighted, the runs had 5.05 and 5.15 calls open on average, and 5 to 7 calls open for 66 % and 70 % of the wall time.
