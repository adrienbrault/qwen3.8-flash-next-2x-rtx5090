# R518: 6 decode slots are not served; c6 gains 17 % on a synthetic load but an 8-agent replay runs 9 % slower

Results directory on the serving host: `results/2026-09-19-r518-slots6`. Raw records: [`2026-09-19-r518-slots6/`](2026-09-19-r518-slots6/). Driver: [`scripts/r518-slots6.sh`](../../scripts/r518-slots6.sh). Date: 2026-09-19.

The served configuration has 4 slots over a 786,432-token pool. R518 measured 6 slots on the same image (`tabbyapi:stack-r4-e3r2`), environment and draft policy. Arms S4 (served) / S6 / S4b / S6b, in that order.

Pool ladder at 6 slots: 786,432 and 753,664 do not boot ("Insufficient VRAM in split for model and cache"). **720,896 boots** (2,091 / 1,295 MiB free after boot) and survives a pass of 6 concurrent 2,048-token requests. Greedy fingerprints are the served ones on all four arms (c1 `ae890c45d1000582`, 30k `4a255910dee2d9c5`).

## Decode

[`bench/probe.py`](../probe.py), 2,048 forced tokens, greedy, two runs per shape. Aggregate is all streams' tokens over the round's wall time; per stream is one request's tokens over its own wall time, averaged.

| load | 4 slots, aggregate (per stream) | 6 slots, aggregate (per stream) | change (aggregate) |
| --- | --- | --- | --- |
| code c1 | 219.1 | 218.3 | −0.4 % |
| prose c1 | 192.2 | 192.1 | 0 % |
| code c4 | 510.4 (132.2) | 505.1 (129.8) | −1.0 % |
| prose c4 | 517.3 (131.6) | 471.0 (122.2) | −9 %; the 4-slot pair is 467.3 and 567.3, the second an outlier |
| code c6 | 448.8 (112.9) | 526.9 (88.5) | +17 % |
| prose c6 | 423.1 (104.8) | 498.2 (83.3) | +18 % |

At 4 slots, 2 of 6 requests wait for a slot, so the per-stream figure at c6 includes the queue. At 6 slots every stream decodes at about 86 t/s instead of about 135.

## Agent replay

[`bench/agent_replay.py`](../agent_replay.py): 16 recorded mini-SWE-agent conversations from the 2026-09-16 SWE-bench runs, the first 24 calls of each, 8 agents in parallel, 2 s between calls in place of tool execution, greedy. Each call re-sends the recorded history. 366 calls per arm, 0 errors.

| arm | wall | calls/min | latency p50 / p90 / mean |
| --- | --- | --- | --- |
| S4 | 422.2 s | 52.0 | 4.47 / 12.16 / 6.21 s |
| S4b | 444.8 s | 49.4 | 4.60 / 12.56 / 6.46 s |
| S6 | 467.8 s | 46.9 | 4.53 / 15.20 / 6.93 s |
| S6b | 477.3 s | 46.0 | 4.33 / 15.68 / 7.19 s |

Six slots take 9.0 % longer on the mean of the two pairs (472.6 against 433.5 s). From the TabbyAPI log ([`bench/tabby_log_stats.py`](../tabby_log_stats.py)), at 4 slots: 4.62M prompt tokens over 293 captured requests, 90.2 % served from cache, 454k new, 130 s of prefill, queue wait p50 / p90 1.35 / 3.92 s, draft acceptance 68.8 %. Six slots cut the queue wait to 0.57–0.68 / 2.9–3.5 s, but each call runs longer (mean 6.8–7.2 s against 6.2 s). For 8 agents that each wait on their previous call, the slower per-stream decode costs more than the shorter queue saves.

## Verdict

Not served. The daily keeps 4 slots at 786,432 tokens.

## Harness notes

- TabbyAPI returned no `usage` object on these non-streamed calls, so the replay's token counts are 0 in `replay.jsonl`. Throughput here is taken from wall time on identical call sets and from the TabbyAPI log.
- The driver's inline log regex matched 24 of 293 records because TabbyAPI wraps its metric lines; `tabby_log_stats.py` joins them. The `docker logs --since` capture of the two 6-slot arms kept only about 150 records each.
- The driver's decision block failed on the per-request records file and fell through to "keep 4 slots"; the tables above are recomputed per round from `records-S4.jsonl` and `records-S6.jsonl`.
