# SWE-bench agent runs: 39 calls and 36k tokens of final context per instance, and where the wall time went

 Driver: [`scripts/r359-swebench.sh`](../../scripts/r359-swebench.sh), [`scripts/r369-swebench-30.sh`](../../scripts/r369-swebench-30.sh). Date: 2026-09-19.

Analysis of the trajectories from the four 2026-09-16 SWE-bench Verified subsets ([R359](r359-swebench.md), results `2026-09-16-r359-swebench-10`, `-r360-swebench-strat`, `-r361-swebench-failed`, `-r369-swebench-30`): 49 unique instances, the later run kept where an instance ran twice. Token counts are re-tokenized with the checkpoint's `tokenizer.json`, because the served image of that day reported no usage in the trajectories; they count every message the agent re-sends, including the reasoning of earlier turns, which the chat template keeps inside a tool-call loop. Serving configuration of that day: 3.05bpw pack, 8 slots, 262,144-token pool, draft policy `[[2, 3], [8, 1]]`, mini-SWE-agent 2.4.6 with 8 workers.

| per instance | median | mean | p90 | max |
| --- | --- | --- | --- | --- |
| agent calls | 39 | 42.6 | — | — |
| final context, tokens | 36,011 | 40,108 | 63,096 | 146,759 |
| new prompt tokens per call (after the first) | 281 | 509 | 1,270 | 8,103 |
| first prompt, tokens | 1,598 | 1,660 | 2,131 | 3,424 |
| prompt tokens re-sent over all calls | 921,765 | 1.26M | — | — |
| generated tokens | 14,545 | 17,684 | — | — |
| of which reasoning | 9,984 | 11,799 | — | — |
| latency per call | 8.9 s | 9.8 s | — | — |
| wall time | 766 s | 864 s | — | — |

The R369 subset, 30 instances in 66.6 minutes (2.2 minutes per instance at 8 workers): 1,246 calls, 487,451 generated tokens (122 t/s over the wall), 34.5M prompt tokens re-sent (8,643 t/s over the wall), 0.63M of them new. Of the 8.9 worker-hours, 5.25 h (59 %) were spent waiting on the model, 1.25 h (14 %) running commands in the task containers and 2.4 h (27 %) in container setup and the run's tail. About 4.7 calls were in flight on average, so each call generated at about 26 t/s against 100 t/s per stream on the c4 decode benchmark of the same stack.

The gap is not explained by decode alone. With exact prefix reuse, each call would process its new tokens plus at most 2,048 tokens of GDN-state replay, about 6 minutes of prefill for the whole run. Two mechanisms can lose that reuse: 8 agents at a mean final context of 40k (p90 63k) exceed a 262,144-token pool, so attention pages are evicted; and a cached prefix is resumable only up to the newest stored GDN checkpoint, taken every 2,048 tokens into a 4 GiB host LRU that holds about 35 checkpoints of about 115 MB (36 GDN layers × 48 heads × 128 × 128 × fp32), and none at the end of a request. The served image of 2026-09-19 has a 786,432-token pool, which removes the first mechanism for 8 agents at p90 context. The agent-replay probe in [R518](../../scripts/r518-slots6.sh) measures cached against new prompt tokens per call on the current configuration.
