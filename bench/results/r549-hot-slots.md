# R549: TTFT and decode with 1 to 4 slots in use, up to 570k tokens resident; the pool does not stall

Results directory on the serving host: `results/2026-09-19-r549-hot-slots`. Raw records: [`2026-09-19-r549-hot-slots/`](2026-09-19-r549-hot-slots/). Driver: [`scripts/r549-hot-slots.sh`](../../scripts/r549-hot-slots.sh). Probe: [`bench/hot_slots.py`](../hot_slots.py). Date: 2026-09-19.

Configuration: the served daily of [R540](r540-promote-r6.md) (4 slots, 819,200-token pool at 8-bit KV, layer split `[30, 30]`, MTP draft depth 3), booted from the served launcher with the NVMe tier off so that the salted test prompts do not enter the daily's tier. Requests are streamed chat completions, greedy, with forced lengths (`min_tokens` = `max_tokens`). Long prompts are `fn_bench` prose filler with a distinct seed per request, so no two requests share a prefix; the server counts about three quarters of the requested filler size as prompt tokens, and every size below is the server's count. TTFT is the time from sending the request to its first streamed token. "Frames" are streamed chunks; with MTP one frame can carry several tokens.

## A: short prompts sent together

About 30 prompt tokens each, 1,024 forced tokens, 2 runs per concurrency.

| concurrency | TTFT (s) | decode per stream (t/s) | aggregate (t/s) |
| --- | --- | --- | --- |
| 1 | 0.23, 0.13 | 217.3, 227.4 | 207.3, 221.4 |
| 2 | 0.22–0.23 | 173.4–181.0 | 334.2, 339.0 |
| 3 | 0.31–0.34 | 142.1–147.6 | 413.9, 409.0 |
| 4 | 0.39–0.45 | 141.1–150.5 | 531.9, 541.2 |

The first c1 run of the unit has the higher TTFT (0.23 s); the second reads 0.13 s.

## B: prompts of about 22,500 tokens sent together

256 forced tokens. TTFT of each request, sorted:

| concurrency | prompt tokens | TTFT (s) |
| --- | --- | --- |
| 1 | 22,682 | 2.65 |
| 2 | 22,556, 22,531 | 5.89, 5.90 |
| 3 | 22,564, 22,682, 22,561 | 5.90, 8.35, 8.36 |
| 4 | 22,539, 22,581, 22,501, 22,477 | 12.00, 12.01, 12.14, 12.14 |

The four prompts at c4 hold 90,098 tokens, and the last first token arrives at 12.14 s: 7,420 prompt tokens per second shared across the queued requests.

## C: one request arriving while other slots decode long contexts

0 to 3 slots first each receive a prompt of about 112,000 tokens (filler target 150,000) and start decoding, one after another. Five seconds after the last one decodes, a short request (512 forced tokens) is sent, then a request of about 22,500 tokens (64 forced tokens).

| slots already decoding | new short request: TTFT / decode | new ~22.5k request: TTFT | running streams: frames/s before → during the ~22.5k prefill | running streams: longest gap before → during |
| --- | --- | --- | --- | --- |
| 0 | 0.16 s / 234.4 t/s | 2.47 s | | |
| 1 | 0.20 s / 161.1 t/s | 3.08 s | 73.8 → 4.22 | 0.022 → 0.532 s |
| 2 | 0.24 s / 133.5 t/s | 3.25 s | 54.8 → 4.31 | 0.046 → 0.536 s |
| 3 | 0.26 s / 128.1 t/s | 3.24 s | 46.0 → 4.32 | 0.050 → 0.546 s |

While the new request prefills, the running streams receive a frame about every 0.54 s; they return to their earlier rate when its prefill ends.

## D: four prompts of about 142,600 tokens sent together

Prompt tokens 142,654, 142,616, 142,692 and 142,579 (570,541 in the 819,200-token pool), 1,024 forced tokens each.

- TTFT, sorted: 24.00, 70.01, 70.05, 70.09 s; 570,541 prompt tokens to the last first token in 70.09 s is 8,140 tokens per second.
- In the 7.93 s window where all four decode: 116.8, 104.3, 114.0 and 108.3 t/s per stream, longest gap between frames 0.051 s on every stream.
- All four requests reached their forced length (`finish_reason: length`), no request failed, and the server answered after the run with no container restart.

## Reading

With four slots in use and 570k tokens resident, decode runs at 104–117 t/s per stream without pauses. A new short request's TTFT goes from 0.16 s with no other load to 0.26 s with three slots decoding long contexts. Waiting time comes from prefill: long prompts that arrive together share about 7.4k–8.1k prompt tokens per second, and each running stream gets about 2 decode steps per second while a new long prompt prefills.
