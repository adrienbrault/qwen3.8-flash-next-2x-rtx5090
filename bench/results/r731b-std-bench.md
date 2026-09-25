# R731b: `vllm bench serve` on ShareGPT V3 and Spec-Bench at stock power: 471 / 525 output tok/s wall clock at 8 streams and 221 / 244 at 1 stream; R731's 400 W limit moved no cell by more than 0.9 %

Results directory on the serving host: `results/2026-09-25-r731b-std-bench-stock`. Raw records: [`2026-09-25-r731b-std-bench-stock/`](2026-09-25-r731b-std-bench-stock/) (`results/<cell>.json`, vLLM's result file with `--save-detailed` per-request arrays, without `generated_texts` and with the time arrays rounded to 1 µs; `results/<cell>.samples.tsv`, the sample manifest in send order; `cells/container-<cell>.log`, the server log of each cell's boot; `cells/client-<cell>.log`; `cells/env-<cell>.txt`, the container's `EXL3_` environment; `boots.tsv`, `runs.tsv`, `foreign.tsv`, `summary.txt`, `summary.json`, `audit.txt`; the boot logs stay on the host). Driver [`scripts/r731-std-bench.sh`](../../scripts/r731-std-bench.sh): the R731 unit with the power-limit gate added after R731's review (a boot whose limits differ from the cards' defaults ends the unit VOID), run at stock power; there is no separate R731b script. Client: the `vllm/vllm-openai:v0.30.0` image running `vllm bench serve` through [`bench/vllm_bench_tabby.py`](../vllm_bench_tabby.py); summary [`bench/std_bench_summary.py`](../std_bench_summary.py) with [`bench/parse_container.py`](../parse_container.py). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, image `tabbyapi:stack-r3-rows32` (`sha256:abe93e1bc178`), the served launcher of [R728](r728-promote-window-off.md) (41 environment keys, 43 `EXL3_` variables in the container, draft-KV window off), 8 slots, 983,040-token page pool at 8-bit KV, draft policy `[[4, 3], [8, 2]]`, NVMe tier off. Power limits 600 / 575 W (the cards' defaults), core clock offset 0 and memory offset +4500 on both cards, read back at every boot.

## Why

`fn_bench` ([`bench/probe.py`](../probe.py)) measures decode rates on prompts of about 110 tokens, all streams starting together. This round measures the served configuration with vLLM's serving benchmark on two public datasets, a harness that third-party serving results also use. R725 and R725b (2026-09-25 02:28 to 03:28 UTC) ran it first and were not published: the ShareGPT run did not log the memory offset, each concurrency drew a different sample, Spec-Bench had 73 to 86 requests at 2 to 8 streams, each dataset ran on one boot, and R728 changed the served configuration afterwards. R731 (2026-09-25, results `2026-09-25-r731-std-bench`) fixed the protocol and ran with both cards limited to 400 W by a limit left over from another engine's launcher. R731b re-runs R731 on the same samples at the cards' default limits, and its tables replace R731's.

## What was measured

2026-09-25 21:25:40 to 22:54:39 UTC. The matrix is ShareGPT V3 and Spec-Bench at 1, 2, 4 and 8 concurrent requests, run twice (pass A, then pass B, the same order), 16 cells. Every cell boots the served launcher afresh, so the prompt cache is empty, warms the kernels with `c` non-streamed 64-token requests on prompts that are in neither dataset, then runs the client closed loop: `--request-rate inf --max-concurrency c`, `--num-warmups 0`, `/v1/chat/completions` with streaming and `include_usage`, temperature 0, thinking on (the template default).

- ShareGPT: `ShareGPT_V3_unfiltered_cleaned_split.json` (sha256 `35f0e213…`), 400 conversations drawn by vLLM's sampler with `--seed 7310`; the prompt is the first user turn and the forced output length is the reference reply's length (mean 210 tokens, maximum 1,439).
- Spec-Bench: `question.jsonl` (sha256 `4b6d33e7…`), all 480 questions of its 13 categories, first turn only, 256 forced tokens each.
- The sample manifest has one md5 across the 8 cells of each dataset and is the same as R731's, so every concurrency level and both runs send the same prompts in the same order.
- Prompts are chat-templated once, by the server: mean 272 / 322 tokens and maximum 1,070 / 1,540 (ShareGPT / Spec-Bench); the client's and the server's sums agree (108,902 and 154,537 tokens).

`bench/vllm_bench_tabby.py` replaces three parts of vLLM v0.30.0 that do not work against TabbyAPI: the chat request function reads `usage` from any frame (TabbyAPI sends it in a frame that also carries a `choices` entry, which stock vLLM skips and then re-tokenises only the text after `</think>`) and splits CRLF-separated events; `min_tokens` is set to the requested output length, because `ignore_eos` is dropped by the exllamav3 backend; and Spec-Bench prompts are not templated on the client before the server templates them again. It also records each cell's sample manifest. The throughput and latency metrics are computed by vLLM's own code; τ comes from the server log.

## Results

Cells are the mean of passes A and B; the p99 columns give pass A and pass B as a range where they differ. Output tok/s and Req/s are wall-clock figures (definitions below).

**ShareGPT V3** (400 prompts, seed 7310, mean 272 input and 210 output tokens)

| streams | output tok/s | A/B spread | req/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | per-stream tok/s (1000 / TPOT p50) | E2E p50 (s) | τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 221.3 | 0.11 % | 1.05 | 139 / 338–346 | 3.42 / 4.64–4.71 | 292 | 0.64 | 2.64 |
| 2 | 304.7 | 0.18 % | 1.45 | 197 / 479–500 | 4.82 / 8.83–8.86 | 208 | 0.90 | 2.63 |
| 4 | 398.5 | 0.06 % | 1.90 | 284 / 684–714 | 7.53 / 20.4–22.8 | 133 | 1.35 | 2.65 |
| 8 | 471.2 | 0.01 % | 2.25 | 389 / 1,333–1,368 | 14.03 / 23.4–25.4 | 71 | 2.24 | 2.30 |

**Spec-Bench** (480 prompts, 13 categories, mean 322 input and 256 output tokens)

| streams | output tok/s | A/B spread | req/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | per-stream tok/s (1000 / TPOT p50) | E2E p50 (s) | τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 244.2 | 0.16 % | 0.95 | 133 / 412–417 | 3.37 / 4.30 | 296 | 1.04 | 2.87 |
| 2 | 341.9 | 0.09 % | 1.34 | 186 / 594–622 | 4.80 / 6.54–6.57 | 208 | 1.48 | 2.87 |
| 4 | 446.3 | 0.12 % | 1.74 | 266 / 676–719 | 7.58 / 10.55–10.59 | 132 | 2.27 | 2.87 |
| 8 | 525.4 | 0.10 % | 2.05 | 380 / 983–1,017 | 13.45 / 16.96–17.05 | 74 | 3.86 | 2.43 |

The per-pass tables are in [`summary.txt`](2026-09-25-r731b-std-bench-stock/summary.txt). τ falls at 8 streams because the draft policy drafts two tokens above 4 streams instead of three. At 1 stream Spec-Bench's τ ranges by category from 2.48 (humanities) to 3.30 (math reasoning) and 3.27 (math); the per-category table is in `summary.txt`.

## Definitions

| column | definition |
| --- | --- |
| output tok/s | Σ completion tokens / (last completion − first request start), closed loop at `c` concurrent requests; it includes prefill, time to the first token, turnover between requests and client overhead, and it is not a decode rate |
| req/s | completed requests over the same wall clock; it depends on the output length |
| TTFT | client-side time from sending the request to the first streamed `reasoning_content` or `content` delta; with thinking on, the first reasoning token |
| TPOT | (E2E − TTFT) / (output tokens − 1) per request, median and 99th percentile across requests; it includes the time a request waits while other requests' prefills run |
| per-stream tok/s | 1000 / TPOT p50, derived from TPOT; it includes those waits and is not the `fn_bench` per-stream decode rate |
| E2E | request latency from sending to the last token |
| τ | tokens per verify step from the server log, Σ generated / (Σ generated − Σ accepted drafts); it counts the prefill step, which puts it about 1 % low. Spec-Bench τ is not comparable to acceptance figures published with Spec-Bench |
| A/B spread | \|A − B\| / mean(A, B) of output tok/s; replication within one session, not the uncertainty of the cell |

vLLM's `max_concurrent_requests` and `max_output_tokens_per_s` fields in the result files are not used: the first counts the requests that touch a one-second bucket (3 to 15 here, while the number of requests running at once, from the per-request start times and latencies, peaks at exactly `c` in every cell), and the second counts streamed frames per second, not tokens. The MTP depth the summary prints (3.00 up to 4 streams, 2.00 to 2.01 at 8) is the draft policy.

## Integrity

- Every cell completed all its requests (400 or 480) with 0 failures and no empty output. The server's per-request lines equal the requests, and the client's Σ output tokens equal the server's Σ generated tokens in all 16 cells. The container logs have 0 out-of-memory, `TORCH_CHECK` and traceback lines.
- 0 cached prompt tokens across all 7,040 request lines: every line reads `none cached`.
- All 16 boots read power limits 600 / 575 W, core offsets 0 / 0 and memory offsets 4500 / 4500 at boot, and the offsets again after each cell (`boots.tsv`). The power limit was read once per boot; there is no draw or clock telemetry for this round.
- All 16 `cells/env-*.txt` are identical, and identical to R731's; the launcher differs from R731's only in the block that sets and reads back the power limit and core offset.
- Foreign traffic: none. The driver's counter (`foreign.tsv`, 0 in every cell) matches only streamed chat requests, so the 0 rests on a count of every request header in each container log: in every cell, `n` streamed chat requests, all carrying `min_tokens`; `c` non-streamed warm-ups with `min_tokens`; and the launcher's one 16-token probe before the run. Nothing else. The counter's pattern needs widening, and the launcher probe excluded, before the driver is reused.
- ShareGPT outputs equal the reference length except for one request per cell that exllamav3's loop detector ended early (none in pass A at 2 streams), at most 305 of 84,120 tokens (0.36 %). Every Spec-Bench output is 256 tokens.
- The longest output is 1,439 tokens, below the 2,048-token requeue boundary, so the server's token counts are exact.
- `bench/std_bench_summary.py`, run on the published copy of the raw records with the driver's arguments (`--sb-out 256 --want-gpc "0 0" --want-mem "4500 4500" --cache-max 0.01 --spread-max 3 --spread-exempt 1 --decide "A B"`), prints `summary.txt` line for line apart from the per-category blocks, which need `--sb-data` and the Spec-Bench file. The rounding of the time arrays changes no printed digit.
- Decision rule, fixed before the run: PUBLISHABLE when both passes are complete, no integrity check fails, every cell has ≤ 1 % cached prompt tokens and the A/B spread of output tok/s is ≤ 3 % at 2 to 8 streams (1 stream reported, not gated). The largest gated spread is 0.18 % (ShareGPT, 2 streams); 1 stream reads 0.11 % and 0.16 %. Decision: PUBLISHABLE.

## Against R731 (400 W, same samples)

Mean of two passes per run; Δ is R731b / R731 − 1.

| cell | output tok/s, stock | output tok/s, 400 W | Δ | A/B spread, stock / 400 W | median ITL Δ | mean ITL Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShareGPT, 1 | 221.3 | 221.2 | +0.01 % | 0.11 / 0.36 % | −0.4 % | −0.3 % |
| ShareGPT, 2 | 304.7 | 302.1 | +0.87 % | 0.18 / 0.63 % | −1.7 % | −1.1 % |
| ShareGPT, 4 | 398.5 | 400.3 | −0.46 % | 0.06 / 0.67 % | −2.1 % | −0.1 % |
| ShareGPT, 8 | 471.2 | 471.4 | −0.03 % | 0.01 / 1.78 % | −2.7 % | −0.4 % |
| Spec-Bench, 1 | 244.2 | 243.9 | +0.11 % | 0.16 / 0.31 % | −0.4 % | −0.3 % |
| Spec-Bench, 2 | 341.9 | 341.3 | +0.16 % | 0.09 / 2.77 % | −1.4 % | −0.6 % |
| Spec-Bench, 4 | 446.3 | 449.5 | −0.72 % | 0.12 / 0.08 % | −2.3 % | +0.3 % |
| Spec-Bench, 8 | 525.4 | 528.7 | −0.62 % | 0.10 / 0.25 % | −2.8 % | +0.2 % |

ITL is vLLM's inter-token latency, one value per streamed frame, and a frame is one verify step. Output tok/s differs by −0.72 to +0.87 % per cell. The median step is 1.4 to 2.8 % shorter at stock at 2 to 8 streams, and the frames no longer than twice the median are 1.2 to 2.5 % shorter; the mean frame time is unchanged (−1.1 to +0.3 %), because time to the first token and the time in frames longer than twice the median are 3 to 4 % higher at stock at 4 and 8 streams (part of the second is the threshold: a shorter median lowers the cut). This round does not separate a prefill-side effect from session variance, and with prompts of at most 1,540 tokens it says nothing about the limit's cost on deep prefill. In R731 the cards drew at most 280 W.

Spec-Bench at 4 and 8 streams reads 0.6 to 0.7 % lower here than in R731 in both passes, while each run's A/B spread is at most 0.25 % there: the difference between two sessions is larger than the replication within one. Each cell is read as ±1 %. ShareGPT's TPOT p99 at 4 streams moves between sessions: 20.4 and 22.8 ms here, 19.4 and 13.6 ms in R731.

## The gap to `fn_bench` at 8 streams (R731)

Measured on R731's records (2026-09-25, results `2026-09-25-r731-std-bench`, 400 W), A/B means. Decoding-only is Σ(output tokens − 1) / Σ(E2E − TTFT) × c. The prefill-free rate drops the frames longer than twice the cell's median ITL, which are the steps in which other requests' prefill chunks run, and applies the same formula.

| | wall clock | share of slot time decoding | decoding-only | prefill-free | time in frames > 2× median |
| --- | --- | --- | --- | --- | --- |
| ShareGPT, 1 / 2 / 4 / 8 streams | 221 / 302 / 400 / 471 | 81 / 82 / 84 / 87 % | 272 / 369 / 473 / 541 | 291 / 441 / 627 / 786 | 0 / 13 / 25 / 35 % |
| Spec-Bench, 1 / 2 / 4 / 8 streams | 244 / 341 / 450 / 529 | 82 / 83 / 86 / 88 % | 295 / 411 / 523 / 596 | 296 / 458 / 650 / 803 | 1 / 12 / 23 / 31 % |

- At 8 streams the prefill-free rate is 98.2 / 100.4 tok/s per stream (ShareGPT / Spec-Bench), against `fn_bench`'s 105.8 / 106.9 per stream and 845 / 853 decode aggregate (code / prose, [R719b](r719b-decode-curve.md)). At 4 streams it equals `fn_bench`: 627 / 650 against 632 / 643.
- 12 to 19 % of slot time is time to the first token: outputs of 210 and 256 tokens turn the slots over continuously.
- τ does not account for the rest at 8 streams (2.30 / 2.44 against 2.26 to 2.32 in `fn_bench`). The remaining 6 to 7 % is a longer decode step, 24.8 ms at 400 W against about 21.7 ms in `fn_bench`. R731b's median step at 8 streams is 24.0 to 24.1 ms, and its frames no longer than twice the median are 1.2 to 2.5 % shorter than R731's, so the prefill-free rung reads up to that much higher at stock; the wall-clock and decoding-only rungs do not change (output tok/s and mean frame time as in the table above).
- Prefill chunks interleaved with decode take 31 to 35 % of decode time at 8 streams on this load, the largest share of the gap between the wall-clock figure and the decode aggregate.
