# Decode depends on content: 238.9 to 341.3 t/s per stream at one stream with thinking off, 2026-11-16

Results directory `20261116T102550Z-decodekinds-r1` on the serving host, records in [`20261116T102550Z-decodekinds-r1/`](20261116T102550Z-decodekinds-r1/), driver and instruments in [`scripts/decodekinds-r1/`](../../scripts/decodekinds-r1/). Measured 2026-11-16 10:25–11:06 UTC on the served image `tabbyapi:tune268-r2-pf2lock-r3` (`sha256:da386167…`), unchanged launcher and configuration, two fresh boots (passes A and B) on a test port, stock 600 / 575 W power limits, core offset zero and memory offset +4500 MHz read back before and after each pass. GPU temperature peaked at 74 / 50 °C.

## Why this run exists

The decode curve’s code/prose requests force 1,024 greedy tokens with thinking on. The corrected 2026-11-16 probe records all output in reasoning and zero answer characters for both prompts; the earlier thinking/answer frame split was incorrect and is withdrawn ([correction](headline-r1.md#thinking-split-correction), [numeric observations](2026-11-16-headline-r1/code-summary.json)). The old 248.7/251.4 t/s curves describe those thinking-on prompts, rather than code/prose answers; frame and character counts do not measure token shares.

## How decode rate is computed here

The usual client formula, `(output_tokens - 1) / (latency - TTFT)`, overstates answers that take only a few decode steps. TabbyAPI streams one frame per decode step, but the last step's tokens arrive in two frames about 0.1 ms apart, so that step's tokens are counted without its time. On a 26-token translation (eight steps) the client formula reads 43 % high; on MBPP and extraction answers (65–94 tokens) 10–15 % high; above 150 tokens 0–3 %.

The rates below are step-based: tokens per decode step, from the server's per-request counters ([`server-steps.jsonl`](20261116T102550Z-decodekinds-r1/A/server-steps.jsonl), extracted from the container log by [`server_steps.py`](../../scripts/decodekinds-r1/server_steps.py)), divided by the mean interval between the client's frames, ignoring intervals under 2 ms. At four streams the mean includes the steps that also carry a new request's prefill. Where answers are long, the step-based rate is within 1–3 % below the client formula at one stream and within 1 % at four.

## Results

**By category** (`vllm bench serve` v0.30.0 through [`vllm_bench_tabby.py`](../vllm_bench_tabby.py): Spec-Bench's 480 prompts and 80 MBPP prompts, test split tasks 11 to 90, one assert shown; `chat_template_kwargs.enable_thinking=false`; temperature 0; output capped at 1,024 tokens without `min_tokens`; all 560 requests at one stream and again at four, per pass). Values are the mean of passes A and B. Drafts accepted per proposed come from TabbyAPI's streamed usage.

| category | prompts | mean output tokens | tokens per step, 1 stream | 1 stream, t/s | 4 streams, t/s per stream | drafts accepted / proposed, 1 stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math (MT-Bench) | 10 | 439 | 3.61 | 341.3 | 168.3 | 0.87 |
| math reasoning (GSM8K) | 80 | 306 | 3.53 | 334.0 | 164.3 | 0.84 |
| coding (MT-Bench) | 10 | 512 | 3.40 | 322.4 | 151.0 | 0.80 |
| extraction (MT-Bench) | 10 | 65 | 3.46 | 320.7 | 163.4 | 0.82 |
| code (MBPP) | 80 | 94 | 3.27 | 311.3 | 152.8 | 0.76 |
| reasoning (MT-Bench) | 10 | 284 | 3.09 | 291.6 | 139.5 | 0.70 |
| retrieval-augmented QA | 80 | 150 | 3.09 | 290.2 | 142.3 | 0.70 |
| STEM (MT-Bench) | 10 | 881 | 2.93 | 278.8 | 136.3 | 0.64 |
| translation | 80 | 26 | 2.95 | 274.7 | 141.7 | 0.65 |
| summarization | 80 | 247 | 2.81 | 264.8 | 128.0 | 0.60 |
| question answering | 80 | 348 | 2.76 | 262.3 | 125.7 | 0.59 |
| roleplay (MT-Bench) | 10 | 541 | 2.70 | 256.4 | 122.9 | 0.57 |
| humanities (MT-Bench) | 10 | 975 | 2.60 | 246.7 | 118.3 | 0.53 |
| writing (MT-Bench) | 10 | 430 | 2.52 | 238.9 | 116.5 | 0.51 |
| all 560 | 560 | 241 | 2.98 | 282.4 / 282.5 (A / B) | 137.3 / 136.5 (A / B) | |

Every request completed, none failed, and none carried a thinking frame. Finish reasons per category are in `summary.md`; 19 of 560 requests per pass reached the 1,024-token cap at one stream, 12 of them humanities, STEM and writing. The decode step takes 10.5–10.8 ms at one stream in every category; the rate follows tokens per step, that is, how many of the MTP head's three drafted tokens the model accepts. It drafts code, maths and extraction better than open-ended prose. Passes A and B agree within 1.3 % per category at one stream. At four streams the 80-prompt categories agree within 3 %; the 10-prompt MT-Bench categories differ by up to 4.2 % (coding) and 18 % (extraction), because each pass batches different neighbours with so few requests.

**MiaAI-Lab protocol** ([`bench_decode.py`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/6278ecb/tests/bench_decode.py), AGPL-3.0, run unmodified from its pinned blob and not copied here; temperature 0, thinking off, one stream, five requests capped at 400 tokens, median). The protocol's own rate uses the client formula above, so it is listed with the server's step counts:

| prompt | protocol rate, t/s, A / B | output tokens | decode steps | tokens per step | finish |
| --- | ---: | ---: | ---: | ---: | --- |
| coding: `clamp_range` function with a docstring | 400.2 / 399.9 | 59 | 17 | 3.47 | stop |
| structured: count from 1 to 200 | 393.3 / 393.2 | 400 | 100 | 4.00 | length |
| prose: explain how a hash map works | 305.5 / 305.0 | 400 | 130 | 3.08 | length |

The coding answer takes 17 steps, so its protocol rate carries the short-answer overstatement; at the 10.5 ms step of the category runs its 3.47 tokens per step is about 330 t/s. The structured prompt accepts every drafted token.

**Code-file rewrite** (R828's 12 rewrites through [`gate.py`](../../docker/overlays/prompt-lookup-r3/gate.py) `client --suite agent`: three revisions of a 160-line Python file in the prompt, the answer rewrites the last one with a one-line edit; thinking off, greedy, 2,048 forced tokens, one stream, one warm-up and three measured rounds): per-stream median 384.5 / 382.2 t/s (passes A / B), minimum 377.2, maximum 388.6. The answer copies its context, and prompt lookup and the MTP head draft most of it.

**Thinking on, control** (`fn_bench.py`, the decode curve's protocol at one stream, 1,024 forced greedy tokens, three measured requests per prompt and pass): code prompt 251.9 / 249.9 t/s and prose prompt 252.2 / 252.3 t/s (passes A / B), within 1.3 % of the decode curve's 248.7 / 251.4.

## Method notes

- Length is not forced in the category and MiaAI runs. A forced continuation past the model's natural end repeats text, which raises draft acceptance and the rate. The thinking-on control and the rewrite suite force length as their protocols do; at 1,024 and 2,048 tokens the client formula's last-step error is under 0.5 %.
- Server counters are matched to client records by generated tokens: at one stream the 560 counters appear in the client's order; at four streams each client record matches a counter with the same generated and prompt tokens within one position. Four or five matches per pass were between requests of equal length and unrecorded prompt length.
- MBPP stands in for a code-generation set because it is the one cached on the serving host, which runs offline. The prompt set is rebuilt by [`prep-dataset.py`](../../scripts/decodekinds-r1/prep-dataset.py) from Spec-Bench's `question.jsonl` and MBPP's test split.
- [`analyze.py`](../../scripts/decodekinds-r1/analyze.py) writes `summary.md` (client-formula rates, finish reasons and draft acceptance); `uv run bench/plot.py` prints the step-based table and draws [`decode-by-content.svg`](../../docs/img/decode-by-content.svg) from the same files.
