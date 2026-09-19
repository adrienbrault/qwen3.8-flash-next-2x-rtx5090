# R355: Quality: GSM8K as served

Results directory on the serving host: `results/2026-09-16-r355-fn-gsm8k`. Raw records: [`2026-09-16-r355-fn-gsm8k/`](2026-09-16-r355-fn-gsm8k/). Driver: [`scripts/r355-fn-gsm8k.sh`](../../scripts/r355-fn-gsm8k.sh). Date: 2026-09-16.


GSM8K, lm-eval, as served: `gsm8k`, 5-shot, `--apply_chat_template`, temperature 0, `max_gen_toks 8192`, limit 200,
`num_concurrent 4`, thinking on.

| arm | exact_match (flexible-extract) | conditions |
| --- | --- | --- |
| **as served here** | **0.925** (strict-match 0.920, stderr ±0.019) | this run, thinking on by configuration |
| **as served here, n=1319** | **0.9158** flexible-extract, **0.9151** strict-match, stderr **±0.0077** | `2026-09-16-r368-gsm8k-1319`, same parameters at 6.6× the sample |

The n=1319 row is the figure to quote; the n=200 row is the same measurement at a wider interval. The vllm-exl3
route's GSM8K figure on the same checkpoint is in the vLLM-route section below.

Thinking is on in this arm and can be seen doing so: a hand-checked item returned 126 chars of
`reasoning_content` plus the answer, and the per-request completion lengths across the 201 requests were
min 13 / p50 232 / p90 449 / max 1,660 tokens. This checkpoint reasons **briefly** — that is its character, not a
template flag left off.
