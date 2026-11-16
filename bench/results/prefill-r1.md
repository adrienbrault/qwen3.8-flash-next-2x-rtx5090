# Cold prefill reaches 12,537 tok/s at 120K prompt tokens

Results directory on the box: `20261116T205758Z-1619449-prefill-r1-{chain,b}`; frozen driver `prefill-r1/unit-b.sh` and checker hashes are in [packet-source-sha256.json](2026-11-16-prefill-r1/packet-source-sha256.json), and [numeric request records](2026-11-16-prefill-r1/prefill.jsonl), [summary](2026-11-16-prefill-r1/analysis.json), [TSV](2026-11-16-prefill-r1/prefill.tsv), [provenance](2026-11-16-prefill-r1/provenance.json) and [validation receipt](2026-11-16-prefill-r1/validation.json) are retained here.

Measured 2026-11-16, 20:58–21:02 UTC, in one boot of `tabbyapi:flashnext-names-fast-r1`, default model name, 2.68 bpw self-calibrated pack with HermiHg 2.6-bpw MTP, pool 819,200, eight available slots, split `[30, 30]`, 8-bit KV and NVMe tier off. The image and original launcher hashes match [headline-r1](headline-r1.md); this round changes no served configuration.

## Protocol and curve

One calibration request precedes three fresh salted prose requests at each target, c1, T=0, template-default thinking, no logprobs, `min_tokens=max_tokens=64`. Salts start at 268000 and advance by 997 per request; filler budgets use `round((target − 86) × 60000 / (calibration_prompt_tokens − 86))`, with calibration prompt count 45,051. All 22 requests finish by length with 64 server completion tokens, and client frames never exceed tokens. Calibration is retained and excluded from the curve. Each displayed prompt count, TTFT and rate is an arithmetic mean of three requests; rate averages the individual `prompt_tokens / ttft_s` ratios, including scheduling and first decode.

| target tokens | mean actual prompt tokens | cold prefill tok/s | mean TTFT s | requests |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 2,012 | 6,381 | 0.315 | 3 |
| 8,000 | 7,972 | 10,061 | 0.793 | 3 |
| 30,000 | 30,039 | 12,173 | 2.468 | 3 |
| 60,000 | 60,079 | 12,406 | 4.843 | 3 |
| 120,000 | 120,133 | 12,537 | 9.582 | 3 |
| 200,000 | 200,319 | 12,474 | 16.059 | 3 |
| 240,000 | 240,329 | 12,357 | 19.448 | 3 |

The observed peak is 12,537 tok/s at 120,133 prompt tokens, c1, prose, with 64 forced greedy output tokens and template-default thinking. These are single-boot means, with no A/B spread or multi-boot replication estimate. Spec-Bench retains its TTFT p50/p99 only: short prompts (median 128 tokens) are dominated by a roughly 0.11 s fixed per-request cost.

## Export and verification

`python3 -B bench/prefill.py` independently checks all request conditions, the calibration budget, target tolerance, seven summary cells and TSV values; `uv run bench/plot.py` draws the cold-prefill rate and TTFT from those checked records. Missing files, targets, repetitions or numeric inconsistencies fail. [Source hashes](2026-11-16-prefill-r1/source-sha256.json) identify the completed source artifacts, and all 40 frozen packet files were verified against their manifest before export by `bench/sanitize_prefill.py`.

The upstream unit is `VALIDATED`, its chain exit code is zero and the restored model ID matches the daily. The upstream checker requires all 22 server requests to have zero cached tokens and matching client/server prompt counts. The supplied source copy excludes server logs, so this publication independently reproduces request arithmetic and retains that validation result without independently repeating the log checks. No generated text, prompts, token-ID arrays, boot logs, private paths or raw request bodies are exported. Kernel caches use an independent copy of the served tuning cache; prompt-cache coldness does not mean cold kernel compilation. Stock GPU limits are 600/575 W, core offset zero and memory offset +4500 MHz.
