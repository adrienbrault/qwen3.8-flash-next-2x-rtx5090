# R787: the README's decode curve, cold prefill, decode at depth and standard benchmark re-measured on `tabbyapi:rebase-dev-r3`

Four units in one chain on 2026-09-27, 15:06 to 17:00 UTC, under one GPU lock. Each repeats the round that drew a README figure until then, with that round's instrument, prompts and metrics:

| unit | replaces | results on the serving host | raw records | driver |
| --- | --- | --- | --- | --- |
| R787a, decode curve, 1 to 8 streams | [R719b](r719b-decode-curve.md) | `results/2026-09-27-r787a-decode-curve` | [`2026-09-27-r787a-decode-curve/`](2026-09-27-r787a-decode-curve/) (`records.jsonl`, `curve.tsv`, `analysis.txt`, `audit.txt`) | [`scripts/r787a-decode-curve.sh`](../../scripts/r787a-decode-curve.sh) |
| R787b, cold prefill to 240k tokens | the prefill part of [R580](r580-decode-curve.md) | `results/2026-09-27-r787b-prefill-curve` | [`2026-09-27-r787b-prefill-curve/`](2026-09-27-r787b-prefill-curve/) (`prefill.jsonl`, `prefill.tsv`, the per-target `fn_bench` logs, `greedy-S.json`, `docker-final.log`, `audit.txt`) | [`scripts/r787b-prefill-curve.sh`](../../scripts/r787b-prefill-curve.sh) |
| R787c, decode at 1 stream on 100k and 200k tokens of context | [R554](r554-depth-decode.md) | `results/2026-09-27-r787c-depth-decode` | [`2026-09-27-r787c-depth-decode/`](2026-09-27-r787c-depth-decode/) (`depth.jsonl`, `fn_bench.log`, `summary.txt`, `audit.txt`) | [`scripts/r787c-depth-decode.sh`](../../scripts/r787c-depth-decode.sh) |
| R787d, `vllm bench serve` on ShareGPT V3 and Spec-Bench | [R731b](r731b-std-bench.md) | `results/2026-09-27-r787d-std-bench` | [`2026-09-27-r787d-std-bench/`](2026-09-27-r787d-std-bench/) (as R731b's: `results/<cell>.json` without `generated_texts` and with the time arrays rounded to 1 µs, `results/<cell>.samples.tsv`, `cells/`, `boots.tsv`, `runs.tsv`, `foreign.tsv`, `summary.txt`, `summary.json`, `audit.txt`) | [`scripts/r787d-std-bench.sh`](../../scripts/r787d-std-bench.sh) |

Chain [`scripts/r787-chain.sh`](../../scripts/r787-chain.sh) with the shared checks of [`scripts/r787-common.sh`](../../scripts/r787-common.sh). Boot logs stay on the host: the launcher prints the host's LAN address. Figures: [`bench/plot.py`](../plot.py) draws `docs/img/decode-scaling.svg` and the solid lines of `std-bench.svg` from R787a, the dashed lines from R787d, and `prefill.svg` from R787b and R787c; a missing R787 file stops it rather than falling back to an older round.

## Configuration

The served launcher of [R785](r785-promote-rebase-r3.md) (md5 `e3db755f…`, checked before the lock and before each unit): image `tabbyapi:rebase-dev-r3` (`sha256:30ed33eb6036`), 42 environment keys with `EXL3_GR_MIX_TILED=1`, 8 slots, 901,120-token page pool at 8-bit KV, split `[30, 30]`, draft policy `[[4, 3], [8, 2]]`, NVMe tier off. Every boot read back power limits 600 / 575 W, core clock offset 0 and memory offset +4500 on both cards, again after the measurement, and 1,181 / 1,759 MiB free VRAM. The agent clients that call the serving port directly were stopped and the API gateway drained for the chain. Requests without the probes' own shape: 0 in each of R787a to R787c (report-only count), 0 in R787d (its gate).

## R787a: decode curve

Two boots, `fn_bench --distinct` (each stream of a round has its own prompt, 118 tokens code, 106 prose), greedy, 1,024 forced tokens, a warm-up round and three recorded rounds per shape; 432 of 432 requests ok. Metrics as in R719b: the decode rate per stream is the median over requests of (tokens − 1) / (time of the last token − time of the first); the decode aggregate is the sum of the decode rates of the requests running together; the end-to-end burst aggregate is all tokens over the round's wall time.

| streams | decode per stream, t/s, code / prose | decode aggregate, t/s, code / prose | time to first token, s, code / prose | end-to-end burst aggregate, t/s, code / prose |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 254.8 / 276.8 | 255 / 277 | 0.13 / 0.13 | 247 / 268 |
| 2 | 208.1 / 201.3 | 418 / 405 | 0.24 / 0.22 | 395 / 379 |
| 3 | 178.5 / 175.4 | 529 / 525 | 0.34 / 0.32 | 487 / 491 |
| 4 | 151.5 / 154.0 | 618 / 619 | 0.45 / 0.43 | 556 / 571 |
| 5 | 136.1 / 137.3 | 684 / 687 | 0.55 / 0.53 | 627 / 632 |
| 6 | 119.1 / 122.7 | 715 / 737 | 0.64 / 0.58 | 655 / 679 |
| 7 | 113.1 / 113.0 | 800 / 790 | 0.67 / 0.64 | 720 / 728 |
| 8 | 104.3 / 106.1 | 834 / 852 | 0.73 / 0.69 | 764 / 768 |

- The two boots differ by at most 1.3 % per cell on the per-stream rate and on the decode aggregate in 14 of 16 cells; at 6 streams they differ by 2.5 % (code) and 3.3 % (prose) per stream and 2.2 % and 2.3 % on the aggregate. R719b's boots differed by at most 1.5 %.
- The decode aggregate rises at every step from 1 to 8 streams for code and for prose.
- At 2 to 8 streams every stream decodes during 91.1 to 98.9 % of the round's mean decode window, so the aggregate overstates the rate the streams sustain together by at most about 9 %.

Against R719b (2026-09-25, the previous image with the same driver and prompts), per stream: code 0.961 to 1.004 at 2 to 8 streams, prose 0.963 to 1.002 at 1 to 8 streams, and code at 1 stream 0.854 (298.4 to 254.8 t/s). Each kind is one prompt. `fn_bench` streams one frame per verify step, so frames count decode steps. On the code prompt at 1 stream the new image takes fewer tokens per decode step: 395 steps for the 1,024 tokens in every request, against 343 on the previous image (2.59 against 2.99 tokens per step), at 1.7 % more time per step (10.19 against 10.02 ms, median per request). Whether the greedy text itself changed is not recorded. Prose at 1 stream, at 2.82 tokens per step on both images, reads 278.6 and 276.8. At 5 to 8 streams the median tokens per step per request is 2.28 to 2.37. These cross-day ratios are two sessions on two images; the same-session comparisons of the two images are [R784](r784-rebase-dev-r3.md) (short prompts: −1.3 to +1.6 % per cell at 1, 4 and 8 streams, code at 1 stream −1.28 % over 24 prompts) and [R786](r786-replay-abba.md) (agent replay: 1.015×). Across three instruments, including R784's same-session ratios of time per step, new image over previous, code and prose (0.996 and 0.996 at 1 stream, 0.998 and 1.005 at 4, 1.018 and 1.020 at 8), the time per decode step on the new image is level at 1 to 4 streams and at most about 2 % higher at 8 streams.

## R787b: cold prefill

One boot, tier off, the budget-to-token calibration of R580 (0.7534 tokens per budget unit), then three salted prompts per target, 64 forced tokens each, time to the first token as the server counts the prompt.

| target | prompt tokens (mean) | time to first token, s | prefill rate, t/s | R580 (2026-09-20), t/s |
| ---: | ---: | ---: | ---: | ---: |
| 30,000 | 29,932 | 2.68 | 11,159 | 9,706 |
| 60,000 | 59,850 | 4.97 | 12,037 | 10,381 |
| 120,000 | 119,530 | 9.87 | 12,108 | 10,538 |
| 200,000 | 199,425 | 16.48 | 12,097 | 10,636 |
| 240,000 | 239,110 | 20.08 | 11,910 | 10,543 |

The rate stays within 11,159 to 12,108 t/s from 30k to 239k tokens. Against R580 it is 1.130 to 1.160×; R580 ran on the image of 2026-09-20 at the stock memory clock ([R726](r726-memoc.md)), so the ratio is not the engine change alone. R784 measured the engine change in one session: 1.134× at 90k tokens.

## R787c: decode at depth

One boot, `fn_bench` at 1 stream, 2,048 forced tokens, greedy; run 0 prefills the context cold and run 1 decodes on the cached prefix. The code targets aim at about 100k and 200k tokens (R554's landed at 179,575 tokens, and its 266,000 target exceeded the window). The figure draws the mean of the two runs.

| kind | prompt tokens | decode, t/s, run 0 / run 1 | tokens per decode step | ms per decode step, run 0 / run 1 |
| --- | ---: | ---: | ---: | ---: |
| code | 101 | 257.3 / 263.3 | 2.70 | 10.50 / 10.26 |
| code | 99,812 | 331.9 / 333.1 | 3.49 | 10.52 / 10.50 |
| code | 199,083 | 294.3 / 298.4 | 3.16 | 10.73 / 10.60 |
| prose | 89 | 262.1 / 271.9 | 2.76 | 10.55 / 10.17 |
| prose | 99,724 | 371.7 / 373.4 | 3.85 | 10.37 / 10.32 |
| prose | 199,735 | 350.0 / 346.0 | 3.73 | 10.67 / 10.79 |

A decode step takes 10.2 to 10.8 ms at every depth, up to 6 % more at about 200k tokens than on a short prompt. The decode rate at depth is higher than at 0 because of draft acceptance: after the filler the model writes near-deterministic text, and 83 % (code) and 95 % (prose) of the 3 drafted tokens per step are accepted at about 100k tokens and 72 % and 91 % at about 200k, against 57 % and 59 % on the short prompts ((tokens per step − 1) / 3). The rise is a property of the filler text, not faster decoding over a long context. `prefill.svg` labels each depth point with its rate and its tokens per step. 4 prose streams on about 100k distinct tokens each (about 400k resident) decoded at 156 to 174 t/s per stream on the cached run.

## R787d: standard benchmark

R731b's protocol unchanged: `vllm bench serve` v0.30.0 through [`bench/vllm_bench_tabby.py`](../vllm_bench_tabby.py), a fresh boot per cell, passes A and B, ShareGPT V3 400 conversations (seed 7310) and all 480 Spec-Bench questions at every concurrency, greedy, thinking on, `min_tokens` forcing each output to the reference length (ShareGPT, mean 210 tokens) or 256 (Spec-Bench). 7,040 requests, 0 failed, 0 cached prompt tokens, 0 foreign requests. Pre-registered gate: the A/B spread of output tok/s at most 3 % per cell. Decision `PUBLISHABLE`, largest spread 0.39 % (Spec-Bench, 1 stream).

**ShareGPT V3**

| streams | output tok/s (wall clock) | A/B spread | req/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | per-stream tok/s (1000 / TPOT p50) | E2E p50 (s) | τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 221.6 | 0.08 % | 1.05 | 137 / 322–359 | 3.44 / 4.56–4.57 | 291 | 0.64 | 2.65 |
| 2 | 307.0 | 0.09 % | 1.46 | 194 / 541–542 | 4.83 / 9.06–9.19 | 207 | 0.90 | 2.65 |
| 4 | 397.0 | 0.36 % | 1.88 | 282 / 639–693 | 7.81 / 17.01–18.99 | 128 | 1.35 | 2.66 |
| 8 | 471.6 | 0.03 % | 2.24 | 382 / 1,343–1,500 | 14.02 / 23.16–24.34 | 71 | 2.23 | 2.30 |

**Spec-Bench**

| streams | output tok/s (wall clock) | A/B spread | req/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | per-stream tok/s (1000 / TPOT p50) | E2E p50 (s) | τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 244.5 | 0.39 % | 0.95 | 130 / 384–392 | 3.40 / 4.34 | 295 | 1.04 | 2.87 |
| 2 | 343.4 | 0.07 % | 1.34 | 183 / 496–503 | 4.79 / 6.58–6.67 | 208 | 1.47 | 2.87 |
| 4 | 445.0 | 0.04 % | 1.74 | 248 / 650–657 | 7.66 / 10.75–10.78 | 131 | 2.27 | 2.87 |
| 8 | 524.7 | 0.10 % | 2.05 | 394 / 978–1,028 | 13.46 / 17.40–17.42 | 74 | 3.83 | 2.43 |

Output tok/s, per-stream and τ are computed by `bench/plot.py` and the unit's summary from the raw records; p50 columns are the mean of the two passes, and p99 columns give both passes as a range where they differ.

The generated texts are removed from every published result JSON, as in R731b: some held a private-range IP address that the model repeats from a ShareGPT prompt. Every field that `bench/plot.py` and `bench/std_bench_summary.py` read is kept, and both reproduce the values above from the published files.

- Against R731b (2026-09-25, the previous image): output tok/s 0.996 to 1.007× in every cell; per-stream 0.990 to 1.001× except ShareGPT at 4 streams, 0.964× (132.7 to 128.0 t/s); TTFT p50 0.7 to 6.8 % lower in 7 cells and 3.7 % higher at Spec-Bench 8 streams (380 to 394 ms); τ within 0.02 of R731b's.
- At 1 stream the per-stream rates (291 ShareGPT, 295 Spec-Bench) sit above R787a's decode alone (255 code, 277 prose), as R731b's (292, 296) sat above R719b's prose (278.6). R787a's rate at 1 stream is one prompt per kind, at 2.59 and 2.82 tokens per step; R787d's τ over 400 and 480 prompts is 2.65 and 2.87 at 1 stream, as in R731b (2.64 and 2.87).
- No request ended on the engine's loop detector (R731b: one per ShareGPT cell except one).
- Spec-Bench τ per category at 1 stream: 2.45 (humanities) to 3.30 (math and math reasoning), in `summary.txt`.

## Limits

- Each unit is one session on one image. The cross-day ratios above compare two sessions and two images; they are not evidence of the engine change. The same-session evidence is R784 and R786.
- R787a is one prompt per kind. At 1 stream its rate follows that prompt's tokens per decode step; a different prompt would read a different rate at the same time per step.
- R787c's rates at depth depend on the filler text and read higher than the rate at short context through draft acceptance; they are not a speed-up from context.
