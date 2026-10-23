# Served self-calibrated 2.68 bpw, 2026-10-23

Results directories: `20261023T000904Z-publish268-{a,b,c}`, `20261023T005148Z-publish268-d` and `20261023T031345Z-publish268-e`. Instruments: [`probe.py`](../probe.py), [`vllm_bench_tabby.py`](../vllm_bench_tabby.py), lm-evaluation-harness GSM8K, tool-eval-bench 2.1.0 and [`needle.py`](../needle.py). Each linked directory contains publishable numeric or score projections; source receipt hashes remain with the operator.

All units were measured 2026-10-23, on image `sha256:2239954c372397b97ec3d570d887fe0afe679e8de214dfda558f5095d8f855fd` (`tabbyapi:tune268-r2`), model `qwen3.8-flash-next-exl3-2.68bpw-selfcal`, window 262,144, pool 819,200, eight slots, KV8/8, split `[30, 30]`, MTP draft GPU split `[0, 32]` and policy `[[4, 3], [8, 2]]`. The private served launcher's SHA256 is `7e36acd2019ffc21082d8a0d50d8c43ea5403c9b0cba5fe39968810e569288c7`; the public mirror changes the existing home-directory lookup and retains public explanatory comments. All 50 launcher selectors match the packet. `EXL3_DENSE_V2_K6=1`, `EXL3_MOE_COOP_HALF=1` and `EXL3_MOE_PREFILL_E3_HALF=1` are set by the launcher; the histogram, static and uncapped round-2 selectors remain image-default 0. NVMe is off, cache trace is on, the served warm-up runs, power limits are 600/575 W, core offset is 0 and memory offset is +4500 MHz on both cards.

Each boot uses an independent copy of the existing served tune cache, seeded before launcher invocation and retained through the cell. The recorded boot interval excludes old-container removal and cache-copy preparation; it includes launcher preflight, model load, ready checks, warm-up and the model check. It does not measure compilation from empty caches. The [unit-a boot table](20261023T000904Z-publish268-a/boots.tsv) records NEW1 at 18.269192979001673 s and NEW2 at 18.319459217003896 s. Both boot launchers report 1,213/815 MiB free on GPUs 0/1 after warm-up; later NVML readbacks are 1,212/814 MiB. Every unit has its own allowlisted `boots.tsv`, with tag, full image ID, source launcher SHA256, pool, selector count, clocks, power, both free-memory samples and boot seconds. No environment or mount paths are included.

## Decode, run 20261023T000904Z unit a

[Records, curve and analysis](20261023T000904Z-publish268-a/) retain all 432 requests: NEW1/NEW2, c1–c8, code/prose, three measured rounds per boot/shape after one unrecorded warm-up. Sampling is greedy; `min_tokens=max_tokens=1024`, every completion and server completion count is 1,024, and every finish is `length`. Each stream has a distinct suffix. Code prompts have 118 tokens and prose prompts 106, at the start of context.

Per-stream rate is the median request `decode_tps`, `(completion_tokens-1)/decode_window_s`. Decode aggregate is the mean of each round's sum of concurrent rates. Round-wall aggregate is a secondary measure: mean round token total divided by that round's `round_wall_s`, including TTFT and tails. TTFT is the median request time to first token. At two to eight streams the common decoding interval is 91.3–98.4% of the mean request decode window. No aggregate is a median per-stream rate multiplied by concurrency.

| streams | decode per stream, t/s, code / prose | decode aggregate, t/s, code / prose | TTFT, s, code / prose | round-wall aggregate, t/s, code / prose |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 248.7 / 251.4 | 249 / 251 | 0.163 / 0.155 | 239.7 / 242.4 |
| 2 | 212.9 / 198.4 | 425 / 397 | 0.293 / 0.281 | 394.9 / 367.8 |
| 3 | 166.2 / 166.6 | 500 / 509 | 0.429 / 0.410 | 459.6 / 451.3 |
| 4 | 145.1 / 143.4 | 586 / 580 | 0.561 / 0.540 | 528.6 / 528.7 |
| 5 | 130.0 / 124.2 | 648 / 624 | 0.696 / 0.667 | 585.9 / 566.2 |
| 6 | 113.0 / 114.4 | 677 / 683 | 0.847 / 0.777 | 609.9 / 621.6 |
| 7 | 107.2 / 105.1 | 749 / 733 | 0.844 / 0.808 | 673.8 / 668.7 |
| 8 | 98.8 / 98.2 | 795 / 786 | 0.919 / 0.910 | 712.6 / 709.0 |

The following table preserves boot medians and aggregate means separately. Spread is `abs(NEW1-NEW2)/mean(NEW1,NEW2)`.

| kind | streams | per-stream median NEW1 / NEW2, t/s | spread | decode aggregate NEW1 / NEW2, t/s | spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 1 | 249.320 / 248.360 | 0.386% | 249.350 / 248.467 | 0.355% |
| code | 2 | 212.510 / 212.870 | 0.169% | 424.963 / 425.527 | 0.132% |
| code | 3 | 164.800 / 167.460 | 1.601% | 497.597 / 502.770 | 1.034% |
| code | 4 | 145.445 / 145.100 | 0.237% | 585.950 / 586.153 | 0.035% |
| code | 5 | 130.690 / 129.110 | 1.216% | 653.937 / 641.110 | 1.981% |
| code | 6 | 111.780 / 113.370 | 1.412% | 672.513 / 680.827 | 1.229% |
| code | 7 | 107.010 / 107.930 | 0.856% | 750.717 / 747.297 | 0.457% |
| code | 8 | 99.815 / 97.890 | 1.947% | 801.017 / 788.513 | 1.573% |
| prose | 1 | 251.990 / 251.380 | 0.242% | 251.613 / 251.170 | 0.176% |
| prose | 2 | 198.265 / 198.630 | 0.184% | 396.410 / 397.260 | 0.214% |
| prose | 3 | 169.180 / 164.900 | 2.562% | 512.140 / 505.290 | 1.347% |
| prose | 4 | 144.030 / 143.395 | 0.442% | 580.787 / 578.513 | 0.392% |
| prose | 5 | 124.260 / 123.650 | 0.492% | 624.053 / 623.447 | 0.097% |
| prose | 6 | 114.445 / 114.090 | 0.311% | 683.553 / 682.273 | 0.187% |
| prose | 7 | 105.730 / 103.630 | 2.006% | 739.930 / 725.627 | 1.952% |
| prose | 8 | 97.725 / 98.280 | 0.566% | 783.650 / 789.007 | 0.681% |

Maximum two-boot spread is 2.562% for per-stream medians and 1.981% for aggregate means. The [standalone decode figure](../../docs/img/decode-scaling.svg) and the solid lines in [the standard figure](../../docs/img/std-bench.svg) read this unit only.

## Cold prefill, run 20261023T000904Z unit b

[Prefill records and budgets](20261023T000904Z-publish268-b/) contain one calibration and 15 measured requests. The chart ignores calibration. Each target has three salted prompts at c1, greedy, `min_tokens=max_tokens=64`, all finishing `length` with 64 completion tokens. X is mean actual server `prompt_tokens`; Y is mean request `prompt_tokens/ttft_s`, rather than a ratio of means. TTFT includes HTTP, queue and first-token work. The upstream unit was marked VALIDATED under its cold-cache server-log assertions; the copied packet lacks those final logs, as detailed under publication limits.

| target | filler budget | mean actual prompt tokens | mean prompt tokens/s | mean TTFT s | measured requests |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30,000 | 39,955 | 30,066.333 | 12,107.955 | 2.483333 | 3 |
| 60,000 | 79,909 | 60,098.000 | 12,416.968 | 4.840000 | 3 |
| 120,000 | 159,819 | 119,964.333 | 12,486.840 | 9.607333 | 3 |
| 200,000 | 266,365 | 199,977.333 | 12,439.795 | 16.075667 | 3 |
| 240,000 | 319,638 | 239,866.333 | 12,355.770 | 19.413333 | 3 |

## Decode at depth, run 20261023T000904Z unit c

[Depth records and summary](20261023T000904Z-publish268-c/) contain six c1 shapes with two 2,048-token greedy requests each, first cold then cached. Code filler budgets are 0/74,000/148,000; prose budgets are 0/133,000/266,000. The chart uses actual prompt counts and mean request decode t/s. `completion_tokens/client_frames` and `1000*decode_window_s/client_frames` retain the probe's token/frame and ms/frame definitions; each frame corresponds to a verify step. These differ from standard τ. No c4 depth series is measured.

| kind | actual prompt tokens | decode t/s | tokens per frame | ms per frame |
| --- | ---: | ---: | ---: | ---: |
| code | 101 | 295.715 | 3.14110 | 10.61887 |
| code | 100,435 | 290.535 | 3.16049 | 10.87423 |
| code | 199,822 | 252.815 | 2.82975 | 11.18570 |
| prose | 89 | 244.275 | 2.57935 | 10.55542 |
| prose | 99,878 | 268.530 | 2.93414 | 10.91797 |
| prose | 199,915 | 379.975 | 3.94605 | 10.38054 |

## Standard serving, run 20261023T005148Z unit d

[Standard results](20261023T005148Z-publish268-d/) retain all 16 cells, passes A/B × ShareGPT/Spec-Bench × c1/c2/c4/c8, and 7,040 measured requests. Each cell boots fresh, uses an out-of-set warm-up, runs closed loop and reaches its forced output count. ShareGPT uses the same 400 prompts at seed 7310 with the reference output lengths, mean 210.3 and total 84,120 tokens per cell. Spec-Bench uses all 480 prompts in 13 categories with 256 forced tokens, total 122,880 per cell. Mean input lengths are 272.255/321.952 tokens, maximum 1,070/1,540. Requests are greedy with template-default thinking.

Every request succeeds and finishes `length`; audited prompt hashes and requested/completion lengths match the sample manifests. The summary records zero cached tokens and no integrity/cache/foreign flags. The frozen standard validator returns `PUBLISHABLE`; maximum throughput spread is 0.467265% at Spec-Bench c2, within the original 3% c2/c4/c8 gate. The operator brief's ≤0.3% description does not match these raw values; the recorded 0.47% is used throughout. Neither pass is replaced or selected by rate.

Aggregate output is mean A/B `output_throughput`, all completion tokens over wall time including prefill and turnover. Per-stream is mean A/B `1000/p50_tpot_ms`, with TPOT `(E2E-TTFT)/(output_tokens-1)` per request. p50 columns are A/B means; p99 columns retain their range. A/B spread is absolute difference divided by the mean. These are the dashed lines in the figure, with each standard series sourced wholly from this run.

**ShareGPT V3**

| streams | output tok/s (wall clock) | A/B spread | req/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | per-stream tok/s (1000 / TPOT p50) | E2E p50 (s) | τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 207.7 | 0.04 % | 0.99 | 172 / 294 | 3.66 / 5.03 | 274 | 0.67 | 2.62 |
| 2 | 285.5 | 0.19 % | 1.36 | 230 / 535–594 | 5.22 / 10.41–10.42 | 192 | 0.97 | 2.60 |
| 4 | 371.0 | 0.14 % | 1.76 | 303 / 643–755 | 8.43 / 17.00–21.28 | 119 | 1.45 | 2.61 |
| 8 | 445.9 | 0.06 % | 2.12 | 374 / 1,910–1,918 | 15.07 / 27.41–30.39 | 66 | 2.37 | 2.28 |

**Spec-Bench**

| streams | output tok/s (wall clock) | A/B spread | req/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | per-stream tok/s (1000 / TPOT p50) | E2E p50 (s) | τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 226.6 | 0.13 % | 0.89 | 163 / 361–371 | 3.67 / 4.66–4.67 | 273 | 1.13 | 2.83 |
| 2 | 318.2 | 0.47 % | 1.24 | 220 / 506–532 | 5.22 / 7.05–7.15 | 192 | 1.59 | 2.84 |
| 4 | 415.0 | 0.02 % | 1.62 | 320 / 749–805 | 8.20 / 10.88–10.94 | 122 | 2.42 | 2.84 |
| 8 | 492.7 | 0.23 % | 1.92 | 394 / 1,479–1,503 | 14.41 / 17.95–18.08 | 69 | 4.10 | 2.42 |

τ is server-log `Σ generated/(Σ generated-Σ accepted drafts)`, including the prefill step. The [summary JSON](20261023T005148Z-publish268-d/summary.json) retains τ, client τ, accepted/proposed fractions, draft depth, ITL, percentile recomputations, category counts and token totals. Numeric detailed arrays are preserved in each projected cell JSON, alongside all request audits and sample manifests. Empty error slots are preserved; failed requests are never removed. No projection exceeds 2 MB, so numeric sharding is unnecessary.

## Quality, run 20261023T031345Z unit e

[Quality score projections](20261023T031345Z-publish268-e/) contain GSM8K, tool and needle summaries plus all score-only outcomes. The complete frozen GSM/tool/needle validators pass against the private raw inputs, including the updated streamed-transport and tool-log checks. The old `20261023T005148Z` unit e is excluded: its streamed proxy responses were HTTP 400. The unit d from that run remains valid. Units d/e from `20261023T000904Z` are excluded.

GSM8K is five-shot, n=500, eight concurrent requests, T=0 and max 8,192 tokens, with lm-eval stop strings removed. Flexible extraction is 491/500 = 0.982; strict extraction is 488/500 = 0.976. The outcome table contains all 500 IDs exactly once per filter, 1,000 rows. The API audit has 500 successful HTTP 200 responses, all naturally finishing `stop`; accuracy counts natural stop and length outcomes rather than rejecting low-scoring samples. Questions, responses and proxy logs are omitted.

Tool-eval-bench is version 2.1.0, all TC-01..TC-69, four sequential trials, parallel 8, T=.6, top_p=.95 and top_k=20, with CLI defaults retained. The final score is the last trial's 88, from 122/138 points. The four derived trial scores are 88/87/89/88, from point totals 122/120/123/122 out of 138 each. The CLI mean is 88, sample standard deviation 0.8, and reported 95% CI is [87.2, 88.8]. The score-only table has 276 scenario/trial rows, matching 276 positive-duration scenario log events. All 774 model requests succeed; finishes are 446 `tool_calls`, 327 `stop` and one `length`. The exact bodiless GET /version detection probe may return 404 and is excluded from model-response success counts. No readiness probe supplies scenario coverage. The CLI seed is omitted by the command and the recorded resolved field is null; no integer seed is available to publish. The tool JSON's autodetected server-model metadata is not an identity source: validated boot model responses and audited request models name the served 2.68 model.

Needles use five planted positions, fractions .08/.30/.55/.80/.96, at requested settings 131,072 and 240,000. Actual prompt counts are 105,680 and 193,464, respectively, with 5/5 hits at each. All ten requests reach the 64-token cap and finish `length`; hits search combined reasoning and visible content. These are substring retrieval checks, not proof that the requested settings equal the actual prompt length.

## Promotion evidence

The [allowlisted promotion-summary projection](2026-10-23-tune268-promotion/summary.json) records values from the operator's round-1 findings and round-2 ABBA summary; it is a projection of those summaries, not a new replay measurement. Round 1 was promoted at 2026-10-23 00:33 Paris = 2026-10-22 22:33 UTC. It uses turboderp's self-calibrated `2.68bpw_h6_ng8`: K3 experts on layers 0–3/44–47 and K2.5 on 4–43, dense/shared K6, 6-bit output head, 8-bit n-gram table and 2.5-bit MTP. Image `48f4ad3a` adds default-off dense K6 twins and HALF MoE decode; the launcher enables both. All 1,847 predecessor SASS functions were preserved and 37 kernels added. Real-weight parity passed 6,466/6,466, and B1/A1 and A2/A1 fork controls were 24/24 bitwise identical.

Round-1 same-image ABBA run `2026-10-22-210133-flashnext-tune268-abba`, pool 819,200: A has both new flags off; B has them on. Greedy decode forces 1,024 tokens; c4 is the sum of concurrent rates. Code/prose c1 is 220.2/227.1 → 242.9/252.8 t/s (+10.3/+11.3%); c4 aggregate 558.6/554.1 → 595.3/586.4 t/s (+6.6/+5.8%). Six-agent replay span is 51.9 → 49.2 s (−5.1%). Both A/B pairs improve each measured shape; within-arm c1 spread is ≤0.32%. Pool 819,200 passes the six-agent replay and a 120,312-token prompt; the round-1 finding records 193 MiB free on GPU 1 afterward. The one-boot 2.50 reference used a different image and pool; it is not used in the served figures.

Round 2 was promoted at 2026-10-23 00:04 UTC: image `2239954c` plus `EXL3_MOE_PREFILL_E3_HALF=1`. HIST, STATIC and UNCAP remain off. Its native build preserves every round-1 function and adds 14 kernels. The bitwise gate passes 380/380; the E3 HALF numerical gate passes 792/792. E3 HALF changes accumulation order and does not claim bitwise kernel parity against the fused fallback. Run `2026-10-22-234824-flashnext-tune268r2-abba` is A1/B1/B2/A2 on one r2 image and pool 819,200, with only E3 HALF differing. The source six-agent replay is a 20-request fixture.

| cell | code c1 t/s | code c4 aggregate t/s | prose c1 t/s | prose c4 aggregate t/s | six-agent span s |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 243.0 | 588.2 | 252.8 | 578.9 | 51.6 |
| B1 | 242.9 | 592.3 | 251.3 | 577.7 | 43.9 |
| B2 | 242.0 | 599.1 | 252.2 | 580.2 | 45.3 |
| A2 | 242.4 | 598.8 | 252.4 | 577.5 | 50.2 |

Mean B/A changes are code c1 −0.1%, code c4 aggregate +0.4%, prose c1 −0.3% and prose c4 aggregate +0.1%. Six-agent replay mean span is 50.9 → 44.6 s (−12.4%). B1/A1 forks are 24/24 identical and the A2/A1 control is 24/24 identical, with no near ties or failures. The pool gate has 20/20 complete agent requests and no engine OOM, a 120,312-token prompt and a separate 120,000-token prompt (64 output tokens, TTFT 9.504268436008715 s).

Cold prefill in that ABBA uses three unique 30,000-token prompts per cell and no long-prompt warm-up; TTFT includes HTTP/queue/first-token work. Cell median TTFTs A1/B1/B2/A2 are 3.000/2.406/2.534/2.971 s; mean B versus mean A cell medians are 2.470 versus 2.985 s (−17.3%). These promotion summaries remain separate from the later served five-point prefill curve.

## Publication limits and omissions

The raw units carry VALIDATED status. Locally repeated numeric checks pass for decode, prefill and depth. Their final container logs were not included in the supplied copy, so the original server-side count and zero-cached-token assertions cannot be independently repeated here. The recorded boot model/image identities, source launcher SHA256, pool, selector count, warm-up marker and clock/power readbacks were checked; no running server or GPU was accessed. Standard and all quality validators pass in full against the supplied inputs. The standard summary's legacy `window=0` field is not used as the request window; boot configs and the pinned launcher specify 262,144.

The public records contain no generations, prompt/answer strings, full tool-eval conversations, full lm-eval samples, endpoint addresses, host mount/config paths, environment dumps or container/boot logs. Standard JSONs retain all numeric values, detailed request arrays and identifying metadata; raw `boots` summary fields are replaced by the allowlisted boot table. Original file hashes and projection descriptions are in the private operator receipt. All public files are below 2 MB.

No memory allocation breakdown, bytes/token, code-edit lookup rate, SWE-bench score/replay/cost, NVMe restoration, structured-output/tool_choice/identity passing list, random workload/Poisson curve or pp2048/tg512 table was remeasured on this configuration. Those claims and the memory figure are omitted from the served README. Historical dated write-ups remain accessible through the index. The old MTP accepted-drafts row is omitted; probe tokens/frame and standard τ retain their distinct definitions. Neither invalid quality run is used to fill missing data.
