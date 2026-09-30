# R811, R811b, R812b, R813: the served `tabbyapi:merge-tok-r1` re-measured with the README's instruments; `vllm bench serve` output tok/s 0.99 to 1.09× the previous image in one session, and at 1 and 2 streams a decode-step speed-up that the previous image acquires at one ~786-token prompt does not occur with the prefill merge on

Results on the serving host: `2026-09-29-r811-std-bench-ab` (R811, 2026-09-29 20:34 to 23:27 UTC), `2026-09-29-r812b-layer-ab` (R812b, 2026-09-29 23:27 to 2026-09-30 01:00 UTC), `2026-09-29-r811b-sharegpt-c4` (R811b, 2026-09-30 01:01 to 01:16 UTC; the directory carries the date the unit was queued) and `2026-09-30-r813-decode-curve` (R813, 2026-09-30 01:17 to 01:34 UTC). Drivers: [`scripts/r811-std-bench-ab.sh`](../../scripts/r811-std-bench-ab.sh) (R811, and R811b as `UNIT=r811b-sharegpt-c4 DATASETS=sharegpt CONCS=4`) and [`scripts/r813-decode-curve.sh`](../../scripts/r813-decode-curve.sh); R812b's driver `r812b-layer-ab.sh`, its two extra launchers and the comparison probes `r811_ab_compare.py`, `r812b_decide.py` and `r813_curve_compare.py` are not in this repository. Raw records:

| unit | raw records |
| --- | --- |
| R813 | [`2026-09-30-r813-decode-curve/`](2026-09-30-r813-decode-curve/): `records.jsonl`, `curve.tsv`, `analysis.txt`, `compare.txt` (against R787a), `audit.txt` |
| R811 | [`2026-09-29-r811-std-bench-ab/`](2026-09-29-r811-std-bench-ab/): per arm (`NEW`, `OLD`) R787d's file set (`results/<cell>.json` without `generated_texts` and with the time arrays rounded to 1 µs, `results/<cell>.samples.tsv`, `cells/` client, container and environment logs, `boots.tsv`, `runs.tsv`, `foreign.tsv`, `summary.txt`, `summary.json`), `compare.txt`, `audit.txt` |
| R811b | [`2026-09-29-r811b-sharegpt-c4/`](2026-09-29-r811b-sharegpt-c4/): the same file set for the one cell |
| R812b | [`2026-09-29-r812b-layer-ab/`](2026-09-29-r812b-layer-ab/): per arm the result JSONs, sample manifests, `boots.tsv`, `runs.tsv`, `foreign.tsv`, `summary.txt`, `summary.json`; `decide.txt`, `decide.json`, `audit.txt`; no cell logs |

Boot logs, readbacks and launcher copies stay on the host. [`bench/plot.py`](../plot.py) draws `docs/img/decode-scaling.svg` and the solid lines of `std-bench.svg` from R813 and the dashed lines from R811's NEW arm with the ShareGPT 4-stream cell from R811 and R811b together, and prints the README's Standard benchmark tables and the NEW / OLD ratios below.

## Configuration

Served image (arm NEW in R811, R811b and R812b; R813): the launcher of [R809p](r803-r810-prefill-merge.md), md5 `6429dfa2…`, image `tabbyapi:merge-tok-r1` (`sha256:ac16920f72cf`), 41 environment keys including `EXL3_GR_MIX_TILED=1`, `EXL3_PREFILL_MERGE=1` and `EXL3_STASH_ASYNC=1`, readback `prefill-merge-r1 1 1`. Previous image (arm OLD): the R808 launcher, md5 `04e347cc…`, `tabbyapi:tokenize-offloop-r2` (`sha256:04b06fa97c4d`), 39 keys, readback `absent exllamav3.cache.prefill_merge`. Both: 8 slots, 901,120-token pool, split `[30, 30]`, draft policy `[[4, 3], [8, 2]]`, NVMe tier off, free VRAM at boot 1,181 / 1,759 MiB; power limits 600 / 575 W, core offset 0 and memory offset +4500 read back at every boot and after every cell. The agent clients that call the serving port directly were stopped and the API gateway drained. No clock or temperature was sampled during the cells.

## R813: the decode curve on the served image

R787a's measurement loop unchanged (`fn_bench --distinct`, 118-token code and 106-token prose prompts, greedy, 1,024 forced tokens, a warm-up round and three recorded rounds per shape, two boots); 432 of 432 requests ok, each 1,024 tokens with `finish_reason` `length`. No prompt is long enough for the prefill merge (more than 257 tokens), and neither container log has a `prefill-merge-r1` line: no merged prefill ran in either process.

| streams | decode per stream, t/s, code / prose | decode aggregate, t/s, code / prose | time to first token, s, code / prose | end-to-end burst aggregate, t/s, code / prose |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 255.9 / 278.1 | 255 / 278 | 0.13 / 0.13 | 247 / 269 |
| 2 | 208.9 / 201.7 | 419 / 406 | 0.24 / 0.23 | 395 / 380 |
| 3 | 179.2 / 176.4 | 535 / 531 | 0.34 / 0.33 | 493 / 497 |
| 4 | 152.6 / 154.4 | 622 / 623 | 0.45 / 0.43 | 561 / 575 |
| 5 | 138.1 / 138.5 | 694 / 693 | 0.56 / 0.53 | 634 / 637 |
| 6 | 121.3 / 121.7 | 735 / 734 | 0.61 / 0.61 | 667 / 675 |
| 7 | 114.6 / 114.6 | 816 / 805 | 0.67 / 0.63 | 728 / 738 |
| 8 | 104.5 / 106.7 | 838 / 857 | 0.79 / 0.69 | 762 / 772 |

- The two boots differ by at most 1.96 % per cell on the per-stream rate and 1.99 % on the decode aggregate (|a − b| / mean).
- At 2 to 8 streams every stream decodes during 90.1 to 98.9 % of the round's mean decode window.
- Tokens per decode step, median per request: 2.59 (code) and 2.82 (prose) at 1 stream, 2.28 to 2.35 at 5 to 8 streams. The container log's count (Σ generated / Σ verify steps, warm-up round included) agrees with the client's frames within 0.09 % on both boots.

Against R787a (2026-09-27, `tabbyapi:rebase-dev-r3`, same driver loop and prompts; `compare.txt`): time per decode step 0.996 (code) and 0.995 (prose) at 1 stream and 0.979 to 0.999 at 2 to 8 streams; tokens per step 1.000 at 1 and 2 streams and 0.986 to 1.010 at 3 to 8; per-stream rate 1.005 at 1 stream and 0.991 to 1.019 at 2 to 8. These are two sessions three days apart, so the ratios carry session-to-session drift as well as the image change. The round's pre-registered expectation of a 2 to 3 % longer step at 1 stream, taken from R811, did not appear. That rules out a fixed per-step cost of the served process on these prompts; it does not decide what R811 and R812b see at 1 and 2 streams, because `fn_bench` never sends the kind of prompt at which the difference starts (below).

## R811: `vllm bench serve`, served image against the previous image

[R787d](r787-bench-refresh.md)'s protocol: `vllm bench serve` v0.30.0 through [`bench/vllm_bench_tabby.py`](../vllm_bench_tabby.py), ShareGPT V3 400 conversations (seed 7310) and all 480 Spec-Bench questions at 1, 2, 4 and 8 streams, greedy, thinking on, `min_tokens` forcing each output to the ShareGPT reference length or to 256 tokens, a fresh boot per cell, passes A and B. The two images alternate cell by cell: in pass A NEW boots first, in pass B OLD. 32 cells, 14,080 requests, 0 failed, 0 cached prompt tokens, 0 foreign requests; every cell's output tokens equal the forced total (84,120 ShareGPT, 122,880 Spec-Bench), so no request stopped early. The per-arm gate (A/B spread of output tok/s at most 3 % per cell) failed for NEW at ShareGPT 4 streams (4.22 %) and for OLD at ShareGPT 4 streams (3.11 %), Spec-Bench 2 streams (3.16 %) and Spec-Bench 8 streams (3.46 %).

**Output tok/s** (NEW / OLD is the mean of the per-pass ratios):

| dataset | streams | NEW, pass A / B | OLD, pass A / B | NEW / OLD | mean TTFT NEW / OLD |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | 1 | 223.7 / 228.3 | 226.4 / 223.5 | 1.005 | 0.869 |
| ShareGPT | 2 | 315.3 / 315.7 | 317.3 / 318.0 | 0.993 | 0.926 |
| ShareGPT | 4 | 437.5 / 419.4; R811b 436.5 / 438.6 | 401.0 / 413.7; R811b 415.5 / 416.6 | 1.052 (four pairs) | 0.874 (four pairs) |
| ShareGPT | 8 | 527.1 / 530.2 | 482.2 / 491.8 | 1.086 | 0.846 |
| Spec-Bench | 1 | 245.7 / 244.5 | 249.4 / 247.1 | 0.987 | 0.917 |
| Spec-Bench | 2 | 358.4 / 359.3 | 342.8 / 353.8 | 1.030 | 0.856 |
| Spec-Bench | 4 | 486.8 / 484.0 | 465.8 / 465.5 | 1.042 | 0.892 |
| Spec-Bench | 8 | 580.5 / 581.2 | 554.2 / 535.4 | 1.067 | 0.863 |

- ShareGPT at 1 stream reads 0.988 in pass A and 1.021 in pass B, inside NEW's own A/B spread there (2.01 %). Spec-Bench at 1 stream is 0.985 and 0.990. At Spec-Bench 2 and 8 streams OLD's spread exceeds the gate, so those two ratios carry that caveat.
- Mean TTFT is 7.4 to 15.4 % lower in every cell. On Spec-Bench's summarization and RAG prompts (mean 779 and 770 prompt tokens) it is 18.2 / 18.6 % lower at 1 stream and 18 to 28 % lower over 1 to 8 streams; qa and math_reasoning read 7.6 % and 5.6 % higher at 1 stream (`compare.txt`). The median TTFT rises 1.6 to 2.7 % at 1 stream and up to 7.1 % at 2, because 60 to 65 % of the prompts are 257 tokens or shorter and never merge; they get their first token 2 to 5 ms later at 1 stream, while the merged longer prompts get theirs 42 to 80 ms sooner.
- At 4 and 8 streams the median decode step (ITL p50, one streamed frame per verify step) of NEW is 0.977 to 1.015 times OLD's. The gain in output tok/s comes from the steps that run other requests' prefill chunks: the frames longer than twice the median ITL hold this share of decode time (passes A / B):

| dataset | streams | NEW | OLD |
| --- | ---: | ---: | ---: |
| ShareGPT | 2 | 11.2 / 11.2 % | 12.7 / 12.8 % |
| ShareGPT | 4 | 21.1 / 22.5 % | 26.0 / 25.1 % |
| ShareGPT | 8 | 30.3 / 30.6 % | 36.3 / 35.0 % |
| Spec-Bench | 2 | 9.9 / 10.0 % | 12.3 / 11.5 % |
| Spec-Bench | 4 | 19.7 / 19.9 % | 22.7 / 23.0 % |
| Spec-Bench | 8 | 27.3 / 28.0 % | 31.5 / 32.2 % |

- At 1 stream the median step is longer on NEW in all four pass pairs: ITL p50 1.034 / 1.021 (ShareGPT) and 1.031 / 1.031 (Spec-Bench); at 2 streams 1.037 / 1.038 and 1.008 / 1.021. τ from the container logs is equal within 0.3 % at 1 stream and within 0.7 % in every cell. The per-stream rate (1000 / TPOT p50) at 1 stream is 2.5 % lower on ShareGPT and 2.9 % lower on Spec-Bench. Token-weighted TPOT (Σ decode time / Σ (tokens − 1)) on the prompts of 257 tokens or fewer, which neither merge nor stash in either image, is 1.040 / 1.013 (ShareGPT) and 1.032 / 1.031 (Spec-Bench) at 1 stream, and 1.016 / 1.016 and 0.980 / 1.003 at 2 streams. The next section shows where in the run this difference arises.

**ShareGPT at 4 streams.** NEW's two boots read 437.5 and 419.4 tok/s. The boots of that cell fall into two states whose whole ITL distribution differs by about 2.3 % (ITL p10 / p50 / p90: NEW A 15.53 / 17.37 / 18.02 ms, NEW B 15.83 / 17.77 / 18.51, OLD A 15.86 / 17.77 / 18.68, OLD B 15.64 / 17.51 / 18.40), in every tenth of the run, and both images hit both states. Paired by state, NEW / OLD is 1.057 (both fast) and 1.046 (both slow). The rule for publishing this cell was committed on 2026-09-29 at 23:57 UTC, after R811 and before R811b's first boot at 01:01 UTC: the published value is the mean of four NEW boots, R811's two and R811b's two, with their range, and the NEW / OLD comparison uses the four boots of each arm. R811b (same unit, both arms, passes A and B, fresh boots) read NEW 436.5 / 438.6 tok/s (spread 0.47 %) and OLD 415.5 / 416.6 (0.26 %), NEW / OLD 1.051 and 1.053; all four R811b boots are in the fast state (ITL p50 17.31 to 17.41 ms), so their small spread shows one state, not a stable cell. Published cell: **433.0 tok/s, range 419.4 to 438.6**, three fast boots and one slow. NEW / OLD over the four pairs is 1.052. At this cell the median decode step is equal between the images; the gain is fewer long frames and a 12.6 % shorter mean TTFT.

**The asynchronous stash log line.** None of the 16 NEW container logs has `EXL3_STASH_ASYNC on: first asynchronous stash`. That line is printed only on the stash path of a prefill that does not merge, which no prompt in either sample reaches: every prompt over 257 tokens merges, and none has (N − 1) a multiple of 256. The stash of every merged prompt (157 to 166 per cell) is finished on the worker thread through a path that prints nothing; the counter line that would show it is printed every 256 events. This is read from the overlay's source, not from a logged counter ([GOTCHAS 38](../../docs/GOTCHAS.md)).

## R812b: which layer carries the difference at 1 and 2 streams

Four arms at ShareGPT 1 and 2 streams, R811's protocol, fresh boot per cell, passes A and B in opposite order (NEW, NOASYNC, KNOBSOFF, OLD, then reversed): NEW the served launcher; NOASYNC without `EXL3_STASH_ASYNC` (40 keys, readback `1 0`); KNOBSOFF the served image without either merge key (39 keys, readback `0 0`); OLD the previous image. 16 cells, 0 failed, 0 foreign; each NEW and NOASYNC log has one `first merged prefill` line, fired by the first 737-token benchmark request, and KNOBSOFF and OLD have none. Ratios per layer, cells in the order pass A 1 stream, pass B 1 stream, pass A 2 streams, pass B 2 streams:

| layer | ITL p50 | median TPOT | token-weighted TPOT |
| --- | --- | --- | --- |
| stash: NEW / NOASYNC | 1.000 0.995 0.996 1.003 | 1.009 0.997 1.000 1.006 | 1.001 0.994 0.992 1.009 |
| merge: NOASYNC / KNOBSOFF | 1.025 1.031 1.028 1.029 | 1.014 1.028 1.021 1.026 | 1.018 1.029 1.009 1.008 |
| image: KNOBSOFF / OLD | 1.005 0.997 0.996 0.985 | 1.007 0.994 0.994 0.979 | 1.007 0.995 0.992 0.978 |
| total: NEW / OLD | 1.030 1.023 1.020 1.017 | 1.031 1.019 1.015 1.010 | 1.026 1.017 0.992 0.995 |

The pre-registered rule (a layer is attributed when its median-TPOT ratio exceeds 1.010 in all four cells) attributes the difference to the merge layer, verdict `ATTRIBUTED merge`; the asynchronous stash and the image's other layers are neutral. Median per-request TPOT is a weak metric here: 37 of the 400 outputs are shorter than 10 tokens, and their frames arrive coalesced. On ITL p50 the merge layer is attributed in all four cells; on token-weighted TPOT only at 1 stream, since at 2 streams the merged prefills also shorten the other stream's long frames and the net per-token difference is 0.8 to 0.9 %, and NEW / OLD at 2 streams is 0.992 / 0.995, NEW faster per output token. At 1 stream the 242 prompts of 257 tokens or fewer produce byte-identical generations in NOASYNC and KNOBSOFF (from the generated texts, which are not published), so the 1-stream difference on them (token-weighted 1.020 / 1.031) compares identical work; 76 of the 158 longer prompts generate different text under the merge.

**Where in the run the difference arises.** Ordering requests by send time: at 1 and 2 streams, the knob-off processes' median decode step falls at one request and stays lower to the end of the run, and the merge-on processes' does not. On ShareGPT it is request 45, a 786-token prompt (its prefill splits at 768 with a 17-row tail); on Spec-Bench request 29, 788 tokens (19-row tail). Earlier split prompts, with tails of 1, 27 and 34 to 254 rows, leave the step unchanged. ITL p50 after the trigger against before it, R811:

| dataset | streams | NEW, pass A / B | OLD, pass A / B |
| --- | ---: | ---: | ---: |
| ShareGPT | 1 | 0.997 / 1.002 | 0.973 / 0.968 |
| ShareGPT | 2 | 0.995 / 0.999 | 0.974 / 0.976 |
| Spec-Bench | 1 | 1.000 / 1.001 | 0.973 / 0.968 |
| Spec-Bench | 2 | 1.004 / 1.003 | 0.976 / 0.979 |

R787d's four ShareGPT cells at 1 and 2 streams on `tabbyapi:rebase-dev-r3` show the same fall at request 45 (0.964 to 0.966), and so do R812b's KNOBSOFF and OLD arms. In R812b the merge layer's ITL p50 ratio is 0.997 / 1.005 / 1.005 / 1.008 on requests 0 to 44 and 1.027 / 1.034 / 1.031 / 1.031 on requests 46 to 399; the stash and image layers stay within 1.6 % in both windows. Before request 45 the four arms' steps are within about 1.5 % of each other. The 1- and 2-stream difference between the images is therefore a speed-up that the knob-off processes acquire at that request and the merge-on processes, which merge the same prompt, never acquire; the step of the merge-on process does not become slower. The mechanism is not known. Reading the overlay's code finds no path that it adds to a decode step: the split hooks run only inside prefill forwards, and the decode and verify forwards take the fused recurrent path. R813's prompts never reach the trigger, so neither R787a nor R813 measured the knob-off fast state; the README's decode curve is the served state for short prompts.

## Figures and README

- `bench/plot.py` reads `2026-09-30-r813-decode-curve/records.jsonl` for the decode curve and, for the standard benchmark, `2026-09-29-r811-std-bench-ab/NEW/results` with its ShareGPT 4-stream cell replaced by the four boots of that directory and `2026-09-29-r811b-sharegpt-c4/NEW/results`; it raises when either directory is missing or R811b holds anything but that cell's two passes. `print_std_bench_tables()` prints the README tables, `print_r811()` the NEW / OLD ratios above, `print_r813()` the comparison with R787a.
- The prefill and depth figure is unchanged: R787b and R787c, 2026-09-27 on `tabbyapi:rebase-dev-r3`.
