# R826: the served daily measures 2.22 to 3.79 % more standard output at 1 and 2 streams; warm-up attribution is supported but not isolated

Results on the serving host: `2026-10-01-r826-std-ab`, 2026-10-01 19:34 to 23:16 UTC. Driver: [`scripts/r826-std-ab.sh`](../../scripts/r826-std-ab.sh). Raw records: [`2026-10-01-r826-std-ab/`](2026-10-01-r826-std-ab/), including `records.jsonl`, `curve.tsv`, `analysis.txt`, `decode-boots.tsv`, `decode-foreign.tsv`, `compare.txt`, `compare.json`, `summary.txt`, sanitized `audit.txt`, and per-arm main and `c4-repeat` standard results, manifests, summaries and boot/run/foreign tables. This round measures the served state; it makes no promotion.

## Configuration and instruments

| arm | launcher md5 | image and ID | launcher selectors / container EXL3 entries | env SHA prefix |
| --- | --- | --- | ---: | --- |
| NEW | `262e9c31f714724409635fbac9df8ac2` | `tabbyapi:r825c-hostprepare`, `sha256:aa04a1cbe94b60bbd91e3d69d15f5e1b6eb679f14885e599404de218d8a897c9` | 46 / 49 | `b80407c61365` |
| OLD | `6429dfa2035a37dba51bb09651d374e8` | `tabbyapi:merge-tok-r1`, `sha256:ac16920f72cf41864ed7f4151bd18bc4ce593015dc0bc263284a8fde8c111454` | 41 / 43 | `c27236961230` |

NEW includes [R818p's boot warm-up](r815-r820-fast-state.md), [R823p's tail checkpoints](r823-tail-checkpoints.md) and [R825p's resumable whole-prompt window](r825-whole-prompt-window.md). OLD is the R809 merge daily measured as NEW in R811 and by R813. Both use eight slots, a 901,120-token pool at 8-bit KV, split `[30, 30]`, draft policy `[[4, 3], [8, 2]]`, the same tuned kernel cache, NVMe tier off, stock power limits 600 / 575 W, core offsets 0 / 0 and memory offsets +4500 / +4500. Settings are read at every boot and after each standard cell. Free VRAM on decode boots is NEW 1,099 / 1,667 MiB against OLD 1,181 / 1,759. These are limit/offset readbacks; no continuous clock, temperature or P-state telemetry is recorded.

The standard instrument is R811's `vllm bench serve` v0.30.0 with [`bench/vllm_bench_tabby.py`](../vllm_bench_tabby.py): ShareGPT V3, 400 conversations, seed 7310, reference-length forced outputs totaling 84,120 tokens per cell; Spec-Bench, 480 requests forced to 256 tokens, totaling 122,880. Requests are greedy, thinking on, closed loop at c1/c2/c4/c8, fresh boot and tier off for every cell. Pass A boots NEW then OLD, pass B OLD then NEW; 32 main cells. A mandatory ShareGPT c4 A/B repeat adds four boots, and the published c4 mean retains all four boots per arm under R811b's composite rule. Input lengths are mean 272 / 322 and maximum 1,070 / 1,540 tokens (ShareGPT / Spec-Bench). This sample does not measure R825p's long-prompt prefill benefit.

The decode instrument is R813's `fn_bench --distinct` loop: c1–c8, code then prose, 118-token code and 106-token prose prompts, greedy, 1,024 forced tokens per request, one unrecorded warm-up and three recorded rounds per shape. Four fresh boots run OLD1, NEW1, NEW2, OLD2, with 216 recorded requests per boot. NEW includes the served launcher warm-up; no standard-instrument warm-up is added to decode boots. The archived R811 helpers, entire `cell()` and matrix loop, and R813's decode loop are byte-identical to R826's before public path sanitization; the analysis changes only arm selection. The decode probe md5 is `b457fa447eee9919b28dfd088c96bc9c`.

## Decode curve, both arms

Per-stream rate is the median over requests after the first token; decode aggregate is the mean over rounds of the sum of request rates, never median rate multiplied by concurrency. TTFT is the median per request. Each row pools two boots and three rounds per boot; code / prose, 1,024 forced greedy tokens on 118 / 106-token prompts, `2026-10-01-r826-std-ab/records.jsonl`.

| streams | NEW per stream, t/s, code / prose | OLD per stream, t/s, code / prose | NEW aggregate, t/s, code / prose | OLD aggregate, t/s, code / prose | NEW TTFT, s, code / prose | OLD TTFT, s, code / prose |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 264.8 / 287.4 | 256.9 / 278.8 | 265 / 287 | 257 / 279 | 0.13 / 0.13 | 0.13 / 0.13 |
| 2 | 215.3 / 209.3 | 208.9 / 203.7 | 432 / 418 | 420 / 407 | 0.23 / 0.22 | 0.24 / 0.23 |
| 3 | 179.3 / 176.1 | 179.1 / 176.1 | 537 / 529 | 532 / 530 | 0.34 / 0.32 | 0.34 / 0.33 |
| 4 | 153.4 / 156.0 | 155.1 / 156.8 | 626 / 629 | 629 / 628 | 0.45 / 0.44 | 0.46 / 0.43 |
| 5 | 137.7 / 139.1 | 137.1 / 139.6 | 689 / 697 | 689 / 698 | 0.55 / 0.53 | 0.56 / 0.53 |
| 6 | 120.1 / 122.7 | 120.8 / 124.4 | 726 / 739 | 730 / 746 | 0.62 / 0.58 | 0.64 / 0.59 |
| 7 | 114.7 / 115.1 | 115.2 / 114.1 | 811 / 808 | 817 / 800 | 0.68 / 0.63 | 0.74 / 0.65 |
| 8 | 105.3 / 106.8 | 105.9 / 106.7 | 844 / 859 | 847 / 859 | 0.73 / 0.70 | 0.79 / 0.69 |

NEW's secondary round-wall aggregates and all per-stream/TTFT values are also in `NEW/decode/curve.tsv`; OLD's are in `OLD/decode/curve.tsv`. Across NEW's two boots the maximum per-cell spread is 2.48 % for per-stream decode rate and 3.07 % for decode aggregate. At c2–c8 the shared decode window covers 90.7 to 98.9 % of the mean window. NEW decode aggregate rises from c5 to c6, code 689 to 726 t/s and prose 697 to 739; median tokens per step across c5–c8 range from 2.27 to 2.34. Prose per-stream rate is 8.5 % above code at c1 and 1.5 % above at c8.

Step time is computed per request as `1000 * decode_window_s / (client_frames - 1)`, then median over both boots for each arm/shape. Tokens per step is mean `completion_tokens / client_frames`; these mean acceptance ratios are distinct from the median acceptance stated above. Ratios are NEW / OLD, from `records.jsonl`.

| streams | code ms/step ratio | prose ms/step ratio | code tokens/step ratio | prose tokens/step ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.970367 | 0.970288 | 1.000000 | 1.000000 |
| 2 | 0.970275 | 0.973117 | 1.000000 | 1.000000 |
| 3 | 1.001488 | 0.999107 | 1.009807 | 1.000468 |
| 4 | 1.003473 | 0.998292 | 0.995443 | 0.997151 |
| 5 | 0.996354 | 0.993558 | 0.998293 | 0.998625 |
| 6 | 1.005827 | 1.003568 | 0.994902 | 0.992904 |
| 7 | 0.997962 | 0.987773 | 0.993731 | 1.001624 |
| 8 | 1.000454 | 0.994357 | 1.000987 | 0.995466 |

All four c1/c2 shapes improve in both boot pairs, with step-time ratios 0.9686–0.9748. Their pooled acceptance ratios are 1.000000: actual tokens/step at c1/c2 are code 2.5924 / 2.6263 and prose 2.8209 / 2.5373. Equal counts establish equal acceptance/frame counts; this instrument stores no generated text and cannot establish token identity. At c3–c8 median step-time ratios span 0.987773–1.005827, including prose c7 at 0.987773 (20.2375 versus 20.4880 ms), boot-pair ratios 0.985944 / 0.989605. Decode-aggregate ratios at c3–c8 span 0.9903–1.0098. The c7 observation is consistent across these two boots, but two boots and many endpoints do not establish a general c3–c8 improvement.

Code decode TTFT ratios are 0.965732 / 0.914980 / 0.934437 at c6/c7/c8. c6 improves in both pairs; c7/c8 are dominated by slower OLD2 admission, with pair ratios 1.0242 / 0.8947 and 1.0089 / 0.8816. These are admission measurements, separate from steady-state decode.

## Standard benchmark, both passes

Output tok/s is `sum(output_lens) / duration`, wall clock including prefill and request turnover. Pooled ratios are ratios of the two pass means, rather than means of pass ratios; ITL p50 pooled is the ratio of the two boot-median means. Main matrix only in this table; ShareGPT c4's repeat and unconditional composite follow separately. All cells are greedy with the forced lengths and input conditions above, `2026-10-01-r826-std-ab/{NEW,OLD}/results/`.

| dataset | streams | NEW output tok/s, A / B | OLD output tok/s, A / B | output NEW/OLD, A / B | output pooled | mean TTFT pooled | ITL p50 pooled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | 1 | 230.2 / 232.7 | 223.7 / 226.8 | 1.029010 / 1.025932 | 1.027461 | 0.989010 | 0.969402 |
| ShareGPT | 2 | 321.1 / 328.5 | 313.7 / 321.8 | 1.023555 / 1.020867 | 1.022194 | 0.987534 | 0.975256 |
| ShareGPT | 4 | 417.7 / 435.2 | 417.9 / 435.9 | 0.999643 / 0.998587 | 0.999104 | 1.003614 | 0.999165 |
| ShareGPT | 8 | 521.9 / 526.0 | 511.5 / 530.7 | 1.020265 / 0.991046 | 1.005386 | 0.996226 | 0.996627 |
| Spec-Bench | 1 | 250.5 / 256.0 | 244.1 / 246.1 | 1.026291 / 1.040249 | 1.033299 | 0.964764 | 0.968997 |
| Spec-Bench | 2 | 364.8 / 364.9 | 347.5 / 355.6 | 1.049895 / 1.026156 | 1.037889 | 0.953573 | 0.969113 |
| Spec-Bench | 4 | 463.8 / 479.1 | 466.1 / 487.8 | 0.995023 / 0.982242 | 0.988487 | 0.995128 | 1.003339 |
| Spec-Bench | 8 | 553.1 / 583.6 | 555.3 / 570.3 | 0.996119 / 1.023195 | 1.009838 | 0.969341 | 0.996320 |

Pooled c1–c2 output gains are 2.22 to 3.79 %, with ITL p50 savings of 2.47 to 3.10 %. Main c4/c8 throughput shifts are smaller than the maximum within-arm A/B spread for their cell: ShareGPT c4/c8 4.21 / 3.69 %, Spec-Bench c4/c8 4.55 / 5.36 %. That comparison describes observed variation; it is not an equivalence test.

Spec-Bench c2 mean TTFT is NEW 203.911 / 206.412 ms and OLD 219.837 / 210.464, ratios 0.927555 / 0.980750, pooled 0.953573 (−4.64 %), only narrowly larger than OLD's 4.36 % A/B spread. Archived summarization and RAG means give pooled ratios 0.934427 / 0.936408 (−6.56 / −6.36 %), 80 requests per group per pass, with pass ratios 0.8963 / 0.9758 and 0.8918 / 0.9851. These group ratios reproduce from `compare.json`, `std.categories`; the source category dataset is not published, so an independent category-to-manifest-SHA remapping cannot be performed here. Other standard mean-TTFT shifts are smaller than their cell's maximum A/B spread.

## ShareGPT c4: retain the mode mixture

All four boots per arm are retained, main A/B and repeat A/B, 400 reference-length forced greedy outputs per boot. Raw cells are under `{NEW,OLD}/results/` and `c4-repeat/{NEW,OLD}/results/` in `2026-10-01-r826-std-ab`; each entry below is wall-clock output tok/s and ITL p50 in ms.

| arm | main A | main B | repeat A | repeat B | output mean | output range | four-boot spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NEW | 417.732; 17.8085 | 435.244; 17.3377 | 420.483; 17.7412 | 432.997; 17.4239 | 426.6143 | 417.7–435.2 | 4.10 % |
| OLD | 417.881; 17.7924 | 435.860; 17.3831 | 437.833; 17.3480 | 441.063; 17.2521 | 433.1593 | 417.9–441.1 | 5.35 % |

The unconditional ratio of four-boot output means is 0.984890, −1.51 % for NEW. Repeat-only output spreads are 2.93 % NEW and 0.73 % OLD; four-boot spreads are 4.10 % and 5.35 %. NEW has two slow-mode boots, ITL p50 17.74–17.81 ms, and two fast, 17.34–17.42; OLD has one slow at 17.79 and three fast at 17.25–17.38. The mode-matched main pass ratios are 0.999643 / 0.998587; the repeat compares unequal modes. This supports a mode-mixture explanation for the unconditional composite rather than a demonstrated stack regression. The composite remains published with its range; no boot is selected out, and the repeat's ≤3 % spread describes the repeat only. The bands match [R811/R811b](r811-r813-std-bench-merge.md).

## Generation identity and attribution

The retained private originals' `generated_texts` are byte-identical across arms for all 400 ShareGPT and 480 Spec-Bench c1 outputs in each pass. At c2, ShareGPT differs on 8/400 in A and 0/400 in B; Spec-Bench differs on 83/480 in A and 7/480 in B. Every forced output length still matches across arms. Identical generation work is directly verified for standard c1 only; decode c1/c2 counts alone establish acceptance identity. Generated texts are omitted from public JSONs under the R811 precedent, so the text-identity check requires the retained originals.

Boot warm-up is the attribution with the most support from prior controlled work. All 20 measured NEW boots record `FASTWARM ok` on uncached prompt counts 20/29/11; OLD does not. NEW c1 code measures 9.8071 ms/step against OLD 10.1066, matching the fast and slow levels. [R818p](r815-r820-fast-state.md) reached 9.797 ms on the same `merge-tok-r1` image as OLD before R823/R825, and R818's controlled launcher comparison measured code c1 ×0.9684 with greedy identity. R826 changes launcher and image together and has no NEW-with-warm-up-off arm; R823/R825 contributions and interactions remain possible. The historical single ~786-token prefill explanation was superseded by R814–R820; the merge daily can acquire the fast state during a ≥4-stream episode. This round does not isolate a CUDA mechanism or exclude all dynamic hardware variation.

## Integrity, publication and review

All 32 main, four repeat and four decode boots match their pinned launcher/image identities and selector counts. Env maps match the independent archived fixtures, readbacks are `prefill-merge-r1 1 1` and `tokenize-offloop-r2 1 1 1 12000 0 1` on every boot, and all 20 NEW landing checks require both return code 0 and the exact R825c PASS line. Boot logs, launcher snapshots, landing/readback files and the archived fixtures remain in the host records. Public boot tables retain the identity hashes, counts, offsets and pool readbacks.

The 36 standard cells have 15,680/15,680 successful requests, rc 0, no errors and failed 0. Manifests match across arms, passes and concurrency; input counts equal manifest lengths plus 52 chat-template tokens. All forced output lengths and cell totals match, parsed server completions equal client completions, and cached prompt tokens are zero. Decode has 864/864 successful 1,024-token, length-finished records, 216 per boot, three rounds and distinct indices per shape. Server completion counts including warm-up are 288 per decode boot; server/client aggregate tokens per step agree within 0.12 % in the retained logs.

All 40 foreign-table entries are zero. The detector counts request headers lacking `min_tokens`; traffic with that field can evade this classification. Independent review scanned all 40 measurement container logs: no OOM, Traceback/ERROR/Exception, TORCH_CHECK/c10::Error, unsupported or merge-disable lines. There are 15 truncated tool-call parse warnings in Spec-Bench, all successful length-finished requests. Both arms log merge activation in all 36 standard cells; short decode prompts have no merged-prefill lines. Fresh-process readbacks alone cannot establish server activity, and absence of an asynchronous-stash log line is not interpreted as inactivity.

Public data follows R811/R813: standard client/container/env logs, JSON results without `generated_texts`, time arrays rounded to 1 µs, manifests, summaries, boot/run/foreign TSVs; decode records, curves and analysis; sanitized audit and comparison reports. Boot logs and generated texts are omitted. Installation paths are removed from data, cache-trace run identifiers are removed and those trace JSON lines compacted, and every public raw file is under 2 MB. The published driver anonymizes the operator home and retains the generic installation root, as R811/R813 do; its runtime helper probes and env fixtures are host prerequisites, not included here.

Independent review on 2026-10-02 returned `CONFIRMED WITH CORRECTIONS`: c1–c2 gains reproduce; warm-up attribution has prior controlled support but is not isolated; c2 generation identity is qualified; c3–c8 includes the prose-c7 −1.22 % step observation; c4's repeat-only spreads are separated from four-boot spreads; TTFT c2 is uneven by pass; hardware readbacks are static. Per-arm `NOT-PUBLISHABLE` summary labels are A/B-spread warnings, not failed integrity checks. Both source launchers remained unchanged, and the served NEW daily was restored at 23:16 UTC. No promotion is implied.

## Figures

[`bench/plot.py`](../plot.py) reads R826 `records.jsonl` with tag prefix NEW for `decode-scaling.svg` and the solid lines of `std-bench.svg`; dashed lines read R826 NEW main cells with ShareGPT c4 pooled with R826 `c4-repeat/NEW`. Missing inputs and incomplete standard matrices raise. Historical R813/R787a and R811/R811b functions retain their original sources. The prefill/depth curve remains R787b/R787c, 2026-09-27 on `tabbyapi:rebase-dev-r3`.
