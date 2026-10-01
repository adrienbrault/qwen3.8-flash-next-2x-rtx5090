# Tail recurrent checkpoints recover edited-turn reuse while preserving the prefill pipeline

Results on flan: `2026-10-01-r823-cache-reuse-hiPd0J`, `2026-10-01-r823b-cache-reuse-7rzKkJ`, `2026-10-01-r823c-cache-reuse-DUO8ur`; drivers `r823-cache-reuse.sh`, `r823b-cache-reuse.sh`, `r823c-cache-reuse.sh`, then `r823p-promote-tailckpt.sh`; public records are the adjacent directories' `decide.json`, `decide.txt`, `audit.txt`, canary `client-*.jsonl`, and R823c's `gpu-equality.*` and `redo-validate-armD/decide-originals.*`.

All three experiments were measured on 2026-10-01. R823p promoted `tabbyapi:r823c-cachetail-inforward` on the same date. R823 and R823b remain `NOT-SUPPORTED`; the six original R823c boots are `RECOVERY-SUPPORTED / FIDELITY-UNRESOLVED`. The subsequent production gates are separate from the registered experimental fidelity test.

## Aggregate traffic observation and mechanism

The real-traffic analysis dated 2026-10-01 covers 473 completed requests on 2026-09-30, with 428 s of total prefill residency. Estimated decoding concurrency was one stream for 2,571.05 s and two for 491.70 s; there was no measured real decode episode above two streams. The cached-prefix plateaus were 35,328 tokens over 17 requests and 72.86 s, 32,768 over 12 requests and 70.36 s, and 65,536 over six requests and 22.14 s. Together these are 35 requests and 165.36 s. Only these aggregates are mirrored; real-client records and content are not published.

This hybrid model can reuse a prefix only when matching KV pages have a saved recurrent-state checkpoint at the resume position. The previous policy saved interior checkpoints every 32,768 rows relative to the restored cached prefix, plus the existing 2,048-row near-end grid and prompt-end checkpoint. An edit inside history cannot use a checkpoint after the mismatch. Synthetic edits at 48,000, 60,000 and 81,000 tokens reproduce the three observed plateaus; unchanged immediate HITs and append-only turns retain prompt-end reuse. This reproduces a mechanism without proving the real clients' edit positions or excluding branch changes and checkpoint eviction.

## Conditions and fixtures

Each round compares boots in one session on Qwen3.8-Flash-Next EXL3 2.50 bpw through TabbyAPI/ExLlamaV3 on two RTX 5090 cards. Controls: eight slots, 901,120-token page pool, 8-bit KV, layer split `[30, 30]`, 2,048-row prefill chunks, MTP depth policy `[[4, 3], [8, 2]]`, FASTWARM and prefill merge/async stash on, 4,096 MiB logical host recurrent-cache budget, NVMe tier off. A round's arms use one image and equal trace instrumentation: `tabbyapi:r823-cachetrace` for R823, `tabbyapi:r823b-cachetail` for R823b, and `tabbyapi:r823c-cachetail-inforward` for R823c.

The synthetic code/tool-history fixtures use seed 823, greedy temperature 0, `top_k=0`, `top_p=1`, `min_p=0`, `min_tokens=max_tokens=64`, disabled loop detection, no BOS and no token healing. Raw completions and rendered chat/tool paths are measured. Cases run at one stream, except C2's two simultaneous independent sessions. First seeds contain 54,465/67,618/88,703 tokens for T35/T32/T65; later edited prompts grow through the fixture sizes. T35 first warms a short prefix and reuses 2,560 tokens in its long seed. APPEND and MOVING-EDIT start at 50,000 tokens; EARLY-EDIT starts at 60,000. The immutable builder records every payload, actual token count and LCP.

Tables report per-case edited-turn median client time to first token, checkpoint gap in rows before the LCP, and cold engine prefill residency. Rows/s is processed rows divided by that residency; C2's timer includes intervening peer work and is not an aggregate throughput rate. Table values come from each round's public decision record except the explicitly corrected R823 T35 long-seed values. The two paired boots remain separate; measurements from different rounds are not paired comparisons.

## R823: global 2,048-row checkpoints

Order A1 B1 B2 A2; A uses the previous 32,768-row interior policy, B sets `EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP=2048`. Source: [`2026-10-01-r823-cache-reuse-hiPd0J`](2026-10-01-r823-cache-reuse-hiPd0J/decide.json).

| Raw case | Prompt tokens, first seed | A1 → B1 TTFT s | A2 → B2 TTFT s | A → candidate gap, rows |
| --- | ---: | ---: | ---: | ---: |
| T35 | 54,465 | 3.2032 → 1.9652 | 3.2145 → 1.9714 | 12672 → 384 |
| T32 | 67,618 | 5.6753 → 2.2760 | 5.6904 → 2.2705 | 27232 → 608 |
| T65 | 88,703 | 3.4945 → 1.6470 | 3.5003 → 1.6463 | 15464 → 1128 |
| MOVING-EDIT | 50,000 | 3.7611 → 1.9153 | 3.7656 → 1.9131 | 16735.5 → 1228 |
| C2 | 2 × 67,618 | 11.2363 → 4.3143 | 11.2343 → 4.3097 | 27232 → 608 |

| Raw seed | Prompt tokens / processed rows | A1 → B1 engine s (rows/s) | A2 → B2 engine s (rows/s) |
| --- | ---: | ---: | ---: |
| T35 | 54,465 / 51,904 | 4.3941 → 7.4531 (11,812 → 6,964) | 4.4301 → 7.3433 (11,716 → 7,068) |
| T32 | 67,618 / 67,617 | 5.4702 → 9.6754 (12,361 → 6,989) | 5.4981 → 9.7180 (12,298 → 6,958) |
| T65 | 88,703 / 88,702 | 7.2981 → 12.6951 (12,154 → 6,987) | 7.3168 → 12.7226 (12,123 → 6,972) |
| APPEND | 50,000 / 49,999 | 4.2338 → 7.1339 (11,809 → 7,009) | 4.2417 → 7.1170 (11,787 → 7,025) |
| C2 per concurrent job | 67,618 / 67,617 | 7.0732 → 18.6812 (9,560 → 3,620) | 7.0646 → 18.6738 (9,571 → 3,621) |

Edited-turn median across the five recovery cases falls 3.7611 → 1.9652 s and 3.7656 → 1.9714 s; median gap falls 16,735.5 → 608 rows. Cold long seeds cost 66–77 % more and concurrent cold engine residency rises about 164 %, exceeding the +10 % gate. The global policy is `NOT-SUPPORTED` despite passing recovery.

The checkpoint ends the LS pipeline window, and a window with fewer than two full chunks is rejected. A 2,048-row interval therefore disables interior lookahead. For the 67,618-token T32 seed, saves rise from three to 33 and engine residency from 5.4702 to 9.6754 s: about 140 ms per extra save including lost overlap, forward scheduling, copying and logging. These timestamps do not isolate device-to-host copy latency. The long T35 seed is already slower before any eviction, so LRU churn cannot explain the initial cold penalty.

## R823b: bounded tail density on the window-cut path

Order A1 B1 C1 C2 B2 A2. B uses global 16,384-row checkpoints; C uses 4,096-row checkpoints only within the last 12,288 prompt rows, retaining the coarse interior and near-end policies. Source: [`2026-10-01-r823b-cache-reuse-7rzKkJ`](2026-10-01-r823b-cache-reuse-7rzKkJ/decide.json). The following paired table is A:C.

| Raw case | Prompt tokens, first seed | A1 → C1 TTFT s | A2 → C2 TTFT s | A → candidate gap, rows |
| --- | ---: | ---: | ---: | ---: |
| T35 | 54,465 | 3.2095 → 1.8294 | 3.2084 → 1.8270 | 12672 → 384 |
| T32 | 67,618 | 5.6731 → 2.2909 | 5.6814 → 2.2945 | 27232 → 2656 |
| T65 | 88,703 | 3.5007 → 1.8052 | 3.5013 → 1.8050 | 15464 → 3176 |
| MOVING-EDIT | 50,000 | 3.7597 → 1.9250 | 3.7311 → 1.9238 | 16735.5 → 2252 |
| C2 | 2 × 67,618 | 11.2296 → 4.3489 | 11.2193 → 4.3476 | 27232 → 2656 |

| Raw seed | Prompt tokens / processed rows | A1 → C1 engine s (rows/s) | A2 → C2 engine s (rows/s) |
| --- | ---: | ---: | ---: |
| T35 | 54,465 / 51,904 | 4.3956 → 4.6613 (11,808 → 11,135) | 4.3731 → 4.6585 (11,869 → 11,142) |
| T32 | 67,618 / 67,617 | 5.4912 → 5.7708 (12,314 → 11,717) | 5.4764 → 5.7975 (12,347 → 11,663) |
| T65 | 88,703 / 88,702 | 7.2901 → 7.5982 (12,167 → 11,674) | 7.2857 → 7.5977 (12,175 → 11,675) |
| APPEND | 50,000 / 49,999 | 4.2345 → 4.5065 (11,808 → 11,095) | 4.2178 → 4.5029 (11,854 → 11,104) |
| C2 per concurrent job | 67,618 / 67,617 | 7.0664 → 8.0264 (9,569 → 8,424) | 7.0561 → 8.0340 (9,583 → 8,416) |

C recovers T32 and C2 TTFT in both raw/chat paths, with single-stream cold cost about 4–7 %. Concurrent cold engine residency rises 13.57–13.86 %, failing the registered +10 % gate; client cold medians rise about 6.7–7.2 %. The two extra tail saves end solo pipeline windows earlier and admit the second job earlier, changing interleaved waits. Once both jobs are active, 4,096-row checkpoints align with the existing two-chunk fairness limit; this is not the one-chunk pipeline exclusion of R823. No checkpoint newly saved by either concurrent seed is evicted during that seed pair. Engine residency includes peer work; the pair's elapsed seed span rises only about 0.22–0.36 s.

The global 16,384 arm B changes raw T32 TTFT 5.6731 → 3.3927 s and MOVING-EDIT 3.7597 → 2.6291 s in pair 1, with pair 2 5.6814 → 3.3917 s and 3.7311 → 2.6251 s. Its T32 gap is 10,848 rows; T35/T65 gaps remain 12,672/15,464. Concurrent cold 67,618-token seeds cost 7.0664 → 11.1788 s and 7.0561 → 11.1654 s per-job engine residency (9,569 → 6,049 and 9,583 → 6,056 processed rows/s), about +58 %. Both B and C are `NOT-SUPPORTED`.

## R823c: capture at chunk ends inside the forward

Order A1 C1 D1 D2 C2 A2 on the common R823c image. C retains the previous round's window-cut path; D has exactly C's density and adds `EXL3_RECURRENT_CHECKPOINT_INFORWARD=1`. Source: [`2026-10-01-r823c-cache-reuse-DUO8ur/redo-validate-armD/decide-originals.json`](2026-10-01-r823c-cache-reuse-DUO8ur/redo-validate-armD/decide-originals.json). The following paired table is A:D.

| Raw case | Prompt tokens, first seed | A1 → D1 TTFT s | A2 → D2 TTFT s | A → candidate gap, rows |
| --- | ---: | ---: | ---: | ---: |
| T35 | 54,465 | 3.2016 → 1.6287 | 3.2222 → 1.6286 | 12672 → 384 |
| T32 | 67,618 | 5.6811 → 2.0781 | 5.6833 → 2.0871 | 27232 → 2656 |
| T65 | 88,703 | 3.4971 → 1.7314 | 3.4836 → 1.7341 | 15464 → 3176 |
| MOVING-EDIT | 50,000 | 3.7604 → 1.7225 | 3.7311 → 1.7212 | 16735.5 → 2252 |
| C2 | 2 × 67,618 | 11.2415 → 4.3350 | 11.2266 → 4.3390 | 27232 → 2656 |

| Raw seed | Prompt tokens / processed rows | A1 → D1 engine s (rows/s) | A2 → D2 engine s (rows/s) |
| --- | ---: | ---: | ---: |
| T35 | 54,465 / 51,904 | 4.4197 → 4.4209 (11,744 → 11,741) | 4.3650 → 4.4015 (11,891 → 11,792) |
| T32 | 67,618 / 67,617 | 5.4797 → 5.4860 (12,340 → 12,325) | 5.4746 → 5.5138 (12,351 → 12,263) |
| T65 | 88,703 / 88,702 | 7.3040 → 7.3454 (12,144 → 12,076) | 7.3033 → 7.3462 (12,146 → 12,075) |
| APPEND | 50,000 / 49,999 | 4.2409 → 4.2604 (11,790 → 11,736) | 4.2152 → 4.2656 (11,862 → 11,721) |
| C2 per concurrent job | 67,618 / 67,617 | 7.0634 → 7.0918 (9,573 → 9,534) | 7.0556 → 7.0982 (9,583 → 9,526) |

Chat T32 TTFT is 5.7108 → 2.1107 s and 5.7081 → 2.1136 s; chat MOVING-EDIT is 3.7582 → 1.7231 and 3.7332 → 1.7235 s; chat C2 is 11.2268 → 4.3104 and 11.2173 → 4.3166 s. Both T32 and C2 clear the registered 25 % and 1.0 s recovery bars. D's gaps are unchanged from C: the gain is execution cost, rather than a denser resume grid.

Every D cold gate passes separately on both paired boots and raw/chat paths. Most changes are below +1 %, but this is not an exhaustive bound: pair 2 raw APPEND cold engine/client is +1.196/+1.245 %, and EARLY-EDIT is +1.736/+1.734 %. Both are below D's +3 % single-stream bar. Concurrent cold engine changes are +0.403/+0.604 % raw and +0.336/+0.252 % chat, below +10 %. Raw APPEND turn medians increase 2.11/10.64 ms, passing both +0.10 s and +20 % limits. Minimum free VRAM equals each paired A at 553/1,057 and 553/1,059 MiB; logical checkpoint occupancy peaks at 4,242,715,760 bytes (4,046.17 MiB). There are no relevant edited-checkpoint losses, under-reuse failures or OOMs. The contemporaneous C controls still fail the concurrent cold cost gate.

D reserves a prefill-merge staging slab before stage A and gives the chunk a private capture descriptor. Each GDN/conv/PLE layer captures the slot slices on its issuing CUDA stream; PLE ID history is cloned at capture time. Publication after both stages and page commit checks completeness, position and full-page ownership, then publishes the captured bytes rather than lookahead-overwritten live state. Existing draft work and the two-job fairness limit remain. Slab exhaustion cuts the window before any later stage A runs, then uses the coherent old stash path; exceptional cleanup drains pending stages before releasing descriptors.

The [`gpu-equality.json`](2026-10-01-r823c-cache-reuse-DUO8ur/gpu-equality.json) probe loads the actual model once, runs CUT, INFORWARD and forced NO_SLAB with KV/recurrent caches reset, and compares every checkpoint position, scalar, tensor shape/dtype, `torch.equal` result and tensor bytes. It passes T35/T32/T65 on both cards, including interior and prompt-end saves; T65 has five captures and zero ordinary fallbacks. NO_SLAB exercises positive fallback counts and matches CUT. Both measured D boots have 216 captures, 84 crossed checkpoints and no tail fallbacks. This tests checkpoint capture equality; it does not establish all A:D output partitions, schedules or NVMe restore fidelity.

## Fidelity and harness corrections

Across four R823c A:D boot pairings, all single-stream token outputs agree. C2 has 40/104 raw and 30/104 chat cross-arm token differences, compared with 17/52 combined A1:A2 differences and 11/52 D1:D2 differences. These reused comparisons are not independent trials. Finish reasons agree; alternatives are in the A-reference top five. Raw margins are at most 0.203125; one chat C2 comparison has margin 0.515625, so the earlier round's statement that every margin is below 0.5 does not carry forward. Same-arm differences limit attribution but do not prove that every cross-arm difference is harmless.

Eight chat APPEND differences disappear when only tool-call UUID fields are removed; tokens, function names/arguments and finish reasons agree. UUID values are redacted in the public records. Edit → immediate-HIT comparisons are identical payloads, contrary to the original R823 shorthand. Different resume/forward partitions change near-tie output even on the same prompt. These failures already occur in A and remain in the strict fidelity result; single-stream HIT-divergence counts match between A and D. The registered R823c verdict remains `RECOVERY-SUPPORTED / FIDELITY-UNRESOLVED`.

R823's original cold statistic selected T35's short 2,689-token warm-up instead of its 54,465-token long seed because the latter had a prior LCP. The archived `decide.*` is retained; the table above uses the reviewed long-seed residency, 4.3941 → 7.4531 and 4.4301 → 7.3433 s, so both cost comparisons fail. R823b and R823c gate the long seed. The harness also initially counted tool-call UUID differences as literal output failures; those remain distinct from token differences.

R823c traffic completed before `validate --arm D` was rejected by argparse's A/B/C choice list. The repair changes only that tuple to A/B/C/D; the archived and re-validation probes otherwise match. D1, D2 and the bug-triggered replacement RD revalidate as VALID without new traffic or changed thresholds. Once D1 is valid, RD is an excess replacement: the seven-boot decision is INCOMPLETE. The six original boots, in their registered order, produce the reported decision. RD is excluded by the replacement rule, not selected by outcome, and remains in the original record; its raw T32 TTFT 2.0801 s and C2 4.3218 s agree with D1/D2 within the observed spread. The [`redo-validate-armD/README.txt`](2026-10-01-r823c-cache-reuse-DUO8ur/redo-validate-armD/README.txt) records this repair.

## R823p production promotion and rollback

The immutable T32 first seed is 67,618 tokens and its first edit LCP is 60,000. The last reusable full-page prefix is 59,904; D's eligible checkpoint is 57,344, followed by immediate HIT reuse of 67,584. A proposed ≥59,000 edit gate was corrected before promotion because 59,392 belongs to R823's global 2,048 grid. The corrected smoke derives the expected position by executing the served checkpoint policy; it does not move the edit, shorten the seed or substitute a HIT.

The production gate checks pinned image/source/config, all baseline and D selectors, FASTWARM, boot free VRAM at least 1,067/1,635 MiB, trace on, disk off and smoke checkpoint/capture evidence. Before greedy tests, `fn_bench --distinct` runs code at one stream, greedy 1,024 forced output tokens, one warm-up and three runs, requiring median time per step ≤9.95 ms. Greedy and chat each must match six R809 reference rows; retrieval must hit all five insertion fractions at both tested depths. Any failure after installation automatically restores the previous launcher and daily.

The first try rolled back on a needle-checker sizing bug after 10/10 retrieval HITs. `fn_needle_oai` sizes filler by words: requested 131,072/240,000 context budgets tokenize to 105,680/193,464 prompt tokens, as in R810. The 90 %-of-requested-budget depth check was therefore wrong; the corrected check uses the measured tokenized depths. The second try promoted on 2026-10-01: FASTWARM passed, free VRAM was 1,099/1,667 MiB, code c1 median was 9.779 ms/step, greedy/chat each 6/6 identical, needles 10/10, T32 edit cache 57,344 then HIT 67,584. These production figures are transcribed from the operator's promotion finding, record `2026-10-01-r823p-promote-tailckpt-wCuc8r`; its raw artifacts were not supplied for this mirror.

The public launcher [`launch-flashnext.sh`](../../scripts/launch-flashnext.sh) and archive [`launch-flashnext-r823p-tailckpt.sh`](../../scripts/launchers/launch-flashnext-r823p-tailckpt.sh) mirror source MD5 `8b644c17` with the existing public substitutions. The production delta from source `f3975d68` is the image tag, three D selectors and explicit NVMe-off default. Image `EXL3_CACHE_TRACE=1` is separate from the 44 launcher keys. Rollback is the archived R818 launcher. NVMe restore on the new capture path remains untested.

## Open real-client attribution and public artifacts

The trace remains on to measure whether real edits fall in the final 12,288 prompt rows and whether retained checkpoints survive multiple sessions. It logs salted digests, role/page positions, LCP/lengths and cache/timing metrics, without prompt text or token IDs; CPU token arrays remain in a bounded registry. Synthetic recovery does not establish the real saved time. A 70–79 s recovery estimate out of the 428 s real prefill total is conditional on matching edit positions and checkpoint survival, rather than a production measurement.

The public record retains decisions, audits, canaries and the real-model equality report. All 17 measured-boot client JSONL files exceed 2 MB and are omitted; the six canary files are below the limit. Raw container/boot logs, real-client records, fixture payloads and tool-eval JSON are not mirrored. Private path prefixes, synthetic API UUIDs and the equality trace's process identifier are scrubbed. Summary records preserve each paired boot and case but do not replace the omitted per-request traffic archive needed for full re-validation.

Regenerate the immutable synthetic payloads with [`bench/r823_fixtures.py`](../r823_fixtures.py), seed 823, the served checkpoint's tokenizer and chat template, and `--nonce synthetic-823-v1`; the builder writes the manifest, gzip payload and SHA-256 manifest. The trace overlay README gives the command. [`r823_reuse.py`](../r823_reuse.py) contains the SSE client, joins, validation and decision logic; [`r823c_checkpoint_equal.py`](../r823c_checkpoint_equal.py) contains the actual-model equality probe. The cumulative image recipe and all SHA-pinned patch stages are in [`docker/overlays/r823c-cachetail-inforward/`](../../docker/overlays/r823c-cachetail-inforward/README.md).
