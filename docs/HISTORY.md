# History of the served configuration

How the served stack changed, oldest first. The current configuration is in [CONFIG.md](CONFIG.md); each experiment's write-up is indexed in [bench/RESULTS.md](../bench/RESULTS.md).

## 2026-09-16: the first served configuration


`qwen3.8-flash-next-exl3-3.05bpw` on TabbyAPI + ExLlamaV3 v1.5.0, port 8022, image `tabbyapi:qsa-cid-pr337`,
262,144-token window and cache, 8-bit KV, 8 slots, layer-split across both cards (TP is not implemented for this
architecture), MTP draft depth 3 with concurrency-indexed depth `[[2, 3], [8, 1]]`, vision on, `qwen3_coder` tools,
sampler preset `qwen38_thinking` (T=0.6, top_k 20, top_p 0.95 as **fallbacks**). See `docs/CONFIG.md` for why each
value.

**Identity of the served configuration, and the two fingerprints that describe it.** They are produced by different
tools and are NOT interchangeable — comparing one against the other is the same error as comparing throughput from two
instruments, and I made it once:

| fingerprint | method | value |
| --- | --- | --- |
| `r363`/`r340` gate | sha256 of the single-file greedy capture, first 16 | `750e1459e177c47e` (1,989 bytes) |
| `greedy_hash()` (`lib/greedy-compare.sh`) | directory hash over the 4-prompt capture: count + sorted (relative name, content) pairs | `8179222fec8df3b8` |

An earlier `18e30f17883a38eb` from this function is superseded: the function was fixed to include the file count,
to list files portably and to refuse the empty-input digest (see `docs/GOTCHAS.md` #13), and every hash it produced
before that is superseded. The directory fingerprint is the one restores check ([`scripts/r373-restore.sh`](../scripts/r373-restore.sh)), and it
prints the measured value when no reference is pinned rather than judging against a number from another method.
`8179222fec8df3b8` is also the value the #290 re-captures and the #246 feature-off arm produced, which is consistent
with those variants being output-identical to the served configuration, as their own gates showed.

Both fingerprints are recorded so a future A/B can establish that it started from the same configuration. The primary
identity is **behavioural**: greedy output on the forced-length probe is `750e1459e177c47e` (1,989 bytes), and the
probe scripts compare against it. The generated config `/srv/qwen5090/flashnext-config.yml` (mounted read-only at
the container's `/app/config.yml`, byte-identical inside and out) was `sha256 12252e838eaa…` at that moment, but
**that hash moves when a comment in the launcher's heredoc moves** — the file carries its own explanation inline.
Verified 2026-09-16: after correcting six comments, the rendered config differed from the served one in comment lines
only, with no differing setting. Read the hash as a provenance marker, not as the contract; grep the file for the
keys when it matters.

## 2026-09-17: the promoted layers, in order (each admitted by its own gate; see [`docker/README.md`](../docker/README.md))


| promoted (CEST) | layer | gate evidence | decode c1 / c4 / c8 (fn_bench code 2048) | prefill 30k / 120k | results |
| --- | --- | --- | --- | --- | --- |
| 02:15 | bszn16 + policy `[[4, 3], [8, 1]]` | c1 fingerprint identical; GSM8K, tool-eval under concurrency | 203 / 375 / 443 | — | [R414](../bench/results/r414-bszn16.md) |
| 03:15 | coopwide | c1 byte-identical; c8 +7.5 % | 206–209 / 369–376 / 469–478 | — | [R421](../bench/results/r421-coopwide-ab.md) |
| 04:32 | hcmix2 (`EXL3_HC_MIX_V2=1`, `MIN_R=1`) + hostgap (`EXL3_HOST_GAP_REWIND=1`) | identical at MIN_R 1; c4 +12 %, c8 +8 % | 213–218 / 422–434 / 537–548 | — | [R428](../bench/results/r428-hcmix2-stack-ab.md) |
| 07:35 | prefill pipeline + nosync + mtpfix2 (`EXL3_LS_PREFILL_PIPELINE=1`) | c1 and 30k fingerprints identical | 213 / 430 / 540 | **3.5 s / 13.1 s** (was 5.5 / 22.3) | [R442, R446](../bench/results/r442-ppipe.md) |
| 12:45 | MoE coop V2 (`EXL3_MOE_COOP_V2=1`) | bit-exact at R = 1..16 in the kernel test, c1 + 30k fingerprints identical, five gates (`2026-09-17-r461-gates-moecoopv2`: GSM8K 0.935, tool-eval 85.5 ± 1.7, needle 5/5) | 207–214 / 425–450 / **550–604** (per stream 75–77) | unchanged | [R460, R461](../bench/results/r460-moecoop-v2-ab.md) |

Pool: 262,144 tokens at 8-bit KV is the ceiling on this box under any split (393,216 and 327,680 fail to boot: [R452](../bench/results/r452-exl3-cache-bits.md), R337). Structured output (llguidance `json_schema` / `response_format` / `regex_pattern`) works, thinking on and off, at c4: [R453](../bench/results/r453-exl3-structured.md).

Not promoted, 2026-09-17 14:20 CEST: MoE coop mode 3 (`EXL3_MOE_COOP_V2=3`, V1 path for singleton expert runs inside the V2 kernel) is bit-identical to mode 1 (same c1 and 30k fingerprints, GSM8K c8 0.935) but 2–3 % slower at c1, c4 and c8 (214 / 115 / 70 per stream against 220 / 118 / 73), because the mode-3 kernel is 8–19 % slower than V1 on rows that route to distinct or partly overlapping experts, which is what decode rows look like: [R462](../bench/results/r462-moecoop-v3-ab.md). The served configuration keeps mode 1.

## 2026-09-18 and 2026-09-19: slots, the 2.50bpw pack, decode round 4, grouped MoE prefill

| promoted (CEST) | change | gate evidence | decode c1 / c4, code (`fn_bench` 2,048) | cold prefill | results |
| --- | --- | --- | --- | --- | --- |
| 2026-09-18 12:24 | 4 slots, pool 360,448 (3.05bpw) | fingerprints identical, needle 5/5, tool-eval 84.8 ± 1.0 | 214–217 / 429–444 | unchanged | [R480](../bench/results/r480-exl3-pool.md) |
| 18:08 | shared expert on a side CUDA stream (`EXL3_SHARED_EXPERT_OVERLAP=1`) | byte-identical; paired tool-eval 83.5 against 84.1 | 218–221 / 448–461 | unchanged | [R490 / R491b](../bench/results/r490-shared-overlap.md) |
| 23:21 | 2.50bpw pack, pool 786,432 | new canonical fingerprints; GSM8K 0.978 against 0.980 without stop strings; tool-eval 86.0 ± 2.6 | 210–214 / 470–510 | 30k 7,558 t/s | [R495b](../bench/results/r495b-2p50-audition.md), [R509](../bench/results/r509-gsm8k-nostop.md), [R511](../bench/results/r511-promote-2p50.md) |
| 2026-09-19 01:10 | decode round 4 group I (pinned draft staging, batched verify, 64K draft head) | byte-identical; tool-eval 84.5 ± 1.9 | 218 / 510 | unchanged | [R499](../bench/results/r499-decode-r4.md), [R514](../bench/results/r514-promote-r4i.md) |
| 01:26 | grouped MoE prefill E3 (`EXL3_MOE_PREFILL_E3=1`), stacked image | c1 identical, 30k changes (prefill order); GSM8K 0.985 = OFF; tool-eval 85.8 ± 2.1 | unchanged | 60k 9,543 t/s, 120k 10,015 t/s | [R513](../bench/results/r513-prefill-e3-r2.md), [R517](../bench/results/r517-promote-stack.md) |

Measured and not promoted in the same period: chunk 1024 for pool ([R483](../bench/results/r483-exl3-pool-chunk.md), [R485](../bench/results/r485-pool-frontier.md)), split [30, 31] at 393,216 ([R487](../bench/results/r487-pool-393k.md)), the n-gram table in RAM ([R484](../bench/results/r484-ngram-ram.md)), the host KV tier ([R493](../bench/results/r493-host-kv-tier.md)), GDN state replay ([R496](../bench/results/r496-gdn-state-r3.md)), dynamic draft ([R497](../bench/results/r497-draft-confidence.md)), a 4-bit MTP graft ([R498](../bench/results/r498-mtp4-graft.md)), prompt lookup ([R501](../bench/results/r501-prompt-lookup.md), a stack candidate), int8 mixer weights as a speed lever ([R499](../bench/results/r499-decode-r4.md)).


## 2026-09-19: pool growth, the NVMe tier, and 8 slots

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 04:15 | int8 hyper-connection mixer weights (`EXL3_HC_MIX_V2_INT8=1`) | tool-eval 85.0 ± 1.4, GSM8K 0.974 | 819,200 | [R525](../bench/results/r525-promote-int8mix.md) |
| 05:16 | MTP draft chain on the GPU, 65,536-row draft embedding copy | byte-identical; tool-eval 84.8 ± 1.5, GSM8K 0.970 | 819,200 | [R528](../bench/results/r528-promote-mtp-pruned.md) |
| 05:43 | `tool_choice` enforcement in TabbyAPI | tool-eval 88.0 ± 1.6 (TC-45 0 → 2 points), GSM8K 0.978 | 819,200 | [R529](../bench/results/r529-promote-tool-choice.md) |
| 06:18 | fix for ExLlamaV3's PLE checkpoint aliasing | tool-eval 84.8 ± 1.3, GSM8K 0.974 | 819,200 | [R530](../bench/results/r530-promote-plefix.md) |
| 08:22 | persistent NVMe prefix tier | restart restore identical to cold; tool-eval 87.0 ± 1.2, GSM8K 0.976 | 819,200 | [R534](../bench/results/r534-promote-nvme-tier.md) |
| 09:23 | deterministic E3 prefill | prefill identical run to run; tool-eval 85.0 ± 0.8, GSM8K 0.972 | 819,200 | [R535](../bench/results/r535-promote-e3det.md) |
| 11:11 | decode kernels round 6 | byte-identical, +1.0 % code / +1.1 % prose at 1 stream; tool-eval 87.2 ± 1.5, GSM8K 0.974 | 819,200 | [R540](../bench/results/r540-promote-r6.md) |
| 16:30 | QSA raw-key ring | byte-identical; tool-eval 86.5 ± 2.4, GSM8K 0.976 | 983,040 | [R546](../bench/results/r546-promote-rawk.md) |
| 17:18 | bf16 GDN recurrent state | new c1 fingerprint, decode +0.3 to +2.1 % on 48 paired prompts; tool-eval 86.8 ± 2.6, GSM8K 0.974 | 1,032,192 | [R548](../bench/results/r548-promote-gdnbf16-ring.md) |
| 19:07 | 8 slots | fingerprints unchanged, prefill 1.00–1.02×; tool-eval 88.2 ± 1.0, GSM8K 0.974 | 966,656 | [R561](../bench/results/r561-promote-slots8.md) |
| 20:15 | n-gram row prefetch after the draft readback | byte-identical, +0.8 % code / +0.9 % prose at 1 stream over 4 boots; tool-eval 84.0 ± 2.4, GSM8K 0.978 | 966,656 | [R565](../bench/results/r565-promote-ngram-prefetch.md) |
| 23:46 | draft depth 2 at 5 concurrent jobs, policy `[[4, 3], [5, 2], [8, 1]]` | fingerprints unchanged, +10.7 % code / +14.1 % prose at 5 streams for −2.0 % at 6-stream prose; needles 5/5 and 5/5, tool-eval 86.0 ± 0.8, GSM8K 0.976 | 966,656 | [R576](../bench/results/r576-promote-c5-policy.md) |

Measured and not promoted in the same period: a deeper single-job draft ([R537](../bench/results/r537-draft-depth.md)), the K=3 MoE kernel without spills ([R536](../bench/results/r536-nospill.md)), the draft embedding copy on cuda:0 ([R555](../bench/results/r555-headdev-mirror.md)), adaptive draft depth ([R556](../bench/results/r556-adaptive-draft-r2.md)), prefill chunk 1,024 / 512 ([R553](../bench/results/r553-chunk-hot-stall.md)), draft depth 2 above 5 jobs ([R560](../bench/results/r560-c8-policy.md)), the draft-cache window, which promoted and then rolled back on a CUDA-graph capture out-of-memory at 5 concurrent ([R575](../bench/results/r575-promote-mtp-kv-window.md)), and prefill chunk 4,096, which does not boot at 8 slots ([R574](../bench/results/r574-chunk4096.md)).

## 2026-09-20: the windowed draft cache

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 00:29 | windowed MTP draft cache (`EXL3_MTP_KV_WINDOW=16384`), image `tabbyapi:mtpwin-r2` | new c1 fingerprint (a layer moves cards), 30k unchanged; decode +0.3 to +2.0 % on 24 paired prompts; restart restore 29,952 tokens in 0.567 s against 3.967 s cold; needles 5/5 and 5/5, tool-eval 85.8, GSM8K 0.976 at 5 concurrent | 999,424 | [R579](../bench/results/r579-promote-mtp-kv-window.md) |
| 11:04 | Prometheus `/metrics` endpoint, image `tabbyapi:mtpwin-r2-metrics1` | c1 and 30k greedy fingerprints identical to the reference; counter deltas verified against driven traffic | 999,424 | [R587](../bench/results/r587-tabby-metrics.md) |

The same overlay one pool step higher, 1,015,808, passed its A/B and four gates on 2026-09-19 and then ran out of memory capturing a decode graph at 5 concurrent, and rolled back ([R575](../bench/results/r575-promote-mtp-kv-window.md)). Every boot now runs a 1-to-8-stream decode ramp before the gates, so no graph is first captured under load.

## 2026-09-22: the batched draft verifier begins running

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 12:10 | image `tabbyapi:bverify-r1` = `mtpwin-r2-metrics1` + verifybatch-r1.patch: the round-4 batched MTP verifier was unreachable — `reqs_past_ids` was aggregated over pre-simplification sampler steps (the frontend appends neutral penalty steps to every stack), and `device_logit_mask` vetoed every `min_tokens` request | canonical gate vs `mtpwin-r2-metrics1`, same salt/shapes: c1 +3.4 %, c4/4k +0.8 %, c8 +3.9 %, c4/26k +3.5 % decode t/s; acceptance per verify unchanged; py-spy leaf `job.py:622` 43.3 % → ~0 | 999,424 | [R646](../bench/results/r646-verifybatch.md) |
| 20:00 | image `tabbyapi:stack-r1` = `bverify-r1` + mtpnorm (MTP input-norm chain → fused `rms_norm`) + mixstate (state row folded into the int8 up kernel, `EXL3_GR_STATE_IN_UP=1`) + prefbatch (accept-path draft prefills grouped by accepted length) | greedy byte-identical on all 6 prompts vs `bverify-r1`; canonical gate: c1 +1.2 %, c4/4k +3.8 %, c8 +4.4 %, c4/26k −0.1 %; acceptance parity; the c8 figure reproduces the prefbatch-only gate exactly | 999,424 | [R653](../bench/results/r653-stack.md) |

## 2026-09-23: the recurrent-state slot pool stops losing slots

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 22:25 | image `tabbyapi:slotfix-r1` = `stack-r1` + slotfix-r1.patch: the three state-allocation paths return the slot handle when the state constructor raises; `release_state` refuses a double release; `reap_failed_job` and `cancel()` release the draft window and the pages under separate handlers | 6 injected constructor faults → 6 slots returned, 0 `no available slots`; greedy byte-identical on all 6 prompts vs `stack-r1`; 18 min churn (12 workers, ~25 % mid-stream cancels): 1,497 requests, 0 errors, 0 restarts; c8 8 of 8 after | 999,424 | [R676](../bench/results/r676-slotfix.md) |

## 2026-09-24: the MTP draft component moves to the second GPU

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 08:34 | launcher `DRAFT_GPU_SPLIT="0, 32"` → `draft_gpu_split: [0, 32]`: the MTP draft component loads on the second GPU; image unchanged (`tabbyapi:slotfix-r1`) | 3 alternating pairs, canonical gate: prose leg A +2.9 / +2.2 / +2.0 % at 1 / 4 / 8 streams, leg B (26k, 4 streams) +2.2 %, means of per-request medians; greedy identical in all 6 boots; 0 OOM; GPU 0 free after the gate 125-139 MiB against 19-33 | 999,424 | [R694](../bench/results/r694-mtp-card1.md) |

## 2026-09-24: the first batch on the stack track

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 10:46 | image `tabbyapi:stack-r2` = `slotfix-r1` + hcfast-r1 (HC boundary mixer: each int8 weight converted once per iteration, batched loads, 35-shuffle reduce-scatter; `EXL3_HC_MIX_V3=1 EXL3_HC_MIX_V3_DOTS_B=2 EXL3_HC_MIX_V3_UP_B=8`) + moefast-r1 (routed-expert MoE decode: cp.async weight ring, activation prefetch, one counter arrival per item, merged narrow down stage; `EXL3_MOE_COOP_V3=2`); 27 environment keys | in-process harness, 5 rounds per shape, stack against OFF: −11.6 % ms per iterate at 1 stream depth 3, −6.5 % at 4 streams depth 3, −4.0 % at 8 streams depth 1, sequence hashes identical in all rounds; canonical gate, 3 alternating pairs, mean ON/OFF of per-request medians: prose leg A 1.093 / 1.057 / 1.051 at 1 / 4 / 8 streams, leg B (26k, 4 streams) 1.064; greedy identical in all 6 boots; 0 OOM; GPU 0 free after the gate 193-199 MiB against 121-131 | 999,424 | [R701](../bench/results/r701-stack-r2.md) |

Measured after the promotion the same day: the decode curve of the served configuration against the one served before R701, two alternating boots each, `fn_bench`, greedy, 1,024 forced tokens, 1 to 8 streams: the decode rate per stream after the first token is 1.05 to 1.15 times higher for code and prose, the end-to-end burst aggregate for prose at 8 streams is 714 against 676 tokens/s, and the time to the first token is unchanged ([R704](../bench/results/r704-decode-curve.md)). The README's decode figure drew the decode rate per stream and the decode aggregate from R704 from then until R719, in place of R580's round-wall aggregate.

Measured and not promoted the same day: hcfast r1 with the default 8/8 tile, a same-sign regression at 8 streams ([R698](../bench/results/r698-hcfast.md)), and mixed draft depth per job to fill 16 verify rows at 5 to 7 streams, 0.79× at 5 streams and 0.84× at 7 ([R678b](../bench/results/r678b-fill16.md)).

## 2026-09-24: the second batch on the stack track

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 22:10 | image `tabbyapi:stack-r3` = `stack-r2` + one patch series (INCLUDE `mf3 dg2`): hcfast r2 (`EXL3_HC_MIX_V3=2` with per-row-count tile tables, replacing hcfast r1's `DOTS_B=2 UP_B=8`), latchain (register-resident GDN recurrence `EXL3_LC_GDN_RR=1`, the QSA indexer as a parallel graph branch `EXL3_LC_QSA_FORK=1`, QSA split and combine compile options), densegemm r2 (`EXL3_DENSE_V2=1`, V2 twins of the dense K=4 decode kernels), moefast r3 (`EXL3_MOE_COOP_V3=3 EXL3_MOE_COOP_V3_MAP=2-4:2 EXL3_SHARED_EXPERT_EARLY=1`) and a guard that gives the QSA branch's dense call its own scratch; 39 environment keys | in-process harness, 7 rounds, union against OFF, stall-excluded: −13.1 % ms per iterate at 1 stream depth 3, −8.5 % at 4 streams depth 3, −8.2 % at 8 streams depth 1, 7 of 7 each, every leave-one-out marginal a gain at one shape or more; sequence hashes identical in all 42 cells of every arm; logits-level parity identical to `stack-r2` at every served decode shape (batches 1 to 8, [R716c](../bench/results/r716b-stack-r3.md)); canonical gate, 3 alternating pairs, mean ON/OFF of per-request medians: prose leg A 1.149 / 1.088 / 1.087 at 1 / 4 / 8 streams, leg B (26k, 4 streams) 1.104; greedy set identical to the first A boot in the other 7 boots; 0 OOM; free VRAM at the UP line 1,033 / 2,421 MiB against 1,041 / 2,431 | 999,424 | [R716b, R716c](../bench/results/r716b-stack-r3.md) |

The components were admitted one by one earlier the same day: hcfast r2 at 1 stream ([R702](../bench/results/r702-hcfast-r2.md)), the latchain recurrence ([R712](../bench/results/r712-latchain-r1.md)), densegemm r2 mode 1 ([R714](../bench/results/r714-densegemm-r2.md), superseding densegemm r1's gemv-only mode 3, [R710b](../bench/results/r710b-densegemm-gv.md)) and moefast r3 ([R713](../bench/results/r713-moefast-r3.md)); the QSA fork and compile options entered as candidates on R712's stall-excluded read and were admitted by the batch gate.

R716b's own unit printed `DECISION UNION: REJECT` on its multi-stream greedy rule alone, which compared each candidate boot against the largest of three A/A divergences. At 4 and 8 streams the served configuration differs from itself on 60 % and 76 % of streams, and the rule failed 64 % of role assignments on that round's boots. The rule was replaced the same day: strict where the A/A boots are identical, INCONCLUSIVE above 25 % A/A divergence with identity taken from in-process parity at every row count the draft policy reaches, a pooled permutation test in between ([`docs/PROMOTION.md`](PROMOTION.md#identity-at-every-served-shape-since-2026-09-24)). R716c ran those in-process cells before the promotion.

Measured and not promoted the same day: moefast r2 without the early shared-expert fork, flat against mode 2 at every stack shape (results `2026-09-24-r703b-moefast-r2`); an L2 prefetch of the next layer's weights, slower in 5 of 5 rounds in the depth-0 cells, and programmatic dependent launch, flat (results `2026-09-24-r705-pdl-gate`); a draft policy with depth 3 only up to 2 jobs, −4.1 to −6.9 % at 3 and 4 streams (results `2026-09-24-r711-draft-policy-p3`); the latchain GDN b/a graph branch, whose marginal inside the four-lever arm was about 0 ([R712](../bench/results/r712-latchain-r1.md)).

## 2026-09-25: draft depth 2 at 6 to 8 jobs

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 01:05 | image `tabbyapi:stack-r3-rows32` = `stack-r3` + rows32-r4.patch (host code only, SASS of every `stack-r3` function unchanged); `EXL3_DENSE_ROWS32=1 EXL3_MOE_COOP_ROWS32=1 EXL3_SHARED_EXPERT_ROWS32=1`, `EXL3_MOE_COOP_V3_MAP=2-4:2,17-32:2`; draft policy `[[4, 3], [8, 2]]`, so 6, 7 and 8 jobs verify at depth 2 (18, 21 and 24 rows) instead of depth 1; 42 environment keys | MoE calls at 17 to 32 rows equal to two served 16-row calls (1,344 of 1,344 per layer, 6 layers, 6 arms); P1 hashes with and without the flags identical in 36 of 36 cells at 16 rows and below, and the same over 4 rounds at 6, 7 and 8 streams depth 2; served A/B, 2 pairs, 110-token prompts, 1,024 forced tokens, greedy: per-stream decode 1.031 / 1.029 / 1.043 code and 1.052 / 1.053 / 1.047 prose at 6 / 7 / 8 streams, 0.995 to 1.003 at 1 to 5; context A/B, 3 pairs: 1.004 to 1.060 on 12k to 44k-token prompts, 1.160 and 1.208 at 6 streams on 3k to 6k-token prompts; greedy output at 8 streams differs from 1 stream on 57 of 64 prompts against the served configuration's own 59 of 64; greedy set identical, 6 of 6; 0 OOM; free VRAM at boot 1,125 / 2,513 MiB against 1,033 / 2,421 | 983,040 | [R717, R717b, R717c](../bench/results/r717-rows32.md) |

The unit R717 printed REJECT for four reasons: a string the decision script looked for in the wrong log, a harness out-of-memory at 7 streams depth 2 on `--gpu-split 30,30`, the flags' 48 MiB on GPU 0 against the 32 MiB headroom rule, and an early-divergence sub-rule of the concurrency greedy check that fires on about a fifth of identical-distribution draws. R717b settled the second at `--gpu-split 28,30` and measured the pool trade; the user accepted 983,040 tokens (−1.6 %) for the flags. The fn_gate cell at 8 streams read +0.9 % on 3,100-token prompts, and R717c measured the gain against context before the promotion. The early-divergence sub-rule is reported and not judged since the review.

Measured after the promotion: the decode curve of the served configuration, two boots, `fn_bench --distinct` (each stream on its own prompt), greedy, 1,024 forced tokens, 1 to 8 streams, 2026-09-25 00:27 to 00:44 UTC. The decode aggregate is 292 / 272 tokens/s at 1 stream and 822 / 831 at 8 streams (code / prose) and rises from 5 to 6 streams (671 to 708, 679 to 716), where R704's aggregate on `stack-r2` fell on code and was flat on prose. The per-stream rate is 0.96 to 1.23 times R704's across the 16 cells; the two rounds differ in image, draft policy, page pool and instrument, since R704 sent one prompt to every stream of a round, so the ratios are not a measured gain of the configuration ([R719](../bench/results/r719-decode-curve.md)). The README's decode figure drew R719 from then until R719b.

## 2026-09-25: the memory clock offset re-applied at boot, and the windowed draft cache off

At 04:22 CEST both cards read a memory clock offset of 0. The host's boot-time service had applied +4500 on 2026-09-02, the offset was recorded intact on 2026-09-03, and the logs of this model's runs from 2026-09-19 on show the stock memory clock under load; the host was not rebooted in between, and the cause is not determined. Every number published here from 2026-09-19 to that moment, R719 included, ran at the stock memory clock. Restoring +4500 raised DRAM bandwidth by 14.3 % (1 GiB device-to-device copy, 1,531.5 to 1,750.8 GB/s on cuda:0) and decode by 1.8 % (code) and 1.7 % (prose) per stream at 1 stream, greedy output identical; at 4 and 8 streams the gain is consistent with that and not resolved ([R726](../bench/results/r726-memoc.md)). Since then the launcher sets the offset before every boot and logs the readback (`MEMOC`).

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 05:52 | launcher: `EXL3_MTP_KV_WINDOW=16384` removed, 41 environment keys; the MTP draft K/V lives in the page-indexed draft cache over the whole pool again, on cuda:1; image unchanged (`tabbyapi:stack-r3-rows32`) | under the window a prompt revived from the prompt cache in a later request group drafted 0.655 accepted per proposed token against 0.868 without it, prompts never seen before 0.911 in both, code at 8 streams on 4k-token prompts 12.5 % faster per stream without the window (R721); agent-shaped replay, 2 pairs, temperature 0.6, 600 s per arm: +2.0 % per stream (R722); at 983,040 cuda:0 unchanged (1,125 MiB free at boot, 229 after the heavy sequence), cuda:1 −940 MiB (1,573 at boot, 763 after), greedy 6 of 6 identical (R723); promotion unit: greedy 6 of 6 identical, 0 out-of-memory under stress at ~19,600 tokens and 4 and 8 streams, needles 5 of 5 at 105,680 and 193,464 prompt tokens, tool-eval 87.8 ± 1.7, GSM8K 0.974 (R728) | 983,040 | [R721, R722, R723, R728](../bench/results/r728-promote-window-off.md) |

The promotion's decode gate (prose) read 1.217 at 8 streams on ~3,100-token prompts and 1.209 at 4 streams on ~19,600-token prompts; both cells revive prompts an earlier cell prefilled, so they bound the gain on revived prompts, and prompts never seen before read 1.01 to 1.02 in R721. The window had been served since [R579](../bench/results/r579-promote-mtp-kv-window.md), where it freed draft cache on cuda:0; since [R694](../bench/results/r694-mtp-card1.md) the draft component is on cuda:1, so turning it off leaves the page pool unchanged.

Measured after the promotion: the decode curve of the served configuration, two boots, `fn_bench --distinct`, greedy, 1,024 forced tokens, 1 to 8 streams, 2026-09-25 07:59 to 08:16 UTC, the R719 driver unchanged. The decode aggregate is 298 / 279 tokens/s at 1 stream and 845 / 853 at 8 streams (code / prose) and rises at every step; the per-stream rate is 1.007 to 1.032 times R719's in all 16 cells. The memory clock offset and the draft cache both changed between the two rounds, and the round does not separate them ([R719b](../bench/results/r719b-decode-curve.md)). The README's decode figure draws R719b since then.

## 2026-09-26: the requeue token-count fix restored

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 13:37 | image `tabbyapi:stack-r3-rows32-tokcount` = `stack-r3-rows32` + [`tokcount-r1/fix.patch`](../docker/overlays/tokcount-r1/fix.patch), one line of `generator/job.py`: `usage.completion_tokens` and the logged T/s count every requeued segment; launcher otherwise unchanged, 41 environment keys | R737: the served image reported 2,969 / 2,888 / 3,714 tokens for generations forced to 9,000 / 13,000 / 20,000, the patched image 9,000 / 13,000 / 20,000 in `usage` and in the log; greedy set identical, 6 of 6; promotion unit: 41 keys, free VRAM at boot 1,125 / 1,573 MiB (unchanged), greedy set identical to the served image's, a 9,000-token generation counted 9,000, 0 OOM, tracebacks or restarts | 983,040 | [R737, R747](../bench/results/r747-tokcount.md) |

The fix was first applied on 2026-09-16 as a `sed` in `docker/Dockerfile.tabbyapi`; every image from `qsa-cid-pr337` on was built from `Dockerfile.tabbyapi-qsa-cid`, which does not carry it, while the documentation said it did. [R583](../bench/results/r583-long-generation.md)'s 65.6 tokens/s per stream for a real agent session came from the server's log in that window and is withdrawn.

## 2026-09-27: a loop in the thinking ends the thinking, not the request

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 11:49 | image `tabbyapi:stack-r3-rows32-tokcount-loopthink3` = `stack-r3-rows32-tokcount` + [`loop-think-r3/fix.patch`](../docker/overlays/loop-think-r3/fix.patch), TabbyAPI `endpoints/OAI/utils/chat_completion.py` and `common/sampling.py`: on a chat request that starts in the thinking, a repeated period of up to 400 tokens in the reasoning forces a one-line message and `</think>` instead of ending the request with reasoning and no content; the engine's loop detector on those requests is `(1600, 4)` instead of `(800, 2)`; launcher otherwise unchanged, 41 environment keys | promotion unit, 2026-09-27 09:43 to 09:49 UTC: 41 keys, free VRAM at boot 1,125 / 1,573 MiB (unchanged); `fn_greedy` 6 of 6 identical to R747's served reference (`/v1/completions`); chat greedy 5 of 5 identical between the old and the new image in the same unit (`/v1/chat/completions`, temperature 0, thinking on: code, arithmetic, prose, a tool call, two turns); a loop prefilled in the thinking 2 of 2 answered "391" at 826 tokens; a thinking-off content loop 2 of 2 stopped at 800 tokens; an agent turn prefilled with a real looping paragraph 2 of 2 ended in tool calls, one after the injection and one that left the loop by itself; 3 injections, 2 engine loop stops; 0 OOM, tracebacks or restarts | 983,040 | [R782, R783](../bench/results/r783-loopthink.md) |

The patch went through three rounds, checked on a Qwen3.8-Flash-Next finetune served with the same chat endpoint. r1 moved the engine's detector to `(1600, 2)`, which doubled the longest period it ends to 800 tokens: reasoning loops with a period of 401 to 800 tokens, which the served detector never caught, were now ended as reasoning-only stops, the bug itself; r1 also armed on thinking-off requests. r2 (R782) keeps 4 repetitions at 1,600 tokens, so the engine ends the same periods as before, and arms only when the request starts in the thinking. r3 does not arm when the 2W window exceeds the 2,048-token output chunk, because the engine rebuilds its detector at every requeue and a window above 2,048 tokens never fills; at the default W = 800 it behaves as r2. The patch covers loop periods of up to 400 tokens; a longer period runs until the model leaves it or reaches `max_tokens`, as before.

## 2026-09-27: the engine rebased onto upstream ExLlamaV3 `dev` 5783a93 (v1.5.2)

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 15:01 (launcher live from 14:23, gates on the live port) | image `tabbyapi:rebase-dev-r3` = the served engine stack, the 41 keys' mechanisms, ported onto upstream ExLlamaV3 `dev` `5783a93` (v1.5.2) with the requeue token-count fix, replacing the whole `exllamav3` package of `tabbyapi:stack-r3-rows32-tokcount-loopthink4` ([`rebase-dev-r3`](../docker/overlays/rebase-dev-r3/)), whose TabbyAPI is the served one plus [`loop-think-r4`](../docker/overlays/loop-think-r4/) (a second, collector-only loop detector for reasoning periods of up to 1,000 tokens); 42 environment keys (`EXL3_GR_MIX_TILED=1` added, upstream's tiled HC prefill mix); pool 983,040 → 901,120; own kernel-cache directory; greedy output changes by design (upstream's GDN fp16 prefill projections, deterministic router GEMM, tiled HC prefill, PLE in-place add) | R784, ABBA against the served image: largest pool with per-card boot free VRAM within 32 MiB of the served image's, 901,120 (1,181 / 1,759 MiB free against 1,125 / 1,573); cold prefill 1.134× at 90k tokens (10,756 → 12,198 t/s, medians of 6 samples), 1.08× and 1.19× by boot pair at 22.6k; short-prompt decode, 24 code and 24 prose prompts, 512 forced tokens, greedy: −1.28 / +0.76 % at 1 stream, +1.58 / −0.77 % at 4, −0.07 / −0.44 % at 8 (code / prose), not a candidate on the pre-registered rule (prose at 8 streams, interval lower bound −2.30 % against −2 %), decode accepted as flat by the operator; `fn_greedy` 4 of 6 and chat greedy 1 of 5 identical to the served image. R785, twice (R785b rolled back on a harness miscount, R785c promoted): 42 keys, boot free 1,181 / 1,759 MiB; ramp 36/36, stress at 4 and 8 streams all ok, 0 OOM; loop-think forced 2/2, thinking-off 2/2, agent turn 2/2; needles 5/5 at 131k and 240k; agent replay 123.0 / 123.7 t/s per stream (bar 120.3); agentic edit 24/24; tool-eval 85.5 / 86.5; GSM8K 0.982 / 0.980 at n = 500; free after the whole run 217 / 749 MiB | 901,120 | [R784](../bench/results/r784-rebase-dev-r3.md), [R785](../bench/results/r785-promote-rebase-r3.md) |

The image is no longer a chain of patches on ExLlamaV3 v1.5.0: the package in site-packages is upstream `5783a93` plus one patch that carries every served mechanism ([`ported-vs-dev.patch`](../docker/overlays/rebase-dev-r3/ported-vs-dev.patch)); TabbyAPI stays at `53da7919` with this repository's patches. [R563](../bench/results/r563-rebase-dev.md) measured an earlier port on 2026-09-19 at −49,152 pool tokens and −1 to −3 % decode, and [R568](../bench/results/r568-rebase-prefill.md) traced the port's prefill gain to the tiled HC prefill mix.

R785's agent replay read 123.0 and 123.7 t/s per stream against 128.6 for an older image in [R728](../bench/results/r728-promote-window-off.md) (2026-09-25). [R786](../bench/results/r786-replay-abba.md) ran the replay in one session on three ABBA pairs of the previous and the new launcher (13:24 to 15:05 UTC, tier off): 1.015× per stream for the new image on the requests common to all arms, 95 % interval 0.980 to 1.053, and one pair resolves about ±5 to 7 %. The R785 against R728 gap is MTP acceptance under sampling (−2.3 %) and higher concurrency (−1.3 %) on an older baseline. The replay does not exercise the smaller page pool: its live KV peaks at 430,000 to 450,000 tokens, inside both pools. The README's decode, prefill, depth and standard-benchmark figures were re-measured on the new image on 2026-09-27 ([R787](../bench/results/r787-bench-refresh.md)).

Rollback: `DAILY_IMG=tabbyapi:stack-r3-rows32-tokcount-loopthink3 CACHE=983040 TUNEDIR=/srv/qwen5090/.exl3cache` and `EXTRA_ENV` without `EXL3_GR_MIX_TILED=1`, the R783 launcher.

## 2026-09-27: sampler fallback temperature 1.0

| changed (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 22:18 | launcher: the `qwen38_thinking` preset's fallback temperature 0.6 → 1.0, the Qwen3.8-Flash-Next model card's thinking-mode value; top_k 20 and top_p 0.95 unchanged, every entry `force: false`, so only requests that omit the temperature change; image and engine environment unchanged | none (sampler fallback only; greedy and explicitly sampled requests are unaffected) | 901,120 | — |

Rollback: `override: 0.6` in the `temperature` entry of the launcher's sampler heredoc.

## 2026-09-28: three unused engine keys removed

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 02:52 | launcher: `EXL3_HOST_GAP_REWIND=1`, `EXL3_GR_STATE_REGRID=1` and `EXL3_HC_MIX_V3_PDL=0` removed from `EXTRA_ENV`, 42 → 39 keys; image unchanged (`tabbyapi:rebase-dev-r3`) | G1-G10 of R785 with greedy byte-identical to R785c (fn_greedy 6/6, chat_greedy 6/6 incl. the long answer); needles 5/5 at 131k and 240k; tool-eval 85.2; GSM8K c8 0.976 | 901,120 | [R789](../bench/results/r789-promote-dropkeys.md), `2026-09-28-r789-promote-dropkeys-0013` |

The first run (00:04 UTC) was rolled back at the loop-think gate: its probe sent no temperature, so it ran at the sampler fallback, 1.0 since the previous evening, while the reference ran at 0.6. The second run pins the probe at 0.6.

Rollback: add the three keys back to `EXTRA_ENV`.

## 2026-09-28: loop-think r5, reasoning loops with periods of up to 4,000 tokens

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 09:49 | image `tabbyapi:rebase-dev-r3-loopthink5` = `rebase-dev-r3` + [`loop-think-r5/r4-to-r5.patch`](../docker/overlays/loop-think-r5/r4-to-r5.patch), TabbyAPI `endpoints/OAI/utils/chat_completion.py`: two more collector-only reasoning-loop rungs, (6000, 2000) and (12000, 4000), beside r4's (3000, 1000), for loop periods of 1,000 to 4,000 tokens; `finish_reason: "length"` instead of `tool_calls` when `max_tokens` cuts tool-call text that does not parse; engine, 39 keys and launcher otherwise unchanged | R792, 07:09 to 07:50 UTC, R789's G1 to G10: boot free 1,181 / 1,759 MiB; greedy byte-identical to R785c (`fn_greedy` 6/6, `chat_greedy` 6/6 incl. the long answer); the overlay's 89 offline checks inside the served container, 0 failures; loop-think forced 2/2, thinking-off 2/2, agent turn 2/2 at temperature 0.6; needles 5/5 at 131k and 240k; agent replay 133.0 t/s per stream (bar 120.3); agentic edit 24/24; tool-eval 84.2; GSM8K c8 0.976; no r5 branch ran during G7 to G10. Report-only: the (6000, 2000) rung fired on a prefilled 1,199-token loop at 6,003 generated tokens (0.6); a tool call cut at 300 tokens returned `length` with 0 calls, streamed and not | 901,120 | [R792](../bench/results/r792-promote-loopthink5.md) |

[R791](../bench/results/r791-temp-incidence.md) found two exact loops in the reasoning with periods of about 1,200 and 3,700 tokens on ordinary agent turns at temperature 0.6, above r4's 1,000-token reach; both ran to `max_tokens` with no answer. The same round found the `tool_calls` label on a cut tool call. At the served fallback temperature 1.0 the prefilled loop of R792's probe did not continue (0 of 8), so the ladder has not fired on the served sampler.

Rollback: `DAILY_IMG=tabbyapi:rebase-dev-r3`, the R789 launcher.

## 2026-09-29: tokenize-offloop r2, one encode per request and long prompts encoded off the event loop

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 19:01 | image `tabbyapi:tokenize-offloop-r2` = `rebase-dev-r3-loopthink5` + [`tokenize-offloop-r2`](../docker/overlays/tokenize-offloop-r2/README.md): TabbyAPI reuses the context-length check's ids in `generate_gen`, so each prompt is encoded once per request, and encodes a prompt longer than 12,000 characters on a one-thread worker; ExLlamaV3's `encode_part_base` encodes through `encode_batch`, which releases the GIL, under a per-tokenizer lock; knobs `TABBY_ENCODE_ONCE`, `EXL3_TOKENIZE_OFFLOOP` (both on by default) and `EXL3_TOKENIZE_OFFLOOP_MIN_CHARS` (12000) not set by the launcher; engine, 39 keys, pool, split and kernel cache unchanged | R808, 14:11 to 16:46 UTC, 5 alternating boot pairs of this image with the knobs on and off (one prompt nonce, greedy peers): a 30,818-token prompt arriving during 4-stream decode stalls the other streams for a median 2.8 ms against 53.1 ms (80 arrivals per arm); time to the first token of a 98-token prompt during 4-stream decode −2.9 ms [−6.2, +0.3] per pair; steady-state time per decode step +0.01 % [−0.25, +0.27] (no cost); greedy output identical on and off (135 of 135 rows, streamed and not) and to R792's reference (12 of 12); `/v1/token/encode` ids equal the served tokenizer file's; boot free VRAM 1,181 / 1,759 MiB on every boot; 0 failed requests. R808p after the boot: image id, 39 keys, pool 901,120, the knob readback, `fn_greedy` 6 of 6 identical to R792 | 901,120 | [R805, R806, R808](../bench/results/r805-r808-tokenize-offloop.md), `2026-09-29-r808-tokenize-offloop-r2-1411`, `2026-09-29-r808p-promote-tokoffloop-1700` |

[R805](../bench/results/r805-r808-tokenize-offloop.md) (2026-09-29) traced the stall to two synchronous encodes of the same prompt on the event loop, about 25 ms each. Round r1 ([R806](../bench/results/r805-r808-tokenize-offloop.md)) sent every encode to the worker: the stall went from 53.5 to 3.5 ms, but a 98-token prompt's first token came 41.6 ms later, because each executor hop waits 1 to 2 decode steps for the loop to pick up its result. r2 hops only above 12,000 characters. R808's decode aggregate reads +0.75 % [+0.21, +1.28]: the removed stall at R808's synthetic arrival rate (the mean gap between decode steps including the stalls, −0.54 % [−1.01, −0.08]) plus MTP acceptance moving with batch timing (tokens per step +0.21 % [−0.07, +0.48]), not faster decode steps. The arriving long prompt's own time to the first token did not change (+3.0 ms per pair, not significant); the gain goes to the other streams.

Rollback: `DAILY_IMG=tabbyapi:rebase-dev-r3-loopthink5`, the R792 launcher.

## 2026-09-29: prefill-merge r1 and the asynchronous recurrent stash

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 22:02 | image `tabbyapi:merge-tok-r1` = `tokenize-offloop-r2` + [`prefill-merge-r1`](../docker/overlays/prefill-merge-r1/README.md) (ExLlamaV3, Python only) and two keys, 39 → 41: `EXL3_PREFILL_MERGE=1` runs the rows after the last full 256-token page in the same forward as the rest of the prompt's remainder when it fits one 2,048-row forward, with every GDN and PLE layer split at that page inside the forward and the last-page stash captured there (forwards per prefill = ceil(N_p / 2048)); `EXL3_STASH_ASYNC=1` copies each recurrent stash to the host through pinned staging slabs, CUDA events and a worker thread; engine kernels, pool, split and kernel cache unchanged | R803, 12:18 to 15:26 UTC, OFF / MERGE / MERGE+ASYNC / OFF2 in two blocks, fresh boot each, 600 s agent replay at 8 slots: agent-turn prefill −13.3 % (MERGE) and −16.6 % (MERGE+ASYNC), client time to the first token −10 %, per-stream decode +3.1 % on the acceptance-free step rate [+1.3, +4.9] (4 merge-class against 4 OFF boots); ASYNC about 10 ms per stash, bitwise against the synchronous copy. R809 on this image (17:03 UTC on): against a partition null (the same prompts behind a prefix 512 tokens shorter), 34 against 29 of 80 prompts diverging within 64 greedy tokens (McNemar p 0.151), 0 confident flips in either arm, first-token KL median 0.0031 against 0.0037; tool-eval 85.0; GSM8K 0.978; boot free VRAM 1,181 / 1,759 MiB; R809t: a tier checkpoint restored from disk after a crash reproduces the warm output over 64 tokens, first-token KL 0.0. R809p after the boot: image id, 41 keys, pool 901,120, both knob readbacks, `fn_greedy` 6 of 6 identical to R792. R810: needles 5 of 5 at 105,680 and 193,464 prompt tokens; the new greedy reference identical on a second boot | 901,120 | [R803, R809, R810](../bench/results/r803-r810-prefill-merge.md), `2026-09-29-r803-prefill-merge-ab-1218`, `2026-09-29-r809-merge-quality-gate-1703`, `2026-09-29-r809t-tier-recheck-1759`, `2026-09-29-r809p-promote-merge-2000`, `2026-09-29-r810-post-checks-2004` |

[R802 and R804](../bench/results/r803-r810-prefill-merge.md) (2026-09-29) found the fixed costs of an agent turn's prefill: a separate forward for the sub-page leftover (up to about 120 ms at 246 rows and more) and two synchronous copies of the recurrent state to the host (8 to 10 ms each). The merge changes the numerics of every prefill it applies to, because the layers above the recurrent ones see a different row count; R803's GPU gate measured the stash difference at the size another served partition of the same prompt produces ([GOTCHAS 37](GOTCHAS.md)), and R809 gated the output against that partition null. The greedy reference rolled over: chat `tool` and `long` changed, the other 10 prompts did not. R803's decode figure was measured on `tabbyapi:prefill-merge-r1` without `tokenize-offloop-r2`; R809's first tier check compared one-token outputs and was repeated as R809t ([GOTCHAS 35](GOTCHAS.md)). GSM8K doc 243 answered `#### 25`; 9 of 23 earlier runs on this checkpoint gave the same answer (R810b).

The served image was measured with the README's instruments on 2026-09-29 and 2026-09-30 ([R811, R811b, R812b, R813](../bench/results/r811-r813-std-bench-merge.md)). `vllm bench serve` against the previous image, alternating boots in one session: output tok/s 1.005 / 0.993 / 1.052 / 1.086× on ShareGPT and 0.987 / 1.030 / 1.042 / 1.067× on Spec-Bench at 1 / 2 / 4 / 8 streams, mean TTFT 7.4 to 15.4 % lower in every cell and 18 to 28 % lower on Spec-Bench's summarization and RAG prompts; at 4 and 8 streams the gain comes from less decode time in the steps that run other requests' prefill, with the median decode step within 2.3 % of the previous image's. At 1 and 2 streams the median decode step is 0.8 to 3.8 % longer than the previous image's. Split by send order, this is a speed-up that processes without `EXL3_PREFILL_MERGE` acquire at one request (ShareGPT request 45, a 786-token prompt; Spec-Bench request 29, 788 tokens) and keep to the end of the run, and that merge-on processes never acquire; before that request the arms are within about 2 %. R812b attributes the difference to the merge key, not to the asynchronous stash or the image's other layers; its mechanism is not known ([GOTCHAS 39](GOTCHAS.md)). The decode curve on the served image (R813, prompts of 106 and 118 tokens, which never reach that request's kind) reads 0.979 to 0.999 times R787a's time per decode step on the base image three days earlier, and replaces R787a in the README.

Rollback: `DAILY_IMG=tabbyapi:tokenize-offloop-r2` and `EXTRA_ENV` without the two keys, the R808 launcher.

## 2026-09-30: a boot warm-up for the faster decode state

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 13:52 | launcher only: after the `Warmup.` request, three raw `/v1/completions` requests of 20, 29 and 11 prompt tokens (synthetic word lists, temperature 0, `max_tokens` 8), checked against the server's own prompt counts, logged as `FASTWARM ok` or `FASTWARM FAILED`, never blocking serving; `FASTWARM=0` skips it; image, 41 keys, pool, split and kernel cache unchanged | R818, 06:18 to 07:44 UTC, 12 fresh boots alternating with the R809 launcher: `FASTWARM ok` 6 of 6; `fn_bench` time per decode step ×0.968 / ×0.966 at 1 stream (code / prose), ×0.977 / ×0.972 at 2, ×1.000 / ×1.008 at 4, ×1.001 / ×1.004 at 8; ShareGPT output tok/s 1.026× at 1 stream and 1.048× at 2; greedy output identical to R809's reference, ShareGPT 1-stream outputs 400 of 400 identical with and without the warm-up; boot free VRAM 1,099 / 1,667 MiB against 1,181 / 1,759, within 8 MiB after traffic, 0 out-of-memory lines. R818b/c, 07:56 to 08:21 UTC: ShareGPT at 8 streams over five boots per arm 0.9906× (cost 1.0095 against a 1.01 bar), boots in two modes about 3.5 % apart in both arms. R818p after the boot: `FASTWARM ok`, `fn_bench` 9.797 ms per step at 1 stream, free VRAM 1,099 / 1,667 MiB, `fn_greedy` 6 of 6 and `chat_greedy` 6 of 6 identical to R809 | 901,120 | [R814 to R820, R818p](../bench/results/r815-r820-fast-state.md), `2026-09-30-r818-warmfast-gate`, `2026-09-30b-r818-warmfast-gate`, `2026-09-30c-r818-warmfast-gate`, `2026-09-30-r818p-promote-warmfast-1152` |

R815 to R820 (2026-09-30, [write-up](../bench/results/r815-r820-fast-state.md)) traced the 1- and 2-stream difference of [GOTCHAS 39](GOTCHAS.md) to a state of the process. Every served process starts at 10.10 to 10.15 ms per decode step at 1 stream and moves to 9.80 to 9.88 ms, where it stays, once the routed-MoE decode path for up to 32 rows (`run_bszN`) has run at two row counts the process has not run before; those first calls launch the shared expert eagerly on its side stream. The prefill merge removed the short prefill forwards that used to set the state; real traffic sets it at the first episode of 4 or more concurrent streams. Clocks, host memory allocation, TabbyAPI, the hyper-connection mixer's row classes, `CUDA_DEVICE_MAX_CONNECTIONS` 1 and 32 and the MoE scratch geometry were excluded; no state that ExLlamaV3's code keeps explains it, and it was not traced below the CUDA API ([GOTCHAS 40](GOTCHAS.md)). The 1- and 2-stream points of the README's decode curve (R813) and standard benchmark (R811) were measured in the slower state. The boot free-VRAM figure is read after the warm-up. A per-boot two-mode effect at 8 streams (about 510 against 529 ShareGPT output tok/s) is separate from this state and is open.

Rollback: `FASTWARM=0`, or the R809 launcher ([`launch-flashnext-r809-merge.sh`](../scripts/launchers/launch-flashnext-r809-merge.sh)).

## 2026-10-01: tail recurrent checkpoints captured inside the prefill pipeline

| promoted (CEST) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 13:40 | `tabbyapi:merge-tok-r1` → `tabbyapi:r823c-cachetail-inforward`; checkpoint interval 4,096 in the last 12,288 prompt rows, in-forward capture, trace on, NVMe default off; source launcher `f3975d68` → `8b644c17` | R823c raw T32 edited-turn TTFT 5.681/5.683 → 2.078/2.087 s at 1 stream; C2 11.242/11.227 → 4.335/4.339 s at 2 streams, greedy 64 forced tokens per request. Cold T32 seed 67,618 prompt tokens: engine 5.480/5.475 → 5.486/5.514 s (12,339/12,351 → 12,325/12,263 processed rows/s); concurrent cold seeds 7.063/7.056 → 7.092/7.098 s per-job residency. CUT/INFORWARD/NO_SLAB checkpoint tensors bitwise equal; single-stream outputs equal, concurrent fidelity unresolved. R823p: FASTWARM passed, free VRAM 1,099/1,667 MiB, code c1 greedy 1,024 forced tokens 9.779 ms/step, greedy and chat each 6/6 equal to R809, needles 10/10, T32 edited cache 57,344 then HIT 67,584; first try rolled back on the needle checker | 901,120 | [R823–R823p](../bench/results/r823-tail-checkpoints.md), `2026-10-01-r823c-cache-reuse-DUO8ur`, promotion record `2026-10-01-r823p-promote-tailckpt-wCuc8r` |

Rollback: [`launch-flashnext-r818-warmfast.sh`](../scripts/launchers/launch-flashnext-r818-warmfast.sh). The separate production promotion leaves R823c's registered `RECOVERY-SUPPORTED / FIDELITY-UNRESOLVED` verdict unchanged.

## 2026-10-01: resumable whole-prompt prefill windows

| promoted (UTC) | change | gate evidence | page pool | results |
| --- | --- | --- | --- | --- |
| 15:37 | `tabbyapi:r823c-cachetail-inforward` → `tabbyapi:r825c-hostprepare`; `EXL3_PREFILL_WHOLE_PROMPT=1`, `EXL3_PREFILL_RESUMABLE=1`, 44 → 46 launcher selectors; literal image-ID guard; CPU preparation before trace attachment, GPU admission deferred; source launcher `8b644c17` → `262e9c31`; trace on, NVMe off, FASTWARM retained | R825c cold solo 20,000 / 50,000 / 90,000 prompt tokens, greedy one output token for timing, six ABBA reps per arm: engine 1,867.914 / 4,205.384 / 7,435.420 → 1,598.833 / 3,845.714 / 6,863.288 ms (processed rows/s 10,707 / 11,889 / 12,104 → 12,508 / 13,001 / 13,113); 27/27 checkpoint/logit/output equality rows. R825p cold50 at 50,001 tokens 4.06 → 3.66 s; health during cold 90,006-token prefill, 200 ms polls with 3 s timeout: 21/35 → 0/33 timeouts, longest 3.018 → 0.334 s; 203-token arrival after 1 s TTFT 6.267 → 1.922 s; decode-first code stream, 8,192 forced greedy output tokens, longest SSE gap 0.467 → 0.488 s. Boot free VRAM 1,099/1,667 MiB, code c1 1,024 forced greedy output tokens 9.766 ms/step, greedy + chat each 6/6 equal to R809, needles 10/10, policy-exact T32 smoke | 901,120 | [R824–R825p](../bench/results/r825-whole-prompt-window.md), `2026-10-01-r825c-resumable-Rh4RGH`, `2026-10-01-r825p-promote-wholeprompt-yj7qj3` |

Rollback: [the R823p launcher](../scripts/launchers/launch-flashnext-r823p-tailckpt.sh). R825's synchronous window could not serve; R825b added resumability, and R825c repaired the observed late-arrival trace abort before the final promotion.
