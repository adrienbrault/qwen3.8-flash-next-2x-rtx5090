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
