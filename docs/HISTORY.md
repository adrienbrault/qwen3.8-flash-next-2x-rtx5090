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

