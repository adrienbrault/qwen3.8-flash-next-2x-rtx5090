# Flash-Next rebase MTP path regression

Measured2026-10-21. Results directories `2026-10-21-170147-flashnext-repair-baseline` (validated API controls) and `2026-10-21-171122-flashnext-repair-profile` (four direct-generator profiles). [Raw control records and kernel tables](2026-10-21-flashnext-rebase-repair/). This compares the original served stack and the native1.6.4 rebase, both loading the original2.50bpw checkpoint, manual split30,30, KV8,8. The original daily remains served between experiments. No candidate promotion occurred.

| Stack | Code c1 per-stream tok/s | Code c4 per-stream tok/s | Six-agent replay seconds |
|---|---:|---:|---:|
| Old, MTP off |147.715|112.5725|not run|
| Rebased, MTP off |121.21|78.0775|not run|
| Old, MTP3 |256.85|156.2425|42.296|
| Rebased, MTP3 |179.435|66.5225|89.403|

API controls use118prompttokens, forced1024outputs, greedy sampling, one warm run and two measured runs at each concurrency. Each table entry is the median of the two within-run per-stream medians. The replay is six sessions,20requests, three tool turns,30K base prompt,512generatedtokens,1200tooltokens and seed20261021. All requests and planned step counts are validated. The original MTP-off control disables prompt lookup, which requires MTP, and retains the other46original selectors. The API KV pool is901120tokens. Original image `sha256:dfaed2cba56f353a99589549b6ccb971fe137f5e7d11cbeed522f207fb8b8724`; native baseline `sha256:4d5132f605016d227fbc934572bb13864bbcfd964870acb300ce3d8345571167`.

The rebased c4 control is31% slower with MTP off. MTP raises old-stack throughput and lowers native c4 throughput. Native source caps fused MoE execution at eight token rows; four jobs with three draft tokens produce16verificationrows and use the general MoE path. In the native MTP profile those general MoE kernels account for16.107summedCUDAms/step. The old profile uses its local MoE kernels. This identifies a dispatch difference; it does not assign the complete API replay slowdown to one function.

Direct profiles use batch4,128prompttokens,64minimumstreamedtokens/job,16warmupsteps,4settlingsteps, KV655360, split30,30 for both target and draft and fixed MTPdepth0/3. Draft placement differs from the API launcher's0,32; profile timings are diagnostic captures, not API throughput measurements. Depth0 captures take9.513 versus13.063ms/step; mixer kernels sum0.889 versus2.064CUDAms/step, and MoE coop plus mgemm sum4.619 versus6.465CUDAms/step. Summed kernel durations are not additive wall-clock attribution. GPU ports and same-image ON/OFF comparisons are still running.

The pruned-head cross-width comparison failed its proposed atol0.04/rtol0.002 bound. A subsequent [same-prefix native API contact](2026-10-21-flashnext-rebase-repair/head-contact.json) passes with exact equality at1–8rows. It compares raw prefix GEMM against a separately constructed native LinearEXL3 at the same width; it does not overturn the full-forward prefix difference. All recorded prefix argmax comparisons match for these contact states. The full-vocabulary winner was not compared. Arithmetic/tools and forced128-token concurrency1/4/8 contact pass; this contact ran during compilation and supplies no throughput claim.

A [source-built native receipt](2026-10-21-flashnext-rebase-repair/native-provenance.json) pins the experimental32-row capacity and carried kernels. This is separate from the initial official-wheel candidate; it compiles the new source plus both manifest patches. Real-weight kernel checks and the recovery performance screen are pending. The original daily remains selected.

The first source-native contact failed closed on an unchanged256-slot frontend validation bound despite the rebuilt512-slot run table. The corrected native patch binds that check to the table capacity; the [updated source receipt](2026-10-21-flashnext-rebase-repair/native-provenance-fixed.json) records the rebuilt extension. The earlier receipt is retained as failed-contact provenance, not a runnable Flash-Next result. The corrected real-weight contact is pending.

## Source-native ablation, 2026-10-21

Same original checkpoint and full 901120-token KV pool. Source-built 1.6.4 image `sha256:ec2ac7be9ebf809bf8cb56a9d2f1e41f548a8c3c132ca46c94c97a804aed2c2a`. Four successive boots; no CPU compilation during these measurements. Two forced 1024-token decode runs after warmup; six-agent replay has 20 complete requests. Results: [raw records](2026-10-21-flashnext-rebase-repair/native-screen-summary.json).

| Candidate additions | c1 tok/s per stream | c4 | c8 | Replay seconds |
|---|---:|---:|---:|---:|
|32-row native capacity|179.62|96.57|66.94|82.076|
|+ carried integer MoE kernels|191.28|113.12|82.32|78.533|
|+ carried int8 mixer|204.00|128.96|88.19|76.965|
|+ pruned draft head|211.24|130.39|87.22|76.570|

This recovers much of the decode regression, but remains below the old stack’s 156.24 tok/s c4 and 42.30-second replay control. It does not establish full target-logit parity or qualify a promotion. The original image remains served.

## Native prefill scheduling, 2026-10-21

Same source-native image `sha256:144ae9a34e458388d32cfba2a7e1444878225415fbe48c5ae0cf066adddd6683`, original checkpoint, full901120pool and all recovered kernel/head flags. Six sessions,20 forced-length complete requests per arm, seed20261021. [Raw summary](2026-10-21-flashnext-rebase-repair/prefill-policy-summary.json).

| Native scheduler | Replay seconds | Median TTFT seconds |
|---|---:|---:|
| Defaults: shared chunk1024, cooldown4 |76.838|2.587|
| Shared chunk2048, cooldown0 |65.626|0.774|

The opt-in policy reduces replay time14.6% in this screen; the old stack’s42.296-second control remains faster. Larger sequential prefill bursts can increase individual decode gaps. This is not the old two-device prefill pipeline. No serving change.

## Remaining prefill costs and E3 recovery, 2026-10-21

Direct-generator profiles used original weights, cache655360, target split30,30, draft split0,32 and MTP3. Cold32768-token prefill took2705ms on the old runtime and5600ms on the repaired runtime before E3/pipeline recovery (two unprofiled repetitions). Summed CUDA kernel times were4285ms and5352ms in the captures; the old two-device pipeline overlaps work, so summed kernel time is not elapsed time. Native weight-table quantization accounts for only35.7ms of the latter capture. The first capture attempt failed because the probe imported an old optional pipeline module; its timed prefills survived but its capture was rerun with module discovery fixed.

The persistent native table cache passed56 internal kernel/output checks, including exact native512/2048-row outputs, but could not place the model with the full901120-token pool. That arm was stopped and the original daily restored. Table caching remains off; no reduced-pool speed comparison is substituted.

The E3 port builds the three original deterministic E3 native files against1.6.4 and preserves native fallback for fractional K or unsupported shapes. On layers3/20 and rows512/2048, the probe observed one E3 dispatch per case; routing IDs, routing weights and full real-layer output tensors matched the old runtime exactly. [Layer receipts](2026-10-21-flashnext-rebase-repair/e3-real-layer-parity.json). This is scoped layer parity, not full-model target-logit fidelity.

With the original checkpoint and full901120pool, image `sha256:94d8fbce504a8e74892fd64b4f3519240141f14b3243e32d827cb80813aaf47c` completed the same20-request/six-session replay in57.964seconds, versus64.101seconds immediately before E3. All planned outputs and steps completed. [Replay summary](2026-10-21-flashnext-rebase-repair/e3-summary.json). The carried mixer’s row-dependent tile settings were also corrected before these two arms; all160 values match the original five launch tables for rows1–32. The original runtime remains faster; no promotion occurred.

## Bounded native lookahead, 2026-10-21

Image `sha256:fae5d087059c4fbfa72c67bba16bfd88d67403d037ab3fe829610b4a7b32b774` passed the [same-native differential](2026-10-21-flashnext-rebase-repair/pipeline-parity.json) at4097/12289-token prompts with checkpoint intervals4096/3072. Actual pipeline dispatch was asserted. Both executions matched full target exports, MTP carries, paired active recurrent stashes, logical valid target/draft KV bytes, page sequences/hashes, native checkpoint keys/bytes and next32greedy tokens. The safety probe used655360cache; the subsequent API replay used the original full901120pool.

The20-request/six-session replay completed in52.128seconds, median TTFT0.761seconds, with all planned outputs/steps validated. [Raw summary](2026-10-21-flashnext-rebase-repair/pipeline-summary.json). The preceding E3-only replay was57.964seconds. The original daily was restored after the chain. The new pipeline uses native checkpoint predicates rather than the old modulo rule; the two disagree for valid irregular interval anchors.

A final experimental Python layer restores per-job stateless MTP verification batching. Native first-token logit checks, forced tokens, stateful samplers/filters, banned-string checkpoints and minimum-length mask transitions retain serial handling. Greedy windows preserve RNG consumption; sampled windows use one Philox seed with row subsequences, preserving row distributions while changing seeded trajectories as the old daily did. Linux eligibility tests and the four-job, 128-token greedy sequence/RNG/acceptance differential pass. The timing screen below rejects this flag for the replay candidate. This layer does not change the native extension.


## Rejected verifier and shared-expert settings

Same-image full-pool screen with the recovered native prefill pipeline: batch verification OFF gives c4 132.0275 tokens/s per stream and replay52.423951seconds; ON gives137.095 and53.294185seconds. [Validated records](2026-10-21-flashnext-rebase-repair/batch-screen-summary.json), [greedy differential](2026-10-21-flashnext-rebase-repair/batch-parity.json). Batch verification remains OFF because the agent replay did not improve.

The existing upstream separate shared-expert path (`EXL3_MOE_SHARED_COOP=0`) gives c4 136.305tokens/s and replay53.670785seconds, with all20requests completed. [Records](2026-10-21-flashnext-rebase-repair/shared-screen-summary.json). It also remains out of the candidate. These are exploratory screens, not final alternating comparisons.

Device-resident drafting also passed exact four-job128-token greedy sequences, RNG and acceptance counters, plus exact prefix/outside-prefix embedding rows. [Differential](2026-10-21-flashnext-rebase-repair/device-parity.json). At full901120pool it gives c4 135.33tokens/s and replay52.109340seconds. [Records](2026-10-21-flashnext-rebase-repair/device-summary.json). This does not establish a useful replay gain over52.128seconds; it remains OFF. No target-model computation changes with this flag.
