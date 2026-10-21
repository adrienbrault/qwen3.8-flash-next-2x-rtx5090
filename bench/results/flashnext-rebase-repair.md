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
