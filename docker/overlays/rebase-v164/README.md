# Experimental ExLlamaV3 1.6.4 rebase

Prepared 2026-10-21. The served image and checkpoint remain unchanged. This packet pins ExLlamaV3 `da68d57de61781e5b74ad68698ac4caf36408d7a` (1.6.4) and TabbyAPI `a270c555db91a5178d72f7dee7bdd34951ac1dad`. It contains a native comparison and a separate source-built performance recovery. [Measurements](../../../bench/results/flashnext-rebase-repair.md) document the regression, individual screens and their limitations. The initial [patch audit](PATCH-AUDIT.md) describes the native comparison before recovery.

Run `bash build.sh` for the official-wheel comparison or `bash build-native.sh` for the source-built recovery on a Linux host containing the exact pinned parent image. `prepare.py` checks upstream archive, wheel and patch hashes. Builds run Linux CPU compatibility and regression checks. Neither script installs a daily. Source-native builds target sm120 and carry a different extension from the official wheel; runtime provenance distinguishes them.

The recovery keeps upstream architecture interfaces, native Job lifecycle, cache ownership and unsupported-shape fallbacks. It adds:

- Python/C++ decode capacities of32 rows, plus integer sm120 MoE dispatch and the original rank320/H4/hidden2560 int8 mixer. Fractional bitrates retain upstream kernels.
- An optional65536-column MTP draft head. Same-prefix native LinearEXL3 outputs match; full-vocabulary-prefix logits differ with width-dependent native dispatch. This is not full-vocabulary fidelity qualification.
- Native full-chunk/no-cooldown scheduling with `EXL3_PREFILL_COMPAT=1`.
- Deterministic E3 prefill via `EXL3_MOE_PREFILL_E3_DET=1`; scoped real-layer routing and output parity pass against the original runtime.
- Bounded target lookahead on disjoint GPUs via `EXL3_NATIVE_PREFILL_PIPELINE=1`: default two full chunks, optional eight with `EXL3_NATIVE_PREFILL_WINDOW=8`, with at most one chunk processed ahead. Native checkpoint predicates and Job draft/carry/page continuation remain in charge. Scoped same-native recurrent state, logical KV, checkpoint, export/carry and continuation differentials pass.

Optional early shared-expert overlap uses one process-lifetime PyTorch stream per GPU and releases its input after the routed dependency is queued. Real-layer output/unload and full-pool residency checks distinguish this path from the rejected private-stream attempts.

Optional performance selectors default off. `repair/candidate.env` records the experimental screen configuration. It is not the served launcher. Full-pool residency is measured with the original901120-token KV pool rather than substituted with a smaller cache.

Three explored settings remain excluded: persistent tiled mixer weights fail full-pool placement; batched MTP verification and device-resident drafting pass scoped correctness checks but do not improve the measured agent replay. Selecting the separate serial shared expert (`EXL3_MOE_SHARED_COOP=0`) also gives no useful replay gain. Batched sampled verification intentionally changes seeded trajectories and remains off.

Checkpoint manifests pin both self-calibrated2.18 and2.68bpw packs. Their small task screen established no accuracy gain over the existing pack. The rebase repair uses the original checkpoint to isolate engine changes. No SWE task-solving result or full BF16-reference fidelity pass is claimed.
