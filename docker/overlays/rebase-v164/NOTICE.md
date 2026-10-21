# Sources and licences

- ExLlamaV3: turboderp-org/exllamav3, pinned 1.6.4 source and official wheel in `manifest.json`, MIT. The retained batch-indexed draft selector, passive cache diagnostics and checkpoint-policy shim derive from the previously published local Flash-Next patch stack; `engine.patch` records the new delta.
- TabbyAPI: theroyallab/tabbyAPI, pinned commit in `manifest.json`, AGPL-3.0. `tabby.patch` carries the prior local loop-think, forced-tool, tokenization and tracing/server changes onto that source. Distribution must retain TabbyAPI's licence and corresponding source.
- Parent image: `tabbyapi:r828-prompt-lookup-r3`, exact image ID in the manifest. Provenance and third-party credits of the existing chain remain in the public recipe's `THIRD_PARTY.md` and `docker/`.
- Checkpoints: Qwen base derivatives published by turboderp, exact revisions and LFS hashes in the checkpoint manifests. Retain the shipped model licences; no modified weights are published by this packet.

For `Dockerfile`, the official 1.6.4 wheel is selected for Python 3.12, CUDA 12.8 and Torch 2.9.0, matching the parent. No native binary from the former patched engine is reused by the candidate.

`Dockerfile.native` instead compiles the pinned new engine source with `engine.patch` followed by `native.patch`, for sm120. The MoE V2/V3/r2 and int8 mixer kernels in the latter patch come from the previously published local daily stack, rebased onto the new native interfaces and upstream fallback handling. Python and native scratch capacities are rebuilt together at32rows. No native binary from the prior daily or official wheel is reused by that image. Its extension hash is in the dated source-build receipt.
