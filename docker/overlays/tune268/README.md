# tune268

Build `tabbyapi:tune268-r1` on `tabbyapi:r828-prompt-lookup-r3` with `bash docker/overlays/tune268/build.sh`, or run `bash docker/build-chain.sh` from the repository root. `BASE`, `TAG`, `DOCKER` and `MAX_JOBS` override the standalone build. The Dockerfile verifies the patch SHA256, applies it at fuzz 0, rebuilds the image's native extension with its existing `rebuild-native.py`, imports it and rejects changes to the predecessor's SASS. The build uses no GPU and disables network access.

Round 1 adds K6 dense V2 GEMM/mGEMM twins and K2.5 MoE V2/V3/r2 decode kernels. Both new selectors default to 0 in the image: `EXL3_DENSE_V2_K6` and `EXL3_MOE_COOP_HALF`. The served launcher sets both to 1. The original build preserved all 1,847 predecessor functions and added 37 kernels; the real-weight differential gate passed 6,466/6,466 checks bitwise.

The installed-source recipe is reproduced by this chain; rebuilding produces a different image ID. The launcher pins the measured operator build and requires a reviewed pin update before a rebuilt image can serve. Promotion measurements and their conditions are in [publish268-served-268](../../../bench/results/publish268-served-268.md). These patches are this repository's work on ExLlamaV3, under MIT; TabbyAPI retains AGPL-3.0.
