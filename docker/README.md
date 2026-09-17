# The served image, layer by layer

The daily on `:8022` runs `tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap-ppipe-nosync-mtpfix2-moecoopv2` (since 2026-09-17 12:45 CEST). The tag is the build history: each suffix is one layer built on the previous one, from the files in this directory. Provenance and licences for every input are in [`../THIRD_PARTY.md`](../THIRD_PARTY.md); the measurement that admitted each layer is in [`../docs/MEASUREMENTS.md`](../docs/MEASUREMENTS.md).

| layer (tag suffix) | recipe | inputs | what it adds | admitted by |
|---|---|---|---|---|
| `tabbyapi:53da7919-rqcount` | `Dockerfile.tabbyapi` | TabbyAPI `53da7919`, ExLlamaV3 v1.5.0 (`cu12` extra), the `sed` requeue count fix | the server + engine, `usage.completion_tokens` correct past the requeue boundary | R331 / R338 |
| `qsa-cid` | `Dockerfile.tabbyapi-qsa-cid` (CUDA devel base, native rebuild) | `qsa-multijob-applied.patch`, `ci-depth-applied.patch`, `tabby-split.patch`, `rebuild-native.py` | multi-job QSA sparse attention; concurrency-indexed draft depth | R341, R340, R350 |
| `-pr337` | `Dockerfile.tabbyapi-pr337` | `upstream-pr337-layer-split-device.patch` (exllamav3#337, creslinux) | current CUDA device pinned during layer-split forwards | R362, R363 |
| `-bszn16` | `Dockerfile.tabbyapi-bszn` (`PATCH=bszn16.patch`, native rebuild) | `bszn16.patch` | fused MoE decode path up to 16 rows (`MAX_BSZN` 8 → 16) | R414 (promoted 2026-09-17 02:15) |
| `-coopwide` | `Dockerfile.tabbyapi-bszn` (`PATCH=coopwide.patch`, `BASE=…-bszn16`) | `coopwide.patch` | wide stage-B MoE tile at ≥ 128 slots | R421 (promoted 03:15) |
| `-hcmix2` | `Dockerfile.tabbyapi-bszn` (`PATCH=hc-mix-v2-r2.patch`, `BASE=…-coopwide`) | `hc-mix-v2-r2.patch` | hyper-connection mixer V2, opt-in `EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1` | R426, R428 (promoted 04:32) |
| `-hostgap` | `Dockerfile.tabbyapi-pyfile` (`SRC=hostgap-gated_delta_net.py`, `DST=exllamav3/modules/gated_delta_net.py`) | `hostgap-gated_delta_net.py` | GDN host-gap rewind, opt-in `EXL3_HOST_GAP_REWIND=1` | R422, R425 |
| `-ppipe` | `Dockerfile.tabbyapi-pypatch` (`PATCH=prefill-pipeline.patch`) | `prefill-pipeline.patch` | two-card prefill pipeline, opt-in `EXL3_LS_PREFILL_PIPELINE=1` | R427 → R442 |
| `-nosync` | `overlays/prefill-nosync-overlay` (`install.py` + `manifest.json`, SHA-256-pinned) | six engine files | no blocking host syncs in the pipelined prefill | R432 |
| `-mtpfix2` | `overlays/prefill-pipeline-mtp-overlay` (cumulative) | same six files | pipeline MTP eligibility + free-VRAM guard fixes | R441, R442 (promoted 07:35: 30k prefill 5.5 → 3.5 s, 120k 22.3 → 13.1 s) |
| `-moecoopv2` | `overlays/moe-coop-v2-overlay` (`install.py --disable-precompiled`, extension JIT-rebuilt in the image) | `exl3_moe_coop.cu`, `exl3_moe_coop_v2_kernel.cuh`, `comp_units/exl3_moe_coop_instances.cuh` | fused MoE decode kernel V2, opt-in `EXL3_MOE_COOP_V2=1` | R460, R461 (promoted 12:45: c4 +4 %, c8 +8 %, byte-identical) |

Every layer above was admitted with byte-identical greedy output on the c1 256-token and 30k prompts (the fingerprints in `../docs/MEASUREMENTS.md`), except where the layer only changes rows ≥ 9 (bszn16, coopwide at ≥ 128 slots), where the gate was GSM8K and tool-eval under concurrency instead.

Rebuild order: `Dockerfile.tabbyapi` → `-qsa-cid` → `-pr337` → `-bszn` ×3 (bszn16, coopwide, hc-mix-v2-r2) → `-pyfile` (hostgap) → `-pypatch` (ppipe) → the three overlays in order (`docker build --build-arg BASE=<previous tag> -f overlays/<name>/Dockerfile overlays/<name>` for the first two; the MoE overlay's build context is the overlay's parent and it takes `--build-arg BASE=<tag>`, not an image id). The Dockerfiles pin TabbyAPI by SHA and ExLlamaV3 by version and assert both at build time.
