# Third-party code and attribution

This repository is MIT-licensed (see [LICENSE](LICENSE)) for the **original** work: documentation, benchmark harnesses and probes, the launcher, the overlay installers, and the measurements. Everything below is derived from, redistributes, or was made possible by someone else's work, and stays under its own licence. If a credit is missing, that is an error: open an issue.

## The model and the checkpoints

| item | origin | licence |
|---|---|---|
| Qwen3.8-Flash-Next (the model) | Qwen team, Alibaba — https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Qwen Community License 1.0 (`license: other`, `license_name: qwen-community-1.0`); the checkpoint's `LICENSE` file governs the weights |
| `qwen3.8-flash-next-exl3-3.05bpw` (served 2026-09-16 to 2026-09-18) and `…-2.05bpw` | EXL3 quantisations by **turboderp** — https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3 (exllamav3 1.4.4 converter, `mul1` codebook, calibration 250 rows × 2048) | the model's licence; the quantisation is turboderp's work |
| `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab` (the served checkpoint since 2026-09-18) | EXL3 quantisation by **r0b0tlab** — https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw (routed experts at K = 2 / 3 / 4, 4-bit MTP layer) | the model's licence |

This repository ships no weights.

## The engine and the server (the image base)

| item | origin | licence | how it is used |
|---|---|---|---|
| ExLlamaV3 v1.5.0 | **turboderp** — https://github.com/turboderp-org/exllamav3 | MIT | the inference engine; every kernel patch below modifies its source and is a derivative work under MIT |
| TabbyAPI (commit `53da7919d4e45c63f4acbcbbc00cbe0f60a1ce65`) | **theroyallab** — https://github.com/theroyallab/tabbyAPI | **AGPL-3.0** | the OpenAI-compatible server; `docker/Dockerfile.tabbyapi` is adapted from their `docker/Dockerfile`; two patches below touch TabbyAPI files and are AGPL-3.0 derivatives |
| llguidance 1.8.0 | Microsoft — https://github.com/guidance-ai/llguidance | MIT | structured output (`json_schema`, `regex_pattern`) through TabbyAPI's exllamav3 grammar backend; unmodified |
| flash-linear-attention, PyTorch cu128, Triton | their respective projects | Apache-2.0 / BSD / MIT | pulled by TabbyAPI's `cu12` extra; unmodified |
| `nvidia/cuda:12.8.1-*-ubuntu24.04` | NVIDIA | NVIDIA Deep Learning Container licence | base images; nothing redistributed here |

## Upstream ExLlamaV3 pull requests carried in the served image

| file here | upstream | author | status | what it is |
|---|---|---|---|---|
| `docker/upstream-pr337-layer-split-device.patch` | [exllamav3#337](https://github.com/turboderp-org/exllamav3/pull/337) | **creslinux** | open (2026-09-17) | keeps the current CUDA device on the module's device during a layer-split forward; **verbatim redistribution**, applied by `docker/Dockerfile.tabbyapi-pr337` |
| `bench/hotvocab/hotvocab-applied.patch` | [exllamav3#303](https://github.com/turboderp-org/exllamav3/pull/303) | **keldenl** | open | "Speed up Qwen MTP decoding by 22% with a selected vocabulary head"; ported to this checkpoint's `qwen4_exp_mtp` by this repo. Measured **inapplicable** on a two-card layer split (`docs/MEASUREMENTS.md`, R377); not in the served image |

## Patches written for this repository (MIT for the patch text; derivative of ExLlamaV3 MIT / TabbyAPI AGPL-3.0 where they modify those files)

All of these were written with **OpenAI Codex** (`gpt-6-astra`, and `gpt-5.6-sol` from 2026-09-17 21:05 CEST) working from static source dumps, then built, measured and accepted or rejected on the box by the operator. Codex never ran on the GPU box; every number attached to them in `docs/MEASUREMENTS.md` is a box measurement. The design notes, audits and verifiers that came with each patch live next to it in the private working repository; the files here are the ones the served image was built from.

| file here | modifies | derived from | what it is |
|---|---|---|---|
| `docker/Dockerfile.tabbyapi` (the `sed` on `exllamav3/generator/job.py`) | ExLlamaV3 | MIT | the "R338 requeue token-count fix": `usage.completion_tokens` and the logged T/s under-counted by ~5× past the requeue boundary |
| `docker/qsa-multijob-applied.patch` | ExLlamaV3 (`exllamav3_ext/libtorch/attention.{cpp,h}`, `modules/attention_fn/{bc_attn,bc_dsa,bc_mla,dsa_triton,qsa_triton}.py`) | MIT | causal multi-job QSA sparse attention within the existing slot limits (the captured QSA path was single-job above its sparse threshold) |
| `docker/ci-depth-applied.patch` + `docker/tabby-split.patch` | ExLlamaV3 (`generator/{generator,job}.py`) **and TabbyAPI** (`common/config_models.py`, `backends/exllamav3/model.py`, `config_sample.yml`) | MIT + **AGPL-3.0** | concurrency-indexed MTP draft depth (`draft_num_tokens_by_batch`, the served policy `[[4, 3], [8, 1]]`) |
| `docker/bszn16.patch` | ExLlamaV3 (`modules/{block_sparse_mlp,mlp}.py`, `exllamav3_ext/libtorch/mlp.h`, `exllamav3_ext/quant/exl3_moe_coop{.cu,_kernel.cuh}`) | MIT | raises the fused decode-shaped MoE path's row limit `MAX_BSZN` 8 → 16 so batched MTP decode stays on the fast path |
| `docker/coopwide.patch` | ExLlamaV3 (`exl3_moe_coop.cu`) | MIT | selects the wide stage-B tile at ≥ 128 slots |
| `docker/hc-mix-v2-r2.patch` | ExLlamaV3 (`exllamav3_ext/hc_mix.{cu,cuh}`, `bindings.cpp`, `modules/hyperconnections.py`) | MIT | bit-exact rewrite of the decode-form hyper-connection mixer kernels (`EXL3_HC_MIX_V2`) |
| `docker/hostgap-gated_delta_net.py` | ExLlamaV3 (`modules/gated_delta_net.py`, whole-file overlay) | MIT | removes host-side gaps in the GDN decode path (`EXL3_HOST_GAP_REWIND`) |
| `docker/prefill-pipeline.patch` | ExLlamaV3 (`generator/{generator,job}.py`, new `generator/prefill_pipeline.py`) | MIT | two-card prefill pipeline for the layer split (`EXL3_LS_PREFILL_PIPELINE`) |
| `docker/overlays/prefill-nosync-overlay` | ExLlamaV3 (`generator/*`, `modules/{block_sparse_mlp,moe_batch_recon}.py`, new `util/prefill_nosync.py`) | MIT | removes the blocking host syncs in the pipelined prefill |
| `docker/overlays/prefill-pipeline-mtp-overlay` | same files, cumulative | MIT | the pipeline's MTP eligibility and free-VRAM guard fixes |
| `docker/overlays/moe-coop-v2-overlay` | ExLlamaV3 (`exl3_moe_coop.cu`, new `exl3_moe_coop_v2_kernel.cuh`, `comp_units/exl3_moe_coop_instances.cuh`) | MIT | bit-exact V2 of the fused MoE decode kernel: bounded work loops and batched completions (`EXL3_MOE_COOP_V2`) |
| `docker/overlays/decode-kernels-r2` | ExLlamaV3 (`exllamav3_ext/libtorch/blocksparse_mlp.{cpp,h}`, `exl3_moe_coop.{cu,cuh}`) | MIT | the shared expert on a side CUDA stream (`EXL3_SHARED_EXPERT_OVERLAP`) |
| `docker/overlays/refbase` | ExLlamaV3 (nine engine files) | MIT | the reference tree the next overlays were written against; both new switches off |
| `docker/overlays/decode-kernels-r4` | ExLlamaV3 (`generator/*`, `modules/hyperconnections.py`, `exllamav3_ext/*`) | MIT | pinned draft staging, batched verify, draft-head pruning (`EXL3_MTP_HEAD_N`), int8 mixer weights |
| `docker/overlays/prefill-e3-r2` | ExLlamaV3 (`modules/block_sparse_mlp.py`, `exllamav3_ext/quant/exl3_moe_prefill_e3*`, `bindings.cpp`) | MIT | grouped MoE prefill for K = 2 / 3 / 4 (`EXL3_MOE_PREFILL_E3`) |
| `docker/overlays/stack-r4-e3r2` | same files as the two above | MIT | both overlays in one image, with a merged `bindings.cpp` |

Each overlay's `manifest.json` pins the SHA-256 of the file it replaces and of the file it installs, and `install.py` refuses to run on a base whose files do not match.

## Ideas and techniques credited (no code copied)

| source | what was taken |
|---|---|
| DominikBucko — https://github.com/DominikBucko/qwen38-flash-next-2x3090 (Apache-2.0) | the same model on two RTX 3090s under vLLM: its exact top-k QSA scratch, 64 MiB score chunks, fused PLE RMSNorm, MTP fixes and dynamic speculative schedule were ported as opt-in experiments on the **vLLM + EXL3** audition track, not into the served ExLlamaV3 image; credited here because that track's measurements are referenced from `docs/` |
| HaberstrohSystems — https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang and https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang | the INT2 AutoRound checkpoint and the 2-bit `moe_wna16` SGLang path were assessed as a route for this box; nothing was ported |
| vcruz305 — https://github.com/vcruz305/vllm-exl3 | the vLLM plugin that loads EXL3 checkpoints, the basis of the vLLM + EXL3 audition track (its patch series is not in this repository) |
| vcruz305 — https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe | the pruned draft head (`EXL3_MTP_HEAD_N=65536`) and int8 hyper-connection mixer weights, both measured here in R499 |
| peonist-ai — https://github.com/peonist-ai/halogen (0.6.0) | prompt lookup beside the MTP draft, measured here in R501 |

## Benchmark and evaluation tools

| tool | origin | licence | use |
|---|---|---|---|
| lm-evaluation-harness (`lm_eval`, GSM8K 5-shot, n=200) | EleutherAI — https://github.com/EleutherAI/lm-evaluation-harness | MIT | quality gate |
| tool-eval-bench v2.1.0 | **SeraphimSerapis** — https://github.com/SeraphimSerapis/tool-eval-bench | MIT | agentic tool-call gate (69 cases × 4) |
| mini-SWE-agent + SWE-bench Verified | SWE-agent / princeton-nlp — https://github.com/SWE-agent/mini-swe-agent | MIT (harness) / SWE-bench data licence | the SWE-bench measurements in `bench/results/r359-swebench.md` |
| `bench/*.py` probes (`probe.py` = `fn_bench`, `needle.py`, `capabilities.py`, `summarize.py`, `multiprompt.py`, `agentic-edit.py`, `nostop_proxy.py`, `agent_replay.py`, `revisit.py`) | this repository | MIT | original |
