# Carried feature audit

Audit date: 2026-10-21. Baseline is the installed `r828-prompt-lookup-r3` image, not only the earlier rebase-dev-r3 snapshot. This experimental candidate replaces the old native extension with the official 1.6.4 wheel and uses the pinned TabbyAPI commit. It is not yet admitted for serving.

| Feature | Candidate treatment | Evidence / limitation |
|---|---|---|
| Batch-indexed MTP | Retained and adapted | Draft config validation, generator capacity maximum and per-ready-batch depth selection in the patches. |
| Reasoning loop recovery | Retained in Tabby | Loop-think ladder and forced end-of-thinking retained; new reasoning/content phase plumbing preserved. Runtime checks pending. |
| Automatic and constrained tool calls | Retained / adapted | Old forcing helper uses the new phase API; automatic parsed tools are smoke-tested. Full tool evaluation pending. |
| Prompt encode once / off event loop | Retained | Ported request ID cache and long-tokenization worker; native encode cache also retained. |
| Prefix KV disk tier | Replaced by upstream | 1.6.4 DiskPageCache and Tabby memory.disk_cache_* fields. Never opens the daily tier during screens. |
| Dense recurrent checkpoints | Replaced by upstream | Native recurrent checkpoint interval fields; custom 12K tail policy is not enabled. Cache-reuse evaluation pending. |
| Whole-prompt/resumable prefill | Replaced by upstream scheduler | Native shared chunk size, cooldown and prefill rotation. No old scheduler flags are claimed active. |
| Adaptive prompt lookup | Native alternative, disabled for first comparison | Native hybrid_draft_tokens exists; old R828 adaptive gate/counters are not part of the native generator. The carried result handler must not import or fabricate old lookup counters. |
| Pruned MTP vocabulary and GPU embedding mirror | Not enabled in native baseline | Old EXL3_MTP_HEAD_N=65536, device draft loop and pruned embedding mirror are absent. This is another configuration difference, not proof that native kernels alone explain the regression. A default-off Python-only vocabulary cap is feasible; its native fallback and half-bit head representation require review/tests before any experiment. |
| GDN recurrent state | Changed | Upstream fp16 store with fp32 scans replaces the local bf16 store. This changes numerics and requires qualification. |
| QSA, MoE, HC, shared expert and rows32 CUDA overlays | Replaced for initial comparison | New upstream implementations run; old extensions are not blindly replayed. This is a whole-stack comparison, not attribution to version alone. Retain/port old kernels only if the comparison establishes a need. |

## Installed engine files with local changes relative to the former upstream pin

The following files were inspected through a three-way source inventory. A clean textual merge does not establish semantic compatibility, so old native/kernel copies are not automatically installed. The candidate patches state exactly what is retained.

- `architecture/qwen4_exp_mtp.py`
- `cache/cache.py`
- `cache/checkpoint_policy.py`
- `cache/mtp_window.py`
- `cache/prefill_merge.py`
- `cache/qsa.py`
- `cache/recurrent.py`
- `cache_trace.py`
- `exllamav3_ext/bindings.cpp`
- `exllamav3_ext/gdn.cu`
- `exllamav3_ext/hc_mix.cu`
- `exllamav3_ext/hc_mix.cuh`
- `exllamav3_ext/hc_mix_v3.cu`
- `exllamav3_ext/hc_mix_v3.cuh`
- `exllamav3_ext/libtorch/attention.cpp`
- `exllamav3_ext/libtorch/attention.h`
- `exllamav3_ext/libtorch/attention_bc.h`
- `exllamav3_ext/libtorch/blocksparse_mlp.cpp`
- `exllamav3_ext/libtorch/blocksparse_mlp.h`
- `exllamav3_ext/libtorch/blocksparse_mlp_bc.h`
- `exllamav3_ext/libtorch/gated_delta_net.cpp`
- `exllamav3_ext/libtorch/lc_fork.h`
- `exllamav3_ext/libtorch/mlp.cpp`
- `exllamav3_ext/libtorch/mlp.h`
- `exllamav3_ext/quant/comp_units/exl3_dense_v2_inst_gemm.cu`
- `exllamav3_ext/quant/comp_units/exl3_dense_v2_inst_gemv.cu`
- `exllamav3_ext/quant/comp_units/exl3_dense_v2_inst_mgemm.cu`
- `exllamav3_ext/quant/comp_units/exl3_moe_coop_inst_k2.cu`
- `exllamav3_ext/quant/comp_units/exl3_moe_coop_inst_k3.cu`
- `exllamav3_ext/quant/comp_units/exl3_moe_coop_inst_k4.cu`
- `exllamav3_ext/quant/comp_units/exl3_moe_coop_instances.cuh`
- `exllamav3_ext/quant/coop_autotune.cu`
- `exllamav3_ext/quant/coop_autotune.cuh`
- `exllamav3_ext/quant/exl3_dense_v2.cu`
- `exllamav3_ext/quant/exl3_dense_v2.cuh`
- `exllamav3_ext/quant/exl3_gemm.cu`
- `exllamav3_ext/quant/exl3_gemm.cuh`
- `exllamav3_ext/quant/exl3_gemm_inner_v2.cuh`
- `exllamav3_ext/quant/exl3_gemm_v2_kernel.cuh`
- `exllamav3_ext/quant/exl3_gemv.cu`
- `exllamav3_ext/quant/exl3_gemv_v2_kernel.cuh`
- `exllamav3_ext/quant/exl3_moe_coop.cu`
- `exllamav3_ext/quant/exl3_moe_coop.cuh`
- `exllamav3_ext/quant/exl3_moe_coop_kernel.cuh`
- `exllamav3_ext/quant/exl3_moe_coop_r2_kernel.cuh`
- `exllamav3_ext/quant/exl3_moe_coop_v2_kernel.cuh`
- `exllamav3_ext/quant/exl3_moe_coop_v3_kernel.cuh`
- `exllamav3_ext/quant/exl3_moe_prefill_e3.cpp`
- `exllamav3_ext/quant/exl3_moe_prefill_e3.cuh`
- `exllamav3_ext/quant/exl3_moe_prefill_e3__cuda.cu`
- `generator/async_generator.py`
- `generator/disk_cache.py`
- `generator/draft_overlap.py`
- `generator/generator.py`
- `generator/job.py`
- `generator/pagetable.py`
- `generator/prefill_pipeline.py`
- `generator/prompt_lookup.py`
- `generator/sampler/custom.py`
- `generator/sampler/sampler.py`
- `model/model_ls.py`
- `modules/arch_specific/qwen4_exp_mtp.py`
- `modules/attention_fn/bc_attn.py`
- `modules/attention_fn/bc_dsa.py`
- `modules/attention_fn/bc_mla.py`
- `modules/attention_fn/dsa_triton.py`
- `modules/attention_fn/qsa_triton.py`
- `modules/block_sparse_mlp.py`
- `modules/embedding.py`
- `modules/embedding_pruned.py`
- `modules/gated_delta_net.py`
- `modules/gated_delta_net_fn/gated_delta_rule.py`
- `modules/hyperconnections.py`
- `modules/mlp.py`
- `modules/moe_batch_recon.py`
- `modules/ngram_embedding.py`
- `modules/ple.py`
- `modules/qsa_indexer.py`
- `tokenizer/tokenizer.py`
- `util/prefill_nosync.py`

## Tabby files with carried changes

- `backends/exllamav3/model.py`
- `common/config_models.py`
- `common/gen_logging.py`
- `common/metrics.py`
- `common/sampling.py`
- `common/tokenize_offloop.py`
- `endpoints/OAI/router.py`
- `endpoints/OAI/utils/chat_completion.py`
- `endpoints/OAI/utils/tool_choice.py`
- `endpoints/core/router.py`
