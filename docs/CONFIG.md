# CONFIG — what is served, and why each value is that value

The launcher `scripts/launch-flashnext.sh` writes `/srv/qwen5090/flashnext-config.yml` on the box and mounts it
into the container as `/app/config.yml`. That generated file is the served configuration; this document explains
it. Every value is either a measurement, an upstream default, or a fit constraint — the tag says which.

## Model

| setting | value | why |
| --- | --- | --- |
| `model_name` | `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab` | [r0b0tlab's 2.50 bpw EXL3 pack](https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw) of `Qwen3.8-Flash-Next`, routed experts at mixed K = 2 / 3 / 4. Served since 2026-09-18 21:21 UTC ([R511](../bench/results/r511-promote-2p50.md)); the 3.05 bpw pack before it, measured equal on GSM8K without stop strings ([R509](../bench/results/r509-gsm8k-nostop.md)) |
| `backend` | `exllamav3` | TabbyAPI's ExLlamaV3 backend; v1.5.0, pinned into the image |
| `max_seq_len` | 262,144 | the checkpoint's `max_position_embeddings` — 262,144 is native, not a limit we chose |
| `cache_size` | 966,656 | the shared page pool, in tokens, served since [R561](../bench/results/r561-promote-slots8.md) with 8 slots: the largest 8-slot pool whose boot and whose minimum free VRAM under a cold 120k prefill plus 8 streams stay within 32 MiB of the 4-slot configuration's on each card (983,040 left cuda:0 36 MiB short under load, [R558](../bench/results/r558-slots8.md)); boot free 2,085 / 867 MiB. At 4 slots, 1,032,192 from [R548](../bench/results/r548-promote-gdnbf16-ring.md): bf16 GDN state frees 864 MiB, and 1,032,192 is the largest pool whose boot leaves each card within 32 MiB of the free VRAM of the configuration before it (1,048,576 leaves cuda:1 577 MiB against 737; a smaller margin made a late CUDA graph capture run out of memory in R548's second try). Before it, 983,040 from [R546](../bench/results/r546-promote-rawk.md): the QSA raw-key ring (`EXL3_QSA_RAWK_RING=1`) frees 2.34 GiB, and 983,040 is the largest pool that boots with the layers on their usual cards (999,424 fails with `RuntimeError: Insufficient VRAM in split for model and cache`; 835,584 to 933,888 move layer 23 to cuda:0, which changes greedy output). Before it, 819,200 from [R525](../bench/results/r525-promote-int8mix.md): the int8 mixer weights free 218 / 258 MiB, and 819,200 was then the largest that boots at 4 slots and 8-bit KV (835,584 fails with `RuntimeError: Insufficient VRAM in split for model and cache`, [R516](../bench/results/r516-int8-mixer-pool.md); without the int8 weights the ceiling is 786,432, [R495b](../bench/results/r495b-2p50-audition.md)). 1,973 / 737 MiB stay free on cuda:0 / cuda:1 after boot with the 320 MiB draft embedding copy ([R528](../bench/results/r528-promote-mtp-pruned.md)), 1,173 / 225 MiB under whole-pool load. On the 3.05 bpw pack the ceiling was 360,448 at 4 slots ([R480](../bench/results/r480-exl3-pool.md)) and 262,144 at 8 ([R452](../bench/results/r452-exl3-cache-bits.md)). Free VRAM is not allocatable VRAM: arithmetic from free VRAM overestimated the pool |
| `cache_mode` | `8,8` | 8-bit KV, the operator's call (no q4) |
| `max_batch_size` | 8 | Eight slots since 2026-09-19 19:07 CEST ([R561](../bench/results/r561-promote-slots8.md)): with bf16 GDN state a slot's recurrent state costs half what it did, and 8 slots cost 65,536 pool tokens against 4. At 8 streams the aggregate is +16 to +23 % over 8 requests on 4 slots and queue wait in an 8-agent replay drops from 1.4 s to 0.12 s at the median ([R558](../bench/results/r558-slots8.md)). From 2026-09-18 to 2026-09-19 the daily ran 4 slots ([R480](../bench/results/r480-exl3-pool.md)), when the state was fp32. **Set explicitly in any case.** exllamav3 clamps the generator's batch size to `cache.num_slots`, and TabbyAPI derives **4** for a recurrent model (128 otherwise). This checkpoint carries GDN recurrent state, so without this line a c8 test silently measures c4 |
| `tensor_parallel` | `false` | `qwen4_exp` raises `NotImplementedError: Tensor-parallel is not currently implemented for Qwen4ExpForConditionalGeneration`. Layer split is the only mode |
| `gpu_split` | `[30, 30]` | a YAML **list**, not the string `"30,30"` — a string fails pydantic with `type=list_type` |
| `gpu_split_auto` | `false` | explicit rather than autosplit: TabbyAPI #405 applies `autosplit_reserve` to device 0 only |
| `cpu_moe_offload_layers` / `cpu_moe_split_experts` | `0` / `0` | zero offload is intended: offloading experts costs decode rate. Both are set explicitly so a stale nonzero value elsewhere cannot silently move the baseline |
| `chunk_size` | 2048 | prefill chunk; also the requeue budget when `output_chunking` is on |
| `output_chunking` | `true` | long outputs are reserved in rounds and requeued, bounding per-job cache growth |
| `vision` | `true` | available in this checkpoint and **off by default in TabbyAPI**. The weight index carries 987 vision tensors plus `vision_config`, `image_token_id` and the token ids, and hidden inside the main shards rather than a separate `vision_*.safetensors` — so their absence from a file listing is not evidence that vision is missing. Used in a live agent session: a DSH session read back its own Chrome screenshots through this path on 2026-09-16 |

## Reasoning and tools

| setting | value | why |
| --- | --- | --- |
| `reasoning` | `true` | without it the model's thinking arrives inline in `content` and `reasoning_content` is null; harnesses render the two fields separately and the reasoning leaks into the visible answer |
| `reasoning_budget_tokens` | 32768 | a bound on runaway thinking, not a quality knob. A budget hit injects the end-of-thinking tokens; a client-side `max_tokens` cap is a different mechanism, because it ends the request instead. The 2026-09-16 DSH breakage was **not** this budget: the request produced ~17k reasoning tokens against a 32768 cap, so it never fired |
| `tool_format` | `qwen3_coder` | without a tool format the model's calls are not parsed into `tool_calls`: they arrive as raw text in `content` and the response finishes `stop` instead of `tool_calls`. `qwen3_coder` matches this checkpoint's template byte for byte |

No `reasoning_budget_message` is set. A budget hit therefore injects the end-of-thinking tokens with no instruction
to wrap up. Measured at a 150-token budget, the model still produces a correct answer in `content`; a cut becomes
reachable as thinking lengthens.

## Drafting (MTP)

| setting | value | why |
| --- | --- | --- |
| `draft_mode` | `mtp` | the checkpoint's own MTP head |
| `draft_num_tokens` | 3 (`DRAFT=` env) | **the schema field is `draft_num_tokens`**; `num_draft_tokens` is not a schema field and is silently ignored, which runs the default depth while the config appears to say otherwise |
| `draft_cache_mode` | `Q8` | the draft schema accepts only FP16/Q8/Q6/Q4 — a pair like `"8,8"` is rejected |
| `draft_num_tokens_by_batch` | `[[4, 3], [5, 2], [8, 1]]` (`DRAFT_POLICY=`) | depth 3 up to 4 decode-ready jobs, depth 2 at 5, depth 1 from 6 on; a verify step stays at 16 rows or fewer, the limit of the cooperative MoE decode kernels (5 jobs × depth 2 = 15 rows). Depth 2 from 6 jobs on costs 32–39 % ([R414](../bench/results/r414-bszn16.md), [R560](../bench/results/r560-c8-policy.md), [R562](../bench/results/r562-profile-c8.md)); depth 2 at 5 jobs is +11 to +20 % and served since 2026-09-19 ([R576](../bench/results/r576-promote-c5-policy.md)) |
| `dynamic_draft` | `false` | measured loss: 184 vs 191 t/s at c1, 229 vs 258 at c4; crashes at c4 with a CUDA-graph out-of-memory ([R497](../bench/results/r497-draft-confidence.md)) |

Drafts are sampled **greedily**; the target is not. Acceptance therefore tracks how predictable the continuation
is, which is why decode rate is content-dependent (see `GOTCHAS.md` #9).

## Launcher knobs (not config keys)

| knob | default | what it does |
| --- | --- | --- |
| `IMG=` | `tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16` | which image to serve; a patch variant is A/B'd without editing the launcher. The chain is in [`docker/README.md`](../docker/README.md) |
| `NVME_TIER=` | `/srv/qwen5090/fast/exl3-nvme-daily` when `IMG` is the served image and the variable is unset; otherwise off | host directory for the NVMe prefix tier, mounted at `/nvme-tier`; `NVME_TIER=` (empty) turns it off. Experiments that launch another image never open the daily's directory ([R534](../bench/results/r534-promote-nvme-tier.md)) |
| `NVME_TIER_GB=` | `64` with the daily default | byte cap for the tier; above it the tier evicts leaf-first on its radix tree, superseded interior checkpoints first |
| `EXTRA_ENV=` | `EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1 EXL3_DRAFT_PINNED_STAGING=1 EXL3_BATCH_VERIFY=1 EXL3_MTP_HEAD_N=65536 EXL3_MOE_PREFILL_E3=1 EXL3_HC_MIX_V2_INT8=1 EXL3_MTP_DEVICE_DRAFT=1 EXL3_EMBED_GPU=1 EXL3_EMBED_GPU_PRUNED=1 EXL3_MOE_PREFILL_E3_DET=1 EXL3_GDN_BA_WARP1=1 EXL3_HC_APPLY_WARP1=1 EXL3_GR_STATE_REGRID=1 EXL3_QSA_RAWK_RING=1 EXL3_GDN_STATE_BF16=1 EXL3_NGRAM_PREFETCH2=1` | the engine patches, each opt-in and default-off in the image; the table below says which result admitted each |
| `CACHE=`, `MAXBS=`, `CACHE_MODE=`, `GPU_SPLIT=`, `CKPT_NAME=` | 966656, 8, `8,8`, `[30, 30]`, the 2.50 bpw pack | pool, slots, KV bits, split and checkpoint for experiments; the defaults are the served values |
| `DRAFT_POLICY=` | empty | expands into `draft_model.draft_num_tokens_by_batch` only when set, so the unpatched path stays byte-identical |
| `SYS_KV=` | 0 | the host KV tier: flat at 8 slots ([R358](../bench/results/r358-hostkv.md)); turns a 12.7 s re-prefill of an evicted 105k session into 0.5–0.7 s for 16 GiB of host RAM ([R493](../bench/results/r493-host-kv-tier.md)); not served |

## Sampling — the preset, not the config

`sampling.override_preset: qwen38_thinking` names a file the launcher writes to
`/srv/qwen5090/sampler_overrides/` and mounts over `/app/sampler_overrides/qwen38_thinking.yml`:

```yaml
temperature: {override: 0.6,  force: false}
top_k:       {override: 20,   force: false}
top_p:       {override: 0.95, force: false}
```

**Why it exists:** TabbyAPI has no sampling fallbacks unless a preset is named, and it warns at boot that requests
omitting samplers run "untruncated: temperature 1.0, top_k 0, top_p 1.0, min_p 0". DSH sends only `max_tokens`, so
every DSH request was sampled at raw T=1.0. The reasoning degenerated into multilingual text, emitted an
end-of-thinking tag inside it, and the remainder was delivered as the visible answer (R338).

**Why these values:** truncation is what stops the runaway thinking (T=1.0 with top_p/top_k added was also
coherent), and 0.6/0.95/20 is the model's thinking-mode recommendation. No sweep of these three values is recorded
in this repository.

**`force: false` matters:** the preset supplies fallbacks only, so a client that samples deliberately keeps its own
values. Verified in the log: a greedy probe still reads `temperature: 0, greedy (req)`, while DSH's requests read
`temperature: 0.6 (preset), top_k: 20 (preset), top_p: 0.95 (preset)`.

## Memory

KV page pool cost: 15,236 B per token, so 1.52 GB of VRAM per 100k tokens and 14.7 GB for the 966,656-token pool. Per token that is 13 attention layers (the checkpoint's 12 full-attention layers plus the MTP block's 1, whose draft cache is `Q8` at the same size), each with 2 KV heads × 256 dims for K and for V at 8 bits (1,024 B), fp16 scales per 32 values (64 B), and the QSA indexer's planes: one pooled 128-dim fp16 key per 4 tokens (64 B) and, with the raw-key ring, 20 raw fp16 keys per 256-token page (20 B). Without the ring the raw plane holds every token's key (256 B) and a token costs 18,304 B. Layout from `exllamav3/cache/quant.py` and `exllamav3/cache/qsa.py`. The figures this page gave before 2026-09-19 16:30 CEST (14,144 B per token, 1.41 GB per 100k) left out the indexer planes.

| setting | value | why |
| --- | --- | --- |
| `sysmem_recurrent_cache` | 4096 (MiB) | host-tier cache for GDN recurrent checkpoints; hybrid prefix reuse needs both the KV pages and a matching stashed checkpoint |
| `sysmem_kv_cache` | `$SYS_KV`, default 0 | the host KV tier. Set to 4096 MiB and **measured**: nothing moves — same 8/8 and 48.4 t/s aggregate on eight unique ~40k-prompt jobs, same 0.43 s repeat TTFT on a 152,761-token prompt. The pool is never spilled to host at these shapes ([R358](../bench/results/r358-hostkv.md)) |

## Engine flags

| flag | what it does | admitted by |
| --- | --- | --- |
| `EXL3_HOST_GAP_REWIND=1` | removes host-side gaps from the GDN decode path | [R428](../bench/results/r428-hcmix2-stack-ab.md) |
| `EXL3_HC_MIX_V2=1`, `EXL3_HC_MIX_V2_MIN_R=1` | bit-exact V2 of the hyper-connection mixer kernels | [R428](../bench/results/r428-hcmix2-stack-ab.md) |
| `EXL3_LS_PREFILL_PIPELINE=1` | pipelines prefill chunks across the two cards | [R442](../bench/results/r442-ppipe.md) |
| `EXL3_MOE_COOP_V2=1` | bit-exact V2 of the fused MoE decode kernel | [R460](../bench/results/r460-moecoop-v2-ab.md) |
| `EXL3_SHARED_EXPERT_OVERLAP=1` | the shared expert on a side CUDA stream | [R490](../bench/results/r490-shared-overlap.md) |
| `EXL3_DRAFT_PINNED_STAGING=1`, `EXL3_BATCH_VERIFY=1`, `EXL3_MTP_HEAD_N=65536` | pinned draft staging, one readback for the verify step, a 65,536-token draft head | [R499](../bench/results/r499-decode-r4.md), [R514](../bench/results/r514-promote-r4i.md) |
| `EXL3_MOE_PREFILL_E3=1` | grouped MoE prefill for every K | [R513](../bench/results/r513-prefill-e3-r2.md), [R517](../bench/results/r517-promote-stack.md) |
| `EXL3_HC_MIX_V2_INT8=1` | int8 hyper-connection mixer weights: frees 218 / 258 MiB for the page pool; changes the c1 greedy output (fingerprint `e7fb377c987d685c`) at unchanged GSM8K and tool-eval | [R516](../bench/results/r516-int8-mixer-pool.md), [R525](../bench/results/r525-promote-int8mix.md) |
| `EXL3_MTP_DEVICE_DRAFT=1`, `EXL3_EMBED_GPU=1`, `EXL3_EMBED_GPU_PRUNED=1` | the MTP draft chain stays on the GPU; the draft token lookup reads a 320 MiB copy of the 65,536 embedding rows the draft head can emit (rows outside it fall back to the full table); byte-identical | [R522](../bench/results/r522-mtp-pruned.md), [R528](../bench/results/r528-promote-mtp-pruned.md) |
| `EXL3_MOE_PREFILL_E3_DET=1` | the E3 grouped MoE prefill writes each expert assignment to its own slot and reduces the slots in a fixed order instead of `atomicAdd`, so cold prefill is identical run to run; 200 MiB of transient scratch at a 2,048-token chunk | [R531, R533](../bench/results/r533-e3-det-precise.md), [R535](../bench/results/r535-promote-e3det.md) |
| `EXL3_GDN_BA_WARP1=1`, `EXL3_HC_APPLY_WARP1=1`, `EXL3_GR_STATE_REGRID=1` | three launch re-grids for small decode kernels, each off unless set to `1`, each keeping every output's arithmetic order (byte-identical): the GDN B/A projection GEMV and the hyper-connection `hc_apply` run one warp per block instead of 256 threads when the served grid has fewer blocks than the card has SMs (1 stream; at 4 streams the served kernels run), and the V2 mixer's state kernel on the int8 path spreads its outputs over more blocks; code c1 +0.99 %, prose c1 +1.10 %, code c4 −0.17 % over 8 boots | [R538](../bench/results/r538-decode-r6.md), [R540](../bench/results/r540-promote-r6.md) |
| `EXL3_QSA_RAWK_RING=1` | the QSA indexer keeps its raw keys in a ring of 20 rows per 256-token page instead of one row per token; only the pool-update kernel reads them, and only the rows of the 4-token block an append touches. The pooled plane, the selection and the attention are bit-identical; 2.34 GiB freed at 819,200 tokens, 163,840 more pool tokens; verify windows above 17 tokens raise | [R546](../bench/results/r546-promote-rawk.md) |
| `EXL3_GDN_STATE_BF16=1` | the Gated-DeltaNet recurrent state (per slot and per draft position) is stored in bf16 instead of fp32; the recurrence math stays fp32 and the state is rounded once per step on store. 864 MiB freed at 4 slots and depth 3. Not byte-identical: the c1 greedy fingerprint becomes `f4add302e176d78e`, the 30k fingerprint stays; decode +0.3 to +2.1 % on 48 paired prompts, GSM8K 0.974, tool-eval 86.8 | [R548](../bench/results/r548-promote-gdnbf16-ring.md) |
| `EXL3_NGRAM_PREFETCH2=1` | each verify forward's per-layer n-gram embedding row reads (the 18.5 GiB table stays on NVMe) start on the worker thread right after the MTP draft ids reach the host, instead of inline when the forward reaches the embedding; prefill staging moves to the worker too. Same rows, same order: byte-identical. +0.8 % code / +0.9 % prose at 1 stream over four boots, prefill 1.01–1.02×. Also turns on `EXL3_NGRAM_TIMING`, a wall-clock line in the log every 30 s | [R559](../bench/results/r559-ngram-prefetch.md), [R565](../bench/results/r565-promote-ngram-prefetch.md) |

## Image

`tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16` (`tabbyapi:stack-r4-e3r2` plus the [`mtp-pruned-r1`](../docker/overlays/mtp-pruned-r1/), [`tool-choice-r1`](../docker/overlays/tool-choice-r1/), [`ple-ckpt-clone-r1`](../docker/overlays/ple-ckpt-clone-r1/), [`nvme-tier-r4`](../docker/overlays/nvme-tier-r4/), [`e3-det-r1`](../docker/overlays/e3-det-r1/), [`decode-kernels-r6`](../docker/overlays/decode-kernels-r6/), [`qsa-rawk-ring-r1`](../docker/overlays/qsa-rawk-ring-r1/) and [`gdn-state-bf16-r1`](../docker/overlays/gdn-state-bf16-r1/) overlays): TabbyAPI pinned at `53da7919`, ExLlamaV3 v1.5.0, the R338 requeue token-count fix, and every layer in [`docker/README.md`](../docker/README.md). Each Dockerfile asserts the versions it builds on and each overlay installer checks the SHA-256 of every file it replaces.
