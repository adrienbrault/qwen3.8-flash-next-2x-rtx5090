# CONFIG — what is served, and why each value is that value

The launcher `scripts/launch-flashnext.sh` writes `/srv/qwen5090/flashnext-config.yml` on the box and mounts it
into the container as `/app/config.yml`. That generated file is the served configuration; this document explains
it. Every value is either a measurement, an upstream default, or a fit constraint — the tag says which.

## Model

| setting | value | why |
| --- | --- | --- |
| `model_name` | `qwen3.8-flash-next-exl3-3.05bpw` | EXL3 3.05 bpw quant of `Qwen3.8-Flash-Next`; the checkpoint shipped for this box |
| `backend` | `exllamav3` | TabbyAPI's ExLlamaV3 backend; v1.5.0, pinned into the image |
| `max_seq_len` | 262,144 | the checkpoint's `max_position_embeddings` — 262,144 is native, not a limit we chose |
| `cache_size` | 360,448 | the shared page pool, in tokens. **Measured with 4 slots (2026-09-18, R480):** 360,448 boots, 393,216 fails with the same error; K8V4 (`cache_mode 8,4`) boots 425,984. **With 8 slots:** 262,144 boots; 393,216 fails with `RuntimeError: Insufficient VRAM in split for model and cache` (R337). Earlier arithmetic from *free* VRAM suggested ~444k and was wrong — free VRAM is not allocatable VRAM |
| `cache_mode` | `8,8` | 8-bit KV, the operator's call (no q4) |
| `max_batch_size` | 4 | Four slots since 2026-09-18 (R481): each slot carries the GDN recurrent state in fp32, one copy per draft position plus one, so four slots instead of eight release ~1.7 GiB that the page pool takes. A fifth to eighth request queues. **Set explicitly in any case.** exllamav3 clamps the generator's batch size to `cache.num_slots`, and TabbyAPI derives **4** for a recurrent model (128 otherwise). This checkpoint carries GDN recurrent state, so without this line a c8 test silently measures c4 |
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
| `dynamic_draft` | `false` | measured loss: 184 vs 191 t/s at c1, 229 vs 258 at c4 |

Drafts are sampled **greedily**; the target is not. Acceptance therefore tracks how predictable the continuation
is, which is why decode rate is content-dependent (see `GOTCHAS.md` #9).

## Launcher knobs (not config keys)

| knob | default | what it does |
| --- | --- | --- |
| `IMG=` | `tabbyapi:53da7919-rqcount` | which image to serve; a patch variant is A/B'd without editing the launcher |
| `DRAFT_POLICY=` | empty | expands into `draft_model.draft_num_tokens_by_batch` only when set, so the unpatched path stays byte-identical |
| `SYS_KV=` | 0 | the host KV tier (measured flat, above) |

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

| setting | value | why |
| --- | --- | --- |
| `sysmem_recurrent_cache` | 4096 (MiB) | host-tier cache for GDN recurrent checkpoints; hybrid prefix reuse needs both the KV pages and a matching stashed checkpoint |
| `sysmem_kv_cache` | `$SYS_KV`, default 0 | the host KV tier. Set to 4096 MiB and **measured**: nothing moves — same 8/8 and 48.4 t/s aggregate on eight unique ~40k-prompt jobs, same 0.43 s repeat TTFT on a 152,761-token prompt. The pool is never spilled to host at these shapes (`2026-09-16-r358-hostkv`) |

## Image

`tabbyapi:53da7919-rqcount` — TabbyAPI pinned at `53da7919` with ExLlamaV3 v1.5.0, plus the R338 requeue
token-count patch. Recipe: `flan/docker/Dockerfile.tabbyapi` in the sibling `kubernetes-home` repo, which asserts
at build time both that the exllamav3 version is the expected one and that the patch applied.
