# GDN state bf16 round 1 — implementation status

Round 1 adds `EXL3_GDN_STATE_BF16=1`, a default-off option that stores the Gated-DeltaNet recurrent state in bf16 instead of fp32. The recurrence math stays fp32. The base is the served chain `tabbyapi:nvme-tier-r4-e3det` (R535). Line numbers without a prefix refer to the local extraction `served-src-r535/exllamav3/`. `p:` marks the patched tree. No GPU, compiler or remote host was used. Nothing here is a measurement. The CUDA changes have not been compiled.

## Files

| file | change |
|---|---|
| `exllamav3_ext/gdn.cu` | both recurrent kernels get a last template parameter `typename state_t = float`; bf16 load via the existing `as_float` overloads, bf16 store via a new `store_state` (round to nearest even); the host wrapper dispatches bf16 state to a separate launch block that returns before the served block |
| `modules/gated_delta_net.py` | import-time selector; `GatedDeltaNet.recurrent_state_dtype`; `GDNLayerState` allocates in that dtype; checkpoint size and rewind job sizes follow the element size; `unstash` refuses a dtype mismatch |
| `modules/gated_delta_net_fn/gated_delta_rule.py` | the chunked (fla) prefill gets an fp32 copy of a bf16 initial state; the existing `state.copy_(new_state)` rounds the fp32 final state once per forward |

`libtorch/gated_delta_net.cpp`, `gdn.cuh`, `bindings.cpp`, the generator, the cache classes and the NVMe tier are unchanged. The patch is 383 lines. It applies with `patch -p1 -F0` to a full scratch copy of the served tree with exit code 0 (macOS `patch`, which is BSD-derived; the box image uses GNU patch).

## Producer/consumer map of `recurrent_state`

Allocation and accounting:
- `modules/gated_delta_net.py:201-205`: `GDNLayerState` allocates `[max_batch_size, max_history + 1, Nv, Dk, Dv]` fp32 on `meta`. `alloc` at `:223-228` materializes and zeroes it on the layer's device. `clear` at `:237-240` zeroes one slot. `free` at `:231-234`.
- `cache/cache.py:160-171` builds one `layer_state_cls(module, max_batch_size, max_history, id(cache))` per recurrent layer. TabbyAPI passes 4 slots and `max_history` 3 on the daily (`launch-flashnext-tabby.sh:77,142,192`).
- `modules/gated_delta_net.py:212-216`: `get_checkpoint_size` hardcoded `* 4` for the state. It feeds `GDNState.checkpoint_size` (`:117-120`), the RAM budget of `RecurrentCache` (`cache/recurrent.py:54-100`), and the NVMe tier namespace (`generator/disk_cache.py:1568-1579`, `checkpoint_bytes`). p: `* self.recurrent_state.element_size()`.
- `:219-220` `storage_size` and `util/memory.py:394-397` use `element_size` already. The loader's placement (`make_tp_allocation` `:1214-1215`) sees the halved size.
- `modules/mamba2.py:10,197` reuses `GDNLayerState`. Mamba2 has no `recurrent_state_dtype`, so it keeps fp32 through the `getattr` default.

Decode and verify (the served path on the daily):
- `modules/gated_delta_net.py:1018-1031` fetches `(conv_state, recurrent_state)` from the layer state. `:1061-1075` runs `BC_GatedDeltaNetSplit.run_bszN` for 1 ≤ bsz ≤ 16 and 1 ≤ seqlen ≤ MAX_QLEN, with `save_history` = MTP verify (`generator/generator.py:1248`).
- `libtorch/gated_delta_net.cpp:303-316` calls `cuda_recurrent_gated_delta_rule_gr` with the tensor. The file never reads the state's dtype. It reads `recurrent_state.size(1)` for the graph geometry snapshot (`:352,360`) and patches the untyped state pointer into the captured graph (`:383,394,406`). `run_bsz1_b` (`:39-62`, fused qkvz models) goes through the same wrapper.
- `exllamav3_ext/gdn.cu:426-658` (generic head dims, plus the MAMBA2 mode) and `:660-849` (128x128) are the only kernels that read or write the state. Each token step reads the state twice from global memory (the `k·S` readback at `:579-593` / `:780-801`, the update at `:614-632` / `:810-832`) and writes it once. With history, step s reads plane s (plane 0 for s = 0) and writes plane s+1, and the last step writes plane 0 (`:507-520` / `:726-741`). Without history, reads and writes are in place on plane 0.
- `gdn.cu:851-1015` validates (`TORCH_CHECK_DTYPE(recurrent_state, kFloat)` at `:914`) and dispatches 14 instantiations. `cuda_recurrent_mamba2_gr` (`:1167-1286`) has its own kFloat check and is untouched.
- Torch fallback, the fused recurrent rule for seqlen < Nv or with history: `gated_delta_rule.py:196-221` → the same wrapper.

Rewind after verification:
- `generator/generator.py:1293-1301` (`reject_remainder`) and `:1582-1583` (`rewind(0)` after full acceptance) → `GDNState.rewind` (`modules/gated_delta_net.py:127-135`) → `_collect_rewind_jobs` (`:45-68`) → `rewind_state_job` (`:293-314`) → `ext.batched_state_rewind` (`gdn.cu:1945-1960,1987-2010`). The kernel copies `num_elements / 4` `float4` vectors, so `num_elements` is a count of 4-byte words. Both branches of `rewind_state_job` passed an element count: the host-gap branch (`EXL3_HOST_GAP_REWIND=1`, on in the daily env) at `:306-309`, the tensor branch at `:310-314`. With bf16 elements that would copy two planes. The kernel checks only `num_elements % 4`, not alignment; a bf16 plane is 1,572,864 B and a slot 4 planes, so every plane start stays 16-byte aligned on a 256-byte-aligned allocation (the harness's sentinel test would expose a wrong copy). p: both pass `_state_words` = `numel * element_size // 4`, which is the served value for fp32 (the CPU test extracts the function and checks 4 → identity, 2 → half).
- `GDNLayerState.rewind` (`:250-262`, non-batched) and `generator/job.py:904,1451` go through `copy_` or the same jobs.

Prefill:
- `gated_delta_rule.py:165-192`: seqlen ≥ Nv without history takes the vendored fla chunk kernel per slot with `initial_state = recurrent_state[s, 0]`, then `state.copy_(new_state)`. The kernel loads `h0` with `.to(tl.float32)` (`vendor/fla/chunk_delta_h.py:130`) and allocates the final state in fp32 (`:363,366`). p: a bf16 state is passed as an fp32 copy (fp32 state is passed as the same object, as before); `copy_` rounds once per forward (one 2,048-token chunk on the daily). The served output path already stores the per-64-token chunk states `h` in the activation dtype (`chunk_delta_h.py:362,365`: `k.new_empty`; `k` is bf16 because `l2norm_fwd` keeps the input dtype, `vendor/fla/l2norm.py:91`). The carried recurrence itself was fp32.
- The KDA chunk path (`:113-137`, `chunk_kda` asserts an fp32 initial state) is unchanged. KDA layers keep fp32 state.

Checkpoints (RAM and NVMe tier):
- `GDNLayerState.stash` (`:317-322`) returns `recurrent_state[slot, :1].cpu()`, so it keeps the live dtype. `unstash` (`:325-329`) used `copy_`, which would convert silently. p: raises `TypeError` on a dtype mismatch.
- `cache/recurrent.py:54-100` (`put`), `:166-182` (TP stash/unstash in the workers), `generator/job.py:910,1524,1630`, `generator/pagetable.py:296-302` only move the stash dict.
- NVMe tier: `serialize_stash` (`generator/disk_cache.py:1160-1201`) records each tensor's dtype in the payload table and writes raw bytes; `deserialize_stash` (`:1204-1228`) rebuilds the same dtype. The namespace (`:1574-1582`) includes `engine_identity()` (`:1098-1134`: every `EXL3_*` variable except `EXL3_NVME_TIER*`, plus the sha256 of all engine sources) and `recurrent_layout[].checkpoint_bytes` (`:1568-1573`). The flag enters the namespace twice: as `EXL3_GDN_STATE_BF16=1` in the env list and through the halved `checkpoint_bytes`. A tier written with fp32 state is therefore never opened by a bf16 process; the old namespace is removed at open. If a mismatch reached `unstash` anyway, it raises instead of converting. No `disk_cache.py` change was needed: the dtype reaches the namespace through these two served inputs, and `unstash` is the only new runtime guard.

Other copies: none. No code clones the state tensor besides `stash`. `GDNState.tp_export` carries no tensors. `util/memory.py` only counts bytes.

## Design

Variant (a) is implemented: every plane is bf16 (the live plane 0 and the MTP history planes 1..max_history), all math in fp32 registers, round to nearest even on every store, exact widening on every load.

Variant (b), bf16 history planes with an fp32 live plane, is not different in the way it would need to be:
- The kernel re-reads the state from global memory at every token. In a verify window, token s > 0 reads the plane written by token s-1. With bf16 history planes, every intermediate step of a window is rounded anyway. Only the last step, which writes plane 0, would stay fp32.
- After a partial acceptance, `batched_state_rewind` copies history plane `a` into plane 0. Plane 0 then holds a bf16-rounded state. After a full acceptance it holds an fp32 state. The committed state's precision would depend on the acceptance pattern.
- It saves less: with 4 planes, 3/8 of the fp32 bytes instead of 1/2.
- It needs two tensors of different dtypes where the kernel uses one tensor and one stride. That changes the kernel signature, the graph pointer patches in `gated_delta_net.cpp` (a second pointer per launch), the rewind kernel (a converting copy), and both rewind job branches.
- Keeping plane 0 in fp32 through a window would need the b12x structure: carry the state in registers across the window's tokens and write only rounded copies (`b12x/sequence/gdn_decode/_cute_kernels.py:361-367` load to fp32 registers, `:500-532` store `self.state_type(state[...])` per destination). That is a rewrite of both kernels. It also breaks the property below.

Property of (a): a verify window and the same tokens decoded one at a time give bit-identical states and outputs, in fp32 (served) and in bf16. Each token reads the stored state of the previous token, the per-token arithmetic and reduction order are the same (the v-split depends only on bsz), and the round happens at the same point. So an MTP accept/rewind sequence commits exactly the state that plain decode of the accepted tokens would. The harness checks this with `torch.equal`, including rewinds through `GDNLayerState` jobs.

The output of token t uses the fp32 state before rounding (`v_out += q·state` right after the update), as in b12x. The next token reads the rounded state.

Rounding cadence differs by path: decode and verify round once per token; chunked prefill rounds once per forward (2,048 tokens on the daily, `chunk_size: 2048`).

## Default-off proof (structural)

- The selector is read once at import: `_gdn_state_bf16 = os.environ.get("EXL3_GDN_STATE_BF16", "0") == "1"` (p: `modules/gated_delta_net.py:41`). Unset, `GatedDeltaNet.recurrent_state_dtype` is `torch.float` (p: `:612`), so `GDNLayerState` allocates the served fp32 tensor.
- For fp32 elements, `_state_words` returns the served element count, `get_checkpoint_size` returns the served value (`element_size() == 4`), `unstash` passes its check and runs the served `copy_`, and the prefill passes the same state object to fla.
- `gdn.cu`: the fp32 host path is the served text. The CPU test deletes the bf16 block and the dtype-check change from the patched host function and gets the served function verbatim, including its 14 `LAUNCH_RULE` lines. None of them names a state type, so they instantiate `state_t = float`.
- The kernels: the CPU test maps the patched kernel region back (`state_t` → `float`, `as_float(*p)` → `*p`, `store_state(p, x)` → `*p = x`, drop the defaulted template parameter) and gets the served region verbatim. For `float`, `as_float` is `return x;` and `store_state` is `*p = x;`, both `__forceinline__`. The fp32 kernels are therefore the served kernels plus identity wrappers. SASS identity is not proven here (no nvcc); the GPU harness `--ref` gate checks bit equality against the served image on real launches.
- The Dockerfile does not set the selector. It asserts `_gdn_state_bf16 is False` with the variable unset.

## VRAM

Geometry: 36 GDN layers, Nv = 48, Dk = Dv = 128, 4 slots, 4 planes (MTP depth 3). One plane of one slot of one layer is 48·128·128 = 786,432 elements.

| | fp32 (served) | bf16 | freed |
|---|---|---|---|
| per layer (4 slots × 4 planes) | 50,331,648 B (48 MiB) | 25,165,824 B (24 MiB) | 24 MiB |
| all 36 layers | 1,811,939,328 B (1.6875 GiB) | 905,969,664 B (0.84375 GiB) | 905,969,664 B = 864 MiB |
| per slot, all layers (4 planes) | 452,984,832 B | 226,492,416 B | 216 MiB |
| per extra MTP depth (1 plane × 4 slots × 36 layers) | 432 MiB | 216 MiB | 216 MiB |
| checkpoint (state part, per layer) | 3 MiB | 1.5 MiB | halves RAM/NVMe bytes per checkpoint |

The conv state (bf16 already) is unchanged.

KV equivalent: the launcher documents 15.75 KiB/token at 8-bit KV (12.00 KiB KV + 3.75 KiB fp16 QSA planes, `launch-flashnext-tabby.sh:60-61`). 864 MiB / 15.75 KiB = 56,173 tokens, 3.4 steps of 16,384. Counting only the 12 KiB KV part it would be 73,728 tokens (4.5 steps), which is not the real per-token cost. Both cards hold GDN layers and KV pages; the card with the smaller margin sets the pool. Expected ladder result: 819,200 + 3 × 16,384 = 868,352; +4 steps (884,736) only if existing slack on the limiting card covers the rest.

## Risks

- Numerics. Every decode token adds one bf16 rounding (relative error ≤ 2^-9 per element) to the carried state. Heads with slow decay (|g| ≈ 1e-4..1e-3, memory horizons of thousands of tokens) accumulate these errors over their horizon; fast heads forget them. The change is not bit-exact by design: flag-on greedy fingerprints will differ from the canonical c1 `e7fb377c987d685c` / 30k `4a255910dee2d9c5`. The harness measures the growth by decay class over 512 steps, over an MTP schedule and over a 32k chunked prefill; the model-level gates decide. vLLM exposes the same choice for SSM/GDN state as `--mamba-ssm-cache-dtype`.
- Rounding cadence. Prefill rounds per 2,048-token forward, decode per token, so the error profile after a long prompt is dominated by the decode phase.
- NVMe tier. Any flag flip is a new namespace, and the previous namespace is deleted at open (`disk_cache.py:290-303`). The patched sources also change `sources_sha256`, so even the flag-off image opens a new namespace and GCs the R534/R535 tier on first boot, like every source round. A/B arms must not share a tier directory, and tier-served latencies are only comparable between arms that were both warmed.
- RAM checkpoint budget. `checkpoint_size` halves with the flag on, so `RecurrentCache` holds about twice as many checkpoints in the same `sysmem_recurrent_cache` budget. This changes eviction behaviour (for the better) only with the flag on.
- Placement. The loader places layers by measured allocations inside the manual `gpu_split [30, 30]`. Halving the state bytes can move a layer boundary between the cards, which changes per-card KV capacity and was the confound in R489 (`gdnstate/r2/impl-status-gdn-state-r2.md`). The box spec records per-card VRAM and the ladder per boot.
- CUDA graphs. `BC_GatedDeltaNetSplit` captures one graph per (bsz, seqlen, history) with the kernel instantiation of the dtype seen at capture. The dtype is fixed per process (import-time selector), so a graph is never replayed on the other dtype. No runtime guard was added, because it would need a `Slot` field in `gated_delta_net.h`. Two caches of different state dtypes in one process are not possible through the selector.
- Compile. Not compiled locally. The bf16 instantiations double the recurrent-kernel instantiations from 14 to 24 (10 new; channelwise/KDA is excluded, and the host wrapper rejects bf16 with channelwise decay).
- Stacking. decode-kernels r6 touches `gdn.cu` at `@@ -8` and `@@ -1629..1720` (B/A GEMV); this round touches `:44-49` and `:426-1015`. The CPU test applies both orders at fuzz 0 and gets identical files. `overlay/manifest-on-r6.json` pins the same patch bytes on the r6 tree (`--build-arg MANIFEST=manifest-on-r6.json`) for the case that R540 is promoted first. adaptive-draft r1 touches only `generator/adaptive_draft.py`, `generator.py` and `job.py`, which this round does not touch.
- Scope. GDN layers only. KDA (GLM5.3/Kimi) and Mamba2 keep fp32 state. Generic head dims (non-128) are supported by the same template and covered by the harness's 64x64 cases, but no served model uses them.

## Local verification

- `served-source.patch` applies to a full scratch copy of `served-src-r535/exllamav3` with `patch -p1 -F0`: exit 0, output files equal the pinned `overlay_sha256`.
- CPU tests: `tests/test_gdn_bf16_cpu.py`, run with the `opus-nvme-tier-r3` venv (pytest, no torch): 25 passed, 0 skipped. They cover the patch/manifest pins, `install.py` on scratch roots (served and r6-stacked, including refusal of a drifted baseline and of a second run), the r6 stacking in both orders, the adaptive-draft file disjointness, the default-off literals, `_state_words`, the kernel and host-function revert-to-served proofs, the untouched Mamba2/rewind kernels, the NVMe-tier identity inputs in the served source, and the Dockerfile contract.
- `tests/gpu_gdn_bf16.py` compiles (`py_compile`). It has not run.
