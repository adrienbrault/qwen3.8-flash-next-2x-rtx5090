# impl-status: MTP device draft chain on a pruned embedding mirror, round 1

The flag is `EXL3_EMBED_GPU_PRUNED=1`. It is off by default and only acts together with `EXL3_MTP_DEVICE_DRAFT=1`, `EXL3_EMBED_GPU=1` and a pruned head (`EXL3_MTP_HEAD_N`).

Status: implemented and packaged. 52 CPU tests pass locally. The GPU work is specified but not run, because this machine has no GPU: two scripts, plus the served A/B in `box-ab-spec.md`.

All `src/` citations are against the pristine tree, which is the `tabbyapi:stack-r4-e3r2` package. "overlay" citations point into `overlay/exllamav3/`.

## 1. Map: which IDs reach the mirror lookup, and on which device

**The chain.** `Generator.iterate_draftmodel_mtp_gen` (src generator.py:809-935) runs in this order:
1. Step-0 IDs are the jobs' last verified tokens: `job.get_input_ids_list()`. They are copied on the host into `draft_input_ids_pinned` (generator.py:852-857). That buffer is pinned only with `EXL3_DRAFT_PINNED_STAGING=1` (generator.py:255-264).
2. The device chain is gated at generator.py:867-873: `_MTP_DEVICE_DRAFT`, no calibrator, `temp_hidden.is_cuda`, no indexed (vision) embeddings, and `target_embed.can_embed_device_ids(temp_hidden.device)`.
3. When the gate passes, the step-0 IDs are uploaded once to `temp_hidden.device` (generator.py:874-881).
4. In the loop (generator.py:883-918):
   - `draft_model.forward(batch_ids, …)` runs `Qwen4ExpMTPInputLayer.forward`. That calls the trunk's `Embedding.forward(x, params, out_dtype=half)` through `attached_model().modules[0]` (src modules/arch_specific/qwen4_exp_mtp.py:154), then moves the result to the input layer's device (:155).
   - The drafted ID is `sample_from_state(...)` (generator.py:896-898), and the chain keeps it on the device as the next `batch_ids` (generator.py:899-905).
5. There is one blocking readback at the end (generator.py:920-924).

**Where the IDs live.**
- `temp_hidden` at step 0 is `job.mtp_last_hidden`. It is sliced from `p_export_states[-1]` (generator.py:1566-1569, 1609-1611), which the trunk's final mixer exports. `attach_to` asserts that this mixer sits directly before lm_head (src architecture/qwen4_exp_mtp.py:159-164), so under the layer split it is on the lm_head's card.
- `sample_from_state` moves the state to the lm_head's device with `lm.prepare_for_device` (architecture/qwen4_exp_mtp.py:184). The pruned GEMV and the argmax run there (:185-208), so every drafted ID is on the lm_head's device.
- R499 (FINDINGS) puts the pruned head's 120 MiB on cuda:1, so that device is cuda:1 in the daily split.

**Which IDs reach the lookup.**
- *Position 0* is the verified target token, which can be any vocab ID. That includes the special tokens at the end of the vocabulary and CJK or other non-top-64K pieces. It is uploaded from the host, so its host copy exists for free.
- *Positions ≥ 1* are argmax indices over the pruned logits, which are `(bsz*seq, n_cols)` (architecture/qwen4_exp_mtp.py:196-208). They are therefore in `[0, n_cols)` by construction.
- `_pruned_head` is a **prefix** slice: `trellis[:, :n_cols//16]` and `svh[:n_cols]`, with `n_cols = min(HEAD_N, n_full)//128*128` (architecture/qwen4_exp_mtp.py:235-242). So the pruned ID set is exactly `{0 … 65,535}`.
- Other embedding calls take host IDs and never reach a mirror:
  - accept-prefill `batch_ids[a:b, 1:accepted_length]` (generator.py:1594-1605);
  - the CPU branch, `self.embedding.forward(x)` (src modules/embedding.py:177-178).

**No other sync in the chain.** `Embedding.forward` also stores `params["input_ids"] = x` (embedding.py:97-98), and the PLE n-gram layer reads it with a host-side gather (modules/ple.py:355-363). The trunk constructor inserts PLE layers in its own layer loop (src architecture/qwen4_exp.py:252-270). They are not part of `build_qwen4_block`. The MTP model holds only its input layer, one `build_qwen4_block` block and `stack_out` (architecture/qwen4_exp_mtp.py:77-117). So the draft chain never runs PLE on device IDs, and the device chain's only DtoH synchronization is the final readback. The routing that also reads `input_ids` is DeepSeek-V4 hash-MoE (modules/block_sparse_mlp_routing.py:249, 325), which is not this architecture.

**Where the table lives.** The trunk Embedding has `prefer_cpu` (embedding.py:44-46). It is built without `normalize` or `multiplier` (src architecture/qwen4_exp.py:237-242), so the only arithmetic after the gather is the `to2` cast to half (embedding.py:179-183).

**The existing r4 mirror.** It is the full `weight.detach().to(x.device)` (embedding.py:165-171), which is 248,320 x 2,560 x 2 B = 1.27 GB on cuda:1, where only about 1,099 MiB is free (R514/R517). That is why the flag stayed off.

## 2. Mechanism (overlay)

**New module `modules/embedding_pruned.py`.** It is torch-only, so the CPU tests can import it.
- `build_remap(keep_ids, num_rows)` (:30) returns an int32 table mapping vocab ID to mirror row, with -1 for absent IDs. It rejects duplicates, out-of-range IDs and empty sets.
- `PrunedEmbeddingMirror` (:59) builds the device copy (:69-87):
  - For a prefix keep set: `mirror = weight[:n].to(device)`. No device remap is allocated, because an in-set ID is its own row.
  - For any other keep set: `mirror = weight[keep]`, plus the device remap.
- Lookups:
  - `lookup(weight, ids, host_ids)` (:148) is the entry point.
  - `host_ids_in_set` (:101) is a host-side membership test: min/max for a prefix, remap otherwise. It does no device work.
  - `gather_device` (:112) is one `F.embedding` on the mirror, through `remap[ids]` for a non-prefix set.
  - `gather_host` (:118) handles a batch that contains an out-of-set ID. It runs `stage_rows` (:44-51), an `index_select` of the exact host table (detached) into a reused pinned buffer, then `.to(device, non_blocking=True)`, and records a CUDA event. Before the buffer is reused, it waits on the previous upload's event.
- Every branch returns exact copies of rows of the same table in the table's dtype. The caller then runs the same `to2` cast on the same device. So the output is bit-identical to the full-mirror gather.

**`modules/embedding.py`:**
- Flag at :17. `_pruned_mirror` is initialised at :43 and cleared at :74.
- `can_embed_device_ids` returns False in pruned mode (:80-82), so the 1.27 GB mirror is never built.
- `prepare_pruned_mirror(device, keep_ids)` (:86-106) builds or reuses the mirror. It requires `EXL3_EMBED_GPU=1`, a host table, a CUDA target, and a size within `EXL3_EMBED_GPU_MAX_MB`.
- In `forward`'s CUDA-ID branch (:195-200), when the flag is on and `params["embed_pruned"]` is set, the gather comes from `pm.lookup(...)`. Everything after the gather is the unchanged tail.
- An un-hinted device-ID call in pruned mode takes the existing synchronous host fallback (:206-210), which is correct and never allocates.

**`architecture/qwen4_exp_mtp.py`.** `draft_head_keep_ids()` (:217-236) returns `arange(n_cols)` as a cached host LongTensor, or None. It uses the same predicate as `sample_from_state`: `_MTP_HEAD_N` must be set and `_pruned_head()` must not be None. It calls `_pruned_head(lm, lm.device)` and reads `n_cols` from it, so the keep set is the head's own cached result. That cache holds the same bytes `sample_from_state` reuses on its first call.

**`generator/generator.py`:**
- Flag at :41.
- In `__init__` (:324-327), when the flag is on, `_prepare_pruned_embed()` (:817-851) builds the mirror at generator creation. That way the boot-time free-VRAM line includes it, and an allocation failure fails the load, not a request. The method returns `(keep_ids, lm_head device)`. It returns None, with one warning naming the reason, if any of these hold:
  - the device-draft flag is off;
  - there is no MTP draft model;
  - dynamic draft is on;
  - the head is not pruned;
  - lm_head is not on CUDA;
  - the mirror is refused.
- In `iterate_draftmodel_mtp_gen`:
  - The gate (:914-920) uses `pruned_embed is not None` in pruned mode and the unchanged `can_embed_device_ids` otherwise.
  - The step-0 upload goes to the mirror's device, and the host tensor is kept as `host_ids0` (:926-930).
  - Each position sets `params["embed_pruned"]=True`. Position 0 also sets `params["embed_pruned_host_ids"]=host_ids0` (:945-948).

**Placement.** The mirror sits on the lm_head's device, which is cuda:1 in the daily. That is where the IDs are:
- every drafted ID is produced there;
- the step-0 IDs are uploaded there, and that is the same card that `temp_hidden` is on.

So each lookup is one same-device gather with no ID hop. The only cross-card move is the existing embedding-to-input-layer `to_device` at arch_specific/qwen4_exp_mtp.py:155, which the r4 full mirror also pays. Putting the mirror on the draft block's card would add a hop for the IDs at every position.

**VRAM per card.**

| card | added | notes |
|---|---:|---|
| cuda:0 | 0 | |
| cuda:1 | 320.0 MiB | 65,536 x 2,560 x 2 B = 335,544,320 B; no device remap for the prefix set |
| host | ≈1 MiB + ≤80 KiB/shape | 0.95 MiB int32 host remap, ≤80 KiB pinned staging per batch shape |

The 120 MiB pruned head is not new; it is only built earlier. Expected cuda:1 free after boot is about 1,099 − 320 ≈ 779 MiB.

## 3. Default path

With `EXL3_EMBED_GPU_PRUNED` unset, the added code works as follows:
- It reads one module constant.
- It evaluates `pm = None` and one short-circuited `if` in `Embedding.forward`'s CUDA-ID branch. That branch is reached only when device IDs meet a host table.
- In the generator, the gate expression is `(… if _EMBED_GPU_PRUNED else target_embed.can_embed_device_ids(temp_hidden.device))`, which is the original call. The upload's device argument is `temp_hidden.device if pruned_embed is None else …`, which is the original device. `host_ids0 = batch_ids` is only a reference.
- No kernel, argument or order changes, and no allocation happens.

`EXL3_EMBED_GPU=1` without the new flag behaves exactly as in r4.

## 4. Verified locally (CPU, `.venv` with torch 2.14 CPU)

Command:

```
cd out/mtp-device-draft-pruned-r1 && ../../.venv/bin/python -m pytest -q tests
```

Result: **52 passed** (43 in round 1, 3 added in round 1b (§5b), 6 added in round 1c (§5c)). Each test file loads the real overlay or baseline source files through `tests/_harness.py`. The harness stubs every `exllamav3.*` module that is not loaded, so the CUDA extension is never needed.

- `test_pruned_mirror_cpu.py` (19):
  - remap values for prefix and general keep sets, and rejection of bad sets;
  - mirror rows equal to `weight[:N]` bitwise (fp16 and bf16);
  - in-set gather equal to the full-table gather bitwise;
  - out-of-set fallback (N-1, N, V-1, a mid ID, a row with ±65504, 6e-8 and −0) equal bitwise, with host-call counters;
  - the fallback never reads the mirror (the test rebinds it to NaN). On CPU the "mirror rows exact" check is trivially true, because `.to("cpu")` aliases the table; the GPU unit test checks the real device copy;
  - host membership edge cases;
  - a non-prefix keep set through the remap;
  - an exhaustive check of every vocab ID in batches of 16;
  - mirror re-keying.
- `test_embedding_module_cpu.py` (6):
  - host-ID `Embedding.forward` output identical across baseline, overlay-off, overlay with `EXL3_EMBED_GPU=1`, and overlay pruned;
  - `can_embed_device_ids` unchanged versus the baseline without the flag and False with it;
  - `prepare_pruned_mirror` gates on the flags, on a non-CUDA target, on keep=None and on the MB cap;
  - `unload` clears the mirror.
- `test_mtp_keep_ids_cpu.py` (9): `draft_head_keep_ids()` equals the `n_cols` of `_pruned_head` for HEAD_N 65536, 65600→65536, 100, unset, 0 and ≥ n_full. It returns None with bias or when unattached, and it is cached as one object.
- `test_generator_chain_cpu.py` (7). These run the real `iterate_draftmodel_mtp_gen` with fakes; a Tensor subclass reports `is_cuda`.
  - With default flags, both baseline and overlay produce the same drafted IDs, per-position IDs and gate calls.
  - In pruned mode, position 0 carries the host copy of the verified tokens (including 150,000 and 248,319) and positions 1-2 carry only `embed_pruned`. The full-mirror gate is never consulted, and the drafted IDs are unchanged.
  - With the flag set but unavailable, the chain falls back to the host chain.
  - `_prepare_pruned_embed` gating is covered, including that the mirror is placed on the lm_head device.
- `test_packaging_cpu.py` (2):
  - the manifest's `pre` hashes equal `src/`, which is still pristine;
  - `post` equals the overlay;
  - `install.py` succeeds on a copy of `src/` and refuses a second install.

The GPU scripts were compiled with `py_compile` but not run.

## 5. What only the box can verify

See `box-ab-spec.md` for the order and the 15-minute budget.

- **`tests/gpu_unit_pruned_embed.py`** checks, on the real 248,320-row table:
  - the mirror rows and the VRAM delta equal exactly 65,536 x 2,560 x 2 B;
  - the pruned, full-mirror and host paths agree bitwise on cuda:1, for in-set, boundary, special-token and mixed IDs;
  - the out-of-set path does not block the host behind queued device work.
  It also runs a per-lookup microbenchmark at 1, 4 and 16 rows.
- **`tests/gpu_chain_identity.py`** loads the real model. It compares draft windows, output tokens, accept and reject counts and `draft_stats` for off, pruned and full (r4) over 8 prompts at c1 and c4. It also reports the out-of-set count and the chain latency with both cards synchronized around each draft call.
- **Served A/B** checks:
  - the boot line and VRAM per card;
  - canonical fingerprints (`ae890c45d1000582` / `4a255910dee2d9c5`);
  - fn_bench c1 and c4 for code and prose;
  - the c4 30k stress pass for VRAM headroom.
- **Device attribution.** Nothing local confirms that lm_head, the trunk's final mixer and therefore the IDs are on cuda:1, or that the draft block is on cuda:0 (the survey marks the latter as inference). The boot line prints the mirror's device, and the chain script prints the lm_head, draft input layer and embedding devices.
- **Speed.** The survey's P2b forecast (−0.6 to −1.2 ms/step at c1) has never been measured for the device chain on this box.

## 5b. R522 box result and the failing unit check (round 1b)

**Box result, R522 (2026-09-19):**
- `gpu_chain_identity.py` passed. Across pruned vs off and full vs off, all 639 draft windows were identical, and so were the tokens and the accept counts. Of the 1,917 lookups (1,668 on the device, 249 through the host fallback), the 249 fallbacks carried 456 rows.
- Chain median: c1 1,585 → 1,377 µs and c4 1,918 → 1,700 µs.
- `gpu_unit_pruned_embed.py` passed everything except `oos result after async upload exact`.

**Root cause: a flaw in the test.** Production is correct.
- The check compared `out`, which is `Embedding.forward`'s output, against `w[h]`, which is raw table rows.
- `Embedding.forward` ends with `to2(x, out_dtype, …)`, and the test passed `out_dtype=torch.half` (src modules/embedding.py:179-181). So `out` is fp16.
- The table is loaded with `allow_bf16 = True` (src modules/embedding.py:55; src loader/safetensors.py:767 and :826 keep bf16 as bf16 when `allow_bf16` is set). So a bf16 checkpoint gives a bf16 `w[h]`.
- The old `bits_equal` returned False on `a.dtype != b.dtype` before it looked at a single bit.
- Every other identity check in the script compared forward output with forward output, so they passed. This was the only mixed-level comparison.
- The real 248,320-row table is bf16 exactly when this check fails and every other check passes. `gpu-unit.json` `dtype` shows it.

**Why this is not a race in the production path** (overlay modules/embedding_pruned.py:118-146 after round 1c):
- The H2D copy (`buf.to(device, non_blocking=True)`) and the recorded event both go on `torch.cuda.current_stream(self.device)`, the stream that the sleep and every consumer also use.
- `out` was read only after `torch.cuda.synchronize(dev)`.
- The buffer is rewritten only in the next out-of-set call, and only after `_staging_event.synchronize()`.
- On the box, the `pruned == full mirror` checks for the out-of-set cases also passed through `forward` (half on both sides), and the chain run used the fallback 249 times with identical draft windows.

Nothing in the overlay changed, so `manifest.json` is unchanged.

**What changed in `tests/`:**
- New `_refs.py`, which keeps the two levels apart: `raw_rows` in the table dtype, `forward_reference` = rows cast to fp16, `bits_equal`, and `describe`. `describe` names the dtype, the shape and the number of differing words, so a mismatch now explains itself.
- In `gpu_unit_pruned_embed.py`, section 3 now checks:
  - raw rows through `PrunedEmbeddingMirror.lookup` against `raw_rows`;
  - forward output against the full-mirror forward (both cast on the device);
  - a buffer-reuse race probe. Two same-shape out-of-set lookups with different IDs run back to back behind the ~100 ms sleep. The second must block for about the sleep, because it waits on the first upload's event, and each result must hold its own rows. This is the test of the race the operator was worried about.
  - The host-cast reference is printed as `INFO` only and does not gate.
- New CPU tests (46 pass in total):
  - `test_embedding_module_cpu.py::test_forward_output_is_cast_rows_not_raw_rows` pins, on the real module, that forward output equals `forward_reference` for bf16 and fp16 tables. It also shows that the round-1 comparison only holds for fp16. This test would have caught the flaw.
  - `test_pruned_mirror_cpu.py::test_staging_reuse_same_shape_distinct_ids` covers the same-shape reuse sequence at the logic level. On CPU there is no staging buffer, so the async part is GPU-only.

## 5c. R522 rerun: a real grad-mode bug in the out-of-set staging (round 1c)

**Box rerun (01:16-01:17 UTC):** `gpu_chain_identity.py` passed again. `gpu_unit_pruned_embed.py` passed every check through `oos_mixed_16`, then crashed in section 3:

`RuntimeError: index_select(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad`

The crash was at the round-1 `embedding_pruned.py:131`, `torch.index_select(weight, 0, flat, out = buf)`.

**Root cause: a real bug in production code.**
- `Embedding.load` wraps the table as `nn.Parameter(weight)` (src modules/embedding.py:62), so `requires_grad` is True.
- Ops with `out=` raise on grad-requiring inputs unless the caller is under `no_grad` or `inference_mode`.
- The served path never hit this, because the generator and `forward_ls` run under inference mode. The chain identity run is served-like and passed with 249 fallbacks for that reason.
- Section 3 of the unit script called `pm.lookup` directly with grad enabled. Round 1b added that direct call, which is why the crash only appeared on the rerun.
- The production code should not depend on the caller's grad mode, so this is a production fix, not only a test fix.

**Fix (overlay `modules/embedding_pruned.py`):**
- The new `stage_rows(weight, flat_ids, buf)` (:44-51) gathers from `weight.detach()` into the pinned buffer.
- `gather_host` calls it (:141).
- The CPU branch also reads `weight.detach()` (:129).
- `detach()` is a metadata-only alias: no copy, no sync, same bytes. The in-set path (`gather_device`, :112-116) is unchanged. It already reads the mirror, which was built from `weight.detach()` (:77).
- An audit of the module's other ops found nothing else: remap construction (:30-41) writes fresh non-grad tensors, and `remap[ids]` and `F.embedding(rows, mirror)` take no `out=`.
- Host-path results no longer carry `requires_grad`, which matches the mirror paths.

**Manifest:** only `modules/embedding_pruned.py` changed, `0f56c579b29f…` → `c1f3f0f5ec98…`. `manifest.json` is regenerated, and the `pre` hashes are unchanged.

**Tests:**
- `gpu_unit_pruned_embed.py` section 3 now runs under `torch.inference_mode()`, the served grad mode (:152). The overlay no longer needs it.
- New CPU test `test_pruned_mirror_cpu.py::test_lookup_with_requires_grad_parameter`, 6 cases: fp16 and bf16, each under grad, no_grad and inference_mode.
  - Each case uses an `nn.Parameter` table and calls `stage_rows` into a plain buffer (the exact op the CUDA path runs), `lookup` with an out-of-set batch, and `lookup` in-set.
  - The results must be bit-exact and must not require grad.
  - Verified to catch the bug: with the old undetached `index_select(out=)` temporarily restored, the two grad-mode cases fail.
- `test_embedding_module_cpu.make()` now builds the table as `nn.Parameter(w)` with `requires_grad=True`, as `Embedding.load` does.
- Suite: **52 passed.**

## 6. Risks

- **VRAM headroom on cuda:1.** The flag takes about 320 MiB of the about 1,099 MiB free. The daily pool was sized with that headroom. The c4 30k stress pass in the spec is the gate. If headroom is short, trim the pool (about 20k tokens at 15.75 KiB/token) or combine with Group M int8 mixers (+258 MiB on cuda:1, R499; that needs a quality gate).
- **Head bookkeeping has two sources.** The keep set derives from the `_pruned_head` cache. If a future change prunes the head differently, for example with a non-prefix hot-vocab head, `draft_head_keep_ids` must follow it. The mirror already supports arbitrary keep sets through the remap. An argmax ID outside the mirror would raise a device-side index assert and crash loudly; it cannot silently mis-embed.
- **In-process arm switching.** The GPU chain script switches arms by setting module globals, which the code reads at call time, and builds a fresh Generator per arm. The served A/B is the env-driven confirmation.
- **Host checks at position 0.** They add two tiny host tensor reductions (min/max) per step. The microbenchmark reports them.
- **Out-of-set position 0.** It costs one host `index_select` of ≤16 rows and one ≤80 KiB async upload. The count is reported. It is expected to be low per step, but it is not zero: special tokens, CJK, and every turn's first reasoning token.
- **Vision jobs** keep the host chain. That is unchanged from r4 (generator.py:870-871).
