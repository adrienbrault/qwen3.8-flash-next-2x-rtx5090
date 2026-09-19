# Impl status: NVMe prefix tier, round 3 (+ 3b fixes after the R526 box run, + round 4 packaging for the served chain)

Status: round 3b passed R526 try 5 on `stack-r4-e3r2` + ple-ckpt-clone r1. Round 4 packages the same tier for the served image chain (`mtp-pruned-r1` → `tool-choice-r1` → `ple-ckpt-clone-r1` = `tabbyapi:mtp-pruned-r1-tc1-plefix`). It is **verified on CPU and awaits the box A/B** (`box-ab-spec.md` §R4). Tier behaviour and on-disk format are unchanged.

## R4. Round 4: the tier on the served chain

### R4.1 What differs between the stack image and the served chain, per tier file

| tier file | stack-r4-e3r2 | served chain (`mtp-pruned-r1-tc1-plefix`) | handling |
|---|---|---|---|
| `generator/generator.py` | `c8692d99…` | `3fbac394…` (mtp-pruned-r1) | new variant, merged generator in `overlay-pruned/` |
| `generator/pagetable.py` | `0cde876b…` | same | shared `overlay/` file |
| `generator/async_generator.py` | `815efe97…` | same | shared `overlay/` file |
| `cache/recurrent.py` | `5bdc3e1e…` | same | shared `overlay/` file |
| `generator/disk_cache.py` | absent | absent | shared `overlay/` file (new) |

- mtp-pruned-r1 replaces `generator.py`, `architecture/qwen4_exp_mtp.py` and `modules/embedding.py`, and adds `modules/embedding_pruned.py` (its `manifest.json`). Only `generator.py` is a tier file.
- tool-choice-r1 changes TabbyAPI files only. ple-ckpt-clone r1 changes `modules/ple.py` only. Neither touches a tier file, so every descendant of mtp-pruned-r1 has the same five baselines.
- The soft dependencies (`constants.py`, `mm_embedding.py`, `job.py`, `cpu_cache.py`, `quant.py`, `qsa.py`, `gated_delta_net.py`) are not touched by any of the three rounds.
- Four of the five installed files are byte-identical to what R526 try 5 ran: `pagetable.py`, `async_generator.py`, `recurrent.py`, `disk_cache.py`. `tools/make_package.py` regenerates `overlay/` and `overlay-tip/` byte-identical to round 3b (checked with `diff -r`; the two `.patch` files differ only in their timestamp lines).

### R4.2 The merge

- `tools/port_generator.py` is unchanged. It applies the same five anchored hunks to mtp-pruned-r1's `generator.py`, and every anchor occurs exactly once there. No hunk touches a line that mtp-pruned-r1 changed. The nearest one is the flag block: its anchor is `class Generator:`, and mtp-pruned-r1 put `_EMBED_GPU_PRUNED` right before that class, so the tier flag now follows the pruned flag.
- `tools/make_package.py` gains the third variant `stack-r4-e3r2+mtp-pruned-r1`. It writes `overlay-pruned/exllamav3/generator/generator.py` and `served-source-pruned.patch`, a diff against mtp-pruned-r1's `generator.py` that applies with `patch -p1 --fuzz=0` and reproduces `overlay-pruned/` byte-identically.
  - pre: the stack's baselines, except `generator.py` = `3fbac394…`.
  - `present` and `depends`: mtp-pruned-r1's other three files. They must exist, and a different version is reported, not refused.
- The tier construction still runs where it did, before the draft-mode setup. `_prepare_pruned_embed()` still runs at the end of `__init__`. The tier allocates pinned host slabs and CUDA streams/events only, no VRAM tensors, so the boot-time free VRAM the pruned mirror sees is unchanged.

### R4.3 The PLE requirement

- `manifest.json` has a new top-level `requires`: `exllamav3/modules/ple.py` must contain ple-ckpt-clone r1's fixed `stash` return line (`self.id_state[slot, :self.ctx].clone()`).
- `install.py` checks it first, on every variant, including `--check` and the already-installed no-op. It exits non-zero naming ple-ckpt-clone r1, and nothing is copied.
- The check is by text, not by hash, so a later `ple.py` that keeps the fix installs. It then prints `note: not the tested file <sha>`; the tested file is `cbd597f6…`.
- `tests/in_image_smoke.py` checks the same line in the installed tree, so the build fails even if `install.py` were bypassed.
- **Consequence:** the default `BASE=tabbyapi:stack-r4-e3r2` no longer builds. It fails at `install.py` by design, because plain stack has the unfixed `ple.py`. The build order is now PLE fix first, then this overlay. R526 did the reverse (tier, then plefix); the files are disjoint, so the result is the same.

### R4.4 Variant detection

- Detection is by the installed files' hashes (`pre` sets), never the tag. The three `generator.py` baselines are distinct: `c8692d99…` stack, `49b8e85d…` tip, `3fbac394…` pruned.
- `in_image_smoke.py` names the variant it finds: `recurrent_tip` importable → tip; `_EMBED_GPU_PRUNED = ` in the installed `generator.py` → pruned, which also requires `embedding_pruned.py` next to it. Tip and pruned together is not a tested combination and fails the smoke check. `install.py` refuses it anyway ("matches no baseline").

### R4.5 CPU tests (all pass)

| suite | result |
|---|---|
| base (`test_store`, `test_adapter`, `test_install`, `test_gpu_script`, `test_merge_pruned`) | **57 passed, 3 skipped** (stacked-only), was 46 + 3 |
| stacked (`NVME_STACKED=1`, store + adapter) | **39/39** |
| tip r1's own suite on our files (`run_tip_suite_stacked.sh`) | **39 + 3** |
| mtp-pruned-r1's own suite on the merged `generator.py` (inside `test_merge_pruned.py`) | **50/50** (its 52 minus the 2 packaging tests that pin its own manifest hashes) |

New in `tests/test_merge_pruned.py` (4):
- `test_pruned_port_has_the_stack_ports_hunks_only`: both ports are insert-only diffs, with the same five inserted blocks before the same lines, and after the same lines except the flag block (R4.2).
- `test_three_way_merge_equals_overlay_pruned`: `git merge-file -p pruned stack stack+tier` exits 0 (no conflicts) and equals `overlay-pruned/` byte for byte.
- `test_tier_off_ast_equals_pruned`: removing the tier's six statements from the merged AST gives mtp-pruned-r1's AST exactly (`ast.dump`). The six are the flag, `self.disk_page_cache = None`, the `if _NVME_TIER:` install block, and the pump / `on_busy` / `on_idle` guards, each matched by shape. No other name in the module refers to the tier. With `EXL3_NVME_TIER` unset, the only differences are a module global `""`, an attribute `None`, and three `is not None` tests that are False.
- `test_pruned_rounds_own_cpu_suite_on_the_merged_generator`: runs mtp-pruned-r1's drafting-chain, pruned-mirror, keep-ids and embedding tests with the merged file in place of its own.

New or extended in `tests/test_install.py`:
- Every simulated base now gets ple-ckpt-clone r1 through its own `fix_ple.py`.
- `test_install_on_pruned_base`: detects `stack-r4-e3r2+mtp-pruned-r1`; `--check` reports the tested `ple.py`; installs 5 files; mtp-pruned-r1's own three files are untouched; a second run is a no-op.
- `test_refuses_without_ple_fix[stack|tip|pruned]`: `--check` and install both refuse, and nothing is copied.
- `test_ple_fix_of_another_shape_is_accepted_with_a_note`.
- `test_pruned_variant_is_not_picked_from_a_tag`: a drifted pruned `generator.py` matches no baseline.
- `test_in_image_smoke_on_installed_tree[stack|tip|pruned]`.
- The port test covers the pruned pair with the same exact list of added statements. The baseline test checks that the pruned `pre` equals the stack's except `generator.py`, which equals mtp-pruned-r1's post hash, and that mtp-pruned-r1 replaces no other tier file.

`test_store.py` / `test_adapter.py` load `pagetable.py`, `recurrent.py` and `disk_cache.py` over `src/`, never `generator.py`. Those files are identical on the pruned variant, so their 46 + 39 results already cover it; there is no separate "pruned" harness mode.

Run commands (from `out/nvme-tier-r4/tests`, `PY=../../../.venv/bin/python`):

```
$PY -m pytest -q test_store.py test_adapter.py test_install.py test_gpu_script.py test_merge_pruned.py   # 57 passed, 3 skipped
NVME_STACKED=1 $PY -m pytest -q test_store.py test_adapter.py                                          # 39 passed
PY=$PWD/$PY bash run_tip_suite_stacked.sh                                                              # 39 + 3 passed
../../../.venv/bin/python ../tools/make_package.py    # regenerates all three overlays, the manifest and the patches
```

### R4.6 Other changes

- `launcher-nvme-tier.patch` was regenerated against the current served launcher (repo `flan/launch-flashnext-tabby.sh`, R530: `IMG` default `mtp-pruned-r1-tc1-plefix`, `EXTRA_ENV` with `EXL3_MTP_DEVICE_DRAFT=1 EXL3_EMBED_GPU=1 EXL3_EMBED_GPU_PRUNED=1`). It applies with `--fuzz=0`. The R525-based one needed `--fuzz=1`. Its content is unchanged.
- `Dockerfile.box`: `/opt/nvme-tier-r4/`, copies `overlay-pruned/`, and has the build commands in its header. `ARG BASE` keeps its default.
- `manifest.json`: `round` `nvme-tier-r4`, `base_images` (informational) in place of `base_image`, the third variant, `requires`.

### R4.7 Risks specific to round 4

1. **The tier has never run with the device draft chain on.** R526 stripped `EXL3_MTP_DEVICE_DRAFT` / `EXL3_EMBED_GPU` / `EXL3_EMBED_GPU_PRUNED` so that its OFF and ON boots matched. The chain changes where draft ids live, not the draft cache contents (R522/R528: bit-equal rows, fingerprints unchanged), and the tier copies only finished pages. Still, a disk restore followed by device-chain drafting is a new path. The box verify step covers it: the restored output hash must equal the warm hash.
2. **Pump cost on top of the device chain.** The device chain removed host syncs from drafting. The pump's copies go on the compute stream, budgeted at 1 % of iteration time. Decode during drain is measured again.
3. **Fresh namespace.** `engine_identity` hashes every installed `.py` plus the `EXL3_*` flags, so the served-chain image cannot reuse R526's tier directories. Use fresh ones.
4. **The default build now fails.** See R4.3. Any unit that builds `Dockerfile.box` with no `BASE` (as R526's script does) needs `--build-arg BASE=<plefix image>`.
5. **The tip variant also needs a PLE-fixed tip image.** It is unchanged otherwise, and it is not on the served chain.

---


## 0. R526 box findings and the round-3b fixes

R526 passed:
- tier-OFF vs tier-ON fingerprints;
- fill;
- drain (692 pages, 12 checkpoints);
- crash restart (same namespace, 0 torn);
- idle decode;
- churn at a 2 GiB cap.

It failed three items.

### 0.1 FAIL 1, restore never hits: what the log proves

After the restart both verify requests ran cold (`restored 0 pages + 0 ckpts`). Their drained line then said `692 pages, 12 ckpts; wrote 0 pages + 2 ckpts (0.22 GiB)`. Two things follow from that line.

- **Page hashes are identical across the two processes.** The two new checkpoints (one per request) needed **0** new pages, so every page of both chains was already on disk under the key the new process computed. This rules out a per-process hash, `PYTHONHASHSEED`, BOS/tokenization differences, the admission rule and the multimodal floor.
- **Two stored checkpoints left the index.** The count stayed at 12 after two writes. At 4.5/128 GiB nothing evicts, and prune/TTL do not run. The only path that removes a checkpoint there is the read quarantine: `_read_record` drops a record whose read fails twice at the same location. `prepare_resume` had found those two checkpoints (the deepest one of each prompt), tried to load them, and the read failed. The cold prefill then stashed the same key again, and it was written anew. That explains the "+2 ckpts, count unchanged".

### 0.2 FAIL 1, root cause: the PLE id context is stashed as a live view (engine bug, file:line in `src/exllamav3/modules/ple.py`)

- `PLELayerState.alloc` keeps the carried token-id context on the host: `self.id_state = torch.full_like(..., device = "cpu")` (ple.py:74-77).
- `PLELayerState.stash` returns `self.id_state[slot, :self.ctx].cpu()` (ple.py:104). On a CPU tensor `.cpu()` is the tensor itself, so the stash holds a **view of the live slot**.
- Every later forward on that slot overwrites it in place: `id_state[s, :ctx].copy_(history[i, -ctx:])` (ple.py:386; the rewind at ple.py:93-100 does the same).
- The `RecurrentCache` entry therefore does not hold the checkpoint-time context. It holds whatever the slot holds now, which after the job ends may be another conversation's tokens.

How this broke the tier:
- The writer serialized the stash later, on its own thread. It hashed the ~111 MB payload with the GIL released, and meanwhile the next forward rewrote the 8×ctx id bytes. Then it wrote the parts. The digest in the header no longer matched the bytes on disk, so every read failed its digest check twice and the record was quarantined.
- Older segments are only header-checked at open, which is why the restart looked clean (0 torn).
- Even without the race, the persisted context would have been the wrong one.

Evidence: a CPU check shows that `.cpu()` of a CPU slice shares storage with the slot and follows later writes. `test_live_view_in_stash_is_snapshotted_at_put[False]` reproduces the wrong persisted content with an aliased stash.

This also affects the tier-OFF daily. Warm RAM-checkpoint hits restore the slot's current id context, not the checkpoint's. This is probably part of why warm revisits differ from cold (R511 attributed it to the MTP zero carry), and it is the same class as the PLE issue behind the tip round's R524 fix. The upstream one-line fix is `self.id_state[slot, :self.ctx].clone()` at ple.py:104. It changes warm-hit outputs on the default path, so it is not in this overlay; it needs its own A/B.

### 0.3 Fixes (file:line in `overlay/exllamav3/`)

- **Snapshot at put.** Under the tier, `RecurrentCache.put` calls `disk_tier.own_stash(state, stash)` right after `state.stash()` (cache/recurrent.py:61-64).
  - `own_stash_tensors` (generator/disk_cache.py:1231) replaces every CPU tensor that is a view (`_base is not None`) with a clone. Here that is the PLE id context, ctx×8 bytes per PLE layer.
  - RAM and disk both hold the checkpoint-time state. Tensors that own their storage are untouched: GPU `.cpu()` copies, the tip round's snapshots, and deserialized checkpoints (`_StashedState.tier_owned`).
  - This changes tier-ON warm-hit outputs (they become correct). Cold fingerprints are unaffected.
- **The writer can no longer hash one thing and write another.**
  - `_snapshot_checkpoint` (disk_cache.py:2202) serializes into one tier-owned buffer, then hashes the source again. A difference is logged as `checkpoint … changed while it was being serialized` and the checkpoint is not stored.
  - The record is then read back from the file (`verify_record` :962 via `_verified` :769). A mismatch is logged and the record is dropped.
  - Pages are also read back with `EXL3_NVME_TIER_VERIFY=2`.
- **The box can see why a restore misses.**
  - `_read_at` (:932) returns the failed check: `open` / `header` / `identity` / `short` / `digest` / `io`. These are counted as `read_fail_*`, and a quarantine prints its reason.
  - `prepare_resume` (:1873) counts every lookup by outcome (`_lookup_outcome` :1934): `hit`, `no-disk-ckpt`, `ram-as-deep`, `in-ram`, `load-read:<check>`, `load-decode:<exc>`, `position a!=b`, `put-dropped`. It prints one line per lookup that found prompt pages on disk.
  - The drained/summary line gains `lookups …`, `integrity: read-fail …, quarantined, write-verify, stash-mutated, views-copied`, `pump N pages (ms/page)` and `last miss`.
  - An **open scan** (`_scan_checkpoints` :2174, background thread, `EXL3_NVME_TIER_SCAN=1`) reads and checks every stored checkpoint once at boot and prints `open scan: M/N checkpoints intact`.
- **Restore abort guard (FAIL 2 request).**
  - When a disk page cannot be restored mid-allocation, `PageTable.allocate_pages` calls `disk_tier.abort_restore(disk_ops)` (pagetable.py:578, :624, :629; disk_cache.py:1962).
  - Every page already copied from disk in that allocation goes back to the state of a freshly claimed page (`kv_position 0`, no link). The checkpoint loaded from disk for it leaves the RAM cache, and a line is logged.
  - The job then resumes from what VRAM/RAM alone hold. Before, the pages restored up to the failure stayed complete: they were valid bit-exact copies, but "half restored".
  - A failed checkpoint load never changed anything, and still does not.

### 0.4 FAIL 2, 30k after the restart generated one EOS token: no tier mechanism found

- That job was a cold prefill. `restored 0 pages + 0 ckpts`, so no H2D copy was issued and no disk checkpoint was put into RAM.
- The failed load changed nothing: it dropped a disk index entry and nothing on the GPU.
- The slot's recurrent and PLE state were cleared by `get_new_state()` (`clear(slot)` fills the PLE id context with EOS, ple.py:85-88).
- The only tier activity during the job was the writer serializing two new checkpoints on the host. That is a read-only use of CPU tensors.
- The PLE live-view bug corrupts only restored state (RAM or disk), never a cold job.
- So I cannot attribute the EOS to the tier. E3 prefill is not bitwise deterministic (R524), and a raw completion ending in a question can put EOS close to "

".
- `gpu_nvme_ab.py first-token` records the first-token top-5 logprobs of R526's own 30k prompt, cold with the tier on and cold on the tier-off daily. A small top-2 margin settles it as a near-tie.
- The requested guard is in place (0.3, abort guard).

### 0.5 FAIL 3, decode −7.3 % at c1 while the pump drained a 120k chain: budget the pump by measured time

- **Cause.** The cost is mostly host time on the generator thread: about 30-40 `cudaMemcpyAsync` launches per page (one per cache tensor), plus about 0.2 ms of D2H per 4.47 MiB page on the compute stream. Four pages per ~14 ms c1 iteration is about 1 ms, which is 7 %.
- **Why not a side stream.** It would take the D2H off the compute stream, but not the launch cost. It would also need a new fence at every point where a complete page can be rewritten.
- **Change.** `pump()` (:1733) now earns `EXL3_NVME_TIER_PUMP_PCT` (default 1 %) of each iteration's measured duration, capped at `PUMP_PAGES` pages' worth, and reset when idle. Each page is charged its measured launch time plus its estimated D2H time (EMA). At c1 this is about one page per three iterations (~1 %).
- **Under sustained load** the pump still makes progress: a 120k chain takes about 20 s of continuous decode. When idle, the drain does the bulk, as before.
- **Off switches.** `EXL3_NVME_TIER_PUMP_PAGES=0` still disables it. The counters `pump_copies` and `pump_seconds` and the per-page cost are in the summary line.

### 0.6 CPU tests added (all pass)

- `test_restore_in_fresh_process_with_another_hash_seed`: writes in one subprocess (`PYTHONHASHSEED=11`) and restores in a fresh one (`PYTHONHASHSEED=12345`). All 60 pages and the checkpoint come back bit-exact, and the summary shows `lookups 1: hit 1`.
- `test_live_view_in_stash_is_snapshotted_at_put[True|False]`: the PLE-style aliased stash. The writer is held until after the "next forward". `True`: RAM and disk hold the checkpoint-time ids. `False` is the control and reproduces R526's wrong content.
- `test_checkpoint_read_failure_is_reported_and_job_runs_cold`: `load-read:digest`, quarantine, a cold job, and a re-persist.
- `test_open_scan_reports_and_quarantines_bad_checkpoints`.
- `test_write_verify_failure_drops_the_record`.
- `test_pump_is_time_budgeted_and_zero_disables_it`.
- `test_corrupt_page_mid_chain_falls_back_without_error` now also asserts the abort guard: `restore_aborts 1` and the digest reason.
- `test_gpu_script.py` covers the new `first-token` subcommand.

Totals, round 3b:

| suite | result |
|---|---|
| base | **46 passed, 3 skipped** (stacked-only) |
| stacked (`NVME_STACKED=1`) | **39/39** |
| tip r1's own suite on our files | 39 + 3 |

`served-source.patch` applies to a fresh `src/` copy (exit 0) and is byte-identical to `overlay/`.

---

Round 3 status as delivered before R526 follows, unchanged except where §0 supersedes it. Round 3b moved `disk_cache.py` line numbers; §0 gives the current ones.

Round 3b risks, in addition to §7:

1. **Tier-ON warm hits differ from tier-OFF warm hits** because of the PLE snapshot. The tier-ON ones are the correct ones; cold fingerprints are unchanged. Anyone comparing warm outputs across arms must know this.
2. **The open scan reads every stored checkpoint once per boot, in the background.** At the 128 GiB cap that could be tens of GB (about 10 s at NVMe speed), competing with early restores. `EXL3_NVME_TIER_SCAN=0` turns it off once the box has shown the checkpoints are sound.
3. **Checkpoint writes now cost more writer time:** one extra ~111 MB copy, one extra hash and a read-back. This is off the generator thread, and the drain takes about 0.2 s longer per checkpoint.
4. **The pump's page-cost estimate starts at 0.3 ms plus the D2H estimate** and then follows measured launch time. Its D2H share is an estimate (20 GB/s), not a measurement.

The overlay, both manifests, `install.py`, `Dockerfile.box`, `served-source.patch`, the CPU tests (all passing), the GPU script, the launcher patch and the box spec are all under `out/nvme-tier-r3/`. The things only the box can show are listed below.

## 1. Review of round 2

Line numbers refer to `ref/nvme-tier-r2/overlay/exllamav3/generator/disk_cache.py` (DC) and the r2 `pagetable.py` (PT) unless noted otherwise.

### Wrong (would fail or crash on the box)

| # | finding | evidence |
|---|---------|----------|
| W1 | **The first full write queue raises `KeyError` on the generator thread.** `metrics["write_queue_skips"]` is incremented, but the key is not in the metrics dict literal. | increments DC:1260, DC:1328; dict DC:228-251 |
| W2 | **A writer error kills the generator.** `store_page` and `persist_checkpoint` re-raise `_writer_error` on the calling thread. `persist_checkpoint` runs inside `RecurrentCache.put` ← `job.maybe_stash_recurrent` ← `recurrent_checkpoint()`, and nothing on that path catches the exception. `AsyncGenerator._run_iteration` then fails every job. | DC:1249, DC:1308, DC:1322; `async_generator.py:47-56` |
| W3 | **The eviction path blocks the generator thread** in `self._pool.acquire()`, deliberately: the comment reads "Eviction must not skip". | DC:1250-1253 |
| W4 | **Every idle transition blocks the generator thread** until the writer's queue is empty. `prune_stranded()` → `self.flush()` (`queue.join()`) runs from `on_queue_drained`. | DC:1436-1437 |
| W5 | **Compaction holds `self.lock` for the whole segment rewrite**, and the generator thread takes that lock in `should_store_page` / `admit_chain` / `has_checkpoint`. So a 256 MB forward stalls token generation, which contradicts brief-r2 point 3. Compaction also raises inside the writer if the cap would be exceeded. | DC:557-611 (raise at :574, :584) |
| W6 | **Host-memory leak on every job start with a disk anchor.** `prefetch_pages` loads a ~115 MB checkpoint into `_cp_prefetch`, but it is consumed only on a RAM miss (PT:301-306). A RAM hit, which is the common case, leaves it in the dict indefinitely. | DC:1343-1359, DC:1428-1431; PT:274-279 |
| W7 | **Slab deadlock.** Page prefetch futures hold staging slabs. They are consumed only if the job actually restores that page from disk, which does not happen when the page is in VRAM, when MTP cuts the last page, or when the job is cancelled or still pending. With ≤ 8 slabs the pool empties, `pump` stalls, and `fetch` → `acquire()` waits forever. `job.py:1162` also prefetches at job preparation. | DC:1343-1375, DC:1399; r2 `job.py:1162` |
| W8 | **The idle drain does not finish a chain.** `on_queue_drained` calls `pump(256)`, but each copied page holds a slab until the writer finishes it. So one call copies about 8 pages and stops, and nothing pumps again while the server is idle. A single request followed by `docker rm -f` leaves the chain incomplete, and `prune_stranded` drops it at the next boot. **The restart gate would most likely fail.** | r2 generator hunk `on_queue_drained` :671-684; DC:1218-1238 |
| W9 | **The packaging cannot build on the image.** `install.py` looks for a `tabbyapi/` package on `sys.path`. The image serves TabbyAPI from `/app` (Dockerfile `WORKDIR /app`, `COPY . .`, `py-modules = []`), and there is no `tabbyapi` package, so the installer exits with "packages not found". The "simulated image layout" test in impl-status-r2 was built to match the installer, not the image. | r2 `overlay/install.py:36-72`; `src/app` layout |
| W10 | **Multimodal chains are persisted.** Image placeholder token ids come from a per-process counter that starts at 1e9. After a restart the same ids stand for other images, so a page hash can alias different K/V. | `tokenizer/mm_embedding.py:4-21`; no filter in DC `persist_checkpoint` |
| W11 | **Interior (superseded) checkpoints are never evicted**, only leaves via `_choose_leaf_key`. A long agent session leaves one checkpoint per 32k prefill interval plus one per 2,048 decode tokens, and the cap fills with dead ~115 MB checkpoints. | `_choose_leaf_key` DC:711; `prune_one_leaf`/eviction only take leaves |
| W12 | **`_expire` pops checkpoints without updating `_seg_live` / `_live_bytes`.** Reclaim then never sees those bytes as dead. | DC:812-822 |

### Unproven or weak

| # | finding | evidence |
|---|---------|----------|
| U1 | Reclaim picks a segment only when its live fraction is below 0.5, with no forced path. With evenly spread evictions, about half the cache can be evicted before any segment qualifies, so `_ensure_room` evicts far more than it needs. | `_ensure_room` DC:520-556, `_reclaim_dirty` DC:613-625 |
| U2 | Boot reads and digests **every payload** of every segment (up to 128 GiB at the default cap) before the server comes up. | DC `_rebuild` :323-392 |
| U3 | `has_checkpoint` is called per page while `Sequence.allocate_pages` scans the prefix, so job start is O(pages²) under the lock. | PT:283 |
| U4 | A compaction that forwards an entry updates it in place. The reader's `fresh is entry` retry test then sees the same object, so the retry is skipped and the result is a miss. | in-place update DC:595-598; retry tests DC:867, :886, :906 |
| U5 | Checkpoints use `torch.save` / `torch.load(weights_only=False)`: pickle on a writable cache directory, plus `b"".join` copies that hold the GIL. | DC:1284, DC:1426 |
| U6 | The r2 `get_cache_stats` hunk iterates the tier's dicts on the API thread while the writer mutates them (`RuntimeError: dictionary changed size`). | r2 `generator.py:451-460` |
| U7 | The dedicated-filesystem check compares against `/`. Inside the container that is the overlayfs root, so any bind mount passes. | DC:1069 |
| U8 | The adapter tests drive a fake pagetable, not the real `PageTable` / `Sequence` / `RecurrentCache`. None of W6-W8 could show up there. | r2 `tests/test_adapter_local.py` |

### Correct, and kept

- Namespace identity without mtime: size + sampled bytes + config.json in full.
- Checksummed 128-byte headers, torn-tail truncation, and the single-owner flock.
- Admission ≥ 32 pages, the placeholder-hash refusal, and target + draft page images in one slab.
- Reads outside the lock via `preadv` into pinned slabs, and `posix_fadvise(DONTNEED)`.

## 2. Rebase onto `src/` (stack-r4-e3r2)

- Of the files the tier touches, only `generator/generator.py` differs between `ref/r2-base-src` and `src`: 234 changed lines, from decode round 4 (draft tables, batch verify, device draft) and the prefill pipeline.
- `pagetable.py`, `recurrent.py`, `job.py`, `async_generator.py` and the TabbyAPI files are byte-identical.
- The tier's generator hunks do not overlap the round-4 hunks. They are applied by `tools/port_generator.py`, which uses exact anchors that must each occur once. The same anchors apply to tip r1's `generator.py`.

## 3. What changed (file:line in `overlay/exllamav3/`)

### `generator/disk_cache.py` (new file, 2,040 lines)

Round 2's storage core is kept; the following was fixed or added.

**`SegmentStore` (:177)**
- **Locking.** The lock is never held across I/O (docstring :181-190).
- **Room.** `_ensure_room` :552 evicts superseded interior checkpoints first (`_superseded_checkpoints` :604, `_victims` :620). It reclaims at ≤ 0.5 live and forces reclaim at ≤ 0.75 live, so rewrite is at most 3x (:68-69, `_pick_segment` :588).
- **Compaction.** `_compact_forward` :643 snapshots under the lock, copies outside it with `copy_file_range`, fsyncs, commits under the lock, then unlinks.
- **Accounting.** `_drop_page_entry` / `_drop_checkpoint_entry` :521-551 keep the accounting exact, `_expire` included.
- **Open.** The cap is enforced at open (:268-270). `_rebuild` :337 checks every header but digests only the newest segment's payloads.
- **Reads.** `_read_record` :940 retries at the moved location and quarantines a record after two failures.
- `metrics` is a `defaultdict`.

**Identity and serialization**
- `engine_identity` :1062: the sha of the installed package's `.py` / `.cu` / `.cuh` / `.cpp` / `.h` sources, plus `EXL3_*` flags except `EXL3_NVME_TIER*`.
- Checkpoints: `serialize_stash` / `deserialize_stash` :1124-1195 (RCP1, no pickle) and `_StashedState` :1195.

**Pools**
- `SlabPool.try_acquire` :1263.
- `_RestoreBatch` :1288 holds bounded parallel reads that live only within one `allocate_pages`.

**`DiskPageCache` (:1370)**
- `install` :1382 prints ON or disabled and never raises.
- `pump` :1634, the non-blocking eviction hook `store` :1662, and `_chain_mm_free` :1676.
- `persist_checkpoint` :1701 skips instead of raising.
- `on_checkpoint_evicted` :1738.
- `prepare_resume` :1748 loads a checkpoint only if it is deeper than RAM.
- `begin_restore` / `end_restore` :1785-1807.
- The idle drain (`on_idle` / `on_busy` / `_drain_step` / `_drainer_main` :1844-1943) waits on the GPU, not the host, and ends with the "drained" line.
- `_writer_main` :1944 counts errors and disables writes after 8 consecutive failures.

### `generator/pagetable.py`

All changes are additive and guarded by `disk_tier is not None`.
- `prepare_resume` :283-287 and the attribute :344.
- Eviction store :525-526.
- Restore batch :578-629.
- `is_resumable` / audit chain walks :702-703, :732-733, :765-766.
- `chain_to_root` :714.
- r2's per-page `has_checkpoint` and the prefetch at job start are gone. `job.py` is not touched.

### `cache/recurrent.py`
- The attribute at :35.
- `on_evict` from the LRU loop :86.
- The persist hook after `put` :96-98.
- The `on_evict` seam :101-107, shared with tip r1's `tip_drop`.

### `generator/async_generator.py`
- `close()` closes the tier (:115-117).

### `generator/generator.py` (both variants)
- `_NVME_TIER` :42.
- Install :308-311.
- `pump` :588-589.
- `on_busy` :646-647 and `on_idle` :657-658.
- The tip variant (`overlay-tip/`) has the same hunks at :45, :311-314, :595, :653, :664.

### Deliberate deviation from brief-r2 point 7

- No TabbyAPI config keys and no status-line counters. The knob is `EXL3_NVME_TIER=<dir>` (plus optional `EXL3_NVME_TIER_GB`, `_MIN_FREE_PCT`, `_SEGMENT_MB`, `_STAGING_MB`, `_MIN_PAGES`, `_TTL_HOURS`, `_PUMP_PAGES`, `_LOG_SECS`), set through the launcher like every other stack flag.
- Reason: TabbyAPI lives in `/app` and is not an installable package (W9). Leaving it untouched keeps the install to one pinned exllamav3 tree.
- Instead of the counters there are periodic ` -- nvme tier: ...` summary lines.

### Packaging

- `install.py` finds the package via `find_spec` (no import). It accepts exactly one of two pinned baselines, verifies the post-install hashes, removes stale `.pyc` files, and is a no-op when already installed.
- `manifest.json` has two variants: `stack-r4-e3r2` and `stack-r4-e3r2+recurrent-tip-r1`. The tip variant pins tip's `generator.py` by hash, because this round replaces that file. `recurrent_tip.py` only has to be present; a different version is reported, not refused.
- `Dockerfile.box` (legacy builder, no heredocs) runs `tests/in_image_smoke.py`.
- `launcher-nvme-tier.patch` adds `NVME_TIER` / `NVME_TIER_GB`.
- `served-source.patch` is `diff -ruN` against `src/`. `served-source-tip.patch` is the tip generator → ported tip generator.
- `tools/make_package.py` regenerates all of this from `work/`.

## 4. Stacking with recurrent-tip-r1

- **Install order:** tip r1's own installer first, then ours. Our `install.py` recognises the tip baseline by hashes: tip's `generator.py` and `recurrent_tip.py` present, the rest equal to `src`. It then installs `overlay-tip/…/generator.py` in place of the base-variant generator.
- **Tip checkpoints at commit:** `put_tip` → `TipPolicyMixin.put` → base `RecurrentCache.put` → `persist_checkpoint`.
- **Tip-policy evictions** go through `tip_drop` → `on_evict` → `on_checkpoint_evicted`.
- **Restore:** a restart restores through the tip-aware cache: `prepare_resume` puts a `_StashedState`, which has `.position` / `.checkpoint_size` like `_HostSnapshot`.
- **Snapshot buffers:** tip r1's pinned snapshot buffers are never handed to the tier. `_build_stashed` makes fresh pageable copies, so the writer serializes immutable tensors.

## 5. Verified locally (CPU, torch 2.14 CPU wheel, Python 3.11, `.venv/`)

The tests drive the **real** `PageTable` / `Sequence` / `RecurrentCache` (and `TipPolicyMixin` when stacked) against CPU fake caches. Page and state contents are deterministic per content hash, so every restore is checked bit for bit.

- `tests/test_store.py`: **15/15**. Covers:
  - round trip and reopen
  - torn tail, and a same-length torn payload
  - quarantine
  - header-only open
  - kill -9 of a writer subprocess mid-append
  - random churn at a small cap (`du` ≤ cap + 4 KiB after every op, rewrite ≤ 3x reclaim)
  - interior checkpoints evicted first
  - readers not blocked by a stalled compaction, and a read racing a compaction
  - lock + namespace GC
  - the free-space floor
  - mtime-insensitive model identity, and engine identity
  - a lowered cap applied at open
- `tests/test_adapter.py`: **14/14** (+3 stacked-only, skipped in the base run). Covers:
  - tier off is inert
  - proactive copy + idle drain + bit-exact restart
  - SIGKILL right after the drained line (subprocess), then restore
  - the eviction path under VRAM pressure
  - a deeper disk checkpoint takes precedence over RAM
  - a corrupt mid-chain page falls back
  - an open restore batch is released
  - a multimodal chain is never persisted
  - placeholder and short chains are refused
  - the drain stops on busy
  - **the generator thread makes no file call** (instrumented `os.*` + `open`)
  - writer errors disable writes without raising
  - the cap holds through adapter churn
  - the checkpoint payload is bit-exact and not pickle
- `NVME_STACKED=1` (overlay-tip over overlay over tip r1 over src), `test_store.py` + `test_adapter.py`: **32/32**. Includes tip `put` persisted, tip-policy eviction persisted via `on_evict`, and restart restore through the tip-aware cache.
- `tests/test_install.py`: **8/8**. Covers:
  - install on the served base
  - install on base + tip (tip's installer, then ours)
  - idempotence
  - refusal of a drifted baseline
  - the generator port adds lines only, and the exact list of added statements
  - the other files add lines only, and the manifest baselines equal `src/`
  - `in_image_smoke.py` against an installed simulated site-packages tree, on the served base and on base + tip
  - a later `recurrent_tip.py` is reported, not refused; a missing one is refused
- `tests/test_gpu_script.py`: **2/2**. A dry run of every `gpu_nvme_ab.py` subcommand (fill, wait-drained, verify, decode, churn) against a stub TabbyAPI that mimics the real response shapes (usage only with `stream_options.include_usage`), with stub docker/du commands; plus a verify FAIL on a hash change.
- `tests/run_tip_suite_stacked.sh`: tip r1's own suite against our pagetable / recurrent / disk_cache / async_generator: **39 + 3 passed**.
- `served-source.patch` applies to a fresh copy of `src/` with `patch -p1 --fuzz=0` (exit 0), and the result is byte-identical to `overlay/`. `src/` itself is untouched; its hashes equal the manifest's `pre` baselines.

Run commands (from `out/nvme-tier-r3/tests`, `PY=../../../.venv/bin/python`):

```
$PY -m pytest -q test_store.py test_adapter.py test_install.py test_gpu_script.py   # 39 passed, 3 skipped
NVME_STACKED=1 $PY -m pytest -q test_store.py test_adapter.py
bash run_tip_suite_stacked.sh
```

## 6. What only the box can verify

See `box-ab-spec.md`: 14 minutes, three boots.
- **Fingerprints:** cold c1 and 30k fingerprints unchanged with the tier on.
- **Restart restore:** the 30k and 120k prefixes survive `docker rm -f` + relaunch, restore from disk with the same cached count and the same greedy output as the pre-restart warm request, and have a lower TTFT.
- **Decode cost:** c1/c4 decode within noise, both drained and during the pump.
- **Cap under churn:** `du` stays under a 2 GiB cap during churn.
- **Not testable on CPU:**
  - real CUDA stream and event ordering (pump on the compute stream, side-stream drain, `on_busy` GPU waits)
  - pinned-memory D2H/H2D speed and NVMe throughput
  - the MTP draft cache restore
  - the size and speed of the real 115 MB checkpoint
  - `engine_identity` time over the installed package (printed by the build smoke test)

## 7. Risks

1. **Restore stalls other jobs.** A disk restore runs inside `allocate_pages` on the generator thread, waiting for reader-pool preads. For 120k that is about 2 GB, roughly 0.5-1 s at 3+ GB/s, during which other slots do not decode. It replaces a ~12 s prefill that would itself take the GPU. Making it asynchronous (restore while other jobs step) is a follow-up.
2. **Pump D2H runs on the compute stream.** It adds about 0.7 ms per iteration only while a chain is pending (4 pages x ~4.3 MB). Box step 8 measures it; `EXL3_NVME_TIER_PUMP_PAGES=0` leaves only the idle drain.
3. **Eviction-path copies are best effort.** When VRAM evicts faster than the staging pool (≤ 16 slabs) drains, pages are skipped rather than waited for. A skipped page makes its chain incomplete, and the checkpoint is pruned at the next open. Only a cache-hit rate is lost; correctness is unaffected.
4. **Namespace resets.** Any change to the installed exllamav3 sources or to an `EXL3_*` flag (except the tier's) starts a fresh namespace and deletes the old one at open. Promoting a new image or flipping a kernel flag empties the tier, by design.
5. **The dedicated-filesystem check is weak inside the container** (a bind mount always differs from the container root). The launcher must point `NVME_TIER` at `/srv/qwen5090/fast`. The free-space floor (15 %, measured on the tier's own filesystem) is the hard guard.
6. **One checkpoint load per job.** If the deepest disk checkpoint fails its digest, the job re-prefills from the RAM checkpoint or from zero. It does not fall back to a shallower disk checkpoint.
7. **Tensor-parallel caches are not supported.** The tier prints "disabled" and stays off; the daily is a layer split, not TP.
8. **Decode-built checkpoints** (every 2,048 tokens, and tip r1's tips) are persisted as-is. A restore from one is exactly as bit-identical as the RAM hit it replaces. The tip round's NUMERICS caveat still applies.
9. **Write volume.** Worst case about 0.5 TB/day at 1,000 agent calls; see design.md. Checkpoints dominate for short calls.
10. **The tip r1 snapshot in `ref/` may be older than the tip image.** The operator's log mentions a later tip fix ("accept PLELayerState"), and `ref/recurrent-tip-r1` does not contain it. A fix confined to `recurrent_tip.py` installs with a note. If the fix also changed tip's `generator.py`, `install.py` refuses the stacked build with "matches no baseline"; `make_package.py` then has to be rerun against the new tip `generator.py`, which takes minutes.
11. **Knob deviation.** The knob is an env var, not the TabbyAPI config keys that brief-r2 point 7 asked to keep (reason in section 3).
12. **Box-only unknowns.** Stream ordering on real hardware, and whether the "drained" line lands within the step 4 budget for 120k: about 2 GB, a few seconds at NVMe speed.
