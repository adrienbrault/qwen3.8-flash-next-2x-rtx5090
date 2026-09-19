# mtp-kv-window r1: implementation status

Status: implemented; CPU tests pass (38/38, output below); box A/B specified in `box-ab-spec.md`. Nothing here has run on a GPU.

The anchor follows the coordinator's correction: 8 slots at pool 966,656, cache 8,8, NVMe tier OFF for the A/B. The overlay is pinned to `tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16`. It is not rebased onto the R565 image `tabbyapi:ngram-prefetch-r1-gdnbf16` (see "Stacking").

Deliverables in `out/mtp-kv-window-r1/`:

| file | content |
| --- | --- |
| `served-source.patch` | 6 files, +375 / −10; `patch -p1 --fuzz=0` exit code 0 on a clean copy of `src/exllamav3`. sha256 `cc2253bb4a00023c873bfa83918d9b2632bc6edfe54f3dc4dd2e77c837ba0e87` |
| `overlay/install.py` | baseline hash check (a null baseline means the file must not exist), fuzz-0 apply judged by exit code, overlay hash check, stale bytecode removal |
| `overlay/manifest.json` | baseline and post-patch sha256 per file, patch sha256 |
| `Dockerfile.box` | `ARG BASE=tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16`, no heredocs, ends with the import assertion |
| `tests/` | `test_mtp_kv_window_cpu.py` (pytest), `mtp_window_scenarios.py` (runs the served or the patched tree's own code in a subprocess), `stubs.py` (extension and triton stand-ins) |
| `box-ab-spec.md`, `impl-status.md` | box A/B spec and this file |

There is no C++ or CUDA change, so there is no `build_extension.py`.

## Mechanism

`EXL3_MTP_KV_WINDOW=<tokens>` (a multiple of 256, at least 512; unset or 0 means off).

1. **Allocation.** A `Cache` built for a model whose caps carry `mtp_draft` gets `max_num_tokens = slots × P × 256` instead of the requested pool, where `P = tokens / 256 + 1` pages per slot (one sink page plus `tokens / 256` ring pages) and `slots` is the Cache's own `max_batch_size`.
   - This happens inside `Cache.__init__`, so it applies whoever builds the draft cache. In the served stack that is TabbyAPI (`backends/exllamav3/model.py`), not `model_init.py`.
   - Every other cache is built exactly as served.
2. **Addressing.** Slot `s` owns physical pages `[s·P, (s+1)·P)`. A job's draft block-table row maps logical page `q` to `s·P` when `q == 0`, and to `s·P + 1 + (q−1) mod (P−1)` otherwise.
   - Below one cycle (`P·256` tokens) this is the identity. Beyond it, each new page lands on the ring page that held the page `P−1` before it.
   - Positions stay absolute. The served kernels compute the K rope, the pooled-key rope at block start, causal bounds, raw-key ring rows and pooled rows exactly as before, and every access goes through the block table. No kernel changed.
   - A row never leaves its slot's pages (tested).
3. **What the draft layer sees.**
   - Up to one cycle: exactly the served draft K/V.
   - Beyond it: the sink page plus the last `P−1` pages, including the page being written. At least the last `tokens − 512` positions are exact at all times (checked in `ring_contract`).
   - The oldest ring page can carry up to 16 rows written by rejected draft steps that crossed a page boundary. The row of the page being written, past the committed end, can hold rows from the previous cycle. Both are older real tokens or rejected drafts, and they affect draft proposals only.
4. **QSA sparse scan.**
   - The graph-captured sparse regime scores `t_total / 4` pooled blocks into a static sized to the draft pool's capacity. With the window, the logical length exceeds that capacity, so it would overflow.
   - `BCAttn.step` therefore clamps `t_total` to one ring cycle (`P·256`), after deciding the regime and the batch-size-1 rope position on the true length. One cycle maps every page of the slot exactly once, so each ring page is scored once.
   - The tail block comes from the device cache lengths (true positions).
   - The eager path (prefill chunks; decode fallbacks only when batch size > 8 or the verify length > 16, which the served config never hits) scans the whole logical length through the aliasing table. In prefill the attention output is discarded, because the draft layer's K/V depend only on its input.
5. **Slots.** A job leases a slot on its first draft-cache access, which is always its first draft prefill chunk: MTP prefills at least the last prompt token.
   - It keeps the slot until it leaves the active set.
   - A lease whose job left the active set without a release is reclaimed when the free list runs dry.
   - The generator refuses to start when the draft cache has fewer slots than `max_batch_size`. The check runs after the recurrent clamp of `max_batch_size` to `cache.num_slots`.
6. **Restored prefixes (prompt-cache revival, CPU tier, NVMe tier, requeue).**
   - A revived span skips its prefill, so nothing writes draft K/V for it. The served full-size cache kept those pages; the ring cannot, and without the target state of those tokens it cannot rebuild them exactly.
   - On release (completion, requeue, cancel), a slot keeps a tag for each whole committed page it still holds: the logical page, the chained page hash (the page table's own blake2b chain, extended over generated pages) and the first token of the next page, because the MTP input at position p pairs the target state of p with the embedding of p+1.
   - A grant picks the free slot with the most tags matching the new job's prompt (ties go to the slot released longest ago), then zeroes every ring page without a matching tag before the job's first draft write.
   - Result: a follow-up turn or a requeued job reads its own earlier draft K/V where the slot still holds it, and zeros elsewhere, never another sequence's.
   - After a restart, or with an NVMe-tier restore, every slot is empty, so the restored span reads zeros until new tokens fill the window. This is a draft-quality regression against served on that request (box: tier restore check).

## Every reader and writer of the draft cache

| site (served tree) | access | with the window |
| --- | --- | --- |
| `generator/job.py:1434` Job.prefill: draft prefill of each chunk. Carries the shifted target states (`mtp_carry_hidden`); also runs under `EXL3_LS_PREFILL_PIPELINE`, which calls Job.prefill | write K/V + QSA planes (raw ring, pooled); attention output discarded | the job's window row (`job.py:1437-1447`). Chunking, carry and target states unchanged (test `job_prefill`: 4 chunks starting at 0/512/1024/1280 with lengths 512/512/256/219, carries 0/511/511/255 as served) |
| `generator/generator.py:952-965` iterate_draftmodel_mtp_gen: depth-d drafting steps (the device-resident chain) | write positions T..T+d−1, read the window | window rows for the batch, in batch order. Pinned staging `mtp_window_draft` exactly when the served table is pinned (`EXL3_DRAFT_PINNED_STAGING=1`), so the async chain never stalls on a pageable copy (`generator.py:1026-1032, 1090`) |
| `generator/generator.py:1638-1675` iterate_gen: MTP accept prefill. Replaces the accepted speculative positions K+1..K+A−1 with target-state inputs (the target-state repair) | write positions K+1..K+A−1 | the jobs' window rows, from the same batch-row list as `block_index`. Staging `mtp_window_verify`, same lifetime as the served `block_index` staging (`generator.py:1331-1336, 1809`). Tests `verify_round` and `verify_two_rounds` run the real iterate_gen: accept prefill rows, cache lengths and target-state slices; rows follow a reordered batch |
| `generator/job.py:1294-1302` partial-page copy on a prompt-cache hit (non-recurrent models only) | copy_page main and draft | main cache only when windowed (never reached on this recurrent model) |
| `generator/pagetable.py:1075` defrag: rotate every cache that shares the page tables | rotate draft pages | skipped when windowed; the window is addressed by slot, not page index (test `defrag`) |
| `generator/generator.py:276` CPU tier (`CPUPageCache`) | store / restore draft pages with main pages | main cache only when windowed (test `gen_init`) |
| `generator/generator.py:314` + `disk_cache.py:1452` NVMe tier | store / restore draft pages | main cache only when windowed (`install(..., None, ...)`). A restore leaves the window as described under restored prefixes |
| `generator/generator.py:787/813` (non-MTP draft model), `1061/1626` (DFlash) | draft cache | not reachable: a windowed cache requires an MTP draft model (asserted in `Generator.__init__`); only `mtp_draft` models get one |
| `modules/attn.py:976` autosplit_extra_measure (load) | zeroes page 0 and runs a synthetic sparse prefill at `num_pages·256 − chunk` | page 0 is slot 0's sink, which is empty at load. The measured transient does not grow with context (tiled scorer, fixed slabs) |
| `modules/attention_fn/bc_attn.py` BCAttn (graph decode) | QSA statics sized to the pooled plane | scan clamp (below) |
| requeue (`prepare_for_requeue` re-inits the same Job object) | none | slot released with tags before the requeue, re-granted with affinity on restart (test `requeue_affinity`) |
| cancel / clear_queue / reap_failed_job / completion | none | release: with tags for completion, requeue and cancel; without tags for clear_queue and failures |

## Changes (patched tree, file:line)

- `cache/mtp_window.py` (new, 165 lines):
  - `_parse_window` / `MTP_KV_WINDOW` (27-43)
  - `window_pages`, `window_pool_tokens`, `window_offsets` (46-61)
  - `MTPWindow` geometry (64-78)
  - `ring_page`, `release_tags` (81-107)
  - `MTPWindowSlots` (110-165)
- `cache/cache.py`:
  - flag import (11)
  - draft-cache resize (142-149)
  - per-layer `mtp_window_scan_tokens` and the check that one cycle exceeds the QSA sparse threshold (169-180)
  - boot log line (199-205)
- `generator/generator.py`:
  - imports
  - windowed assert branch (191-199)
  - `draft_window` and tier exclusion (281-293)
  - slot check after the recurrent clamp (319-327)
  - NVMe exclusion (341-344)
  - releases in clear_queue (483), cancel (539), completion/requeue (1833-1835), reap_failed_job (1861)
  - slot, tag, clear and rows methods (756-846)
  - drafting round (1017-1032, 1090)
  - verify round and accept prefill (1320-1336, 1809)
- `generator/job.py`: partial-page copy guard (1295-1299); draft prefill row (1437-1447).
- `generator/pagetable.py`: defrag skips a windowed draft cache (1076-1080).
- `modules/attention_fn/bc_attn.py`: `qsa_scan_cap` from the cache layer, with its bounds assert (249-257); the clamp in `step` (594-598).

## Decisions the brief asked for

- The draft prefill still runs over the whole prompt in r1, unchanged except for the table.
  - Window K/V need the target state of their own tokens, and those states exist only while the chunk is prefilled.
  - Chunks that end before `prompt_end − window − 256` could skip the MTP block's forward (saving its share of prefill time, about one of 49 blocks). This is not done here, to keep the prefill path identical. It is an r2 candidate.
- **QSA indexer planes.**
  - The raw-key ring rows and the pooled rows live per physical page, so they travel with the ring page.
  - Pooled keys are roped at their absolute block start and stay valid under aliasing.
  - The raw-key ring's rewind guarantee (verify window + 2 < 20 rows) is per page and unchanged.
  - The graph scan is clamped to one cycle, and `Cache.__init__` refuses a window whose cycle does not exceed the sparse threshold (`4·top_k + 3`, 2,051 tokens at top-k 512). The smallest valid window at top-k 512 is 2,048 (cycle 2,304).
- **What the NVMe tier stores for draft pages:** nothing. Main pages only.
- **Sink.** Page 0 (the first 256 tokens) is kept for every sequence at no extra cost.

## Main output: what holds and what does not

Holds on every path:

- The flag does not touch the main cache (same shapes, bytes and page tables; test `alloc`: the main cache is identical with the flag on), the target forward, or the sampler.
- Every emitted token is the target's own choice for the verify forward that produced it. The draft only proposes.

Does not hold beyond one ring cycle: byte-identical greedy text.

- Proposals change, so acceptance changes, and with it the verify layout (which tokens share a verify forward).
- In this engine the target's numerics depend on that layout. R537 and R542 fingerprints moved with the draft depth, and R552 showed batch position changes greedy text.
- Below one cycle, the draft input is identical to served, so fingerprints must match. This is the box rule for the c1 fingerprint.
- For the same reason, with the window on, greedy text at depth can depend on a request's cache history (cold, warm, restored). The served stack's GT gate ("output identical to warm and cold") can then fail at 30k without a bug. The box spec logs it instead of gating on it.

## Byte and pool arithmetic (8 slots, pool 966,656)

Per attention layer at 8,8 with the raw-key ring: K 512 + V 512 + scales 64 + raw-key ring 20 + pooled 64 = 1,172 B per token. Test `window_allocation_and_bytes` reproduces this from the real `CacheLayer_qsa_quant.storage_size()` with 2 × 256 K/V and a 128-dim indexer.

| | tokens | bytes | MiB |
| --- | --- | --- | --- |
| MTP layer cache today (on cuda:0) | 966,656 | 1,132,920,832 | 1,080.4 |
| window W=4096, 8 slots × 17 pages | 34,816 | 40,804,352 | 38.9 |
| window W=16384, 8 slots × 65 pages | 133,120 | 156,016,640 | 148.8 |
| freed, W=4096 | | 1,092,116,480 | 1,041.5 |
| freed, W=16384 | | 976,904,192 | 931.6 |

If TabbyAPI builds the draft Cache without `max_batch_size` (the default is 16 slots; the boot log line shows it), the window pools double: 77.8 / 297.6 MiB.

**New pool at equal bytes** (13 layers' bytes, 12 layers + window pool). Total bytes are 13 × 1,132,920,832 = 14,727,970,816:

- W=4096: (14,727,970,816 − 40,804,352) / (12 × 1,172) = 1,044,309 tokens, which floors to **1,032,192** (63 × 16,384; +4 steps, +65,536, +6.8 %; +8.0 % before flooring).
- W=16384: 1,036,117, which also floors to **1,032,192** (+4 steps, +6.8 %; +7.2 % before flooring).
- 16 slots: W=4096 1,041,408 → 1,032,192 (+4); W=16384 1,025,024 → 1,015,808 (+3).

**Per card, which is what decides.**

- The bytes are freed on cuda:0, where the MTP layer lives (R539 census).
- A pool step costs cuda:1 7 × 16,384 × 1,172 B = 128.2 MiB, and cuda:0 5 × 16,384 × 1,172 B = 91.6 MiB plus the MTP layer's 18.3 MiB (0 with the window).
- cuda:1 bound the pool at every earlier boot of this model (R539, R541, R546, R548). The flag frees nothing there, so **at unchanged placement the expected gain is 0 steps**.
- The +4 needs layer 23 to move onto cuda:0 (box S3; with 6 + 6 cache layers a step costs 109.9 MiB per card). That move changes greedy text (R544b) and requires the full gates.

**The zero-code alternative (`draft_cache_mode` Q6 / Q4 on the MTP layer).**

| | draft layer B/token | saved B/token | share of cache bytes | equal-bytes pool |
| --- | --- | --- | --- | --- |
| Q6 | 916 | 256 | 1.68 % | 983,040 (+1 step) |
| Q4 | 660 | 512 | 3.36 % | 999,424 (+2 steps) |
| window | 42–161 B/token equivalent | 1,011–1,130 | 7.2–8.0 % | 1,032,192 (+4 steps) |

The window frees more pool than both. Stacking Q6 on the window only shrinks its 39–149 MiB pool, which is not worth an arm. Two constraints apply to Q6/Q4 as well:

- They free only cuda:0 bytes, so the same placement limit applies.
- They break the standing KV floor (user 2026-09-17: no KV below 8-bit on any engine) unless the user exempts the draft layer.

## Verified here, and how

CPU only (macOS, Python 3.11.15, torch 2.14.0 CPU, pytest 9.1.1). The served tree is `src/exllamav3`, untouched. The patched tree is a copy patched with `patch -p1 --fuzz=0`, judged by exit code. Every behavioural test runs the tree's own functions in a subprocess with only `exllamav3.ext` and triton stubbed:

- `Cache.__init__` + `CacheLayer_qsa_quant` (real `Attention.cache_layer_type`, real `QSAIndexer.sparse_threshold`)
- `Generator.__init__` (real PageTable, RecurrentCache; recorders for the CPU and NVMe tier constructors)
- `Generator.iterate_draftmodel_mtp_gen`
- `Generator.iterate_gen` (verify, rejection rewind, EOS completion, MTP accept prefill, release)
- `Job.prefill` (real Sequence, CachePage objects, chunking, recurrent last-page split)
- `PageTable.defrag` (the rotation list reaches `ext.cache_rotate`)
- `BCAttn.__init__`, `BCAttn.step`

What the tests establish:

- **Flag unset:** the patched tree prints exactly what the served tree prints for the allocation (with and without the raw-key ring), generator init (tier lists, batch clamp), the drafting round, both verify rounds, chunked prefill, defrag and BCAttn.step.
- **Flag set:**
  - window pool and shapes, with the main cache unchanged;
  - 1,172 B per token and the freed bytes and equal-bytes pools above; the Q6/Q4 bytes and pools;
  - small windows refused;
  - tiers exclude the draft cache, and the slot check runs after the clamp;
  - drafting rows, lengths and target states per step;
  - accept-prefill rows, lengths and target-state slices; the carry unchanged; the completed job's slot released; rows following a reordered batch;
  - one window row across all prefill chunks with the served carry chain;
  - defrag rotating main only;
  - scan clamp values (regime and rope position from the true length);
  - `BCAttn.__init__` deriving the clamp from the real cache layer;
  - slot reuse: an unrelated job's slot fully zeroed; a continuation's slot keeps exactly its tagged pages (sink + the two newest whole pages at P = 5) and zeroes the rest; a prompt differing inside page 9 keeps only the sink; other active slots untouched.
- **Window bookkeeping** (`ring_contract`): 3 random schedules of about 900 operations over 3 interleaved slots (prefill chunks aligned and not, MTP rounds of depth 1/3/15 with rejected drafts, replays), about 740k position checks per schedule. No violation, no access outside a slot's pages. Two broken maps (a ring one page short, no sink) are caught.
- **Mutation check** (by hand, not in the suite): routing the accept prefill or the draft prefill back to `block_index`, or dropping the completion release, changes the scenario output the tests assert on.
- **Packaging:** install.py applies once, refuses a second run, refuses a drifted baseline and a pre-existing `cache/mtp_window.py`; the patch reverses cleanly; the Dockerfile rules hold.

CPU test output (`../../.venv311/bin/python -m pytest tests -v -p no:cacheprovider` from `out/mtp-kv-window-r1/`):

```
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- <venv>/bin/python
rootdir: <round dir>
plugins: anyio-4.15.1
collecting ... collected 38 items
tests/test_mtp_kv_window_cpu.py::test_patch_and_manifest_pins PASSED     [  2%]
tests/test_mtp_kv_window_cpu.py::test_patch_targets_are_manifest_files PASSED [  5%]
tests/test_mtp_kv_window_cpu.py::test_served_baselines PASSED            [  7%]
tests/test_mtp_kv_window_cpu.py::test_patch_applies_fuzz0_by_exit_code PASSED [ 10%]
tests/test_mtp_kv_window_cpu.py::test_install_py_on_scratch_root PASSED  [ 13%]
tests/test_mtp_kv_window_cpu.py::test_install_py_refuses_drift PASSED    [ 15%]
tests/test_mtp_kv_window_cpu.py::test_dockerfile_rules PASSED            [ 18%]
tests/test_mtp_kv_window_cpu.py::test_flag_parsing[-0] PASSED            [ 21%]
tests/test_mtp_kv_window_cpu.py::test_flag_parsing[0-0] PASSED           [ 23%]
tests/test_mtp_kv_window_cpu.py::test_flag_parsing[4096-4096] PASSED     [ 26%]
tests/test_mtp_kv_window_cpu.py::test_flag_parsing[16384-16384] PASSED   [ 28%]
tests/test_mtp_kv_window_cpu.py::test_flag_parsing[100-None] PASSED      [ 31%]
tests/test_mtp_kv_window_cpu.py::test_flag_parsing[-256-None] PASSED     [ 34%]
tests/test_mtp_kv_window_cpu.py::test_flag_parsing[abc-None] PASSED      [ 36%]
tests/test_mtp_kv_window_cpu.py::test_flag_parsing[256-None] PASSED      [ 39%]
tests/test_mtp_kv_window_cpu.py::test_flag_off_identical_to_served[alloc] PASSED [ 42%]
tests/test_mtp_kv_window_cpu.py::test_flag_off_identical_to_served[alloc_small_window] PASSED [ 44%]
tests/test_mtp_kv_window_cpu.py::test_flag_off_identical_to_served[gen_init] PASSED [ 47%]
tests/test_mtp_kv_window_cpu.py::test_flag_off_identical_to_served[draft_round] PASSED [ 50%]
tests/test_mtp_kv_window_cpu.py::test_flag_off_identical_to_served[verify_round] PASSED [ 52%]
tests/test_mtp_kv_window_cpu.py::test_flag_off_identical_to_served[verify_two_rounds] PASSED [ 55%]
tests/test_mtp_kv_window_cpu.py::test_flag_off_identical_to_served[job_prefill] PASSED [ 57%]
tests/test_mtp_kv_window_cpu.py::test_flag_off_identical_to_served[defrag] PASSED [ 60%]
tests/test_mtp_kv_window_cpu.py::test_flag_off_bc_step_identical_to_served PASSED [ 63%]
tests/test_mtp_kv_window_cpu.py::test_flag_off_rawk_alloc_identical PASSED [ 65%]
tests/test_mtp_kv_window_cpu.py::test_window_allocation_and_bytes PASSED [ 68%]
tests/test_mtp_kv_window_cpu.py::test_draft_cache_mode_alternative_bytes PASSED [ 71%]
tests/test_mtp_kv_window_cpu.py::test_small_window_refused PASSED        [ 73%]
tests/test_mtp_kv_window_cpu.py::test_generator_init PASSED              [ 76%]
tests/test_mtp_kv_window_cpu.py::test_drafting_round_rows PASSED         [ 78%]
tests/test_mtp_kv_window_cpu.py::test_verify_round_accept_prefill_and_release PASSED [ 81%]
tests/test_mtp_kv_window_cpu.py::test_verify_rows_follow_batch_order PASSED [ 84%]
tests/test_mtp_kv_window_cpu.py::test_job_prefill_rows_across_chunks PASSED [ 86%]
tests/test_mtp_kv_window_cpu.py::test_defrag_leaves_windowed_draft_cache PASSED [ 89%]
tests/test_mtp_kv_window_cpu.py::test_bc_step_scan_clamp PASSED          [ 92%]
tests/test_mtp_kv_window_cpu.py::test_bc_init_derives_the_cycle PASSED   [ 94%]
tests/test_mtp_kv_window_cpu.py::test_slot_reuse_keeps_only_matching_pages PASSED [ 97%]
tests/test_mtp_kv_window_cpu.py::test_ring_contract PASSED               [100%]
============================= 38 passed in 28.32s ==============================
```

To reproduce: create a venv with CPU `torch`, `numpy`, `pytest`, `typing_extensions`, `pyyaml`, `pydantic`, `tokenizers`, `rich` and `safetensors` (the tree's import-time dependencies), then run pytest from `out/mtp-kv-window-r1/`. `MTP_WINDOW_SERVED_SRC` points at another served tree.

## Only the box can verify

- How TabbyAPI builds the draft Cache: whether it passes `max_batch_size` (8 or 16 slots; the boot line tells), and whether anything in TabbyAPI reads `draft_cache.max_num_tokens`.
- Free VRAM per card and layer placement at boot. The pool gain is 0 steps unless cuda:0's freed ~1 GiB changes what binds (box S2/S3).
- The graph path with a clamped `t_total`: capture, replay patching and top-k over one cycle on real kernels. The CPU test only sees the value handed to `bc.run`.
- Acceptance and decode at short and deep context, c1/c4/c8, and whether the c1 fingerprint is unchanged.
- The zeroing on grant under `EXL3_LS_PREFILL_PIPELINE` streams. It runs on the draft layer device's current stream before the first draft prefill of the job.
- Tier restore: correctness, and acceptance of the restored request.

## Risks

1. **No pool gain at normal placement** (cuda:1 binds). Moved placement (layer 23 onto cuda:0, either by itself at boot as in R543, or through a forced split in S3) is the expected and only gain path. It changes greedy text (R544b), so the box spec compares moved W arms against an OFF control at the same moved placement (S1.0) and applies a placement-aware headroom rule (tightest card of OFF at 966,656 minus 32 MiB, plus a survival run: cold 120k, c1 to c8 fresh graph captures, one c8 round). A pool won this way needs new canonical fingerprints and the full gates.
2. **Greedy text at depth is no longer independent of cache history or of the window size.** See "Main output". It affects gates written as byte-identity checks at 30k (GT).
3. **Deep-context draft quality.**
   - Top-k over one ring cycle sees at most W + 256 tokens.
   - Restored spans read zeros.
   - A revived prefix keeps only tagged pages, which excludes the page that was being written and the one after it.
   - Acceptance at 100k+ may drop; at c8 each pp of acceptance costs about 0.6 % throughput.
4. **Eager-path duplicates.** An eager sparse decode (batch size > 8 or verify length > 16; not in the served config) scores aliased copies of ring pages, so top-k picks fewer distinct blocks. Proposals only.
5. **Host overhead.** The two row builds per step are memoised when the slots and width do not change. A slot grant with tag matching costs up to `P` hash comparisons per free slot, once per job.
6. **The flag enters the NVMe tier namespace** (every `EXL3_*` variable does), so W arms open their own namespace. Tier pages from OFF boots are not reused by W boots and the other way round.

## Stacking with the R565 image (`tabbyapi:ngram-prefetch-r1-gdnbf16`, `EXL3_NGRAM_PREFETCH2=1`)

That overlay patches `generator/generator.py` and `generator/prefill_pipeline.py`. Its sources are not in this workspace, so coexistence could not be checked here.

- This patch does not touch `prefill_pipeline.py`.
- Its `generator.py` hunks are local: imports, the draft assert, the tier lines, the slot check after the recurrent clamp, the release calls, one method block, and the row construction in the drafting and verify rounds.
- `install.py` refuses the R565 `generator.py` by hash, so a wrong stack cannot build silently.
- Before a promotion: rebase the `generator.py` hunks onto the R565 file (regenerate the manifest), rerun this suite against an extraction of the R565 tree (`MTP_WINDOW_SERVED_SRC=<extracted>/exllamav3`), and repeat S1 on that base.

## What was wrong in the previous worker's partial output

The previous patch applied with fuzz 0 and its manifest hashes matched. The problems were in what it did:

1. **The served path allocated nothing less.** The draft cache was resized in `model_init.py` only. The daily is served by TabbyAPI, which builds its own draft `Cache(draft_model, max_num_tokens=cache_size, ...)`. Under TabbyAPI the flag would have kept the full-size draft cache (with the loosened `<=` assert it would even have booted) and freed 0 bytes. Now fixed in `Cache.__init__`.
2. **The scan clamp used the whole window pool** (`slots × P` pages), not one cycle. Each ring page was scored `slots` times, so top-k selected duplicates.
3. **Zero-state backfill.** On a revived prefix it ran a draft forward over the last W tokens with zero target states. That overwrote K/V the slot might still hold for the same conversation with K/V computed without target information, and it cost a W-token draft forward per request. Replaced by tagged slot reuse plus zeroing.
4. **Slot check before the batch clamp.** It compared slots with `max_batch_size` before the generator clamps `max_batch_size` to the main cache's `num_slots`, so a valid config (generator 8, cache 4) could be refused.
5. **Unbounded cache and blocking copies for the window tables.** One `(slots × width)` tensor was cached per distinct width, for the life of the process. The draft-step and verify tables were fresh pageable tensors, which would turn every drafting step's upload into a synchronous copy and stall the device-resident draft chain.
6. **Dead or needless hunks.** The non-MTP draft path, `disk_cache.py` (the generator already passes None), and `model_init.py`.
7. **The Dockerfile could not build.** Its last check indexed `t[0, 100]` on a `(1, 64)` table (IndexError).
8. **Tests did not run the served functions.** They were mostly source-text checks plus a stand-alone simulation that never called `Generator`, `Job.prefill` or `BCAttn`. The accept-prefill and prefill paths were checked only by reading the source.
