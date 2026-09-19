# QSA raw-key ring round 1 — implementation status

Round 1 adds `EXL3_QSA_RAWK_RING=1`, a default-off option that replaces the QSA indexer's full per-token `raw_k` plane with a ring of 20 rows per cache page. The pooled plane, the K/V pages, the selection and the attention are unchanged. The base is the served chain `tabbyapi:nvme-tier-r4-e3det` (R535). Line numbers without a prefix refer to the local extraction `served-src-r535/exllamav3/`. `p:` marks the patched tree. No GPU, compiler or remote host was used. Nothing here is a measurement.

## Verdict on the census claim

The claim holds in substance: `raw_k` is only read by the pool-update kernels, and only for the rows of the blocks an append touches. The pooled plane is the only long-term product. Two details of the census entry (`flan/probes/vram_census/README.md`, candidate 1) are wrong:

1. "8 rows per page" is not enough without a C++ change. The graph-captured decode path (`attention.cpp:611-651`) launches the raw append first and the pool update second, and the pool kernel gets no pointer to the staged new keys, only the plane. So one BC call must fit the partial block before `pos0` plus all of its own rows in the ring without two positions sharing a slot: `q_len + compress_ratio - 1` = 16 + 3 = 19 rows for `MAX_QLEN` 16 (`bc_attn.py:52`, `attention.h:38`). Speculative rewinds need `q_len + 2 < rows` (below). The ring is 20 rows: the smallest multiple of the block size that covers both.
2. The net saving is therefore 2.34 GiB, not 2.46 GiB, and the projected pool at equal bytes is 984,064 tokens (+20.1 %), not 994,000.

A ring of 8 rows would need the pool kernel to read new rows from the staging buffer in the BC path too (a C++ argument change and an extension rebuild) and would cap every verify window at 6 tokens. It gains 0.12 GiB more and, at equal bytes, the same number of 16,384-token ladder steps (10). Not done.

## Reader/writer map

`raw_k` (fp16, `[pages, PAGE_SIZE, 128]`) and `pooled` (fp16, `[pages, PAGE_SIZE / 4, 128]`) live on `QSAPlanes` (`cache/qsa.py:11-59`), mixed into `CacheLayer_qsa` (fp16 K/V) and `CacheLayer_qsa_quant` (quantized K/V, the daily's 8,8). `attn.py:960-973` maps every QSA module's cache layer to one of the two, main model and MTP draft alike.

Writers of `raw_k`:
- Eager path `QSAIndexer.update_planes` (`qsa_indexer.py:446-503`): `_mla_plane_update_kernel` (`mla_triton.py:150-177`) writes every appended row at `phys * page_size + pos % page_size` (`qsa_indexer.py:490-493`). Called from `attn.py:1077` whenever the BC graph declines (bsz > 8, q_len > 16, non-causal spans, `EXL3_BC_ATTN=0`), which includes every prefill chunk.
- Graph path: the same kernel compiled ahead of time (`bc_attn.py:450-454`) and launched from C++ with the plane pointer set once in `set_qsa` (`bc_attn.py:258`, `attention.cpp:160-186`, launch `attention.cpp:611-628`).
- `copy_page` (`cache/qsa.py:46-50`): rows `[0, num_tokens)` of one page into another.
- Warmup (`attn.py:1017`): zeroes page 0 before the synthetic max-context prefill.
- Defrag (`pagetable.py:1084-1087`): `ext.cache_rotate` over every tensor of `get_all_tensors()`, whole pages, page-major, byte-generic (`exllamav3_ext/generator/cache.cu:12-40`).
- CPU tier and NVMe tier restores: write whole page images back through the segment table built from `get_tensors()` (`cpu_cache.py:80-91`, `disk_cache.py:1547-1559`).
- TP workers: `model_tp_fn.py:364` (`copy_page`), `:383`, `:477` (tensor lists).

Readers of `raw_k`:
- `_qsa_pool_update_kernel` (`qsa_triton.py:66-144`), eager (`qsa_indexer.py:496-502`) and graph (`bc_attn.py:456-462`, `attention.cpp:630-651`). Grid `(bsz, append_len // P + 1)`. Program `pi` rebuilds pool `pos0 // P + pi` from rows `pool * P + j < pos0 + append_len` (`:89-117`). The rows read are `[floor(pos0 / P) * P, pos0 + append_len)`: the rows appended by this call plus at most `P - 1` = 3 rows written by an earlier call (the partial block before `pos0`). Nothing else.
- `copy_page`, defrag, tiers, TP: byte copies of whole pages or page prefixes; they never interpret the rows.
- Test-only references: `update_planes_ref` (`qsa_indexer.py:505-556`, reads the same block rows) and `select_indices_paged_ref` (`:585-631`, only uses `raw_k.shape[1]` as the page size).
- Not a reader: the non-cached path (`attn.py:870-872`, `pool_keys` over a local tensor), `build_mask` (`qsa_indexer.py:407-424`, local), `select_indices_paged`/`_select_rows` (read `pooled` only, `:573-582`), `sparse_attend` (K/V only, `:633-671`), the BC sparse regime (`attention.cpp:655` onwards reads `qsa_pool_plane` only; `qsa_raw_plane` appears at `:184`, `:616`, `:634` and nowhere else), `bc_attn.py:474` (sizes the score buffer from `pooled`), `bc_attn.py:715` (checks `raw_k` exists and sits on the module's device).

Writers of `pooled`: the pool-update kernel (both paths), `copy_page`, defrag, tiers. Readers: the selection kernels (eager `_select_rows`, graph `k_qsa_fewq`), both over complete blocks only; a partial block's pooled value is written but never selected, and it is rewritten when the block completes.

## Which raw rows each path needs

A pooled key is final once its 4-token block is complete, and every block lies inside one page (`PAGE_SIZE % compress_ratio == 0`, `cache/qsa.py:26`). So the rows needed after a call returns are the rows of the partial block at the sequence's next start position. Per path on the served chain (Qwen3.8-Flash-Next is a recurrent hybrid, GDN state class `GDNState`, `qwen4_exp.py:313`):

- Prefill: chunk starts are page-aligned (`job.py:1231-1254`, `prefill_end` rounded down at `:1234`; the two-device lookahead only runs full 2,048-token page-aligned chunks, `prefill_pipeline.py:17-35`; the multimodal chunk rules that end chunks mid-page, `atomic_mm_prefill` and `mm_exact_chunks`, are set only by `gemma4.py:427` and `deepseek_v4.py:277`, not by `qwen4_exp`, so vision inputs do not change this). A page-aligned start reads no earlier row. The last chunk ends at `len - 1`, so the next call (decode) needs that chunk's last `(len - 1) % 4` rows.
- Decode and MTP verify: the call at `pos0 = kv_position` needs rows `[pos0 - pos0 % 4, pos0)`. After a verify window of `L` tokens at `pos0`, the next start is `pos0 + a` with `a ≥ 1` accepted, so the rows it needs are at `≥ pos0 - 2` while the window wrote up to `pos0 + L - 1`. MTP draft steps write `kv + idx` one token at a time (`generator.py:950-979`); the accept prefill rewrites `kv + 1 .. kv + a - 1` (`:1655-1676`); the next round starts at `kv + a`. Same bound.
- Checkpoint rewinds (stop strings, banned strings): `GDNState.rollback_capacity()` is 0 (`gated_delta_net.py:138-141`), so `rewind_checkpoint` (`job.py:882-944`) always restores a page-aligned recurrent stash (`:906-909`) and replays prefill from that page boundary. The replay rewrites every row from the boundary to `len - 1`.
- Requeue: `maybe_stash_recurrent(..., PAGE_SIZE)` stashes at a page boundary, pages are released, the job resumes by replay from a page boundary.
- Prefix reuse: only full pages carry a real hash and are shared or revived; a sequence continuing after a shared prefix starts at a page boundary. The partial-last-page copy (`job.py:1276-1310`, `c.copy_page` at `:1297`) is skipped for recurrent models (`:1276`, `self.generator.recurrent_cache is None`), so it never runs on Flash-Next: `qwen4_exp` sets the `recurrent_states` cap (`qwen4_exp.py:309`) and the generator then always creates a `RecurrentCache` (`generator.py:291-293`), whatever the RAM budget. If it ran with `num_tokens % 4 != 0`, the destination would need the source page's rows `[num_tokens - num_tokens % 4, num_tokens)`, which a ring only holds if the source page stopped exactly there.
- NVMe/CPU tier: stores and restores full pages only (keyed by page hash); a restored page is never continued mid-page.
- Page sharing: a full page shared by several sequences is written again only by a replay, with the same values (the served code already relies on that, `job.py:935-937`).

## Design

Flag off: nothing changes. The allocation shapes, the kernels, the launch code and the compile signatures are the served ones. The patch only adds code behind `raw_ring_rows > 0`, plus two lines in the test-only torch references (below).

Flag on (`EXL3_QSA_RAWK_RING=1`, read at import of `cache/qsa.py`):

- p: `cache/qsa.py`. `QSAPlanes._init_planes` sets `raw_ring_rows` = 20 (`ceil((MAX_QLEN + P - 1) / P) * P` with `MAX_QLEN` 16 and `P` 4) and allocates `raw_k` as `[pages, 20, 128]`. Row slot of position `t` in physical page `p`: `p * 20 + t % 20`. The ring is page-major like every other cache tensor, so defrag, `get_tensors()`, the CPU tier and the NVMe tier carry it without change. `copy_page` copies `pooled` as before and no raw rows; it raises for `num_tokens % 4 != 0` (unreachable on recurrent models, see above). `storage_size` follows the shape.
- p: `qsa_triton.py`, three new kernels next to the served ones (the served kernels are byte-identical):
  - `_qsa_raw_ring_append_kernel`: the runtime signature of `_mla_plane_update_kernel` (so the C++ launch and its graph-parameter patch indices 2/3/4 are unchanged), writes the row of position `t` to its ring slot, and skips rows before `pos0 + append_len - RING` (only the last 20 rows of a longer eager chunk are kept; they occupy distinct slots, so there is never a write race).
  - `_qsa_pool_update_ring_bc_kernel`: the runtime signature of `_qsa_pool_update_kernel` (graph-parameter indices 4/5/6 unchanged), the same math line for line, raw rows read from the ring.
  - `_qsa_pool_update_ring_kernel` (eager): one extra pointer, the staged new keys. Rows at or after `pos0` come from the staging buffer, rows before `pos0` from the ring. Same math line for line.
  - `qsa_ring_plane_update(...)`: the eager launch sequence, pool update FIRST, then the ring append. The pool kernel reads the pre-`pos0` rows before the append can overwrite a slot.
- p: `qsa_indexer.py`. `update_planes` takes the ring branch when `layer.raw_ring_rows > 0`, after the served projection, stage kernel and query rope; the ring launcher derives the page size from the pooled plane (`pooled.shape[1] * P`), not from `raw_k.shape[1]`. The ring branch raises if a verify forward (`params["recurrent_history"]`) is longer than `RING - P + 1` = 17 tokens (it would break the rewind bound; the daily's longest window is 4). `update_planes_ref` refuses ring layers; `select_indices_paged_ref` takes the page size from `pooled.shape[1] * P` (the same value, 256, on a full plane).
- p: `bc_attn.py`. `_configure_qsa` compiles the ring append and the ring BC pool kernel instead of the served pair when the layer has a ring, with the same runtime signatures, and asserts `q_len + P - 1 <= RING`. The factory `_qsa_ring_graph_kernels(...)` holds both ring kernels so the GPU harness can AOT-compile exactly what a graph slot compiles.
- Unchanged: `attention.cpp` and every other C++/CUDA file (no extension rebuild), `attn.py`, the generator, the tiers.

### Why 20 rows are enough (bit-exactness argument)

The ring slot of position `r` holds `raw(r)` as long as no position `r' ≡ r (mod 20)` of the same page was written after `r`'s last write. Two conditions cover every path:

- (A) Within one graph call (append, then pool): the rows the pool kernel reads, `[pos0 - pos0 % 4, pos0 + L)`, span at most `L + 3` = 19 ≤ 20 positions, so they occupy distinct slots. The eager path has no such constraint: it reads new rows from the staging buffer and old rows before any ring write.
- (B) Across calls: a later call at `pos0'` needs rows `r ≥ pos0' - 3`. Every write since those rows were last written landed below `kv + 17`, where `kv ≤ pos0'` is the committed length at the time of the write: a verify window starts at `kv` and is at most 16 tokens in the graph or 17 in the eager path (guarded), MTP draft steps write `kv + idx` for `idx` below the depth, and committed positions only decrease through a replay, which rewrites from a page boundary. An aliasing write would have to land at `r + 20 ≥ pos0' + 17`. After a replay or any long prefill chunk, the rows needed next are within the chunk's last 20 rows, which the chunk wrote to the ring after every stale write.

The pooled values are then computed from the same fp16 inputs, in the same order, by the same arithmetic as the served kernel, so the pooled plane is bit-identical and the selections and attention outputs follow.

## Bytes

Per token and per QSA layer on the daily (8,8, 2 KV heads × 256, indexer head 128, compress ratio 4):

| tensor | served | ring |
|---|---|---|
| K packed (`qk`, int32) | 512 B | 512 B |
| V packed (`qv`) | 512 B | 512 B |
| K scales (`sk`, fp16) | 32 B | 32 B |
| V scales (`sv`) | 32 B | 32 B |
| `raw_k` | 256 B | 20 B (20 rows × 256 B per 256-token page) |
| `pooled` | 64 B | 64 B |
| total | 1,408 B | 1,172 B |

The main model has 12 QSA layers and the MTP draft cache 1 (same token count, the generator asserts it; census README line 121): 13 layers, 18,304 B/token served, 15,236 B/token with the ring.

At 819,200 tokens (3,200 pages):
- served cache-side total for these layers: 14,994,636,800 B (13.96 GiB); `raw_k` 2,726,297,600 B (2.54 GiB).
- ring: 212,992,000 B for `raw_k` (0.198 GiB); saved 2,513,305,600 B = 2.34 GiB (2.51 GB).
- per layer: 209,715,200 B of `raw_k` → 16,384,000 B; 184.4 MiB saved per layer.
- pool at equal bytes: 14,994,636,800 / 15,236 = 984,158 tokens → 984,064 in whole pages, +20.1 %. In 16,384-token ladder steps from 819,200: +10 steps = 983,040. This is arithmetic, not a measurement. The saving lands on whichever card holds each QSA layer and the MTP layer; the ladder is limited by the tighter card, and a layer that moves between cards under `gpu_split [30, 30]` changes the result (the R489 placement confound). The census noted about 1.2 GiB unused on cuda:0 at R541, so the achieved gain depends on the split.

## NVMe tier and CPU tier impact

The page image shrinks by 60,416 B per QSA layer per page: 360,448 B → 300,032 B per layer, 4.47 MiB → 3.72 MiB per page for the 13 layers (−16.8 %, before the 256-byte segment alignment). The tier stores about 20 % more pages in the same bytes and writes 16.8 % fewer bytes per page. The flag-on namespace differs from the flag-off one three ways, each sufficient: the env list in `engine_identity()` (`disk_cache.py:1098-1134`) contains `EXL3_QSA_RAWK_RING=1`; `page_layout[].shape` of the `raw_k` segment is `[20, 128]` instead of `[256, 128]` (`:1551-1558`); `page_image_bytes` changes. A flag-on process therefore never opens a flag-off tier, and vice versa; the other namespace is deleted at open (`gc_stale_namespaces`). Installing the overlay also changes `sources_sha256` with the flag off, so the patched image starts a new namespace in either mode. The tier stores the ring rows of a page as they are; nothing reads them after a restore, because a restored page is complete.

## Risks

- Nothing was compiled here (no Triton on this machine). The C++ graph path launches the ring kernels through the served `TritonKernel` launcher with the served argument vectors; the ring kernels keep the runtime parameter lists and order, so the vectors and the graph-parameter patch indices match. The GPU harness AOT-compiles both graph kernels with `_compile_kernel` exactly as a slot does, and JIT-compiles the eager `_qsa_pool_update_ring_kernel` in its first scenario. That eager kernel is the likeliest first-run compile failure: it selects between two pointer arguments in a runtime `if`/`else` inside the unrolled block loop, a construct the served kernels do not use. Either failure shows in the harness, before any boot.
- The bound (B) assumes the generator's rewind behaviour on a recurrent model (replay from a page boundary). A future non-recurrent QSA model with partial-page reuse would hit the `copy_page` error instead of wrong output. A future in-place GDN rollback of more than a verify window (`rollback_capacity() > 0`) would break (B); the served code has none.
- Verify windows longer than 17 tokens (draft depth ≥ 17) raise in the eager path with the flag on. The daily uses depth 3 (policy `[[4, 3], [8, 1]]`).
- The ring rows of warmup page 0 alias across the synthetic block table, as the full plane's did; the values are zeros either way.
- `vram_census` attributes `qsa_planes` by attribute name; its per-token figures will show 20 B for `raw_k` with the flag on.
- Defrag moves ring pages like every other cache tensor (`cache_rotate` works on the page's byte size, 5,120 B here, 16-byte aligned).

## Status

Files in this directory:

| file | content |
|---|---|
| `served-source.patch` | 436 lines over four files: `cache/qsa.py`, `modules/qsa_indexer.py`, `modules/attention_fn/qsa_triton.py`, `modules/attention_fn/bc_attn.py` |
| `overlay/install.py`, `overlay/manifest.json` | hash-pinned install (baseline and post-patch sha256 per file, patch sha256, `patch -p1 --fuzz=0`, aborts on any mismatch or `.rej`); keeps the prebuilt extension |
| `Dockerfile.box` | `ARG BASE=tabbyapi:nvme-tier-r4-e3det`; install, then asserts the selector reads `False` unset and `True` with `=1`, the ring symbols exist and the served extension (`exl3_moe_prefill_e3_det`) is still the one loaded; no rebuild, no heredocs |
| `tests/test_rawk_ring_cpu.py` | 32 pytest cases, CPU torch 2.14 |
| `tests/rawk_scenarios.py` | the generator-shaped call schedules (9 positive, 2 negative controls), shared by the CPU tests and the GPU harness |
| `tests/fake_triton.py` | sequential CPU interpreter of the `tl` subset the plane kernels use, with a per-launch write-race detector |
| `tests/gpu_rawk_ring.py` | GPU harness (`--device`, `--json`, exit 1 on any mismatch) |
| `box-ab-spec.md` | build, harness on both cards, flag-off and flag-on fingerprint boots, long greedy identity, pool ladder, A/B, gates, NVMe tier sequence |

Patch application: `patch -p1 -F0 --no-backup-if-mismatch` on a full scratch copy of the served tree exits 0 with no reject and no `.orig` (macOS `patch`; the box image uses GNU patch). A second application with `--forward --dry-run` fails, as it should. The patched files are byte-identical to the working tree the patch was generated from. The four files are disjoint from decode-kernels r6, gdn-state-bf16 r1 and adaptive-draft r1.

CPU tests: 32 passed (`python -m pytest -q -p no:cacheprovider tests/test_rawk_ring_cpu.py`, 6 s). They cover: patch and manifest pins, exit code, install on a scratch root, refusal of a drifted baseline; every served function and class unchanged except the ring branches, and the flag-off paths of `update_planes` and `_configure_qsa` equal to the served source once the ring branch is removed; the ring kernels' runtime parameter lists equal to the served kernels' (the C++ argument vectors and graph patch indices); the AOT signature dicts equal to the kernel parameters; the ring pool kernels' arithmetic textually identical to the served kernel from the fp16 mean to the stores, and the loops differing only in the row address; eager launch order pool-then-append; the verify-window guard; allocation through the real `CacheLayer_qsa`/`CacheLayer_qsa_quant` classes with a stubbed extension (flag unset, `0` and empty give the served shapes, dtypes, storage and page bytes; flag on gives `[pages, 20, 128]`, storage smaller by `pages × 236 × 128 × 2`, a 5,120-byte ring page that is 16-byte aligned for `cache_rotate`, block-aligned `copy_page` without raw rows, misaligned `copy_page` refused before any write); `engine_identity()` listing the flag; the byte and pool arithmetic; the nine schedules run through the interpreter on the actual served and patched kernel sources with pooled planes `torch.equal` after every call and no write race; both negative controls mismatching; the interpreter's served pooling against an independent torch reference (atol 2e-3); the ring append keeping exactly the last 20 rows. Two mutations of the patched launcher (append before pool; an off-by-one ring slot) were each caught by at least two schedules.

Not done here, and required before any promotion: the GPU harness on both cards (the first real Triton and AOT compile of the ring kernels), the flag-off and flag-on fingerprint boots, the long greedy identity, the ladder, the A/B and the tier sequence (`box-ab-spec.md`).
