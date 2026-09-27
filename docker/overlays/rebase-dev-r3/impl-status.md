# impl-status — rebase-dev r3 (r2 port + upstream dev `73a6229..5783a93`, v1.5.2)

2026-09-27. Local work only: no GPU, no nvcc, no docker. The box build is the first real compile.

In this repository: `out/` is not carried; `prepare-tree.sh` rebuilds `out/rebase-dev-r3/` from upstream `5783a93` and `ported-vs-dev.patch` and checks the tree SHA-256 below. `Dockerfile.vanilla`, `out/dev-vanilla/` and `tools/` (the clang harness and its outputs, which record local paths of the machine it ran on) are not carried either; the harness results are quoted below. The box build, pool and speed results are in [`bench/results/r784-rebase-dev-r3.md`](../../../bench/results/r784-rebase-dev-r3.md). The r2 port map is [`port-map-r2.md`](port-map-r2.md).

## Inputs (exact)

| what | value |
|---|---|
| previous port | rebase-dev r2: `r2/out/rebase-dev-r2/exllamav3`, 585 files, tree sha256 `ca252232e05467282c4abdfa2f79dab99a696b4d46b5ea23e3efb5e1fd87759f` (re-verified 2026-09-27), = served `stack-r3-rows32` on upstream `73a6229524748be023d651b39a83e07abcdb8256` + tokcount-r1 |
| upstream target | exllamav3 `dev` HEAD `5783a9360b749a1d5bde6862dc35e7766e542ca1` (2026-09-27 00:31 +0200, "Loader: Reuse file streams across threads (Linux only)", version 1.5.2) |
| upstream range | `8cb1009` merge PR #325 (Turing sm_75, 23 PR commits `fb1be5d..9f811ba`), `f4db698` rework of #325 (per-device budget, ladders, eager fallback), `988fa66` v1.5.2, `5783a93` loader file streams. 30 paths, +1,061 / −235 |
| ported tree (r3) | `out/rebase-dev-r3/exllamav3/`: 587 files, tree sha256 **`150497b44e6f49b06b8a13f2568e68fc41841bc28964928509e181b633285501`** (`find exllamav3 -type f \| LC_ALL=C sort \| xargs sha256sum \| sha256sum`) |
| vanilla tree | `out/dev-vanilla/exllamav3/` = `git archive 5783a93 exllamav3`, 564 files, tree sha256 **`3f7b139b86f0aba7a860278c61c5c93ed2a944901660f64f6e775004dd6786ad`**; `out/dev-vanilla/UPSTREAM_SHA` |
| build base | `tabbyapi:stack-r3-rows32-tokcount-loopthink4` (served stack-r3-rows32 → tokcount-r1 in site-packages → loop-think r4 in `/app`, [`docker/overlays/loop-think-r4`](../loop-think-r4/)). The r3 image replaces the whole site-packages exllamav3 (r3 carries tokcount-r1 itself) and keeps `/app` |

## What was done

1. Scratch clone of upstream dev. Commit `r2` = r2's `exllamav3/` tree on parent `73a6229` (all other paths as at 73a6229). `git diff 73a6229 r2 -- exllamav3` is byte-identical to `r2/ported-vs-dev.patch` (cmp), so the synthetic commit is exactly r2. (It carries r2's one cosmetic artefact, `architecture/solar_open_moe.py` mode 755 → 644; the tree sha256 ignores modes.)
2. `git merge 5783a93` on `r2`. The merge-base is 73a6229, so git applied exactly the four upstream commits. **One conflicted file** (below); `bindings.cpp`, `exl3_gemm.cu` (rest of it), `bc_attn.py`, `bc_dsa.py`, `bc_mla.py`, `dsa_triton.py`, `hyperconnections.py` were two-sided and auto-merged.
3. **Proof that r2 → r3 is upstream and nothing else:** for each of the 30 upstream paths, the `+`/`−` lines of `git diff r2 r3` equal those of `git diff 73a6229 5783a93` (hashed per file). All 30 equal, including `exl3_gemm.cu`, whose resolution produces the same line set. So `r2-to-r3.patch` (28 files under `exllamav3/`, +927 / −235) is upstream's four commits verbatim. `doc/env_vars.md` and `tests/test_sm75_gemm.py` are outside `exllamav3/` and not exported.
4. Exported `out/rebase-dev-r3/exllamav3` (r2's 585 files + `exllamav3_ext/arch.cuh` + `modules/attention_fn/smem.py`), `out/dev-vanilla/`, `ported-vs-dev.patch` (`git diff 5783a93 r3 -- exllamav3`: 77 files, +15,758 / −341, identical shape to r2's vs 73a6229), `r2-to-r3.patch` (`git diff r2 r3 -- exllamav3`).
5. Adapted the build: `Dockerfile.box` (FROM `...-tokcount-loopthink4`), `landing_r3.py`, `rebuild-native.py` (byte-identical to r2's; upstream's `setup.py` did not change in the range), `.dockerignore`, tests (below). Beyond the deliverable list: `Dockerfile.vanilla` (the upstream arm, `tabbyapi:dev-vanilla-r3`) and `tools/` (the harness and its outputs). Unlike r2, both Dockerfiles COPY `landing_r3.py` and `tests/` AFTER the extension build RUN, so editing either reuses the cached nvcc layer.

## The conflict and its resolution

| file:line (r3) | ours (r2) | upstream | resolution |
|---|---|---|---|
| `exllamav3_ext/quant/exl3_gemm.cu:173-176` (`exl3_gemm_gr`) | `int* locks = DevCtx::instance().get_locks(device) + lc_gemm_locks_offset;` (densegemm lcguard) | `int* locks = DevCtx::instance().get_locks(device);` + `int smem_max = DevCtx::instance().get_smem_request(device);` | both: our locks line, then upstream's comment + `smem_max`. Every `SMEM_MAX` → `smem_max` in the file auto-merged beside the densegemm V2 twin blocks (`:282`, `:313`, `:331`, `:341`; mgemm `:553`, `:643`, `:676`, `:708`, `:718`); the twin gates `if (!half_k && ...)` are unchanged <!-- prose-ok: C++ negation in code --> |

No served mechanism was touched by upstream outside this line: the served kernels (hc_mix V2/V3, densegemm V2, moe coop V2/V3/rows32, latchain, E3) and the served `bc_attn.py` constexprs (`BLOCK_ROWS`/`ROWS_SUB`/`D_SUB`/`V_DIM`, `num_stages = _LC_QSA_SPLIT_STAGES` / `_LC_QSA_COMBINE_STAGES or 1`) are byte-identical to r2 (step 3).

Two-sided merges worth naming, all auto-merged and checked:
- `bindings.cpp`: upstream's `exl3_gemm_shape_compat` kwargs (`half_k = false`) and `g_get_smem_max` land next to our binding groups; served revision markers intact.
- `bc_dsa.py` / `bc_mla.py` / `dsa_triton.py`: r1-era `SHARED_BOUNDS` constexpr on `_dsa_indexer_fewq_kernel` (ours, default 0) + upstream's ladder in `dsa_indexer_scores`, which launches the kernel without `SHARED_BOUNDS`, so it takes the default 0 = upstream behaviour.
- `hyperconnections.py`: upstream only adds `and torch.cuda.get_device_capability(dev)[0] >= 8` to the tiled gate (`:403`); served V2/V3 dispatch untouched.
- `bc_attn.py`: upstream adds `smem_limit` + `BCKernelTooLarge` in `_compile_kernel` (`:127-132`) and a try/except around `_configure` in `forward` (`:681-684`); every one of the 12 served compile sites runs inside `_configure` (`_configure_qsa` and `_qsa_ring_graph_kernels` are reached from it, `:451`, `:518`).

## sm_120 answer (step 9): nothing changes on the 5090

The 5090 is CC 12.0 with a 99 KiB opt-in (101,376 B, CUDA CC table). The operator can confirm on the box with `python3 -c "import torch, exllamav3_ext as e; print(e.g_get_smem_max(0), torch.cuda.get_device_properties(0).shared_memory_per_block_optin)"` (expect `101376 101376`).

| mechanism (r3 file:line) | on sm_120 | why |
|---|---|---|
| EXL3 GEMM/mgemm launch smem (`exl3_gemm.cu:176,553` → `exl3_devctx.cu:64-66`) | 92,160 B, as before | `get_smem_request = MIN(optin, EXL3_SMEM_MAX_DEFAULT)` = MIN(101376, 90·1024) = 92,160 = the old `SMEM_MAX` (`exl3_gemm_inner.cuh:7`) |
| autotune shape filter (`exl3_kernel_map.cu:99-111`) | keeps every shape; = the old divisibility test | largest footprint over shapes 1–4 × K 1..8 × {int, +0.5} is 69,632 B (shape 4, K 8.5) ≤ 92,160 (table below; `test_flag_selection_r3.py` recomputes it from the `EXL3_GEMM_SHAPE_n` macros). Same candidate set → same autotune picks |
| forced-shape gate `exl3_gemm_check_smem` (`exl3_kernel_map.cu:116-124`, called `:198`, `:234`) | never throws | same table |
| autotune key `gemm_autotune_hash` | unchanged in the range | **r2's `TUNEDIR` is reusable for r3** (r2 risk #2 does not recur) |
| MoE launch (`exl3_moe.cu:272,308,378`) | 92,160 B, as before | same `get_smem_request` |
| served densegemm V2 twins (`exl3_dense_v2.cu:203,296,326`) | still request `SMEM_MAX` = 92,160 | not touched upstream; equal to the request cap on sm_120 |
| int8 GEMV gate (`exl3_gemv_int8.cu:273`) | taken as before | declines only below 80 KiB opt-in |
| deterministic router GEMM (`routing_gemm.cu:253-256`) | taken as before | declines only for `cc_major < 8` |
| tiled HC prefill (`hyperconnections.py:403`) | on, as before | `get_device_capability(dev)[0] >= 8` (12) |
| `ptx.cuh`, `exl3_gemv_kernel.cuh`, `det_gemm.cuh`, `hgemm_f16acc.cu` | identical device code | all new code is under `EXL3_SM75` (`__CUDA_ARCH__ < 800`, `arch.cuh:22-26`) or `__CUDA_ARCH__ >= 800` guards whose true branch is the old asm |
| Triton ladders (`smem.py:54-81`; `triton_paged.py:1235,1967,2265`; `dsa_triton.py:1086,1138,1209,1226`) | stock config, one compile-only probe per key | `pick_config` returns the first candidate that fits; every ladder leads with the stock config; the limit is `props.shared_memory_per_block_optin` (`smem.py:45`), which torch binds for CUDA builds from 2.7 on (upstream PR commit `1873964`; `torch/csrc/cuda/Module.cpp` v2.14.0, `#ifndef USE_ROCM`), so the 96 KiB fallback (`smem.py:47`) is not taken on the image's torch (≥ 2.11 per the repo's image notes; `landing_r3.py` prints the torch version and whether the attribute exists, and puts a WARNING in its last line if not). A stock config above 101,376 B could not have launched before either (Triton raises OutOfResources at load), so on the served paths stock always fits |
| `paged_attn_triton_prefill` rule change (`triton_paged.py:1882`) | no effect | was `if qc is not None:` (overrode a caller's `block_n`), now `... and block_n is None`. No in-tree caller passes `block_m`/`block_n`/`num_stages`/`num_warps`/`block_h` to any Triton attention entry point (AST scan in `test_flag_selection_r3.py`) |
| **BC graph capture** (`bc_attn.py:127-132`, `:681-684`) | **every served graph kernel still captures** | `_compile_kernel` declines only when `ck.metadata.shared > smem_limit(device)` = the opt-in. `triton_kernel.cpp:20-21` already sets `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = shared_bytes` under `cuda_check_drv`, which fails for anything above the opt-in, so every kernel the served image captures today satisfies `shared ≤ opt-in` and cannot hit `BCKernelTooLarge`. The served constexprs and stage knobs are byte-identical to r2 |
| unused by Qwen3.8-Flash-Next | — | `mla_triton.py`, `mla_attn.py`, `dsv4.py`, `bc_mla.py`, `bc_dsa.py` ladders and `dsa_attn` (DeepSeek/MLA paths; `qwen4_exp*.py` imports none of them). The QSA indexer reaches only `dsa_indexer_scores` (`qsa_indexer.py:291-322`, eager prefill; decode is the BC kernel at `bc_attn.py:555`) |

GEMM shared-memory footprint (bytes, `exl3_gemm_smem_bytes`, `shmem_out_had` makes no difference because `tilesize_n · tilesize_m` = the reduction scratch for every shape):

| shape (M,K,N,stages) | K=2 | K=3 | K=4 | K=8.5 (max) |
|---|---|---|---|---|
| 1 (16,16,128,6) | 14,336 | 15,872 | 17,408 | 24,320 |
| 2 (16,32,128,4) | 16,384 | 18,432 | 20,480 | 29,696 |
| 3 (16,32,256,4) | 28,672 | 32,768 | 36,864 | 55,296 |
| 4 (16,16,512,4) | 43,008 | 47,104 | 51,200 | 69,632 |

**New host work per launch** (not a blocker; for the record): `select_exl3_gemm/mgemm_kernel` now calls `cudaGetDevice` + a `DevCtx` mutex (`exl3_gemm_check_smem`) on non-autotuned launches; `exl3_gemm_shape_compat` does the same inside the autotune candidate loop; `routing_gemm_det_fits` adds a `cudaDeviceGetAttribute`; each eager Triton attention launch adds `pick_config`'s dict lookup + `torch.cuda.get_device_capability`. Under the BC graph (decode) these run at capture only; in eager prefill they are microseconds against a chunk of hundreds of ms.

## VRAM (step 10): nothing in the four commits moves R741's numbers

R741 measured r2 at ~550 MiB more per card pair than the served stack at pool 983,040: tiled HC prefill 342/300 MiB, `EXL3_AUTOSPLIT_PREPARE` 86/62, residual 118/110 unattributed. Read against the four commits:

- **No new device allocation on sm_120.** `DevCtx` gains an `int smem_max[MAX_DEVICES]` (host). The tiled HC gate adds a capability test only (tiled stays on, same workspaces). `EXL3_AUTOSPLIT_PREPARE` / `model_ls.py` are not in the range.
- **Triton wrappers:** `paged_attn_triton_decode` / `_prefill` / `varlen_attn_triton` moved their partial-buffer allocation into `prepare(cfg)` with the same sizes as before. On the first call per key the probe allocates them once, drops them (the args tuple dies when `shared_bytes` returns), and the launch's `prepare` re-allocates the same size from the caching allocator: no new peak, and later calls skip the probe. On sm_120 only the stock candidate is probed. `dsa_attn`'s workspace-per-candidate (`dsa_triton.py:1046-1050`) could hold two sizes in `g_tensor_cache` only if the ladder stepped down, and `dsa_attn` is not on the Flash-Next path.
- **Loader (5783a93):** on Linux `stloader_open_file` opens one `FILE*` per shard and repeats it for the 8 workers instead of 8 `fopen`s; all reads are `pread` (`stloader.cpp:87`, via `read_range`, 3 call sites), so no stdio buffer is ever allocated. Host descriptors only; the pinned staging pool is unchanged. No VRAM effect.
- **Binary:** a few new host functions and `static_assert`s; no new kernels for sm_120, so no module-load memory change (lazy loading).

So r3 should reproduce r2's pool deficit. The operator's evidence that settles it: the R741-style P search at pool 983,040 on r3 (expect r2's number within noise), or `nvidia-smi` per-card used MiB after load at the same pool on r2 and r3 images.

## TabbyAPI pairing (loop-think r4, coordinator's check)

- **Signatures:** no signature TabbyAPI calls changed in `73a6229..5783a93`: the range touches no file under `generator/`, `model/`, `cache/` or the package `__init__` (`git diff r2 r3 -- exllamav3/generator` is empty). `test_tabby_callsites.py` passes on r3 (65 import/call/probe sites over 7 TabbyAPI files, was 59 over 5 in r2) and on dev-vanilla with `ALLOW_ABSORBED=num_draft_tokens_by_batch` (the known r2 gap, unchanged).
- **The three loop-think r4 dependencies, confirmed in the r3 tree (byte-identical to served `stack-r3-rows32` for `generator/loop_detect.py`; `job.py` differs from served only in the tokcount-r1 line; `async_generator.py` only in upstream's `cpu_page_cache.close()` from r2's merge):**
  1. `mc.max_rq_tokens` is TabbyAPI's own attribute (`backends/exllamav3/model.py:433`, `chunk_size` when output chunking). Its engine meaning is unchanged: `Job(max_rq_tokens = ...)` requeues at `new_tokens > max_rq_tokens - max_num_draft_tokens` (`job.py:636`, `:727`), aligns it to a page/recurrent boundary (`:1118-1128`), and the requeued job gets `stop_on_loop = self.stop_on_loop` (`:1083`), so the engine's detector restarts at every requeue, which r4's `2W <= max_rq` guard relies on.
  2. `exllamav3.generator.loop_detect.LoopDetector(window_size, max_period)` (`loop_detect.py:10`): `max_period = min(max_period or W // 3, W // 2)`, `feed_many()` returns the detection indices or None (`:124-132`), `_total` counts tokens fed. Collector calls `LoopDetector(W, W // min_reps)` and `LoopDetector(3L, L)` fit. Job builds `LoopDetector(window_size, window_size // min_reps)` from `stop_on_loop` (`job.py:317-320`), the same (W, reps) contract r4's `(2W, 4)` backstop uses.
  3. `constrain_output_now`: `Job.constrain_output_now` (`job.py:497`) and `AsyncJob.constrain_output_now` (`async_generator.py:174`, forwards to the job). TabbyAPI reaches it through a `hasattr` probe (`model.py:1152`) that would silently disable the reasoning budget and the loop injection if it were missing; the audit now fails on that.
- The same three hold on dev-vanilla 5783a93 (upstream code; `job.py:318`, `:1080`, `async_generator.py:171,177`).
- **Audit widened** (`tests/test_tabby_callsites.py`): besides `backends/exllamav3`, it scans `TABBY_EXTRA` (default `tests/tabby-extra`, TabbyAPI's two files with loop-think r4's `fix.patch` applied; in the image `/app/endpoints/OAI/utils/chat_completion.py:/app/common/sampling.py`) for exllamav3 imports and constructor calls; checks every `hasattr` probe for `constrain_output_now` / `error` against the engine classes; and runs the loop-think contract behaviourally (a period-3 loop is detected once an 8-token window fills; `_total`; `Job` params; detector rebuilt per requeue; `AsyncJob` forwarding). `tabby-backends/model.py` is byte-identical to loop-think r4's pristine `base/backends/exllamav3/model.py` (r4 does not touch backends).
- `landing_r3.py` also re-runs the base image's own `/opt/loopthink-r4/landing_loopthink.py` and (r3 only) `/opt/tokcount-r1/landing_tokcount.py` against the new exllamav3.

## Verification done here

- **Clang CUDA syntax harness** (`tools/clang-harness.sh`; Homebrew clang 23.1.1, `-x cuda --cuda-gpu-arch=sm_120 -fsyntax-only -std=c++20`, CUDA 12.8.x headers from the pypi wheels + conda-forge cusparse/cusolver headers, torch 2.14.0 headers + c10/cuda from pytorch v2.14.0, host side macOS libc++; the flags and two header work-arounds are documented in the script). All 188 TUs of both trees: r2 22 failing / 49 error kinds, r3 22 / 49. **0 kinds new in r3, 0 gone, identical per-TU exit codes, no TU hit the error limit** (`tools/harness-kinds-r2.txt`, `tools/harness-kinds-r3.txt`, `tools/harness-rc-r3.txt`). The 49 baseline kinds are harness noise present in both: clang rejects `ptx.cuh`'s `"n"(bytes)` asm operand (nvcc accepts it), macOS lacks `intrin.h` / `_fseeki64` / `__int64` (stloader's non-Linux branch is Windows code), plus `lm_clamp_`/`max` overloads and a `half2` assignment. The instance TUs (`exl3_comp_unit_*`, `exl3_dense_v2_inst_*`) parse clean, so upstream's new per-instantiation `static_assert(exl3_gemm_smem_bytes(...) == layout)` (`exl3_gemm_inner.cuh:82-88`) holds for every compiled GEMM instance, and the new `#include "exl3_kernel_map.cuh"` in `exl3_gemm_inner.cuh` resolves in every TU that pulls it (including `exl3_gemm_inner_v2.cuh:57` → `exl3_dense_v2_inst_*.cu`).
- **Harness blind spots:** device code generation (ptxas registers, SASS), the asm-heavy TUs, the real link, and Linux-only host branches (`#ifdef __linux__`, e.g. the new `streams = 1` path in `stloader_open_file`) are parsed as macOS. The box build is the first real compile.
- **CPU tests** (`tests/run_cpu_tests.sh`, torch 2.14.0 CPU, extension and triton stubbed): results in "CPU suite result" below. `test_flag_selection_r2.py` is r2's suite, run on the r3 tree; its only change is the version check (1.5.2). It includes the reader map: **all 41 live launcher keys are read by exactly the files that read them in the served tree**. New `test_flag_selection_r3.py` (10 checks): version; GEMM footprint table ≤ 92,160; the exl3_gemm.cu resolution (lcguard offset + 2× `smem_max`, no launch left on `SMEM_MAX`, V2 gates, V2 twins still on `SMEM_MAX`); `pick_config` stock-first / one probe per key / cached (with a fake 99 KiB sm_120); the torch opt-in binding (n/a on CPU torch; reported by landing_r3.py in the image); 12 BC compile sites + served constexprs + the decline condition; no forced Triton tiles in-tree; the HC tiled gate; stloader pread-only; `EXL3_TRITON_SMEM_*` unset.

### CPU suite result (final run, 2026-09-27, after the last edit; `tools/cpu-tests-r3.txt`)

`run_cpu_tests.sh` on `out/rebase-dev-r3`, EXL3_* stripped: flags 21/21 (r1 suite), r2 flags 14/14 (includes the 41-key reader map: **every live launcher key is read by exactly the files that read it in the served tree**), r3 10/10, cache/rewind 8/8, E3-DET 15/15, pagetable 4/4, tabby surface 6/6, tabby call sites PASS (65 import/call/probe sites over 7 TabbyAPI files). `ALL SUITES: 0`. Also run here: the call-site audit on `out/dev-vanilla` with `ALLOW_ABSORBED=num_draft_tokens_by_batch` PASS; a negative control (r3 tree with `AsyncJob.constrain_output_now` renamed) FAILS with 2 findings (the hasattr probe and the contract), so the audit catches a silently disabled loop-think; `landing_r3.py`'s `loopthink_contract()` executed under `_runner` against both `out/rebase-dev-r3` and `out/dev-vanilla`: OK. The rest of `landing_r3.py` is r2's landing with names/version changed and runs only in the image.

## Flag status

Unchanged from r2 (r2 `impl-status.md`, "Flag status"): every served key is present, flag-selectable, and read by the same file(s) as in served. Upstream adds two debug-only knobs, `EXL3_TRITON_SMEM_LIMIT` (caps the Triton limit; would shrink served tiles, must stay unset: the Dockerfile and `landing_r3.py` assert it) and `EXL3_TRITON_SMEM_DEBUG` (prints ladder picks and BC declines; harmless, useful to prove on the box that nothing declines: run one boot with `EXL3_TRITON_SMEM_DEBUG=1` and grep the log for `declining to eager` / `over`).

## Build

```
B=tabbyapi:stack-r3-rows32-tokcount-loopthink4
docker build -f Dockerfile.box --build-arg BASE=$B \
  --build-arg BASE_ID=$(docker image inspect $B --format '{{.Id}}') \
  --build-arg DGV2_NVCC_DEFS="$(docker image inspect $B --format '{{index .Config.Labels "local.stack.dgv2_defs"}}')" \
  --build-arg TREE_SHA=150497b44e6f49b06b8a13f2568e68fc41841bc28964928509e181b633285501 \
  --build-arg MAX_JOBS=4 -t tabbyapi:rebase-dev-r3 .
docker build -f Dockerfile.vanilla --build-arg BASE=$B --build-arg BASE_ID=$(docker image inspect $B --format '{{.Id}}') \
  --build-arg TREE_SHA=3f7b139b86f0aba7a860278c61c5c93ed2a944901660f64f6e775004dd6786ad --build-arg MAX_JOBS=4 -t tabbyapi:dev-vanilla-r3 .
```

Build context = this directory after `bash prepare-tree.sh` (`.dockerignore` drops `*.patch`, `*.md`, `tools/`). Landing last lines: `rebase-dev-r3 landed: ...` / `dev-vanilla-r3 landed: ...`.

## Risk list

1. **No device compile here.** Register or shared-memory overflow or a ptxas failure surfaces only in the box build. r3 adds no TU and no kernel for sm_120; upstream's new `static_assert` in `exl3_gemm_inner.cuh` is evaluated by the harness and holds.
2. **Pool.** Expected equal to r2 (the VRAM section). R741's P search result for r2 applies until the box says otherwise.
3. **Triton fallback limit.** If the image's torch lacked `shared_memory_per_block_optin` (torch < 2.7), `smem.py` would use 96 KiB and a served BC kernel between 96 and 99 KiB would decline to eager (slower decode, no error). Reported, not fatal: the landing's last line carries `WARNING: torch lacks shared_memory_per_block_optin` in that case, and one boot with `EXL3_TRITON_SMEM_DEBUG=1` settles whether anything declines.
4. **A future non-integer or larger-shape pack** would meet upstream's filter only below 90 KiB, which is also the old `SMEM_MAX`: no change in behaviour from r2 on sm_120.
5. **`EXL3_TRITON_SMEM_LIMIT`** set by accident (e.g. copied from an upstream doc into EXTRA_ENV) shrinks tiles silently. Guarded at build and landing, not at launch: the launcher's EXTRA_ENV must not carry it.
6. **Carried from r2 unchanged:** the numerics changes by upstream (fingerprints differ from S by design), half-integer bitrates silently take upstream kernels, the pinned-draft-buffer inconsistency with `EXL3_DRAFT_PINNED_STAGING` unset, EP incompatibilities, dev-vanilla needs `DRAFT_POLICY=` (empty).
7. **Loader stream sharing (5783a93)** is correct only because every read is positional; a future served change that `fseek`/`fread`s a handle from a worker would race on Linux. `test_flag_selection_r3.py` pins "no fseek, 3 read_range sites".

## Not done / unresolved

- No GPU work (by instruction). The build, pool, decode and prefill answers come from the box.
- No conflict needed evidence beyond the code: the single conflict is a two-line union.
