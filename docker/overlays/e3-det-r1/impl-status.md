# e3-det r1: deterministic E3 grouped MoE prefill (implementation status)

**Status:** implemented and packaged. Checked locally on CPU and with clang's CUDA front end. Nothing has been compiled with nvcc or run on a GPU.

**Flag:** `EXL3_MOE_PREFILL_E3_DET=1`, literal `1` only.

- It is a module global, read by `forward()` at call time, like `MOE_PREFILL_E3`, so tests can flip `block_sparse_mlp.MOE_PREFILL_E3_DET` in-process.
- It only acts where atomic E3 would run: `MOE_PREFILL_E3` on and a chunk of at least 512 rows.
- Unset, the image issues exactly `tabbyapi:stack-r4-e3r2`'s extension calls with the same arguments, in the same order. The existing kernels are untouched: every CUDA/C++ change is an addition.

**Base:** `tabbyapi:stack-r4-e3r2`. `src/` equals that image: all five baseline sha256s match `ref/stack-r4-e3r2/manifest.json` `post`.

## Design

### Two sources of order-dependence, not one

With E3 on, two tiers add into `final_hidden_states` with float atomics:

1. The E3 down epilogue: `e3_atomic_float4` for experts with more than `thin_rows` (32) rows.
2. The served fused kernel `exl3_moe` for the thin experts (1 to 32 rows). The E3 branch runs it with `scratch=None`, because `FUSED_DET and not e3_active` disables its slot mode. Its epilogue `had_hf_r_128_d_inner<true>` is an `atomicAdd` into the token row.

Both have to go. Fixing only the E3 down kernel would leave the thin-tier atomics, and prompts would still diverge.

### Chosen: per-assignment slots plus one fixed-order reduction

Each routed assignment owns one slot. The slot row is the assignment's position `p` in the expert-sorted order. Order on the stream:

1. **Phase 1**, `exl3_moe_prefill_e3_det`:
   - The existing metadata, gather and gate/up launches, unchanged and with the same arguments.
   - A new index-prep launch:
     - `row_pos[fat_row] = p`
     - `expert_start[e]` (the thin tier's slot base)
     - `inv_order[order[p]] = p`
   - A down kernel with the atomic epilogue replaced by a store: `e3_down_det_kernel`. It writes the **half-rounded, pre-Hadamard** output into fat slot `p`, as 2 bytes per element.
2. **Thin tier:** the served `exl3_moe` kernel in its existing slot mode (`output_scratch = slots`, `fused_base = expert_start`), over the same two count windows as today (`[1,16]` m16, `[17,32]` m32). It writes the final weighted fp32 contribution into thin slot `p`.
3. **Phase 2**, `exl3_moe_prefill_e3_det_reduce`: one warp per (token, 128-column block).
   - For k = 0..9 in router top-k order, it reads assignment `a = t*10+k` from slot `inv_order[a]`.
   - Fat slots get the atomic kernel's epilogue, operation for operation: `e3_had_float_raw`, then `×(HAD_SCALE·route)`, then `×svh[e]`. Thin slots are used as stored.
   - It sums from `+0.0f` with `__fadd_rn` (`__fmul_rn` for the scales, so nvcc cannot contract a scale into the sum) and **stores** the row.

No floating-point atomics remain, and every output value comes from one thread in a fixed order. The result is a pure function of the inputs:

- Integer histogram.
- Stable argsort under DET.
- Integer scans.
- Per-row GEMM arithmetic, which is independent of CTA scheduling: each (segment, N-tile) belongs to one CTA, E3 has no split-K, and the thin tier's arithmetic was already bitwise stable in FUSED_DET mode (R524: E3-off repeats exact).

### Why store half and pre-Hadamard for fat rows

The atomic epilogue does `hv = e3_float4_to_half4(acc)` before the Hadamard (`exl3_moe_prefill_e3.cu`, `e3_down_kernel`). The half value is therefore already the exact input of everything that follows, so storing it drops no information.

It also halves the slot traffic of fat rows compared with fp32 post-epilogue slots. Fat assignments are most of them at 2,048 rows: about 40 rows per expert on average against a threshold of 32.

The fused thin tier's epilogue (`had_hf_r_128_d_inner`) is the same operation sequence on its own half pre-Hadamard row. Its slot mode stores the identical value it would have added atomically.

Consequence: for every assignment, the contribution DET sums is bitwise the contribution atomic E3 adds. **DET and atomic E3 differ only in summation order.** This gives the box a sharp correctness check. NRMSE(DET, atomic) should be at float-reassociation level, about 1e-7; `gpu_e3_det.py` gates it at < 1e-5. Any slot, index or column-map bug would show up as order-1 error.

### Scratch memory: zero net VRAM

The slot scratch is fp32 `[A, 2560]` (A = assignments = rows × 10). That is 209.7 MB at 2,048 rows, the brief's figure.

Atomic E3 already allocates `h13_g` and `h13_u`, two half `[A, 2560]` buffers of the same 209.7 MB total. Neither is read after the gate/up kernel: the down kernel reads only `h2`. DET allocates one fp32 buffer and carves `h13_g`/`h13_u` out of its two halves. The same bytes then serve as the slots, and stream order makes this safe:

1. gate/up is the last reader of h13;
2. `e3_down_det` writes fat slots after it;
3. the thin tier writes thin slots after that (so it must run after phase 1, not before as in atomic E3);
4. then the reduction.

The only extra memory is `row_pos` (A×4), `inv_order` (A×8) and `expert_start` (513×8): 0.25 MB at 2,048 rows.

With the prefill pipeline, each card's chunk allocates its own per-call workspace through the caching allocator, exactly as atomic E3 does today. The per-card peak is unchanged. For context, cuda:1 had 1,057 MiB free at boot and 569 MiB under load in R525.

Half-precision fat slots are not a numerics change: they are the atomic kernel's own values. fp16/bf16 *post*-epilogue slots would change numerics, and the design does not use them.

### Expected cost

This is an estimate, not a measurement. Per MoE layer and 2,048-row chunk, A = 20,480, with f = the fraction of assignments in fat experts (estimated 0.85-0.95):

| | atomic E3 today | DET |
|---|---|---|
| fat epilogue | 13.1M `red.v4.f32` into a 21 MB `out` (L2-resident) | 2 B/elem store: f × 105 MB |
| thin epilogue | scalar `atomicAdd` into `out` | fp32 slot store: (1−f) × 210 MB |
| reduction | — | read f × 105 + (1−f) × 210 MB, write 21 MB |
| index work | — | 1 prep launch (40 blocks; block 0: 2 shuffle scans + row_pos) |

Extra DRAM traffic is about 230 MB per layer at f = 0.9, less where the reduction hits L2 (96 MB): about 0.10-0.15 ms at ~1.6 TB/s. The removed L2 atomics give some of that back.

Net: roughly +0.05 to +0.14 ms on a 2.7 ms E3 layer at 2,048 rows (R507), about 49 MoE launches per chunk. The expected end-to-end cold prefill change is **−1 % to −3.5 %**. That straddles the −3 % target, and I cannot promise it from here.

The kernel test prints `det_over_atomic` per layer and row count. End-to-end ≈ 0.6 × (ratio − 1), because MoE is about 60 % of a chunk.

If the box shows more than −3 %, round 2 has two levers:

1. **All-half slots:** a slot variant of the fused kernel's thin epilogue that stores the half pre-Hadamard row instead of the fp32 result. All slots become 2 B, which removes (1−f) × 105 MB of writes and of reads.
2. **Fold the reduction into the routed/shared-expert combine**, saving one pass over `out`.

**Found in passing (not changed, default path):** the served `e3_metadata_kernel` does its 512-expert scan on thread 0, with a dependent L2 load per expert. That is latency-bound, about 50-100 µs per layer, roughly 1-2 % of prefill in atomic E3 and in DET alike. The DET prep kernel uses a parallel shuffle scan instead. Doing the same in the metadata kernel would speed up E3 itself, but it changes the default path and belongs in its own round.

### Alternatives considered

- **Token-owned reduction** (one CTA owns a token block's full down projection across its experts). E3 is expert-major: 64-row segments of one expert, which is where its weight reuse and its +16-21 % come from. A token's 10 contributions come from 10 different segments. Token ownership means re-tiling E3 into the non-grouped path it replaced.
- **Sorted-segment reduction keyed by (token, slot).** This is what is implemented, with the key realised as `inv_order`.
- **All-fp32 post-epilogue slots**, the simplest option: it reuses `exl3_moe_gather` unchanged. It doubles fat-slot traffic, which is about +1.5 points of cost at the −3 % budget.
- **All assignments through E3** (thin_rows = 0, uniform half slots). This moves the 1-32-row experts off the fused kernel, whose 16/32-row tiles beat E3's 64-row tiles there. It also changes the thin experts' numerics. Unmeasured, and rejected for r1.
- **Order-independent integer (fixed-point) atomics.** No scratch, but it needs a global scale: overflow and underflow risk with unknown activation ranges. It has no vector form (4× the atomic ops), changes numerics, and still has to touch the served fused kernel's atomics.

## Changed files

All paths are under `overlay/exllamav3/`. `e3-det-r1.patch` is the unified diff against `src/`.

| file | change | lines (overlay) |
|---|---|---|
| `modules/block_sparse_mlp.py` | Flag at `:51-56`. Stable argsort under DET at `:1045-1051`. `run_fused(..., det_slots=None)` at `:1153-1161`; the default call passes the same `scratch, tables[0]` as before (`:1193`). DET branch at `:1199-1240` (alias allocation, phase 1, thin tier into slots, reduction). The atomic E3 branch at `:1241-` is unchanged. | 5 lines replaced, 63 added |
| `exllamav3_ext/quant/exl3_moe_prefill_e3.cu` | Additions only. Design comment `:662-679`. `e3_det_prep_kernel` `:683-765`. `e3_down_det_kernel` `:767-838`. `e3_det_reduce_kernel` (with its half2 helper) `:840-937`. Launch struct and dispatch `:939-977`. Entry points `exl3_moe_prefill_e3_det_cuda` `:1075-1187` and `exl3_moe_prefill_e3_det_reduce_cuda` `:1189-1217`. The atomic kernels and `exl3_moe_prefill_e3_cuda` (`:981-1073`) are byte-identical to the baseline. | +440 |
| `exllamav3_ext/quant/exl3_moe_prefill_e3.cpp` | Validating wrappers `exl3_moe_prefill_e3_det` `:142-258` and `exl3_moe_prefill_e3_det_reduce` `:260-295`: dtype, contiguity, shape, pinned geometry, device guard, thin range. | +147 |
| `exllamav3_ext/quant/exl3_moe_prefill_e3.cuh` | Declarations `:76-172` | +94 |
| `exllamav3_ext/bindings.cpp` | `m.def` for the two new functions, `:266-267` | +2 |

Packaging (round root):

- `manifest.json`: pre = src sha256, post = overlay sha256.
- `install.py`: verifies pre against the installed package and the payload, copies, then verifies post.
- `build_extension.py`: the stack's, unchanged.
- `Dockerfile.box`: `ARG BASE=tabbyapi:stack-r4-e3r2`, no heredocs. It ends with the import assertion for both new symbols plus a default-off check.
- `tests/`:
  - `test_e3_det_cpu.py`
  - `gpu_e3_det.py`
- `local_syntax_check/`: local-only compile checks (script and header stand-ins). It sits at the round root, outside `tests/`, so its fake `torch/extension.h` never lands in the image.

## Verified locally, and how

- **CPU tests** (`.venv`, torch 2.14 CPU, `python3 tests/test_e3_det_cpu.py`): **7 tests, all pass, 6.4 s.**
  - `IndexModel.test_random_routings`: 27 cases = rows {512, 1,024, 2,048} × routing {uniform, zipf, hot} × thin_rows {1, 16, 32}. Python ports of the unchanged metadata kernel, the new prep kernel (warp-shuffle scans step for step), the down-slot store, the fused slot mode and the reduction lookup. Checks:
    - every slot is written exactly once, by the right tier, for the right (expert, token, route weight);
    - the reduction reads assignment (t, k) at k;
    - the parallel scan equals the serial layout, and `expert_start` equals the cumsum;
    - the down store and the reduction read use the same (column block, lane) map;
    - the fat half region sits inside its slot row;
    - the slot bytes equal h13_g|h13_u;
    - segment count stays within the Python `seg_cap` bound.
  - `test_boundaries`: counts placed exactly at thin, thin+1, 16/17, 32/33, 64/65, 128/129 for thin ∈ {1, 16, 32, 64}.
  - `Dispatch`: the baseline `block_sparse_mlp.py` (src) and the overlay are loaded side by side with stub packages, and a stub extension that applies the C++ wrappers' checks and emulates the kernels at slot level. Checks:
    - **Flag off:** identical extension call sequence and arguments (dtype, shape, stride, content hash), identical argsort calls and bitwise-identical output against the baseline for E3 on at 512 and 2,048 rows, E3 off, and E3 on below 512 rows.
    - **Flag on without E3:** identical to the baseline.
    - **Flag on with E3:** call sequence `e3_det → exl3_moe[1,16] → exl3_moe[17,32] → e3_det_reduce`, with slot mode and `fused_base = expert_start`. No atomic E3 and no gather. Stable sort. Slots alias h13. The output matches the reference routed sum. The output does not change when fat rows are processed in shuffled orders.
- **Mutation check:** four deliberate bugs in the overlay's Python were each caught by the dispatch tests: `order` passed instead of `inv_order`, a zero slot base, a non-stable sort, and the thin tier moved before phase 1.
- **Compile checks without nvcc** (`local_syntax_check/run.sh`), using Homebrew clang 23 with NVPTX, the CUDA 12.8 headers from the nvidia wheels, and stand-ins for torch/c10/libcu++:
  - the full patched `exl3_moe_prefill_e3.cu` parses and instantiates for sm_120a, device side and host side, with `-Wall -Werror`: 0 diagnostics;
  - the `.cpp` wrapper and `bindings.cpp` parse against the real torch 2.14 headers (pybind `m.def` instantiated): 0 errors;
  - PTX is emitted for all kernels. The DET kernels contain no atomics by source construction. The PTX scan (no `atom.`/`red.` in them) adds little: clang without libdevice lowers `atomicAdd` to calls, so the baseline atomic kernel also scans as 0.
- **Packaging:**
  - `install.py` run against a copy of `src/` as fake site-packages: it installs, and the installed tree equals the work tree;
  - a second run against the already-patched tree fails with all five pre-hash mismatches, as it should;
  - `manifest.json` pre hashes equal the stack-r4-e3r2 post hashes;
  - `Dockerfile.box` has no heredocs;
  - no chat-template tags in any file.
- `src/` stays pristine. The `__pycache__` two early test runs wrote into `src/` was removed, and the test now sets `sys.dont_write_bytecode`.

## Only the box can verify

- The nvcc compile of the whole extension for sm_120, including register and shared-memory limits.
- The runtime correctness of the new kernels on real weights: NRMSE(DET, atomic) at about 1e-7.
- Bitwise determinism across repeats, under a concurrent side-stream load, across processes and across fresh boots.
- End-to-end greedy identity on the cold 8k and 30k prompts.
- The cold-prefill cost at 60k and 120k against the −3 % target.
- VRAM parity.
- That the c1 fingerprint does not move.

## Operator commands

```bash
# build (context = this directory)
sudo docker build --build-arg BASE=tabbyapi:stack-r4-e3r2 -f Dockerfile.box -t tabbyapi:e3-det-r1 .
# kernel + e2e (process 1), then kernel cross-process (process 2); full docker run lines in box-ab-spec.md
python3 /opt/e3-det-r1/tests/gpu_e3_det.py --model <ckpt> --kernel --e2e --out /out/gpu_e3_det_p1.json
python3 /opt/e3-det-r1/tests/gpu_e3_det.py --model <ckpt> --kernel --repeats 3 --iters 5 --warmup 2 \
    --compare /out/gpu_e3_det_p1.json --out /out/gpu_e3_det_p2.json
# served: live launcher with IMG=tabbyapi:e3-det-r1 and EXTRA_ENV="<live EXTRA_ENV> EXL3_MOE_PREFILL_E3_DET=1"
# local (this workspace, no GPU)
.venv/bin/python out/e3-det-r1/tests/test_e3_det_cpu.py
out/e3-det-r1/local_syntax_check/run.sh "$PWD"
```

## Risks

1. **nvcc-only compile issues.** clang accepted everything, but nvcc (with `--use_fast_math`, `-Xcudafe` suppressions) is a different front end. The likeliest spots are the `long long` `__shfl_sync` and `half2_uint32` in the reduction. Both are standard CUDA.
2. **Cost:** −1 % to −3.5 % expected against the −3 % target (see above). Most exposed are chunks with many thin assignments: 512-1,024-row tail chunks, where thin fp32 slots dominate. Their share of a 60k/120k prompt is small.
3. **`--use_fast_math` implies `-ftz`.** If nvcc emits the reduction's `__fmul_rn` without `.ftz` while the atomic kernel's plain `*=` uses `.ftz`, the per-contribution values could differ from atomic E3 for denormal intermediates. That affects only the DET-vs-atomic comparison, far below the 1e-5 gate, never determinism.
4. **The reduction stores the output row instead of adding to it.** I checked that nothing writes `final_hidden_states` before it in the DET branch:
   - `fhs_ext` is a fresh `torch.zeros` at `block_sparse_mlp.py:1005`;
   - the thin tier in slot mode writes only slots;
   - batched-reconstruct groups and the per-expert loop are empty when `expert_count_list is None`;
   - shared experts are added after the routed sum;
   - `EXL3_SHARED_EXPERT_OVERLAP` lives only in the bsz ≤ 16 decode path (`BC_BlockSparseMLP`, `libtorch/blocksparse_mlp.cpp:19-23, 303`);
   - `MOE_PREFILL_E3` is read nowhere outside `block_sparse_mlp.py` (grepped `src/exllamav3` and `src/app`).

   The sum starts at +0.0f, so `+=` onto the zero row would give the same bits at one extra 21 MB read per layer. A future caller that pre-accumulates would need it.
5. **DET is not bitwise equal to atomic E3 or to E3-off, by design.** The DET 30k fingerprint is a new canonical; c1 must not move.
6. **Determinism assumes the serving stack is otherwise deterministic,** as R524 showed with E3 off. DET removes E3's contribution only.
7. **Licensing provenance of E3 is unchanged** (Mia AGPL-3.0 note in the r2 impl-status). The new kernels are original, but they live in the same file.
