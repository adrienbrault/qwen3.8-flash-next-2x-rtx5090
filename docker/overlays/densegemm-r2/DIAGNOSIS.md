# densegemm r2: why every gemm / mgemm V2 twin is 3-6x slower (R710)

Final, 2026-09-24, dense-GEMM Opus sub-agent round 2. No GPU and no nvcc on this workstation. The evidence is the R710 raw files (`out-densegemm/r710-results/`), the r1 source (`densegemm-r1/work/b`), and a clang CUDA compile to PTX used as a proxy (§1b). The r2 patch is written blind; its first gate step is a CPU-only compile check that decides whether any GPU time is spent.

## Verdict, ranked by evidence

1. **Cause: the gemm and mgemm twins run their inner loop out of local memory.** Every one of the 24 gemm/mgemm twins has a per-thread stack frame of 544-1832 bytes. The served kernels of the same shapes have 8-80 bytes, and the 8 gemv twins (the ones that win) have 8. For shapes 1, 2 and 3 the register count collapses from 122-128 (served) to 40: the compiler did not spill under a cap, it placed the fragment and pipeline state in memory. Shape 4 hits the 255-register cap and spills 640-864 bytes. r1's install reported "LOCAL:0 = no spills", but `LOCAL` in `cuobjdump -res-usage` counts statically declared `.local` data. Spills and demoted arrays are counted in `STACK`, which r1 printed and did not check.
2. Not the cause: occupancy and cooperative residency, the gathered-fixup wait, per-call workspace zeroing, and the P0 method. Each is ruled out below from source or from the R710 numbers.
3. Minor, real, and pointing the wrong way: the r1 B prologue commits all weight stages as one cp.async group and then runs `cp_async_wait<0>`, so it drains the whole pipeline before the first MMA. The served kernel waits for stage 0 only. This costs about 1 µs per call at most and does not explain 3-6x. r2 fixes it anyway.

## 1. The evidence for local memory

From `r710-results/res-usage.txt` (cuobjdump -res-usage of the rebuilt .so), K=4, codebook 2, fp16 / fp32 C:

| kernel, shape (TM, TK, TN, SH, FS) | threads | served REG / STACK | V2 twin REG / STACK |
|---|---|---|---|
| gemm, shape 1 (16, 16, 128, 6, 5) | 256 | 128 / 8-32 | 40 / 1752 |
| gemm, shape 2 (16, 32, 128, 4, 3) | 512 | 122-128 / 16 | 40 / 1072-1088 |
| gemm, shape 3 (16, 32, 256, 4, 3) | 512 | 128 / 56 | 40 / 1368 |
| gemm, shape 4 (16, 16, 512, 4, 3) | 256 | 179 / 16 | **255** / 640-760 |
| mgemm, shape 1 | 256 | 128 / 8-24 | 40 / 1824-1832 |
| mgemm, shape 2 | 512 | 128 / 16 | 40 / 1168 |
| mgemm, shape 3 | 512 | 128 / 56-80 | 40 / 1512-1528 |
| mgemm, shape 4 | 256 | 194-242 / 16 | **255** / 768-864 |
| gemm / mgemm rows32 (32, 32, 128 / 256, 4, 3) | 512 | (no served twin) | 64-128 / 544-864 |
| gemv twins (all 8) | | 57-80 / 8-16 | 57-121 / 8 |

Served REG / STACK are for codebook 2 (the mul1 codebook the model uses); the other codebooks are the same within a few registers.

Why this fits the P0 numbers and nothing else does:
- The slowdown does not depend on the stream-K segment count. attn.index_qk at rows 16 has one segment per column (no fixup at all) and is still 2.6x slower (13.9 → 36.7 µs). shared.gate_up is 2.8x slower at every row count. A local-memory inner loop slows every call; a fixup, barrier or zeroing defect would scale with handoffs or be a constant few µs.
- The ratio follows the stack size. Shape 3 twins (1368-1528 B) are the slowest: in_proj r8/r16 26 → 171-175 µs (6.6x), qkv 24-26 → 141-145 µs (5.8x). Shape 2 (1072-1168 B): out_proj/o_proj r16 20.0 → 77.5 µs (3.9x), shared.down r16 10.1 → 20.4 µs (2x; its K is 640, so fewer loop iterations). Shape 4 (255 regs, 640-864 B spill): in_proj r4 26.3 → 104 µs (4x). rows32 twins with smaller stacks (544-864 B) at rows 24/32: in_proj 44.8 → 84.7 µs (1.9x), where the served two-pass kernel is the reference.
- The times are identical to 0.1 µs across rounds and between m1 and m2 (e.g. out_proj r16 77.50 / 77.50, r24 and r32 both 159.5). That is a deterministic per-instruction cost, not contention or a spin.
- P1 round 1 of R710 agrees: c4d3 20.15 ms OFF → 31.06 ms ON.

What in the r1 source triggers it: the r1 `reduce()` lambda is much larger than the served one. It adds the batched `gather` (a runtime loop over producer slots with `buf[GU][M][F]` float4 / float2 arrays, `ok[GU]`, 64-bit `tile_beg` / `owner` division, both the compact and the full slot paths), `store_slot` (both paths), and for mgemm the output Hadamard that the served mgemm runs as a separate pass. `reduce` is expanded once per fragment stage in the unrolled main loop (3 or 5 copies). The served source warns that this loop structure can "confuse the compiler and make it place the fragment arrays in local memory". The most likely mechanism for shapes 1-3 (REG 40) is that `reduce` (or `gather`) was not inlined: the closure then captures `frag_c` and the pipeline counters by reference, their addresses escape, and every MMA accumulates through local memory. For shape 4 the same extra arrays are live next to `frag_b[3][8]` and push a kernel that is already at 179-242 registers over the 255 cap. The 64-bit divisions are a second suspect: they compile to subroutine calls. Which construct is decisive cannot be settled without a compiler, so r2 removes all of them and the gate's first step measures the result (§5).

## 1b. Proxy check with clang (no nvcc here)

To test the "non-inlined lambda" mechanism and to syntax-check r2, the twin TUs were compiled on this Mac with Homebrew clang 23 in CUDA device-only mode (sm_120, -O3, CUDA 12.8 headers from the PyPI wheels, a linux libc/libc++ header set from the zig wheel). Scripts: `r2/clang-proxy/dcc.sh`, `ptx_local.py`; full table: `r2/clang-proxy/ptx-local-summary.txt`. clang's inliner is not NVVM's, so absolute numbers do not transfer (clang also outlines a lambda in the served TN256/TN512 kernels, which nvcc does not). The comparison between r1, r2 and served under one compiler does transfer as a direction.

Inner function per shape, codebook 2, fp16 C (local depot bytes / call instructions in the inner function; in r1, 3 of them are calls to the outlined `gather`):

| shape (TM, TK, TN) | served | r1 gemm | r2 gemm | r2 mgemm | r2 gemm, `DGV2_LAMBDA_AI=1` |
|---|---|---|---|---|---|
| 16, 16, 128 (shape 1) | 4 / 0 | 144 / 15 | 4 / 0 | 4 / 0 | 4 / 0 |
| 16, 32, 128 (shape 2) | 20 / 0 | 144 / 9 | 20 / 0 | 20 / 0 | 4 / 0 |
| 16, 32, 256 (shape 3) | 384-512 / 9 | 176 / 9 | 20 / 0 | 20 / 0 | 4 / 0 |
| 16, 16, 512 (shape 4) | 448-576 / 9 | 240 / 9 | 4 / 0 | 4 / 0 | 4 / 0 |
| 32, 32, 128 (rows32) | n/a | 576 / 9 | 12 / 0 | 12 / 0 | 4 / 0 |
| 32, 32, 256 (rows32) | n/a | 640 / 9 | 544 / 9 | 544 / 9 | 4 / 0 |

In r1 the outlined function is the in-loop fold `gather`. Its body has no `mma` and no `ldmatrix`, but it does have 24 `ld.global.cg` (the slot loads) and 76 `cvt.rn.f16.f32` (the fp16 round trips of the running sum), and it is called from each of the 3 (or 5) unrolled fragment stages. Its mangled name, `...UliE4_clEi` (the sixth `(int)` lambda), agrees. That is the mechanism proposed in §1: the in-loop fold is not inlined, its closure takes the address of `frag_c` and the loop state, and they move to local memory. With r2 no 16-row shape outlines anything under clang. `DGV2_LAMBDA_AI=1` also compiles (the attribute syntax is accepted) and removes the last outlined lambda in the rows32 TN256 twin. The r2 sources compile without errors in the three variants tried (defaults, `DGV2_LAMBDA_AI=1`, `DGV2_BPRO=0`), gemm and mgemm TUs.

## 2. Hypotheses ruled out

| hypothesis | verdict | evidence |
|---|---|---|
| Register pressure → occupancy → cooperative grid no longer fits | ruled out as stated | The grid is the served tuned grid, at most one CTA per SM. A cooperative launch that does not fit fails with `cudaErrorCooperativeLaunchTooLarge`, which `cuda_check` turns into an exception. It cannot run slowly. Register pressure matters through spills (row 1), not residency. |
| Gathered fixup serializes or spins | ruled out | Producers publish (`__stcg` + `red.add`) without waiting. Only heads wait, and only on later CTAs' first segments, so there is no cycle. index_qk r16 has no producers at all and is still 2.6x slower. The spin is the served `barrier_acquire` (no backoff, same as served). |
| Per-call workspace or counter zeroing | ruled out | `ws_ready()` returns at its first line after the first call. The one-time `cudaMalloc` + `memsetAsync` + `cudaStreamSynchronize` runs in P0's untimed warm-up. Locks and counters return to 0 in-kernel (`*lock = 0` by the head). |
| B prologue exceeds shared memory / cp.async limits | ruled out | Same stage buffers as served (the static_assert on SMEM_MAX is unchanged), two commit groups instead of three. The real defect is the `wait<0>` drain (row 3 of the verdict), worth ≤ 1 µs. |
| P0 bench artefact | ruled out | Timing is GPU events with the queue pre-filled behind a sleep. `us_each` ≈ `us_batch` in every cell. The autotune cache is warmed on the served arm before timing. m3 (gemv only) re-measures the served gemm/mgemm kernels in every cell and matches m0 within 0.1-4 %. m0 and m1 run the same callables on the same weights. |

## 3. What r2 changes (densegemm-r2.patch)

The gemv twin, the host dispatch and the flags are unchanged. `exl3_gemm_inner_v2.cuh` is restructured; `exl3_gemm_v2_kernel.cuh` only gains the optional inline attribute on its two lambdas; `exl3_dense_v2.cuh` bumps the revision marker to 2.

1. **Deferred head fold.** A segment that starts at k = 0 but does not reach the top k tile must end its CTA's slice, so it is always the CTA's last segment (checked on the CPU model for every served config and 2000 random ones, `test_densegemm_cpu.py` test 5). r2 does not fold it inside the unrolled loop. It records the column, lets the loop exit, and folds after the loop, at one call site, when `frag_a` and `frag_b` are dead. The invariant is structural: `reduce()` runs only when `slice2_k == tiles_k - 1 || slice2_iters == 1`, and the deferred branch is the not-top case, so `slice2_iters == 1` there, `advance2()` takes it to 0, and the loop's next statement breaks. Inside the loop `reduce()` now does only what the served one does in size: the served threadblock reduction, then either publish (non-head) or the in-tile epilogue (a whole column in one CTA).
2. **The fold is sequential and has no arrays.** One slot at a time, from the top producer down: S = P_top; S = P_b + rnd(S); final = own + rnd(S). Same order and rounding as r1 and as the served chain. The loop over producers holds one accumulator per fragment, with no `buf[GU]` batch and no `ok[]` mask. The price is one dependent L2 round trip per producer (about 0.1-0.3 µs each; 5-6 producers for out_proj and gate_up heads) where r1 batched the loads. If P0 shows the fixup lever paying nothing, staging the producer slots through `sh_c` (free after the loop) is the r3 change.
3. **32-bit partition arithmetic.** `tile_beg` and `owner` use the served expression `tiles_k * tiles_n * b / num_slices` in `int`. The host already refuses any launch with tiles × (grid + 1) ≥ 2^31, so the results are the same and there are no 64-bit division calls.
4. **The new code is in `__forceinline__` free functions** (`dv2_store_slot`, `dv2_fold`) that take `frag_c` by reference, which is the construct the rest of the extension uses for force-inlined helpers (exl3_moe_coop_v2_kernel.cuh). The only new lambda, `epilogue`, wraps the served epilogue code that r1 already ran inside `reduce`.
5. **B prologue waits for stage 0 only.** Commit groups are now [B stages 0..S-2] [A stage 0] ... [A stage S-2] followed by the served `wait_stage()` (`cp_async_wait<S-2>`), which completes the B group and A stage 0 and leaves the rest outstanding, as the served prologue does.
6. Compile-time switches `DGV2_BPRO` (1 = B prologue, the default; 0 = served order) and `DGV2_LAMBDA_AI` (1 = `__attribute__((always_inline))` on the main-loop lambdas; default 0). The compile bisect picks the first combination that passes the stack bar, and the build uses it (§5).

**Why compile-time switches and not runtime P0 sub-flags.** The brief allowed P0 sub-flags that turn the gathered fixup, the B prologue and the barrier trim on and off. They were not built. The 3-6x is decided when ptxas compiles the kernel, so switching mechanisms at run time cannot show it. Each runtime switch would also add code paths to the same kernels whose inlining is the problem. The discriminating tool is the compile bisect (§5). The consequence: if r2 compiles clean but P0 lands at served ± 1 µs (`P0-ARMS none`), this run cannot say which lever failed to pay, and r3 would need those runtime switches (or separate compile-time images).

## 4. Expected per-call time if fixed (estimates, P0 measures)

The first bar is that the twins stop losing: with a served-sized stack, a twin should land at served ± 1 µs before any lever pays. attn.index_qk r16 (no fixup) is the control cell for that. The lever estimates are r1's §4 figures, less the batched gather (r2's sequential fold costs about one L2 round trip, ~0.1-0.3 µs, per producer).

| projection (kernel) | rows | served µs (R710 m0) | r1 twin µs | r2 expected µs |
|---|---|---|---|---|
| gdn.in_proj (mgemm) | 4 | 26.3 | 104.1 | 20-23 |
| gdn.in_proj (mgemm) | 16 | 28.3 | 174.9 | 24-26.5 |
| attn.qkv (mgemm) | 4 | 23.6 | 141.0 | 19.5-22 |
| attn.qkv (mgemm) | 16 | 25.9 | 144.7 | 21.5-24 |
| gdn.out_proj (gemm) | 16 | 20.0 | 77.5 | 15.5-17.5 |
| attn.o_proj (gemm) | 16 | 18.3 | 77.4 | 14-16 |
| attn.index_qk (gemm) | 16 | 13.9 | 36.7 | 12.5-13.9 |
| shared.gate_up (mgemm) | 4 / 16 | 13.7 / 15.4 | 38.5 / 42.6 | 9-11.5 / 10.5-13 |
| shared.down (gemm) | 16 | 10.1 | 20.4 | 8.5-9.8 |
| gdn.out_proj, o_proj, index_qk, shared.down | 4-8 | gemv | gemv twin (m3) | unchanged, gemv twin |

Critical path per step at rows 16 (c4d3 / c8d1): served 2437 µs; r2 estimate 2.03-2.26 ms, i.e. −0.18 to −0.41 ms (0.9-2 % of a 20 ms step). At rows 4 (c1d3) the gemm/mgemm twins only touch in_proj and qkv: −0.15 to −0.33 ms on top of the gemv twin's −0.30 ms.

## 5. The discriminating probe: compile-only, no GPU

`bisect-compile.sh` runs in the base image without `--gpus` (CPU only). It applies the r1 and r2 patches to scratch copies of the installed package, compiles the two twin TUs for each variant with the real build's nvcc flags plus `-Xptxas -v`, dumps `cuobjdump -res-usage` and `-sass`, and compares every twin with the served kernel of the same shape from the base .so (`compile_bar.py`):

| variant | defines | role |
|---|---|---|
| r1 | the R710 patch | must FAIL the bar: reproduces §1 and validates the probe (PROBE VALID), else nothing is built |
| r2 | defaults | first choice |
| r2-ai | `-DDGV2_LAMBDA_AI=1` | fallback if nvcc still outlines a lambda |
| r2-nobpro | `-DDGV2_BPRO=0` | fallback if the B prologue's saved pipeline state is what spills |
| r2-nobpro-ai | both | last fallback |

Bar per twin: `STACK ≤ served STACK + 64` bytes (rows32 twins against the 16-row served kernel of the same TK/TN). The choice (`BISECT-CHOICE`) is the first r2 variant whose 16-row and rows32 twins all pass; if no variant passes on the rows32 twins, it is the first whose 16-row twins pass, and `ROWS32-COMPILE FAIL` is reported. The gate builds the image with those defines (`--build-arg DGV2_DEFS`, passed to nvcc for every TU; the served-SASS identity check proves it touches no served code). If none passes, the gate stops before the daily goes down, so no GPU time is used. Only the 16-row result can stop the gate. The rows32 result is reported and feeds the ROWS32 verdict (§8), because a mild spill in a 32-row twin could still be fast, and P0 decides that. The install repeats the 16-row bar on the real .so and fails the build on a violation. The per-variant table (REG, STACK, LDL / STL / CALL counts from the SASS, served reference) says which construct is responsible if every variant fails: CALL > 0 inside a twin names an outlined function, and LDL / STL counts show whether the main loop touches local memory. That table is the next step in that case, not ncu.

`compile_bar.py` was checked against R710's own res-usage: it fails all 16 r1 16-row twins (0/16, max 1832 bytes). The bisect's control flow was dry-run here with fake nvcc / cuobjdump shims: probe valid + r2 passing picks r2; r2 failing picks the next passing variant; rows32 failing everywhere except with `DGV2_LAMBDA_AI=1` picks r2-ai; rows32 failing everywhere picks r2 with `ROWS32-COMPILE FAIL`; a variant that does not compile is skipped; probe invalid gives `BISECT-CHOICE none`.

## 6. Gate (densegemm-gate.sh, r2)

0a compile bisect (CPU, in-lock, the daily keeps serving) → 0b build with the chosen defines (the install re-checks the bar) → 1 parity (r1's test, revision 2) → 2 P0, with the arm gate at its end:

- `P0-CONTROL`: attn.index_qk r16, m2 against m0. This cell has no fixup, so it is the first place a twin that still runs out of local memory shows up (R710: 13.9 → 36.7 µs).
- `P0-CALL`: each engaged critical-path gemm/mgemm cell at rows 4 and 16, m2 against m0.
- GM (`EXL3_DENSE_V2=2`) goes to P1 only if no such cell is more than 3 % slower than served, the rows-16 critical-path total is below served, and rows 4 is within the c1 bound. ON (`EXL3_DENSE_V2=1`) additionally needs GM eligible. `P0-ARMS none` stops the gate FLAT before P1, with the daily restored. The rule was run offline on R710's p0.json (prints `P0-ARMS none`) and on synthetic 0.9x / 1.05x data (`GM ON` / `none`).
- `P0-ROWS32` / `P0-ROWS32-VERDICT`: arm m4 = `EXL3_DENSE_ROWS32=1` with `EXL3_DENSE_V2` unset, per projection at rows 18 / 21 / 24 / 32 against the served two-pass kernel. The gate logs it as a separate `DECISION ROWS32` line right after P0, before any P1/FLAT branch, and again at the end (§8).

→ 3 P1: OFF + the eligible arms, ROWS32 off, 6 rounds rotated so each arm runs first equally often, × c4d3 / c8d1 / c1d3, plus the draft-0 cells. Hashes must equal OFF. → 4 greedy (off + each P1 arm) → DECISION per §16, same clauses as r710.

Worst cases for GPU time: a compile-bar stop costs 0 GPU minutes; a P0 stop costs about 20 (parity + P0).

## 7. Files and operator steps

In `out-densegemm/r2/`:
- `DIAGNOSIS.md` (this file).
- `densegemm-r2/`, the patch directory, laid out like r1's: `densegemm-r2.patch` (sha256 `2bc1580c…`, 13 files, made by `make-patch.sh`: work/a == stack-r2, fuzz-0 round trip, composes with pdl-l2-r1 both ways, and r1 → r2 touches only `exl3_gemm_inner_v2.cuh`, `exl3_gemm_v2_kernel.cuh`, `exl3_dense_v2.cuh`), `Dockerfile.box`, `install-densegemm.sh`, `rebuild-native.py`, `compile_bar.py`, `bisect-compile.sh`, `densegemm-gate.sh`, `test_densegemm_cpu.py` (r1 tests + 5 deferred head + 6 prologue groups; PASS), `test_densegemm_parity.py`, `bench_densegemm_p0.py`, `sass_hashes.py`, `d0/`, `ref/densegemm-r1.patch`, `work/a`, `work/b`, `make-patch.sh`, `quick-nvcc.sh`.
- `clang-proxy/`: `dcc.sh`, `ptx_local.py`, `ptx-local-summary.txt` (§1b).

Operator:
0. Precondition: BASE must be a daily without densegemm. The r2 patch creates the `exl3_dense_v2*` files, so it applies only to stack-r2 (or another densegemm-free stack). If the gemv twin (R710b) is batched into a new daily before this unit runs, the bisect fails loudly and without GPU (`dense_v2_revision present`); rebase with `make-patch.sh` against the new stack first.
1. Copy `r2/densegemm-r2/` (without `work/`) to `flan/patches/exllamav3/densegemm-r2/` in the repo and to `/srv/qwen5090/patches/exllamav3/densegemm-r2/` on flan. `ref/densegemm-r1.patch` must come along (the bisect's probe uses it).
2. Copy `densegemm-gate.sh` to `flan/rNNN-densegemm-r2.sh` and `/srv/qwen5090/rNNN-densegemm-r2.sh`, then queue `rNNN-densegemm-r2`. It is independent of R710b (GV): ON includes the gemv twin, GM does not.
3. Read, in order: `audit.log` `[bisect]` lines (PROBE, BAR per variant, BISECT-CHOICE), `[build] densegemm: bar` lines, the parity line, then `P0-CONTROL` / `P0-CALL` / `P0-ARMS`.
4. If the bisect prints `BISECT-CHOICE none`, send `bisect/bar-*.txt` and `bisect/*.log` back: the LDL / STL / CALL columns name the construct for r3.
5. An independent review of the round applies as usual.

## 8. ROWS32 (piece 4a) for rows32-r3 (operator addendum)

**The ROWS32 twins share the slow mechanism, from source and from R710.** `EXL3_DENSE_ROWS32=1` does not have kernels of its own. It launches the same `exl3_gemm_v2_kernel` / `exl3_mgemm_v2_kernel` templates, instantiated with `EXL3_GEMM_V2_SHAPE_2_M32` / `_3_M32` (TILESIZE_M = 32), through the same `exl3_gemm_kernel_inner_v2`. They therefore use the gathered fixup, the B prologue and the barrier trim (gemm: no trailing `grid.sync`; mgemm: no per-chunk barrier and no separate output pass). `resolve_v2` takes the rows32 path on the ROWS32 flag alone, so `EXL3_DENSE_V2` unset + `EXL3_DENSE_ROWS32=1` runs exactly these kernels at 17-32 rows and the served ones at ≤ 16 rows. In R710 they carry the same defect: REG 64 / 128 with STACK 544-864 bytes (§1 table). R710's only rows32 timing ("rows32 at rows 24: m1 −8454.9 µs/step saved") is contaminated twice: it ran the local-memory twins, and under m1, so rows 24/32 in that table measured the broken kernels. The rows32 twins alone were never timed.

**r2 covers them.** They go through the same restructured inner loop. Under the clang proxy the TN128 rows32 twin is clean with the r2 defaults (12 B depot, no calls). The TN256 one (gdn.in_proj and attn.qkv at 17-32 rows) still outlines one lambda with the defaults (544 B) and is clean with `DGV2_LAMBDA_AI=1` (4 B). So the bisect prefers a variant whose rows32 twins also pass the bar (§5). One risk remains that no restructuring removes: the TN256 twin runs 512 threads, so it has a 128-register cap, and the served 16-row TN256 kernel already sits at 128 registers with a 56-80 B stack. A second 16-row block adds 16 fp32 accumulators and 4 A registers per thread, which may spill in ptxas whatever the code structure. If the bar reports that twin over budget and P0 shows it slow, the r3 lever is `FRAG_STAGES = 2` for the M32 twins (fewer prefetched B fragments in registers). It changes only the fragment prefetch depth, not the MMA sequence or the reduction order, so it stays bitwise identical.

**Verdict, reported separately.** The gate logs `DECISION ROWS32 (EXL3_DENSE_ROWS32=1, EXL3_DENSE_V2 unset)`. It says READY when parity passed (its rows 17 / 24 / 32 cells are bitwise against served two-pass) and `P0-ROWS32-VERDICT READY`: no engaged critical-path cell at 18 / 21 / 24 rows is more than 3 % slower than served two-pass, and the critical-path total is lower at each of those rows. The compile-bar status is printed next to it. There is no P1 / §16 verdict for ROWS32 in this gate, because no P1 cell of the daily config reaches 17 rows. rows32-r3's own gate, rebased on `tabbyapi:densegemm-r2` with the same defines, measures it end to end. The P0 arm gate for GM / ON is unaffected (m4 is a fifth arm, rows 18 and 21 were added).

**Expected per call for a working 4a at 18-24 rows (estimates).** One pass reads each weight once, like a 16-row call. The served two-pass kernel pays a second pass, which R710 measured at +56-60 % over 16 rows (in_proj 28.3 → 44.8 µs). A 32-row tile doubles the MMAs, the A tile, the partial sums and the epilogue rows, but those are small next to the weight stream at decode sizes, so the estimate is served-16-row × 1.05-1.15:

| projection | served 16 rows (R710 m0) | served two-pass, 24 rows | 4a one-pass, 18-24 rows (estimate) |
|---|---|---|---|
| gdn.in_proj (mgemm, TN256) | 28.3 | 44.8 | 29.7-32.6 |
| attn.qkv (mgemm, TN256) | 25.9 | 40.6 | 27.2-29.8 |
| gdn.out_proj (gemm, TN128) | 20.0 | 32.3 | 21.0-23.0 |
| attn.o_proj (gemm, TN128) | 18.3 | 30.7 | 19.2-21.0 |
| attn.index_qk (gemm, TN128) | 13.9 | 22.3 | 14.6-16.0 |
| shared.gate_up (mgemm, overlap stream) | 15.4 | 25.1 | 16.1-17.7 |
| shared.down (gemm, overlap stream) | 10.1 | 14.9 | 10.6-11.6 |

Critical path per step at 24 rows: served two-pass 3.90 ms; 4a estimate 2.56-2.80 ms, i.e. −1.1 to −1.3 ms per step (shared-expert calls excluded, they run on the overlap stream). The served two-pass times at 18 and 21 rows should be close to 24 rows (the second pass is a 16-row tile whatever its fill); P0 now measures 18 and 21 directly. These numbers hold only if the TN256 rows32 twin passes the bar or at least P0. If it spills, in_proj and qkv, about half of the estimated saving, could land anywhere between the one-pass estimate and R710's 85 µs.
