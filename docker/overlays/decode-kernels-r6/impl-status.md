# Decode kernels round 6 — implementation status

Written before the first build. The box results are in [R538](../../../bench/results/r538-decode-r6.md) (kernel equality and the 8-boot A/B) and [R540](../../../bench/results/r540-promote-r6.md) (served since 2026-09-19 11:11 CEST).

Round 6 rebases round 5 onto the served chain `tabbyapi:nvme-tier-r4-e3det` (R535). The served sources are the local extraction at `served-src-r535/exllamav3/`; line numbers below refer to that tree unless marked `r6:` (the patched tree). No GPU, compiler, or remote host was used. Nothing here is a measurement.

## Answer to the central question

Under the daily environment (`scripts/launch-flashnext.sh:206`), three of the four r5 selectors change a kernel that the daily launches. One of them needed a port to the int8 mixer. The fourth is a no-op.

| selector | status under the daily env | r6 |
|---|---|---|
| `EXL3_GDN_BA_WARP1` | live, independent of the mixer | carried unchanged |
| `EXL3_HC_APPLY_WARP1` | live, independent of the mixer | carried unchanged |
| `EXL3_GR_STATE_REGRID` | dead as written in r5 (fp16 V2 branch only); the int8 mixer launches the same state kernel | ported to the int8 launcher |
| `EXL3_GR_FREE_UP_H` | no-op: frees 0 bytes | dropped |

### The mixer path on the daily

- `EXL3_HC_MIX_V2=1`, `EXL3_HC_MIX_V2_MIN_R=1` and `EXL3_HC_MIX_V2_INT8=1` make `_prepare` call `_quantize_v2_int8` (`modules/hyperconnections.py:322-323`). With `MIX_V2_MIN_R <= 1` that function sets `fn_h = None` and `upx_h = None` (`:342-344`). `proj_h` and `up_h` stay resident for the R > 32 prefill GEMMs (`:439,443`).
- Every decode, verify and MTP call has 1 ≤ R ≤ 32, so it takes the V2 branch (`:399,410`). Because `fn_q` is set, that branch calls `ext.gr_mix_v2_int8` (`:418-422`). The fp16 branch `gr_mix_v2` / `gr_mix_v2_fused` (`:424-426`) is never reached on the daily. The MTP head's mixers are the same `GatedResidual` class (`architecture/qwen4_exp_mtp.py:104-115`, `architecture/qwen4_exp.py:95-110,283-292`), so they take the same branch.
- The int8 launcher `gr_v2_launch_i8` (`exllamav3_ext/hc_mix.cu:1437-1469`) runs three kernels: `gr_v2_dots_i8_kernel`, the shared fp32 `gr_v2_state_kernel` (`:1454-1455`), and `gr_v2_up_i8_kernel`. The fp16 launcher calls the same state kernel (`:1421-1422`). The state kernel reads only the fp32 `dots` workspace and writes `state` and `post` (`:1051-1078`). It does not read weights.

### `EXL3_GR_STATE_REGRID`: ported

In r5 the selector only switched `gr_mix_v2` to `gr_mix_v2_regrid` (fp16). On the daily that is dead code. The port adds a `state_regrid` argument to `gr_v2_launch_i8`. It launches the r5 `gr_v2_state_regrid_kernel` in place of `gr_v2_state_kernel` (r6: `hc_mix.cu:1541-1546`). It also adds the entry point `gr_mix_v2_int8_regrid`, which sets the flag only on SM 12.0 (r6: `hc_mix.cu:1766-1781`), and the Python dispatch `ext.gr_mix_v2_int8_regrid if self.STATE_REGRID else ext.gr_mix_v2_int8` (r6: `hyperconnections.py:420`).

The port preserves bit order for the same reason r5's fp16 variant does. The dots kernel and the up kernel are unchanged, and the state kernel's input (`dots`) is byte-identical. Each state output `i` still runs the same H-ordered `fmaf` chain from the same four `rmr` values, the same `1/H` scale, and the same `sigmoidf_`/SiLU and store (r6: `hc_mix.cu:1128-1157` vs served `:1051-1078`). The only difference is which CTA computes output `i`. Every CTA recomputes `rmr` from the same `dots` tail, and only x-block 0 writes the global `rmr` copy.

`gr_mix_v2_int8` keeps its served behaviour: it calls `gr_mix_v2_int8_impl(..., false)`. The fp16 `gr_mix_v2_regrid` from r5 is kept so the harness can check both weight variants. It is not reached on the daily.

### `EXL3_HC_APPLY_WARP1`: live

`GatedResidual.apply_` calls `ext.hc_apply` (`modules/hyperconnections.py:471-476`) after every attention and MLP site (`modules/transformer.py:165,191`). This covers the target's 96 sites and the MTP block's two sites. The call does not depend on the mixer path. `hc_mix.cu` still has the r4 baseline hash (`912db071…`), so the r5 hunk applies unchanged.

### `EXL3_GDN_BA_WARP1`: live

Decode and verify GDN calls take `bc.run_bszN` (`modules/gated_delta_net.py:1060-1072`), which calls `gdn_ba_gemv_gr(x, ba_weight_t, ba_bias, s.ba, graph)` (`exllamav3_ext/libtorch/gated_delta_net.cpp:278`). The mixer does not touch this path. The daily is layer-split rather than tensor-parallel (`tensor_parallel: false`, `gpu_split: [30, 30]`, `scripts/launch-flashnext.sh:284-286`), so B/A keeps its full N = 96. `gdn.cu` has the same hash as the r5 baseline (`a06effe5…`). The MTP block is `full_attention` (`architecture/qwen4_exp_mtp.py:95`) and has no B/A call.

### `EXL3_GR_FREE_UP_H`: no-op, dropped

- r5 releases `up_h` only when `upx_h is not None` (r5 `r4-source.patch:375`). On the daily `upx_h` is already `None` (`:342-344`), so the guard is false and `up_h` stays.
- `up_h` also cannot be released there. It is the only lossless copy of the up weight once `upx_h` is gone, and the R > 32 prefill GEMM reads it (`:443`). Rebuilding it from `upx_q`/`upx_s` would change prefill numerics.
- **VRAM freed by `FREE_UP_H` on the daily: 0 bytes.**
- The memory that r5 targeted has already been released by the served int8 path. For H = 4, D = 2560, rank 320 (r5 `static-derivations.json`), each site mixer frees `upx_h` 6,553,600 B plus `fn_h` 6,635,520 B. It adds `fn_q` 3,317,760 B, `fn_s` 1,296 B, `upx_q` 3,276,800 B and `upx_s` 40,960 B, for a net 6,552,304 B. Each final mixer nets 6,511,360 B. With 98 site mixers (48 target blocks × 2 plus 2 MTP) and 2 final mixers, the total is 655,148,512 B (0.610 GiB). That is within 0.03 % of r5's 655,360,000 B `FREE_UP_H` figure. R516/R525 measured 218 / 258 MiB more free VRAM per card from the int8 weights (`scripts/launch-flashnext.sh:71`). The tensor arithmetic does not explain the ~150 MiB difference from 625 MiB. That difference needs a box measurement.
- The only way left to free the 6,553,600 B `up_h` per mixer (655,360,000 B total) without changing numerics is to keep it off-GPU and copy it back for each prefill chunk. That costs 97 × 6.55 MB of host-to-device traffic per chunk. It is not implemented.

## What r6 contains

`served-source.patch` touches five files and applies to the served tree with `patch -p1 -F0` (exit 0, no rejects, verified on a fresh copy). Baseline and post-patch sha256 are pinned in `overlay/manifest.json`:

| file | served baseline | r6 overlay |
|---|---|---|
| `exllamav3_ext/gdn.cu` | `a06effe5…` | `b2624556…` (= r5 r4-variant overlay) |
| `exllamav3_ext/hc_mix.cu` | `912db071…` | `16a1c799…` |
| `exllamav3_ext/hc_mix.cuh` | `33ef3437…` | `4be278ba…` |
| `exllamav3_ext/bindings.cpp` | `59ad68ef…` (carries `exl3_moe_prefill_e3_det`) | `3cd3a4d7…` |
| `modules/hyperconnections.py` | `a17978dc…` (= r5's r4 baseline) | `220e185c…` |

The served `hyperconnections.py` is byte-identical to r5's r4 baseline. The only served file that moved since r5 is `bindings.cpp`, and the r6 hunk there adds two `m.def` lines.

All selectors are off by default:
- The C++ flags (`EXL3_GDN_BA_WARP1`, `EXL3_HC_APPLY_WARP1`) are process-lifetime statics that are true only for the exact string `"1"`. They also require SM 12.0 and a served grid below the SM count.
- `EXL3_GR_STATE_REGRID` is read once as a class attribute (`== "1"`). When it is unset, Python calls the served entry points `gr_mix_v2_int8` / `gr_mix_v2` / `gr_mix_v2_fused` exactly as before.
- The served C++ entry points pass `state_regrid = false`.

With every selector unset, the patched build launches the same kernels with the same grids as the served build.

## Expected effect per selector (estimates, not measurements)

| selector | shape at c1/d3 (R = 4) | shape at c4/d3 (R = 16) | estimate |
|---|---|---|---|
| `GDN_BA_WARP1` | 36 calls/step; `grid (12,4)` × 256 threads → `(96,4)` × 32 | 192 served CTAs ≥ 170 SMs → served fallback | −72.7 µs/step at c1/d3 (r5: [exllamav3#369](https://github.com/turboderp-org/exllamav3/pull/369)'s 49.36 % kernel reduction applied to the r1 147.3 µs/step B/A total); 0 at c4 |
| `HC_APPLY_WARP1` | 96 target calls/step (+2 MTP); `(3,4)` × 256 → `(20,4)` × 32 | `(3,16)` × 256 → `(10,16)` × 32 | none predicted; r1 total 128.5 µs/step (c1), 143.8 (c4), near launch floor |
| `GR_STATE_REGRID` (int8) | 97 target calls/step (+MTP); `R` CTAs × 128 → `(11,R)` × 32 site, `(10,R)` final | 16 → 176 CTAs | none predicted; r1 total 120.5 µs/step (c1), 125.0 (c4) |

The r1 per-kernel totals come from a trace that predates the int8 mixer, bszn16, coop V2 and E3. The B/A and `hc_apply` kernels themselves have not changed since r1 (same source hashes). The int8 path launches the same state kernel with the same grid as the fp16 path, so the per-launch state figure still applies. What has changed is the per-step denominator (the daily step is shorter), so treat the −72.7 µs as an upper-bound planning number that the box must check. `BATCH_VERIFY` at c4 gives 16 verify rows per GDN call, so GDN falls back there by construction.

## Risks

- **Not compiled.** The patch has not been through nvcc locally. The int8 hunk only reuses symbols that the fp16 hunk already defines earlier in the same file (`gr_v2_state_regrid_kernel` at r6 `hc_mix.cu:1129`, `CEIL_DIVIDE` from `util.h:5`), and the r5 hunks were already written against this `hc_mix.cu`/`gdn.cu` baseline. The Docker build is the first compile.
- **Equality is by construction plus a harness, not yet by run.** The compiler could in principle schedule the re-grid kernel's `rsqrtf`/`__expf` differently. Both kernels use the same source expressions and flags, so bitwise equality is expected, and `tests/test_r6_kernels.py` checks it with `torch.equal` on every output.
- **The harness guards against vacuous equality.** A candidate that silently fell back to the served kernel would still compare equal. `--launch-check` (default on) profiles one R = 4 call per family and fails unless `gdn_ba_gemv_kernel<1>`, `hc_apply_kernel<…,32>` and `gr_v2_state_regrid_kernel` (fp16 and int8) actually launched.
- **Host cost.** `gr_mix_v2_int8_regrid` and the two C++ selectors call `at::cuda::getDeviceProperties` on every call (cached by ATen). Decode runs under CUDA graphs, so this only matters for eager calls.
- `EXL3_GR_STATE_REGRID=1` together with `EXL3_DECODE_FUSE=1` makes the re-grid take precedence on the fp16 branch (r5 behaviour). The daily does not set `DECODE_FUSE`, and on the int8 branch `DECODE_FUSE` has no effect.
- `tests/test_round6_cpu.py` reads the served source from `R6_SERVED_SRC` and has no default; the tests that need it skip when it is unset.
- r6 touches different files from decode-kernels r3b (`exl3_moe_coop*`). The two can be stacked in either order, but their A/Bs are separate.

## Local validation

`tests/test_round6_cpu.py` (pytest, torch-free): **17 passed, 0 skipped** (Python 3.11, pytest 9.1.1). The tests cover:
- patch hash; touched files equal the manifest; served baseline hashes;
- apply at `-F0` with exit 0, no `.rej`, and overlay hashes; the patch refuses a second forward application;
- `install.py --root` on a scratch copy passes, and a second run aborts; a drifted baseline aborts before anything is patched;
- literal default-off selectors; served entry points pass `false`; `FREE_UP_H` is absent;
- arithmetic-order markers (the FMA chain appears once in each state kernel); the int8 launcher branches only on the state kernel;
- bindings keep `exl3_moe_prefill_e3_det` and `gr_mix_v2_int8`; Dockerfile contract;
- the daily-env routing facts from the central question;
- GPU harness `--dry-run` without torch (`python -S`), which plans 32 B/A + 64 `hc_apply` + 128 V2-state cases.

No CUDA result is implied.
