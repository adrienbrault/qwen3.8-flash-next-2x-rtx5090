# port-map — served `stack-r3-rows32` onto upstream dev `73a6229`

The map of the r2 port (upstream `73a6229`). r3 is r2 plus upstream's four commits `73a6229..5783a93`, merged with one conflict ([`impl-status.md`](impl-status.md)), so every row below holds for r3. r1's port map (rows 1 to 25) is not in this repository.

Method (details in impl-status.md). The r1 port (`16f940d` on `1d64111`) was re-expressed as a merge commit `M1` with parents upstream `1d64111` and served `A`. The served `A..B` delta (`B` = `stack-r3-rows32`) was merged onto it (M2, merge-base `A`), then upstream `73a6229` was merged (M3, merge-base `1d64111`). Two fix commits followed. Every live flag is checked by its reader file (`test_flag_selection_r2.py`). r1's port-map rows 1–25 still hold unless a row below supersedes them.

## Part 1 — served mechanisms added since r1 (the `A..B` delta, M2)

| # | feature / hunk group | flag(s) | files | status | risk |
|---|---|---|---|---|---|
| 26 | hcfast r1/r2: V3 int8 decode mixer with per-R tile tables | `EXL3_HC_MIX_V3=2`, `_DOTS_B`, `_UP_B`, `_DOTS_J`, `_DOTS_PF`, `_UP_Q`, `_PDL` | `exllamav3_ext/hc_mix_v3.{cu,cuh}` (NEW, 1,651 ln), `hyperconnections.py` (`_hc_mix_v3_level/_table/_cap`, class knobs, `_mix` dispatch), `bindings.cpp` (`gr_mix_v2_int8_v3`, `hc_mix_v3_revision`) | applied; `hc_mix_v3.cu` byte-identical to served | none at decode; rows > 32 go to upstream tiled prefill (as in r1) |
| 27 | mixstate: state derived inside the up kernel | `EXL3_GR_STATE_IN_UP=1` | `hc_mix.cu` (`gr_mix_v2_int8_statein`), `hyperconnections.py` | applied | none |
| 28 | latchain r1: BC-graph side branches and stage knobs | `EXL3_LC_GDN_RR`, `EXL3_LC_QSA_FORK`, `EXL3_LC_QSA_SPLIT_STAGES`, `EXL3_LC_QSA_COMBINE_STAGES`, `EXL3_LC_QSA_DIV16` | `libtorch/lc_fork.h` (NEW), `attention.cpp`, `attention_bc.h`, `gated_delta_net.cpp`, `gdn.cu`, `bc_attn.py` | merged with upstream `V_DIM` (4163ecb) in `k_sp_combine` | the combine kernel's constexpr set grew on both sides; both compile sites carry all of them |
| 29 | moefast r1/r3: coop V3 + r2 kernels, per-row-count map, early shared-expert fork | `EXL3_MOE_COOP_V3=3`, `EXL3_MOE_COOP_V3_MAP`, `EXL3_SHARED_EXPERT_EARLY` | `exl3_moe_coop_v3_kernel.cuh`, `exl3_moe_coop_r2_kernel.cuh` (NEW), `exl3_moe_coop.cu/.cuh`, `comp_units/exl3_moe_coop_instances.cuh`, `coop_autotune.*`, `blocksparse_mlp.{cpp,h,_bc.h}` (`start_shared`, `shared_early`, `shared_prio`), `block_sparse_mlp.py` (`_shared_early_ok`) | **bridged** to upstream's float-K ABI (row 34) | half-integer K never reaches the served kernels |
| 30 | densegemm r1/r2 + lcguard: V2 dense gemm / mgemm / gemv twins, 17–32-row one-pass | `EXL3_DENSE_V2=1`, `EXL3_DENSE_ROWS32=1` | `exl3_dense_v2.{cu,cuh}`, `exl3_gemm_inner_v2.cuh`, `exl3_gemm_v2_kernel.cuh`, `exl3_gemv_v2_kernel.cuh` (NEW), `comp_units/exl3_dense_v2_inst_{gemm,mgemm,gemv}.cu` (NEW), `exl3_gemm.cu`, `exl3_gemv.cu` | **gated to integer K** (row 34); instance macros updated for upstream's `half_k` template argument | device codegen unverified locally |
| 31 | rows32 r4: routed coop and shared expert at 17–32 rows in one launch | `EXL3_MOE_COOP_ROWS32=1`, `EXL3_SHARED_EXPERT_ROWS32=1` | `exl3_moe_coop.cu` (`exl3_moe_coop_rows_cap`), `libtorch/mlp.{h,cpp}` (`MAX_BSZN 32`), `mlp.py`, `block_sparse_mlp.py` (import guards) | applied; guards and `moe_rows32_revision 2` kept | none |
| 32 | n-gram prefetch 2 (timing only) | `EXL3_NGRAM_PREFETCH2=1` | `ngram_embedding.py`, `ple.py`, `generator.py`, `prefill_pipeline.py` | applied; PLE rebased on upstream's 1k slabs (4b7f13a) | the in-place PLE add returns `delta = None`; audited, nothing reads it |
| 33 | MTP KV window r2 (off since R728) | `EXL3_MTP_KV_WINDOW` (unset) | `cache/mtp_window.py` (NEW), `cache/cache.py`, `generator.py`, `job.py`, `pagetable.py`, `bc_attn.py` | applied, default off | none while off |

## Part 2 — upstream `1d64111..73a6229` (40 commits, M3) and the r2 fixes

| # | upstream change | commits | status in r2 | effect on the served config |
|---|---|---|---|---|
| 34 | fractional trellis: float K across the C++ boundary, `half_k` template argument, h1–h3 instances, `bits_k.cuh`, `frac.cu` | 07b8a2e, 6b84a21 | **kept**; served dense V2 / coop V2/V3/r2 dispatch gated on integer K (`!half_k`, `bits_from_K`) | none for the integer K = 2/3/4 pack; autotune keys change (`half_k` in the hash) <!-- prose-ok: C++ negation in code --> |
| 35 | GDN/KDA: token-major conv1d, prefill sub-ranges (`_chunked_scan`), fp16 prefill output | 421f590 | kept; bf16 state guard added | numerics change (accepted); less prefill scratch |
| 36 | PLE 1k slabs, in-place add | 4b7f13a | kept | numerics order change in PLE prefill (accepted) |
| 37 | warmup without full logits | 623d197 | kept | less warmup VRAM |
| 38 | qcache staging cap + worst-case measurement | bd5b1a3 | kept | inert: every Qwen4Exp attention (and the MTP block) is QSA |
| 39 | HC: no statics for prefill shapes | a1adfbf | kept | prefill workspaces per call (pow2-rounded) |
| 40 | autosplit: resident state allocated up front (`EXL3_AUTOSPLIT_PREPARE`, default 1) | 47dc7ce | kept | the same load loop serves the manual `[30, 30]` split; pool effect unknown (diagnostic knob) |
| 41 | long-query attention 4 warps, FP16ACC GEMM layout for Ada, tiled HC path without cc gate | 0315f22 | kept | sm_120: routing/det-GEMM code simplified; no served-path change found |
| 42 | shared expert before CPU-MoE sync; CPU MoE priming / decode unroll | 383e1a6, 2735c4f, 5e5b4a3, b630b97, 8271af4 | kept | inert (no CPU MoE offload) |
| 43 | MiMo-V2 (arch, SWA ring, TP, vision, MTP with `draft_step`), DFlash sliding-window, asymmetric V dim | 32d43da, f933857, fd9e8ba, febded2, b2fd6ef, 4163ecb, 73ca7c5, 6020e21, f94d75e, 7a051a3 | kept; `draft_step` added to the MTP draft params; `V_DIM` merged into `k_sp_combine` | inert except the two merge points |
| 44 | arch guards for the deterministic GEMM / router (sm_80+), docs | cc43d8c, 2d9751f, e6cfced, f74b324, 3c8dd45 | kept | none on sm_120 |
| 45 | convert.py / YAQA / qbench / perf / tests / docs / MSVC / aarch64 / iGPU memory | d1feab1, 7e2e6b0, 9bbcf1c, 9ac1ca1, 958ec93 (v1.5.1), 181eea1, 887ba49, 5547dca, e3b52f4, e91a7e4, 73a6229 | kept | none (version string 1.5.1) |
| 46 | requeue token count (upstream still one-segment) | — | **tokcount-r1 applied**, `generator/job.py:1053`, flag-free | usage / log / metrics counts exact on generations > 4,096 tokens (R737) |

Upstream commits the brief named that were already in r1's base (`≤ 1d64111`) and stay as r1 merged them: 04fc12d (HC block dots/finalize, fused threshold 8), 825db5b (tiled strong-deterministic HC prefill), 12981f4 (sparse attention tiling), 3b1a57d (sh_coop fused expert), a093e9c (deterministic router GEMM, replicated TP routing), and 9d18a7f (TP for Qwen3.8-Flash-Next + MTP). 9d18a7f is dormant under layer split and live only in the TP arms.

## The kept-both pairs (what the box can A/B by flipping one flag)

1. **Shared expert:** upstream `sh_coop` (`EXL3_MOE_SHARED_COOP`, default on) vs ours `EXL3_SHARED_EXPERT_OVERLAP` + `EXL3_SHARED_EXPERT_EARLY` (served). Arm FSC drops both of ours; EARLY without OVERLAP is refused by the C++ constructor.
2. **HC prefill mixer:** upstream tiled int8 (`EXL3_GR_MIX_TILED=1`, default) vs the cuBLAS path (`=0`, arm PNT). R568 measured about +11–14 % prefill for the tiled path, at about 300 MiB per card.
3. **HC decode mixer:** upstream fused `gr_mix` up to 8 rows vs ours V2 → V3 (`EXL3_HC_MIX_V2`/`_V3` family, served).
4. **Dense / MoE decode kernels:** upstream (flags unset, or any half-integer K) vs ours (`EXL3_DENSE_V2`, `EXL3_DENSE_ROWS32`, `EXL3_MOE_COOP_V2/_V3/_ROWS32`, integer K). The K gate is automatic, not a flag.
5. From r1, still valid: GDN rewind (`EXL3_HOST_GAP_REWIND`), pinned draft staging (`EXL3_DRAFT_PINNED_STAGING`).

## Deliberate behavior changes vs the served image (accepted)

These are all upstream's: the deterministic router GEMM (r1), tiled HC prefill (r1), GDN fp16 prefill projections, token-major conv and scan sub-ranges, PLE in-place slabs, and warmup without full logits. Ours: tokcount-r1. The greedy bytes of P differ from S by design; decode is judged on paired prompts.
