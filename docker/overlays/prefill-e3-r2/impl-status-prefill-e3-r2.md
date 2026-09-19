# Implementation status — grouped MoE prefill E3 round 2

## Status

Implemented as a default-off replacement overlay against
`tabbyapi:decode-kernels-r2-refbase`. It replaces round 1 rather than stacking on it: both served
files are checked against the original refbase hashes before the patch is applied.

- `exllamav3_ext/bindings.cpp` baseline:
  `4f099e82c2610f596d22e308cce7f8249bd3a3cfa15fd280c913ce18153b197b`
- `modules/block_sparse_mlp.py` baseline:
  `97461cdc2e14f156b9f9738ed45a60f952bb1ea2186b067c92d888c25bff7cec`
- patched bindings:
  `e855634c456c9a980987d0fa9390d7347b8bf5a98e45197c022c610823b68ddd`
- patched block-sparse MLP:
  `9a4f4279e026320db5ff3e73cf878dfcc433e865637d13ef5599456cb383dada`

The flag remains literal-`1` gated. Unset, `0`, or any other value uses the served planner and
arithmetic. The minimum is clamped to 512 rows, so decode-sized calls can never enter E3.

## What changed from round 1

Round 1 compiled one K3/mul1 device path and required gate, up and down all to be K3. Round 2:

1. passes gate K, up K and down K independently from each layer;
2. accepts each projection K independently in `{2,3,4}` while still requiring mul1;
3. instantiates all nine gate/up K pairs and all three down K variants;
4. sizes packed-B shared-memory staging separately for the two streams; and
5. dispatches the gate/up and down launches independently, so a mixed projection signature such
   as `[2,3,4]` is valid.

The B-stage layout is the only K-dependent data layout. A trellis tile holds `16*K` uint16 words:
32/48/64 words for K2/K3/K4. Each specialization calls the existing exact mul1 dequant primitive
`dq_dispatch<K,2>`. Gate and up keep independent SUH/SVH pointer tables; down keeps its own K and
scales.

The CTA tile remains 64×128×32 for every K. M/N/K matrix dimensions and MMA fragments do not vary
with trellis width, and the native dequant helper already has optimized aligned K2 and K4 paths plus
the generic K3 path. Worst-case gate/up dynamic shared memory is 48 KiB at K4/K4 (44 KiB at K3/K3,
40 KiB at K2/K2); down is 32/28/24 KiB. All retain the intended two-CTA launch bound on sm_120, so
shrinking the row tile would lose B reuse without solving a resource limit.

The four device phases are unchanged structurally:

1. build fat-row and 64-row segment metadata from the GPU histogram;
2. gather and independently rotate gate/up inputs;
3. run the selected paired gate/up specialization and fused SwiGLU/down-input transform; and
4. run the selected down specialization and fp32 atomic routed scatter.

Experts with 1–32 rows remain on the served `exl3_moe_kernel`; only >32 rows enter E3. E3 suppresses
the host count readback and the deterministic slot/gather allocation for the active call.

## Served pack layout

The supplied aggregate counts are 38,400 K2 tensors, 35,328 K3 tensors and 1,536 K4 tensors,
equivalent to 25, 23 and 1 complete 512-expert × three-projection layers. Aggregate counts alone do
not prove that every layer has a uniform gate/up/down K. The checkpoint is not present in this
worktree, so the exact layer signatures could not be inspected locally.

`tests/inspect_checkpoint_k_layout.py` reads only safetensors headers inside the image and emits
every layer's gate/up/down signature plus `mixed_projection_layers`. The box protocol makes this a
mandatory pre-benchmark step. Mixed signatures require no fallback in the implementation and are
also printed by the real-weight benchmark.

## Expected performance by K

No performance result is claimed without the box. At 2048 rows, the engineering priors for an
isolated routed-MoE layer are:

| K | Expected OFF/ON speedup | Reasoning |
|---|---:|---|
| 2 | 1.05–1.20x | 64-byte packed 16×16 tile and cheap aligned unpack leave less dequant cost to amortize. |
| 3 | 1.10–1.30x | 96-byte tile and generic K3 unpack make grouping/control and B reuse relatively more valuable. |
| 4 | 1.05–1.25x | 128-byte tile raises B traffic, but K4 has an aligned unpack path; the pack has only one layer-equivalent. |

At 512 rows the uniform mean is ten assignments per expert, so usually no expert exceeds the
32-row threshold. Expected speed is about 0.97–1.00x for all K: the test is primarily an overhead
gate. At 2048 rows the mean is 40, making a substantial fat tier plausible, but real routing skew
controls the fraction.

These ranges are deliberately much smaller than Mia's reported +37–45% cold prefill. Mia used
K4/MCG, H=4096, I=1024, 288 experts/top-8 and much fatter per-expert batches; this target is
H=2560, I=640, 512 experts/top-10 at 512/2048-row chunks. Only measured per-K and end-to-end box
results can promote the change.

## Workspace and numerical contract

Workspace is independent of K because packed weights are resident model tensors and only fat-row
activation/metadata capacity is allocated per call:

| Input rows | Assignments | Upper-bound workspace |
|---:|---:|---:|
| 512 | 5,120 | 59,061,192 bytes = 56.33 MiB |
| 2,048 | 20,480 | 236,226,312 bytes = 225.29 MiB |

The formula is `2*A*2560*2 + A*640*2 + A*(8+2+4) + 3*4*(ceil(A/64)+512) + 8`.
It excludes allocator rounding, output/sort/histogram, existing served fused buffers, shared expert,
attention and other layers. Dynamic shared memory listed above is per resident CTA, not per row.

ON is not numerically identical to served prefill. The selected E3 specialization preserves the
same codebook values and fp16 boundaries around Hadamard, SUH/SVH, SiLU and gate multiplication,
but changes MMA reduction/rounding and uses atomic top-10 accumulation. For all K, expected
OFF-vs-ON NRMSE is roughly order `1e-3`; nothing predicts a discontinuity at K2 or K4,
because K changes packed decoding rather than matrix dimensions or arithmetic precision. A result
at or above order `1e-2`, or one K behaving materially worse than the others, needs diagnosis.
ON-repeat error should normally be float-atomic order noise, roughly `1e-7`–`1e-6` NRMSE. Absolute
errors are activation-scale dependent and cannot be predicted usefully. These are unverified priors,
not acceptance thresholds.

## Packaging and validation

`Dockerfile.box` is at the round root and its build context is the round directory. Every `COPY`
path is relative to `out/prefill-e3-r2/`, and the Dockerfile contains no heredoc. Image-run tests
discover the installed package with `importlib.util.find_spec("exllamav3")` and otherwise read only
`/opt/prefill-e3-r2`; they never reference `out/...` or `ref/served-src/...`.

`tests/bench_prefill_e3_layer.py` enforces `--use-per-device 30,30` and rows `512,2048`, requires K2
and K3 on every card, adds K4 on the card where it exists, uses identical input for OFF/ON, checks
finite output, and records CUDA-event samples and error statistics. `verify_local.py` is explicitly
local-only and is not copied into the image.

Local CPU integrity checks cover hashes, fuzz-free patch application, Python compilation, K
specialization tokens, literal gating, decode exclusion, Docker context/heredoc rules, installed-test
path independence and benchmark coverage. This machine has no usable GPU; native CUDA compilation,
runtime correctness, actual pack layer signatures, real-weight errors, performance, VRAM headroom,
end-to-end serving and quality remain unverified.

## Delivered files

- `served-source.patch`: replacement integration patch against refbase;
- `overlay/manifest.json`, installer, build helper and three native sources;
- round-root `Dockerfile.box`;
- installed-package smoke test, checkpoint-header inspector and per-K/card GPU benchmark under
  `tests/`;
- `static-derivations.json`;
- `box-ab-spec.md`; and
- this status report plus local-only verifier/result.

The grouped-kernel design and substantial device-code structure remain adapted from the Mia source
under `ref/mia/`, whose changelog identifies AGPL-3.0. Preserve provenance and review licensing
before distributing the derived overlay or image.
