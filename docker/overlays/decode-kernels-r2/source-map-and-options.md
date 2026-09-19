# Shared-expert decode path: source map and options

## Disposition

Option A is the only implementable bit-identical candidate in this source snapshot. It is shipped
behind literal opt-in `EXL3_SHARED_EXPERT_OVERLAP=1`; unset or `0` retains the served execution
path. It launches the existing shared-expert CUDA graph on one nonblocking stream, launches routed
rotation/stage A on the original stream, and inserts an event dependency immediately before routed
stage B. No device arithmetic, buffers, weights, routing decisions, or combine order change
(`out/decode-kernels-r2/overlay/payload/exllamav3_ext/libtorch/blocksparse_mlp.cpp:18-23,78-126`;
`out/decode-kernels-r2/overlay/payload/exllamav3_ext/quant/exl3_moe_coop.cu:143-176`).

This is a correctness-ready measurement candidate, not a claimed speedup. The routed V2 kernels
are ordinary `cudaLaunchKernel` launches, not cooperative launches, but their grid is explicitly
capped at occupancy capacity. At the 40- and 160-slot shapes it is intended to keep all available
CTA residency busy. Meanwhile, the shared EXL3 GEMMs are cooperative, grid-synchronising launches
with 90 KiB dynamic shared memory. Useful concurrent residency is therefore doubtful at every
served shape and must be demonstrated in a GPU timeline. c1 d0 selects the narrow 32-column routed
tile, giving a 400-CTA A base grid capped at two occupancy waves; c1 d3/c4 d3 select the wide tile
and an occupancy-capacity wave. None has an obvious pool of idle SMs
(`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu:65-86,122-170`;
`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop_v2_kernel.cuh:16-19,522-553`;
`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh:5-7`;
`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_gemm.cu:289-304,644-662`).

Option B is not bit-identically expressible through the served GEMM API as one generic launch (or
two generic launches), and Option C is format-incompatible with the routed V2 instantiation in this
checkpoint. Neither is shipped.

## Important format correction

The shared and routed expert dimensions are both 2560↔640 and there are 48 target layers
(`ref/r464-r465/r464/run-123307/long/ctx30720_b4_d0/environment.json:649-691`). They use the same
mul1 codebook: the checkpoint config says `codebook: mul1`, the loader tests each linear's `.mul1`
tensor, and `LinearEXL3` maps it to `mul1=True`
(`ref/r464-r465/r464/run-123307/long/ctx30720_b4_d0/environment.json:718-730`;
`ref/served-src/exllamav3/modules/linear.py:385-425`;
`ref/served-src/exllamav3/modules/quant/exl3.py:60-88`).

Their bit widths are **not** the same:

- Routed expert gate/up/down linears use the ordinary `bits` allocation
  (`ref/served-src/exllamav3/architecture/qwen4_exp.py:188-205`).
- The shared expert passes `select_hq_bits=2`
  (`ref/served-src/exllamav3/architecture/qwen4_exp.py:207-220`).
- With checkpoint `bits=3.05`, allocation starts at `floor(3.05)=3` and the HQ minimum is
  `3 + 2 = 5` when HQ conversion is enabled
  (`ref/served-src/exllamav3/conversion/allocation.py:55-87,161-166`).
- Runtime K is read from `trellis.shape[-1] // 16`; the R464 trace resolves shared kernels as K=5,
  cb=2 and routed V2 kernels as K=3, cb=2
  (`ref/served-src/exllamav3/modules/quant/exl3.py:60-77`;
  `out/decode-kernels-r1/inventory.tsv:22-26,71-76`).

The captured checkpoint metadata does not include its generated `quantization_config.json`
tensor-storage map or raw safetensor headers, so unavailable headers are not claimed as inspected.
The served generator would enumerate every leaf's stored tensors, derive `bits_per_weight` from the
trellis shape and record a present `.mul1` multiplier (`ref/served-src/exllamav3/conversion/quant_config.py:15-62`).
The architecture supplies `mlp.shared_expert.{gate,up,down}_proj`; the loader appends `.trellis`,
scale and `.mul1`, and the traced specialisations validate what loaded (`ref/served-src/exllamav3/modules/linear.py:385-425`).

Consequently, trellis payload alone is:

```text
48 layers × 3 matrices × 2560 × 640 weights × 5 bits / 8
= 147,456,000 bytes per target step
```

That excludes SU/SV metadata and activations. It is not 90 MB. Dividing this corrected lower-bound
payload by the measured shared group gives 146.07, 129.55 and 120.65 GB/s for c1 d0, c1 d3 and c4
d3 respectively. At the routed estimate of 431 GB/s the payload-only floor is 342.1 µs, leaving a
derived 0.667/0.796/0.880 ms, or 7.49/5.38/3.75% of the unprofiled wall. These are model estimates,
not new measurements; inputs and formulas are in `out/decode-kernels-r2/overlap-estimates.tsv`.

## Source map: construction and execution

### Architecture and block order

`build_qwen4_block` constructs each transformer block with hyperconnection mixers, either GDN or
QSA attention, and a `BlockSparseMLP`. The routed expert is 512-way/top-10 with 640 intermediate
width; the shared expert is a separate `GatedMLP` with learned fp16 `shared_expert_gate` and the HQ
setting described above (`ref/served-src/exllamav3/architecture/qwen4_exp.py:86-110,188-221`;
`ref/r464-r465/r464/run-123307/long/ctx30720_b4_d0/environment.json:649-691`).

At the MLP site the transformer performs hyperconnection mix, converts the MLP input to fp16,
applies any MLP norm, calls the MoE, then applies the hyperconnection residual update
(`ref/served-src/exllamav3/modules/transformer.py:173-195`). Inside the MoE:

1. The fp16 input is flattened and routed with gate matmul → top-k → softmax
   (`ref/served-src/exllamav3/modules/block_sparse_mlp.py:918-963`;
   `ref/served-src/exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp:13-64`).
2. Rows 1..16 select the bound cooperative decode path
   (`ref/served-src/exllamav3/modules/block_sparse_mlp.py:937-943,1262-1274`).
3. During load, eligible shared expert and fp16 shared gate bound classes are embedded in
   `BC_BlockSparseMLP`; shared output scratch is fp32 `(1,16,2560)`
   (`ref/served-src/exllamav3/modules/block_sparse_mlp.py:567-589,618-673`).
4. If embedding is unavailable, the Python fallback separately evaluates the shared expert and
   uses `add_sigmoid_gate_proj`, or an ordinary add without a gate
   (`ref/served-src/exllamav3/modules/block_sparse_mlp.py:1303-1317`). This fallback is not the
   served rows-1..16 path being optimised.

### Shared expert: kernels, stream and workspace

`GatedMLP.load_local` fuses compatible gate/up matrices into `MultiLinear`, allocates four global
cache entries named `gu`, `a1`, `a2` and `down_xh`, and constructs `BC_GatedMLP`
(`ref/served-src/exllamav3/modules/mlp.py:631-686`). The cache key includes device, exact shape,
dtype and name, so equal-shaped layers reuse the same allocation; a layer completes before the next
can reuse it (`ref/served-src/exllamav3/util/tensor.py:214-229`). Per runtime row count, gate/up
transform and output buffers are lazily cached at exact shapes
(`ref/served-src/exllamav3/exllamav3_ext/libtorch/mlp.cpp:14-36`).

The device sequence is:

1. fused K=5 mul1 `exl3_mgemm` for gate and up, producing fp16 gate/up scratch;
2. `act_mul_kernel_h<ACT_SILU>`, reading fp16 gate/up and writing an fp16 activation;
3. K=5 mul1 down `exl3_gemm` (or selected int8 GEMV at c1 d0), writing fp32 shared output.

The call sites and dtypes are explicit at
`ref/served-src/exllamav3/exllamav3_ext/libtorch/mlp.cpp:38-90` and
`ref/served-src/exllamav3/exllamav3_ext/activation.cu:24-85`; the activation's half operations and
half store are at `ref/served-src/exllamav3/exllamav3_ext/activation_kernels.cuh:142-185`.

The first call runs eagerly, the second captures, and subsequent calls launch the same nodes as a
CUDA graph on PyTorch's current stream
(`ref/served-src/exllamav3/exllamav3_ext/libtorch/mlp.cpp:93-145`;
`ref/served-src/exllamav3/exllamav3_ext/graph.cu:30-51,129-186`). A graph reduces CPU launch work
but still produces three device kernel nodes per layer, matching the measured 144 launches per
48-layer target step (`out/decode-kernels-r1/inventory.tsv:24-26,74-76`).

### Routed V2 and exact combine precision/order

`BC_BlockSparseMLP` validates the routed gate/up/down codebook, prepares one static `MoeCoopParams`,
and passes the optional fp16 shared-gate weight
(`ref/served-src/exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp:203-252`). Per call,
`exl3_moe_coop_run` binds input, selected experts, fp16 routing weights and optional fp32 shared
output, then launches on the current PyTorch stream
(`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu:337-380`).

With `EXL3_MOE_COOP_V2=1`, rows 1..16 execute optional input rotation, routed A, then routed B. All
three use ordinary launch syntax; there is no `cudaLaunchCooperativeKernel` and no CTA-wide grid
barrier. Bounded work loops and last-arrival atomics only elect epilogue owners
(`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu:100-170`;
`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop_v2_kernel.cuh:522-553`). The launcher
queries compiled occupancy and caps the grid to two waves below 32 slots and one wave at/above 32
slots (`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu:122-140`;
`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop_v2_kernel.cuh:16-19`).

The exact served combine happens inside routed B, not in an fp16 add afterward:

- each fp16 routing weight is converted to float;
- down split-K partials and top-k routed contributions are accumulated in fixed fp32 order;
- shared-gate dot consumes fp16 input/weight, accumulates in fp32, then uses `__expf` for sigmoid;
- after routed warp partial reduction, B executes `routed_float += gate_float * shared_float` and
  stores fp32 output.

See `ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop_kernel.cuh:201-223` and
`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop_v2_kernel.cuh:437-515`. The separate
fallback add implements the same high-level fp32 update
(`ref/served-src/exllamav3/exllamav3_ext/activation_kernels.cuh:369-410`).

## Option A — exact stream overlap

```text
main: prior/norm/routing ─ record input ─ rot ─ routed A ─ wait shared ─ routed B
                                  │                         ▲
side:                       wait input ─ shared GU ─ act ─ down ─ record done
```

The flag is parsed strictly as unset/`0` or `1`; other values fail. Resources are created only for
layers with an embedded shared expert and only under the opt-in. OFF creates no side stream/events
(`out/decode-kernels-r2/overlay/payload/exllamav3_ext/libtorch/blocksparse_mlp.cpp:18-23,302-324`).
ON records input readiness, redirects PyTorch's current stream with `CUDAStreamGuard`, calls the
unchanged shared graph, records completion, and passes that event to the routed launcher
(`out/decode-kernels-r2/overlay/payload/exllamav3_ext/libtorch/blocksparse_mlp.cpp:96-126`). Both V2
and fallback launchers wait after A and before B
(`out/decode-kernels-r2/overlay/payload/exllamav3_ext/quant/exl3_moe_coop.cu:143-175,179-224`).

Bit identity follows from the dependency graph: input is ready before either branch; every kernel
within each branch is unchanged and stream-ordered; B cannot read shared output before shared down
completes; B retains the existing routed reduction and final shared-add order. GPU `torch.equal`
remains an absolute release gate rather than a source-only claim.

### Measured-window ceiling, not a prediction

| shape | shared µs/layer | rot+A µs/layer | ideal hidden µs/step | ideal wall share |
|---|---:|---:|---:|---:|
| c1 d0 (1 row, 10 slots) | 21.030 | 25.277 | 1,009.463 | 11.33% |
| c1 d3 (4 rows, 40 slots) | 23.712 | 41.881 | 1,138.162 | 7.69% |
| c4 d3 (16 rows, 160 slots) | 25.463 | 95.075 | 1,222.213 | 5.20% |

These are derived from R464 kernel durations (`out/decode-kernels-r1/inventory.tsv:22-26,71-76,
178-182`; `out/decode-kernels-r2/overlap-estimates.tsv:2-4`). The windows fit, but time-window
overlap is not resource overlap. Shared gate/up and regular down use cooperative EXL3 GEMMs with
grid barriers and 90 KiB launch shared memory
(`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_gemm_kernel.cuh:8-49,88-167`;
`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_gemm.cu:289-304,644-662`). c1 d3 and c4 d3
cross the routed busy-slot threshold and use one occupancy-capacity wave. c1 d0 is below the
threshold but uses the narrow tile, so its A base is
`10 slots × 2 projections × (640 / 32 columns) = 400` CTAs, capped at two occupancy waves
(`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu:65-86,122-165`).

If 90-KiB cooperative GEMMs cannot co-reside with routed A, a more realistic source-derived ceiling
is only the non-cooperative tail of the shared branch: c1 d0 can at most hide its int8 down plus
activation, 343.60 µs/step (3.86% wall); c1 d3 can at most hide 36.99 µs of activation (0.25%);
c4 d3 can at most hide 38.35 µs (0.16%). Even those bounds assume the scheduler overlaps that tail
and omit event overhead. Zero is the only guaranteed overlap. Kernel classifications/durations come
from `out/decode-kernels-r1/inventory.tsv:24-26,74-76,180-182`; wall denominators come from
`out/decode-kernels-r2/overlap-estimates.tsv:2-4`.

Plain conclusion: source gives no basis to promise the ideal 1.0–1.2 ms at any served shape. Every
routed A launch initially covers the SM set, and the cooperative shared GEMMs reserve heavy
resources, so A is likely to expose little or no overlap unless the two kernels have complementary
residency. **If CUDA cannot co-reside the 90-KiB cooperative GEMM block with a routed CTA, A cannot
help at all; it will serialize and add event/stream overhead.** If the ON timeline serializes the
shared GEMMs or either card's one-layer median is not faster, A is a NO-GO.

## Option B — one/two fused shared launch

Exactness is not available by composing the served GEMM entry points:

- `exl3_mgemm` and regular `exl3_gemm` are whole cooperative kernels with internal `grid.sync()`;
  the autotuner can choose distinct block counts, block shapes and concurrency for gate/up and down
  (`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_gemm_kernel.cuh:8-49,88-167`;
  `ref/served-src/exllamav3/exllamav3_ext/quant/exl3_gemm.cu:238-308,575-666`).
- Exact activation fusion must preserve the gate/up fp16 store/load boundary and half operation
  sequence (`ref/served-src/exllamav3/exllamav3_ext/activation_kernels.cuh:142-185`).
- c1 d0 uses a distinct int8-activation GEMV down path while c1 d3/c4 d3 use regular K=5 GEMM
  (`out/decode-kernels-r1/inventory.tsv:24-26,74-76,180-182`).

Device code cannot inline-invoke those existing parent kernels. A new monolithic kernel has one
physical grid/block/shared-memory geometry, so it cannot generically retain two independently tuned,
`gridDim`-dependent work assignments. A two-launch gate/up+activation variant would require a new
specialised epilogue after a grid-wide barrier in every selected mgemm variant; it removes only the
0.736–0.800 µs activation kernel per layer before host-gap effects.

In principle a new kernel family could reimplement every algorithm with logical grids and explicit
fp16 boundaries, but that is not fusion through the served GEMM scheme and cannot be called exact
without GPU differential coverage of every row/card/path. No `EXL3_SHARED_EXPERT_FUSED` flag is
shipped.

## Option C — always-selected routed expert

The existing routed instantiation cannot represent this checkpoint: routed matrices are K=3/cb=2
and shared matrices are K=5/cb=2. `BC_BlockSparseMLP` and `exl3_moe_coop_prepare` accept one routed
codebook/K per stage, and the launcher selects one `<K,cb>` template
(`ref/served-src/exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp:203-212,229-249`;
`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu:110-120,243-251,331-334`). C needs
mixed-width pointer metadata and K=5 decode in the same launch, not only one extra expert ID.

It would also be non-exact. Current shared output comes from the shared GEMM reduction and is added
after the complete routed fp32 reduction. An extra routed slot would use coop GEMV reduction, change
fp32 summation order, and either round sigmoid gate to fp16 routing-weight representation or need a
new float special-weight path. Minimum expected change is fp32 last-bit movement; GEMM-vs-GEMV
reduction can be larger, so no source-only error bound is defensible. It would need tensor error
histograms and the requested GSM8K n=200 quality gate. Because the format work is not a bounded
extra-slot change, C and its separate flag are not shipped.
