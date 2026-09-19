# Implementation status — decode kernels round 2

## Status

**Implemented, CPU-integrity-checked, not GPU-validated:** Option A is shipped as an opt-in native
overlay under `EXL3_SHARED_EXPERT_OVERLAP=1`; OFF/unset follows the original same-stream call path
and allocates no overlap resources
(`out/decode-kernels-r2/overlay/payload/exllamav3_ext/libtorch/blocksparse_mlp.cpp:18-23,96-126,
302-310`). Option B and C are analysis-only and have no flags or device-code changes
(`out/decode-kernels-r2/source-map-and-options.md:216-255`).

All created implementation, test, analysis and build artifacts are inside
`out/decode-kernels-r2/` (`out/decode-kernels-r2/overlay/Dockerfile:4-9`;
`out/decode-kernels-r2/overlay/manifest.json:1-21`). No GPU result is reported in the local
validation record (`out/decode-kernels-r2/local-validation.json:1-16`).

## Evidence-backed claims

| Claim | File:line evidence |
|---|---|
| The target has 48 layers, hidden width 2560, routed/shared intermediate width 640, 512 routed experts and top-10 routing. | `ref/r464-r465/r464/run-123307/long/ctx30720_b4_d0/environment.json:649-691` |
| Checkpoint config is 3.05 nominal bpw and mul1 codebook. | `ref/r464-r465/r464/run-123307/long/ctx30720_b4_d0/environment.json:718-730` |
| Routed matrices are K=3/cb=2 in the trace; shared matrices are K=5/cb=2. | `out/decode-kernels-r1/inventory.tsv:22-26,71-76,178-182` |
| Shared K=5 is explained by `select_hq_bits=2`, floor(3.05)=3 and HQ clamping to the 5-bit minimum. | `ref/served-src/exllamav3/architecture/qwen4_exp.py:188-220`; `ref/served-src/exllamav3/conversion/allocation.py:55-87,161-166` |
| EXL3 runtime derives K from trellis shape and codebook from `.mcg`/`.mul1` tensors. | `ref/served-src/exllamav3/modules/quant/exl3.py:60-88`; `ref/served-src/exllamav3/modules/linear.py:385-425` |
| Corrected shared trellis payload is 147.456 MB/step; measured-duration payload rates and 431-GB/s projections are derived, not measurements. | `out/decode-kernels-r2/overlap-estimates.tsv:1-4`; formula in `out/decode-kernels-r2/source-map-and-options.md:62-76` |
| Transformer order is HC mix/norm → MoE → HC residual apply. | `ref/served-src/exllamav3/modules/transformer.py:173-195` |
| Served rows 1..16 route through `BC_BlockSparseMLP`, with embedded shared expert and fp16 shared gate. | `ref/served-src/exllamav3/modules/block_sparse_mlp.py:567-589,618-673,937-943,1262-1274` |
| Shared launch order is K=5 gate+up → half activation → K=5 down-to-fp32, using `gu`, `a1`, `a2`, `down_xh` workspaces and a per-row-count CUDA graph. | `ref/served-src/exllamav3/modules/mlp.py:631-686`; `ref/served-src/exllamav3/exllamav3_ext/libtorch/mlp.cpp:14-90,93-145` |
| CUDA graph replay preserves device kernel nodes; it does not fuse the three launches. | `ref/served-src/exllamav3/exllamav3_ext/graph.cu:30-51,129-186`; measured counts at `out/decode-kernels-r1/inventory.tsv:24-26,74-76` |
| Routed V2 uses ordinary `cudaLaunchKernel`, not `cudaLaunchCooperativeKernel`; kernels have bounded loops and no CTA grid sync. | `ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu:143-170`; `ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop_v2_kernel.cuh:522-553` |
| Routed grid size is capped using queried occupancy, with two waves below 32 slots and one at/above 32. | `ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu:122-140`; `ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop_v2_kernel.cuh:16-19` |
| Shared regular EXL3 GEMMs are cooperative grid-synchronising launches using 90 KiB launch shared memory. | `ref/served-src/exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh:5-7`; `ref/served-src/exllamav3/exllamav3_ext/quant/exl3_gemm_kernel.cuh:8-49,88-167`; `ref/served-src/exllamav3/exllamav3_ext/quant/exl3_gemm.cu:289-304,644-662` |
| Current combine is already inside routed B: routing and shared-gate math reduce in fp32, then shared fp32 output is gate-scaled and added after the routed sum, with fp32 output store. | `ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop_kernel.cuh:201-223`; `ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop_v2_kernel.cuh:437-515` |
| Option A creates one nonblocking side stream and two disable-timing events per eligible bound layer only when the flag is enabled. | `out/decode-kernels-r2/overlay/payload/exllamav3_ext/libtorch/blocksparse_mlp.h:88-94`; `out/decode-kernels-r2/overlay/payload/exllamav3_ext/libtorch/blocksparse_mlp.cpp:302-324` |
| ON records main-stream input readiness, makes the side stream wait, launches the unchanged shared graph under `CUDAStreamGuard`, then records shared completion. | `out/decode-kernels-r2/overlay/payload/exllamav3_ext/libtorch/blocksparse_mlp.cpp:96-120` |
| Both V2 and fallback routed launchers enqueue their shared-completion wait after A and before B. | `out/decode-kernels-r2/overlay/payload/exllamav3_ext/quant/exl3_moe_coop.cu:143-175,179-224` |
| The routed run accepts the optional event without changing any tensor binding or device kernel arguments. | `out/decode-kernels-r2/overlay/payload/exllamav3_ext/quant/exl3_moe_coop.cuh:80-128`; `out/decode-kernels-r2/overlay/payload/exllamav3_ext/quant/exl3_moe_coop.cu:347-380` |
| Option B cannot reuse the existing parent GEMM kernels inside one launch while retaining independently autotuned physical grids and the separate fp16 activation boundary. | `ref/served-src/exllamav3/exllamav3_ext/quant/exl3_gemm.cu:238-308,575-666`; `ref/served-src/exllamav3/exllamav3_ext/activation_kernels.cuh:142-185`; analysis at `out/decode-kernels-r2/source-map-and-options.md:216-238` |
| Option C is incompatible with the existing uniform K/codebook stage instantiation and would alter reduction/weight precision even after mixed-K support. | `ref/served-src/exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp:203-212,229-249`; `ref/served-src/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu:110-120,243-251,331-334`; analysis at `out/decode-kernels-r2/source-map-and-options.md:240-255` |
| The overlay is baseline-hash-gated, copies four payloads, removes only package-local prebuilt extensions, and rebuilds for sm_120. | `out/decode-kernels-r2/overlay/install.py:15-47`; `out/decode-kernels-r2/overlay/Dockerfile:1-9`; `out/decode-kernels-r2/overlay/manifest.json:1-21` |
| The GPU harness uses fresh OFF/ON children, both devices, one eligible layer per card, deterministic inputs and every row count 1..16, then requires `torch.equal`. | `out/decode-kernels-r2/tests/test_shared_expert_overlap.py:66-150,153-207` |
| The same harness reports one-layer CUDA-event medians at rows 1, 4 and 16 for both cards. | `out/decode-kernels-r2/tests/test_shared_expert_overlap.py:50-63,136-147,208-226` |
| OFF/ON/OFF2/ON2 serving order, c1/c4 code+prose 2,048-forced ×2, and fingerprints are specified without inventing unavailable `fn_bench` flags. | `out/decode-kernels-r2/box-ab-spec.md:80-117` |
| CPU-only validation passed source/payload hashes, patch reproduction, source invariants and Python compilation. | `out/decode-kernels-r2/local-validation.json:1-16`; checker at `out/decode-kernels-r2/verify_local.py:1-78` |

## What remains unverified without the GPU box

- Native CUDA/C++ compilation and sm_120 link compatibility remain unverified; the Docker rebuild
  is the first mandatory box step (`out/decode-kernels-r2/overlay/Dockerfile:7-9`;
  `out/decode-kernels-r2/local-validation.json:10-16`).
- Bit identity remains unverified until the checkpoint-backed two-card rows-1..16 `torch.equal`
  harness passes (`out/decode-kernels-r2/tests/test_shared_expert_overlap.py:124-150,178-207`).
- Actual concurrent residency remains unverified until a GPU timeline shows simultaneous kernel
  execution and B after shared down (`out/decode-kernels-r2/box-ab-spec.md:48-63,114-117`).
- One-layer OFF/ON latency remains unmeasured; the harness defines rows 1/4/16, 50 warmups and 500
  median samples but contains no stored result (`out/decode-kernels-r2/tests/test_shared_expert_overlap.py:24-35,50-63,136-147`;
  `out/decode-kernels-r2/local-validation.json:10-16`).
- Serving throughput, thermals, MTP acceptance, token identity and fingerprints remain unmeasured;
  the complete paired protocol is operator work (`out/decode-kernels-r2/box-ab-spec.md:65-144`).
- CUDA graph capture's one-time `cudaDeviceSynchronize` can obscure an unprimed measurement; the
  harness deliberately primes eager/capture/replay before retention and timing
  (`ref/served-src/exllamav3/exllamav3_ext/graph.cu:30-44`;
  `out/decode-kernels-r2/tests/test_shared_expert_overlap.py:128-139`).
- Side-stream resource destruction during normal model unload/process shutdown remains unexercised
  on CUDA; lifecycle code is at
  `out/decode-kernels-r2/overlay/payload/exllamav3_ext/libtorch/blocksparse_mlp.cpp:313-324`.

## Files delivered

- Source/precision/option analysis: `out/decode-kernels-r2/source-map-and-options.md:1-255`.
- Exact implementation diff: `out/decode-kernels-r2/served-source.patch:1-236`.
- Hash-gated overlay: `out/decode-kernels-r2/overlay/manifest.json:1-21`,
  `out/decode-kernels-r2/overlay/install.py:1-51`, and
  `out/decode-kernels-r2/overlay/Dockerfile:1-9`.
- GPU parity and microbenchmark: `out/decode-kernels-r2/tests/test_shared_expert_overlap.py:1-238`.
- Serving protocol: `out/decode-kernels-r2/box-ab-spec.md:1-144`.
- Derived timing/payload table: `out/decode-kernels-r2/overlap-estimates.tsv:1-4`.
- CPU-only verifier/result: `out/decode-kernels-r2/verify_local.py:1-78` and
  `out/decode-kernels-r2/local-validation.json:1-16`.
