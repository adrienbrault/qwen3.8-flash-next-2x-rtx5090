# Box validation and OFF/ON/OFF2/ON2 specification

Nothing in this file was run on the GPU box from this workspace. It is an operator-run rejection
or promotion protocol for `EXL3_SHARED_EXPERT_OVERLAP`.

## Build the overlay

Build from the worktree root:

```bash
docker build \
  --build-arg BASE=tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap-ppipe-nosync-mtpfix2-moecoopv2 \
  -f out/decode-kernels-r2/overlay/Dockerfile \
  -t tabbyapi:decode-kernels-r2 .
```

The overlay replaces four native source files:

- `exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp`
- `exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.h`
- `exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu`
- `exllamav3/exllamav3_ext/quant/exl3_moe_coop.cuh`

The installer checks the exact served baseline and payload SHA-256 for every file, aborts on any
mismatch, removes only the package-local precompiled extension, and the next image layer rebuilds
for `TORCH_CUDA_ARCH_LIST=12.0`
(`out/decode-kernels-r2/overlay/install.py:15-47`;
`out/decode-kernels-r2/overlay/build_extension.py:1-20`;
`out/decode-kernels-r2/overlay/Dockerfile:1-9`). Archive the image digest and rebuilt extension
SHA-256.

## Mandatory checkpoint-backed GPU parity and microbenchmark

Run before TabbyAPI, with both 5090s visible and the served checkpoint mounted:

```bash
docker run --rm --gpus all \
  --mount type=bind,src="$CKPT",dst=/checkpoint,readonly \
  tabbyapi:decode-kernels-r2 \
  python3 /opt/decode-kernels-r2/tests/test_shared_expert_overlap.py \
    --model /checkpoint --gpu-split 30,30 --expect-devices 2 \
    --bench --warmup 50 --repeats 500
```

The parent launches fresh OFF and ON children, so the constructor-latched flag cannot leak between
arms (`out/decode-kernels-r2/tests/test_shared_expert_overlap.py:153-175`). Each child:

- verifies the installed patched source hash and exactly two visible devices;
- loads the checkpoint layer-split at 30/30;
- chooses one embedded shared-expert MoE layer on each card;
- primes eager execution, graph capture and replay;
- saves block outputs for every row count 1..16 from deterministic identical fp16 inputs;
- when `--bench` is present, reports CUDA-event medians for one layer at rows 1, 4 and 16.

The parent requires the same selected layer per card and `torch.equal` for every output tensor
(`out/decode-kernels-r2/tests/test_shared_expert_overlap.py:66-150,178-226`). Any unequal element,
missing card/layer, source mismatch, load/build/CUDA failure, non-finite timing or runtime warning is
an immediate abort. Save complete stdout plus driver, CUDA, PyTorch, clocks, power and temperature.

Run one short profiler capture for each card/row setting after warmup. ON must show shared kernels
on a second stream; routed rot/A may overlap them; routed B must begin only after shared down has
ended. If shared kernels remain on the main stream, B overlaps shared down, or the extension still
executes the old launch order, reject the build. Use unprofiled medians for performance decisions.

## Fixed serving configuration

Every arm is a fresh process/container because `EXL3_SHARED_EXPERT_OVERLAP` is read while bound MoE
objects are constructed. Keep checkpoint, GPU placement `[30,30]`, PCIe placement, KV 8,8, vision
setting, cache sizing, prompts, server arguments and MTP policy `[[4, 3], [8, 1]]` fixed. Retain the
promoted settings in every arm:

```text
EXL3_HOST_GAP_REWIND=1
EXL3_HC_MIX_V2=1
EXL3_HC_MIX_V2_MIN_R=1
EXL3_LS_PREFILL_PIPELINE=1
EXL3_MOE_COOP_V2=1
```

Only vary:

- OFF/OFF2: `EXL3_SHARED_EXPERT_OVERLAP=0` or unset;
- ON/ON2: `EXL3_SHARED_EXPERT_OVERLAP=1`.

Do not enable the round-1 `EXL3_DECODE_FUSE` proposal. It was not promoted and is unrelated to this
overlay. Use the operator's existing daily boot, health, warmup, fingerprint and `fn_bench`
procedures; the private `fn_bench` CLI is not in this checkout, so no flags are invented here.

## Arm order and required matrix

Run in this order to bracket thermal and clock drift:

| arm | overlap flag | after health and warmup |
|---|---:|---|
| OFF | 0 | fingerprints; code and prose at c1 and c4, 2,048 forced tokens ×2 each |
| ON | 1 | identical work |
| OFF2 | 0 | identical work |
| ON2 | 1 | identical work |

That is eight `fn_bench` observations per arm: code c1 ×2, code c4 ×2, prose c1 ×2 and prose c4
×2. Retain aggregate tokens/s, per-stream decode tokens/s, per-stream wall tokens/s, TTFT, exact
generated-token counts, MTP proposed/accepted counts or histogram, errors and device telemetry.
Compare aggregate throughput at c4 and per-stream decode throughput at c1; do not mix them.

Run both canonical greedy checks in **every** arm, retain full token IDs, and verify:

- c1 fingerprint: `1474eee2f5945248`
- 30k-context fingerprint: `4a255910dee2d9c5`

OFF must first reproduce the daily values. Every ON output must match its bracketed OFF output
token-for-token, not only by hash. MTP proposed/accepted counts and acceptance histograms must
also match for identical requests.

For one warmed code c1 and c4 request in OFF and ON, retain a short Kineto timeline. OFF must show
the shared graph before routed rot/A/B on one stream. ON must show a distinct shared stream and the
pre-B dependency. Report actual overlap in microseconds for each shared kernel; do not infer it from
different stream IDs alone.

## Decision rules

Correctness gates are absolute:

- both-card, rows-1..16 unit test is fully `torch.equal`;
- all fingerprints, full token IDs and MTP acceptance data match in all four serving arms;
- no extension, CUDA graph, event, cooperative launch, server or teardown error occurs.

Performance GO requires all of:

- ON one-layer median is faster at rows 1, 4 and 16 on both cards, with no tested row slower;
- timeline proves real simultaneous execution, not only queued work on two streams;
- paired `(ON vs OFF)` and `(ON2 vs OFF2)` move in the same direction for code and prose at c1 and
  c4;
- median paired gain is at least 1% at both c1 and c4, with no ON observation more than 1% slower
  than its bracketed OFF baseline;
- clocks, temperature, power and MTP acceptance do not explain the direction.

Immediate NO-GO: any correctness difference; no real kernel overlap; either card's isolated median
regresses; serving regresses by more than 1%; direction flips across brackets; unstable acceptance;
OOM; wrong architecture build; or invalid thermal/clock pairing. Given the cooperative shared GEMMs
and occupancy-capped routed A, **no material gain is the prior expectation at any served shape**.
Rollback is the unchanged base image with the overlap flag unset.

Option C was not implemented because shared K=5 and routed K=3 are incompatible with the uniform
routed kernel instantiation. Therefore no GSM8K gate is applicable to this build.
