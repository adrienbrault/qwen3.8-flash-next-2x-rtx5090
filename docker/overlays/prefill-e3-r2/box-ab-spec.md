# GPU-box validation and promotion specification — prefill E3 round 2

Nothing in this file was run on the GPU box. This is the operator-run rejection or promotion
protocol for `EXL3_MOE_PREFILL_E3=1`; no timing result is checked into the artifact.

## 1. Build from the round directory context

Run from the worktree root. The final argument is intentionally the round directory, not `.`:

```bash
docker build \
  --build-arg BASE=tabbyapi:decode-kernels-r2-refbase \
  -f out/prefill-e3-r2/Dockerfile.box \
  -t tabbyapi:prefill-e3-r2 \
  out/prefill-e3-r2
```

`Dockerfile.box` uses no heredocs. It hash-checks the two refbase files, installs the three new
sources, applies the replacement patch with fuzz disabled, rebuilds for sm_120, runs an
installed-package static smoke test, and checks the native symbol. A baseline/payload mismatch,
compile/link/import failure, or wrong architecture is an immediate NO-GO. Archive the image
digest, rebuilt extension SHA-256, driver, CUDA, PyTorch, clocks, power limit and both GPU names.

## 2. Check the checkpoint's actual per-projection layout

This CPU-only command reads `/checkpoint` safetensors headers from inside the image. It does not
read the worktree:

```bash
docker run --rm \
  --mount type=bind,src="$CKPT",dst=/checkpoint,readonly \
  --mount type=bind,src="$RESULTS",dst=/results \
  tabbyapi:prefill-e3-r2 \
  python3 /opt/prefill-e3-r2/tests/inspect_checkpoint_k_layout.py \
    --model /checkpoint \
    --strict-served-pack \
    --out /results/prefill-e3-k-layout.json
```

The strict check requires tensor counts K2=38,400, K3=35,328 and K4=1,536, 512 experts in each
projection, and K only in `{2,3,4}`. Preserve `mixed_projection_layers`: an empty list proves that
the served pack has no gate/up/down K mismatch; a nonempty list is supported and must be exercised
by the benchmark. Header inspection was not possible in this worktree because the checkpoint is
only mounted on the box.

## 3. Real-weight per-layer microbenchmark on both cards

```bash
docker run --rm --gpus all \
  --mount type=bind,src="$CKPT",dst=/checkpoint,readonly \
  --mount type=bind,src="$RESULTS",dst=/results \
  -e EXL3_MOE_PREFILL_E3=1 \
  tabbyapi:prefill-e3-r2 \
  python3 /opt/prefill-e3-r2/tests/bench_prefill_e3_layer.py \
    --model /checkpoint \
    --use-per-device 30,30 \
    --rows 512,2048 \
    --warmup 5 \
    --iterations 20 \
    --out /results/prefill-e3-r2-layer.json
```

The harness fails unless every visible card has an eligible K2 and K3 layer. It reports one real
checkpoint layer for K2 and K3 on each card, plus K4 on whichever card contains it. For a mixed-K
layer it records the complete `[gate, up, down]` signature. Every layer/row case runs OFF and ON
on one identical input, requires finite outputs, reports max/mean absolute error, RMSE, NRMSE,
ON-repeat error, routing skew, and all CUDA-event samples.

Capture a short warmed trace for each K/row/card case. An ON case with experts above 32 rows must
contain metadata, gather, the matching templated gate/up specialization, and the matching down
specialization. Verify 1–32-row experts remain in `exl3_moe_kernel`, every >32-row expert runs
exactly once in E3, and no reconstruct/cuBLAS tier also consumes it.

Microbenchmark GO requires:

- both cards report K2 and K3, and the card holding K4 reports it;
- both 512 and 2048 rows exist for every selected layer, with finite positive event timings and
  finite OFF/ON/ON-repeat outputs;
- 2048-row ON median is below OFF for K2 and K3 on both cards and for K4 where present;
- 512-row ON is no more than 3% slower for any K (this mostly prices metadata/empty-fat overhead);
- a repeated run has the same speedup direction; and
- peak memory leaves serving headroom. Candidate global workspace bounds are 56.33 MiB at 512
  rows and 225.29 MiB at 2048 rows, independent of K and excluding allocator rounding/existing
  served buffers. Per-CTA dynamic shared memory is 40/44/48 KiB for uniform K2/K3/K4 gate+up and
  24/28/32 KiB for down; it is not an additional per-row allocation.

Numerical equality is not a gate. The engineering expectation for OFF-vs-ON NRMSE is the same
order for K2, K3 and K4: roughly `1e-3`, with routing/activation-dependent excursions possible.
Investigate an order-of-magnitude jump (`>=1e-2`), a strong K-specific discontinuity, or any
non-finite result before end-to-end testing. ON-repeat differences should be only float-atomic
summation noise, typically `1e-7`–`1e-6` NRMSE; these are priors, not measured limits.

## 4. Fixed serving A/B

Use a fresh server/container per arm because the flag is read at import. Hold image, checkpoint,
`[30,30]` placement, topology, KV 8,8, cache sizing, chunk size 2048, MTP policy, prompts,
sampling and promoted decode/pipeline flags fixed. Vary only:

- OFF/OFF2: flag unset or `EXL3_MOE_PREFILL_E3=0`;
- ON/ON2: `EXL3_MOE_PREFILL_E3=1`.

Run `OFF → ON → OFF2 → ON2`. After identical warmup and the established cold-prefix procedure:

```bash
/srv/qwen5090/probes/fn_bench.py --ctx 30000 120000 --unique
```

Retain complete output and, separately at 30k/120k, input tokens, chunk count, prefill wall,
prompt tok/s, TTFT p50/p95, decode tok/s, errors, peak allocated/reserved VRAM, and card telemetry.
Confirm full 2048-token chunks enter E3 and no prefix-cache hit contaminates the result.

Promotion requires both brackets to improve at both contexts, median paired prompt-throughput gain
at least 5%, no worse p95 TTFT, no OOM/retry, and no card starvation/pipeline bubble. The plausible
isolated-layer prior at 2048 rows is modest and geometry-dependent: about 1.05–1.20x for K2,
1.10–1.30x for K3, and 1.05–1.25x for K4. K2 has the smallest/cheapest aligned packed stream;
K3's generic unpack leaves the most dequant/control cost to amortize; K4 doubles K2 packed bytes
but has aligned unpack. At 512 rows expect roughly 0.97–1.00x because usually no expert exceeds
32 rows. These are planning ranges, not acceptance evidence. Mia's +37–45% used K4/MCG,
H=4096/I=1024, 288 experts/top-8 and much fatter per-expert batches, so it is not transferable.

## 5. Decode isolation and quality

Run the established warmed code/prose decode matrix at c1 and c4, 2,048 forced output tokens,
twice per arm. An ON decode-only trace must contain zero `e3_*` launches: E3 requires at least 512
rows, while served decode is at most 16. Decode passes inside the established noise band with no ON
arm more than 1% slower than its bracketed OFF arm.

Run the fixed GSM8K n=200 manifest in OFF and ON with identical greedy settings and archive raw
responses, token IDs, extracted answers and per-item correctness. Require all requests finite and
complete, ON correct count not below OFF, manual review of every correctness flip, and no new
truncation/empty/malformed-answer cluster. Greedy identity is not required because prefill values
and downstream routing/cache state may differ.

## 6. Promotion and rollback

Promote only if build, header layout, every per-K/card microbenchmark, both cold-prefill contexts,
decode isolation and GSM8K pass. Otherwise leave the flag unset. Runtime rollback is a restart with
the flag unset; strict source rollback uses `tabbyapi:decode-kernels-r2-refbase`.
