# Round-4 box build and A/B gate

Run only on the operator box under its normal lock. This checkout has not contacted the box.
Round 2 and round 3 must remain disabled throughout this gate; stack them only after round 4 has an
independent result.

## 1. Build the candidate image

From the checkout root:

```bash
docker build \
  -f out/decode-kernels-r4/overlay/Dockerfile.box \
  --build-arg BASE=tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap-ppipe-nosync-mtpfix2-moecoopv2 \
  -t tabbyapi:decode-kernels-r4 .
```

The recipe first hash-checks all nine installed source files against the captured served tree, copies
the edited files directly into `/opt/venv/lib/python3.12/site-packages/exllamav3/`, removes the old
extension, clears the round-specific `TORCH_EXTENSIONS_DIR`, sets `TORCH_CUDA_ARCH_LIST=12.0`, and
imports once. CUDA sources changed, so clearing this cache is required in this round. Abort on a
baseline mismatch, compile error, missing `gr_mix_v2_int8` binding, or a non-sm_120 image.

Smoke-check the installed selectors in a fresh container before loading the checkpoint:

```bash
python3 - <<'PY'
import os
for key in (
    "EXL3_DRAFT_PINNED_STAGING", "EXL3_BATCH_VERIFY", "EXL3_MTP_DEVICE_DRAFT",
    "EXL3_EMBED_GPU", "EXL3_MTP_HEAD_N", "EXL3_HC_MIX_V2_INT8",
):
    assert key not in os.environ
from exllamav3.ext import exllamav3_ext as ext
assert hasattr(ext, "gr_mix_v2_int8")
print("round-4 extension imported; selectors unset")
PY
```

## 2. Offline mixer correctness and timing

Run the 96-site synthetic weight audit on CPU. It uses the real `D=2560, H=4, rank=320` dimensions,
reports mixed/post max absolute, max relative, and mean absolute error, and reports exact storage:

```bash
mkdir -p /results/decode-kernels-r4
python3 /opt/decode-kernels-r4/mixer_int8_parity.py \
  --sites 96 --rows 1 --seed 1234 \
  --json /results/decode-kernels-r4/mixer-int8-synthetic-96.json
```

Then exercise the actual extension on both GPUs in separate processes. This compares the int8 V2
kernel with a torch reference made from the same dequantized weights, records the numerical delta to
served fp16 V2, and CUDA-event-times site and final mixers at rows 1, 4, and 16:

```bash
for R4_GPU in 0 1; do
  python3 /opt/decode-kernels-r4/tests/test_gr_mix_v2_int8.py \
    --device "cuda:${R4_GPU}" --warmup 50 --iterations 500 \
    --json "/results/decode-kernels-r4/mixer-gpu${R4_GPU}.json"
done
```

Require the dequantized-weight reference assertions to pass and all values to be finite. Int8 versus
fp16 is intentionally not an equality gate. Retain per-card medians, clocks, temperature, power,
and peak allocated/reserved VRAM. Do not transfer the fork's V1 1.14–1.34x result to V2; the test
above and serving A/B are the round-4 measurements.

## 3. Common serving configuration

Use the same candidate image for every arm, the existing daily boot/health/warmup procedure, served
placement `[30,30]`, four decode slots, KV 8/8, and draft policy `[[4,3],[8,1]]`. Keep these values
fixed:

```text
EXL3_HOST_GAP_REWIND=1
EXL3_HC_MIX_V2=1
EXL3_HC_MIX_V2_MIN_R=1
EXL3_LS_PREFILL_PIPELINE=1
EXL3_MOE_COOP_V2=1
EXL3_SHARED_EXPERT_OVERLAP=0
EXL3_MOE_COOP_V2_NOSPILL=0
```

Set every round-4 selector explicitly for each process; never rely on inherited environment. The
all-off values are:

```text
EXL3_DRAFT_PINNED_STAGING=0
EXL3_BATCH_VERIFY=0
EXL3_MTP_DEVICE_DRAFT=0
EXL3_EMBED_GPU=0
EXL3_HC_MIX_V2_INT8=0
EXL3_MTP_HEAD_N unset
```

For each group below run `OFF, ON, OFF2, ON2` in that order. Restart the server between arms because
all selectors are read at import/load time. Change only the listed group values:

| group | OFF / OFF2 | ON / ON2 | identity contract |
|---|---|---|---|
| host I/O chain | staging/device/embed all `0` | `EXL3_DRAFT_PINNED_STAGING=1`, `EXL3_MTP_DEVICE_DRAFT=1`, `EXL3_EMBED_GPU=1` | greedy and sampled token/RNG identical |
| batched verify | `EXL3_BATCH_VERIFY=0` | `EXL3_BATCH_VERIFY=1` | greedy token-identical; eligible sampling has the same distribution but a different RNG stream |
| pruned draft head | `EXL3_MTP_HEAD_N` unset | `EXL3_MTP_HEAD_N=65536` | target output token-identical; serial sampled RNG unchanged |
| V2 int8 mixer | `EXL3_HC_MIX_V2_INT8=0` | `EXL3_HC_MIX_V2_INT8=1` | numerics change; no served-fingerprint equality expected |

The host-I/O group intentionally tests the useful dependency combination. Add short pin-only and
device+embed-only smoke arms if attribution is needed; `EXL3_MTP_DEVICE_DRAFT=1` without
`EXL3_EMBED_GPU=1` must safely use the served host chain.

## 4. Correctness gates

For the host-I/O, batched-verify greedy, and pruned-head groups, run both canonical greedy checks in
all four arms, retain full token IDs, and require:

- c1 fingerprint `1474eee2f5945248`;
- 30k-context fingerprint `4a255910dee2d9c5`.

Also require OFF=ON=OFF2=ON2 token IDs. Run explicit early-EOS, stop-token, stop-string,
max-token, and banned-string-rewind cases; compare emitted IDs/text, final cache position, and MTP
accepted/rejected totals. These exercise the serial `receive_sample` authority retained after
batched sampling and the existing `EXL3_HOST_GAP_REWIND=1` path.

For the common sampled request (`temperature=0.6, top_p=0.95, top_k=20`, no penalties, no filters,
no logprob/top-token export), fixed-seed ON output is allowed to differ from OFF only in the batched
verify group. Run all 12 code and 12 prose prompts over a seed sweep and retain raw token IDs,
request settings, output lengths, and acceptance counts. Compare empirical token/output metrics with
paired bootstrap intervals; do not apply an equality fingerprint to this group because one Philox
seed supplies all q rows instead of q Python RNG draws.

Prove exact serial fallback with fixed seeds for each of these requests under batched verify OFF/ON:

- non-default repetition penalty;
- nonzero presence penalty;
- grammar/filter or device logit mask;
- returned probabilities or top tokens;
- forced token/token healing, if exposed by the unchanged runner.

Each fallback case must have identical full token IDs and auxiliary outputs. The regular
temperature/top-p/top-k stack is eligible only because TabbyAPI collapses it to one stateless fused
sampler step; adding any history-dependent leading step must not enter the batched path.

At c4/depth3, confirm four independent jobs and 16 target verification rows. The implementation
issues one q-row sampler call per eligible job (four calls), performs draft comparisons on device,
and shares one pinned readback/event wait; it does not merge different jobs' samplers or RNGs.

For the pruned head, record draft IDs and acceptance counts. Any full-head argmax outside the first
65,536 columns may become an earlier rejection; the target verifier must still produce the same
tokens. Confirm the installed effective width is 65,536 (a multiple of 128) and the cached EXL3
trellis prefix uses 4,096 complete 16-column tiles.

For int8, require ON and ON2 to repeat under identical seeds, but do not compare them to the served
fingerprints. Record first greedy divergence from OFF, acceptance by draft position, output checks,
and any quality/tool-eval results. Confirm the load-time VRAM delta separately from allocator
reserved memory; with the served `MIN_R=1` configuration the source-derived prediction is
655,148,512 bytes (0.610 GiB) for target plus MTP fp16 decode layouts replaced by int8+scales.
Also record the host-I/O group's embedding mirror placement and size: a new fp16/bf16 mirror of the
served 248,320×2,560 table is 1,271,398,400 bytes (1.184 GiB), unless the table is already resident
on the incoming IDs' device.

## 5. Performance matrix

For every arm in every group, run the operator's unchanged `fn_bench` runner for code and prose at
c1 and c4, 2,048 forced tokens, twice per cell. Its private CLI is not in this checkout, so no flags
are invented here. Retain aggregate and per-stream decode/wall tokens/s, TTFT, exact generated token
count, MTP proposed/accepted/rejected counts and per-position histogram, errors, clocks,
temperature, power, and peak allocated/reserved VRAM.

Run the 12-prompt probe per kind and concurrency:

```bash
python3 /srv/qwen5090/probes/multiprompt.py \
  --url http://127.0.0.1:8022/v1 --model "<served-model-id>" \
  --tag "<GROUP>-<OFF|ON|OFF2|ON2>" --tokens 2048 --conc 1 4 \
  --out /results/decode-kernels-r4/multiprompt.jsonl
```

Keep the model ID, prompt set, seeds, request body, and warmup unchanged. For the host-I/O group,
profile a short in-process window and retain the count/duration of draft-loop host gaps and
synchronous memcpy/synchronize calls; do not quote the fork's 8.4→3.1 ms as this box's result. For
the head group retain `exl3_gemm` head duration and bytes for all three drafts. For the int8 group
retain separate `gr_v2_dots_i8_kernel` and `gr_v2_up_i8_kernel` totals and compare with the served
~1.7 ms/step mixer baseline under matched workload.

## 6. Decision and rollback

Identity-preserving groups are NO-GO on any canonical mismatch, fallback mismatch, cache/state
error, bad stop handling, c4 cross-job contamination, build error, OOM, or reproducible performance
regression. The sampled batched path is NO-GO on any fallback leak or distribution/quality anomaly;
fixed-seed divergence alone is expected. Int8 is a separate numerics-changing decision: require its
kernel/reference tests, repeatability, VRAM delta, quality/acceptance gate, and paired performance
direction before promotion.

For performance, require ON-vs-OFF and ON2-vs-OFF2 direction to agree at c1 and c4, with the
12-prompt results supporting the single-prompt benchmarks and no thermal/clock or acceptance-rate
explanation. Report raw samples or confidence intervals; never promote from one prompt.

Rollback is the same candidate image with the affected round-4 selector(s) set to `0` or
`EXL3_MTP_HEAD_N` unset, followed by health check. Then restore the recorded daily image if needed.
