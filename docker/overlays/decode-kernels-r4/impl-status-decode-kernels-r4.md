# Decode kernels round 4 — implementation status

## Outcome

The three fork patches are ported as a nine-file overlay against the captured served exllamav3
tree. Every new behavior is import/load-time opt-in and defaults off; no TabbyAPI file is changed.
The exact baseline and overlay hashes are recorded in
`out/decode-kernels-r4/overlay/manifest.json:1`, and the replayable patch is
`out/decode-kernels-r4/served-source.patch:1`. The six CPU structural tests and zero-fuzz patch
dry-run pass locally; PyTorch, nvcc, a checkpoint, and a GPU are unavailable here, so no numerical
or performance result is claimed (`out/decode-kernels-r4/local-validation.json:1`).

## 1. Pinned staging, batched verification, and device-resident MTP

### Pinned staging and reconciliation with upstream

`EXL3_DRAFT_PINNED_STAGING=1` pins the reusable draft input-ID and result-ID buffers instead of
adding a second copy of the upstream change (`overlay/exllamav3/generator/generator.py:248-264`).
One helper now owns distinct reusable block-table/cache-length buffers for generic AR, MTP, and
DFlash drafters; the served pageable `torch.zeros` branch is retained verbatim when the flag is off
(`overlay/exllamav3/generator/generator.py:672-688`). This is the reconciliation of fork 0003(a)
with `a3b249543`: there is one implementation, not stacked duplicate allocations.

Every per-step cache-length source gets its own immutable pinned buffer because the previous H2D
copy may still be pending when the CPU increments the logical lengths
(`overlay/exllamav3/generator/generator.py:691-697`). The helper is used by the generic autoregressive
draft loop, MTP loop, and DFlash path (`overlay/exllamav3/generator/generator.py:724-795`,
`:826-909`, `:964-999`). Accepted MTP target-state prefill also receives a private staged length
buffer and per-job embedding slot (`overlay/exllamav3/generator/generator.py:1583-1610`).

The served target/verification path was already pinned: block table, cache lengths, positions, and
the uniform-width concatenated input IDs use `_staging`, and target parameters already request
pinned embedding output (`overlay/exllamav3/generator/generator.py:1070-1081`, `:1101-1148`,
`:1170-1185`). Round 4 therefore does not duplicate that work; its flag covers the remaining draft
paths. CPU embedding output is copied into a reusable pinned buffer only when the generator marks the
call safe, with separate slots for back-to-back accepted-prefill jobs
(`overlay/exllamav3/modules/embedding.py:184-199`).

### Device-resident MTP chain

`EXL3_MTP_DEVICE_DRAFT=1` is independent, but the hot path engages only when
`EXL3_EMBED_GPU=1` can provide an exact device mirror, the state is CUDA-resident, dynamic draft
calibration is off, and the job has no indexed/multimodal embedding
(`overlay/exllamav3/generator/generator.py:838-880`). One initial ID upload feeds the first draft;
subsequent argmax IDs and the whole draft window remain on the head's device, followed by one
whole-window blocking readback before CPU job logic consumes it
(`overlay/exllamav3/generator/generator.py:881-925`). This naturally follows the served split: IDs
returned by the shared head stay on that head's CUDA device, and the embedding mirror is created on
the incoming IDs' device (`overlay/exllamav3/modules/embedding.py:164-178`).

The mirror is size-capped by `EXL3_EMBED_GPU_MAX_MB` (default 4096), is exact-copy rather than
requantized, and is released on unload (`overlay/exllamav3/modules/embedding.py:13-14`, `:64-76`,
`:164-176`). If either device flag is missing, the size cap is exceeded, IDs are not CUDA, dynamic
drafting is active, or vision/indexed embeddings are present, the served host chain runs. The mixed
vision branch itself is unchanged (`overlay/exllamav3/modules/embedding.py:100-160`).

If the served 248,320×2,560 embedding is CPU-resident, its fp16/bf16 mirror costs 1,271,398,400
bytes (1.184 GiB) on the IDs/head device; if the table already resides on that same device, no mirror
copy is made (`overlay/exllamav3/modules/embedding.py:53-62,164-178`;
`out/decode-kernels-r1/weight-streaming.tsv:26`). This is a larger placement cost than the int8
mixer saving and is an explicit box VRAM/OOM gate, not a free host optimization.

### Batched verify

`EXL3_BATCH_VERIFY=1` labels only a lone argmax or lone fused sampler as batchable. A regular
temperature/top-p/top-k stack is eligible because the served sampler collapses it to one stateless
`SS_Fused`; any leading penalty/ban/history step leaves a multi-step sampler and therefore has no
batch mode (`overlay/exllamav3/generator/sampler/custom.py:1129-1185`). TabbyAPI constructs
repetition/presence/frequency penalties before temperature/top-k/top-p and builds the same
`CustomSampler` without any round-4 request plumbing change
(`ref/served-src/tabbyapi/backends/exllamav3/sampler.py:45-126`, `:212-226`;
`ref/served-src/tabbyapi/backends/exllamav3/model.py:1364-1369`).

The eligibility guard rejects penalties/history, filters/grammar masks, forced IDs, device masks,
probability or top-token exports, negative/new-token special states, and multi-sequence jobs. Every
rejected case falls through to the exact served `job.receive_logits` path; nothing is approximated
(`overlay/exllamav3/generator/draft_overlap.py:46-68`;
`overlay/exllamav3/generator/generator.py:1422-1430`).

For each eligible MTP job, all q verify logits enter one sampler call. Draft comparison stays on
device, all eligible concurrent jobs copy tokens plus match bits to one pinned buffer, and one CUDA
event is synchronized (`overlay/exllamav3/generator/generator.py:1299-1354`). At c4/depth3 this is
four independent q=4 sampler calls (16 target rows), not one sampler shared across jobs; per-job
samplers and RNGs remain isolated, while their D2H completion is shared
(`overlay/exllamav3/generator/generator.py:1307-1354`).

Precomputation does not decide acceptance. The existing serial loop still calls `receive_sample`
position by position, stops immediately on requeue/EOS/stop, handles banned-string checkpoint
rewinds, advances filters/past IDs after accepted drafts, and performs cache/recurrent rejection
rollback (`overlay/exllamav3/generator/generator.py:1399-1526`). In particular, the host-gap rewind
signal suppresses the stale MTP carry exactly as before (`overlay/exllamav3/generator/generator.py:1467-1477`,
`:1577-1581`; `ref/served-src/exllamav3/generator/job.py:882-946`).

The served concurrency-indexed policy is untouched. It still selects a round-wide depth from the
decode-ready job count and reserves buffers/cache history for the maximum configured depth
(`overlay/exllamav3/generator/generator.py:196-228`, `:663-669`). TabbyAPI still validates and passes
`draft_num_tokens_by_batch` into `AsyncGenerator`
(`ref/served-src/tabbyapi/backends/exllamav3/model.py:268-275`, `:830-845`).

### Output/RNG contract by flag

| selector | greedy | sampled |
|---|---|---|
| `EXL3_DRAFT_PINNED_STAGING` | Token-identical; same RNG. Only source allocation/copy mode changes (`generator.py:248-264,672-704`). | Token- and RNG-stream-identical for the same reason. |
| `EXL3_MTP_DEVICE_DRAFT` | Token-identical; same RNG. The same greedy draft IDs are retained on device then read once (`generator.py:869-925`). | Token- and RNG-stream-identical; target sampling order is untouched. |
| `EXL3_EMBED_GPU` | Token-identical; same RNG. It gathers from an exact weight copy (`embedding.py:164-178`). | Token- and RNG-stream-identical. Alone it is normally dormant because served IDs are CPU. |
| `EXL3_BATCH_VERIFY` | Token-identical. Ignored greedy seeds are advanced only for positions actually consumed, preserving Python RNG state (`generator.py:1411-1425`). | For the eligible lone fused sampler, output is distribution-identical but RNG-stream-different: one seed supplies independent Philox row subsequences instead of q serial Python seeds (`custom.py:1179-1185`; `generator.py:1324-1342,1413-1421`). Fixed-seed token traces may differ. Ineligible settings are exact serial fallback. |
| `EXL3_MTP_HEAD_N` | Token-identical through full-head target verification; draft acceptance may change (`qwen4_exp_mtp.py:172-242`; `generator.py:1170-1186`). | Distribution- and serial RNG-stream-identical when batch verify is off; only proposal/round boundaries change. |
| `EXL3_HC_MIX_V2_INT8` | Not identical; weight quantization changes mixer numerics (`hyperconnections.py:325-337`). | Not identical and not the same distribution as fp16 in the strict numerical sense; quality/acceptance require a separate gate. |

The served fused kernel keys Philox by `(random, flat row * dim + token index)`, which is why q
rows under one seed retain the intended per-row categorical distribution while changing the serial
seed stream (`ref/served-src/exllamav3/exllamav3_ext/generator/sampling_fused.cu:722-726`).

The identity statements above are implementation contracts awaiting the box fingerprints; they are
not claimed GPU observations. Default-off executes the served allocation/sampling branches
(`overlay/exllamav3/generator/generator.py:34-38`,
`overlay/exllamav3/modules/embedding.py:13-14`).

## 2. Pruned 5-bit EXL3 draft head

`EXL3_MTP_HEAD_N` is unset by default. When set to a positive value below the full head width, only
`Qwen4ExpMTPModel.sample_from_state` uses a cached prefix; the target model's verification forward
still computes `batch_logits` through its full head (`overlay/exllamav3/architecture/qwen4_exp_mtp.py:17-18`,
`:172-215`; `overlay/exllamav3/generator/generator.py:1170-1186`). Therefore a draft outside the
prefix is an ordinary speculative rejection and cannot replace the full-head target token. Greedy
and serial sampled output are token-identical by construction; acceptance/window shape may change.
When stacked with sampled `EXL3_BATCH_VERIFY`, only that verifier flag changes the RNG stream.

The quantized slice is legal without reconstructing or requantizing weights. EXL3 stores its B
matrix as `(k/16, n/16, 16*K)` and requires n divisible by 128
(`ref/served-src/exllamav3/exllamav3_ext/quant/exl3_gemm.cu:23-36`). The port rounds down to 128,
slices trellis dimension 1 by `N/16`, and slices the matching output-scale vector
(`overlay/exllamav3/architecture/qwen4_exp_mtp.py:217-242`). For `N=65536`, that is 4,096 complete
EXL3 column tiles.

The r1 inventory measured a full 5-bit head payload of 397,312,000 bytes per draft invocation and
three-invocation head time of 828.275 us at c1/d3, 841.139 us at c4/d3, and 804.666 us at 30k c1/d3
(`out/decode-kernels-r1/weight-streaming.tsv:26`, `:42`, `:58`). A 65,536-column prefix is
104,857,600 bytes (26.3918% of full), so d3 removes 877,363,200 bytes (0.817 GiB) of streamed head
weights per step. A strictly bandwidth-proportional estimate is 609.7 us saved at c1, 619.1 us at
c4, and 592.3 us at 30k, leaving about 218.6/222.0/212.4 us of draft-head time. Those are derived
estimates, not round-4 measurements; the cached prefix also costs about 104.9 MB additional VRAM
because the full verification head remains resident.

## 3. V2 int8 mixer

`EXL3_HC_MIX_V2_INT8=1` is implemented specifically on top of served V2 and only engages when
`EXL3_HC_MIX_V2=1`. Load-time symmetric quantization stores one fp32 scale per folded fn row and one
per up output channel (`overlay/exllamav3/modules/hyperconnections.py:242-249`, `:293-337`). With the
served `EXL3_HC_MIX_V2_MIN_R=1`, fp16 `fn_h` and decode-layout `upx_h` are released; checkpoint-layout
`proj_h`/`up_h` remain for R>32 prefill (`overlay/exllamav3/modules/hyperconnections.py:339-344`,
`:410-445`). If `MIN_R>1`, the fp16 copies are deliberately retained for the smaller-row fallback,
so the quoted net VRAM saving does not apply.

The existing `upx_h` is not dead in served fp16 decode: both V1 and V2 fused kernels consume its
lane-contiguous `(H,D/4,rank,4)` layout (`overlay/exllamav3/modules/hyperconnections.py:318-321`,
`:424-432`). It can be freed only after another decode representation exists. This port replaces it
with `upx_q`; simply deleting it from the fp16 path would not be bit-identical or runnable. `up_h`
itself remains necessary for prefill (`overlay/exllamav3/modules/hyperconnections.py:433-445`).

The new V2 dots kernel keeps V2's row batching/reduction shape, loads eight int8 weights per vector,
and applies the per-row scale after reduction
(`overlay/exllamav3/exllamav3_ext/hc_mix.cu:954-1042`). The new up kernel keeps V2's tiled
warp-per-channel geometry and applies a per-channel scale after the rank reduction
(`overlay/exllamav3/exllamav3_ext/hc_mix.cu:1161-1239`). The host wrapper validates dtype, layout,
device, workspaces, and shared-memory bounds, dispatches the same B=1/2/4/8 row buckets, and uses
`C10_CUDA_CHECK` with the required CUDAException include for every new launch
(`overlay/exllamav3/exllamav3_ext/hc_mix.cu:1-8`, `:1437-1468`, `:1565-1632`). The symbol is declared
and bound at `overlay/exllamav3/exllamav3_ext/hc_mix.cuh:73-86` and
`overlay/exllamav3/exllamav3_ext/bindings.cpp:122`.

This flag changes numerics. The fp16 FMA weights are replaced by rounded int8 values, and factoring
the scale out of each contracted loop changes fp32 rounding as well; no token-equality promise is
made (`overlay/exllamav3/modules/hyperconnections.py:325-337`;
`overlay/exllamav3/exllamav3_ext/hc_mix.cu:954-1042,1161-1239`).

### Storage accounting

The served dimensions are hidden 2560, H=4, rank=320, 48 trunk blocks and one MTP block
(`ref/r464-r465/r464/run-123307/short/ctx1024_b1_d3/environment.json:579-584`, `:657-667`). Each
block creates two site mixers (`ref/served-src/exllamav3/architecture/qwen4_exp.py:86-110`), and the
target and MTP each add a combine-less final mixer
(`ref/served-src/exllamav3/architecture/qwen4_exp.py:282-302`;
`overlay/exllamav3/architecture/qwen4_exp_mtp.py:102-117`).

For one site, fp16 `fn_h+upx_h` is 13,189,120 bytes; int8 weights plus fp32 scales are 6,636,816
bytes, freeing 6,552,304. Exact totals from the same formulas are:

| scope | freed bytes | GiB |
|---|---:|---:|
| 96 trunk sites | 629,021,184 | 0.5858 |
| target: 96 sites + final | 635,532,544 | 0.5919 |
| target + MTP: 98 sites + 2 finals | 655,148,512 | 0.6102 |

The unchanged norm weights are excluded from both sides. The formulas and all three scopes are in
`out/decode-kernels-r4/mixer_int8_parity.py:66-82,116-130`, with the 96-site arithmetic also locked
by `out/decode-kernels-r4/tests/test_round4_cpu.py:75-83`. Thus the source-derived net confirms the
low end of the proposed 0.6–0.8 GB estimate, not 0.8 GB.

### Expected time, clearly not measured

The served r1 mixer total is 1,716.645 us/step at c1/d3, 2,985.450 us at c4/d3, and 1,721.816 us
at 30k c1/d3 (`out/decode-kernels-r1/groups.tsv:18`, `:30`, `:42`). Applying the fork's reported
V1 isolated 1.14–1.34x speedup only as a planning bracket gives roughly 1.28–1.51 ms for the served
1.7166 ms workload. A more optimistic traffic-only floor, halving the r1 dots/up time while leaving
state/apply fixed, is about 0.98 ms; int8 conversion instructions and V2 geometry make that a bound,
not a forecast (`out/decode-kernels-r1/inventory.tsv:80-83`). The supplied box test must establish
the real V2 time (`out/decode-kernels-r4/tests/test_gr_mix_v2_int8.py:70-116`).

The offline script defaults to all 96 real-dimension synthetic sites and reports max absolute,
max relative, and mean absolute error for mixed and post outputs versus fp16 weights, plus the exact
storage accounting (`out/decode-kernels-r4/mixer_int8_parity.py:14-17`, `:39-63`, `:85-135`). It
marks `checkpoint_tested=false` because no checkpoint is readable here (`mixer_int8_parity.py:130`).

## 4. Tests and packaging

Local tests compile every overlay Python file without importing torch, validate literal default-off
selectors, exercise every batched-verify fallback class, lock the EXL3 128/16 tile arithmetic, lock
the 96-site storage result, and require C10 launch checks
(`out/decode-kernels-r4/tests/test_round4_cpu.py:23-89`). `verify_local.py` additionally verifies
baseline/overlay/patch hashes and a zero-fuzz dry-run
(`out/decode-kernels-r4/verify_local.py:23-105`). All required local checks pass
(`out/decode-kernels-r4/local-validation.json:1-16`).

The box CUDA harness compares the new kernel to a reference using the same dequantized weights,
compares numerical drift to fp16 V2, covers site/final forms at rows 1/4/16, and records separate
CUDA-event medians (`out/decode-kernels-r4/tests/test_gr_mix_v2_int8.py:27-54`, `:82-116`). The build
first verifies installed served hashes, copies exactly the nine changed files, clears a dedicated
extension cache because CUDA changed, sets architecture 12.0, and imports once
(`out/decode-kernels-r4/overlay/Dockerfile.box:4-27`;
`out/decode-kernels-r4/overlay/verify_installed.py:1-29`;
`out/decode-kernels-r4/overlay/build_extension.py:1-20`).

The serving matrix, canonical fingerprints, sampled/fallback cases, c1/c4 fn_bench, 12-prompt
probe, acceptance accounting, VRAM gate, and OFF/ON/OFF2/ON2 order are specified in
`out/decode-kernels-r4/box-ab-spec.md:1`.

## 5. Unverified / operator work

- CUDA compilation and binding import are unverified locally because nvcc and torch are absent
  (`out/decode-kernels-r4/local-validation.json:6,13`).
- The 96-site synthetic parity script has not run here because torch is absent; no max-abs/max-rel
  numbers are fabricated (`out/decode-kernels-r4/local-validation.json:13-16`).
- Kernel parity, rows 1/4/16 timings, SASS/resource behavior, and both-card behavior await the box
  (`out/decode-kernels-r4/tests/test_gr_mix_v2_int8.py:70-116`).
- The checkpoint is not present, so actual module counts/VRAM deltas must be confirmed from the
  loaded model audit even though the served config/source arithmetic is exact
  (`out/decode-kernels-r4/mixer_int8_parity.py:116-130`).
- Device placement (expected cuda:1 for MTP/head), mirror allocation, pinned-copy engagement, one
  verify wait across c4, early stop/EOS/rewind behavior, and all identity/RNG contracts await the
  serving gates (`out/decode-kernels-r4/box-ab-spec.md:55-136`).
- No round-4 performance number is reported. The fork's host and V1-int8 figures and r1 timings are
  inputs to prioritization only; promotion requires the paired box matrix
  (`out/decode-kernels-r4/box-ab-spec.md:138-177`).
