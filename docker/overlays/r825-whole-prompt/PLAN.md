# Cold prefill throughput, 30–90k new tokens — 2026-10-01

The largest likely lever is **routed-MoE prefill**, including E3 fat-expert GEMMs and its fused thin-expert tier. Budget **10–18% of cold engine-prefill time saved** for a successful combined kernel round, not a measured gain. The curve supports depth-independent work as the main limit; it does not identify which fixed component dominates. R824 measures that distinction before implementation.

This packet is private planning work. No SSH, GPU, Docker, git, deployment, serving changes, or public-repo writes were performed. The operator runs R824. The LIVE daily is launcher `8b644c17e60049a8069fc901b6f091fa`, image `tabbyapi:r823c-cachetail-inforward`, image ID `sha256:f5a3c35e2e47647f2200ff33ca822405036835f0729544fad11bad7c56c226b7`, promoted 2026-10-01 11:40 UTC. Restore that daily, never the R823c experiment's old `f3975d68` baseline.

## 1. Measured curve and what the timestamps mean

Inputs: the complete local `raw/2026-10-01-r823c-cache-reuse-DUO8ur/container-*.log` and corresponding `client-*.jsonl`. [curve.json](curve.json) includes input SHA-256 hashes, request identities, source line numbers, every selected completion delta, and excluded boundary flags. [fixtures/provenance.json](fixtures/provenance.json) identifies verbatim copied log/client samples. The parser accepts the actual Docker timestamp plus compact JSON format:

```text
2026-10-01T10:13:52.384568942Z [R823-cache] {"v":1,"run":"089e155cdef94a0bbd27b86bf50814dd","t":1790849632.3844974,"event":"prefill_forward","key":"r823/raw/T35/s0/-1/warm","conversation":"ef07f4d3e9a25755e7ae9ef4b49f2f1e","serial":4,"start":0,"end":2048,"pipeline":false,"prompt_prefill":true}
```

`generator/job.py` emits this event **after** target prefill, MTP draft prefill dispatch, KV-position/page updates, and before the subsequent stash calls. `start/end` are token positions, not a pair of wall timestamps. Adjacent same-job `t` differences are **completion-to-completion wall cadence**, including the preceding stash/commit and next forward, not isolated CUDA kernel duration. The first 0–2048 completion has no preceding same-job completion: its isolated duration cannot be recovered from these events. Do not subtract the previous request's completion or silently call allocation-to-completion GPU time. R824 supplies that missing measurement.

Selection: A1/A2 (prior daily policy on the same instrumented image) and D1/D2 (the promoted in-forward policy); `status=VALID`, `kind=seed`, ≥30k prompt tokens, cached tokens exactly zero, start position zero, contiguous completion chain ending at `prompt_tokens−1`. Exclude C2 concurrent seeds, edited turns even when cached zero, warm prefixes, decode/requeue forwards, partial chunks, and all `*-pre.log` snapshots. This leaves **32 solo cold requests, 872 full-chunk cadence deltas, 36 coarse restarts**. Group by file + run + key + serial; compare numeric JSON `t`, not rounded server TTFT. The raw directory's other arms are parsed and checked but do not estimate the promoted cold curve.

The table excludes a chunk starting at a 32,768-token checkpoint and chunks within the last 4,096 prompt tokens. These have pipeline fill/drain effects. Bins use the completed chunk's start position; lookahead means some cadence includes stage 0 of the following chunk.

| Chunk start depth | A1/A2 n / median ms | D1/D2 n / median ms |
|---|---:|---:|
| 0–8,192 (first measurable chunk starts 2,048) | 48 / 148.24 | 48 / 148.41 |
| 8,192–16,384 | 64 / 149.49 | 64 / 149.55 |
| 16,384–24,576 | 64 / 150.36 | 64 / 150.58 |
| 24,576–32,768 | 64 / 150.91 | 64 / 151.24 |
| 32,768–40,960, excluding restart | 48 / 152.26 | 48 / 152.36 |
| 40,960–49,152 | 48 / 152.61 | 48 / 153.22 |
| 49,152–57,344 | 30 / 152.97 | 30 / 153.60 |
| 57,344–65,536 | 20 / 154.35 | 20 / 155.64 |
| 65,536–73,728, excluding restart | 6 / 154.76 | 6 / 154.92 |
| 73,728–81,920 | 8 / 155.80 | 8 / 156.69 |
| 81,920–90,112, excluding tail | 2 / 156.62 | 2 / 163.18 |

The last bin is too sparse to fit a new attention regime: two observations per policy, from one T65 request per boot. At 74–82k, the D curve is **5.6% above** its early cadence, A **5.1% above**. A local linear reading is about 0.10 ms per 1k depth added to a 148 ms intercept. At 90k it suggests ~9 ms extra per chunk (~6% at the deepest chunk; ~3% integrated over a whole cold prompt). This is an estimate of depth-dependent excess, not the total attention share. Routing/content changes, checkpoint capture, and staging also vary along these prompts; these bins are correlated observations within a few requests, not hundreds of independent trials.

The coarse restart medians are **307.29 ms A / 305.80 ms D**, versus ~152–155 ms interior cadence: approximately **150 ms bubble** at each 32k restart. A 50k request pays one interior coarse restart, 67.6k and 88.7k two. Removing just those extra bubbles would save roughly **3.5%, 5.5%, 4.1%** of their A engine time, respectively. Initial fill, near-tail windows, MTP serialization and stage imbalance are additional costs, not all removable together.

Engine-prefill medians cross-check the stated ruler:

| Prompt tokens | A engine s / tok/s | D engine s / tok/s |
|---|---:|---:|
| 50,000 | 4.22 / 11,848 | 4.25 / 11,765 |
| 67,618 | 5.48 / 12,339 | 5.50 / 12,294 |
| 88,703 | 7.30 / 12,151 | 7.35 / 12,068 |

**Verdict:** long cold prefill is mostly a repeated fixed cost per 2,048-row chunk, plus window bubbles. Growing KV attention is a modest additional slope below 90k. The curve alone cannot distinguish MoE/dense GEMMs from GDN or host issue time. R804's kernel profile makes MoE the leading hypothesis. The nearly identical A/D curves support applying that hypothesis to the new daily; they do not substitute for its GPU profile.

## 2. Source and historical constraints

Read together: FINDINGS R801–R806, R803, R804, R809p; [thin-tier-PLAN.md](../thin-tier-PLAN.md); [EXPERIMENT-BACKLOG.md](../EXPERIMENT-BACKLOG.md) prefill items; [docs-GOTCHAS-flashnext.md](../docs-GOTCHAS-flashnext.md), especially §§19–22; S21 gr-mix-tiled and S22 fuse-gdn specs; the served extracted Python tree and `patches/exllamav3/r823c-cachetail-inforward`; native E3 sources under `patches/exllamav3/rebase-dev/r3/out/rebase-dev-r3/exllamav3/exllamav3_ext/quant/`.

* **R804's two MoE costs differ.** Bulk fixed part ~71–76 ms; ~51–55 ms of that is the fused `exl3_moe_kernel` thin tier. At 2,048 rows the measured thin tier is 46.4 ms; the **separate** E3 gate/up + down GEMMs cost 52.5 ms, plus metadata/gather/reduce/prep ~12.9 ms. These are both-card serial kernel sums from an older harness. Do not divide their sum directly by today's pipeline cadence: A(i+1) overlaps B(i).
* **Thin tier is decoder-issue-bound.** K2/K3 times are nearly equal despite different packed bytes. The all-expert byte floor is ~22 ms, but the demonstrated decoder rate suggests a practical ~32–35 ms floor, not 22. It already skips inactive experts. `THIN_ROWS=16` is expected to lose 3–8 ms by moving middle experts into padded E3; it is not a free tuning win. Wider groups or a coop rewrite change arithmetic order.
* **E3 mainloop is fixed MB=4 (64 rows), fp32 `FragC` accumulation.** Fat experts are segmented into 64-row tiles; partly filled tiles decode and issue padded MMA work. Its per-weight cost is ~2× the thin tier's in the prior plan. MB=1/2 shape dispatch and better staging are plausible larger levers; the required fat-count/padding distribution is not measured by R823 logs. `EXL3_GEMM_H_ACC` belongs to sibling GEMM paths; it does not turn this E3 `FragC` mainloop into fp16 accumulation. H_ACC's backlog estimate (0–5% E3 prefill) is a hypothesis requiring separate dispatch proof and fidelity gates.
* **Actual attention dispatch is important.** The served `modules/attention_fn/dispatch.py` prefers Triton paged/varlen kernels, with quant-aware `_fns_qc` for packed q8 KV. `attn_mode="flash_attn"` in Job is not proof an external FlashAttention kernel ran. R824 retains kernel names. Tune the actual `_paged_attn_prefill_kernel`, quant staging, split counts, BLOCK_M/N, warps/stages on sm120 first; replacing it with FlashAttention requires q8 KV compatibility, workspace accounting and measured evidence.
* **GDN does not attend over the whole KV history.** Its chunked FLA scan carries a fixed-size recurrent state. `_scan_sub_range` defaults to 2,048; source records that larger scans were slower and 1,024 was +38%. Its projections remain depth-independent GEMMs. S22 fuse-gdn is a **decode** fusion proposal; its guessed `op_3` harness never matched the served `op_2` call, and does not optimize the long-prefill FLA recurrence. Do not resurrect it as a cold-prefill win.
* **S21's tiled HC prefill is already ON** (`EXL3_GR_MIX_TILED=1`) after the rebase. The old ~2× isolated mixer speedup is not an unclaimed daily gain. R577's static table residency surprise and R788's pool cost warn against adding persistent per-layer workspaces.
* **Merge, async stash, encode-once/off-loop are already served** (R803/R808/R809p). Their past gains cannot be counted again. Tokenization is outside engine prefill; at ≤3 streams the R805 peer-gap exposure is also not today's cold-prefill GPU ceiling. R801's sub-1% decode argument concerned mostly reused 2–4k turns; today's ≥16k-new traffic criterion reopens the long-prefill question, not that old denominator.

The LS pipeline is deliberately narrow: exact 2,048-row chunks, two contiguous device stages, at most one A lookahead, independent host issue threads, peer consumer-stream copy, no TP/CPU-offload/requeued/exported-state paths. With one active job, a window ends at a coarse 32k checkpoint or near the last two chunks; with peers it is capped at **two chunks**. Both stage streams join before Job's draft/commit continuation because scratch and MTP may touch both devices. R823c tail captures let A(i+1) cross the added 4k tail checkpoints, but preserve the old coarse/near window ends. Crossing an uncaptured recurrent checkpoint would save the wrong state. Extending lookahead can race global fused-kernel locks/schedulers or MTP scratch.

The present eligibility guard adds allocator reserve to driver-free memory. GOTCHAS §21 shows why that sum can overstate usable headroom. R824 records driver-free/allocated/reserved and allocator retries separately; it does not alter the guard. CHUNK=4096 previously failed boot (R574 E3-DET scratch), and today `gen.max_chunk_size != 2048` **disables the LS pipeline**. A 4k knob-only sweep would compare larger **serial** chunks to the 2k pipelined daily, not simply halve fixed cost.

## 3. Ranked levers — percentages are cold engine time saved

The ranges are planning estimates for solo 30–90k cold prompts, not throughput improvements, real-traffic wall improvements, or additive promises. Throughput gain is `1/(1−saved)−1`; e.g. 10–18% time saved means 11–22% more tok/s. Real workload weighting must be recomputed from the ≥16k-new 61% prefill-seconds population; TTFT includes queue/frontend/decode-first-token costs.

| Rank | Lever | Estimated share exposed / expected time saved | Cost and risk |
|---|---|---|---|
| 1 | **Routed MoE: E3 fat first, fused thin second** | ~30–45% critical-path budget combined; **10–18% saved** if fat −30–45%, thin −25–35% survive overlap. Fat alone ~6–10%; thin rewrite ~4–7%; bitwise thin changes ~1–3%. | Multi-day CUDA round. Padding-sensitive MB dispatch, register/staging changes, dequant issue efficiency. Numerics gates for changed tiles/accumulation; maintain deterministic slot reduction, q8 KV pool and real headroom. First measure actual limiting card. |
| 2 | **Window bubbles / two-card balance / draft boundary serialization** | Coarse bubbles alone 3.5–5.5% at observed sizes; combined **5–10% saved**, conditional on stage idle/balance. | Days, scheduling/state correctness risk. Capture coarse checkpoints before permitting lookahead, isolate scratch, preserve MTP/state order. Rebalance only within measured memory budget. A 26/22 layer count is not a measured 26/22 work split. Already overlapped cards cannot yield another 2×. |
| 3 | **Attention backend/tile choice at depth** | Depth-dependent excess ~3% integrated to 90k; total attention could be ~10–20% of critical cadence (old profile, uncertain). **1–4% saved** expected; higher only if R824 finds a backend step change. | Small tile sweep to larger backend port. q8 staging, wide-index bounds, sm120 shared memory, partial buffers, retrieval/fidelity gates. No evidence yet for quadratic dominance below 90k. |
| 4 | **Larger chunks (4096) plus redesigned pipeline window geometry** | Naive half-thin-cost ceiling ~8–10%; **0–10% net**, potentially slower from lost overlap/OOM. Overlaps rank 1/2 and is not additional. | Native scratch + Python pipeline redesign; preserve checkpoint/reuse semantics. R574 failed boot, stock pipeline is hardcoded 2048. Do not spend R824 on this sweep. |
| 5 | **GDN chunked recurrence, excluding its dense projections** | Historical core kernel sum ~10 ms/serial forward, likely few % of critical cadence; **0–2% saved** expected. Dense projections belong in the GEMM budget. | Kernel work and recurrent-state fidelity; S22 decode fusion is unrelated. Escalate only if measured ≥17% critical share at all depths. |
| 6 | **Host gaps between forwards / PLE staging / metadata/readbacks** | **0–5% saved** expected, higher only if both-card idle is ≥15%. Distinguish host issue starvation from peer wait/checkpoint bubbles; old R804 host idle ≤14 ms was serial. | Mostly Python/staging work, relatively cheap. Avoid a sync added by the probe; never sum CUDA-event envelopes and call the result GPU busy. Ngram prefetch2 is already served; full table in RAM remains out of scope. |

These rankings intentionally keep fat and thin MoE separately measurable. R824 can pick thin rather than fat, or host issue rather than either; the current biggest-lever estimate remains conditional.

## 4. THE measurement: R824 cold pipeline critical-path census

One immutable daily image, one stack load, one solo cold **90,113-token** job shape (90,112 prefilled rows). Use the existing `profile_decode.py --prefill --one-process` implementation through a small process-local wrapper; three unprofiled cold repetitions plus two CPU+CUDA Kineto captures. Every repetition has a unique first-page token prefix and asserts cached=0. No prefix priming or warm-state resume at 32k/80k: both depths are reached within the same cold sequence. Exact raw R818 warm-up strings (20/29/11 tokens) and one long shapes warm-up occur before timing. No nsys/ncu, build, flags A/B or quality round.

The wrapper loads **max_batch_size=8, pool=901120, q8/q8, MTP depth/history=3, target split [30,30], draft split [0,32]**, enqueues one request and checks every target module's placement against the real D1 LS layout in `fixtures/daily-layout.json`. This fixes R804's accidental 0–33|34–47 harness partition; matching split flags alone was insufficient. The image import landing is checked before the lock. Image digest, LIVE launcher full MD5, resolved selectors, runtime source mounts, config, model identity, restart count and hardware controls are checked. The harness has no TabbyAPI frontend/vision tower or sysmem-KV tier; it measures text prefill, not total API TTFT or new capacity. Any placement mismatch aborts before inference, rather than silently accepting a different split.

Read full 2,048-row chunks at **0–2048, 32768–34816, 81920–83968**. Show initial/coarse fills explicitly. For a sustained comparison, also use the immediately following 2k/34k interior chunks, plus 80k (there is no 80k coarse boundary). Never label the restart bubble attention growth. Both captures retain **all** chunk ledgers so neighboring depths can be inspected.

CUDA event pairs wrap stage 0, stage module dispatch, MoE, Attention, GDN, HC/PLE, and E3 fat/thin/reduce entry points on the **actual current streams**. They are recorded from both the main and pipeline worker threads. There is no module-level `synchronize()`. Read events only after the profiler's existing final two-card join. Stage and leaf envelopes include host submission starvation; parent and child event durations are not summed. Kineto GPU intervals, including real PtoP `inDevice` schema, determine actual per-card busy, overlap, and both-card idle on a common timeline. Kernel names distinguish `e3_gateup_kernel` / `e3_down_det_kernel` from `exl3_moe_kernel`; dense/GDN/attention envelopes contextualize projections and unclassified kernel names.

**Pre-registered reading:**

1. Require both captures, all three requested pipeline stage depths, zero cached tokens in three timings and two captures, exact placement, both device kernel coverage, unchanged hardware controls, no VOID file, and capture X ≤1.15× unprofiled median X. Missing kernels/ranges or unmatched REAL schema fails closed. If profiler inflation exceeds 15%, retain CUDA envelopes and say INCONCLUSIVE; do not promote a guessed gain.
2. For each sustained band, report wall ms, per-card busy ms and family kernel ms, overlap ms, both-idle ms, module/stage envelopes and source/placement hashes. The busier card is a **screening proxy** for the limiting stage; confirm against stage envelopes before engineering. Time-window family sums can include A(i+1), which is the work actually competing with B(i), not a same-chunk attribution.
3. Score each family as critical-card kernel time / band wall × a declared local reduction: **fat 40%, thin 30%, attention 30%, GDN 30%**. Score host/pipeline as both-idle share ×50%. Average the three sustained depth bands. These are counterfactual screening scores, not measured gains. Pick a family only if it projects ≥5% cold time saved and exceeds second place by ≥2 percentage points in **both captures**. Otherwise TIE (inspect timelines) or STOP (no big enough candidate). The probe prints the result; do not choose a lever from an inclusive operator total.
4. **If MoE leads:** large fat share → MB/padding/staging round first; large thin share → revisit the decoder issue/barrier/occupancy route from thin-tier-PLAN, not inactive-expert compaction or THIN_ROWS=16 by assumption. A routed-MoE combined score can justify doing both, but an individual subfamily must identify the first implementation.
5. **If attention leads:** require actual attention kernel time increasing with depth, not only higher wall time; inspect actual Triton/q8 dispatch and the emitted kernel shapes before a FlashAttention port. **If GDN leads:** require large GDN module/core share at all three depths, then the chunked scan rather than S22 decode fusion. **If host/pipeline leads:** ≥10% projection corresponds to ≥20% both-idle; inspect checkpoint/draft/host gaps and stage imbalance. Do not confuse one-card idle during productive overlap with both-card idle.
6. Independently report coarse restart versus its next interior chunk and stage imbalance. The simple automated family rule is conservative about imbalance (a busy peer is not both-idle). If a stage envelope counterfactual yields ≥5% while family scores tie, choose a bounded scheduling follow-up rather than a kernel rewrite. No unconditional kernel GO from this packet.

Expected time: load/warm-up ~2–4 min; six ~8 s prefills and exports ~2–5 min; restore ~3–5 min, normally **10–15 GPU-exclusive minutes**. Probe hard timeout **900 s +30 s kill grace**; restored launcher hard timeout **600 s +30 s grace**; stop ~60 s and endpoint-stop polls ~90 s. The critical section budgets ≤30 min, excluding queue/drain wait. A failed restore exits nonzero with its log; it does not disguise a wrong daily as success. Queue-aware restoration skips intermediate daily boots and restores only when this unit is last.

## 5. Operator packet and offline verification

Files: `r824-prefill-profile.sh`, `r824_probe.py`, `test_r824.py`, `prompt.txt`, `fixtures/`, plus existing `probes/profile_decode.py` and `lib/{gpu-queue,serve-ctl,gateway-drain}.sh`. Deploy from these artifacts using the normal operator file-copy procedure. No transfer or run was attempted here. On flan:

```sh
sudo systemd-run --unit=r824-prefill-profile --collect \
  -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
  -p Environment=HOME=${HOME} \
  /usr/bin/bash /srv/qwen5090/prefill-throughput/r824-prefill-profile.sh
```

Stop with `systemctl stop r824-prefill-profile`: the trap invalidates the current capture, removes the named probe container, restores archived LIVE daily if last in queue, and restarts only clients previously running. Before flock, register in gpu-queue; before stopping, quiesce clients and gateway-drain/wait-idle. Recheck immutable identity after queueing. The GPU child closes inherited lock/drain descriptors. Restoration never rewrites the LIVE launcher file. Source landing/manifests are verified in a CPU-only container before the lock.

Result directory contains archived packet, LIVE launcher, preflight landing/self-tests, pre/post identities, probe log, source hashes/environment, exact placement, CUDA-event ledgers, unprofiled X, complete compressed Kineto traces, per-forward ledgers, `reading.json`/`reading.txt`, VOID reasons, cleanup and restore logs. Offline reanalysis:

```sh
python3 prefill-throughput/r824_probe.py analyze --root RESULT_DIR/gpu --out reading.json
R824_RAW=RAW_DIR python3 prefill-throughput/test_r824.py
python3 prefill-throughput/r824_probe.py curve --raw RAW_DIR --out curve.json
```

The offline tests use **verbatim real R823c logs/client JSONL**, a real R804 Kineto sample containing outer `phase/prefill_job`, CPU `phase/job_prefill` and duplicate GPU annotations, plus its actual peer-copy schema. Lifecycle functions run under shell mocks for successful restore, queued successor, pre-boot exit, queued signal and restore failure. The full raw run parses all **18 container logs / 53,079 trace events**, and reproduces the **32 / 872 / 36** selected curve counts. This is offline readiness; GPU execution and its reading remain pending the operator.
