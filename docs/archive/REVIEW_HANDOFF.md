# Repository review handoff and improvement plan

Prepared 2026-09-16 for the agent responsible for this repository.

Repository: this one

Reviewed revision: `53ee6dd` (`push-to-flan.sh: verified copies, because bash -n cannot tell an empty file from a valid one`). The initial review started at `fb8c73f`; the three commits through `53ee6dd` were also inspected. Recheck current source before applying findings: other work was landing during the review.

## Purpose and scope

The user requested a repository review, then this handoff and ideas for making the setup better and faster. The nine findings below are the original review feedback, with evidence and suggested acceptance checks. Later sections are proposed improvements, not measured speedups or instructions to enable every experiment.

The review made no code fixes, contacted no GPU server, and performed no deployment, image build, or GPU benchmark. It used local source inspection, temporary fixtures, mocked HTTP responses and mocked shell commands. Eight Python files passed AST parsing and 39 shell scripts passed `bash -n` at the reviewed revision. ShellCheck was unavailable. Temporary reproduction fixtures were removed; the scenarios below describe how to turn them into maintained tests. This handoff is the only repository artifact created by this task.

Read [CLAUDE.md](../../CLAUDE.md) first. In particular, this track is private, shares both GPUs with the box's other engine, and has a mirrored launcher in the private `kubernetes-home` repository. Carry forward the user's existing authorization in your session and follow that operating agreement for live work. Decide explicitly which service should be restored; an experiment must not silently change that decision.

Existing background should be read in place rather than reconstructed from this handoff:

- [Configuration](../../docs/CONFIG.md)
- [Measurements](../../bench/RESULTS.md)
- [Promotion assessment](../../docs/PROMOTION.md)
- [Operating runbook](../../docs/RUNBOOK.md)
- [Known measurement and operational traps](../../docs/GOTCHAS.md)

## Review findings

P1 means fix before relying on unattended deployment or experiment execution. P2 means a correctness problem in a measurement or validation path.

### F1 — P1: failed validation still completes deployment

Location: [r363-enable.sh:102](../../scripts/r363-enable.sh#L102), plus the workload checks at the end of the script.

A mismatching canonical fingerprint only logs `FAIL`. The replay prints parsed tool names and finish reason without asserting them, and retrieval output is filtered for the word `retrieved`, which also matches `0/5 retrieved`. The script then leaves the selected image serving, prints `DONE`, and returns success.

**Reproduced:** a fully local mock supplied a different fingerprint, no tool calls, `finish_reason=stop`, and 0/5 retrieval. The script printed all those failures and exited 0.

**Fix direction:** return explicit success/failure from each gate; aggregate required gates into a structured result; restore the validated configuration when promotion fails. Preserve the original failure if rollback also fails. Apply the same principle to the image/fingerprint checks in `r373-restore.sh`, which currently log mismatches without failing.

**Acceptance:** wrong fingerprint, invalid/missing tool call, retrieval miss, missing result, and interrupted capture each make promotion fail. Success requires every expected check from this attempt. A failure restores the intended service and reports whether recovery succeeded.

### F2 — P1: a historical PASS can approve a failed current run

Location: [r363-enable.sh:41](../../scripts/r363-enable.sh#L41).

The child `r362-pr337.sh` exit status is ignored. Eligibility is determined by searching its entire append-only audit log for `PASS baseline == pr337`. An old success therefore authorizes a newly failed, aborted, or missing measurement.

**Reproduced:** seeded the audit log with an earlier PASS; the current child appended FAIL and returned 1. The parent still set `PR337=1` and selected the combined PR #337 image.

**Fix direction:** consume a result belonging to this invocation, with a unique attempt ID, expected arm identities and explicit gate status. Make the child return a meaningful status too; changing only the parent's exit-code check is insufficient while the child itself logs failures and finishes successfully.

**Acceptance:** prior PASS + current FAIL must fail; prior FAIL + current PASS must pass; an empty child script or missing current result must never pass.

### F3 — P1: failed experiment boot skips restoration

Location: [r364-hotvocab.sh:121](../../scripts/r364-hotvocab.sh#L121); restoration is later in the normal success path.

The launcher removes the currently serving container before booting the candidate. If the treatment boot fails, this runner exits before reaching the live-launcher restore. Its TERM trap also only logs and exits. Several other experiment runners have the same lifecycle pattern.

**Reproduced:** a local launcher mock changed the server state to DOWN and failed the treatment boot. The runner returned 1, left DOWN as the final state, and never reached restoration.

**Fix direction:** centralize experiment ownership and recovery. Record the intended restoration target, install cleanup before the first disruptive action, and restore on error, timeout and termination while still holding the GPU lock. Coordinate with the existing queue helper's EXIT trap instead of overwriting it accidentally. Plan for process-group timeout behavior and ensure the restoration work has time to finish.

**Acceptance:** fail each stage after teardown; send TERM; exercise nested runners; verify restoration occurs once, before lock release, and that failed recovery is visible to the caller. Do not rely on `set -e` alone to encode this lifecycle.

### F4 — P1: direct launches bypass the GPU lock

Location: [launch-flashnext.sh:247](../../scripts/launch-flashnext.sh#L247).

The documented direct invocation writes shared configuration and removes serving containers without acquiring the GPU-exclusive lock. A benchmark holding that lock is protected from other cooperating runners but not from this launcher.

**Evidence:** source inspection; the launcher has no `gpu_lock`, `flock`, or queue-helper invocation. No live race was induced.

**Fix direction:** serialize all mutations, including config writes and STOP, through the same lock. Use the existing inherited-descriptor-aware helper rather than reopening the lock in a child and reintroducing the deadlock documented in GOTCHAS. Also check direct restore calls made by chain scripts.

**Acceptance:** an independent launcher waits behind an active experiment; a child launcher invoked by the lock holder proceeds without deadlock; no shared config or container changes occur before acquisition. Run the lock test on Linux or an isolated Linux environment because the helper uses `/proc`.

### F5 — P2: single-frame responses lose their benchmark record

Location: [probe.py:153](../../bench/probe.py#L153).

A response with multiple tokens in one text-bearing frame sets `t_first == t_last`. The decode-rate division then raises `ZeroDivisionError` outside the request exception handler. In a worker thread, that request never reaches `sink.append`.

**Reproduced:** one SSE message frame containing text and `usage.completion_tokens=16` produced `ZeroDivisionError` and zero records.

**Fix direction:** preserve successful response/accounting evidence even when steady-state decode is unmeasurable; use null for an absent or zero decode window. Ensure the summary handles null rates mixed with valid rates. Use a monotonic clock for elapsed durations. Every scheduled request, including a worker failure, must produce exactly one terminal record.

**Acceptance:** one frame/many tokens, one token, multiple frames, failure before text, and failure after partial text each produce one record. Zero-window cases have no decode-rate estimate.

### F6 — P2: explicit null usage crashes greedy capture

Location: [hotvocab-greedy-capture.py:44](../../bench/hotvocab/hotvocab-greedy-capture.py#L44).

`data.get("usage", {})` handles a missing key but returns `None` for `"usage": null`. The chained `.get` raises after writing the first prompt's text, leaving a partial output directory and skipping the remaining prompts.

**Reproduced:** a nonempty first response with null usage raised `AttributeError` and left only `0.content.utf8`.

**Fix direction:** normalize optional usage to an object; separately distinguish capture completeness from availability of token accounting. Do not invent counts when usage is unavailable.

**Acceptance:** missing usage and null usage both permit all four prompts to be captured; invalid response structure or empty required output produces a failed-capture result with no success marker.

### F7 — P2: reruns can validate stale captures

Location: [r366-ourkernel.sh:58](../../scripts/r366-ourkernel.sh#L58); analogous fixed-directory captures appear in other runners.

The capture tool uses `mkdir(..., exist_ok=False)`, but the runner reuses fixed output paths and only logs capture failures. A rerun therefore can compare old files against newly collected performance data. This is especially relevant to wrapper scripts explicitly intended to rerun experiments.

**Reproduced component behavior:** rerunning capture against an existing directory raised `FileExistsError` before making any HTTP request. The runner's source then continues into its comparison.

**Fix direction:** create immutable attempt directories. Require a complete capture manifest tied to the current attempt and image/config identity before comparison. Preserve old attempts rather than clearing historical evidence in place.

**Acceptance:** populate old matching captures, make current capture fail, and verify the current gate fails. A fresh successful attempt must compare only its own complete expected prompt set.

### F8 — P2: distinct suffixes still share cached prefixes

Location: [probe.py:72](../../bench/probe.py#L72), including the `--distinct` CLI description.

Appending a suffix does not prevent reuse of the long shared prefix. Tests using that mode cannot establish independent context memory requirements.

**Reproduced:** two `--distinct` prompts with the 30k requested code filler shared 126,264 of 126,313 characters, or 99.96%, at the start.

**Fix direction:** describe this mode as shared-prefix fan-out; use the existing `--unique` mode when testing unrelated contexts, and calibrate actual token counts after randomization. Update the remaining mislabeled admission tests, including the slot ladder. Existing documentation already corrects some historical rows; do not relabel every shared-prefix result as invalid.

**Acceptance:** independent-context tests diverge near the beginning and record actual server prompt/cached-token counts. Shared-prefix tests retain their intended common prefix and are labeled accordingly.

### F9 — P2: summary reintroduces the samples it excluded

Location: [summarize.py:32](../../bench/summarize.py#L32).

The `... or grp` fallback restores all early-stopped samples when no length-finished samples remain. Missing finish reasons are also accepted. A partially valid batch can furthermore get an aggregate calculated from only some requests while retaining the whole batch wall time.

**Reproduced:** a fixture generating 8 of 4,096 requested tokens with `finish_reason=stop` still printed 200 decode tokens/s and 8 aggregate tokens/s.

**Fix direction:** define throughput eligibility explicitly: authoritative token count, expected forced length, a valid completed response, and `finish_reason=length`. Keep failed/short requests visible but separate from throughput samples. Require a complete valid round for a comparable round aggregate; report partial work separately. Never replace missing token counts with SSE frame counts for token-throughput claims.

**Acceptance:** all-short, missing-finish, partial-round, duplicate-attempt, missing-usage and valid-round fixtures produce distinct, truthful outcomes. No valid samples means no throughput estimate.

### Additional source-level follow-up: empty capture directories can compare equal

While preparing improvement pointers, I inspected the sibling `greedy-compare.sh` (in the private infrastructure repository). Its comments say empty directories never pass, but `greedy_hash` hashes an empty file listing into the nonempty SHA-256 prefix `e3b0c44298fc1c14`; `greedy_same` only checks that the resulting hash is nonempty and equal. Two existing empty directories therefore satisfy the predicate on the intended GNU/Linux tools. This is a source-level finding, not a live-host test, and was not included in the original nine.

Fix this shared helper alongside F7. Check expected prompt IDs, successful capture completion and usable text, not just directory existence or a hash. Matching partial captures must also fail. Preserve its source provenance when updating the copy used on flan.

## Reliability improvements that make optimization trustworthy

These are design recommendations, not additional claims that every historical measurement is wrong.

1. **One experiment lifecycle implementation.** Put locking, preflight, candidate boot, health/inference checks, recovery and final status in one shared implementation. Keep individual experiment definitions small. Explicitly distinguish a standalone run from a child already holding the lock. Define how queue-marker cleanup composes with recovery.
2. **One immutable manifest per attempt.** Record repository commit, script hashes, image ID/digest, engine versions and patch hashes, checkpoint/tokenizer/template identity, effective config, sampler, endpoint, actual prompt tokens, cache state, repetitions, attempted/completed/valid request counts, and structured gate results. Hash normalized effective config separately from commented YAML bytes.
3. **One response parser and record schema.** Share handling of `content`, `reasoning_content`, `reasoning`, completion `text`, tool calls, usage, terminal markers and protocol errors. Keep raw evidence for failures. Record one outcome per request and make invalid/missing accounting explicit.
4. **Fresh, complete captures.** A manifest should identify all expected prompts and contain a completion marker written only after success. Equality must compare both completeness and content. Preserve reasoning/content channel distinctions rather than depending on arbitrary directory contents alone.
5. **A small offline regression suite.** Turn F1–F9 into behavior tests using fake responses, temporary result directories and mocked process actions. Include failure propagation through actual parent/child scripts. Run Python checks and ShellCheck in CI; use Linux for lock and signal tests. No GPU is needed for most of this coverage.
6. **Generate the current config documentation.** `docs/CONFIG.md` still lists the older image and an empty draft-policy default while the live launcher defaults have moved. Generate or check a concise served-config table from the same source used by the launcher; retain historical conditions alongside historical results.
7. **Keep operations diagnosable.** Emit a compact status containing image/config identity, last successful inference, active/queued requests, oldest queue age, cache/recurrent-state pressure, restart reason and last recovery outcome. Treat a health endpoint and a successful warmup inference as separate checks.

## Performance ideas, in recommended order

The recorded bottleneck is weak concurrency scaling with a two-GPU layer split. That suggests optimizing queueing, repeated prompt work and the actual CPU/GPU critical path before another speculative kernel rewrite. The following are experiments with falsifiers, not promised percentage gains.

### 1. Try bounded admission before increasing slot count

**Why:** the recorded slot ladder has roughly flat aggregate throughput from offered concurrency 4 through 16, with much worse TTFT at 16. More offered work is adding waiting time. This is evidence for testing admission control, not proof that c4 is optimal for every workload.

**Experiment:** keep the current engine's eight slots and compare client-side active-request limits of 2, 4 and 8 under the same incoming workload. Queue excess work outside the engine. Include short tool turns mixed with long prefills and long generations; test cancellation and a fairness policy so long requests do not starve.

**Measure:** latency from original client submission, including the external queue; p50/p95 TTFT and completion time; inter-token stalls; completed useful tasks/minute; throughput and errors. A lower server-side TTFT obtained only by moving the queue is not an improvement.

**Success:** equal or better completed-task throughput with better end-to-end latency/fairness. Do this before reducing `max_batch_size`, which changes allocations and introduces another variable.

### 2. Preserve and measure prefix-cache reuse in real agent requests

**Why:** the repository records a 152,761-token prompt taking about 24 s to first token cold versus 0.43 s on a repeat, in its documented depth experiment. That is an existing measured opportunity in the prefill path, not a forecast for all requests.

**Pointers:** preserve stable system instructions and tool-schema ordering; avoid inserting timestamps or random identifiers ahead of stable text; append changing material after the invariant prefix where semantics allow. Measure the actual repeated prefix on real tool turns. Keep compaction deliberate: rewriting the whole conversation every turn can trade prompt length for cache misses.

For this hybrid model, verify usable recurrent-state checkpoints as well as KV hits. Track prompt tokens reused, not just the existence of a prefix-cache setting. Prefix caching reduces repeated prefill work; it does not directly accelerate generation of new tokens. [vLLM's APC documentation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) explains that distinction; its flags are not TabbyAPI configuration instructions.

### 3. Use the existing profiler, but first match the served configuration

Read `the profiler how-to` (in the private infrastructure repository) and `profile_decode.py` (in the private infrastructure repository). These already exist in the private sibling repository. Their own validation statement is CPU/static only; no successful GPU profile is assumed by this handoff.

**Important mismatch:** the harness defaults to FP16 KV and automatic placement, sizes cache slots to the offered batch, and constructs the generator with a fixed draft depth. It does not automatically reproduce the launcher's YAML or concurrency-indexed policy. A trace of those defaults must not be labeled a profile of the served setup.

Before using it for that claim, explicitly match or record deviations from:

- KV `8,8`, manual budgets `[30,30]`, cache 262,144 tokens and eight resident slots;
- MTP mode, draft-cache format, resident draft weights and effective `[[2,3],[8,1]]` policy;
- chunk size 2,048, relevant engine environment, image/patch identity and module placement;
- the code/prose/tool workload and actual context lengths.

Expose resident slot count independently from offered batch in the harness if necessary. Use fitting contexts for full-batch kernel comparisons: four independent 152k contexts cannot simply be forced resident in a 262k pool. Measure queueing/eviction through the serving endpoint separately. Verify that chosen deep contexts actually cross the checkpoint's QSA sparse threshold.

Start with short c1/c4/c8 shapes and one fitting long-context shape. Inspect CPU launch gaps, host/device and inter-device copies, synchronization, per-device work, target-verification windows and draft acceptance. Use a timeline for critical-path attribution: summed kernel time or sampled GPU utilization alone does not prove which work limits wall time.

PyTorch supports combined CPU/CUDA profiling and scheduled capture; stack tracing and initial profiling have overhead. Keep profiler runs separate from the unprofiled throughput measurement used to claim a gain. [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

### 4. Make MTP tuning workload-specific and measurable

The current concurrency-indexed policy already has measured value. Keep it as the reference. Compare supported fixed depths and a small set of explicit policies using the same served base, resident resources and prompts. If testing no-draft, distinguish disabling speculative work from unloading the draft model, which can change placement and memory availability.

Record accepted tokens per verification, actual verification-window histograms, draft/verification work where measurable, wall throughput, TTFT and completion latency. Include prose and tool arguments, not just predictable code. An adaptive policy based on acceptance is a follow-up only if the fixed-policy results show a stable opportunity; the previous dynamic-draft setting regressed on recorded tests.

Greedy output identity is a useful gate. Changes that alter sampled behavior additionally need the matched tool/agent quality checks, not just the greedy fingerprint.

### 5. Investigate mixed prefill/decode interference

After establishing a clean baseline, test supported chunk-size values around the current 2,048 setting with simultaneous long-prefill and active-decode requests. First inspect the exact backend: this repository documents chunking as affecting the output requeue path as well as prefill, so changing it is not automatically a one-dimensional prefill experiment.

Measure decode stalls and p95 time to useful output, not only total prefill throughput. Smaller chunks might improve responsiveness while adding scheduling overhead; larger chunks may do the reverse. Validate token accounting across several requeue boundaries and confirm memory fit for every arm.

### 6. Let traces choose CPU, transfer or CUDA-graph work

If the timeline shows CPU starvation, isolate model-serving CPU resources from builds/scoring jobs and examine sampling, logging, tokenization and synchronization costs. The repository already records a native build degrading a soak; enforce exclusion rather than only printing that nothing else is running.

If kernel-launch gaps dominate, examine additional graph coverage or fewer graph boundaries. If copies or synchronization dominate, improve that path first. CUDA graphs mainly address CPU launch overhead and can fail to help when the expensive work lies elsewhere. [NVIDIA's CUDA-graph performance guidance](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html).

For cross-GPU copies, measure actual topology, negotiated links, transfer direction/size and synchronization before changing placement or recommending hardware. Rebalancing memory budgets is not automatically a speedup when the GPUs execute sequentially.

### 7. Optimize useful work, not only tokens/second

Replay representative coding sessions and track successful tool turns, total task completion time, retries, output tokens, failures and cache reuse. Investigate avoidable repeated reasoning, oversized tool logs and repeated prompt reconstruction. Any trimming or budget change needs a quality check; reducing output length can improve completion time while reducing task success.

For a quality comparison, add a held-out subset chosen independently of either model's observed outcomes and run matched harness settings. The existing outcome-stratified SWE subsets are useful diagnostics, but a simple fair-sign probability such as `2^-19` is not justified by symmetry after selecting on a prior run's failures. Preserve their descriptive tallies and test broader claims on an independent selection. Likewise, zero changed outcomes on 19 repeated instances is limited evidence, not a general equivalence guarantee.

### 8. Treat multi-GPU parallelism as an engineering project

Read `the EP status assessment` (in the private infrastructure repository) and recheck the current QSA/PLE follow-up work before planning. The documented prerequisites include QSA transport, PLE transport, replica/output-selection policy and MTP adapters. Their current status must be reconciled across worktrees; a compiled prerequisite patch is not a working EP engine.

A staged route is transport/state tests, a small trunk-only GPU correctness milestone, real memory/communication measurements, then MTP integration and quality/performance evaluation. Another research direction is overlapping independent microbatches across layer-split stages, if the engine can support safe cache/state ownership and asynchronous scheduling. Neither is a launcher switch or a promised twofold speedup. Communication over the actual PCIe topology may limit the result.

The box's other engine can remain a workload-specific alternative, but both complete stacks require both GPUs. Immediate per-request routing between simultaneously resident full stacks is not available on the documented setup. A simultaneous router would require a separately proven fitting smaller model or other hardware.

## Avoid repeating low-value experiments without new evidence

| Already recorded | Consequence for the next plan |
| --- | --- |
| Slots 12 and 16 fail to boot at the current cache/split | Do not retry unchanged. Any smaller-cache tradeoff must preserve an explicitly agreed context/service requirement. |
| Host KV tier 4,096 MiB did not improve the tested cases | Revisit only with demonstrated eviction/recomputation in a different workload. |
| The 32-row MoE envelope was output-identical and flat in measured performance | Require a trace showing useful time in the affected path before more dispatch work. |
| #246 changed outputs with small/noisy latency improvements | Do not promote based on a single favorable TTFT sample. |
| c8 TTFT has large within-configuration variation | Use repeated, interleaved comparisons. A large min/max range is not itself a statistical threshold below which all effects are impossible to detect. |
| Existing kernel caches and warmup already matter | Preserve them and verify representative shapes are warm; do not claim adding already-enabled caching as a new optimization. |

These are references to the repository's recorded experiments, not measurements independently repeated during this review.

## Suggested implementation sequence and completion evidence

1. **Repair deployment and recovery (F1–F4).** Use a common lifecycle and current-attempt results. Add offline failure/lock tests. Verify the mirrored source and host artifact identity through the existing operating workflow.
2. **Repair instruments and capture validity (F5–F9 plus the shared hash helper).** Add protocol fixtures and immutable attempt manifests. Regenerate affected summaries from identifiable raw attempts; do not fabricate missing historical records or claim that all prior results are invalid.
3. **Run one controlled workload experiment.** Admission limits 2/4/8 are a useful first candidate. Keep arrivals, prompts, sampler and server constant; count external waiting time. Select the next optimization from a served-matching profile.

For each performance A/B, record the exact baseline/candidate identities and raw per-request data. Interleave or randomize arm order and keep cold/warm states explicit. Treat requests within a concurrent round as correlated; compare independent rounds/blocks and report uncertainty. Choose a practical minimum improvement before examining results. Expand repetition when uncertainty could change the decision; confirm any win with unprofiled end-to-end measurements and quality gates.

The handoff is complete when the responsible agent can report which findings were fixed or superseded, the tests that establish each outcome, the actual restored/served identity after authorized live work, and measured before/after results for any chosen optimization. No speedup estimate in this document should be treated as a promised result.

Optional next-session skills: `diagnose` for reproducing failures and bottlenecks, and `tdd` for durable behavior tests. Read their setup requirements before using them; this handoff did not initialize or change the repository's issue workflow.
