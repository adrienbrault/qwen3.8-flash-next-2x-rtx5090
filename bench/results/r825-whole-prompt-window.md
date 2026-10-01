# R824–R825p: resumable whole-prompt windows reduce solo cold prefill time and bound event-loop blocking

Results on flan, 2026-10-01: [`2026-10-01-r824-prefill-profile-A3h6iN`](2026-10-01-r824-prefill-profile-A3h6iN), [`2026-10-01-r825-whole-prompt-YhItMe`](2026-10-01-r825-whole-prompt-YhItMe), [`2026-10-01-r825b-resumable-DyjHvu`](2026-10-01-r825b-resumable-DyjHvu), [`2026-10-01-r825c-resumable-Rh4RGH`](2026-10-01-r825c-resumable-Rh4RGH), and [`2026-10-01-r825p-promote-wholeprompt-yj7qj3`](2026-10-01-r825p-promote-wholeprompt-yj7qj3); drivers [`r824-prefill-profile.sh`](../../scripts/r824-prefill-profile.sh), [`r825-whole-prompt.sh`](../../scripts/r825-whole-prompt.sh), [`r825b-resumable.sh`](../../scripts/r825b-resumable.sh), [`r825c-resumable.sh`](../../scripts/r825c-resumable.sh), [`r825p-promote-wholeprompt.sh`](../../scripts/r825p-promote-wholeprompt.sh); raw reading, decision, equality, results, frontend comparison and request records are in those directories, with each audit copied as `audit.txt`.

The promotion completed at 15:37 UTC on 2026-10-01. The served image is `tabbyapi:r825c-hostprepare`, operator-built ID `aa04a1cbe94b`, source launcher MD5 `262e9c31`; rollback is `tabbyapi:r823c-cachetail-inforward`, source launcher `8b644c17`. The public launcher preserves its predecessor's public path adaptations and carries the same runtime delta: image, literal image-ID guard, `EXL3_PREFILL_WHOLE_PROMPT=1` and `EXL3_PREFILL_RESUMABLE=1`. It keeps the checkpoint policy, trace on, NVMe off, pool, split, sampler and FASTWARM.

## R824 census and corrected attention classification

R824 ran from 12:06 to 12:21 UTC on the R823c daily configuration. The solo cold prompt has 90,113 input tokens, zero cached tokens and 2,048-row prefill chunks. The profile harness uses greedy sampling with a 64-token output budget. CUDA events and two Kineto captures cover early, middle and late chunks; only extracted kernel/forward records and reading JSON are published here. Three timed repetitions read 7,289.646 / 7,339.927 / 7,346.393 ms engine prefill, median 7,339.927 ms, approximately 12,277 processed rows/s. The two captures inflate elapsed time by 1.015× and 1.013×. Core offsets were zero, memory offsets +4,500 on both cards, power limits 600 / 575 W; allocation retry deltas were zero. These conditions and records are in `2026-10-01-r824-prefill-profile-A3h6iN`.

The original classifier assigned approximately 26 % of the busier card's band to attention, 24 % to dense/other, 19 % to fat MoE, 15 % to thin MoE and 3.5 % to GDN. Its `attn|qsa|flash|fmha|paged|lse` expression also matched `lse` inside `false` in generic kernel templates, while excluding DSA indexer kernels. The original automated TIE remains in the raw reading; it describes those original buckets. Corrected attention durations on card 0 are 21.412 / 23.024 / 24.991 ms at band starts 2,048 / 34,816 / 81,920, and 22.600 / 27.591 / 29.777 ms on card 1. These are clipped band kernel sums including lookahead overlap, not isolated same-chunk module times. Generic projections remain dense unless launch attribution identifies their owner; worker-thread module ranges are absent for card 0.

The corrected attention screen projects 4.532 / 4.563 % saving in the two captures. The fat-MoE screen exceeds the thin runner-up by 3.004 / 2.992 percentage points and clears the registered family rule conditionally. It establishes a follow-up candidate, not a measured kernel gain. The scheduling counterfactual is independently supported by all six cold forward chains: exactly four cadence deltas exceed 250 ms, at starts 32,768, 65,536, 86,016 and 88,064. Their durations are approximately 290–308 ms against per-run steady medians of 149–152 ms, exposing 588–600 ms per prompt. One initial fill and final drain remain necessary.

## Four cuts and the R825 design

The predecessor cuts at coarse saves, at near-end saves, at the last full page before the leftover, and at the final-merge eligibility boundary. Coarse saves at 32,768 and 65,536 restart the pipeline. Near-end saves reduce the last two forwards to one-chunk windows, which the old planner sends through the serial path. The planner rejects partial chunks and clamps to the last full page. The merge path rejects pipeline-attached jobs and assumes the logical position and reservation position coincide; stage-A lookahead violates that assumption unless the original split descriptor is retained.

R825 plans every eligible solo prompt chunk, including the existing merged final chunk, into one window. It reserves capture descriptors before current/lookahead issue, captures coarse, tail and last-page states inside the existing layer forwards, and publishes only after matching page commits. The existing two pinned staging slabs are reused; current plus lookahead remain the activation bound. It changes no target arithmetic, draft forward, KV row, checkpoint position or stash layout. The two target stages join before each MTP draft prefill. Existing peers keep the two-chunk cap and all prior eligibility exclusions. Slab exhaustion closes before the checkpoint and falls back coherently; a timed B fallback fails the whole-window performance gate.

R825 ran from 12:57 to 13:07 UTC, results `2026-10-01-r825-whole-prompt-YhItMe`, direct engine, one stream, greedy one output token for timing and 32 for equality, zero cached tokens. Three in-process ABBA blocks per prompt size give six repetitions per arm, identical IDs within each block, cache resets and common shape/fast-state warmups. A disables the whole-prompt selector on the candidate stack; B enables it. Both are checked against a separately loaded actual R823c daily reference. Engine time is `time_first_token - time_first_prefill`, excluding HTTP tokenization and network time; rates below use N−1 processed prompt rows.

| Prompt tokens | A engine ms | B engine ms | A → B processed rows/s | Engine time saved |
| ---: | ---: | ---: | ---: | ---: |
| 20,000 | 1,853.8 | 1,596.5 | 10,788 → 12,527 | 13.88 % |
| 50,000 | 4,204.5 | 3,841.3 | 11,892 → 13,016 | 8.64 % |
| 90,000 | 7,422.5 | 6,855.6 | 12,125 → 13,128 | 7.64 % |

All 27 equality rows pass: T35 with its cached-prefix offset, T32, T65, aligned 20,000 / 50,000 / 90,000 and ragged 20,001 / 50,001 / 90,113-token cases, each in A, B and forced NO_SLAB. The probe compares checkpoint position sets, every saved stash's scalars and tensors, complete final-prompt and first-verification logits, and 32 greedy output IDs, including tensor byte views. Candidate tensors are not a second replayable tensor archive; the equality JSON records executed comparisons. Every timed B request has one window, final pipeline engagement, no serial prompt rows and no fallback. Added persistent GPU workspace and pinned-slab capacity are zero by source structure; post-request free-memory samples do not establish transient serving headroom.

The first R825 try aborted before GPU work because the probe required `EXL3_PREFILL_NOSYNC=1`; the daily leaves it unset, effectively zero. That try supplies no measurement. The successful probe checks the daily's actual setting.

## R825 could not serve synchronously; R825b resumes between chunks

`AsyncGenerator._run_iteration` calls `generator.iterate()` synchronously on TabbyAPI's event loop. An uninterrupted whole-prompt window would hold that loop for the prompt, preventing arrivals, health handlers and existing streams from progressing. R825's direct-engine equality and throughput verdict therefore did not authorize serving that synchronous path.

R825b ran from 14:00 to 14:16 UTC, results `2026-10-01-r825b-resumable-DyjHvu`, with the same cold solo ABBA, greedy one-token timing and 32-token equality conditions. A resumable window advances one chunk per iterate and retains completed stage-A lookahead, events, captures and slab ownership. A pending arrival consumes the retained lookahead and closes at the next boundary before admission. Cancel, reset and shutdown drain before release. A positive `asyncio.sleep(0.000001)` follows a resumable chunk.

| Prompt tokens | A engine ms | B engine ms | A → B processed rows/s | Saved | A max heartbeat gap ms | B max heartbeat gap ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 1,867.294 | 1,598.828 | 10,710 → 12,509 | 14.377 % | 1,923.538 | 281.460 |
| 50,000 | 4,202.943 | 3,840.122 | 11,896 → 13,020 | 8.633 % | 4,056.823 | 282.097 |
| 90,000 | 7,420.818 | 6,855.966 | 12,128 → 13,127 | 7.612 % | 6,865.584 | 282.803 |

The old path's longest iterate is at most 2,617.780 ms in these timings, shorter than its longest heartbeat gap. `sleep(0)` immediately queues the generator continuation; an overdue timer first completes its Future and then queues its coroutine wakeup. Ready generator iterations can precede both stages, so the timer gap spans several blocking iterates. This is finite ready-queue delay. The probe uses the selector event loop, while the server uses uvloop; this heartbeat experiment alone did not establish an HTTP outage. R825p later measured that path directly.

All 27 equality rows pass. The separate GPU mid-window peer case arrives at chunk 3 and closes the retained window at chunk 4; its short job produces 32 reference-equal IDs and the long job's checkpoints/logits/output remain equal. With equality instrumentation and peers its heartbeat maximum is 1,497.691 ms, so the solo timing maximum is not a universal latency bound. Cancellation evidence here is CPU/source coverage, not a CUDA disconnect recovery run.

## R825c repairs the observed frontend abort

The second R825p attempt rolled back after a request arriving during a long cold prompt aborted in the trace hook with `NoneType page_hashes`. R825b deferred all `prepare_for_queue` work; TabbyAPI attached its trace immediately after enqueue and read unprepared hashes. Attachment preceded cancellation/cleanup registration, so the engine subsequently ran the orphaned request. The failing arrival had a 205-token prompt; it is separate from the successful promotion's 203-token arrival.

R825c completes CPU-only hashes and MRoPE preparation at enqueue, before async construction returns, while deferring GPU allocation/admission until the window closes. Failed preparation does not append the job or advance its serial, and the async wrapper removes its temporary mapping. Trace attachment also accepts missing digests and emits them once after preparation. This repairs the observed abort and failed-preparation contract; unrelated exceptions after successful enqueue can still cross the inherited frontend cleanup seam.

R825c ran from 15:10 to 15:26 UTC, results `2026-10-01-r825c-resumable-Rh4RGH`, under the same cold solo ABBA, greedy one-token timing and 32-token equality conditions. All 27 equality rows and the chunk-3/chunk-4 peer case pass. Imported real-frontend CPU tests cover the aborting seam; the GPU peer probe covers engine enqueue/admission, not HTTP trace attachment.

| Prompt tokens | A engine ms | B engine ms | A → B processed rows/s | Saved | A max heartbeat gap ms | B max heartbeat gap ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 1,867.914 | 1,598.833 | 10,707 → 12,508 | 14.405 % | 1,921.913 | 280.737 |
| 50,000 | 4,205.384 | 3,845.714 | 11,889 → 13,001 | 8.553 % | 4,059.435 | 281.684 |
| 90,000 | 7,435.420 | 6,863.288 | 12,104 → 13,113 | 7.695 % | 6,874.943 | 282.270 |

Every timed B request has one window, no serial rows/fallback and final pipeline engagement. Allocation retries remain zero; minimum post-request driver-free memory is 984.5625 / 1,142.5625 MiB. The equality peer run has 13 windows, maximum heartbeat 1,481.013 ms and maximum iterate 567.862 ms. These are instrumented peer observations, not solo or serving bounds.

## R825p frontend gates and promotion

R825p ran from 15:32 to 15:37 UTC, results `2026-10-01-r825p-promote-wholeprompt-yj7qj3`, comparing the old live daily and candidate on the same prompts in one session. Frontend requests are greedy, with equal `min_tokens` and `max_tokens`, loop detection disabled. Cold solo requests force 32 output tokens; the decode-first case starts an 8,192-token code generation before the cold prompt, with at most three requests. The 90k frontend prompts tokenize to 90,006 input tokens; cold50 has 50,001. The first promotion attempt failed its trace-key checker before a candidate measurement; request keys must start with `r823/` for verbatim trace logging. The second attempt exposed the R825b abort described above. Only the final run promoted.

| Frontend gate | Old daily | R825c candidate | Condition |
| --- | ---: | ---: | --- |
| Cold50 engine prefill | 4.06 s | 3.66 s | 50,001 input tokens, zero cached, solo; 12,315 → 13,661 processed rows/s; B one-window source/trace inference |
| Health timeouts during prefill | 21/35 | 0/33 | cold 90,006-token solo prompt; `/health` polled every 200 ms, 3 s timeout |
| Longest health latency | 3.018 s | 0.334 s | same health case; zero HTTP errors in both arms |
| Arriving short-request TTFT | 6.267 s | 1.922 s | 203-token short prompt sent 1 s into a cold 90,006-token prompt; two requests |
| Decode-first longest SSE gap | 0.467 s | 0.488 s | code stream already active before cold 90,006-token prompt; ratio 1.046×, bar 1.5× |

The health denominator counts samples during prefill, not all polls. The trace has no explicit window counter; the whole-window check combines pinned source, contiguous pipelined forwards, interior checkpoint crossings, enabled selectors and absence of fallback. SSE gaps are delivery timestamps, with grouped MTP tokens sharing a timestamp and boundary-spanning gaps included. They do not measure kernel step time. A reverse proxy health-checking `/health` with a 3 s timeout can drop this backend when a check falls inside the old cold-prefill block; the recorded HTTP probe establishes the timeout condition without replaying the proxy's polling schedule.

The final run records FASTWARM passed, boot free VRAM 1,099 / 1,667 MiB, code at one stream with 1,024 forced greedy output tokens and short distinct prompts at 9.766 ms per decode step across three runs, greedy and chat each 6/6 identical to R809, needles 10/10 at measured depths 105,680 / 193,464 tokens, and policy-exact T32 edited-prefix reuse followed by immediate HIT. `smoke.json`, request JSONL and `audit.txt` retain those checks. The literal image-ID guard runs before serving-container removal/start; the installed launcher does not depend on an experiment-directory pin file.

## Open measurements and public-record limits

Solo direct-engine gains and the recorded frontend cases do not establish production latency distributions or a universal heartbeat bound with peers. GPU cancellation/disconnect recovery, CUDA failure cleanup and transient serving-memory minima remain unmeasured. Existing peers still use the two-chunk path. Broader frontend construction cleanup, sustained load, allocator fragmentation, NVMe restore with this stack and the earlier R823 concurrent-fidelity question remain open. A throughput or equality gate does not settle those separate questions.

No container/boot logs, Kineto traces, baseline tensor dumps or client-session content are published. R824 `reading.json` lists two complete capture files because the combined reading exceeds 2 MB; concatenate them in listed order to reconstruct its `captures` array. Kernel/forward extracts, worker measurements, original classification, cadence and decision records remain available. Private paths in provenance and audits are replaced with public or deployment-relative paths. Corrected R824 review evidence is also split by capture without dropping records. The frontend build self-test reads extracted failure JSON instead of a container-log excerpt. Engine patches, snapshot bytes and source SHA manifests are unchanged; operator scripts retain historical deployment pins and require repinning their adapted bytes before reuse. Packet designs preserve their pre-run state and can contain superseded pending-run statements; this write-up records the completed result.
