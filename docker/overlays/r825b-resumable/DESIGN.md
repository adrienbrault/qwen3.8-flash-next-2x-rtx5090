# R825b — resumable solo whole-prompt window, 2026-10-01

**Ready for an operator build and registered GPU run. No GPU result is claimed.**
Only files in this packet were written. No SSH, git, Docker, host, cluster,
launcher, or public-repo operation was performed.

## Chosen mechanism and the necessary MTP join

R825b keeps one logical window for a solo prompt, but commits **one chunk per
`Generator.iterate()`**, including its original merged final chunk. It retains
R825's stage-A lookahead, activation, parameters, CUDA ready event, captures and
slab ownership across returns. It does **not** retain an executing target kernel
across the MTP draft call: R825 already waits for host issue and joins both target
streams before draft prefill. Removing that join would race the shared per-device
scratch and kernel locks identified in R825's design. Retaining the *completed*
lookahead gives the same A-next/B-current overlap without changing this ordering.
Thus resumability is feasible without the proposed 12-chunk fallback.

References below name packet snapshots; `base/` is the R825 image's predecessor
code for the affected files and `src/` is the candidate output.

* [base/generator/async_generator.py:22](base/generator/async_generator.py#L22)
  calls synchronous `iterate()` at line 36 and only then yields at line 41.
  R825's [base/generator/prefill_pipeline.py:64](base/generator/prefill_pipeline.py#L64)
  consumes the entire window in that call.
* [src/generator/prefill_pipeline.py:87](src/generator/prefill_pipeline.py#L87)
  advances at most `count` chunks; new solo windows use count=1 at the end of
  `prefill_job`. It retains the attachment only after successful chunk commit,
  capture publication and a completed pending Future. The persistent `index`
  advances once in `forward`, not once per window call.
* [src/generator/prefill_pipeline.py:395](src/generator/prefill_pipeline.py#L395)
  still issues A-next before B-current, then completes both target streams at
  lines 405–416. [src/generator/job.py:1438](src/generator/job.py#L1438) retains
  the identical MTP hidden-state shift, carry clone, draft prefill and page commit.
  No target math, draft math, logits slicing, KV rows or stash policy changes.
* [src/generator/generator.py:689](src/generator/generator.py#L689) resumes before
  tier pumping, admission, recurring checkpoints or decode, and immediately
  returns its progress results. A newly started resumable window also returns
  at line 723. Even the final chunk returns before first decode; the next normal
  iteration performs it. No extra prefill is issued in that iteration.
* [src/generator/async_generator.py:44](src/generator/async_generator.py#L44)
  uses `sleep(0.000001)` following a resumable chunk. A positive timer lets overdue
  heartbeat/HTTP timers run before another chunk; `sleep(0)` can otherwise put
  this task ahead of overdue timers for multiple ready-queue turns. Ordinary
  daily iterations retain `sleep(0)`.

`EXL3_PREFILL_WHOLE_PROMPT=0` remains the image default: it selects the served
R823c planner and normal iteration behavior. With WHOLE_PROMPT=1,
`EXL3_PREFILL_RESUMABLE` defaults to 1. Setting RESUMABLE=0 is an optional R825
control, not a prerequisite for the normal daily arm. An open window latches its
mode and is always finished or cleaned up even if environment selectors change.

## Carried state and between-iterate invariants

The `_Runtime` contains the owning Job/model, planned `ends`, next `index`, stage
layout/devices/caller streams/executor/peer-copy function, `pending` completed
Future, `captures`, `merges`, `prepared`, whole-prompt mode and resumable mode.
The Future owns the next IDs, private host metadata/parameters, cloned stage-A
activation and producer ready event. Captures and merged descriptors retain the
existing pinned slab until the existing stash worker consumes it or cleanup
abandons it. No extra slab capacity, stream, GPU workspace or persistent VRAM
allocation is introduced. Retention stays at one current/one lookahead chunk.

After committed chunk i, while its window remains attached:

1. KV/page metadata, logical recurrent position, MTP draft KV, carry and progress
   have committed through end(i). Target stage B has completed through end(i).
2. Target stage A's live recurrent/PLE state and ID history have advanced through
   end(i+1). **They are ahead of the logical position** and cannot be read as a
   live checkpoint at end(i). The next Future owns the exact activation/params
   needed to complete B(i+1); it must never be discarded and that chunk reissued.
3. Both target stages were joined before draft(i). Draft(i) may still have work
   queued on the same caller stream; stream ordering and cleanup joins remain
   unchanged. There is no stage-A host worker still issuing. Captured checkpoint
   bytes are private copies, and any submitted stash worker owns immutable slabs.
4. Only the owning window may issue target or draft inference until retained
   lookahead commits or cancellation drains it. No peer admission, KV restore,
   tier pump, live checkpoint or decode runs between these steps.
5. Submitted captures from chunk i have been published by R825's unchanged stash
   path. Remaining reserved captures belong to lookahead, not an unpublished
   committed chunk. At most R825's two slabs are reserved/worker-owned; no-slab
   drain/retry and cut-before-issue fallback remain unchanged.

Ownership is recorded on Job, Generator and Cache. The Cache attachment permits
cleanup before a replacement Generator resets the shared page/state pools.

## Late arrivals, peers and event-loop-accessible paths

[src/generator/prefill_pipeline.py:121](src/generator/prefill_pipeline.py#L121)
checks pending jobs and other active jobs **before every resumed chunk**. On an
arrival it truncates `ends` to `index+1`, updates a retained boundary descriptor's
`crossed` reason, consumes already issued A-next, runs its B/draft/commit and
publishes its captures. It issues no A-after-next and closes the window. Pending
jobs are admitted on the following normal iteration.

The owner was solo when its resumable window was created, so peer decode is
impossible during retained lookahead. Already admitted peers use R825's
**existing two-chunk cap in a synchronous window** (the `active_jobs` condition
in `_Runtime.__init__`, line 257, and planner `limit`). They then reach their own
prefill and decode in that normal iteration. Carrying peer windows across
iterations would repeatedly return on the first Job and starve later Jobs;
R825b deliberately keeps the existing peer scheduling path. The one-chunk bound
here is for solo/resumed windows, not a bound on a full three-peer decode round.

The event-loop entry paths were checked:

* Async enqueue reaches `Generator.enqueue`. At
  [src/generator/generator.py:573](src/generator/generator.py#L573) it assigns the
  serial and queues the incoming Job, but defers `prepare_for_queue` while a
  window owns state. This also prevents incoming MRoPE/embedding preparation
  from touching the GPU. `iterate_start_jobs` at line 2080 performs preparation
  and activation only after the window closes. Preparation failures are reaped.
* `constrain_output_now` at [src/generator/job.py:500](src/generator/job.py#L500)
  changes forced-token/filter host metadata, not prompt IDs, KV, recurrent state
  or pipeline activation. The first forced-token device upload happens in decode.
* Cache statistics at [src/generator/generator.py:430](src/generator/generator.py#L430)
  traverse host page/hash metadata; they do not stash or execute a forward.
  Recurring checkpoints at line 792 are suppressed while suspended; the window
  already publishes matching captured boundaries. Iterate's early return also
  suppresses visualization and the normal draft/target decode dispatch.
* The registered environment has no NVMe tier and the probe uses CPU tier size
  zero, matching R825. Iterate's tier pump cannot run while suspended. This packet
  does not authorize external callers to directly run Model.forward or rewrite
  cache tensors during active generation; that was never a supported concurrent
  generator entry point.

## Cancellation, exceptions, shutdown and reset

[src/generator/prefill_pipeline.py:68](src/generator/prefill_pipeline.py#L68)
centralizes R825's cleanup: wait for outstanding host issue, attempt **both**
stream joins even after worker/first-stream failure (runtime `drain`, line 357),
then abandon every retained ordinary/merged capture and clear all three ownership
attachments in `finally`. No partially executed chunk is retried.

Generator `cancel` (line 600), `clear_queue` (line 534), and `reap_failed_job`
(line 2036) close the owner's window **before** draft-slot release, recurrent-slot
release or page deallocation. Direct Job deallocation has the same guard at
[src/generator/job.py:1564](src/generator/job.py#L1564). A resumed chunk exception
uses the existing per-job error result and reap path. A first-chunk exception also
returns immediately, avoiding another forward in that iteration.

Async shutdown first stops/awaits the iteration task, then clears a suspended
queue at [src/generator/async_generator.py:114](src/generator/async_generator.py#L114).
Generator takeover clears/drains the old owner and its Cache attachment before constructing a new
PageTable at [src/generator/generator.py:187](src/generator/generator.py#L187).
Direct PageTable reset (line 367) and Cache state reset (line 451) clear a suspended
owner before reset. Direct defragmentation refuses a suspended owner (PageTable
line 896); ordinary idle housekeeping runs after cancellation has drained it.
A CUDA failure remains an error; cleanup attempts do not establish that a failed
CUDA context is usable.

## Expected gain and freeze bound

The measured R825 medians below are replayed verbatim from the successful
2026-10-01 `r825-whole-prompt-YhItMe/gpu/results.json`, copied into `fixtures/`
with hashes/provenance. Solo R825b has exactly the same chunk/stage overlap and
one startup/final drain. The added cost is C event-loop resumptions/returns,
plus result delivery and the positive timer, where C is the prompt chunk count.
A planning estimate is **0.2 ms per extra yield**, with a deliberately broad
0.1–1.0 ms range. This is host overhead, not a GPU measurement; the ABBA run will
replace it. First decode is deferred by one yield and is included in engine time.

| Input tokens | Chunks | R825 A/B ms | R825 saving | Added host ms at 0.2 ms/chunk | Expected R825b saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 10 | 1853.756 / 1596.503 | 13.877% | 2.0 | 13.769% |
| 50,000 | 25 | 4204.496 / 3841.325 | 8.638% | 5.0 | 8.519% |
| 90,000 | 44 | 7422.548 / 6855.598 | 7.638% | 8.8 | 7.520% |

At 1 ms/chunk the corresponding estimates are 13.338%, 8.043%, 7.045%.
No windows are added at coarse/near boundaries or before the merged final tail.
A per-window chunk cap would lose startup/drain overlap repeatedly; it is not
implemented because retaining the completed lookahead is safe under the guards
above.

A solo synchronous call now covers one chunk, at most 2048 rows, including the
first startup chunk or final merged chunk: **approximately <=350 ms expected**.
A 50 ms heartbeat may have a gap up to roughly 400 ms including its timer period;
registration requires **<=500 ms**, using the largest gap over all six B reps at
each size. GPU stalls, slab-worker draining and scheduling delays mean this is
an experimentally gated bound, not a hard realtime guarantee. A's old 32768-row
windows have the supplied approximately 2.5 s synchronous burst expectation.
The unmodified A wrapper's `sleep(0)` can further delay a timer across ready-queue
turns, so its measured heartbeat gap may be larger; the actual maximum is
reported without forcing either number.

## Build, landing, registered GPU unit and offline evidence

`base/` contains R825's three output files plus its inherited landed
Generator/PageTable and unchanged AsyncGenerator/Cache files needed for lifecycle
guards. The supplied plateau extraction predates the R823 overlays: Generator
and PageTable were reconstructed from the exact bundled R823b and R823 trace
artifacts, respectively, and verified against their inherited image manifests.
`BASE-PROVENANCE.json` records the composition; no overlay is silently removed.
`SHA256SUMS.exl3.base/src` pin all seven files; `fix.patch` applies with fuzz zero
and no offsets. `landing_r825b.py` composes the R823/R823b/R823c/R825 manifests,
then overrides only this delta and verifies loaded paths and file bytes.
Dockerfile is `FROM tabbyapi:r825-whole-prompt`; expected parent image is
`sha256:d8a81e276dce0e86b511a243971696a8819012e7c1c2c92b45e055aa305fa0ff`.
The operator must verify that tag resolves to this ID before building:

```sh
test "$(docker image inspect -f '{{.Id}}' tabbyapi:r825-whole-prompt)" = sha256:d8a81e276dce0e86b511a243971696a8819012e7c1c2c92b45e055aa305fa0ff
docker build --network=none -t tabbyapi:r825b-resumable prefill-throughput/r825b
```

The operator deploys the packet/dependencies listed in `r825b-resumable.sh` and
uses its systemd-run recipe. The unit reuses R825's GPU queue, lock, client/gateway
drain, archived LIVE rollback, pinned daily identity, source landings, placement,
clocks/power gates, timeout and last-in-queue restoration. It checks the R825
parent tag too. It neither builds nor promotes. Daily rollback remains launcher
MD5 `8b644c17e60049a8069fc901b6f091fa` / `tabbyapi:r823c-cachetail-inforward`.

`r825b_probe.py` reuses the nine-case, three-arm byte matrix (actual serial daily
reference vs candidate A, B and forced NO_SLAB), warm T35 cached offset, full
final-prompt/first-verification logits and 32 greedy IDs. It adds a cold90k long
Job with a 128-token short Job enqueued after retained chunk 3. It requires the
retained window to finish at chunk 4, and compares the long checkpoints/logits/
tokens and short's 32 tokens with their separate serial daily references. The
short's prompt+output stays below one page, so it adds no checkpoint positions to
the long Job's snapshot set. The short finishes before long final logits; equality
hooks select the primary owner's prefill and its first verification.

Every probe run drives the **actual AsyncGenerator** on an asyncio loop, attaching
the already loaded Generator without constructing another GPU stack. A 50 ms
heartbeat starts before enqueue and records every elapsed tick gap through cold
first token, including the tick delayed across completion. The probe also records
maximum synchronous iterate duration. Equality copying/hooks are removed before
three ABBA blocks at 20k/50k/90k, with six cold reps per arm and identical IDs in
each block. Arm A uses WHOLE_PROMPT=0, arm B uses WHOLE_PROMPT=1/RESUMABLE=1.

Registration fails closed on incomplete original/late-peer equality, warm timing,
wrong order/prompt, B fallback/serial rows/multiple windows/missing final pipeline,
allocator retries or <64 MiB actual driver-free/card. SUPPORTED additionally
requires B median engine savings >=5% **and** max heartbeat gap <=500 ms at every
size. A and B heartbeat maxima are reported per size. Missing heartbeat/peer
fields cannot be filled from R825's earlier GPU result.

Environment assertions compare the complete EXL3/EXLLAMAV3 selector mapping with
`fixtures/live-container-env.txt`, excluding only the two experiment selectors.
The archived real daily leaves NOSYNC unset; the effective value must be zero.
The probe does not introduce NOSYNC=1. System/container metadata is not confused
with serving selectors.

`python3 -B prefill-throughput/r825b/test_r825b.py` runs 21 packet checks and executes
all 15 unchanged R825 checks from the bundled `r825/` predecessor packet. They
exercise real AST-loaded planner/runtime/driver and generator lifecycle methods,
actual AsyncGenerator heartbeat/shutdown with CPU sleeps, both cleanup joins,
patch/hash/syntax landing, and the new unit's queue-aware restore mocks. Real
R825 results/equality/logs replay succeeds under the old gate and fails under the
new gate because responsiveness and late-peer evidence are absent. Additional
synthetic heartbeat/peer fields are used only for explicitly labeled negative
registration tests. These CPU checks do not establish CUDA equality, speed or
responsiveness; all GPU steps remain with the operator.
