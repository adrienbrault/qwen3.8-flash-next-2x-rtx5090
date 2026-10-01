# R825c

> 2026-10-01 operator note: IMAGE_ID.env is filled (sha256:aa04a1cbe94b...); the promoted launcher (262e9c31) carries the ID literally and no longer reads IMAGE_ID.env at boot (commit cb66e9e). Statements below about an UNSET pin and a launcher that requires the file are superseded.
 — frontend-safe resumable enqueue, 2026-10-01

Ready for operator build and registered GPU runs. Only files and CPU checks were
performed. No SSH, git, Docker, GPU, host-service or cluster operation was run.
No new GPU equality, speed, health or late-arrival result is claimed.

## Confirmed cause

R825p's production failure is reproduced through the **imported, unchanged**
TabbyAPI `backends/exllamav3/model.py`, not a replacement `generate_gen`:

* `base/generator/generator.py:573–583` assigns generator, page-table and serial,
  but skips `Job.prepare_for_queue` while a resumable window owns lookahead.
* `base/generator/pagetable.py:216` initializes `Sequence.page_hashes=None`.
  `Sequence.prepare` at line 232 normally computes the list from CPU tokens.
* `fixtures/app/backends/exllamav3/model.py:1542–1558` constructs AsyncJob, whose
  constructor enqueues synchronously. Line 1559 immediately calls
  `r823_trace.attach(job.job, trace_record)`.
* The served, inherited R823c `base/cache_trace.py:131` iterates `page_hashes`.
  For the real 205-token short arrival this should be a prepared **empty list**,
  but is still None. The exception happens before cleanup registration at
  `model.py:1561` and outside its consumption try block at line 1569.
* `fixtures/app/endpoints/OAI/utils/completion.py:212` consumes stream_generate;
  the collector queues its exception and the stream consumer raises it at
  line 313. The raw server log shows the engine subsequently allocated serial
  39 and continued it even though its frontend had already aborted.

`fixtures/late-arrival-failure.log` copies the real server lines 3481–3517,
including `14:42:19.127 ERROR`, the exact NoneType exception, the full call chain,
and subsequent allocation. The full candidate client and audit log are also
copied. `BASE-PROVENANCE.json` pins their bytes and records their original paths.
`repro-before.txt` is the CPU reproduction with the R825b generator and strict
inherited trace: the same TypeError at model.py:1559 / cache_trace.py:131.
The supplied production passes and failed gate are accepted as evidence; this
packet does not reinterpret or pre-pass any promotion gate.

## Source proof: the entire existing queue preparation is host-only

Dependency inspection shows MRoPE preparation is CPU-only in this stack too.
The patch runs **all of the existing prepare_for_queue immediately** and leaves
`job.py` byte-identical to R825b.

| Existing source | Work and state | Why it is host-only |
| --- | --- | --- |
| `base/generator/job.py:151`, `:193–210` | Assert CPU input IDs; create sequence IDs and optional healed prefix | CPU tensor slicing/cat and per-job Sequence construction |
| `base/generator/job.py:1104–1108` | serial, generator/page-table references, skips | Assign job-local references; no page-table mutation |
| `base/generator/job.py:1110–1131` | draft compatibility, default output budget, requeue alignment | Read immutable limits/checkpoint interval and cache presence; integer arithmetic |
| `base/generator/job.py:1133–1146` | banned-string rollback compatibility | Read state **class** guaranteed_rollback; tokenize strings on CPU; no live state access |
| `base/generator/job.py:1148–1162` | page hashes, uniqueness, MTP cached-page cap | Sequence.prepare mutates only sequence-local fields; CPU IDs and hashes |
| `base/generator/pagetable.py:23–28`, `:232–251` | rolling BLAKE2b hashes and new-page count | tensor.numpy().tobytes(), hashlib, Python lists/sets; no page lookup/claim or CUDA |
| `base/generator/job.py:1164–1173` | capacity and batch validation | Read max_pages/max_batch_size; assert arithmetic capacity, not current free-state admission |
| `base/generator/job.py:1175–1184` | held token/logprob/logit buffers, text, SAM | SeqTensor defaults to CPU and is lazy (`fixtures/util/tensor.py:19`, `:35–39`); BC_SAM constructor/reset only changes C++ vectors (`fixtures/exllamav3_ext/sam.cpp:13–53`, `sam.h:7–22`) |
| `base/generator/job.py:1186` | enqueue timestamp | Host clock; now records actual enqueue instead of delayed admission |
| `base/generator/job.py:1188–1201` | MRoPE frequencies and offset, or defaults | Architecture constructs g_rope as RoPE("cpu", ...) (`fixtures/architecture/qwen3_5.py:494`); frequencies stay CPU |
| `fixtures/util/rope.py:436–468` | MRoPE positions and frequency math | CPU zeros/input IDs, embedding span/grid **metadata** only, CPU inv_freq and fresh result tensors; does not access image embedding GPU payloads or mutate g_rope/recurrent state |
| `fixtures/exllamav3_ext/rope.cu:475–548` | gen_mrope_pos_ids native routine | Despite the .cu filename, this is a plain host loop over CPU data_ptr pointers, with no CUDA launch, stream, synchronization, or shared scratch |

These dependency snapshots came from the supplied image-source extraction.
The copied app model SHA256 matches the inherited R823 app manifest exactly:
`d9d6a6ceb290632aae7264f21c71c65a24b02765a95c54df5b2d58a07cac7e56`.
The inherited trace is **R823c output**, SHA256
`f16c2b953ebcd386e500f50b430cd018f67a0ef48f1b4ba5a175df78793a6d50`,
not the superseded original R823 module. Its tail and in-forward fields survive.

There is no GPU/page-pool operation inside prepare_for_queue to defer. The
existing separation is already correct once enqueue stops bypassing it:
`src/generator/generator.py:572–580` prepares before appending to pending_jobs;
`iterate` resumes/returns at lines 685–689 before admission, tier pumping,
checkpointing or decode; `iterate_start_jobs:2076` asserts no owning runtime.
Only then may it activate (`Job.activate:1595`) and allocate
(`Job.allocate_pages:1523`, `Sequence.allocate_pages:259`), including recurrent
cache lookup/restore and page-table changes. Retained lookahead therefore keeps
exclusive ownership of GPU and page/state pools. Pending arrival still causes
the existing pipeline to commit its already-issued next chunk and close.

The obsolete deferred-preparation admission loop is removed. Host preparation
cannot later rerun and reset enqueue time, buffers or externally injected output.
Requeue still calls the unchanged complete preparation with rq=True at
`job.py:1090`; its original preservation semantics remain intact.

## Defence in depth and preparation errors

`src/cache_trace.py:125` still emits the same render event and digest list for
normal prepared jobs; the normal-record parity test compares all fields with
the strict inherited trace using the same salt. If a different caller attaches
while hashes are None, render.page_digests is null (unknown, not empty). A
job-local flag triggers one keyed `page_digests` event in before_allocation
at line 164 after the list exists. It contains the complete digests, key,
conversation and serial. Cancellation before allocation leaves the value
unknown; no fabricated empty list or synthetic measurements are emitted.

The audit also found AsyncGenerator's already-latched error path
(`src/generator/async_generator.py:85–87`) does not prepare or enqueue at all:
it places the original error on the new AsyncJob queue. Trace policy reads now
use nullable generator metadata so attachment does not mask that original error
when job.generator is None. Ordinary policy fields and trace-disabled behavior
are unchanged. The real generate_gen test confirms the original latched error
reaches its recovery handler.

Immediate validation failures occur before AsyncJob construction returns and
before frontend cleanup registration. `AsyncGenerator.enqueue:93–99` now removes
the temporary async map entry before reraising. The request never enters pending
or active queues, does not increment the serial, does not drain the owning long
window, and cannot become a ghost request. This also covers an MRoPE exception.

## Exhaustive frontend access audit

`app/` below means the pinned copied source under `fixtures/app/` (identical
line numbers to the supplied served tree). The table covers every direct Job,
AsyncJob or sequence access after enqueue and before the first generator result,
plus callbacks that can run while the consumer is parked. Endpoint searches
covered the entire copied app tree, not just completion.py.

| Caller / file:line | Exact access / call | Preparation assumption and disposition |
| --- | --- | --- |
| `app/backends/exllamav3/model.py:1559` | job.job passed to trace.attach | Real failing seam; complete host preparation now exists before AsyncJob returns |
| `base/cache_trace.py:126–128` | write job._r823_trace; iterate job.sequences; write seq._r823_owner | Sequences exist from Job constructor; ownership tags are host-only |
| `base/cache_trace.py:130–135` | job.serial_number; job.sequences[0].page_hashes; job.generator.recurrent_checkpoint_interval_pp / recurrent_checkpoint_tail_pp / recurrent_checkpoint_interval / max_chunk_size | Hashes were the missed prerequisite; all now immediate. Nullable fallback also handles the latched wrapper path |
| `app/backends/exllamav3/model.py:1560` | store AsyncJob in self.active_job_ids[request_id] | Identity only |
| `app/backends/exllamav3/model.py:1561` | id(job), bound job.cancel; register cleanup callback | No preparation fields read; callback may run before first result |
| `app/backends/exllamav3/model.py:1570` | async for result in job | AsyncJob iterator reads cancelled and queue; no GPU state prerequisite |
| `src/generator/async_generator.py:174`, `:180–185` | self.cancelled; await self.queue.get(); sentinel/exception check, then yield result | Constructor fields only; the generator owns result production |
| `app/common/networking.py:150–163` | store registered func/args; await func(*args) on disconnect | Calls AsyncJob.cancel, never reads raw Job fields |
| `src/generator/async_generator.py:195–199`, `:135–147` | AsyncJob.generator.cancel(self), self.cancelled; raw job.job and jobs map; put sentinel | Pending cancel removes job without page allocation/release; owner cancellation retains existing drain-before-release |
| `src/generator/generator.py:596–620` | runtime.job identity, pending/active membership; owner close; active slot/pages release | Pending late arrival cannot release owner pages or close its window. Tested through real generate_gen cleanup registration |
| `app/backends/exllamav3/model.py:709–711` | iterate active_job_ids; await job.cancel() during unload | Same safe cancellation path, including requests waiting for first result |
| `app/backends/exllamav3/model.py:1186–1208` | get active AsyncJob; hasattr(job, constrain_output_now); encode CPU IDs; job.constrain_output_now(ids) | API can be called while pending; real frontend injection tested before first result |
| `src/generator/async_generator.py:187–193`, `job.py:500–546` | job.constrain_output_now; generator/tokenizer reference, forced IDs/index, filters/is_active, checkpoint | CPU metadata only; constructor initializes filters/forced state. No KV/sequence/page rewrite; first device upload is _pop_forced_token at job.py:554 |
| `app/endpoints/OAI/utils/completion.py:201–212`, `:309–313` | stream_generate / async iteration; receive generation dict or exception | No raw Job/sequence access; consumes result dictionaries, forwards exact failure |
| `app/endpoints/OAI/utils/chat_completion.py:850`, `:879`, `:896` | consume generation dictionaries; call constrain_generation_output for reasoning budget/loop | Actual calls follow generated chunks; pending invocation is nevertheless CPU-safe as above |
| `app/endpoints/OAI/utils/tool_choice.py:462–472`, `:508–510` | stream_generate; active_job_ids.get(job_id); await job.cancel() | Tool cancellation follows a generated chunk; same safe path |
| `app/common/metrics.py:276–279` | get active_job_ids, len(jobs) | Counts identities, no underlying Job state |
| `app/common/status_display.py:260–278`, model.py:1562 | JobStatus UI fields / add_job(request_id,label,context_len) | JobStatus is a separate UI object, not exllamav3.Job; no prepare prerequisite |

After the first result, the only additional direct accesses in generate_gen are
job.queue.empty/get_nowait at model.py:1587–1588, job.job at trace.finished
line 1649, id(job) at disconnect finish line 1650, and job.cancelled/job.cancel
at lines 1659–1660 and the contained-error path at 1378–1379. These rely on
constructor/queue fields or the trace owner, not new GPU preparation fields.

| Requested area | Configuration/consumption source | Audit result |
| --- | --- | --- |
| Logprobs / logits | app model.py:1518–1519, :1553–1554; Job constructor:230–234; app model.py:1638–1639 | Set before enqueue; held tensors are prepared immediately; frontend reads result dicts after output |
| Stop / banned / loop config | app model.py:1420–1433, :1547–1556; Job:235–269, :325–331 | Constructor host state; compatibility validation immediate; no post-enqueue setter requiring allocation |
| Token counting / healing | app model.py:1445, :1605–1620; Job:193–210; app token encode/validate:920–1029 | Count encoded CPU IDs before enqueue or result token IDs later; healed sequence IDs available from constructor, hashes prepared immediately |
| Embeddings / vision | app model.py:1425–1440, :1552; app chat_completion.py:323–334; common/multimodal.py:23; vision.py:135 | Image encoding is a separate pre-enqueue model operation, not an entry on the pending Job. No post-enqueue vision forward; queue preparation reads embedding span/grid metadata and computes CPU MRoPE. This packet tests preparation, not concurrent vision inference |
| Tool / grammar constraints | app model.py:1476–1497, :1557; chat_completion.py:879/:896; tool_choice.py:508–510 | Grammar built before enqueue; filters passed to constructor. No pre-result grammar compilation by generator; forced IDs/filter suspension host-safe |
| Cache/admission counting | Job.current_new_pages_required:1204; Generator.iterate_start_jobs:2076 onward | Hashes/counts now ready immediately; actual live-page admission remains behind the runtime guard |

No other frontend Job entry assumes GPU preparation before the first result.
The two extra lifecycle issues found (failed enqueue map cleanup and latched
trace metadata) are covered above and by CPU regressions.

## Packet, CPU verification and limits

`base/` is all seven R825b output files plus inherited R823c cache_trace.py.
`src/` is their final output. Only Generator, AsyncGenerator and cache_trace
change. Job, PageTable, pipeline driver, cache and merge math are byte-identical.
`SHA256SUMS.exl3.base/src` pin all eight files. `fix.patch` applies to base with
fuzz 0, no offsets, and produces src exactly. `refresh_packet.py` regenerates
these local manifests/diff/provenance after an intentional source edit.

`landing_r825c.py` composes manifests in order R823, R823b, R823c, R825, R825b,
then R825c, verifies all served bytes and imported paths, and checks the unchanged
frontend overlay too. Running an old landing verifier directly against the new
image would incorrectly reject changed files; both operator units use the new
composed landing. Dockerfile retains WHOLE_PROMPT=0 / RESUMABLE=1 defaults.

CPU evidence is saved in `offline-validation.txt`:

* 14 frontend/contract tests import the complete real copied model.py,
  Job, Sequence, SeqTensor, RoPE, AsyncJob/AsyncGenerator and cache_trace. Only
  Generator's needed methods are AST-loaded to avoid constructing a GPU model;
  allocation/forward, unrelated frontend dependencies, native SAM and native
  MRoPE position loop have explicit CPU stand-ins. Real CPU PyTorch hashing,
  frequency math and queue/trace operations run. The held-window owner is a
  deterministic CPU sentinel; retained chunk scheduling is covered separately
  by the inherited runtime tests and remains a GPU gate.
* Real production log replay extracts the short prompt size 205 and serial 39.
  The test drives the real generate_gen seam with those values. A second test
  runs the fixed generator with the **original strict attach**, proving the
  repair does not depend on the defensive trace change.
* Host validation, default/requeue budgets, token healing, stop/top-logprob
  config, immediate CPU MRoPE, forced injection, pending cancellation, disabled
  trace, normal trace field parity, unknown hashes emitted once at allocation,
  and the already-latched engine error path are exercised. Forbidden CUDA/page
  pool methods fail immediately if called; CUDA is never initialized.
* 23 stdlib packet/candidate runtime/lifecycle checks pass. All 21 unchanged
  R825b checks (which run the 15 R825 predecessor checks) also pass against their
  bundled immutable predecessor packet. Candidate behavior reuses 16 unchanged
  runtime/lifecycle checks and replaces the one enqueue test whose old assertion
  explicitly demanded the broken whole-preparation deferral. That replacement
  requires immediate preparation and no deferred flag. Patch/hash/base identity,
  unchanged math/registration, shell syntax and the adapted unit's rollback/queue
  mocks are checked separately.
* 27 R825p promotion CPU checks pass, including all original gate tests and
  added unset/malformed/duplicate image-pin failures. `promotion-gate-integrity.json`
  compares the AST of every existing probe function against the real production
  snapshot: only candidate launcher generation changed; every existing gate and
  measurement function is identical. Pin-file updates do not require changing
  launcher/probe MD5 pins.

The filled-pin portability run also passes both packet and frontend checks in
a temporary archive, importing the bundled app instead of the scratchpad copy.
To repeat the combined local CPU checks, run `validate_cpu.py --frontend-python
/path/to/cpu-torch-python`; it also requires the root updated R825p probe/tests.

The copied model defaults to the supplied app tree when it exists; an explicit
R825C_APP can select another copied tree. Otherwise tests import the bundled
byte-identical app snapshot. The model bytes must match the pinned SHA either
way. The Dockerfile runs frontend CPU checks with CUDA_VISIBLE_DEVICES empty;
the operator experiment repeats them before obtaining GPU exclusivity.
These CPU checks do not establish CUDA numerical equality or responsiveness.

## Operator build, pin and registered runs

The parent tag must be the supplied image ID. Operator commands (not run here):

```sh
test "$(docker image inspect -f '{{.Id}}' tabbyapi:r825b-resumable)" = sha256:01c967e088a35f3ffb02f98a458fefa5fb3da1604a18591c05f71399c6d548b9
docker build --network=none -t tabbyapi:r825c-hostprepare prefill-throughput/r825c
# Fill the repo artifact after building; deploy this file with the packet.
printf 'R825C_IMAGE_ID=%s\n' "$(docker image inspect -f '{{.Id}}' tabbyapi:r825c-hostprepare)" > prefill-throughput/r825c/IMAGE_ID.env
```

The supplied `IMAGE_ID.env` is deliberately UNSET. The experiment, promotion
and candidate launcher reject missing/unset/malformed IDs and compare the tag's
actual ID with the pin. All three read the same operator-filled artifact. The
installed launcher retains that guard, so its pin file must remain deployed.
No auto-inspected candidate ID is accepted as approval of an unrecorded build.

Deploy the complete R825c packet and the dependencies listed at the top of
`r825c-resumable.sh` to the matching `/srv/qwen5090` paths. Do not run the nested
`r825b/` predecessor units: those are immutable CPU-test dependencies.
Also deploy the updated root R825p unit/template, updated probes and existing
R825p fixtures/dependencies listed in that unit. Leave the live launcher alone;
the promotion unit handles its reviewed swap and rollback.

```sh
sudo systemd-run --unit=r825c-resumable --collect -p RuntimeMaxSec=43200 \
  -p TimeoutStopSec=900 -p Environment=HOME=${HOME} \
  /usr/bin/bash /srv/qwen5090/prefill-throughput/r825c/r825c-resumable.sh
# After reviewing this run's equality and registration result:
sudo systemd-run --unit=r825p-promote-wholeprompt --collect \
  -p RuntimeMaxSec=18000 -p TimeoutStopSec=600 \
  /usr/bin/bash /srv/qwen5090/r825p-promote-wholeprompt.sh
```

The R825c unit runs the **byte-identical R825b probe**: daily reference equality,
three-arm nine-case matrix plus mid-window short request, identical ABBA timing
order, six cold reps per arm at 20k/50k/90k, and heartbeat. Registration is
unchanged: equality and memory/retry/cold-window gates mandatory; SUPPORTED needs
>=5% median savings and <=500 ms max B heartbeat gap at every size. Queue source,
lock ordering, archived LIVE rollback, client/gateway drain, timeboxes and
last-in-queue daily restoration are retained. It does not promote.

R825p now selects the R825c pinned image and composed landing. All promotion
gates and their order remain unchanged: FASTWARM, VRAM floors, fn c1 <=9.95 ms,
R809 greedy/chat 6+6 identity, needles 10/10, T32 edit/HIT, paired cold50 engine
<=0.95x old, concurrent SSE gap <=1.5x old, /health zero timeouts/errors and
<=1 s worst latency, +1 s short arrival TTFT <= old. Every frontend trace request
key still starts with `r823/`, preserving the try-1 lesson. A late-arrival abort
still fails promotion and restores the archived R823c launcher.
