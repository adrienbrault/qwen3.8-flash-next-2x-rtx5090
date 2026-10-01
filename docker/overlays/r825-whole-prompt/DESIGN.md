# R825: one cold-prompt pipeline window — 2026-10-01

**Ready for a separate operator build and gated GPU experiment.** This packet
implements the scheduling change; it has no measured GPU equality or speedup.
No SSH, git, Docker, serving mutation or public-repo write was performed.
The operator restores LIVE `8b644c17e60049a8069fc901b6f091fa`, image
`tabbyapi:r823c-cachetail-inforward`, not the obsolete pre-R823 launcher.

[REVIEW-R824.md](../../REVIEW-R824.md) rejects R824's attention classification
(`lse` matches `false`) while confirming the four window penalties and rule 6's
scheduling counterfactual. Expected engine time saved: **~15% at 20k, ~10–11% at
50k, ~8% at 90k**; these are optimistic estimates from the 2026-10-01 R824 raw
results, not measured R825 gains. Exact ragged-tail sizes may recover less.

## Exact predecessor and cut reasons

`base/` is the served extracted Python tree after applying the bundled R823
trace, R823b policy and R823c overlay in their Dockerfile order, all with fuzz 0.
Its hashes match R823c's manifest. `fix.patch` changes only those three files;
`SHA256SUMS.exl3.base/src` pin the input and output. Line references below are to
these **R823c predecessor snapshots**, unless a repository module path is given.

1. **Coarse saves:** [base/generator/prefill_pipeline.py:338](base/generator/prefill_pipeline.py#L338)
   explicitly retains the old 32768 coarse grid outside the final two chunks.
   `window_ends()` stops at the first checkpoint at **:35**. R823c's capture
   already exists at a coarse end, but `split.crossed` at **:189** is false
   because no later end is in that window. This is conservative geometry, not
   an inherent need to read live recurrent state.
2. **Near-end grid:** that same planner chooses `recurrent_checkpoint_interval`
   (daily 2048) at **:339** when completed position is within 4096 of prompt end.
   Each final end therefore terminates a one-chunk window; **:342** returns
   False when fewer than two ends remain. At 90112 prefill rows, the earlier
   window ends at 86016, and 86016–88064 and 88064–90112 run serially. The
   checkpoint policy itself remains unchanged: `cache/checkpoint_policy.py:21`
   selects near=2048, then tail=4096 within12288, else coarse=32768, with grid
   offset by restored cached pages in `job.py:1613`.
3. **Last full page and leftover:** `window_ends()` at **:27–33** clamps to the
   last full 256-row page and rejects every non-2048 chunk. Its runtime at
   **:265** also asserts exactly2048 rows. A real 90000-token prompt has
   prefill_end89999, last_page89856 and a1935-row final merged forward starting
   88064. It cannot enter that old runtime. `base/generator/job.py:1327` requires
   the latest possible last-page checkpoint, not only a checkpoint-grid save.
4. **Existing final merge is incompatible with lookahead reservation:**
   [base/cache/prefill_merge.py:477](base/cache/prefill_merge.py#L477)
   rejects a pipeline-attached Job and at **:478** requires live recurrent
   position==start. Lookahead reservation is one chunk ahead of logical commit.
   Removing those guards globally would be unsafe. A distinct planner-only
   reservation uses validated exclusive pages and allows at most one chunk of
   position lag. `Job` must consume that descriptor; reserving a second slab in
   `try_split()` would change final partition or fail under pressure.
5. **MTP and final hidden states:** `base/generator/job.py:1424` dispatches target
   prefill; **:1438–1461** exports post-final-norm states, shifts by the previous
   chunk's cloned carry, and prefills the draft for every prompt token. This
   remains after the target join and before page commit at **:1470**. Skipping
   draft prefill for the final target chunks would change draft KV and
   speculative behavior. This is an ordering constraint; it does not require
   the window cuts. R825 changes no draft operation, KV row, carry or token count.
6. **Logits:** `base/generator/prefill_pipeline.py:199–207` traverses all target
   modules for MTP and slices only at the logits head; **:218** requests one
   token's logits while preserving the full exported hidden states. Serial
   `model/model_ls.py:395` uses the same last-token slice. Keep final norm,
   exports and the logits head in stage B for the final chunk. Do not terminate
   at the last KV writer on this MTP stack. The equality probe captures the full
   final prompt logits and first decode-verification logits, rather than only
   comparing argmax.
7. **Scratch and locks:** `base/generator/prefill_pipeline.py:289–300` waits for
   worker host issue, then joins **both** stage streams before returning to MTP.
   `util/prefill_nosync.py:54–68` implements the daily event join after B_i and
   A_(i+1). Native
   `patches/exllamav3/rebase-dev/r3/out/rebase-dev-r3/exllamav3/exllamav3_ext/quant/exl3_devctx.cu:81`
   provides a **device-global** locks/scheduler allocation, not a per-module
   resource. `modules/block_sparse_mlp.py:611` also documents shared completion
   counters; **:1361** documents allocator-shared per-call prefill scratch.
   Independent streams on one device for draft/target would require resource
   isolation. R825 retains one target issue stream per card and the existing
   two-card join before every draft call. No draft scratch or locks are cloned.
8. **Peers:** `base/generator/prefill_pipeline.py:331` caps windows at two chunks
   whenever another job is active. Keep that rule and every existing eligibility
   exclusion (TP, exported states, requeue, alternate RoPE, multimodal,
   offload, non-exclusive pages, unsupported placement).

## Implemented change

`EXL3_PREFILL_WHOLE_PROMPT=0` is the image default and runs R823c's old window
planner. `=1`, together with the daily merge and in-forward flags, uses
[src/generator/prefill_pipeline.py:48](src/generator/prefill_pipeline.py#L48).
A solo job plans all original full chunks and its original merged final partial
chunk in one window. Peers retain the two-chunk cap. Each step still executes
`Job.prefill`, existing draft prefill, page commits and existing publication.

Before current/lookahead stage A, `_capture_params()` at **:210** reserves an
end-of-chunk descriptor for every policy checkpoint, including coarse and near
saves. It additionally captures an aligned last full page regardless of grid
phase. R823c's unchanged hooks copy all GDN recurrence/conv slices and PLE conv
and cloned host-ID history on the layer's issuing stream. Captured A_i bytes
cannot be overwritten by A_(i+1). `stash_boundary()` publishes only after the
matching commit. Page hash, checkpoint size, positions and insertion policy
remain the predecessor's.

For a final merged chunk, `_prefill_split` carries the exact original last-page
split into both stages. The unchanged R823c layer paths are
`patches/exllamav3/r823c-cachetail-inforward/src/modules/gated_delta_net.py:1289`
and `.../src/modules/ple.py:578`; they run the same convolution/recurrence
partition on each side and capture at the same position. `Job.prefill` adopts
the reserved split and still calls `stash_split()` after page commit. It
suppresses a duplicate live last-page stash when an aligned pipeline capture
already owns that snapshot. The aligned last-page capture also retains the original `prompt_end` save
reason in diagnostics. The last sub-page remainder is not silently dropped.
All target GEMMs, layer forwards, recurrent arithmetic, logits and draft
operations remain unchanged.

`try_pipeline_capture()` at [src/cache/prefill_merge.py:570](src/cache/prefill_merge.py#L570)
uses the existing pinned staging pool (daily default **two slabs**). At most
current and lookahead captures are unpublished. On exhaustion it drains already
submitted stash worker tasks and retries before issuing lookahead. This lets a
previous worker-owned slab become reusable without persistent workspaces or a
new pool. If unavailable (including forced failure), it shortens the window
before crossing that checkpoint. For a final split failure it truncates to the
served last-page cut **before slicing the next IDs**; Job retains the live save
and subsequent serial leftover. A measured B fallback invalidates the whole
window performance claim, even if its state remains correct.

Every exception drains both streams before abandoning every ordinary/merged
capture. Descriptors are reserved on the owning thread; raw lookahead params
clear the preceding capture. No partially executed chunk is retried.

## Memory and remaining risks

**Added persistent VRAM: 0 bytes. Added pinned-slab capacity: 0 bytes.** The pool
count/shape, existing streams, global kernel locks and MTP scratch stay the same.
The already served one-lookahead activation retention remains bounded at two
2048-row chunks; a final ragged chunk is smaller. O(chunks) ends/prepared entries
are host Python metadata only. R577/R788's per-layer workspace/pool trap is
therefore not introduced. Allocator fragmentation and reserved-byte peaks are
still empirical questions, especially with a new ragged shape under overlap.
Record driver-free, allocated and reserved separately; never call their sum
usable headroom. The existing eligibility guard is retained, not repaired here.

GPU-only risks: recurrence split fidelity, current/lookahead capture ordering,
final exports/logits lifetime, draft/global-lock ordering, final ragged scratch
shapes, slab worker latency and real allocator headroom. Slab drain can add a
host wait and reduce gain. CPU scheduling fakes do not prove CUDA arithmetic.
Peer fairness is preserved structurally and covered offline; production
concurrent performance is not established by this solo throughput round.
The strict daily-versus-derived-A equality gate may expose existing cross-boot
nondeterminism; that is a failure, not permission to add numerical tolerance.

## Equality and registered GPU unit

[r825_probe.py](r825_probe.py) extends R823c's checkpoint equality workflow by
reusing its `equal_stash()` byte gate and immutable raw T35/T32/T65 IDs, then
adding exact 20000/50000/90000 and 20001/50001/90113-token shapes. T35 retains
its real warm prefix to cover nonzero cached-base grid offsets. All other cases
are cold. The matrix runs actual pinned daily → derived A → B → forced NO_SLAB.
For every case require exactly the same set of checkpoint positions, every
scalar, every tensor's dtype/shape/`torch.equal`/uint8 bytes, complete final
prompt and first verification logits, and all32 greedy output token IDs. There
are27 candidate case/arm rows; missing evidence fails offline replay. Whole
path and forced fallback must each engage. A failure aborts before A/B timing.

[r825-whole-prompt.sh](r825-whole-prompt.sh) uses the queue/lock/drain/lifecycle
conventions of R824 and R823c. It archives all files, immutable fixtures and LIVE;
checks baseline image digest, both source landings, launcher selectors, runtime
identity, exact placement, clocks/power and cached counts; quiesces clients;
then runs a baseline GPU container on the actual pinned daily image. A second
container loads the candidate once for equality and timing. The env selector
changes per window within that process; there are no boots between A/B arms.
The candidate tag resolves once to a frozen image ID, rechecked after the queue.
Build is strictly separate from the unit.

Timing at **20000,50000,90000 input tokens** (N−1 prefilled) runs **three ABBA
blocks per size**, six cold reps/arm/size. Each block uses the identical input in
all four slots and resets KV/recurrent caches between slots. Three distinct
first-page prefixes per size prevent accidental reuse; cached must still be0.
The same R818 fast-state warmups and a full90k shape warmup precede tests.
Equality CPU copies/logit hooks are removed from measured traffic.

Registration: complete byte equality matrix; B exactly one pipeline window,
zero serial prefill rows/fallbacks and final pipeline engagement; zero allocator
retries; at least64 MiB actual driver-free/card after each request; **at least5%
median engine time saved at each size**. Report arm medians and percent saved,
not a pooled throughput claim. Lower gain is NOT-SUPPORTED; missing/correctness
or memory evidence aborts. This round does not promote a launcher.

Each GPU container has3600s+30s kill grace; restore is600s+30s. Failure, signal,
timebox and early lock-owner exits remove the tracked probe then restore the
archived LIVE when last in queue. A successor skips the daily down/up. Restore
verifies LIVE MD5, image/config/selectors, restart count and served model. The
unit does not install or replace a launcher.

Operator build from flan: `docker build --network=none -t tabbyapi:r825-whole-prompt prefill-throughput/r825`.
Deploy the unit header's packet/dependencies, then invoke the header's
`systemd-run` recipe. No GPU gate has been pre-passed here.

## Offline validation

`python3 -B prefill-throughput/r825/test_r825.py`:15 checks pass. They execute the
actual planner, `_capture_params`, `Runtime.forward` and `run_window` with fake
CPU stage issue, covering coarse overwrite, near saves, final merged capture,
no-slab cut-before-issue, final no-slab cut, two-chunk peers and exception cleanup.
They also execute the real slab reservation/drain/retry function and reject
more than one chunk of position lag. They check zero-fuzz/no-offset patch landing
and hashes, unchanged draft block,
Python/shell syntax, ABBA/cold/memory/equality fail-closed decisions and mocked
queue-aware LIVE restore. They do not import torch or invoke Docker.

Fixtures are **verbatim real lines** from the R824 non-timestamped `probe.log`
and R823c's Docker-timestamped container log; hashes and source line numbers are
in [fixtures/provenance.json](fixtures/provenance.json). Raw replay parses all18
R823c container logs (53079 events), and reconstructs the32-request/872-delta
cold curve. R824 replay reads both complete compressed Kineto captures and all
six full cold chains; evidence is `r824-review-evidence.json` and
`offline-raw-replay.json`. No parser was validated only against invented
`prefill done` strings. Docker's import landing and every GPU gate remain for
the operator.
