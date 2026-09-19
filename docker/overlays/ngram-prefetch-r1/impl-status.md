# ngram-prefetch round 1 — implementation status

Round 1 adds two default-off options to the served chain's exllamav3 tree:

- `EXL3_NGRAM_PREFETCH2=1` — timing-only prefetch changes:
  1. **decode/verify**: start the PLE row gather of every verification forward as soon as its
     exact id history exists (the generator hands the batch's `(bsz, 1 + window)` ids to the PLE
     worker right after the MTP draft readback), instead of staging inline at the forward's
     start. The stock path declines prefetch for decode-sized inputs (`PREFETCH_MIN_TOKENS = 256`),
     so the whole gather — R465: 0.72 ms/step mean, p90 1.66 ms at c4 d3 — is exposed today.
  2. **prefill staging issue**: under the flag, `NGramEmbedding.prefetch` acquires its staging
     set (and waits on the previous set's CUDA event) on the worker thread, not on the issuing
     thread — the prefill pipeline's `_stage0` for chunk k+1 can no longer stall on chunk k's
     cold gather or still-streaming H2D uploads. The stock pipeline already issues chunk k+1's
     staging before B_i runs, so this is the second-order part; the flag keeps it because the
     instrument would otherwise still show the issue-thread stall in the p90 tail.
- `EXL3_NGRAM_TIMING=1` (also on implicitly with PREFETCH2) — per-forward instrument: the wall
  time each PLE forward spends obtaining its rows (a prefetch entry's `future.result()` or an
  inline `_stage`), split prefill vs decode, aggregated by `NGramEmbedding.timing_report()` and
  printed every `EXL3_NGRAM_TIMING_INTERVAL` (default 30) seconds by the generator.

Base: the served image's exllamav3 tree (`tabbyapi:nvme-tier-r4-e3det-r6-rawk`, exllamav3
v1.5.0 + overlays, extracted to `src/exllamav3/` in this directory — verified byte-identical to
the newest in-cache image extraction except the served tree's own newer files: the cache in
`/T/tmpivgghvzo` predates the e3det r6 / rawk / ple-ckpt-clone r1 stack; the files this patch
touches are identical in both). Line numbers below refer to the patched tree `work/exllamav3/`.
No GPU, no remote host, nothing here is a measurement.

## What changed, file by file (patched line numbers)

`modules/ngram_embedding.py`
- :48-59 — the two selector constants. `PREFETCH2` from `EXL3_NGRAM_PREFETCH2`, `_TIMING` from
  `EXL3_NGRAM_TIMING` or implicitly PREFETCH2. Both literal default-off; an unset environment is
  the served code.
- :135-136 — per-module timing record list (capped at 4096 entries) and a forwards-seen counter.
- :413-424 — `_timed(kind, out_len)`: a context manager that records the wall time of the
  forward's row-staging wait when `_TIMING`, inert otherwise.
- :427-448 — `timing_report()`: aggregate `{prefill, decode}` cells (calls, mean/p50/p90/max/total
  ms) plus the staging counters — the box reads this through the generator's periodic printer or
  in-process.
- :564-586 — `prefetch()` under PREFETCH2: the staging set is NOT acquired on the caller
  thread. The entry (`{"history", "pin", "future"}`) is appended with `pin=None` and
  `_stage_deferred` runs on the single worker: acquire (with the new `skip` guard), set
  `entry["pin"]`, run the unchanged `_stage`, return U. Flag off: the served statements verbatim
  (guarded by `if not PREFETCH2:`).
- :588-600 — `_stage_deferred(history, entry)`: worker-side acquisition. The executor stays
  single-worker, so staging order is exactly the served order; only acquisition moves off the
  caller.
- :602-620 — `prefetch_ids(history)`: the PREFETCH2 hook for the generator. Same entry
  machinery, without the decode-sized decline (a verify history is exactly the small window the
  flag exists for). Dedups against already-queued histories like `prefetch()`.
- :472-503 (`_acquire_pin`) — one parameter `skip: dict | None = None` and the retire-candidate
  selection now (a) skips the caller's own deferred entry and (b) prefers candidates whose
  staging already finished AND owns its set. With `skip=None` and the served pending list the
  candidate order is the served one (done-with-pin first, else oldest). A new degenerate branch
  (`others` empty — every held set belongs to an inline forward, no queued entry to retire)
  spin-waits for the holder instead of retiring nothing and re-looping.
- :461-472 (`_retire`) — `entry["pin"]` may be `None` while a deferred stage is still acquiring;
  the release is skipped in that case (the set is created later and released by its own stage).

`modules/ple.py`
- :355-382 — `PLELayer.prefetch_ids(ids, params)`: builds the verify forward's history exactly
  as the forward will. With recurrent states in params it reads the carried context from
  `rsg[0].cache.get_recurrent_layer((layer_idx, 0))`'s `id_state[s, :ctx]` per slot, where the
  slots come from `params["recurrent_states"]` handles (`r.slot`) — not
  `params["recurrent_slots"]`, which `prepare_for_recurrence` only creates later inside
  `prepare_inputs`. Stateless path (position 0) takes `self._history(ids)`. The result goes to
  `NGramEmbedding.prefetch_ids`.

`generator/generator.py`
- :44-51 — flag constants (`_NGRAM_PREFETCH2`, `_NGRAM_TIMING`, `_NGRAM_TIMING_INTERVAL`) and
  the `time` import.
- :268 — `self._ngram_timing_last = None` in `__init__`.
- :1296-1303 — in `iterate_gen`, after `params.update(self.draft_model.draft_verifier_params)`
  and before `batch_logits = self.model.forward(...)`: `if _NGRAM_PREFETCH2:` then
  `for m in self.model._get_prefetch_layers: m.prefetch_ids(batch_ids, params)`. `batch_ids` is
  the verify forward's exact input tensor (pinned staging, draft tokens included, already read
  back at this point), and `params` already carries `recurrent_states`.
- :649-651 and :702-730 — `iterate` calls `_ngram_timing_maybe_report()` when `_NGRAM_TIMING`;
  the helper rate-limits by wall clock and prints one line per PLE embedding with the
  prefill and decode exposed-wait cells.

`generator/prefill_pipeline.py`
- :196-201 — comment on the existing `module.prefetch(x, params)` call in `_stage0` documenting
  the new (flag-gated) non-blocking acquisition. No functional change here: the served call site
  already issues A_(i+1)'s staging before B_i runs; the gain comes from the module side.

Unchanged: every `.cu`/`.cpp`/`.cuh` (no extension rebuild), the tiers, `job.py`, `pagetable.py`,
`model.py`, the cache classes, the sampler. The stock `prefetch()` path, `forward()`'s
consumption contract (`_match` on exact history, retire on mismatch), the `_PinSet` event
machinery and `MAX_PIN_SETS = 2` are the served ones.

## Why byte-identical in output holds

Staged sets are consumed only via `_match(history)` — an exact `torch.equal` of the queued
history against the one `forward()` builds. The verify hook builds the history from the same
recurrent-state slots and the same `ids` the forward will use; if anything advanced between the
call and the forward (it cannot on the generator thread: the only PLE state writer between the
draft readback and the verify forward is the verify forward itself), the set is simply not
matched and the forward stages inline — the served behaviour. The timing context manager and the
periodic print are read-only. The prefill path's change is thread placement of `_acquire_pin`,
not arithmetic: hashes, gathers, dequant and the inverse gather run the same code in the same
order on the same inputs.

## What was verified locally (no GPU)

CPU venv (torch 2.14 CPU, py3.14) in `.venv/`; the full exllamav3 package imports on CPU with a
stubbed `exllamav3_ext` and stubbed `triton` (the served tree's non-ngram modules never execute
CUDA paths in these scenarios).

- `tests/test_ngram_prefetch_cpu.py` — 13 passed (`python -m pytest -q -p no:cacheprovider`).
  - patch + manifest pins: patch sha256, file list (`patch -p1 -F0` exit 0, post-patch hashes,
    second application fails), `install.py` on a scratch root and its refusal of a drifted
    baseline.
  - flag-off identity: every served function the patch touches is textually unchanged except
    the allowlisted ones (`_acquire_pin`/`_retire`/`prefetch`/`forward`/class containers in
    `ngram_embedding.py`, `__init__`/`iterate`/`iterate_gen` in `generator.py`, `_stage0` in
    `prefill_pipeline.py`), and each allowlisted symbol is pinned by a targeted check: the
    served statements survive verbatim inside the `if not PREFETCH2:` guards; `Generator.__init__`
    differs by exactly the timing-clock line; `iterate` by exactly the flag-gated report call;
    `iterate_gen` by exactly the flag-gated verify prefetch; `_stage0` by a comment.
  - runtime through `cpu_runtime.py` under five flag arms (off / TIMING / PREFETCH2+TIMING /
    PREFETCH2 / `EXL3_NGRAM_PREFETCH=0`): forward equals the module's torch reference
    (`compute_ngram_ids` + `fetch_rows`) for prefill and decode histories with eos boundaries;
    prefetch-off vs prefetch-on rows `torch.equal`; a queued history that never matches is
    evicted by the acquire path when the two staging sets run out (`retired` grows) and only
    costs time; repeated mismatching prefetches keep `len(_pins) <= 2` and `_pending <= 2`
    (the "host pinned buffers for one chunk ahead only" bound); `prefetch_ids` builds the same
    history `forward()` builds and its set is consumed (`hit`); the timing instrument records
    prefill and decode waits and `timing_report()` aggregates them (mean/p50/p90/max all ≥ 0);
    flag-off inertia (`_timing` empty, counter zero).
- `tests/fake_ext.py` — a faithful CPU port of `ngram.cu`'s `ngram_hash_cpu` (uint64 xor chain,
  the signed-cast `%` with the adjust, the sorted-unique dedup and the head assignment) and
  `ngram_gather_cpu` (run-coalesced reads delegated to index_select on a registered store);
  `ngram_dequant` delegates to the tree's own `ngram_codec.dequant_rows` with the per-head bias,
  matching `fetch_rows`' rounding. Its hash arithmetic was checked equal to the torch reference
  on a deliberately int64-wrapping case (token 200000, multiplier 2^62+1): the bit patterns and
  the modulo agree, which is the C++/reference agreement the served HF parity test pins.

Not verified locally and known limits:

- Nothing compiled (no CUDA toolchain here); nothing ran on CUDA. The deferred-acquire path's
  first real run against live CUDA events and a queued stream is `gpu_ngram_prefetch.py` on the
  box (step 2 of box-ab-spec.md) — that is the boot blocker to catch before serving.
- The timing cells' actual values on the served stack (what is exposed at c1/c4 and on a 60k
  cold prefill today) only the box can measure — that is the instrument's purpose.
- `prefetch_ids` uses `params.get("layer_instance", 0)` for the recurrent-layer lookup, the same
  default as the forward's stateless path. With a `layer_map` that repeats the PLE layer, the
  prefetch would read instance 0's carried context — a mismatch then only discards the set
  (timing-only contract); the served pack has one PLE instance.

## Risks

- `_acquire_pin` under PREFETCH2 now also runs on the worker thread (deferred path). The served
  code's callers were the generator thread and the prefill-pipeline stage-0 thread; both exist
  today, so a cross-thread acquire is not new to the design, but the `skip`/empty-pending
  branches are new logic. The CPU scenarios exercise them (deferred order, pressure, discard);
  the GPU harness exercises them against real events. The worst case in the new branches is a
  bounded spin-wait (a holder frees its set at the end of its inline stage), never a deadlock:
  `skip` prevents `_retire` from joining the future running on the calling thread itself.
- A stale PREFETCH2 entry that no forward matches sits in `_pending` until the staging sets run
  out and `_acquire_pin` retires it — the served eviction discipline, unchanged. Its worker may
  be writing the set while the retire joins it (served cost, unchanged).
- `prefetch_ids` is called with `batch_ids`, which the generator builds once per verify round;
  if a future change feeds unpinned ids the H2D-only claim in the comment weakens (correctness
  unaffected).
- The timing list grows only under the instrument; 4096 capped entries ≈ a few hundred KB.
- `engine_identity()` already includes every EXL3_* variable, so a flag-on boot opens its own
  NVMe tier namespace; verified from served `disk_cache.py` (no code change needed).

## Status

Files in this directory:

| file | content |
|---|---|
| `served-source.patch` | 370 lines over four files: `modules/ngram_embedding.py`, `modules/ple.py`, `generator/generator.py`, `generator/prefill_pipeline.py` |
| `overlay/install.py`, `overlay/manifest.json` | hash-pinned install (baseline + post-patch sha256 per file, patch sha256, `patch -p1 --fuzz=0`, aborts on any mismatch or `.rej`); keeps the prebuilt extension |
| `Dockerfile.box` | `ARG BASE=tabbyapi:nvme-tier-r4-e3det-r6-rawk`; install, py_compile of the tests, then asserts the selectors read False unset / TIMING on / PREFETCH2 on and that the generator + PLE hooks exist; no rebuild, no heredocs |
| `tests/test_ngram_prefetch_cpu.py` | 13 pytest cases, CPU torch 2.14, run locally |
| `tests/cpu_runtime.py`, `tests/fake_ext.py` | the scenario runner (five flag arms) and the faithful CPU port of the n-gram extension functions it drives |
| `tests/gpu_ngram_prefetch.py` | the GPU/box harness (real CUDA events, real dequant kernel, synthetic table; one card per run) |
| `box-ab-spec.md` | build, harness, flag-off and flag-on fingerprint boots, the timing-instrument runs at c1/c4 + 60k cold prefill, prefill and mp_decode A/Bs, tier sequence — inside the 15-minute cap |

Patch application: `patch -p1 -F0 --no-backup-if-mismatch` on the full served tree exits 0 with
no reject (macOS patch; the box image uses GNU patch), verified by the pytest suite.

What only the box can verify: the harness on both cards (first real CUDA events + dequant run of
the deferred path), the served fingerprints on both arms, the actual exposed-wait cells at
c1/c4/60k, the A/B tables, and the tier sequence (`box-ab-spec.md`).
