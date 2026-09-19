# NVMe prefix tier, round 3: what changed since round 2 and why

Round 2 got the idea and the on-disk format mostly right, and its storage core (append-only checksummed segments with a hash-chain index) is kept. Its integration with the generator and several storage paths were wrong or unproven. See `impl-status.md` for the file:line review. This round fixes those, rebases onto `stack-r4-e3r2`, and stacks with `recurrent-tip-r1`.

## Mechanism (unchanged in shape)

- A recurrent checkpoint is resumable only together with its whole page chain, from the root to its anchor page. The tier therefore persists **chains**: when `RecurrentCache.put` stores a checkpoint whose chain is at least 32 pages, is content-hashed and has no multimodal ids, the checkpoint goes to the writer. The chain's pages that are not yet on disk are queued for copy-out.
- The pages are copied out of VRAM (main cache plus MTP draft cache) into pinned staging slabs. A writer thread appends them to 256 MiB segment files under `<dir>/ns-<digest>/`.
- On a VRAM and RAM miss, `Sequence.allocate_pages` restores the deepest checkpoint whose chain is complete on disk, then reads the pages back into freshly allocated VRAM pages.

## What changed, and why

1. **The generator thread never blocks on the tier and never sees a tier exception.**
   - r2 blocked in `_pool.acquire()` on the eviction path, in `flush()` on every idle transition, and behind a lock held for a whole compaction.
   - r2 also re-raised writer errors inside `RecurrentCache.put`, which kills the generator.
   - Now slab acquisition is `try_acquire`, and a page is skipped when no slab is free (it is copied later or re-prefilled; nothing is lost).
   - The store lock is never held across file I/O. Compaction snapshots its work under the lock, copies with `copy_file_range` outside it, and commits under the lock.
   - Writer errors are counted and logged. After 8 consecutive failures writes are disabled, and reads continue.
   - A CPU test instruments `os.*` and `open` and checks that the generator thread makes no file call at all.
2. **Copy-out is guaranteed to finish before a crash can matter.**
   - `pump()` at the top of `iterate()` copies pages on the compute stream, so stream order makes a later overwrite of the page wait for the copy.
   - Round 3b budgets it by measured time: `EXL3_NVME_TIER_PUMP_PCT`, default 1 % of each iteration, at most `EXL3_NVME_TIER_PUMP_PAGES` (4) pages.
   - R526 measured −7.3 % c1 decode at a flat 4 pages per iteration. The cost is mostly per-tensor copy launches on the generator thread, which a side stream would not remove.
   - When the queue drains, a drainer thread copies the rest on a side stream, until the next `iterate()`. `on_busy()` then makes the compute streams wait on the GPU for the side-stream copies, so the host does not wait.
   - It prints ` -- nvme tier: drained (...)` only after the writer has consumed everything. The restart gate keys on that line.
   - r2's drain stopped after about 8 slabs, so a short request followed by `docker rm -f` left chains incomplete.
3. **Restore is bounded and cannot leak or deadlock.**
   - r2 prefetched a ~115 MB checkpoint on every job start whose anchor was on disk, and consumed it only on a RAM miss (a host-memory leak). Its page-prefetch futures held slabs that were never returned when the pages turned out to be in VRAM, which deadlocks the pool.
   - Now `prepare_resume` loads a checkpoint only when it is deeper than anything in RAM. Page reads live in a `_RestoreBatch` that exists only within one `allocate_pages` call; `finish()` returns every slab, and `pump()` also releases a batch that a failed allocation left open.
   - r2 called `has_checkpoint` per page, which was O(n^2) at job start. That is gone.
4. **The byte cap is enforced as physical bytes, including while compacting.**
   - `disk_bytes()` = namespace file + every segment's size, and it is at most the cap whenever an append returns.
   - While a segment is forwarded, the transient overshoot is less than one segment. Segments are capped at cap/8.
   - The cap also applies at open: a lowered `EXL3_NVME_TIER_GB` shrinks the store before the first append.
   - Eviction goes leaf-first on the radix, and interior (superseded) checkpoints go first. r2 never evicted them, so the cap filled with dead checkpoints.
   - r2's reclaim could evict about half the cache before any segment dropped below 50 % live. Now: plain reclaim at ≤ 50 % live, forced at ≤ 75 %, so the rewrite is at most 3x the reclaimed bytes. `_expire` and every drop keep per-segment live bytes exact.
   - A free-space floor (`EXL3_NVME_TIER_MIN_FREE_PCT`, 15 %) refuses writes whatever the cap.
5. **Restart correctness.**
   - Boot verifies every header and the payload digests of the newest segment only. r2 read and hashed every byte, up to 128 GiB. Older segments are checked on each read (digest per record). A read that fails twice at the same place quarantines the record.
   - A torn tail is truncated, and chains left incomplete by a kill are pruned at open.
   - Page and checkpoint keys are the engine's own content hashes, which are stable across processes.
   - Multimodal chains are never persisted: their token ids come from a per-process counter and alias across restarts.
   - The namespace = model identity without mtimes + sha of the installed exllamav3 sources (including the .cu files the extension is built from) + every `EXL3_*` flag except the tier's own + the generator's chunk and checkpoint intervals. A different image or kernel flag set gets a fresh namespace, and stale ones are removed at open.
6. **No pickle.** Checkpoints are serialized as `RCP1` + a JSON table + 64-byte-aligned raw tensors, and are read back as `torch.frombuffer` views. r2 used `torch.save` / `torch.load(weights_only=False)`.
7. **Packaging that works on the image.**
   - TabbyAPI is not touched. The knob is the env var `EXL3_NVME_TIER=<dir>`, like every other stack flag. It is set through the launcher (`launcher-nvme-tier.patch`: `NVME_TIER=<host dir>`).
   - r2's installer searched `sys.path` for a `tabbyapi/` package that does not exist in the image (the server lives in `/app`), so its build would have failed.
   - `install.py` pins two baselines by sha256: `stack-r4-e3r2`, and `stack-r4-e3r2 + recurrent-tip-r1` with tip installed first.

## Default path

- With `EXL3_NVME_TIER` unset, `disk_cache` is never imported.
- `generator.py` gains 11 statements. Each is the flag read, the constructor, or a call under `if self.disk_page_cache is not None`.
- `pagetable.py`, `recurrent.py` and `async_generator.py` only add `disk_tier is not None` branches. `test_install.py` checks this with a line diff.

## Stacking with recurrent-tip-r1

- Tip r1 swaps the `RecurrentCache` instance's class for a `TipPolicyMixin` subclass. Its `put_tip` goes through `put`, and so through the base `put` that carries the persist hook. So tip checkpoints are persisted at commit like any other.
- The tip policy evicts through `tip_drop`, which calls `self.on_evict(key, state)` when that exists. The base LRU loop now calls the same `on_evict`, and `on_evict` hands the checkpoint to the tier if the put-time attempt had not written it. That happens when the queue was full, or when the chain was not yet complete.
- The generator hunks are the same anchors in both variants (`tools/port_generator.py`). The tip variant ships its own `generator.py` (`overlay-tip/`).

## Costs

- Copy-out: about 4.3 MB of D2H per page (the 12 attention layers + QSA planes + the draft layer).
  - The pump costs about 0.25 ms of generator time per page. The 1 % budget holds c1 decode to about 1 %, and only while a chain is pending.
  - Idle-drain copies run on a side stream and cost nothing while serving.
- Writes: a 120k agent prefix is about 2 GB of pages written once, plus about 115 MB per checkpoint. Checkpoints are taken every 32k tokens of prefill, every 2,048 decoded tokens, and at tips with tip r1.
  - A typical agent call adds about 2 checkpoints + its new pages, 0.3-0.5 GB.
  - At 1,000 calls a day that is at most 0.5 TB/day. This is the endurance number to weigh against the drive's TBW.
- Restore: 120k = about 2 GB read with bounded parallel preads, plus one checkpoint.
  - The restore runs inside `allocate_pages`, so it stalls other jobs for about 0.5-1 s at 3+ GB/s.
  - That is still far below the 12 s cold prefill it replaces.

## Round 3b (after R526)

- **Checkpoints are snapshots.** The PLE layer stashes its host-resident token-id context as a live view of the slot (`modules/ple.py:104`, `.cpu()` of a CPU tensor), and later forwards overwrite that view in place.
  - With the tier on, `RecurrentCache.put` copies any CPU view in a stash before storing it, so RAM and disk hold the checkpoint-time state.
  - The writer serializes into a buffer it owns and checks that the source did not change, then reads the record back.
  - This was R526's restore failure: the digest was computed over bytes that the next forward then rewrote. The tier-off engine bug remains, and its one-line fix belongs to its own A/B.
- **Diagnosable misses.** Every lookup has a recorded outcome, every read failure has a reason, and an open scan checks every stored checkpoint at boot. The summary line carries all of them.
- **Abort guard.** A failed page restore resets that allocation's disk-restored pages and drops the checkpoint it loaded from disk, so the job resumes from VRAM/RAM only.
- **Time-budgeted pump** (above).
