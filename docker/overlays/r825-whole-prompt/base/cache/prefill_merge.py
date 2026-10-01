"""
prefill-merge r1 (flan R803): two opt-in changes to how a recurrent (GDN / PLE) checkpoint is taken.

EXL3_PREFILL_MERGE=1
    Job.prefill cuts the forward that reaches the last full page (last_page_b) there and runs the sub-page leftover
    as one more forward, so the last-page stash can be read from the live state. With the knob on, when the whole
    remainder fits one forward (<= max_chunk_size rows), the Job runs it as ONE forward and every recurrent layer
    splits itself at last_page_b: it processes the rows before the split exactly as a forward ending there would,
    writes its slot, lets the Split copy the slot (the same slices its stash() reads), then processes the rest from
    the slot. The Job stashes that copy at last_page_b. Serial forwards per request become ceil(N_p / 2048).

EXL3_STASH_ASYNC=1
    GDNState.stash() copies each slot slice into a pinned staging slab on the current stream (no host sync) and
    returns a PendingStash whose pageable tensors a worker thread fills; readers (GDNState.unstash, the NVMe tier's
    serialize_stash) wait on its ready flag. Bytes in == bytes out.

Both knobs are read per call and default off; off, every caller takes its served path.
"""
from __future__ import annotations
from exllamav3 import cache_trace as r823_trace

import os
import queue
import threading

import torch

from ..constants import PAGE_SIZE

REVISION = "prefill-merge-r1"

_ALIGN = 64

metrics = {
    "merged_prefills": 0,           # forwards that ran merged with a split
    "merge_no_slab": 0,             # merge applicable but no free staging slab: served cut
    "snapshot_stashes": 0,          # last-page stashes stored from a split
    "snapshot_skipped": 0,          # split captured but not stored (checkpoint already current / key present)
    "async_stashes": 0,             # GDNState.stash() taken asynchronously
    "async_snapshots": 0,           # snapshot stashes (merged forward) finished on the worker (EXL3_STASH_ASYNC on)
    "async_no_slab": 0,             # EXL3_STASH_ASYNC on, no free slab: served synchronous stash
    "incomplete_snapshots": 0,      # a split whose layers did not all capture (merge then disables itself)
}
_said = set()
_disabled_reason = None


def merge_enabled() -> bool:
    return _disabled_reason is None and os.environ.get("EXL3_PREFILL_MERGE", "0") == "1"


def async_enabled() -> bool:
    return os.environ.get("EXL3_STASH_ASYNC", "0") == "1"


def _say(key, msg):
    if key not in _said:
        _said.add(key)
        print(f" -- {REVISION}: {msg}", flush = True)


def _report(counter):
    n = metrics[counter]
    if n and n % 256 == 0:
        print(f" -- {REVISION}: " + ", ".join(f"{k} {v}" for k, v in metrics.items()), flush = True)


def _disable(reason):
    global _disabled_reason
    _disabled_reason = reason
    print(f" -- {REVISION}: EXL3_PREFILL_MERGE disabled for this process: {reason}", flush = True)


# ---------------------------------------------------------------------------------------------------- the Job rule

def split_point(prefill_start: int, prefill_end: int, prompt_end: int, page_size: int = PAGE_SIZE):
    """
    The split position (absolute, = last_page_b) when the forward [prefill_start, prefill_end) can run MERGED, else
    None. Merged means: the served chunk rule already ends this forward at the prompt end (the remainder fits one
    forward and no shared page stopped it), and the last full page boundary lies strictly inside it (there is a
    sub-page leftover the served cut would run as its own forward).
    """
    last_page_b = prompt_end // page_size * page_size
    if prefill_end != prompt_end:
        return None
    if not prefill_start < last_page_b < prompt_end:
        return None
    return last_page_b


# ---------------------------------------------------------------------------------------------------- layout

def _layer_types():
    from ..modules.gated_delta_net import GDNLayerState, GatedDeltaNet
    from ..modules.ple import PLELayerState, PLELayer
    return GDNLayerState, GatedDeltaNet, PLELayerState, PLELayer


def _nbytes(shape, dtype):
    n = 1
    for s in shape:
        n *= int(s)
    return n * torch.empty((), dtype = dtype).element_size()


def stash_layout(cache):
    """
    ([(key, layer_state, kind, [(offset, shape, dtype, nbytes), ...]), ...], slab_bytes) for the device-resident parts
    of one stash, in get_all_recurrent_layers() order (= GDNState.stash()'s dict order), or None when a recurrent layer
    type is not supported. Cached on the cache object. Shapes come from the layer-state tensors (valid on meta too).
    """
    lay = getattr(cache, "_pm_layout", None)
    if lay is not None:
        return lay or None
    GDN, GDN_MOD, PLE, PLE_MOD = _layer_types()
    entries, off = [], 0
    ok = True
    for key, l in cache.get_all_recurrent_layers().items():
        # the layer state AND its module must be the ones whose forward splits (Mamba2 also uses GDNLayerState)
        if isinstance(l, GDN) and isinstance(l.module, GDN_MOD):
            cdim = l.module.conv_kernel_size
            shapes = [((1,) + tuple(l.recurrent_state.shape[2:]), l.recurrent_state.dtype),
                      ((l.conv_state.shape[1], cdim), l.conv_state.dtype)]
            kind = "gdn"
        elif isinstance(l, PLE) and isinstance(l.module, PLE_MOD):
            shapes = [((l.conv_state.shape[1], l.win), l.conv_state.dtype)]
            kind = "ple"
        else:
            ok = False
            break
        parts = []
        for shape, dtype in shapes:
            nb = _nbytes(shape, dtype)
            parts.append((off, tuple(shape), dtype, nb))
            off = -(-(off + nb) // _ALIGN) * _ALIGN
        entries.append((key, l, kind, parts))
    lay = (entries, off) if ok and entries else False
    cache._pm_layout = lay
    if not lay:
        _say("layout", "a recurrent layer type other than GDN / PLE is present: merge and async stash stay off")
    return lay or None


# ---------------------------------------------------------------------------------------------------- staging

def _pinned(nbytes):
    return torch.empty(nbytes, dtype = torch.uint8, pin_memory = True)


class StagingPool:
    """
    A bounded pool of host staging slabs (pinned in service; `alloc` is injectable for CPU tests). A slab released
    with pending CUDA events is synchronized by its next acquirer only. try_acquire() never waits for a slab.
    """

    def __init__(self, nbytes: int, count: int, alloc = None):
        self.nbytes = nbytes
        self.count = max(1, count)
        self._alloc = alloc or _pinned
        self._free = []
        self._made = 0
        self._events = {}
        self._lock = threading.Lock()

    def try_acquire(self):
        make = False
        with self._lock:
            if self._free:
                slab = self._free.pop()
                events = self._events.pop(id(slab), None)
            elif self._made < self.count:
                self._made += 1
                make, events = True, None
            else:
                return None
        if make:
            try:
                # a normal (non-inference) tensor whatever the caller's mode: it is written in place on the generator
                # thread (inference mode in service) and read on the worker; an inference-tensor slab would refuse an
                # in-place write from any thread / call outside inference mode
                with torch.inference_mode(False):
                    slab = self._alloc(self.nbytes)
            except BaseException:
                with self._lock:
                    self._made -= 1
                raise
        for e in events or ():
            e.synchronize()
        return slab

    def release(self, slab, events = None):
        with self._lock:
            if events:
                self._events[id(slab)] = list(events)
            self._free.append(slab)


def _pool(cache, lay, alloc = None):
    pool = getattr(cache, "_pm_pool", None)
    if pool is None:
        count = int(os.environ.get("EXL3_STASH_ASYNC_SLABS", "2"))
        pool = StagingPool(lay[1], count, alloc)
        cache._pm_pool = pool
        _say("pool", f"staging {pool.count} slab(s) x {lay[1] / 1024**2:.1f} MiB "
                     f"({'pinned' if alloc is None else 'injected allocator'}), {len(lay[0])} recurrent layers")
    return pool


# ---------------------------------------------------------------------------------------------------- stash objects

class PendingStash(dict):
    """A stash dict (GDNState.stash() layout) whose tensors may still be filling; wait_stash() blocks until ready."""
    __slots__ = ("_pm_ready", "_pm_error")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pm_ready = threading.Event()
        self._pm_error = None


def wait_stash(stashed):
    """Block until an asynchronously taken stash holds its bytes. A no-op for every other stash (plain dicts)."""
    ready = getattr(stashed, "_pm_ready", None)
    if ready is None:
        return
    if not ready.is_set():
        ready.wait()
    if stashed._pm_error is not None:
        raise RuntimeError(f"{REVISION}: asynchronous stash failed: {stashed._pm_error}")


def _fill(jobs):
    for dsts, views in jobs:
        for d, v in zip(dsts, views):
            d.copy_(v)


class _Worker:
    def __init__(self):
        self.q = queue.Queue()
        self.t = threading.Thread(target = self._run, name = "exl3-stash-async", daemon = True)
        self.t.start()

    def submit(self, task):
        self.q.put(task)

    def _run(self):
        while True:
            st, events, jobs, pool, slab, inference = self.q.get()
            done = False
            try:
                for e in events:
                    e.synchronize()
                # inference mode is thread-local: the destinations were allocated on the generator thread, under
                # torch.inference_mode() in service (inference tensors, as the served .cpu() stash returns them), and
                # an in-place fill of an inference tensor outside inference mode raises. Fill in the finisher's mode.
                with torch.inference_mode(inference):
                    _fill(jobs)
                done = True
            except BaseException as exc:  # noqa: BLE001 -- surfaced to every reader by wait_stash
                st._pm_error = f"{type(exc).__name__}: {exc}"
            finally:
                pool.release(slab, None if done else events)
                st._pm_ready.set()
                self.q.task_done()


_worker = None
_worker_lock = threading.Lock()


def _get_worker():
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _Worker()
        return _worker


def drain_worker():
    """Wait for every submitted asynchronous stash (tests, shutdown)."""
    if _worker is not None:
        _worker.q.join()


class Capture:
    """
    Copies of one slot's recurrent state, layer by layer, into a staging slab; finish() turns them into a stash dict
    in GDNState.stash()'s layout ({"position", "checkpoint_size", key: tuple}, get_all_recurrent_layers() order).
    """

    def __init__(self, lay, slot: int, slab, pool, site_events=False):
        self.entries, _ = lay
        self.by_layer = {id(l): (k, kind, parts) for k, l, kind, parts in self.entries}
        self.slot = slot
        self.slab = slab
        self.pool = pool
        self.host = {}
        self.devices = []
        self.site_events = [] if site_events else None
        self.done = set()
        self.consumed = False

    def _view(self, part):
        off, shape, dtype, nb = part
        return self.slab[off:off + nb].view(dtype).view(shape)

    def capture(self, rsl):
        """The slices rsl.stash(slot) reads, copied now (stream-ordered before any later write to the slot)."""
        ent = self.by_layer.get(id(rsl))
        if ent is None:
            # a layer state object the cached layout does not know (the cache re-created its states): leave this
            # split incomplete; stash_split then skips the stash and disables merge for the process
            _say("unknown_layer", "a recurrent layer state is not in the stash layout: the split stays incomplete")
            return
        k, kind, parts = ent
        if kind == "gdn":
            cdim = rsl.module.conv_kernel_size
            srcs = (rsl.recurrent_state[self.slot, :1], rsl.conv_state[self.slot, :, :cdim])
        else:
            srcs = (rsl.conv_state[self.slot, :, :rsl.win],)
            # the id context lives on the host: a clone is the snapshot (ple.py stash, ple-ckpt-clone r1)
            self.host[k] = rsl.id_state[self.slot, :rsl.ctx].clone()
        for part, src in zip(parts, srcs):
            v = self._view(part)
            if src.is_cuda:
                with torch.cuda.device(src.device):
                    v.copy_(src, non_blocking = True)
                if src.device not in self.devices:
                    self.devices.append(src.device)
            else:
                v.copy_(src)
        # R823c: record on the layer's issuing stream, including the stage-0 worker.
        # A later main-thread event must not stand in for this capture's completion.
        if self.site_events is not None:
            for device in {src.device for src in srcs if src.is_cuda}:
                with torch.cuda.device(device):
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    self.site_events.append(event)
        self.done.add(k)

    def complete(self) -> bool:
        return len(self.done) == len(self.entries)

    def _events(self):
        if self.site_events is not None:
            return list(self.site_events)
        # Preserve the original prompt-end / whole-state stash path in A and C.
        evs = []
        for d in self.devices:
            e = torch.cuda.Event()
            e.record(torch.cuda.current_stream(d))
            evs.append(e)
        return evs

    def finish(self, position: int, checkpoint_size: int, asynchronous: bool) -> PendingStash:
        assert not self.consumed
        self.consumed = True
        events = self._events()
        st = PendingStash()
        st["position"] = position
        st["checkpoint_size"] = checkpoint_size
        jobs = []
        for k, l, kind, parts in self.entries:
            # storage-owning pageable tensors, shaped and typed as .cpu() of the slices returns them: never views, so
            # the NVMe tier's own_stash_tensors (which clones views) cannot read them before they are filled
            dsts = tuple(torch.empty(shape, dtype = dtype) for _, shape, dtype, _ in parts)
            jobs.append((dsts, [self._view(p) for p in parts]))
            st[k] = dsts + ((self.host[k],) if kind == "ple" else ())
        if asynchronous:
            _get_worker().submit((st, events, jobs, self.pool, self.slab, torch.is_inference_mode_enabled()))
        else:
            try:
                for e in events:
                    e.synchronize()
                _fill(jobs)
            finally:
                self.pool.release(self.slab)
                st._pm_ready.set()
        self.slab = None
        return st

    def abandon(self):
        """Return the slab without producing a stash (copies may be in flight: the next acquirer waits for them).
        The events are recorded here, after the whole forward, so a skipped snapshot (stash_split: key present or
        checkpoint current) costs its next acquirer one host sync on them -- no worse than the served .cpu(), rare."""
        if self.slab is not None and not self.consumed:
            self.consumed = True
            try:
                events = self._events()
            except Exception:  # noqa: BLE001 -- best effort at teardown
                events = None
            self.pool.release(self.slab, events)
            self.slab = None

    def __del__(self):
        try:
            self.abandon()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------------------------------- (d) async stash

def async_stash(state, alloc = None):
    """
    GDNState.stash() under EXL3_STASH_ASYNC=1: a PendingStash, or None to take the served synchronous path (knob off,
    tensor-parallel, unsupported layer type, or no free staging slab).
    """
    if not async_enabled():
        return None
    cache = state.cache
    if cache.model.loaded_tp:
        return None
    lay = stash_layout(cache)
    if lay is None:
        return None
    slab = _pool(cache, lay, alloc).try_acquire()
    if slab is None:
        metrics["async_no_slab"] += 1
        _say("async_no_slab", "EXL3_STASH_ASYNC: no free staging slab, a stash took the synchronous path "
                              "(counted in async_no_slab)")
        return None
    cap = Capture(lay, state.slot, slab, cache._pm_pool)
    for _, l, _, _ in lay[0]:
        cap.capture(l)
    st = cap.finish(state.position, state.checkpoint_size, asynchronous = True)
    metrics["async_stashes"] += 1
    _say("async_on", "EXL3_STASH_ASYNC on: first asynchronous stash")
    _report("async_stashes")
    return st


# ---------------------------------------------------------------------------------------------------- (a) merge

class Split:
    """The params["_prefill_split"] object of a merged forward: `at` = rows before the split (relative to the forward's
    first row), capture(rsl) called by each recurrent layer after it has written its slot at the split."""

    def __init__(self, at: int, last_page_b: int, cap: Capture):
        self.at = at
        self.last_page_b = last_page_b
        self.cap = cap

    def capture(self, rsl):
        self.cap.capture(rsl)


class _SnapshotState:
    """Duck-typed state for RecurrentCache.put(): put() calls .stash() only when the key is new; the tip policy reads
    .position and .checkpoint_size."""

    def __init__(self, split: Split, checkpoint_size: int):
        self.split = split
        self.position = split.last_page_b
        self.checkpoint_size = checkpoint_size

    def stash(self):
        asynchronous = async_enabled()
        st = self.split.cap.finish(self.position, self.checkpoint_size, asynchronous = asynchronous)
        if asynchronous:
            metrics["async_snapshots"] += 1
        return st


def try_split(job, prefill_start: int, prefill_end: int, prompt_end: int, alloc = None):
    """A Split when this forward runs merged (EXL3_PREFILL_MERGE=1 and split_point() applies and the job/model are a
    supported shape and a staging slab is free), else None: the caller takes the served last-page cut."""
    if not merge_enabled():
        return None
    last_page_b = split_point(prefill_start, prefill_end, prompt_end)
    if last_page_b is None:
        return None
    gen = job.generator
    rs = job.recurrent_state
    if (len(job.sequences) != 1 or job.embeddings or getattr(job, "_ls_prefill_pipeline", None) is not None or
            rs is None or getattr(rs, "exported", False) or rs.position != prefill_start or
            gen.model.loaded_tp):
        return None
    lay = stash_layout(rs.cache)
    if lay is None:
        return None
    slab = _pool(rs.cache, lay, alloc).try_acquire()
    if slab is None:
        metrics["merge_no_slab"] += 1
        _say("merge_no_slab", "EXL3_PREFILL_MERGE: no free staging slab, a prefill took the served cut "
                              "(counted in merge_no_slab)")
        return None
    metrics["merged_prefills"] += 1
    _say("merge_on", f"EXL3_PREFILL_MERGE on: first merged prefill ({prompt_end - prefill_start} rows, split at "
                     f"+{last_page_b - prefill_start})")
    _report("merged_prefills")
    return Split(last_page_b - prefill_start, last_page_b, Capture(lay, rs.slot, slab, rs.cache._pm_pool))


def stash_split(job, split: Split):
    """After a merged forward: the last-page stash from the split's copies, with maybe_stash_recurrent's guards at
    position last_page_b (job.py maybe_stash_recurrent / is_checkpoint_boundary(PAGE_SIZE))."""
    cache = job.generator.recurrent_cache
    lpb = split.last_page_b
    try:
        if not split.cap.complete():
            metrics["incomplete_snapshots"] += 1
            _disable(f"incomplete snapshot at {lpb}: {len(split.cap.done)} of {len(split.cap.entries)} recurrent "
                     "layers captured (this stash is skipped; the next turn restores from an earlier one)")
            return
        if job.last_recurrent_checkpoint_pos == lpb:
            metrics["snapshot_skipped"] += 1
            return
        seq = job.sequences[0]
        page = seq.allocated_pages[(lpb - 1) // PAGE_SIZE]
        assert page.kv_position == PAGE_SIZE
        job.last_recurrent_checkpoint_pos = lpb
        cache.put(page.phash, _SnapshotState(split, job.recurrent_state.checkpoint_size),
                  trace_owner=job, trace_reason="prompt_end")
        if split.cap.consumed:
            metrics["snapshot_stashes"] += 1
        else:
            metrics["snapshot_skipped"] += 1
    finally:
        split.cap.abandon()


# R823c: an END-of-chunk capture, without changing any recurrence/kernel partition.
def tail_capture_enabled():
    return os.environ.get("EXL3_RECURRENT_CHECKPOINT_INFORWARD", "0") == "1"


def try_boundary(job, start, end, alloc=None):
    """Reserve before stage A, never from the worker. None retains the old window cut."""
    rs = job.recurrent_state
    if (not tail_capture_enabled() or _disabled_reason is not None or
            rs is None or rs.exported or job.generator.model.loaded_tp or
            len(job.sequences) != 1 or job.embeddings or end - start != 2048):
        return None
    lay = stash_layout(rs.cache)
    if lay is None:
        return None
    slab = _pool(rs.cache, lay, alloc).try_acquire()
    if slab is None:
        return None
    return Split(end - start, end, Capture(lay, rs.slot, slab, rs.cache._pm_pool, site_events=True))


def stash_boundary(job, split):
    """Publish only the captured bytes, after both stages and the matching page commit."""
    try:
        if not split.cap.complete():
            metrics["incomplete_snapshots"] += 1
            raise RuntimeError(f"R823c incomplete chunk snapshot at {split.last_page_b}")
        pos = split.last_page_b
        seq = job.sequences[0]
        assert seq.kv_position == job.recurrent_state.position == pos
        page = seq.allocated_pages[(pos - 1) // PAGE_SIZE]
        assert pos % PAGE_SIZE == 0 and page.kv_position == PAGE_SIZE
        if job.last_recurrent_checkpoint_pos == pos:
            return
        job.generator.recurrent_cache.put(
            page.phash, _SnapshotState(split, job.recurrent_state.checkpoint_size),
            trace_owner=job, trace_reason=r823_trace.save_reason(job, None))
        job.last_recurrent_checkpoint_pos = pos
        r823_trace.emit("tail_capture", **r823_trace.owner(job), position=pos,
                        stored=split.cap.consumed, layers=len(split.cap.done),
                        crossed=getattr(split, "crossed", False))
    finally:
        split.cap.abandon()
