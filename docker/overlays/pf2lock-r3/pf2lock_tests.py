#!/usr/bin/env python3
"""
pf2lock deterministic tests (CPU only, stdlib unittest + torch): the ownership rules of the staging sets in
exllamav3/modules/ngram_embedding.py, driven into exact interleavings with gates instead of iterations.

  python3 pf2lock_tests.py [--module PATH] [-v]       (unittest arguments after the options)

--module loads that file as exllamav3.modules.ngram_embedding (pf2lock_harness.py: the real package inside the
image, stand-ins on a host without exllamav3); without it the installed module is tested. The harness shims are the
same as cpu_stress.py's (pageable allocations, a Python ngram_hash_cpu stand-in, one fp16 RAM shard with row r =
value r). Each test builds a fresh module, flips ne.PREFETCH2 as needed (the module reads it at call time), and
checks the module's state afterwards: no pending entry, no held set, at most MAX_PIN_SETS sets.

Contract under test (documented at NGramEmbedding._lock): one issuing thread per module (these tests' main thread)
plus the module's single prefetch worker. A set marked held has exactly one owner, which ends the hold on every
path, failures included.

Coverage (review r1 R1 and R4; review r2 N1):
  - claimed hit vs retire (a retire of an entry a forward already claimed is a no-op; the reverse: a retired entry
    makes the forward miss), publication during a retire scan, cancellation of a pin-less queued task while both
    sets stay held (the served free[0] IndexError) and the sleep/retry release;
  - failures on the same instance, each followed by the module serving correct rows again: allocation MemoryError
    on a fresh set, on growth of a warm set (old buffers kept) and on the worker; a failed deferred staging retired
    by drain (executor still shut down) and by acquisition pressure; a failed inline stage; a failed upload/dequant
    (set released) and an upload with no event possible (set discarded, never reused); a failed submit on both the
    flag-off and the PREFETCH2 path, rejected before enqueue (a shut-down executor) and after it (review r2 N1: a
    worker Thread.start failure inside submit leaves the task queued with no future returned; the task must not
    touch a set, on both flags, followed by same-instance serving and drain); an entry abandoned while its staging
    still runs (released only on completion);
  - PREFETCH2-on long prefetch() (deferred path), warm-set sequences (no allocation when the set already fits);
  - no module lock held by the waiting thread during a future wait, an event synchronize, a poll sleep or an
    allocation.
CPU-only caveat (not the served path): on a CPU device forward()'s inverse map aliases the staging buffer, which is
released before the final index_select; on CUDA it is a device copy. No test here relies on that window.
"""
import argparse
import concurrent.futures as cf
import os
import sys
import threading
import time
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf2lock_harness as H  # noqa: E402
import torch  # noqa: E402

ne = None
h = H.h


class OwnedLock:
    """threading.Lock that knows its owner thread, so a test can ask "does the waiting thread hold it?"."""

    def __init__(self):
        self._l = threading.Lock()
        self.owner = None

    def acquire(self, *a, **k):
        ok = self._l.acquire(*a, **k)
        if ok:
            self.owner = threading.get_ident()
        return ok

    def release(self):
        self.owner = None
        self._l.release()

    def locked(self):
        return self._l.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def mine(self):
        return self.owner == threading.get_ident()


def until(pred, timeout = 5.0, what = "condition"):
    t = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > t:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.001)


class Base(unittest.TestCase):
    prefetch2 = True

    def setUp(self):
        self._saved = dict(PREFETCH2 = ne.PREFETCH2, time = ne.time, ext = ne.ext, result = cf.Future.result,
                           alloc = H._ALLOC_HOOK)
        ne.PREFETCH2 = self.prefetch2
        self.m = H.make_module(ne)
        self.m._lock = OwnedLock()
        self.violations = []
        self.allocs = []
        self.sleeps = 0
        lock = self.m._lock

        def alloc_hook(shape, dtype):
            if lock.mine():
                self.violations.append("allocation under the module lock")
            self.allocs.append((tuple(shape) if isinstance(shape, (tuple, list)) else (shape,), dtype))
        H._ALLOC_HOOK = alloc_hook

        def sleep(s):
            if lock.mine():
                self.violations.append("sleep under the module lock")
            self.sleeps += 1
            time.sleep(s)
        ne.time = types.SimpleNamespace(sleep = sleep, perf_counter = time.perf_counter)

        real_result = self._saved["result"]

        def result(fut, timeout = None):
            if lock.mine():
                self.violations.append("future wait under the module lock")
            return real_result(fut, timeout)
        cf.Future.result = result
        self.gate = None

    def tearDown(self):
        if self.gate is not None:
            self.gate.set()
        H.shutdown(self.m)
        ne.PREFETCH2 = self._saved["PREFETCH2"]
        ne.time = self._saved["time"]
        ne.ext = self._saved["ext"]
        cf.Future.result = self._saved["result"]
        H._ALLOC_HOOK = self._saved["alloc"]

    # helpers
    def block_worker(self):
        """Occupy the single worker until self.gate is set, so queued tasks wait (not started, cancellable)."""
        self.gate = threading.Event()
        started = threading.Event()
        if self.m._executor is None:
            self.m._executor = cf.ThreadPoolExecutor(max_workers = 1, thread_name_prefix = "ngram_prefetch")

        def blocker():
            started.set()
            self.gate.wait(10)
        self.m._executor.submit(blocker)
        started.wait(5)

    def rows_ok(self, history, want = None):
        before = dict(self.m.prefetch_stats)
        out = self.m.forward(history, {})
        self.assertTrue(torch.equal(out.to(torch.float16), H.expected(history)), f"wrong rows for {history}")
        if want:
            self.assertEqual(self.m.prefetch_stats[want], before[want] + 1, f"{history} was not a {want}")

    def clean(self, drain = True):
        if drain:
            self.m._drain_prefetch()
        self.assertEqual(self.m._pending, [])
        self.assertEqual([p for p in self.m._pins if p.held], [])
        self.assertLessEqual(len(self.m._pins), ne.MAX_PIN_SETS)
        self.assertEqual(self.violations, [])

    def serves_again(self):
        """The module keeps serving after a failure: an inline miss, a deferred hit and a long prefetch hit."""
        saved = ne.PREFETCH2
        ne.PREFETCH2 = True
        try:
            self.rows_ok(h(900, 901, 902), "miss")
            a = h(910, 911, 912, 913)
            self.m.prefetch_ids(a)
            self.rows_ok(a, "hit")
            hist = torch.arange(100, 100 + ne.PREFETCH_MIN_TOKENS + 9, dtype = torch.int64).view(1, -1)
            self.m.prefetch(hist)
            self.rows_ok(hist, "hit")
        finally:
            ne.PREFETCH2 = saved
        self.clean()

    def hold_both_sets(self):
        p1 = self.m._acquire_pin(4, ne.ROW_DIM, torch.float16)
        p2 = self.m._acquire_pin(4, ne.ROW_DIM, torch.float16)
        self.assertIsNot(p1, p2)
        return p1, p2


class Races(Base):
    def test_claimed_hit_vs_retire_is_noop(self):
        a = h(1, 2, 3, 4)
        self.block_worker()
        self.m.prefetch_ids(a)
        entry = self.m._pending[0]
        out = {}
        t = threading.Thread(target = lambda: out.setdefault("r", self.m.forward(a, {})))
        t.start()
        until(lambda: not self.m._pending, what = "the forward claiming the entry")
        self.m._retire(entry)                       # the forward owns it: no cancel, no release, no count
        self.assertFalse(entry["future"].cancelled())
        self.assertEqual(self.m.prefetch_stats["retired"], 0)
        self.gate.set()
        t.join(5)
        self.assertTrue(torch.equal(out["r"].to(torch.float16), H.expected(a)))
        self.assertEqual(self.m.prefetch_stats["hit"], 1)
        self.clean()

    def test_retire_first_then_forward_misses(self):
        a = h(1, 2, 3, 4)
        self.block_worker()
        self.m.prefetch_ids(a)
        entry = self.m._pending[0]
        self.m._retire(entry)                       # not started: cancelled, nothing held
        self.assertTrue(entry["future"].cancelled())
        self.assertEqual(self.m.prefetch_stats["retired"], 1)
        self.gate.set()
        self.rows_ok(a, "miss")
        self.clean()

    def test_publish_during_retire_scan(self):
        # Both sets held by finished queued entries; a forward miss must retire one. Pause it inside _retire (after
        # the scan chose its candidate, outside the lock) and publish a new entry meanwhile: the new entry stays
        # queued and untouched, the chosen one is retired, the miss gets a set and correct rows.
        x, y, z, miss = h(1, 2, 3), h(4, 5, 6), h(7, 8, 9), h(10, 11, 12)
        self.m.prefetch_ids(x)
        self.m.prefetch_ids(y)
        until(lambda: all(e["future"].done() for e in self.m._pending), what = "x and y staged")
        self.assertTrue(all(p.held for p in self.m._pins))
        paused, go = threading.Event(), threading.Event()
        real_retire = self.m._retire
        chosen = []

        def slow_retire(entry):
            chosen.append(entry)
            paused.set()
            go.wait(5)
            return real_retire(entry)
        self.m._retire = slow_retire
        out = {}
        t = threading.Thread(target = lambda: out.setdefault("r", self.m.forward(miss, {})))
        t.start()
        paused.wait(5)
        self.m._retire = real_retire
        self.block_worker()                         # z stays queued (no set) while the scan's retire completes
        self.m.prefetch_ids(z)
        zentry = self.m._pending[-1]
        go.set()
        t.join(5)
        self.assertTrue(torch.equal(out["r"].to(torch.float16), H.expected(miss)))
        self.assertIn(chosen[0]["history"].tolist(), (x.tolist(), y.tolist()))
        self.assertTrue(any(e is zentry for e in self.m._pending))
        self.assertFalse(zentry["future"].cancelled())
        self.gate.set()
        self.rows_ok(z, "hit")
        self.clean()

    def test_cancel_pinless_queued_while_both_held_then_poll(self):
        # Served: the worker retires a not-started entry (no set), recomputes free = [] and raises IndexError on
        # free[0]. Patched: it cancels Q, finds no other candidate, polls (sleep outside the lock) and proceeds when
        # an inline holder releases.
        p1, p2 = self.hold_both_sets()
        w, q = h(21, 22, 23), h(24, 25, 26)
        self.block_worker()
        self.m.prefetch_ids(w)
        self.m.prefetch_ids(q)
        wentry, qentry = self.m._pending
        self.gate.set()                             # W runs, Q queued behind it
        until(lambda: qentry["future"].cancelled(), what = "W retiring the pin-less Q")
        until(lambda: self.sleeps > 0, what = "W polling for a set")
        self.assertFalse(wentry["future"].done())
        p1.held = False                             # the inline holder ends its stage (as forward does)
        until(lambda: wentry["future"].done(), what = "W proceeding")
        self.assertIsNone(wentry["future"].exception())
        self.assertIsNone(qentry["pin"])
        self.rows_ok(w, "hit")
        self.rows_ok(q, "miss")
        p2.held = False
        self.clean()


class Failures(Base):
    def fail_alloc_on(self, k, exc = MemoryError):
        calls = []
        prev = H._ALLOC_HOOK

        def hook(shape, dtype):
            prev(shape, dtype)
            calls.append(1)
            if len(calls) == k:
                raise exc("injected allocation failure")
        H._ALLOC_HOOK = hook

        def restore():
            H._ALLOC_HOOK = prev
        return restore

    def test_alloc_failure_fresh_set_inline(self):
        restore = self.fail_alloc_on(3)             # uids and inverse allocated, heads fails
        with self.assertRaises(MemoryError):
            self.m.forward(h(1, 2, 3, 4), {})
        restore()
        self.assertEqual([p.held for p in self.m._pins], [False])
        self.assertIsNone(self.m._pins[0].uids)     # nothing published: the next grow allocates all four
        self.clean()
        self.serves_again()

    def test_alloc_failure_growing_warm_set_keeps_old_buffers(self):
        self.rows_ok(h(1, 2, 3), "miss")
        pin = self.m._pins[0]
        old = (pin.uids, pin.inverse, pin.heads, pin.packed)
        restore = self.fail_alloc_on(4)             # three new buffers allocated, packed fails
        with self.assertRaises(MemoryError):
            self.m.forward(torch.arange(1, 40, dtype = torch.int64).view(1, -1), {})
        restore()
        self.assertIs(self.m._pins[0], pin)
        self.assertEqual((pin.uids, pin.inverse, pin.heads, pin.packed), old)
        self.assertFalse(pin.held)
        self.rows_ok(torch.arange(1, 40, dtype = torch.int64).view(1, -1), "miss")
        self.clean()
        self.serves_again()

    def test_alloc_failure_on_worker(self):
        a = h(1, 2, 3, 4)
        restore = self.fail_alloc_on(1)
        self.m.prefetch_ids(a)
        until(lambda: self.m._pending[0]["future"].done(), what = "the deferred stage failing")
        restore()
        self.assertIsNone(self.m._pending[0]["pin"])
        with self.assertRaises(MemoryError):
            self.m.forward(a, {})                   # the hit carries the worker's failure
        self.clean(drain = False)
        self.rows_ok(a, "miss")
        self.serves_again()

    def stage_fails_once(self, exc = OSError):
        real = ne.ext.ngram_hash_cpu
        state = {"armed": True}

        def hash_cpu(*a, **k):
            if state["armed"]:
                state["armed"] = False
                raise exc("injected staging failure")
            return real(*a, **k)
        ne.ext = types.SimpleNamespace(ngram_hash_cpu = hash_cpu, ngram_dequant = getattr(ne.ext, "ngram_dequant", None))

    def test_failed_future_retired_by_drain(self):
        a = h(1, 2, 3, 4)
        self.stage_fails_once()
        self.m.prefetch_ids(a)
        e = self.m._pending[0]
        until(lambda: e["future"].done(), what = "the deferred stage failing")
        self.assertTrue(e["pin"].held)              # owned by the queued entry until someone claims it
        with self.assertRaises(OSError):
            self.m._drain_prefetch()
        self.assertIsNone(self.m._executor)         # drain still shut the executor down
        self.assertFalse(e["pin"].held)
        self.clean(drain = False)
        self.serves_again()

    def test_drain_retires_all_and_shuts_down_when_one_raises(self):
        a, b = h(1, 2, 3, 4), h(5, 6, 7, 8)
        self.stage_fails_once()
        self.m.prefetch_ids(a)
        self.m.prefetch_ids(b)
        until(lambda: all(e["future"].done() for e in self.m._pending), what = "both stages done")
        with self.assertRaises(OSError):
            self.m._drain_prefetch()
        self.assertIsNone(self.m._executor)
        self.assertEqual(self.m.prefetch_stats["retired"], 2)
        self.clean(drain = False)
        self.serves_again()

    def test_failed_future_retired_by_acquire_pressure(self):
        # A failed, finished entry holds one set, a finished good entry the other; a forward miss must retire
        # the done one first (served preference) and gets its failure (served f.result() propagation); nothing
        # stays held, and the next forwards serve.
        bad, good, miss = h(1, 2, 3), h(4, 5, 6), h(7, 8, 9)
        self.stage_fails_once()
        self.m.prefetch_ids(bad)
        self.m.prefetch_ids(good)
        until(lambda: all(e["future"].done() for e in self.m._pending), what = "both stages done")
        with self.assertRaises(OSError):
            self.m.forward(miss, {})
        self.assertEqual([p for p in self.m._pins if p.held and p is not self.m._pending[0]["pin"]], [])
        self.rows_ok(good, "hit")
        self.rows_ok(miss, "miss")
        self.clean()
        self.serves_again()

    def test_failed_inline_stage(self):
        self.stage_fails_once()
        with self.assertRaises(OSError):
            self.m.forward(h(1, 2, 3), {})
        self.clean()
        self.serves_again()

    def trellis_module(self, dequant):
        K = 1
        words = ne.words_per_row(K)
        m = self.m
        m.mode = "trellis_ram"
        m.K = K
        m.tables = [torch.arange(H.ROWS, dtype = torch.int16).view(-1, 1).expand(H.ROWS, words).contiguous()]
        m.head_bias = None
        ne.ext = types.SimpleNamespace(ngram_hash_cpu = ne.ext.ngram_hash_cpu, ngram_dequant = dequant)

    def test_failed_upload_dequant_releases_set(self):
        state = {"fail": True}

        def dequant(packed_d, K, heads_d, bias, rows):
            if state["fail"]:
                state["fail"] = False
                raise RuntimeError("injected dequant failure")
            rows.copy_(packed_d[:, :1].to(torch.half).expand_as(rows))
        self.trellis_module(dequant)
        a = h(1, 2, 3)
        with self.assertRaises(RuntimeError):
            self.m.forward(a, {})
        self.assertEqual(len(self.m._pins), 1)      # CPU: no async copy, the set is reused
        self.assertFalse(self.m._pins[0].held)
        out = self.m.forward(a, {})
        self.assertTrue(torch.equal(out.to(torch.float16), H.expected(a)))
        self.clean()

    @unittest.skipIf(torch.cuda.is_available(), "needs a host without CUDA (the upload must fail with no event)")
    def test_failed_upload_without_event_discards_set(self):
        a = h(1, 2, 3)
        self.rows_ok(a, "miss")
        pin = self.m._pins[0]
        self.m.device = torch.device("cuda")        # the H2D raises; no event can be recorded either
        with self.assertRaises(Exception):
            self.m.forward(a, {})
        self.assertNotIn(pin, self.m._pins)         # never reused: an upload might still read it
        self.assertFalse(pin.held)
        self.m.device = torch.device("cpu")
        self.clean()
        self.serves_again()

    def test_failed_submit_offpath_releases_set(self):
        ne.PREFETCH2 = False
        hist = torch.arange(1, ne.PREFETCH_MIN_TOKENS + 10, dtype = torch.int64).view(1, -1)
        self.m._executor = cf.ThreadPoolExecutor(max_workers = 1)
        self.m._executor.shutdown()                 # submit raises RuntimeError
        with self.assertRaises(RuntimeError):
            self.m.prefetch(hist)
        self.assertEqual(self.m._pending, [])
        self.assertEqual([p for p in self.m._pins if p.held], [])
        self.m._executor = None
        ne.PREFETCH2 = True
        self.clean()
        self.serves_again()

    def test_failed_submit_prefetch_ids(self):
        self.m._executor = cf.ThreadPoolExecutor(max_workers = 1)
        self.m._executor.shutdown()
        with self.assertRaises(RuntimeError):
            self.m.prefetch_ids(h(1, 2, 3))
        self.assertEqual(self.m._pending, [])
        self.m._executor = None
        self.clean()
        self.serves_again()

    # review r2 N1: ThreadPoolExecutor.submit queues the work item, then starts the worker; a failing Thread.start
    # raises out of submit with the task queued and no future returned. The task runs as soon as a later submit
    # starts a worker (or never, if none does). Patched (r3): the task waits for its entry's publication decision,
    # sees the abort and touches no set; the issuer released its set. r2 released the flag-off set while the orphan
    # still staged into it (an inline miss or a later prefetch could write the same set), and the PREFETCH2 orphan
    # acquired a set that no entry owned (held forever).
    def fail_worker_start_once(self):
        real = threading.Thread.start
        state = {"failed": 0}

        def start(t):
            if t.name.startswith("ngram_prefetch") and not state["failed"]:
                state["failed"] = 1
                raise RuntimeError("can't start new thread")
            return real(t)
        threading.Thread.start = start

        def restore():
            threading.Thread.start = real
        self.addCleanup(restore)
        return restore

    def record_stages(self):
        staged = []
        real = self.m._stage

        def stage(history, pin):
            staged.append(history.tolist())
            return real(history, pin)
        self.m._stage = stage
        return staged

    def orphan_after_failed_submit(self, issue, hist):
        staged = self.record_stages()
        restore = self.fail_worker_start_once()
        with self.assertRaisesRegex(RuntimeError, "can't start new thread"):
            issue(hist)
        restore()
        self.assertEqual(self.m._pending, [])
        self.assertEqual([p for p in self.m._pins if p.held], [])     # flag off: the issuer's set is released
        q = self.m._executor._work_queue
        self.assertEqual(q.qsize(), 1, "the failed submit queued its task (post-enqueue failure)")
        item = q.get_nowait()                       # the work item holds the future submit never returned
        q.put(item)
        return staged, item.future

    def check_orphan(self, staged, orphan, hist):
        until(orphan.done, what = "the orphaned task to run")      # a never-decided task would block the worker
        self.assertIsNone(orphan.exception())
        self.assertFalse(hist.tolist() in staged, "the unpublished (aborted) task staged into a set")

    def test_post_enqueue_submit_failure_offpath(self):
        ne.PREFETCH2 = False
        a = torch.arange(1, ne.PREFETCH_MIN_TOKENS + 10, dtype = torch.int64).view(1, -1)
        b, c = (a + 200) % H.ROWS, (a + 400) % H.ROWS
        staged, orphan = self.orphan_after_failed_submit(self.m.prefetch, a)
        self.m.prefetch(b)                          # starts the worker: the orphaned A task runs first
        self.check_orphan(staged, orphan, a)
        self.rows_ok(c, "miss")                     # inline acquisition while B is queued/staged
        self.rows_ok(b, "hit")
        self.rows_ok(a, "miss")
        self.clean()
        self.serves_again()

    def test_post_enqueue_submit_failure_prefetch2(self):
        a, b = h(1, 2, 3, 4), h(5, 6, 7, 8)
        staged, orphan = self.orphan_after_failed_submit(self.m.prefetch_ids, a)
        self.m.prefetch_ids(b)
        self.check_orphan(staged, orphan, a)
        self.rows_ok(b, "hit")
        self.rows_ok(a, "miss")
        self.clean()                                # no set left held by an ownerless task
        self.serves_again()

    def test_post_enqueue_submit_failure_prefetch2_long(self):
        a = torch.arange(10, 10 + ne.PREFETCH_MIN_TOKENS + 20, dtype = torch.int64).view(1, -1)
        staged, orphan = self.orphan_after_failed_submit(self.m.prefetch, a)
        self.m.prefetch((a + 300) % H.ROWS)
        self.check_orphan(staged, orphan, a)
        self.rows_ok((a + 300) % H.ROWS, "hit")
        self.clean()
        self.serves_again()

    def test_post_enqueue_submit_failure_then_drain(self):
        # Drain right after the failure (no later submit starts a worker): the queued task never runs, nothing is
        # held, drain returns; the same instance then serves on a fresh executor.
        for flag, issue, hist in ((False, "prefetch", torch.arange(1, ne.PREFETCH_MIN_TOKENS + 10,
                                                                 dtype = torch.int64).view(1, -1)),
                                  (True, "prefetch_ids", h(1, 2, 3, 4))):
            ne.PREFETCH2 = flag
            staged, orphan = self.orphan_after_failed_submit(getattr(self.m, issue), hist)
            self.clean()
            self.assertIsNone(self.m._executor)
            self.assertFalse(hist.tolist() in staged)
            self.serves_again()

    def test_abandon_running_stage_releases_on_completion(self):
        # An entry claimed while its staging still runs (e.g. a forward whose wait was interrupted) is released by
        # the worker when the staging ends, never before (the worker is still writing the set).
        a = h(1, 2, 3, 4)
        go = threading.Event()
        real = ne.ext.ngram_hash_cpu

        def slow_hash(*args, **kw):
            go.wait(5)
            return real(*args, **kw)
        ne.ext = types.SimpleNamespace(ngram_hash_cpu = slow_hash)
        self.m.prefetch_ids(a)
        e = self.m._pending[0]
        until(lambda: e["pin"] is not None, what = "the worker acquiring")
        self.assertTrue(self.m._claim(e))
        self.m._abandon(e)
        self.assertTrue(e["pin"].held)              # still running: not released yet
        go.set()
        until(lambda: not e["pin"].held, what = "release on completion")
        self.clean()
        self.serves_again()


class Paths(Base):
    def test_prefetch2_long_prefetch(self):
        hist = torch.arange(10, 10 + ne.PREFETCH_MIN_TOKENS + 20, dtype = torch.int64).view(1, -1)
        miss = (hist + 300) % H.ROWS
        self.m.prefetch(hist)
        e = self.m._pending[0]
        self.assertIsNotNone(e["future"])           # deferred: published only with its future
        self.rows_ok(miss, "miss")
        self.rows_ok(hist, "hit")
        self.assertIsNotNone(e["pin"])              # the worker acquired it (PREFETCH2), not the issuer
        self.clean()

    def test_prefetch2_off_long_prefetch(self):
        ne.PREFETCH2 = False
        hist = torch.arange(10, 10 + ne.PREFETCH_MIN_TOKENS + 20, dtype = torch.int64).view(1, -1)
        self.m.prefetch(hist)
        self.assertIsNotNone(self.m._pending[0]["pin"])     # acquired on the issuing thread
        self.rows_ok((hist + 300) % H.ROWS, "miss")
        self.rows_ok(hist, "hit")
        self.clean()

    def test_warm_set_sequence(self):
        for k in range(30):
            a, b = h(k, k + 1, k + 2, k + 3), h(k + 50, k + 51, k + 52, k + 53)
            self.m.prefetch_ids(a)
            self.rows_ok(b, "miss")
            self.rows_ok(a, "hit")
        self.assertEqual(len(self.m._pins), ne.MAX_PIN_SETS)
        self.assertEqual(len(self.allocs), 4 * ne.MAX_PIN_SETS)     # warm sets are reused, never regrown
        n0 = len(self.allocs)
        self.rows_ok(torch.arange(1, 60, dtype = torch.int64).view(1, -1), "miss")
        self.assertEqual(len(self.allocs), n0 + 4)                  # growth: one set, four buffers
        self.rows_ok(h(5, 6, 7), "miss")
        self.assertEqual(len(self.allocs), n0 + 4)                  # a smaller request fits a grown set
        self.clean()

    def test_no_lock_held_during_waits(self):
        class FakeEvent:
            def __init__(ev, lock, sink):
                ev.lock, ev.sink, ev.calls = lock, sink, 0

            def synchronize(ev):
                ev.calls += 1
                if ev.lock.mine():
                    ev.sink.append("event synchronize under the module lock")
        a, b = h(1, 2, 3), h(4, 5, 6)
        self.rows_ok(a, "miss")
        self.rows_ok(b, "miss")
        events = []
        for p in self.m._pins:
            p.event = FakeEvent(self.m._lock, self.violations)
            events.append(p.event)
        for k in range(10):                         # hits, misses, retires (spurious entries), a running join
            x, s = h(k, k + 1, k + 2), h(k + 30, k + 31, k + 32)
            self.m.prefetch_ids(x)
            self.m.prefetch_ids(s)
            self.rows_ok(h(k + 60, k + 61, k + 62))
            self.rows_ok(x)
        self.assertGreater(sum(e.calls for e in events), 0)
        self.assertGreater(self.m.prefetch_stats["retired"], 0)
        self.clean()


def main():
    global ne
    p = argparse.ArgumentParser(add_help = False)
    p.add_argument("--module")
    a, rest = p.parse_known_args()
    ne = H.load_module(a.module)
    H.install_shims(ne)
    with open(ne.__file__, "rb") as f:
        import hashlib
        print(f"pf2lock_tests: module {ne.__file__} sha256 {hashlib.sha256(f.read()).hexdigest()} "
              f"stubbed={H.STUBBED} torch {torch.__version__}", flush = True)
    prog = unittest.main(argv = [sys.argv[0]] + rest, exit = False)
    sys.stdout.flush()
    os._exit(0 if prog.result.wasSuccessful() else 1)   # a stranded worker of a failing (served) test must not hang


if __name__ == "__main__":
    main()
