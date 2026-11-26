"""CPU ownership, persistence and alias regressions using the candidate's modules."""
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import unittest
from collections import Counter, deque
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from _iso import import_package

import_package()
from exllamav3.cache.prefill_merge import PendingStash
from exllamav3.cache.recurrent import RecurrentCache
from exllamav3.generator import fast_accept, name_policy
from exllamav3.generator.disk_cache import DiskPageCache, engine_identity
from exllamav3.generator.pagetable import Sequence
from exllamav3.model.model_ls import _split_boundaries
from exllamav3.modules import row_table


class RowTableOwnership(unittest.TestCase):
    def setUp(self):
        self.alloc = patch.object(row_table, "_pinned_empty", lambda shape, dtype: torch.empty(shape, dtype=dtype))
        self.alloc.start()
        self.addCleanup(self.alloc.stop)
        self.table = row_table.RowTable("fixture", [])
        self.table._set_stores([torch.arange(256).reshape(64, 4)])
        self.addCleanup(self.table.close)

    @staticmethod
    def resolve(ids, pin):
        return ids.flatten(), True

    def lookup(self, ids, decode=lambda rows, *_: rows):
        return self.table.lookup(ids, ids.numel(), self.resolve, decode, torch.device("cpu"))

    def test_submit_that_enqueues_then_raises(self):
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                a, b = torch.tensor([[1, 2]]), torch.tensor([[3, 4]])
                stage_calls = []
                stage = self.table._stage

                def record(ids, *args):
                    stage_calls.append(ids.tolist())
                    return stage(ids, *args)

                with patch.object(self.table, "_stage", side_effect=record):
                    with patch.object(threading.Thread, "start", side_effect=RuntimeError("after enqueue")):
                        with self.assertRaisesRegex(RuntimeError, "after enqueue"):
                            self.table.prefetch(a, 2, self.resolve, deferred=deferred)
                    self.table.prefetch(b, 2, self.resolve, deferred=deferred)
                    self.assertEqual(self.lookup(b).tolist(), [[12, 13, 14, 15], [16, 17, 18, 19]])
                    self.table.drain_prefetch()
                    self.assertEqual(stage_calls, [b.tolist()])
                    self.assertFalse(self.table._pending)
                    self.assertFalse(any(pin.held for pin in self.table._pins))

    def test_partial_growth_is_not_published_and_lookup_recovers(self):
        a = torch.tensor([[1, 2]])
        self.lookup(a)
        pin = self.table._pins[0]
        old = (pin.uids, pin.inverse, pin.heads, pin.packed)
        with patch.object(row_table, "_pinned_empty", side_effect=[torch.empty((3, 4)), MemoryError("pin")]):
            with self.assertRaises(MemoryError):
                self.lookup(torch.tensor([[1, 2, 3]]))
        self.assertEqual(tuple(map(id, old)), tuple(map(id, (pin.uids, pin.inverse, pin.heads, pin.packed))))
        self.assertFalse(pin.held)
        self.assertEqual(self.lookup(a).shape, (2, 4))

    def test_failed_stage_and_failed_decode_release_holds(self):
        a = torch.tensor([[1, 2]])
        with patch.object(self.table, "_stage", side_effect=RuntimeError("stage")):
            self.table.prefetch(a, 2, self.resolve, deferred=True)
            with self.assertRaisesRegex(RuntimeError, "stage"):
                self.lookup(a)
        with self.assertRaisesRegex(RuntimeError, "decode"):
            def fail(*_):
                raise RuntimeError("decode")
            self.lookup(a, fail)
        self.assertFalse(any(pin.held for pin in self.table._pins))
        self.assertEqual(self.lookup(a).shape, (2, 4))

    def test_deferred_hit_and_inline_miss_overlap(self):
        hits = 0
        for i in range(5000):
            a = torch.tensor([[i % 63, (i + 1) % 64]])
            b = torch.tensor([[(i + 2) % 63, (i + 3) % 64]])
            self.table.prefetch(a, 2, self.resolve, deferred=True)
            query = a if i % 2 else b
            expected = query.flatten().reshape(-1, 1) * 4 + torch.arange(4)
            self.assertTrue(torch.equal(self.lookup(query), expected))
            hits += i % 2
        self.table.drain_prefetch()
        self.assertGreaterEqual(self.table.prefetch_stats["hit"], hits)
        self.assertFalse(any(pin.held for pin in self.table._pins))


class DiskReadiness(unittest.TestCase):
    def setUp(self):
        self.key = b"new content hash"
        self.old = b"\0" * 8 + b"a" * 8
        self.disk = object.__new__(DiskPageCache)
        disk = self.disk
        disk._lock = threading.RLock()
        disk.states, disk.pending_states, disk.state_candidates = {}, {}, {}
        disk.pages, disk.pending_pages = {self.key: {}}, {}
        disk._queue = deque()
        disk.total_size = 0
        disk.recurrent_sides = {}
        disk.metrics = Counter()
        disk._can_write = lambda: True
        disk._spaced = lambda *_: True
        disk._make_room = lambda _: True
        disk._enqueue = disk._queue.append
        disk._state_layout = lambda *args: {}
        self.cache = RecurrentCache(NS(loaded_tp=False), 4096)
        self.cache.disk_tier = disk

    @staticmethod
    def pending_pair():
        return {"position": 256, "checkpoint_size": 64, "target": PendingStash(), "draft": PendingStash()}

    def test_insert_rekey_and_forced_flush_wait_for_target_and_draft(self):
        for route in ("insert", "rekey", "flush"):
            with self.subTest(route=route):
                stash = self.pending_pair()
                seen = threading.Event()

                def layout(*args):
                    self.assertTrue(stash["target"]._pm_ready.is_set())
                    self.assertTrue(stash["draft"]._pm_ready.is_set())
                    seen.set()
                    return {}

                self.disk._state_layout = layout
                if route == "rekey":
                    self.cache[self.old] = stash
                    run = lambda: self.cache.rekey(self.old, self.key)
                elif route == "insert":
                    run = lambda: self.cache._insert(self.key, stash, persist=True)
                else:
                    run = lambda: self.disk.write_state(self.key, stash, force=True)
                with patch.object(self.disk, "note_state", self.disk.write_state):
                    t = threading.Thread(target=run)
                    t.start()
                    try:
                        self.assertFalse(seen.wait(0.02))
                        stash["target"]._pm_ready.set()
                        self.assertFalse(seen.wait(0.02))
                    finally:
                        for side in ("target", "draft"):
                            stash[side]._pm_ready.set()
                        t.join(2)
                self.assertFalse(t.is_alive())
                self.assertTrue(seen.is_set())
                self.assertIs(self.disk.pending_states.pop(self.key), stash)
                self.cache.clear()

    def test_failed_fill_never_described_or_enqueued(self):
        stash = self.pending_pair()
        stash["target"]._pm_ready.set()
        stash["draft"]._pm_error = OSError("fill")
        stash["draft"]._pm_ready.set()
        with self.assertLogs("exllamav3.generator.disk_cache", level="WARNING"):
            self.assertFalse(self.disk.write_state(self.key, stash, force=True))
        self.assertFalse(self.disk._queue)
        self.assertEqual(self.disk.total_size, 0)

    def test_pipeline_rekey_waits_for_settlement_and_drops_failure(self):
        for fails in (False, True):
            with self.subTest(fails=fails):
                stash = {"position": 256, "checkpoint_size": 64}
                self.cache[self.old] = stash
                self.cache.unwritten = [(self.old, stash)]
                self.cache.pipeline = NS(failed_stashes={id(stash)} if fails else set())
                with patch.object(self.disk, "note_state", self.disk.write_state):
                    self.cache.rekey(self.old, self.key)
                    self.assertFalse(self.disk.pending_states)
                    self.cache.write_settled()
                self.assertEqual(self.key in self.cache, not fails)
                self.assertEqual(self.key in self.disk.pending_states, not fails)
                self.disk.pending_states.clear()
                self.cache.clear()


class Contracts(unittest.TestCase):
    def test_split_boundaries_validated_before_loading(self):
        modules = [NS(key=k, caps={}) for k in ("layers.24", "layers.25", "layers.26")]
        for value, devices, expected in [
            ("", [0, 1], {}),
            ("layers.25", [0, 1], {"layers.25": 1}),
            ("layers.25", [0], None),
            ("layers.25,layers.25", [0, 1, 2], None),
            ("layers.26,layers.25", [0, 1, 2], None),
            ("other.component", [0], {}),
        ]:
            with self.subTest(value=value), patch.dict(os.environ, EXL3_LS_SPLIT_BEFORE=value):
                if expected is None:
                    with self.assertRaisesRegex(ValueError, "EXL3_LS_SPLIT_BEFORE"):
                        _split_boundaries(modules, devices)
                else:
                    self.assertEqual(_split_boundaries(modules, devices), expected)

    def test_names_default_off_and_native_root_byte_compatible(self):
        P = name_policy
        with patch.dict(os.environ, clear=True):
            self.assertIs(P.resolve(P.RAW, P.RAW), P.NATIVE)
            self.assertEqual(P.advertised(P.RAW), ())
            with self.assertRaises(ValueError):
                P.resolve(P.FAST, P.RAW)
        with patch.dict(os.environ, EXL3_FLASHNEXT_FAST="1"):
            fast = P.resolve(P.FAST, P.RAW)
            self.assertEqual(P.advertised(P.RAW), (P.FAST,))
            with self.assertRaises(ValueError):
                P.resolve(P.FAST, "other")
        ids = torch.arange(256).reshape(1, -1)
        root = P.content_hash(ids, None)
        self.assertEqual(root, hashlib.blake2b(ids.numpy().tobytes(), digest_size=16).digest())
        fast_root = P.content_hash(ids, None, fast.cache_domain)
        self.assertNotEqual(root, fast_root)
        self.assertEqual(P.content_hash(ids, fast_root, fast.cache_domain), P.content_hash(ids, fast_root))
        self.assertEqual(Sequence(ids, ids, fast.cache_domain).cache_domain, fast.cache_domain)

    def test_fast_rescue_is_single_and_requires_finite_control(self):
        native, proposals = [1, 2, 7, 4], [1, 3, 7]
        # Native first rejection is the one allowed rescue. The next position stays native.
        allowed = [True, True, True]
        self.assertEqual(fast_accept.choose(native, proposals, allowed, 1), (3, True))
        self.assertEqual(fast_accept.choose(native, proposals, allowed, 2), (7, False))
        self.assertEqual(fast_accept.choose(native, proposals, [False] * 3, 1), (2, False))

    def test_capability_flag_does_not_fragment_disk_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "kernel.py").write_text("pass\n")
            original = engine_identity(tmp, {})
            self.assertEqual(original, engine_identity(tmp, {"EXL3_FLASHNEXT_FAST": "1"}))
            self.assertNotEqual(original, engine_identity(tmp, {"EXL3_BATCH_VERIFY": "1"}))


if __name__ == "__main__":
    unittest.main()
