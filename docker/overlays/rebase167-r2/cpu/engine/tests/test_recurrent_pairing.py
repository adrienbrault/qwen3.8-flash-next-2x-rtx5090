"""rebase165: recurrent-state glue between the served prefill stack and 1.6.5's PairedState. CPU torch only (skipped
without torch); modules are imported without the package __init__ (no compiled extension).

- HostPool: pools only storage-owning tensors, and at most as many idle buffers per key as take() created
- prefill_merge.target_state / _paired: captures use the target side; a recurrent draft side disables them; a
  published snapshot has PairedState.stash()'s layout
- prefill_merge.stash_layout refuses upstream's two-rows-per-slot GDN layout
- RecurrentCache.put accepts the served trace keywords
"""
import types
import unittest

from _iso import have_torch, load, stub_module, stub_packages

HAVE_TORCH = have_torch()


def _modules():
    stub_packages("exllamav3.util", "exllamav3.cache")
    stub_module("exllamav3.util.memory", malloc_trim=lambda: None)
    rec = load("exllamav3.cache.recurrent")
    pm = load("exllamav3.cache.prefill_merge")
    return rec, pm


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class HostPoolTests(unittest.TestCase):

    def setUp(self):
        import torch
        self.torch = torch
        self.rec, _ = _modules()

    def test_take_give_reuse(self):
        pool = self.rec.HostPool()
        a = pool.take((4, 8), self.torch.float16)
        pool.give({"position": 3, "checkpoint_size": 1, "layers": [a]})
        b = pool.take((4, 8), self.torch.float16)
        self.assertIs(a, b)
        self.assertEqual((pool.allocated, pool.reused), (1, 1))

    def test_foreign_buffers_are_not_accumulated(self):
        pool = self.rec.HostPool()
        mine = pool.take((16,), self.torch.float32)
        foreign = [self.torch.empty((16,)) for _ in range(5)]
        pool.give(foreign + [mine])
        self.assertEqual(len(pool.free[((16,), self.torch.float32)]), 1)

    def test_views_are_never_pooled(self):
        pool = self.rec.HostPool()
        base = pool.take((2, 16), self.torch.float32)
        pool.give(base[0])
        self.assertEqual(pool.free.get(((16,), self.torch.float32), []), [])

    def test_release_drops_idle_and_their_live_count(self):
        pool = self.rec.HostPool()
        a = pool.take((8,), self.torch.int32)
        b = pool.take((8,), self.torch.int32)
        pool.give([a])
        pool.release()
        key = ((8,), self.torch.int32)
        self.assertEqual(pool.live[key], 1)        # b is still out
        pool.give([b])
        self.assertEqual(len(pool.free[key]), 1)
        pool.give([self.torch.empty((8,), dtype = self.torch.int32)])
        self.assertEqual(len(pool.free[key]), 1)

    def test_put_accepts_served_trace_keywords(self):
        import inspect
        params = inspect.signature(self.rec.RecurrentCache.put).parameters
        self.assertIn("trace_owner", params)
        self.assertIn("trace_reason", params)


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class PrefillMergePairing(unittest.TestCase):

    def setUp(self):
        import torch
        self.torch = torch
        self.rec, self.pm = _modules()

    def test_target_state(self):
        t = object()
        self.assertIsNone(self.pm.target_state(types.SimpleNamespace(recurrent_state = None)))
        self.assertIs(self.pm.target_state(types.SimpleNamespace(recurrent_state = self.rec.PairedState(t, None))), t)
        self.assertIsNone(self.pm.target_state(
            types.SimpleNamespace(recurrent_state = self.rec.PairedState(t, object()))))
        # served (unpaired) state objects are passed through
        self.assertIs(self.pm.target_state(types.SimpleNamespace(recurrent_state = t)), t)

    def test_paired_layout_matches_pairedstate_stash(self):
        inner = {"position": 4096, "checkpoint_size": 123, "layer.0": "x"}
        out = self.pm._paired(inner)
        self.assertEqual(set(out), {"position", "checkpoint_size", "target"})
        self.assertIs(out["target"], inner)
        self.assertEqual((out["position"], out["checkpoint_size"]), (4096, 123))

        class S:
            def __init__(self, s):
                self.s = s
                self.position = s["position"]
                self.checkpoint_size = s["checkpoint_size"]

            def stash(self):
                return self.s

        paired = self.rec.PairedState(S(inner), None).stash()
        self.assertEqual(set(paired), set(out))
        self.assertIs(paired["target"], inner)
        self.assertEqual((paired["position"], paired["checkpoint_size"]), (out["position"], out["checkpoint_size"]))

    def _two_row_layer(self, bsz = 3):
        """GDNLayerState stand-in with 1.6.5's two-row pool and NATIVE's stash_views (wt-native gated_delta_net.py)"""
        torch = self.torch

        class GDN:
            def stash_views(self, slot, parity = 0):
                row = 2 * slot + parity
                return self.recurrent_state[row:row + 1], self.conv_state[slot, :, :self.module.conv_kernel_size]

        class GDNMod:
            conv_kernel_size = 4

        l = GDN()
        l.module = GDNMod()
        l.recurrent_state = torch.empty((2 * bsz, 1, 2, 4, 4), dtype = torch.bfloat16)
        for r in range(2 * bsz):
            l.recurrent_state[r].fill_(float(r))
        l.conv_state = torch.empty((bsz, 8, 4 + 3), dtype = torch.bfloat16)
        for b in range(bsz):
            l.conv_state[b].fill_(100.0 + b)
        return GDN, GDNMod, l

    def test_two_row_pool_flag(self):
        self.assertTrue(self.pm.TWO_ROW_POOL)

    def test_capture_reads_the_live_row(self):
        torch = self.torch
        self.pm._gdn_capability = True
        GDN, GDNMod, l = self._two_row_layer()
        orig = self.pm._layer_types
        self.pm._layer_types = lambda: (GDN, GDNMod, type("P", (), {}), type("PM", (), {}))
        try:
            cache = types.SimpleNamespace(get_all_recurrent_layers = lambda: {"l0": l})
            lay = self.pm.stash_layout(cache)
            self.assertIsNotNone(lay)
            (key, _, kind, parts), = lay[0]
            # the stash shapes are exactly what GDNState.stash() copies
            self.assertEqual([p[1] for p in parts], [tuple(v.shape) for v in l.stash_views(0, 0)])

            class Pool:
                def release(self, slab, events = None):
                    pass

            for slot in range(3):
                for parity in (0, 1):
                    slab = torch.zeros(lay[1], dtype = torch.uint8)
                    cap = self.pm.Capture(lay, slot, slab, Pool(), parity = parity)
                    cap.capture(l)
                    self.assertTrue(cap.complete())
                    st = cap.finish(4096, 1, asynchronous = False)
                    rs, cs = st[key]
                    self.assertEqual(float(rs.float().mean()), float(2 * slot + parity))
                    self.assertEqual(float(cs.float().mean()), 100.0 + slot)
                    self.assertEqual((rs.dtype, cs.dtype), (torch.bfloat16, torch.bfloat16))
                    # a stash restores into the same row unstash() writes
                    ref_rs, ref_cs = l.stash_views(slot, parity)
                    self.assertTrue(torch.equal(rs, ref_rs) and torch.equal(cs, ref_cs))
        finally:
            self.pm._layer_types = orig

    def test_stash_layout_refuses_a_gdn_state_without_stash_views(self):
        torch = self.torch
        self.pm._gdn_capability = True

        class GDN:
            pass

        class GDNMod:
            conv_kernel_size = 4

        l = GDN()
        l.module = GDNMod()
        l.recurrent_state = torch.empty((6, 1, 2, 4, 4), device = "meta")
        l.conv_state = torch.empty((3, 8, 4), device = "meta")
        orig = self.pm._layer_types
        self.pm._layer_types = lambda: (GDN, GDNMod, type("P", (), {}), type("PM", (), {}))
        try:
            self.assertIsNone(self.pm.stash_layout(types.SimpleNamespace(get_all_recurrent_layers = lambda: {"l0": l})))
        finally:
            self.pm._layer_types = orig

    def test_callers_pass_the_target_parity(self):
        import ast
        from _iso import text
        tree = ast.parse(text("exllamav3/cache/prefill_merge.py"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Capture"]
        self.assertEqual(len(calls), 4)
        for c in calls:
            kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
            self.assertIn(kw.get("parity"), ("rs.parity", "state.parity"))


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class CapabilityGate(unittest.TestCase):
    """Daily flags with the GDN capture capability absent: nothing is reserved, the flags are inert"""

    DAILY = {"EXL3_LS_PREFILL_PIPELINE": "1", "EXL3_PREFILL_MERGE": "1", "EXL3_STASH_ASYNC": "1",
             "EXL3_RECURRENT_CHECKPOINT_INFORWARD": "1", "EXL3_PREFILL_WHOLE_PROMPT": "1",
             "EXL3_PREFILL_RESUMABLE": "1"}

    def setUp(self):
        import os
        self.os = os
        self.env = dict(os.environ)
        os.environ.update(self.DAILY)
        self.rec, self.pm = _modules()
        stub_packages("exllamav3.modules")
        self.gdn = stub_module("exllamav3.modules.gated_delta_net")
        self.pm._gdn_capability = None
        self.pm._disabled_reason = None

    def tearDown(self):
        self.os.environ.clear()
        self.os.environ.update(self.env)
        self.pm._gdn_capability = None

    def job(self, rs):
        gen = types.SimpleNamespace(model = types.SimpleNamespace(loaded_tp = False))
        seq = types.SimpleNamespace(kv_position = 0)
        return types.SimpleNamespace(recurrent_state = self.rec.PairedState(rs, None), generator = gen,
                                     sequences = [seq], embeddings = [], _ls_prefill_pipeline = None)

    def test_absent_capability_declines_every_reservation(self):
        self.assertFalse(self.pm.gdn_capture_supported())
        self.assertFalse(self.pm.merge_enabled())
        self.assertFalse(self.pm.tail_capture_enabled())
        rs = types.SimpleNamespace(position = 0, slot = 0, parity = 0, exported = False,
                                   cache = types.SimpleNamespace(get_all_recurrent_layers = lambda: {}))
        job = self.job(rs)
        self.assertIsNone(self.pm.try_split(job, 0, 4096 + 100, 4096 + 100))
        self.assertIsNone(self.pm.try_boundary(job, 2048, 4096))
        self.assertIsNone(self.pm.try_pipeline_capture(job, 2048, 4096))

    def test_query_present_enables(self):
        self.gdn.prefill_capture_supported = lambda: True
        self.pm._gdn_capability = None
        self.assertTrue(self.pm.gdn_capture_supported())
        self.assertTrue(self.pm.merge_enabled())
        self.assertTrue(self.pm.tail_capture_enabled())

    def test_stash_layout_declines_without_capability(self):
        torch = __import__("torch")

        class GDN:
            def stash_views(self, slot, parity = 0):
                return self.recurrent_state[0:1], self.conv_state[0, :, :4]

        class GDNMod:
            conv_kernel_size = 4

        l = GDN()
        l.module = GDNMod()
        l.recurrent_state = torch.empty((2, 1, 2, 4, 4), device = "meta")
        l.conv_state = torch.empty((1, 8, 7), device = "meta")
        orig = self.pm._layer_types
        self.pm._layer_types = lambda: (GDN, GDNMod, type("P", (), {}), type("PM", (), {}))
        try:
            cache = types.SimpleNamespace(get_all_recurrent_layers = lambda: {"l0": l})
            self.assertIsNone(self.pm.stash_layout(cache))
        finally:
            self.pm._layer_types = orig


if __name__ == "__main__":
    unittest.main()
