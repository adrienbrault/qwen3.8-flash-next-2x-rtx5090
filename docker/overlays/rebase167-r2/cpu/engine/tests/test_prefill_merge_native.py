"""rebase165 NATIVE R1 against the real GDNLayerState (two-row pool, stash_views): prefill_merge.Capture copies the same
row and conv head as GDNLayerState.stash(slot, parity), and the copy restores through unstash. CPU only; imports the
integrated package from EXL3_SRC (real extension when importable, else a no-op stub; nothing launches)."""
import types
import unittest

from _iso import have_torch, import_package

gdn = pm = torch = None
if have_torch():
    try:
        import_package()
        import torch
        import exllamav3.modules.gated_delta_net as gdn
        import exllamav3.cache.prefill_merge as pm
    except Exception as e:  # noqa: BLE001
        print(f"(skipping: cannot import the package: {e!r})")
        gdn = None


def _layer(bsz = 3):
    mod = object.__new__(gdn.GatedDeltaNet)
    mod.__dict__.update(dict(fdim_qkv = 64, conv_kernel_size = 4, num_v_heads = 2, num_k_heads = 1,
                             k_head_dim = 32, v_head_dim = 32, kda = False, mamba2 = False,
                             recurrent_state_dtype = torch.bfloat16))
    l = gdn.GDNLayerState(mod, bsz, 3, 0)
    l.alloc("cpu")
    for r in range(l.recurrent_state.shape[0]):
        l.recurrent_state[r].fill_(float(r))
    for b in range(bsz):
        l.conv_state[b].fill_(100.0 + b)
    return l


class _Pool:
    def release(self, slab, events = None):
        pass


@unittest.skipIf(gdn is None, "integrated exllamav3 package not importable")
class CaptureVsLayerStash(unittest.TestCase):

    def setUp(self):
        self._host_copy = gdn.host_copy
        gdn.host_copy = lambda t: t.clone()
        self._cap = pm._gdn_capability
        pm._gdn_capability = True       # the module side is exercised directly here

    def tearDown(self):
        gdn.host_copy = self._host_copy
        pm._gdn_capability = self._cap

    def test_two_row_pool_flag_is_set(self):
        self.assertIs(pm.TWO_ROW_POOL, True)

    def test_capture_equals_layer_stash(self):
        for slot in (0, 2):
            for parity in (0, 1):
                with self.subTest(slot = slot, parity = parity):
                    l = _layer()
                    cache = types.SimpleNamespace(get_all_recurrent_layers = lambda: {"l0": l})
                    lay = pm.stash_layout(cache)
                    self.assertIsNotNone(lay)
                    cap = pm.Capture(lay, slot, torch.zeros(lay[1], dtype = torch.uint8), _Pool(), parity = parity)
                    cap.capture(l)
                    self.assertTrue(cap.complete())
                    got = cap.finish(4096, 1, asynchronous = False)["l0"]
                    ref = l.stash(slot, 0, parity)
                    self.assertEqual(len(got), len(ref))
                    for a, b in zip(got, ref):
                        self.assertEqual((a.shape, a.dtype), (b.shape, b.dtype))
                        self.assertTrue(torch.equal(a, b))
                    self.assertEqual(float(got[0].float().mean()), float(2 * slot + parity))
                    l.unstash(1, got, 0, 1 - parity)
                    self.assertEqual(float(l.recurrent_state[2 + 1 - parity].float().mean()), float(2 * slot + parity))



DAILY = {"EXL3_LS_PREFILL_PIPELINE": "1", "EXL3_PREFILL_MERGE": "1", "EXL3_STASH_ASYNC": "1",
         "EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP": "4096", "EXL3_RECURRENT_CHECKPOINT_TAIL_PP": "12288",
         "EXL3_RECURRENT_CHECKPOINT_INFORWARD": "1", "EXL3_PREFILL_WHOLE_PROMPT": "1", "EXL3_PREFILL_RESUMABLE": "1"}


@unittest.skipIf(gdn is None, "integrated exllamav3 package not importable")
class DailyFlagsCapabilityGate(unittest.TestCase):
    """
    Daily prefill flags with the GDN module's capture capability absent (codex NATIVE review finding 1): the planner
    must reserve nothing, so no "R823c incomplete chunk snapshot" can be raised; the flags degrade to the served A
    path (window cuts at checkpoints, synchronous stashes). With the capability present the same planner reserves.
    """

    def setUp(self):
        import os
        self.os = os
        self.env = dict(os.environ)
        os.environ.update(DAILY)
        self.cap = pm._gdn_capability
        self.query = getattr(gdn, pm.GDN_CAPTURE_QUERY, None)
        if hasattr(gdn, pm.GDN_CAPTURE_QUERY):
            delattr(gdn, pm.GDN_CAPTURE_QUERY)
        pm._gdn_capability = None
        pm._disabled_reason = None
        import exllamav3.generator.prefill_pipeline as pp
        self.pp = pp

    def tearDown(self):
        self.os.environ.clear()
        self.os.environ.update(self.env)
        if self.query is not None:
            setattr(gdn, pm.GDN_CAPTURE_QUERY, self.query)
        elif hasattr(gdn, pm.GDN_CAPTURE_QUERY):
            delattr(gdn, pm.GDN_CAPTURE_QUERY)
        pm._gdn_capability = self.cap

    def runtime(self, l, prompt_end = 3 * 4096 + 100):
        rs = types.SimpleNamespace(position = 0, slot = 0, parity = 0, exported = False, checkpoint_size = 1,
                                   cache = types.SimpleNamespace(get_all_recurrent_layers = lambda: {"l0": l}))
        lay = pm.stash_layout(rs.cache)
        if lay is not None:
            # pageable slabs: pin_memory needs a CUDA driver, absent in the CPU test container
            rs.cache._pm_pool = pm.StagingPool(lay[1], 2, alloc = lambda n: torch.empty(n, dtype = torch.uint8))
        gen = types.SimpleNamespace(model = types.SimpleNamespace(loaded_tp = False), max_chunk_size = 2048,
                                    recurrent_checkpoint_interval = 2048, recurrent_checkpoint_interval_pp = 4096,
                                    recurrent_checkpoint_tail_pp = 0)
        seq = types.SimpleNamespace(kv_position = 0, sequence_ids = [0] * (prompt_end + 1))
        from exllamav3.cache.recurrent import PairedState
        job = types.SimpleNamespace(recurrent_state = PairedState(rs, None), generator = gen, sequences = [seq],
                                    embeddings = [], cached_pages = 0, _ls_prefill_pipeline = None)
        ends = [2048, 4096, 6144, 8192]
        rt = types.SimpleNamespace(job = job, ends = list(ends), whole_prompt = self.pp.whole_prompt_enabled(),
                                   prepared = set(), merges = {}, captures = {})
        return rt

    def plan(self, rt):
        reserved = []
        for index in range(len(rt.ends)):
            if index >= len(rt.ends):
                break
            params = self.pp._Runtime._capture_params(rt, index, {})
            if "_prefill_tail_capture" in params or "_prefill_split" in params:
                reserved.append(index)
        return reserved

    def test_capability_absent_reserves_nothing(self):
        self.assertFalse(pm.gdn_capture_supported())
        self.assertFalse(pm.merge_enabled())
        self.assertFalse(pm.tail_capture_enabled())
        self.assertFalse(self.pp.whole_prompt_enabled())
        rt = self.runtime(_layer())
        self.assertEqual(self.plan(rt), [])
        self.assertEqual((rt.captures, rt.merges), ({}, {}))

    def test_capability_present_reserves_at_checkpoints(self):
        setattr(gdn, pm.GDN_CAPTURE_QUERY, lambda: True)
        pm._gdn_capability = None
        self.assertTrue(self.pp.whole_prompt_enabled())
        l = _layer()
        rt = self.runtime(l)
        reserved = self.plan(rt)
        self.assertTrue(reserved)
        for split in list(rt.captures.values()) + list(rt.merges.values()):
            self.assertEqual(split.cap.parity, 0)
            split.cap.abandon()


if __name__ == "__main__":
    unittest.main()
