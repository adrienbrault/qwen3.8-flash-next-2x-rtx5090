#!/usr/bin/env python3
"""CPU tests of the TP=2 bounding probe (run in the workspace .venv: torch CPU + the real exllamav3 sources in src/).

    cd out/tp-bound-r1/tests && ../../../.venv/bin/python -m unittest -v test_cpu.py

1. TestExl3Slicing     EXL3 column / row slices made by the REAL LinearEXL3.tp_import_split(_n) reconstruct exactly the
                       columns / rows of the full layer's effective weight (real get_weight_tensor: 128-block Hadamards
                       + sign vectors; the tile decoder is a tile-local stand-in, reconstruct.cu decodes each 16x16 tile
                       from its own packed words only). A 64-aligned split is shown to FAIL (the test sees the Hadamard
                       block structure). Row-split partial products sum to the full product.
2. TestExtBindings     the signature database parsed from bindings.cpp / *_bc.h agrees with known facts.
3. TestDryRun          the whole GPU probe runs on the meta device over a synthetic checkpoint with served tensor shapes,
                       every exllamav3 / extension call bound against src/ (dryrun_probe.py); served BC paths used by
                       full AND half modules; every EXL3 slice the TP import makes is 128-aligned; the GDN half norm is
                       rebuilt with the sigmoid gate.
4. TestJoin            family rules cover R519, arithmetic identities of the budget, decision line, AR count, pick_ar.
5. TestProbeHelpers    kernel-name normalization, trace parsing, bench() host-ahead logic, plan arithmetic.
6. TestScripts         shell syntax, the operator command, EXTRA_ENV == the served launcher's, NCCL transport parser,
                       data/src_sha256.json matches src/ (written when missing).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import dryrun_probe  # noqa: E402

EXT = dryrun_probe.setup()          # stubs installed before anything imports exllamav3

import torch  # noqa: E402
import tpb_common as C  # noqa: E402
import tp_bound_gpu as G  # noqa: E402
import tp_bound_join as J  # noqa: E402
import tp_bound_ar as AR  # noqa: E402
from ext_sigs import ExtSigs  # noqa: E402

ROOT = os.path.dirname(HERE)
WS = os.path.abspath(os.path.join(ROOT, "..", ".."))
SRC = os.path.join(WS, "src", "exllamav3")
DATA = os.path.join(ROOT, "data")


# ----------------------------------------------------------------------------------------------------------------------
# 1. EXL3 slicing
# ----------------------------------------------------------------------------------------------------------------------

def _tile_decoder(seed=7):
    """Tile-local stand-in for the trellis decode: tile (k, n) -> 16x16 values from trellis[k, n, :] only."""
    proj = {}

    def decode(trellis, K):
        if K not in proj:
            g = torch.Generator().manual_seed(seed + K)
            proj[K] = torch.randn((16 * K, 256), generator=g, dtype=torch.float64) / (16 * K) ** 0.5
        kt, nt, _ = trellis.shape
        v = (trellis.to(torch.float64) / 4096.0) @ proj[K]             # (kt, nt, 256)
        return v.view(kt, nt, 16, 16).permute(0, 2, 1, 3).reshape(kt * 16, nt * 16)
    return decode


class TestExl3Slicing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.decode = _tile_decoder()
        # ext.reconstruct(w, trellis, K, mcg, mul1) fills w (in, out) with the decoded inner weight
        EXT.returns["reconstruct"] = lambda w, trellis, K, mcg, mul1: w.copy_(cls.decode(trellis, K).to(w.dtype))
        from exllamav3.modules.quant.exl3 import LinearEXL3
        cls.LinearEXL3 = LinearEXL3

    def make(self, inf, outf, K, seed):
        g = torch.Generator().manual_seed(seed)
        trellis = torch.randint(-32768, 32767, (inf // 16, outf // 16, 16 * K), generator=g, dtype=torch.int16)
        suh = (torch.randint(0, 2, (inf,), generator=g) * 2 - 1).half()
        svh = (torch.rand((outf,), generator=g) * 0.5 + 0.75).half()
        return self.LinearEXL3(None, inf, outf, suh=suh, svh=svh, trellis=trellis)

    def half(self, full, split):
        ctx = C.local_context(torch.device("cpu"))
        exported = full.tp_export({}, C.StubProducer())
        return self.LinearEXL3.tp_import_split(ctx, exported, {}, split)

    def weight(self, lin):
        return lin.get_weight_tensor().double()

    def assert_close(self, a, b, what):
        err = float((a - b).abs().max())
        scale = float(a.abs().max())
        self.assertLess(err, 2e-3 * scale, f"{what}: max err {err} (scale {scale})")

    def test_column_split_selects_columns(self):
        # (in, out, K, split points): lm_head-like, attention q, GDN z, expert up (640 -> 384 | 256)
        for inf, outf, K, cut in ((2560, 2048, 4, 1024), (2560, 1536, 4, 768), (1024, 640, 2, 384),
                                  (1024, 640, 3, 384)):
            full = self.make(inf, outf, K, seed=inf + outf + K)
            W = self.weight(full)
            for first, last in ((0, cut), (cut, outf)):
                h = self.half(full, (True, first, last))
                self.assertEqual((h.in_features, h.out_features), (inf, last - first))
                self.assert_close(W[:, first:last], self.weight(h), f"col {first}:{last} of {inf}x{outf} K{K}")

    def test_row_split_selects_rows_and_partials_sum(self):
        for inf, outf, K, cut in ((6144, 1024, 4, 3072), (640, 1024, 2, 384), (640, 1024, 3, 384)):
            full = self.make(inf, outf, K, seed=3 * inf + outf)
            W = self.weight(full)
            x = torch.randn((4, inf), dtype=torch.float64)
            parts = []
            for first, last in ((0, cut), (cut, inf)):
                h = self.half(full, (False, first, last))
                self.assertEqual((h.in_features, h.out_features), (last - first, outf))
                Wh = self.weight(h)
                self.assert_close(W[first:last, :], Wh, f"rows {first}:{last} of {inf}x{outf} K{K}")
                parts.append(x[:, first:last] @ Wh)
            self.assert_close(x @ W, parts[0] + parts[1], "row-parallel partial sum")

    def test_split_n_sections(self):
        # GDN in_proj_qkv under the K-head plan: q | k | v sections, rank 0 takes the first half of each
        full = self.make(512, 2560, 4, seed=11)          # q 512, k 512, v 1536 (scaled-down 2048 / 2048 / 6144)
        W = self.weight(full)
        ctx = C.local_context(torch.device("cpu"))
        exported = full.tp_export({}, C.StubProducer())
        splits = [(True, 0, 256), (True, 512, 768), (True, 1024, 1792)]
        h = self.LinearEXL3.tp_import_split_n(ctx, exported, {}, splits)
        ref = torch.cat([W[:, a:b] for _, a, b in splits], dim=1)
        self.assert_close(ref, self.weight(h), "split_n q|k|v")

    def test_misaligned_split_is_detected(self):
        # a 128-wide slice that starts inside a 128-channel Hadamard block (64) does NOT give those columns, for
        # columns and for rows: the equalities above hold because the plan cuts are 128-aligned
        full = self.make(512, 512, 4, seed=5)
        W = self.weight(full)
        h = self.half(full, (True, 64, 192))
        err = float((W[:, 64:192] - self.weight(h)).abs().max())
        self.assertGreater(err, 0.05 * float(W.abs().max()))
        h = self.half(full, (False, 64, 192))
        err = float((W[64:192, :] - self.weight(h)).abs().max())
        self.assertGreater(err, 0.05 * float(W.abs().max()))

    def test_plan_cuts_are_128_aligned(self):
        rs = C.ratio_split_ref
        for units, width in ((16, 1), (2, 1), (5, 128), (1940, 128), (512, 128)):
            for a, b in C.rank_ranges(units, width, rs):
                if width == 128:
                    self.assertEqual(a % 128, 0)
                    self.assertEqual(b % 128, 0)
        self.assertEqual(C.rank_ranges(5, 128, rs), [(0, 384), (384, 640)])      # MoE / shared: 0.6 : 0.4
        self.assertEqual(C.vocab_ranges(248320, rs), [(0, 124160), (124160, 248320)])


# ----------------------------------------------------------------------------------------------------------------------
# 2. Extension bindings
# ----------------------------------------------------------------------------------------------------------------------

class TestExtBindings(unittest.TestCase):

    def test_known_signatures(self):
        s = ExtSigs(os.path.join(SRC, "exllamav3_ext"))
        self.assertEqual(s.unresolved, [])
        self.assertEqual(s.funcs["exl3_moe_coop"].n_params, 36)       # exl3_moe_coop.cuh:128-157
        self.assertEqual(s.funcs["exl3_gemm"].n_params, 10)
        init = s.classes["BC_GatedRMSNorm"]["init"].sigs[0]
        self.assertEqual((init.n_params, init.n_defaults), (6, 3))      # gated_rmsnorm_bc.h
        self.assertEqual(s.classes["BC_LinearEXL3"]["init"].sigs[0].n_params, 8)
        self.assertIn("run_bszN", s.classes["BC_BlockSparseMLP"]["methods"])
        self.assertIn("run_bszN", s.classes["BC_GatedMLP"]["methods"])
        self.assertEqual(sorted(sg.n_params for sg in s.classes["BC_GatedDeltaNetSplit"]["init"].sigs), [15, 18])

    def test_stub_rejects_unknown_and_bad_arity(self):
        with self.assertRaises(AttributeError):
            EXT.no_such_function(1)
        with self.assertRaises(TypeError):
            EXT.exl3_moe_coop(*range(35))
        with self.assertRaises(TypeError):
            EXT.BC_GatedRMSNorm(1)


# ----------------------------------------------------------------------------------------------------------------------
# 3. Dry run of the GPU probe
# ----------------------------------------------------------------------------------------------------------------------

class TestDryRun(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tpb_dry_")
        n0 = len(EXT.calls)
        cls.logs = []
        cls.res, _, cls.recv = dryrun_probe.run(os.path.join(cls.tmp, "ckpt"), log=cls.logs.append)
        cls.calls = EXT.calls[n0:]
        cls.names = [c[0] for c in cls.calls]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_probe_ran_clean(self):
        r = self.res
        self.assertEqual(r["errors"], [], "\n".join(e[:2000] for e in r["errors"]))
        self.assertEqual(r["path_problems"], [])
        self.assertEqual(r["missing_primary_families"], [])
        self.assertTrue(r["ok"])
        for fam in ("GDN", "ATTN", "MOE_K2", "MOE_K3", "MTP_ATTN", "MTP_MOE", "HC", "HEAD"):
            self.assertIn(fam, r["families"])

    def test_every_shape_timed(self):
        F = self.res["families"]
        self.assertEqual(sorted(F["GDN"]["shapes"]), sorted(G.TRUNK_SHAPES))
        self.assertEqual(sorted(F["ATTN"]["shapes"]), sorted(G.TRUNK_SHAPES + G.DRAFT_SHAPES))
        for k in ("MOE_K2", "MOE_K3"):
            self.assertEqual(sorted(F[k]["rows"], key=int), [str(r) for r in G.MOE_ROWS])
            for rows in G.MOE_ROWS:
                st = F[k]["rows"][str(rows)]
                for arm in ("full", "rank0", "rank1", "routed", "shared"):
                    self.assertIn(arm, st)
        self.assertEqual(sorted(F["MTP_ATTN"]["shapes"]), sorted(G.DRAFT_SHAPES))

    def test_served_paths_used(self):
        n = self.names
        # GDN: fused split BC for full and half, with the qkvz bundle
        self.assertGreater(n.count("BC_GatedDeltaNetSplit.run_bszN"), 0)
        self.assertGreater(n.count("BC_GatedDeltaNetSplit.set_qkvz_bundle"), 0)
        self.assertGreater(n.count("BC_Attention.run"), 0)
        self.assertGreater(n.count("BC_Attention.set_qsa"), 0)        # indexer armed (full and halves)
        self.assertGreater(n.count("BC_BlockSparseMLP.run_bszN"), 0)
        self.assertGreater(n.count("exl3_moe_coop"), 0)               # the routed-only arm (r521 call)
        self.assertGreater(n.count("BC_GatedMLP.run_bszN"), 0)
        self.assertGreater(n.count("gr_mix_v2"), 0)
        self.assertGreater(n.count("hc_apply"), 0)
        info = self.res["families"]["GDN"]["info"]
        for r in ("rank0", "rank1"):
            self.assertTrue(info[r]["bc_split"])
            self.assertEqual(info[r]["num_k_heads"], 8)
            self.assertEqual(info[r]["num_v_heads"], 24)
        ai = self.res["families"]["ATTN"]["info"]
        self.assertEqual((ai["rank0"]["num_q_heads"], ai["rank0"]["num_kv_heads"]), (12, 1))
        self.assertTrue(ai["full"]["indexer"])
        self.assertGreater(ai["full"]["sparse_threshold"], ai["full"]["ctx"])   # 1k context = dense regime, as served
        mi = self.res["families"]["MOE_K2"]["info"]
        self.assertEqual((mi["rank0"]["intermediate_size"], mi["rank1"]["intermediate_size"]), (384, 256))
        self.assertEqual((mi["rank0"]["shared_interm"], mi["rank1"]["shared_interm"]), (384, 256))
        self.assertTrue(mi["rank0"]["bc_sh_exp"] and mi["full"]["bc_sh_exp"])
        self.assertEqual(mi["rank0"]["num_local_experts"], mi["full"]["num_experts"])   # tensor split, not EP

    def test_gdn_half_norm_uses_served_sigmoid_gate(self):
        # BC_GatedRMSNorm(weight, eps, constant_bias, w_groups, gate_first, gate_act): gate_act 1 = sigmoid
        acts = [c[1][5] if len(c[1]) > 5 else c[2].get("gate_act", 0) for c in self.calls if c[0] == "BC_GatedRMSNorm"]
        self.assertIn(1, acts)
        info = self.res["families"]["GDN"]["info"]
        self.assertEqual(info["full"]["gate_activation"], "sigmoid")
        self.assertTrue(info["rank0"]["norm_gate_activation_fixed"])      # engine TP drops it (impl-status finding)

    def test_exl3_slices_are_128_aligned(self):
        n = 0
        for r in self.recv:
            if r["slice_dim"] is None or r["shape"] is None:
                continue
            if r["dtype"] == "torch.int16" and len(r["shape"]) == 3:          # trellis: units of 16 channels
                self.assertEqual(r["first"] % 8, 0, r)
                self.assertEqual(r["last"] % 8, 0, r)
                n += 1
            elif r["dtype"] == "torch.float16" and len(r["shape"]) == 1 and r["shape"][0] % 128 == 0:
                self.assertEqual(r["first"] % 128, 0, r)                     # suh / svh
                n += 1
        self.assertGreater(n, 50)

    def test_coop_call_binds_36(self):
        coop = [c for c in self.calls if c[0] == "exl3_moe_coop"]
        self.assertTrue(coop)
        self.assertTrue(all(len(c[1]) == 36 for c in coop))

    def test_steps_recorded(self):
        r = self.res
        self.assertEqual(r["requested_steps"], list(G.STEP_ORDER))
        self.assertEqual({k: v["status"] for k, v in r["steps"].items()}, {k: "ok" for k in G.STEP_ORDER})
        self.assertEqual(r["incomplete_steps"], [])
        self.assertEqual(r["respawns"], [])
        for fams in G.STEP_FAMILIES.values():
            for f in fams:
                self.assertTrue(r["families"][f]["complete"], f)

    def test_moe_split_recorded(self):
        for k in ("MOE_K2", "MOE_K3", "MTP_MOE"):
            sp = self.res["families"][k]["info"]["split"]
            self.assertEqual(sp["moe"], {"rank0": [0, 384], "rank1": [384, 640]}, k)
            self.assertEqual(sp["shared"], {"rank0": [0, 384], "rank1": [384, 640]}, k)
            self.assertEqual((sp["critical_rank"], sp["critical_share"], sp["unit_channels"]), ("rank0", 0.6, 128))
        self.assertEqual(self.res["moe_layers_tried"], {"K2": [[0, 2]], "K3": [[0, 2], [1, 3]]}
                         if isinstance(self.res["moe_layers_tried"]["K2"][0], list) else
                         {"K2": [(0, 2)], "K3": [(0, 2), (1, 3)]})

    def test_shared_experts_prewarmed_before_overlap(self):
        """R527 root cause guard: every shared expert (full + both halves) runs once per row count on the current
        stream before the first overlapped BC_BlockSparseMLP call, so its lazily allocated scratch is never owned by
        the side stream that ~BC_BlockSparseMLP destroys."""
        n = self.names
        starts = [i for i, c in enumerate(n) if c == "BC_BlockSparseMLP.run_bszN"
                  and (i == 0 or n[i - 1] != "BC_BlockSparseMLP.run_bszN")]
        first_moe = n.index("BC_BlockSparseMLP.run_bszN")
        pre = n[:first_moe].count("BC_GatedMLP.run_bszN")
        self.assertGreaterEqual(pre, 3 * len(G.MOE_ROWS))
        self.assertTrue(starts)
        for fam in ("MOE_K2", "MOE_K3", "MTP_MOE"):
            self.assertIn("allocator", self.res["families"][fam])

    def test_no_empty_cache_in_probe(self):
        with open(os.path.join(HERE, "tp_bound_gpu.py")) as f:
            tree = ast.parse(f.read())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "empty_cache"]
        self.assertEqual(calls, [], "torch.cuda.empty_cache() after an overlap BC unload raised in R527")


class TestIsolation(unittest.TestCase):
    """One family failing records an error for that family and the rest still run (in-process without a spawner,
    in a fresh process with one); ok / errors / steps are always written."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tpb_iso_")
        self.ckpt = os.path.join(self.tmp, "ckpt")
        self.orig = (G.Probe.prewarm_shared, G.Probe.run_hc)

    def tearDown(self):
        G.Probe.prewarm_shared, G.Probe.run_hc = self.orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fail_k2(self, times):
        """Raise inside run_moe_layer for the trunk K=2 layer (after its family entry exists), `times` times."""
        orig, left = self.orig[0], [times]

        def bad(probe, mlp, *a, **k):
            if mlp.multi_up.K == 2 and not mlp.key.startswith("mtp") and left[0] > 0:
                left[0] -= 1
                raise RuntimeError("CUDA error: invalid device context (injected)")
            return orig(probe, mlp, *a, **k)
        G.Probe.prewarm_shared = bad

    def _spawner(self, log):
        calls = []

        def spawn(a, steps, retried, deadline, _log):
            calls.append((list(steps), list(retried)))
            res, _, _ = dryrun_probe.run(self.ckpt, extra_args=["--families", ",".join(steps), "--retried",
                                                               ",".join(retried), "--deadline-at", repr(deadline)],
                                         log=log.append, spawn=spawn)
            return res | {"rc": 0 if res["ok"] else 1, "out": None, "elapsed_s": 0.0}
        return spawn, calls

    def test_in_process_failure_isolated(self):
        self._fail_k2(99)

        def hc_fails(probe, modules):
            raise RuntimeError("HC exploded (injected)")
        G.Probe.run_hc = hc_fails
        res, _, _ = dryrun_probe.run(self.ckpt)
        self.assertFalse(res["ok"])
        self.assertEqual(res["missing_primary_families"], ["MOE_K2"])
        self.assertIn("MOE_K2", res["families"])                   # the stub entry does not count as measured
        self.assertFalse(res["families"]["MOE_K2"].get("complete"))
        st = {k: v["status"] for k, v in res["steps"].items()}
        self.assertEqual(st, {"GDN": "ok", "ATTN": "ok", "MOE_K2": "failed", "MOE_K3": "ok", "MTP": "ok",
                              "HC": "failed", "HEAD": "ok"})
        self.assertEqual(sorted(res["incomplete_steps"]), ["HC", "MOE_K2"])
        self.assertTrue(any(e.startswith("MOE_K2 failed") and "invalid device context" in e for e in res["errors"]))
        self.assertTrue(any(e.startswith("HC failed") for e in res["errors"]))
        for f in ("GDN", "ATTN", "MOE_K3", "MTP_MOE", "HEAD"):
            self.assertTrue(res["families"][f]["complete"], f)

    def test_secondary_failure_keeps_ok(self):
        def hc_fails(probe, modules):
            raise RuntimeError("HC exploded (injected)")
        G.Probe.run_hc = hc_fails
        res, _, _ = dryrun_probe.run(self.ckpt)
        self.assertTrue(res["ok"])                                  # HC is a sensitivity family
        self.assertEqual(res["incomplete_steps"], ["HC"])
        self.assertTrue(res["errors"])

    def test_respawn_retries_failed_family_once_and_merges(self):
        self._fail_k2(1)
        logs = []
        spawn, calls = self._spawner(logs)
        res, _, _ = dryrun_probe.run(self.ckpt, log=logs.append, spawn=spawn)
        self.assertEqual(calls, [(["MOE_K2", "MOE_K3", "MTP", "HC", "HEAD"], ["MOE_K2"])])
        self.assertTrue(res["ok"], res["missing_primary_families"])
        self.assertEqual(res["missing_primary_families"], [])
        self.assertEqual(res["steps"]["MOE_K2"]["status"], "ok")
        self.assertEqual(res["steps"]["MOE_K2"]["earlier_attempt"]["status"], "failed")
        self.assertEqual(len(res["respawns"]), 1)
        self.assertTrue(res["families"]["MOE_K2"]["complete"])
        self.assertTrue(any(e.startswith("MOE_K2 failed") for e in res["errors"]))
        self.assertEqual({k for k, v in res["steps"].items() if v["status"] == "ok"}, set(G.STEP_ORDER))

    def test_respawn_does_not_retry_twice(self):
        self._fail_k2(99)
        logs = []
        spawn, calls = self._spawner(logs)
        res, _, _ = dryrun_probe.run(self.ckpt, log=logs.append, spawn=spawn)
        self.assertEqual(calls, [(["MOE_K2", "MOE_K3", "MTP", "HC", "HEAD"], ["MOE_K2"]),
                                 (["MOE_K3", "MTP", "HC", "HEAD"], ["MOE_K2"])])
        self.assertFalse(res["ok"])
        self.assertEqual(res["missing_primary_families"], ["MOE_K2"])
        self.assertEqual(len(res["respawns"]), 2)
        for f in ("GDN", "ATTN", "MOE_K3", "MTP_MOE", "HC", "HEAD"):
            self.assertTrue(res["families"][f]["complete"], f)
        self.assertEqual(res["steps"]["MOE_K2"]["status"], "failed")

    def test_main_writes_verdict_when_setup_fails(self):
        from unittest import mock
        out = os.path.join(self.tmp, "sc.json")
        with mock.patch.dict(os.environ, G.REQUIRED_ENV):
            rc = G.main(["--model", os.path.join(self.tmp, "nope"), "--device", "meta", "--out", out])
        self.assertEqual(rc, 1)
        with open(out) as f:
            r = json.load(f)
        self.assertFalse(r["ok"])
        self.assertTrue(r["fatal"])
        self.assertFalse(r["partial"])
        self.assertEqual(r["missing_primary_families"], ["GDN", "ATTN", "MOE_K2", "MOE_K3"])

    def test_main_writes_verdict_when_env_missing(self):
        from unittest import mock
        out = os.path.join(self.tmp, "sc.json")
        env = {k: v for k, v in os.environ.items() if k not in G.REQUIRED_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            rc = G.main(["--out", out])
        self.assertEqual(rc, 2)
        with open(out) as f:
            r = json.load(f)
        self.assertFalse(r["ok"])
        self.assertIn("EXL3_", r["fatal"])

    def test_spawn_child_command_and_results(self):
        from unittest import mock
        a = G.parse_args(["--model", "/model", "--out", os.path.join(self.tmp, "tp_bound_scaling.json")])
        seen = {}

        def fake_run(cmd, timeout):
            seen["cmd"], seen["timeout"] = cmd, timeout
            out = cmd[cmd.index("--out", len(cmd) - 2) + 1]
            with open(out, "w") as f:
                json.dump({"ok": True, "pid": 42, "families": {"MOE_K2": {"complete": True, "rows": {}}}}, f)
            return subprocess.CompletedProcess(cmd, 0)
        with mock.patch.object(G.subprocess, "run", fake_run):
            child = G.spawn_child(a, ["MOE_K2", "HEAD"], ["MOE_K2"], time.time() + 300, lambda m: None)
        cmd = seen["cmd"]
        self.assertEqual(cmd[1], os.path.join(HERE, "tp_bound_gpu.py"))
        self.assertEqual(cmd[2:6], ["--model", "/model", "--out", a.out])       # original argv first ...
        tail = cmd[6:]                                                           # ... overrides last (argparse: last wins)
        self.assertEqual(tail[tail.index("--families") + 1], "MOE_K2,HEAD")
        self.assertEqual(tail[tail.index("--retried") + 1], "MOE_K2")
        self.assertTrue(tail[tail.index("--out") + 1].endswith("tp_bound_scaling.from_MOE_K2.json"))
        self.assertLessEqual(seen["timeout"], 311)
        self.assertEqual((child["rc"], child["pid"]), (0, 42))

        def boom(cmd, timeout):
            raise subprocess.TimeoutExpired(cmd, timeout)
        with mock.patch.object(G.subprocess, "run", boom):
            child = G.spawn_child(a, ["HEAD"], [], time.time() + 300, lambda m: None)
        self.assertEqual(child["rc"], "timeout")
        self.assertTrue(child["errors"])
        self.assertIn("no time left", G.spawn_child(a, ["HEAD"], [], time.time() + 5, lambda m: None)["errors"][0])

    def test_steps_parse(self):
        self.assertEqual(G.parse_steps("HEAD,MOE,GDN"), ["GDN", "MOE_K2", "MOE_K3", "HEAD"])
        with self.assertRaises(SystemExit):
            G.parse_steps("MOE_K4")


# ----------------------------------------------------------------------------------------------------------------------
# 4. Join
# ----------------------------------------------------------------------------------------------------------------------

def _load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def synthetic_scaling(s=0.6, full_us=100.0, hidden_us=5.0):
    def arms(sv):
        return {"full": {"median_us": full_us}, "rank0": {"median_us": full_us * sv},
                "rank1": {"median_us": full_us * sv * 0.9}, "s": sv}
    F = {"GDN": {"shapes": {k: arms(s) for k in G.TRUNK_SHAPES}},
         "ATTN": {"shapes": {k: arms(s) for k in G.TRUNK_SHAPES + G.DRAFT_SHAPES}},
         "MTP_ATTN": {"shapes": {k: arms(s) for k in G.DRAFT_SHAPES}},
         "MTP_MOE": {"rows": {str(r): arms(s) for r in (1, 4)},
                     "overlap": {str(r): {"hidden_us": hidden_us} for r in (1, 4)}},
         "HC": {"shapes": {k: arms(0.7) for k in G.TRUNK_SHAPES + G.DRAFT_SHAPES}},
         "HEAD": {"rows": {k: arms(0.55) for k in ("lm_4", "lm_16", "draft_1", "draft_4")}}}
    split = {"unit_channels": 128, "moe_intermediate": 640, "moe": {"rank0": [0, 384], "rank1": [384, 640]},
             "shared_intermediate": 640, "shared": {"rank0": [0, 384], "rank1": [384, 640]},
             "critical_rank": "rank0", "critical_share": 0.6}
    for k in ("MOE_K2", "MOE_K3"):
        F[k] = {"rows": {str(r): arms(s) for r in G.MOE_ROWS}, "info": {"split": split}, "complete": True,
                "overlap": {str(r): {"hidden_us": hidden_us} for r in G.MOE_ROWS}}
    return {"families": F, "ok": True}


def synthetic_ar(us=10.0):
    return [{"method": "nccl", "mode": m, "rows": r, "width": w, "dtype": d, "median_us": us + (1 if m == "eager" else 0),
             "valid": True} for m in ("eager", "graph") for r in (1, 4, 8, 16) for w in (2560, 10240)
            for d in ("fp32", "fp16")]


class TestJoin(unittest.TestCase):

    def test_rules_cover_r519(self):
        for cfg in ("c1d3", "c4d3"):
            k = _load(f"r519_{cfg}_kernels.json")
            ft = J.family_totals(k)
            self.assertGreater(ft["explicit_ms"] / ft["total_ms"], 0.95, cfg)
            for name, ms in ft["unmatched"]:
                self.assertLess(ms / ft["total_ms"], 0.005, f"{cfg}: unmatched kernel above 0.5 %: {name}")
            self.assertAlmostEqual(sum(ft["families"].values()), ft["total_ms"], places=6)
            self.assertAlmostEqual(ft["total_ms"], k["total_kernel_ms"] / k["capture"]["iterations"], places=6)

    def test_every_rule_pattern_matches_something(self):
        names = [k["name"] for c in ("c1d3", "c4d3", "c6d1") for k in _load(f"r519_{c}_kernels.json")["kernels"]]
        tree = ast.parse(open(os.path.join(HERE, "tp_bound_join.py")).read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "classify")
        doc = ast.get_docstring(fn)
        pats = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value != doc and len(n.value) >= 4 and not n.value.isupper()]
        self.assertGreater(len(pats), 30)
        for p in pats:
            self.assertTrue(any(p in n for n in names), f"join rule pattern matches no R519 kernel: {p!r}")

    def test_required_kernels_are_served(self):
        served = G.served_kernel_names(DATA)
        for fam, req in G.REQUIRED_KERNEL.items():
            self.assertTrue(any(req in n for n in served), f"{fam}: {req}")

    def test_identity_s1_ar0(self):
        k = _load("r519_c1d3_kernels.json")
        ones = {f: (1.0, "t") for f in list(J.family_totals(k)["families"]) + ["GDN_REWIND"]}
        b = J.budget(k, ones, 0.0, 0.0, overlap=3.0, s_moe_eff=1.0)
        self.assertAlmostEqual(b["step_ms"], k["capture"]["elapsed_ms_per_step"], places=9)
        self.assertAlmostEqual(b["speedup"], 1.0, places=9)
        self.assertEqual(b["n_ar"], 102)

    def test_half_everything(self):
        k = _load("r519_c1d3_kernels.json")
        half = {f: (0.5, "t") for f in list(J.family_totals(k)["families"]) + ["GDN_REWIND"]}
        b = J.budget(k, half, 0.0, 0.0, overlap=0.0, s_moe_eff=0.5)
        E, K = b["E_ms"], b["K_ms"]
        self.assertAlmostEqual(b["step_ms"], K / 2 + (E - K), places=9)
        b2 = J.budget(k, half, 10.0, 10.0, overlap=0.0, s_moe_eff=0.5)
        self.assertAlmostEqual(b2["step_ms"] - b["step_ms"], 102 * 10.0 / 1000.0, places=9)
        b3 = J.budget(k, half, 0.0, 0.0, overlap=2.0, s_moe_eff=0.5)
        self.assertAlmostEqual(b3["step_ms"] - b["step_ms"], 1.0, places=9)

    def test_evaluate_and_decision(self):
        sc = synthetic_scaling(s=0.6)
        evals = {}
        for cfg in ("c1d3", "c4d3"):
            kern = _load(f"r519_{cfg}_kernels.json")
            wall = _load(f"wall_r519_{cfg}.json")
            ev = J.evaluate(cfg, kern, wall, sc, synthetic_ar(10.0))
            evals[cfg] = ev
            p = ev["primary"]
            # hand check of the primary: sum(s K) + AR + gap + overlap correction
            fam = J.family_totals(kern)["families"]
            smap = J.scaling_for(sc, J.CONFIGS[cfg])
            exp = sum(ms * smap.get(f, (1.0,))[0] for f, ms in fam.items()) + 102 * 10.0 / 1000 + p["gap_ms"] \
                + (1 - ev["s_moe_effective"]) * p["overlap_ms"]
            self.assertAlmostEqual(p["step_ms"], exp, places=9)
            self.assertEqual(ev["ar_trunk"]["mode"], "graph")
            self.assertLess(ev["bound_perfect_s05_ar0"]["step_ms"], p["step_ms"])
            self.assertIn("head_sharded", ev)
            self.assertIn("hc_sharded", ev)
            text = J.fmt_eval(ev)
            self.assertIn("PROJECTED TP step", text)
            self.assertIn("ASSUMPTION", text)
        line = J.decide(evals)
        self.assertTrue(line.startswith("DECISION: GO") or line.startswith("DECISION: NO-GO"))
        self.assertEqual(evals["c1d3"]["go"], evals["c1d3"]["best"]["speedup"] >= 1.25)
        # s = 1 everywhere cannot be a GO
        ev1 = J.evaluate("c1d3", _load("r519_c1d3_kernels.json"), None, synthetic_scaling(s=1.0),
                         synthetic_ar(10.0))
        self.assertFalse(ev1["go"])
        self.assertIn("NO-GO", J.decide({"c1d3": ev1}))

    def test_pick_ar_prefers_valid_graph(self):
        res = [{"method": "a", "mode": "eager", "rows": 4, "width": 2560, "dtype": "fp32", "median_us": 3.0,
                "valid": True},
               {"method": "b", "mode": "graph", "rows": 4, "width": 2560, "dtype": "fp32", "median_us": 5.0,
                "valid": True},
               {"method": "c", "mode": "graph", "rows": 4, "width": 2560, "dtype": "fp32", "median_us": 1.0,
                "valid": False}]
        self.assertEqual(J.pick_ar(res, 4)["method"], "b")
        self.assertIsNone(J.pick_ar(res, 8))

    def test_missing_primary_family_raises(self):
        sc = synthetic_scaling()
        del sc["families"]["MOE_K3"]
        with self.assertRaises(KeyError):
            J.evaluate("c1d3", _load("r519_c1d3_kernels.json"), None, sc, synthetic_ar())

    def test_non_p2p_file_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ar.json")
            with open(p, "w") as f:
                json.dump({"results": synthetic_ar(), "transport": {"ok": False, "kinds": {"SHM": 4}}}, f)
            notes = []
            self.assertEqual(J.load_ar([p], notes), [])
            self.assertTrue(notes)

    def test_main_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            sp, ap_, op = (os.path.join(d, n) for n in ("sc.json", "ar.json", "budget.json"))
            with open(sp, "w") as f:
                json.dump(synthetic_scaling(), f)
            with open(ap_, "w") as f:
                json.dump({"results": synthetic_ar(), "transport": {"ok": True}}, f)
            rc = J.main(["--scaling", sp, "--ar", ap_, "--out", op])
            self.assertEqual(rc, 0)
            txt = open(os.path.join(d, "budget.txt")).read()
            self.assertIn("DECISION:", txt)
            self.assertIn("MoE tensor split (128-channel units): experts 640 -> rank0 [0, 384), rank1 [384, 640)", txt)
            self.assertIn("critical rank0 holds 0.60", txt)
            self.assertIn("s = max(rank0, rank1) / full", txt)

    def test_main_incomplete_probe(self):
        """R527 shape: MOE_K2 missing -> no decision, exit 1, the probe's status printed."""
        with tempfile.TemporaryDirectory() as d:
            sp, ap_, op = (os.path.join(d, n) for n in ("sc.json", "ar.json", "budget.json"))
            sc = synthetic_scaling()
            del sc["families"]["MOE_K2"]
            sc.update({"ok": False, "missing_primary_families": ["MOE_K2"], "incomplete_steps": ["MOE_K2"],
                       "errors": ["MOE_K2 failed: RuntimeError('x')\ntrace"]})
            with open(sp, "w") as f:
                json.dump(sc, f)
            with open(ap_, "w") as f:
                json.dump({"results": synthetic_ar(), "transport": {"ok": True}}, f)
            self.assertEqual(J.main(["--scaling", sp, "--ar", ap_, "--out", op]), 1)
            txt = open(os.path.join(d, "budget.txt")).read()
            self.assertNotIn("DECISION:", txt)
            self.assertIn("missing_primary_families=['MOE_K2']", txt)
            self.assertIn("probe error: MOE_K2 failed", txt)


# ----------------------------------------------------------------------------------------------------------------------
# 5. Probe helpers
# ----------------------------------------------------------------------------------------------------------------------

class TestProbeHelpers(unittest.TestCase):

    def test_normalize(self):
        n = "void exl3_moe_coop_v2_ns::exl3_moe_coop_a_kernel<3, 2, true>(MoeCoopParams)"
        self.assertEqual(G.normalize_kernel(n), "void exl3_moe_coop_v2_ns::exl3_moe_coop_a_kernel<3, 2, true>")
        self.assertEqual(G.kernel_base(n), "exl3_moe_coop_v2_ns::exl3_moe_coop_a_kernel")
        self.assertEqual(G.normalize_kernel("_paged_attn_decode_split_kernel"), "_paged_attn_decode_split_kernel")

    def test_trace_table(self):
        tr = {"traceEvents": [{"ph": "X", "cat": "kernel", "name": "k1", "dur": 10},
                              {"ph": "X", "cat": "kernel", "name": "k1", "dur": 30},
                              {"ph": "X", "cat": "cpu_op", "name": "aten::add", "dur": 99}]}
        t = G.kernel_table_from_trace(tr, 2)
        self.assertEqual(t, {"k1": {"us": 20.0, "launches": 1.0}})

    def test_bench_host_ahead(self):
        class Gpu:
            def __init__(self, gpu_ms):
                self.gpu_ms = gpu_ms

            def sync(self):
                pass

            def sleep_ms(self, ms):
                pass

            def events(self):
                g = self

                class E:
                    def record(self):
                        pass

                    def synchronize(self):
                        pass

                    def elapsed_time(self, o):
                        return g.gpu_ms
                return E(), E()

        class Clock:
            def __init__(self, step):
                self.t, self.step = 0.0, step

            def __call__(self):
                self.t += self.step
                return self.t

        class A:
            warmup, iters, rounds, sleep_ms_min = 2, 10, 3, 10.0
        # host 1 us per clock tick -> far ahead of a 10 ms sleep
        st = G.bench(Gpu(1.0), {"full": lambda i: None, "rank0": lambda i: None}, A(), n_rot=4, clock=Clock(1e-6))
        self.assertTrue(st["full"]["host_ahead"])
        self.assertAlmostEqual(st["full"]["median_us"], 100.0)
        self.assertEqual(len(st["full"]["samples_us"]), 3)
        # a very slow host (sleep capped at 250 ms) is flagged
        st = G.bench(Gpu(1.0), {"full": lambda i: None}, A(), clock=Clock(0.5))
        self.assertFalse(st["full"]["host_ahead"])

    def test_s_from_uses_slower_rank(self):
        st = {"full": {"median_us": 100.0}, "rank0": {"median_us": 60.0}, "rank1": {"median_us": 45.0}}
        self.assertAlmostEqual(G.Probe.s_from(st), 0.6)

    def test_ratio_split_matches_engine(self):
        from exllamav3.util.misc import ratio_split
        for d in (128, 256, 640, 1280, 248320, 5 * 128, 7 * 128):
            for w in ([1, 1], [2, 1], [1, 3]):
                self.assertEqual(C.ratio_split_ref(d, w), ratio_split(d, w))
        for d in (2, 5, 16, 1940):
            self.assertEqual(C.ratio_split_ref(d, [1, 1], chunk_size=1), ratio_split(d, [1, 1], chunk_size=1))

    def test_sum_check(self):
        f = torch.randn(64)
        a = f * 0.3
        self.assertTrue(C.sum_check(f, [a, f - a])["ok"])
        self.assertFalse(C.sum_check(f, [a, a])["ok"])

    def test_attn_params_shapes(self):
        from exllamav3.modules.attn import prepare_for_attn
        p = C.attn_params(torch, prepare_for_attn, object(), 4, 4, 1024, 256)
        self.assertEqual(tuple(p["block_table"].shape), (4, C.attn_pages_per_seq(1024, 4, 256)))
        self.assertTrue(bool((p["cache_seqlens"] == 1024).all()))
        self.assertGreaterEqual(C.attn_cache_tokens(4, 1024, 4, 256), 4 * (1024 + 4))


# ----------------------------------------------------------------------------------------------------------------------
# 6. Scripts
# ----------------------------------------------------------------------------------------------------------------------

SRC_FILES = [
    "modules/gated_delta_net.py", "modules/attn.py", "modules/block_sparse_mlp.py", "modules/mlp.py",
    "modules/linear.py", "modules/quant/exl3.py", "modules/hyperconnections.py", "modules/gated_rmsnorm.py",
    "modules/qsa_indexer.py", "modules/attention_fn/bc_attn.py", "cache/qsa.py", "cache/quant.py",
    "cache/recurrent_util.py", "model/model_tp_shared.py", "model/model_tp_alloc.py", "util/misc.py",
    "architecture/qwen4_exp.py", "architecture/qwen4_exp_mtp.py", "loader/safetensors.py", "constants.py",
]


class TestScripts(unittest.TestCase):

    def test_shell_syntax(self):
        for sh in ("box_entry.sh", "run_box_probe.sh"):
            r = subprocess.run(["bash", "-n", os.path.join(HERE, sh)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_python_parses(self):
        for f in os.listdir(HERE):
            if f.endswith(".py"):
                ast.parse(open(os.path.join(HERE, f)).read(), f)

    def test_operator_wrapper(self):
        s = open(os.path.join(HERE, "run_box_probe.sh")).read()
        self.assertEqual(len(re.findall(r"^\s*sudo docker run ", s, flags=re.M)), 1)
        self.assertIn("tabbyapi:stack-r4-e3r2", s)
        self.assertIn("/probe/tests/box_entry.sh", s)
        m = re.search(r'^EXTRA_ENV=\$\{EXTRA_ENV:-"([^"]+)"\}', s, flags=re.M)     # env override, default = daily
        launcher = open(os.path.join(WS, "ref", "launch-flashnext-r517.sh")).read()
        ml = re.search(r"^EXTRA_ENV=\$\{EXTRA_ENV:-([^}]+)\}", launcher, flags=re.M)
        self.assertEqual(m.group(1).split(), ml.group(1).split())
        for k, v in G.REQUIRED_ENV.items():
            self.assertIn(f"{k}={v}", m.group(1))
        b = open(os.path.join(HERE, "box_entry.sh")).read()
        for script in ("tp_bound_p2p.py", "tp_bound_ar.py", "tp_bound_gpu.py", "tp_bound_join.py"):
            self.assertIn(script, b)
            self.assertTrue(os.path.exists(os.path.join(HERE, script)))
        self.assertIn("NCCL_P2P_LEVEL=SYS", b)
        self.assertIn("--nproc_per_node=2", b)

    def test_nccl_transport_parser(self):
        p2p = ["NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read",
               "NCCL INFO Channel 01/0 : 1[1] -> 0[0] via P2P/direct pointer"]
        self.assertEqual(AR.judge_transport(p2p), (True, {"P2P": 2}))
        self.assertFalse(AR.judge_transport(p2p + ["NCCL INFO Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct"])[0])
        self.assertFalse(AR.judge_transport([])[0])

    def test_src_hashes(self):
        """data/src_sha256.json: the files of src/ the probe was bound against (the probe warns on the box if the
        image's package differs). Written when missing; must match src/ when present."""
        cur = G.source_hashes(SRC, SRC_FILES)
        self.assertTrue(all(cur.values()), [k for k, v in cur.items() if not v])
        path = os.path.join(DATA, "src_sha256.json")
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(cur, f, indent=1, sort_keys=True)
        with open(path) as f:
            self.assertEqual(json.load(f), cur)


if __name__ == "__main__":
    unittest.main()
