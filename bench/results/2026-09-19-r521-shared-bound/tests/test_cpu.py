#!/usr/bin/env python3
"""CPU tests for the shared-expert-fused-r1 deliverable (stdlib only; a C++ compiler is optional).

    python3 -m unittest -v test_cpu.py

1. slice_partition.py reproduces the served GEMM work partition: every column's k-range is covered exactly once, folded
   in lock order, and the result depends on gridDim.x (the argument that a regridded shared-expert kernel cannot be
   bit-identical).
2. autotune_geometry.py's key port equals the verbatim C++ (hash_ref.cpp) and its cache reader matches
   coop_autotune.cu:load_disk_cache_locked on a synthetic file (with an embedded second header).
3. join_stall.py attributes the stall correctly on synthetic traces.
4. bound_arithmetic.py still produces the numbers quoted in impl-status.md.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import autotune_geometry as ag  # noqa: E402
import bound_arithmetic as ba  # noqa: E402
import join_stall as js  # noqa: E402
from slice_partition import TILESIZE_K, TILESIZE_N, f32, fold_column, segments  # noqa: E402


class TestPartition(unittest.TestCase):

    def test_segments_cover_each_column_once_in_lock_order(self):
        rnd = random.Random(1)
        cases = [(tk, tn, g) for tk in (1, 2, 3, 16, 32, 80, 160) for tn in (1, 2, 4, 5, 20) for g in (1, 2, 3, 7, 64, 170)]
        cases += [(rnd.randint(1, 200), rnd.randint(1, 40), rnd.randint(1, 340)) for _ in range(300)]
        for tk, tn, g in cases:
            segs = segments(tk, tn, g)
            for n, lst in segs.items():
                covered = sorted(k for k0, k1, _ in lst for k in range(k0, k1 + 1))
                self.assertEqual(covered, list(range(tk)), (tk, tn, g, n, lst))
                ends = [k1 for _, k1, _ in lst]
                self.assertEqual(ends, sorted(ends, reverse=True))
                self.assertEqual(lst[0][1], tk - 1)          # the "first" block holds the last k-tile
                # lock bookkeeping (inner.cuh:834-841): the fold visits lock_i = 0, d1, d1+d2, ... and ends at tiles_k
                lock = 0
                for k0, k1, _ in lst:
                    self.assertEqual(tk - k1 - 1, lock)
                    lock += k1 - k0 + 1
                self.assertEqual(lock, tk)

    def test_grid_size_changes_bits_for_shared_expert_shapes(self):
        # Shared down projection k = interm (512 assumed), n = hidden 2560; gate/up k = 2560, n = 512. For every tile
        # shape the kernel map offers, two different grids give different fp32 results for some output element.
        rnd = random.Random(7)
        found = 0
        for size_k, size_n in ((512, 2560), (2560, 512)):
            for shape in (1, 2, 3, 4):
                tk = size_k // TILESIZE_K[shape]
                tn = size_n // TILESIZE_N[shape]
                g1, g2 = max(2, min(tk * tn, 170) // 2), min(tk * tn, 170)
                s1, s2 = segments(tk, tn, g1), segments(tk, tn, g2)
                if s1 == s2:
                    continue
                cols = [n for n in range(tn) if [x[:2] for x in s1[n]] != [x[:2] for x in s2[n]]]
                diff = 0
                for trial in range(40):
                    partials = [f32(rnd.gauss(0.0, 1.0) * 10 ** rnd.uniform(-3, 1)) for _ in range(tk)]
                    for n in cols:
                        if fold_column(partials, s1[n]) != fold_column(partials, s2[n]):
                            diff += 1
                if diff:
                    found += 1
        self.assertGreater(found, 0)

    def test_same_grid_same_bits(self):
        rnd = random.Random(3)
        partials = [f32(rnd.gauss(0.0, 1.0)) for _ in range(80)]
        s = segments(80, 4, 64)
        for n in range(4):
            self.assertEqual(fold_column(partials, s[n]), fold_column(partials, segments(80, 4, 64)[n]))


class TestAutotuneKeys(unittest.TestCase):

    def test_row_buckets(self):
        # gate/up key buckets by MIN(roundup_pow2(m), 16); down by MIN(roundup_pow2(MAX(m, 2)), 16)
        gu = {r: ag.gate_up_key(r, 2560, 512, 5, 0, 170, 2) for r in range(1, 17)}
        dn = {r: ag.down_key(r, 512, 2560, 5, 0, 170, 2) for r in range(1, 17)}
        self.assertEqual(len(set(gu.values())), 5)            # {1} {2} {3,4} {5..8} {9..16}
        self.assertEqual(len(set(dn.values())), 4)            # {1,2} {3,4} {5..8} {9..16}
        self.assertEqual(gu[3], gu[4]); self.assertNotEqual(gu[4], gu[5]); self.assertEqual(gu[9], gu[16])
        self.assertEqual(dn[1], dn[2]); self.assertEqual(dn[5], dn[8])
        # served verify shapes: c1 d3 = 4 rows, c4 d1 = 8, c4 d3 = 16 are three different geometries
        self.assertEqual(len({gu[4], gu[8], gu[16]}), 3)
        # down at 1-2 rows (draft block, c1) takes the int8 path on sm_120 with the default EXL3_INT8_GEMV=2
        self.assertTrue(ag.down_path(1, 5, 2).startswith("int8"))
        self.assertTrue(ag.down_path(4, 5, 2).startswith("autotuned"))

    def test_python_key_matches_verbatim_cpp(self):
        cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if not cxx:
            self.skipTest("no C++ compiler")
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "hash_ref")
            subprocess.run([cxx, "-std=c++17", "-O1", "-o", exe, os.path.join(HERE, "hash_ref.cpp")], check=True)
            cases = []
            for dev in (0, 1):
                for m in (1, 2, 3, 4, 7, 8, 9, 16, 17, 300):
                    for K in (2, 5, 8):
                        cases.append(("gemm", m, 512, 2560, K, 1, dev, 5, 170, 2))
                        cases.append(("mgemm", m, 2560, 512, K, 0, dev, 5, 170, 2, 1, 2))
            cases.append(("mgemm", 4, 2560, 640, 5, 0, 1, 5, 170, 1, 30, 1))
            for c in cases:
                out = subprocess.run([exe, *map(str, c)], check=True, capture_output=True, text=True).stdout.strip()
                if c[0] == "gemm":
                    py = ag.salt(ag.gemm_autotune_hash(c[1], c[2], c[3], c[4], bool(c[5]), c[6], c[7], c[8], c[9]))
                else:
                    py = ag.salt(ag.mgemm_autotune_hash(c[1], c[2], c[3], c[4], bool(c[5]), c[6], c[7], c[8], c[9], c[10], c[11]))
                self.assertEqual(out, f"{py:016x}", c)

    def test_cache_reader_matches_cpp_parse(self):
        recs = [(0x1111, 2, 512, 64, 1), (0x2222, 4, 256, 80, 2), (0x3333, 3, 512, 170, 1), (0x1111, 1, 256, 32, 1)]
        blob = struct.pack("<8sII", b"EX3ATUNE", 1, 32)
        blob += struct.pack("<QiiiiII", *recs[0], 0, 0) + struct.pack("<QiiiiII", *recs[1], 0, 0)
        # a second process appended header + first record in one write (coop_autotune.cu:185-199)
        blob += struct.pack("<8sII", b"EX3ATUNE", 1, 32) + struct.pack("<QiiiiII", *recs[2], 0, 0)
        blob += struct.pack("<QiiiiII", *recs[3], 0, 0)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(blob)
            path = f.name
        try:
            got = ag.read_cache(path)
        finally:
            os.unlink(path)
        self.assertEqual(got, {0x1111: (1, 256, 32, 1), 0x2222: (4, 256, 80, 2), 0x3333: (3, 512, 170, 1)})


def _k(name, ts, dur, stream, dev=0):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": {"device": dev, "stream": stream}}


A = "void exl3_moe_coop_v2_ns::exl3_moe_coop_a_kernel<5, 2, false>(MoeCoopParams)"
B = "void exl3_moe_coop_v2_ns::exl3_moe_coop_b_kernel<5, 2, false>(MoeCoopParams)"
MG = "void exl3_mgemm_kernel<5, false, 2, 16, 32, 128, 4, 3>(...)"
ACT = "void act_mul_kernel_f<0>(...)"
GM = "void exl3_gemm_kernel<5, true, 2, 16, 32, 128, 4, 3>(...)"


class TestJoinStall(unittest.TestCase):

    def _run(self, events, steps=None):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            res = js.analyze(js.load_events(path))
        finally:
            os.unlink(path)
        return res, js.summarize(res, steps)

    def test_hidden_late_and_absent(self):
        ev = [
            # layer 1: shared (side stream 9) ends at 30, A ends at 50 -> hidden
            _k(MG, 0, 10, 9), _k(ACT, 11, 2, 9), _k(GM, 14, 16, 9), _k(A, 2, 48, 7), _k(B, 52, 20, 7),
            # layer 2: A ends at 140, shared ends at 145, B starts at 147 -> exposed 5
            _k(MG, 100, 15, 9), _k(ACT, 116, 2, 9), _k(GM, 119, 26, 9), _k(A, 102, 38, 7), _k(B, 147, 20, 7),
            # layer 3: no side work (e.g. a layer without a shared expert) -> 0
            _k(A, 200, 30, 7), _k(B, 232, 20, 7),
            # unrelated side-stream kernel (not a shared-expert kernel name) inside layer 3's window -> ignored
            _k("void some_other_kernel(...)", 205, 60, 11),
            # other device, independent
            _k(MG, 0, 10, 3, dev=1), _k(GM, 12, 30, 3, dev=1), _k(A, 1, 20, 2, dev=1), _k(B, 43, 10, 2, dev=1),
        ]
        res, s = self._run(ev, steps=1)
        by = [(p["dev"], round(p["exposed_us"], 3)) for p in res["pairs"]]
        self.assertIn((0, 0.0), by)
        self.assertIn((0, 5.0), by)
        self.assertIn((1, 21.0), by)          # dev 1: A ends 21, shared ends 42, B at 43 -> 21 exposed
        self.assertEqual(s["moe_calls"], 4)
        self.assertEqual(s["calls_with_side_stream_work"], 3)
        self.assertEqual(s["calls_stalled_gt_0p5us"], 2)
        self.assertAlmostEqual(s["exposed_total_us"], 26.0)
        self.assertAlmostEqual(s["exposed_ms_per_step"], 0.026)

    def test_window_is_bounded_by_previous_b(self):
        # side kernel belonging to layer 1 must not be charged to layer 2
        ev = [_k(MG, 0, 60, 9), _k(A, 0, 40, 7), _k(B, 61, 5, 7),
              _k(A, 70, 40, 7), _k(B, 111, 5, 7)]
        res, _ = self._run(ev)
        self.assertEqual([round(p["exposed_us"], 3) for p in res["pairs"]], [20.0, 0.0])
        self.assertEqual([p["side_kernels"] for p in res["pairs"]], [1, 0])


class TestArithmetic(unittest.TestCase):

    def test_quoted_numbers(self):
        r = ba.report()
        self.assertEqual(r["shared_us_per_layer_c1d3"], 23.71)
        self.assertEqual(r["shared_us_per_layer_c4d3"], 25.46)
        self.assertEqual(r["ceiling_gain_pct_c1d3"], 8.32)
        self.assertEqual(r["ceiling_gain_pct_c4d3"], 5.49)
        self.assertEqual(r["payload_floor_us_per_layer"], 1.71)
        self.assertEqual(r["mean_gain_pct_c1"], 4.23)
        self.assertEqual(r["mean_gain_pct_c4"], 5.65)
        self.assertEqual(r["residual_ms_c1d3_point"], 0.536)
        self.assertEqual(r["residual_ms_c4d3_point"], -0.034)
        self.assertEqual(r["residual_ms_c4d3_microbench_bound"], 0.137)
        self.assertEqual(r["residual_ms_c1d3_microbench_bound"], 0.0)


if __name__ == "__main__":
    unittest.main()
