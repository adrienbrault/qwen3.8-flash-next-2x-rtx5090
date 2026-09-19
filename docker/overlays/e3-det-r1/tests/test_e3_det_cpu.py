#!/usr/bin/env python3
"""CPU tests for e3-det r1 (deterministic E3 grouped MoE prefill). No GPU, no compiled extension.

Part 1 (index model): Python ports of the index arithmetic of e3_metadata_kernel (unchanged),
e3_det_row_pos_kernel, e3_down_det_kernel's slot store, the served exl3_moe kernel's slot mode
(slot = fused_base[e] + row) and e3_det_reduce_kernel's slot lookup. Over random, skewed and
boundary routings it checks: every assignment's slot is written exactly once, by the right tier
(fat = count > thin_rows), for the right (token, expert, weight); the reduce reads token t's
k-th slot as assignment (t, k); the fat half region and the thin fp32 region of a slot are inside
the slot row; the whole slot scratch is exactly the bytes of h13_gate|h13_up; and the down store's
(128-column block, lane) map equals the reduce's.

Part 2 (dispatch): loads the served block_sparse_mlp.py (baseline, from src/) and the overlay copy
side by side with stub packages and a stub extension that validates arguments like the C++
wrappers and emulates the kernels at slot level. Checks:
  * flag unset / "0": the overlay issues exactly the baseline's extension calls with identical
    arguments (E3 on at 512 / 2,048 rows, E3 off, below the E3 minimum) and returns identical bits;
  * flag set but E3 inactive (E3 off, or < 512 rows): identical to the baseline too;
  * flag set and E3 active: the call sequence is e3_det -> exl3_moe (thin windows, slot mode,
    fused_base = expert start) -> e3_det_reduce, no atomic E3 and no exl3_moe_gather; argsort is
    stable; the slot scratch aliases h13_g|h13_u; the output equals the reference routed sum; the
    output is independent of the fat-row processing order (emulated CTA order shuffles).

Run: python3 test_e3_det_cpu.py  (paths default to the workspace layout; override with
E3DET_BASELINE=<.../exllamav3> and E3DET_OVERLAY=<.../exllamav3>).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import random
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True  # never leave __pycache__ in the baseline tree or the overlay

HERE = Path(__file__).resolve().parent
BASELINE = Path(os.environ.get("E3DET_BASELINE", HERE.parent.parent.parent / "src" / "exllamav3"))
OVERLAY = Path(os.environ.get("E3DET_OVERLAY", HERE.parent / "overlay" / "exllamav3"))

EXPERTS = 512
TOPK = 10
HIDDEN = 2560
INTER = 640
TILE_M = 64


# ----------------------------------------------------------------------------------------------
# Part 1: index model
# ----------------------------------------------------------------------------------------------

def metadata_model(expert_count, token_sorted, weight_sorted, thin_rows):
    """e3_metadata_kernel (served, unchanged): fat rows + 64-row segments."""
    starts, fat_base, seg_base = {}, {}, {}
    src = rows = segs = 0
    for e in range(EXPERTS):
        c = int(expert_count[e])
        starts[e] = src
        src += c
        fat_base[e] = rows
        seg_base[e] = segs
        if c > thin_rows:
            rows += c
            segs += (c + TILE_M - 1) // TILE_M
    A = len(token_sorted)
    row_token = np.full(A, -1, np.int64)
    row_weight = np.zeros(A, np.float16)
    row_expert = np.full(A, -1, np.int32)
    seg = []
    for e in range(EXPERTS):
        c = int(expert_count[e])
        if c <= thin_rows:
            continue
        for r in range(c):
            row_token[fat_base[e] + r] = token_sorted[starts[e] + r]
            row_weight[fat_base[e] + r] = weight_sorted[starts[e] + r]
            row_expert[fat_base[e] + r] = e
        for s in range((c + TILE_M - 1) // TILE_M):
            r0 = s * TILE_M
            seg.append((seg_base[e] + s, e, fat_base[e] + r0, min(c - r0, TILE_M)))
    return rows, sorted(seg), row_token, row_weight, row_expert


def prep_model(expert_count, order, thin_rows):
    """e3_det_prep_kernel, step for step: warp shuffle-up inclusive scans (32 lanes, 16 warps),
    a second shuffle scan over the warp totals made exclusive, per-expert row_pos writes, and the
    permutation inversion done by every block."""
    E, W = EXPERTS, 32
    NW = E // W
    c = [int(expert_count[e]) for e in range(E)]
    f = [ci if ci > thin_rows else 0 for ci in c]
    s_inc, f_inc = list(c), list(f)
    for w in range(NW):
        lanes = range(w * W, (w + 1) * W)
        d = 1
        while d < 32:
            s_prev, f_prev = list(s_inc), list(f_inc)   # shfl_up reads the pre-step values
            for t in lanes:
                lane = t - w * W
                if lane >= d:
                    s_inc[t] = s_prev[t] + s_prev[t - d]
                    f_inc[t] = f_prev[t] + f_prev[t - d]
            d <<= 1
    warp_s = [s_inc[w * W + 31] for w in range(NW)]
    warp_f = [f_inc[w * W + 31] for w in range(NW)]
    ws0 = warp_s + [0] * (32 - NW)
    wf0 = warp_f + [0] * (32 - NW)
    ws, wf = list(ws0), list(wf0)
    d = 1
    while d < 32:
        sp, fp = list(ws), list(wf)
        for lane in range(32):
            if lane >= d:
                ws[lane] = sp[lane] + sp[lane - d]
                wf[lane] = fp[lane] + fp[lane - d]
        d <<= 1
    warp_s = [ws[l] - ws0[l] for l in range(NW)]
    warp_f = [wf[l] - wf0[l] for l in range(NW)]
    A = len(order)
    row_pos = np.full(A, -1, np.int64)
    expert_start = np.zeros(E + 1, np.int64)
    for e in range(E):
        w = e // W
        src0 = warp_s[w] + s_inc[e] - c[e]
        dst0 = warp_f[w] + f_inc[e] - f[e]
        expert_start[e] = src0
        if e == E - 1:
            expert_start[E] = src0 + c[e]
        if f[e]:
            for r in range(c[e]):
                row_pos[dst0 + r] = src0 + r
    inv = np.full(A, -1, np.int64)
    for p_ in range(A):
        inv[int(order[p_])] = p_
    return row_pos, expert_start, inv


def row_pos_model(expert_count, thin_rows, A):
    """Serial reference for row_pos (what e3_metadata_kernel's fat layout implies)."""
    row_pos = np.full(A, -1, np.int64)
    src = rows = 0
    for e in range(EXPERTS):
        c = int(expert_count[e])
        if c > thin_rows:
            for r in range(c):
                row_pos[rows + r] = src + r
            rows += c
        src += c
    return row_pos


def down_store_addrs(pos, hidden = HIDDEN):
    """e3_down_det_kernel store: half offsets of (n-block, s, lane) -> 4 halves."""
    out = {}
    for bx in range(hidden // 256):
        n_base = bx * 256
        for s in range(2):
            for lane in range(32):
                off = pos * 2 * hidden + n_base + lane * 4 + s * 128
                out[(n_base + s * 128, lane)] = off
    return out


def reduce_read_addrs(pos, hidden = HIDDEN):
    """e3_det_reduce_kernel fat read: warp (token, block b), lane -> half offset."""
    out = {}
    for b in range(hidden // 128):
        for lane in range(32):
            col = b * 128 + lane * 4
            out[(b * 128, lane)] = pos * 2 * hidden + col
    return out


def make_routing(T, mode, seed):
    g = torch.Generator().manual_seed(seed)
    if mode == "uniform":
        probs = torch.ones(EXPERTS)
    elif mode == "zipf":
        probs = 1.0 / (torch.arange(EXPERTS, dtype = torch.float64) + 1.0) ** 1.1
    elif mode == "hot":
        probs = torch.full((EXPERTS,), 1e-3, dtype = torch.float64)
        probs[:10] = 1.0  # ten experts take (nearly) every token: counts ~= T
    else:
        raise ValueError(mode)
    probs = probs[torch.randperm(EXPERTS, generator = g)]
    sel = torch.multinomial(probs.float().expand(T, EXPERTS), TOPK, replacement = False, generator = g)
    w = torch.rand((T, TOPK), generator = g).half()
    return sel.long(), w


def boundary_routing(T, thin_rows, seed):
    """Counts placed exactly on thin_rows, thin_rows + 1, 16/17, 32/33, 64/65, 128/129."""
    rng = random.Random(seed)
    targets = [thin_rows, thin_rows + 1, 16, 17, 32, 33, 64, 65, 128, 129, 1, 2]
    experts = rng.sample(range(EXPERTS), len(targets))
    need = {e: c for e, c in zip(experts, targets)}
    sel = [[] for _ in range(T)]
    order = list(range(T))
    for e, c in need.items():
        rng.shuffle(order)
        placed = 0
        for t in order:
            if placed == c:
                break
            if len(sel[t]) < TOPK and e not in sel[t]:
                sel[t].append(e)
                placed += 1
    others = [e for e in range(EXPERTS) if e not in need]
    for t in range(T):
        while len(sel[t]) < TOPK:
            e = rng.choice(others)
            if e not in sel[t]:
                sel[t].append(e)
    w = torch.tensor([[rng.random() for _ in range(TOPK)] for _ in range(T)]).half()
    return torch.tensor(sel, dtype = torch.long), w


def check_slots(testcase, sel, w, thin_rows):
    T = sel.shape[0]
    flat_e = sel.reshape(-1)
    flat_w = w.reshape(-1)
    A = flat_e.numel()
    order = flat_e.argsort(stable = True)
    flat_t = torch.arange(T).repeat_interleave(TOPK)
    token_sorted = flat_t[order].numpy()
    weight_sorted = flat_w[order].numpy()
    count = torch.bincount(flat_e, minlength = EXPERTS + 1).numpy()
    start = np.cumsum(count) - count

    fat_rows, segs, row_token, row_weight, row_expert = metadata_model(count, token_sorted, weight_sorted, thin_rows)
    row_pos = row_pos_model(count, thin_rows, A)
    row_pos_p, expert_start_p, inv_p = prep_model(count, order.numpy(), thin_rows)
    testcase.assertTrue(np.array_equal(row_pos_p, row_pos), "parallel prep scan != serial layout")
    testcase.assertTrue(np.array_equal(expert_start_p[:EXPERTS], start[:EXPERTS]))
    testcase.assertEqual(int(expert_start_p[EXPERTS]), A)
    testcase.assertEqual(fat_rows, int(sum(c for c in count[:EXPERTS] if c > thin_rows)))
    testcase.assertLessEqual(len(segs), (A + 63) // 64 + EXPERTS)  # seg_cap bound used by Python

    written = {}  # slot -> (tier, expert, token, weight)

    # E3 fat tier: every (segment, mb, r) row of every CTA column; slot = row_pos[fat_row]
    for _, e, r0, rows in segs:
        for mb in range(4):
            rows_mb = rows - mb * 16
            if rows_mb <= 0:
                break
            for r in range(min(rows_mb, 16)):
                fr = r0 + mb * 16 + r
                p = int(row_pos[fr])
                testcase.assertGreaterEqual(p, 0)
                testcase.assertNotIn(p, written, "fat slot written twice")
                written[p] = ("fat", e, int(row_token[fr]), float(row_weight[fr]))

    # Thin tier: exl3_moe slot mode over the windows the Python issues
    windows = [(1, min(16, thin_rows))] + ([(17, thin_rows)] if thin_rows > 16 else [])
    for lo, hi in windows:
        for e in range(EXPERTS):
            c = int(count[e])
            if c == 0 or c < lo or c > hi:
                continue
            for r in range(c):
                p = int(start[e]) + r  # fused_base[e] + row
                testcase.assertNotIn(p, written, "thin slot written twice")
                written[p] = ("thin", e, int(token_sorted[start[e] + r]), float(weight_sorted[start[e] + r]))

    testcase.assertEqual(sorted(written), list(range(A)), "every assignment slot written exactly once")

    # Reduce: assignment a = t * TOPK + k reads slot inv_order[a] (as the prep kernel wrote it)
    inv = inv_p
    testcase.assertTrue(np.array_equal(inv, torch.empty_like(order).scatter_(0, order, torch.arange(A)).numpy()))
    for a in range(A):
        t, k = divmod(a, TOPK)
        e = int(flat_e[a])
        tier, we, wt, ww = written[int(inv[a])]
        testcase.assertEqual(tier, "fat" if count[e] > thin_rows else "thin")
        testcase.assertEqual((we, wt), (e, t))
        testcase.assertEqual(ww, float(flat_w[a]))  # route weight the atomic kernel used

    # Byte layout: slot scratch fp32 [A, 2560] == h13_g|h13_u (2 x [A, 2560] half)
    testcase.assertEqual(A * HIDDEN * 4, 2 * A * HIDDEN * 2)
    for p in (0, A // 2, A - 1):
        st = down_store_addrs(p)
        rd = reduce_read_addrs(p)
        testcase.assertEqual(st, rd, "down store and reduce read map (column block, lane) identically")
        halves = sorted(o + i for o in st.values() for i in range(4))
        testcase.assertEqual(halves, list(range(p * 2 * HIDDEN, p * 2 * HIDDEN + HIDDEN)))
        # fat half region [p*5120, p*5120 + 2560) and thin fp32 row [p*2560, (p+1)*2560) floats
        testcase.assertLessEqual(p * 2 * HIDDEN + HIDDEN, (p + 1) * 2 * HIDDEN)
    return count


class IndexModel(unittest.TestCase):
    def test_random_routings(self):
        n = 0
        for T in (512, 1024, 2048):
            for mode in ("uniform", "zipf", "hot"):
                for thin in (1, 16, 32):
                    sel, w = make_routing(T, mode, seed = T * 7 + len(mode) + thin)
                    count = check_slots(self, sel, w, thin)
                    n += 1
                    if mode == "hot":
                        self.assertGreater(int(count.max()), T // 2)
        self.assertEqual(n, 27)

    def test_boundaries(self):
        for thin in (1, 16, 32, 64):
            sel, w = boundary_routing(512, thin, seed = 11 + thin)
            count = check_slots(self, sel, w, thin)
            for c in (thin, thin + 1, 64, 65):
                self.assertIn(c, set(int(x) for x in count))

    def test_thin_32_windows_cover_exactly(self):
        # Windows [1, 16] and [17, 32] plus E3's (32, inf) partition every positive count
        for c in range(1, 3000):
            tiers = int(1 <= c <= 16) + int(17 <= c <= 32) + int(c > 32)
            self.assertEqual(tiers, 1)


# ----------------------------------------------------------------------------------------------
# Part 2: dispatch with stub packages and an emulating stub extension
# ----------------------------------------------------------------------------------------------

class CallLog:
    def __init__(self):
        self.calls = []

    @staticmethod
    def digest(v):
        if isinstance(v, torch.Tensor):
            t = v.detach()
            h = hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16] \
                if t.numel() else "empty"
            return ("T", str(t.dtype), tuple(t.shape), tuple(t.stride()), h)
        if isinstance(v, float):
            return ("f", float(v).hex())
        return ("v", repr(v))

    def add(self, name, args):
        self.calls.append((name, tuple(self.digest(a) for a in args)))


class World:
    """Per-expert synthetic weights shared by the stub kernels and the reference."""

    def __init__(self, seed = 5):
        g = torch.Generator().manual_seed(seed)
        self.W = (torch.rand((EXPERTS, HIDDEN), generator = g) * 2 - 1).half()     # stands in for gate/up/down
        self.S = (torch.rand((EXPERTS, HIDDEN), generator = g) * 0.5 + 0.75).half()  # down svh

    def hv(self, y_row, e):
        # "half-rounded pre-Hadamard down output" of token row y through expert e
        return (y_row.float() * self.W[e].float()).half()

    def contrib(self, y_row, e, w):
        # the epilogue both tiers share (Hadamard omitted in the emulation)
        return self.hv(y_row, e).float() * float(w) * self.S[e].float()


def check(t, dtype, name, shape = None):
    assert isinstance(t, torch.Tensor), name
    assert t.dtype == dtype, f"{name}: dtype {t.dtype} != {dtype}"
    assert t.is_contiguous(), f"{name} must be contiguous"
    if shape is not None:
        assert tuple(t.shape) == tuple(shape), f"{name}: shape {tuple(t.shape)} != {shape}"


def make_ext(log: CallLog, world: World, fat_order_seed = None):
    ext = types.SimpleNamespace()

    def exl3_moe(y, out, expert_count, token_sorted, weight_sorted, tsg, tsu, tig, tiu, act_idx,
                 Kg, Ku, Kd, gtr, gsuh, gsvh, utr, usuh, usvh, dtr, dsuh, dsvh,
                 gmcg, gmul1, umcg, umul1, dmcg, dmul1, act_limit, num_active,
                 output_scratch, fused_base, count_lo, count_hi, m_tile):
        log.add("exl3_moe", (y, out, expert_count, token_sorted, weight_sorted, act_idx, Kg, Ku, Kd,
                             gmcg, gmul1, umcg, umul1, dmcg, dmul1, act_limit, num_active,
                             output_scratch if output_scratch is not None else "none",
                             fused_base if fused_base is not None else "none", count_lo, count_hi, m_tile))
        if output_scratch is not None:
            check(output_scratch, torch.float, "output_scratch")
            assert output_scratch.dim() == 2 and output_scratch.size(1) == y.size(1)
            check(fused_base, torch.long, "fused_base")
        max_rows = tsg.size(1)
        counts = expert_count.tolist()
        s = 0
        for e in range(len(counts) - 1):
            c = counts[e]
            if 0 < c <= max_rows and count_lo <= c <= count_hi:
                for r in range(c):
                    tok = int(token_sorted[s + r])
                    v = world.contrib(y[tok], e, weight_sorted[s + r])
                    if output_scratch is not None:
                        output_scratch[int(fused_base[e]) + r] = v
                    else:
                        out[tok] += v
            s += c

    def exl3_moe_gather(out, scratch, flat_expert, inv_order, expert_start, slot_base, slot_kind, weight_sorted):
        log.add("exl3_moe_gather", (out, scratch, flat_expert, inv_order, expert_start, slot_base, slot_kind, weight_sorted))
        T = out.size(0)
        k = flat_expert.numel() // T
        for t in range(T):
            acc = torch.zeros(out.size(1))
            for j in range(k):
                a = t * k + j
                e = int(flat_expert[a])
                kind = int(slot_kind[e]) if e < slot_kind.numel() else 0
                if kind:
                    pos = int(inv_order[a])
                    wt = float(weight_sorted[pos]) if kind == 2 else 1.0
                    acc += scratch[int(slot_base[e]) + pos - int(expert_start[e])] * wt
            out[t] += acc

    def _meta(expert_count, token_sorted, weight_sorted, row_token, row_weight, row_expert,
              seg_expert, seg_row0, seg_rows, num_rows, num_segs, thin):
        fat_rows, segs, rt, rw, re_ = metadata_model(expert_count.tolist(), token_sorted.numpy(),
                                                     weight_sorted.numpy(), thin)
        row_token.copy_(torch.from_numpy(rt))
        row_weight.copy_(torch.from_numpy(rw))
        row_expert.copy_(torch.from_numpy(re_))
        for i, e, r0, rows in segs:
            seg_expert[i], seg_row0[i], seg_rows[i] = e, r0, rows
        num_rows[0] = fat_rows
        num_segs[0] = len(segs)
        return segs

    def _check_e3_common(x, expert_count, token_sorted, weight_sorted, h13_g, h13_u, h2,
                         row_token, row_weight, row_expert, seg_expert, num_rows, num_segs, thin):
        check(x, torch.half, "x")
        assert x.size(1) == HIDDEN
        check(expert_count, torch.long, "expert_count", (EXPERTS + 1,))
        check(token_sorted, torch.long, "token_sorted")
        check(weight_sorted, torch.half, "weight_sorted", token_sorted.shape)
        cap = token_sorted.numel()
        assert cap == x.size(0) * TOPK
        check(h13_g, torch.half, "h13_gate", (cap, HIDDEN))
        check(h13_u, torch.half, "h13_up", (cap, HIDDEN))
        check(h2, torch.half, "h2", (cap, INTER))
        check(row_token, torch.long, "row_token", (cap,))
        check(row_weight, torch.half, "row_weight", (cap,))
        check(row_expert, torch.int, "row_expert", (cap,))
        assert seg_expert.numel() >= (cap + 63) // 64 + EXPERTS
        check(num_rows, torch.int, "num_rows", (1,))
        check(num_segs, torch.int, "num_segs", (1,))
        assert 1 <= thin <= 256

    def exl3_moe_prefill_e3(x, out, expert_count, token_sorted, weight_sorted,
                            gtr, gsuh, gsvh, utr, usuh, usvh, dtr, dsuh, dsvh,
                            h13_g, h13_u, h2, row_token, row_weight, row_expert,
                            seg_expert, seg_row0, seg_rows, num_rows, num_segs,
                            gb, ub, db, thin, act_limit):
        log.add("exl3_moe_prefill_e3", (x, out, expert_count, token_sorted, weight_sorted, gb, ub, db, thin, act_limit))
        _check_e3_common(x, expert_count, token_sorted, weight_sorted, h13_g, h13_u, h2,
                         row_token, row_weight, row_expert, seg_expert, num_rows, num_segs, thin)
        segs = _meta(expert_count, token_sorted, weight_sorted, row_token, row_weight, row_expert,
                     seg_expert, seg_row0, seg_rows, num_rows, num_segs, thin)
        for _, e, r0, rows in segs:
            for r in range(rows):
                fr = r0 + r
                out[int(row_token[fr])] += world.contrib(x[int(row_token[fr])], e, row_weight[fr])

    def exl3_moe_prefill_e3_det(x, expert_count, token_sorted, weight_sorted,
                                gtr, gsuh, gsvh, utr, usuh, usvh, dtr, dsuh,
                                h13_g, h13_u, h2, row_token, row_weight, row_expert,
                                row_pos, order, expert_start, inv_order,
                                seg_expert, seg_row0, seg_rows, num_rows, num_segs, slots,
                                gb, ub, db, thin, act_limit):
        log.add("exl3_moe_prefill_e3_det", (x, expert_count, token_sorted, weight_sorted, gb, ub, db, thin, act_limit))
        _check_e3_common(x, expert_count, token_sorted, weight_sorted, h13_g, h13_u, h2,
                         row_token, row_weight, row_expert, seg_expert, num_rows, num_segs, thin)
        cap = token_sorted.numel()
        check(row_pos, torch.int, "row_pos", (cap,))
        check(order, torch.long, "order", (cap,))
        check(expert_start, torch.long, "expert_start", (EXPERTS + 1,))
        check(inv_order, torch.long, "inv_order", (cap,))
        check(slots, torch.float, "slots", (cap, HIDDEN))
        # order must be the permutation that produced token_sorted
        assert torch.equal(token_sorted, torch.arange(x.size(0)).repeat_interleave(TOPK)[order])
        # The Python path is meant to alias the slot scratch over h13_g|h13_u
        assert h13_g.data_ptr() == slots.data_ptr()
        assert h13_u.data_ptr() == slots.data_ptr() + cap * HIDDEN * 2
        segs = _meta(expert_count, token_sorted, weight_sorted, row_token, row_weight, row_expert,
                     seg_expert, seg_row0, seg_rows, num_rows, num_segs, thin)
        rp, es, inv = prep_model(expert_count.tolist(), order.numpy(), thin)
        row_pos.copy_(torch.from_numpy(rp).int())
        expert_start.copy_(torch.from_numpy(es))
        inv_order.copy_(torch.from_numpy(inv))
        # gather + gate/up write h13_g/h13_u (all rows of the scratch get clobbered here)
        h13_g.fill_(7.0)
        h13_u.fill_(-3.0)
        # down: fat slot rows get the half pre-Hadamard values, in an arbitrary "CTA" order
        work = [(e, r0 + r) for _, e, r0, rows in segs for r in range(rows)]
        if fat_order_seed is not None:
            random.Random(fat_order_seed).shuffle(work)
        sh = slots.view(torch.half)
        for e, fr in work:
            sh[int(row_pos[fr]), :HIDDEN] = world.hv(x[int(row_token[fr])], e)

    def exl3_moe_prefill_e3_det_reduce(out, slots, flat_expert, inv_order, flat_weight, expert_count, dsvh, thin):
        log.add("exl3_moe_prefill_e3_det_reduce", (out, flat_expert, inv_order, flat_weight, expert_count, thin))
        check(out, torch.float, "out")
        T = out.size(0)
        cap = T * TOPK
        check(slots, torch.float, "slots", (cap, HIDDEN))
        check(flat_expert, torch.long, "flat_expert", (cap,))
        check(inv_order, torch.long, "inv_order", (cap,))
        check(flat_weight, torch.half, "flat_weight", (cap,))
        check(expert_count, torch.long, "expert_count", (EXPERTS + 1,))
        assert 1 <= thin <= 256
        sh = slots.view(torch.half)
        for t in range(T):
            acc = torch.zeros(HIDDEN)
            for k in range(TOPK):
                a = t * TOPK + k
                e = int(flat_expert[a])
                p = int(inv_order[a])
                if int(expert_count[e]) > thin:
                    v = sh[p, :HIDDEN].float() * float(flat_weight[a]) * world.S[e].float()
                else:
                    v = slots[p]
                acc = acc + v
            out[t] = acc

    ext.exl3_moe = exl3_moe
    ext.exl3_moe_gather = exl3_moe_gather
    ext.exl3_moe_prefill_e3 = exl3_moe_prefill_e3
    ext.exl3_moe_prefill_e3_det = exl3_moe_prefill_e3_det
    ext.exl3_moe_prefill_e3_det_reduce = exl3_moe_prefill_e3_det_reduce
    return ext


def load_bsm(tag: str, bsm_file: Path, ext_holder: dict):
    """Import block_sparse_mlp.py as <tag>.modules.block_sparse_mlp with stub siblings."""
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class _Any:
        def __init__(self, *a, **k):
            pass

    root = mod(tag); root.__path__ = []
    mods = mod(f"{tag}.modules", Module = type("Module", (), {}), Linear = _Any); mods.__path__ = []
    model = mod(f"{tag}.model"); model.__path__ = []
    util = mod(f"{tag}.util", profile_opt = lambda *a, **k: (lambda f: f)); util.__path__ = []
    mod(f"{tag}.model.config", Config = _Any)
    mod(f"{tag}.model.model_tp_alloc", TPAllocation = _Any)
    mod(f"{tag}.util.tensor", to2 = lambda x, *a: x, g_tensor_cache = None,
        buffered_interleaved_arange = lambda r, k, device: torch.arange(r, device = device).repeat_interleave(k))
    spec = importlib.util.spec_from_file_location(f"{tag}.util.prefill_nosync",
                                                  BASELINE / "util" / "prefill_nosync.py")
    pn = importlib.util.module_from_spec(spec); sys.modules[spec.name] = pn; spec.loader.exec_module(pn)

    class ExtProxy:
        def __getattr__(self, n):
            return getattr(ext_holder["ext"], n)

    mod(f"{tag}.ext", exllamav3_ext = ExtProxy())
    mod(f"{tag}.modules.multilinear", MultiLinear = _Any)
    mod(f"{tag}.modules.mlp", MLP = _Any, GatedMLP = _Any)
    mod(f"{tag}.modules.rmsnorm", RMSNorm = _Any)
    mod(f"{tag}.modules.layernorm", LayerNorm = _Any)
    mod(f"{tag}.modules.block_sparse_mlp_cpu", BlockSparseMLP_CPU = type("BlockSparseMLP_CPU", (), {}))
    mod(f"{tag}.modules.moe_batch_recon", PAD_MAX = 0, plan_groups = None, BatchReconLayer = _Any)
    names = ["RoutingCFG", "ROUTING_ACT_SIGMOID", "ROUTING_ACT_SQRTSP", "routing_std", "routing_std_bias",
             "routing_ds3", "routing_dots", "routing_sqrtsp", "routing_sqrtsp_hash"]
    mod(f"{tag}.modules.block_sparse_mlp_routing", **{n: _Any for n in names})
    spec = importlib.util.spec_from_file_location(f"{tag}.modules.block_sparse_mlp", bsm_file)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    return m


def make_layer(bsm, sel, w, K = (3, 2, 4)):
    L = object.__new__(bsm.BlockSparseMLP)
    ptrs = lambda: torch.arange(EXPERTS, dtype = torch.long)

    def multi(k):
        return types.SimpleNamespace(K = k, mul1 = True, mcg = False, ptrs_trellis = ptrs(),
                                     ptrs_suh = ptrs(), ptrs_svh = ptrs())

    L.__dict__.update(dict(
        alt_residual_channel = False, hidden_size = HIDDEN, expert_size = HIDDEN, bc = None,
        router_pre_norm = None, routing_gate = object(), routing_cfg = None,
        routing_fn = lambda bsz, cfg, z, params: (sel.clone(), w.clone()),
        num_experts_per_tok = TOPK, device = torch.device("cpu"), routed_pre_norm = None, latent_in = None,
        routing_device = None, cpu_split_first = None, cpu_offload = False, intermediate_size = INTER,
        num_local_experts = EXPERTS, num_experts = EXPERTS, f_threshold = 0, is_quantized = True,
        config = types.SimpleNamespace(infer_params = types.SimpleNamespace(no_reconstruct = False)),
        support_quant_paths = True, routing_first = 0,
        fused_mode_buffers = types.SimpleNamespace(
            temp_state_g = torch.zeros((2, 256, HIDDEN), dtype = torch.half),
            temp_state_u = torch.zeros((2, 256, HIDDEN), dtype = torch.half),
            temp_intermediate_g = torch.zeros((2, 256, INTER), dtype = torch.half),
            temp_intermediate_u = torch.zeros((2, 256, INTER), dtype = torch.half)),
        fused_rows = 4096, support_fused = True, gated = True, activation_fn = "silu",
        activation_fn_idx = 0, interm_dtype = torch.half, intermediate_size_padded = INTER,
        multi_gate = multi(K[0]), multi_up = multi(K[1]), multi_down = multi(K[2]), act_limit = 0.0,
        mtile_ok = False, latent_out = None, tp_reduce = False, routed_post_norm = None,
        shared_experts = None, shared_experts_post_norm = None, shared_gate = None, bc_sh_exp = False,
    ))
    L._batch_recon_layer = lambda y: None
    L.cpu_split_combine = lambda fhs, p, q, shape: fhs
    return L


class Dispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert (BASELINE / "modules" / "block_sparse_mlp.py").is_file(), BASELINE
        assert (OVERLAY / "modules" / "block_sparse_mlp.py").is_file(), OVERLAY
        cls.holder = {}
        cls.base = load_bsm("e3det_base", BASELINE / "modules" / "block_sparse_mlp.py", cls.holder)
        cls.new = load_bsm("e3det_new", OVERLAY / "modules" / "block_sparse_mlp.py", cls.holder)
        cls.world = World()
        cls._cap = torch.cuda.get_device_capability
        torch.cuda.get_device_capability = lambda *a, **k: (12, 0)
        cls._empty, cls._empty_like = torch.empty, torch.empty_like
        # Uninitialised workspaces would make argument digests differ run to run: zero them
        torch.empty = lambda *a, **k: torch.zeros(*a, **k)
        torch.empty_like = lambda *a, **k: torch.zeros_like(*a, **k)

    @classmethod
    def tearDownClass(cls):
        torch.cuda.get_device_capability = cls._cap
        torch.empty, torch.empty_like = cls._empty, cls._empty_like

    def run_layer(self, bsm, T, e3, det, mode = "zipf", seed = 3, thin = 32, fat_order_seed = None):
        sel, w = make_routing(T, mode, seed)
        g = torch.Generator().manual_seed(seed + 1)
        x = (torch.rand((T, HIDDEN), generator = g) * 2 - 1).half()
        log = CallLog()
        self.holder["ext"] = make_ext(log, self.world, fat_order_seed)
        bsm.MOE_PREFILL_E3 = e3
        if hasattr(bsm, "MOE_PREFILL_E3_DET"):
            bsm.MOE_PREFILL_E3_DET = det
        bsm.MOE_PREFILL_E3_THIN_ROWS = thin
        argsorts = []
        real_argsort = torch.Tensor.argsort

        def spy(self_, *a, **k):
            argsorts.append(bool(k.get("stable", False)))
            return real_argsort(self_, *a, **k)

        torch.Tensor.argsort = spy
        try:
            out = make_layer(bsm, sel, w).forward(x, {})
        finally:
            torch.Tensor.argsort = real_argsort
        return out, log.calls, argsorts, (x, sel, w)

    def reference(self, x, sel, w):
        ref = torch.zeros((x.size(0), HIDDEN), dtype = torch.float64)
        for t in range(x.size(0)):
            for k in range(TOPK):
                ref[t] += self.world.contrib(x[t], int(sel[t, k]), w[t, k]).double()
        return ref

    def test_default_off_is_identical(self):
        cases = [(512, True), (2048, True), (1024, False), (256, True)]
        for T, e3 in cases:
            with self.subTest(T = T, e3 = e3):
                ob, cb, ab, _ = self.run_layer(self.base, T, e3, False)
                on, cn, an, _ = self.run_layer(self.new, T, e3, False)
                self.assertEqual(cb, cn, "extension call sequence/arguments differ from the baseline")
                self.assertEqual(ab, an)
                self.assertTrue(all(not s for s in an), "default path must use the default argsort")
                self.assertTrue(torch.equal(ob, on))
                self.assertTrue(any(c[0] == ("exl3_moe_prefill_e3" if (e3 and T >= 512) else "exl3_moe") for c in cn))

    def test_flag_without_e3_is_identical(self):
        for T, e3 in [(1024, False), (256, True)]:
            with self.subTest(T = T, e3 = e3):
                ob, cb, ab, _ = self.run_layer(self.base, T, e3, False)
                on, cn, an, _ = self.run_layer(self.new, T, e3, True)
                self.assertEqual(cb, cn)
                self.assertEqual(ab, an)
                self.assertTrue(torch.equal(ob, on))

    def test_det_path(self):
        for T, mode, thin in [(512, "zipf", 32), (2048, "zipf", 32), (1024, "hot", 32),
                              (512, "uniform", 16), (1024, "zipf", 1), (512, "zipf", 64)]:
            with self.subTest(T = T, mode = mode, thin = thin):
                out, calls, argsorts, (x, sel, w) = self.run_layer(self.new, T, True, True, mode, thin = thin)
                names = [c[0] for c in calls]
                expect = ["exl3_moe_prefill_e3_det", "exl3_moe"] + (["exl3_moe"] if thin > 16 else []) + \
                         ["exl3_moe_prefill_e3_det_reduce"]
                self.assertEqual(names, expect)
                self.assertEqual(argsorts, [True], "DET path must sort stably")
                # thin windows run in slot mode with fused_base = expert start
                for c in calls[1:-1]:
                    self.assertEqual(c[1][17][0], "T")  # output_scratch present
                    self.assertEqual(c[1][17][1], "torch.float32")
                    self.assertEqual(c[1][18][1], "torch.int64")
                wins = [(c[1][19][1], c[1][20][1], c[1][21][1]) for c in calls[1:-1]]
                self.assertEqual(wins, [("1", str(min(16, thin)), "16")] + ([("17", str(thin), "32")] if thin > 16 else []))
                ref = self.reference(x, sel, w)
                err = (out.double() - ref).abs().max().item()
                self.assertLess(err, 1e-4 * max(1.0, ref.abs().max().item()), f"max abs err {err}")
                # atomic E3 emulation agrees with the reference too
                out_a, calls_a, _, _ = self.run_layer(self.new, T, True, False, mode, thin = thin)
                self.assertLess((out_a.double() - ref).abs().max().item(), 1e-4 * max(1.0, ref.abs().max().item()))

    def test_det_independent_of_fat_order(self):
        outs = [self.run_layer(self.new, 1024, True, True, "zipf", fat_order_seed = s)[0] for s in (None, 1, 2, 3)]
        for o in outs[1:]:
            self.assertTrue(torch.equal(outs[0], o))


if __name__ == "__main__":
    unittest.main(verbosity = 2)
