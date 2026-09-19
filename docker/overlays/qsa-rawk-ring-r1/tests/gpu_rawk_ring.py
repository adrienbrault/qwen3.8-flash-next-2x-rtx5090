#!/usr/bin/env python3
"""GPU harness for qsa-rawk-ring r1 (EXL3_QSA_RAWK_RING). Runs inside the patched image, one card per run.

Flag off vs flag on, torch.equal throughout:

1. kernels: the generator-shaped schedules of rawk_scenarios.py with the real Triton kernels. The served pair
   (_mla_plane_update_kernel + _qsa_pool_update_kernel, served order) against the ring kernels in the BC-graph order
   (ring append, ring pool) and the eager order (qsa_ring_plane_update): decode, MTP verify with random acceptance and
   the draft-cache pattern, q_len 16 graph windows, eager windows at the guard limit, stop-string rewind + replay,
   bsz 3 batches, prefix sharing, mid-block chunk starts, full-width rope. Pooled planes after every call.
   Two negative controls (a ring of 8 rows under q_len 16; an eager window past the guard) must MISMATCH.
2. update_planes: the real QSAIndexer.update_planes (served branch on a full-plane layer, ring branch on a ring
   layer; projection and norms stubbed with fixed random weights) over a 5,000-token chunked prefill, MTP verify
   windows with rewinds (recurrent_history set), prefix reuse from shared full pages, a continuation after a
   mid-block prefill end. At checkpoints: pooled planes, the returned indexer queries, select_indices_paged indices
   and sparse_attend outputs over shared random K/V. The ring branch must refuse an 18-token verify window.
3. aot: bc_attn._qsa_ring_graph_kernels (the exact AOT compile a graph slot does) for q_len 1, 4, 16 and rotary
   widths 64 and 128; q_len 17 must compile (20 rows) and q_len 18 must be refused.
4. cache_layer: CacheLayer_qsa_quant allocated with the selector off and on (shapes, storage, copy_page).
5. bench_us (informational): plane upkeep per call, served vs ring, eager 2,048-token chunk and q1/q4.

Exit 0 only if every check passes. --json writes the full record.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rawk_scenarios as sc   # noqa: E402


def real_kernels():
    from exllamav3.modules.attention_fn import qsa_triton as qt
    from exllamav3.modules.attention_fn.mla_triton import _mla_plane_update_kernel
    return SimpleNamespace(
        mla = _mla_plane_update_kernel,
        pool = qt._qsa_pool_update_kernel,
        ring_append = qt._qsa_raw_ring_append_kernel,
        ring_bc_pool = qt._qsa_pool_update_ring_bc_kernel,
        ring_plane_update = qt.qsa_ring_plane_update,
        ctx = lambda dev: torch.cuda.device(dev),
    )


# ---- 2. update_planes end to end ----------------------------------------------------------------------------------

class Rope:
    """The pieces of RoPE update_planes touches: queries are roped in place (identical in both arms, a fixed
    rotation keeps the selection non-trivial), the pool kernels read inv_freq and attn_factor."""

    def __init__(self, dev, rope_r = 64):
        self.inv_freq = (1.0 / (10000.0 ** (torch.arange(0, rope_r, 2, dtype = torch.float32) / rope_r))).to(dev)
        self.attn_factor = 1.0

    def apply(self, q, k, position, positions, position_ids, in_place, *a):
        q.mul_(1.0)


def make_indexer(dev, hidden, seed):
    from exllamav3.modules.qsa_indexer import QSAIndexer
    g = torch.Generator().manual_seed(seed)
    idx = QSAIndexer.__new__(QSAIndexer)
    idx.n_heads, idx.head_dim, idx.compress_ratio = 4, 128, 4
    idx.token_budget, idx.block_topk = 2048, 512
    idx.scale = 1.0 / math.sqrt(128)
    w = (torch.randn((hidden, 5 * 128), generator = g) / math.sqrt(hidden)).half().to(dev)
    idx.index_qk_proj = SimpleNamespace(forward = lambda x, params: (x.reshape(-1, hidden) @ w).view(*x.shape[:-1], -1))
    idx.q_layernorm = SimpleNamespace(weight = SimpleNamespace(data = (torch.randn(128, generator = g) * 0.1).half().to(dev)),
                                      rms_norm_eps = 1e-6)
    idx.k_layernorm = SimpleNamespace(weight = SimpleNamespace(data = (torch.randn(128, generator = g) * 0.1).half().to(dev)),
                                      rms_norm_eps = 1e-6)
    return idx


def make_layer(dev, pages, ring, kv):
    rows = ring or sc.PAGE_SIZE
    return SimpleNamespace(
        raw_k = torch.zeros((pages, rows, 128), dtype = torch.half, device = dev),
        pooled = torch.zeros((pages, sc.PAGE_SIZE // 4, 128), dtype = torch.half, device = dev),
        raw_ring_rows = ring,
        k = kv[0], v = kv[1],
    )


def run_update_planes(dev, ring, seed = 31):
    from exllamav3.modules.qsa_indexer import QSAIndexer   # noqa: F401
    hidden = 256
    g = torch.Generator().manual_seed(seed)
    idx = make_indexer(dev, hidden, seed)
    rope = Rope(dev)
    pages = 64
    kv = [(torch.randn((pages, sc.PAGE_SIZE, 2, 256), generator = g) * 0.5).half().to(dev) for _ in range(2)]
    full = make_layer(dev, pages, 0, kv)
    rg = make_layer(dev, pages, ring, kv)
    attn = SimpleNamespace(num_q_heads = 16, num_kv_heads = 2, head_dim = 256, sm_scale = 1.0 / 16.0)
    perm = torch.randperm(pages, generator = g).tolist()
    tables = {"a": perm[:24], "b": perm[:12] + perm[24:36]}      # b shares a's first 12 full pages
    rec = {"calls": 0, "checks": 0, "mismatch": [], "guard": None}

    def call(names, pos0s, length, verify = False, check = False):
        x = torch.randn((len(names), length, hidden), generator = g).half().to(dev)
        bt = torch.tensor([tables[n] for n in names], dtype = torch.int32, device = dev)
        cpu = torch.tensor(pos0s, dtype = torch.int32)
        params = {"recurrent_history": True} if verify else {}
        qf = idx.update_planes(full, x, rope, bt, cpu, params).clone()
        qr = idx.update_planes(rg, x, rope, bt, cpu, params).clone()
        rec["calls"] += 1
        tag = (tuple(names), tuple(pos0s), length)
        if not torch.equal(full.pooled, rg.pooled):
            rec["mismatch"].append(("pooled", tag))
        if not torch.equal(qf, qr):
            rec["mismatch"].append(("q_idx", tag))
        if check and max(pos0s) + length > idx.sparse_threshold():
            rec["checks"] += 1
            sf = idx.select_indices_paged(full, qf, bt, cpu)
            sr = idx.select_indices_paged(rg, qr, bt, cpu)
            if not torch.equal(sf, sr):
                rec["mismatch"].append(("selection", tag))
            q = torch.randn((len(names), length, 16, 256), generator = g).half().to(dev)
            of = idx.sparse_attend(full, attn, q, qf, bt, cpu)
            orr = idx.sparse_attend(rg, attn, q, qr, bt, cpu)
            if not torch.equal(of, orr):
                rec["mismatch"].append(("attention", tag))

    # chunked prefill (page-aligned chunk ends, the last at len - 1 = mid-block)
    pos, end = 0, 5003
    while pos < end:
        stop = min((pos + 2048) // 256 * 256, end)
        call(["a"], [pos], stop - pos)
        pos = stop
    kv_a = end
    rng = torch.Generator().manual_seed(seed + 1)
    for r in range(40):
        call(["a"], [kv_a], 4, verify = True, check = (r % 5 == 0))
        kv_a += int(torch.randint(1, 5, (1,), generator = rng))
    # q_len 16 windows (graph-sized, eager here) with acceptance down to one token
    for r in range(10):
        call(["a"], [kv_a], 16, verify = True, check = (r % 3 == 0))
        kv_a += int(torch.randint(1, 17, (1,), generator = rng))
    # prefix reuse: b continues after the 12 shared full pages, then both decode in one batch
    kv_b = 12 * 256
    call(["b"], [kv_b], 301)
    kv_b += 301
    for r in range(12):
        call(["a", "b"], [kv_a, kv_b], 1, check = (r % 4 == 0))
        kv_a += 1
        kv_b += 1
    # the guard: an 18-token verify window on the ring layer
    x = torch.randn((1, ring - 4 + 2, hidden), generator = g).half().to(dev)
    bt = torch.tensor([tables["a"]], dtype = torch.int32, device = dev)
    try:
        idx.update_planes(rg, x, rope, bt, torch.tensor([kv_a], dtype = torch.int32), {"recurrent_history": True})
        rec["guard"] = "not raised"
    except RuntimeError as e:
        rec["guard"] = "raised" if "EXL3_QSA_RAWK_RING" in str(e) else f"other: {e}"
    rec["pass"] = not rec["mismatch"] and rec["guard"] == "raised" and rec["checks"] >= 8
    return rec


# ---- 3. AOT compile of the graph kernels -------------------------------------------------------------------------

def run_aot(dev, ring):
    from exllamav3.modules.attention_fn.bc_attn import _qsa_ring_graph_kernels
    out = {}
    ok = True
    for rope_r in (64, 128):
        for q in (1, 4, 16):
            try:
                a, p = _qsa_ring_graph_kernels(dev, 128, 4, rope_r, 1.0, 1e-6, q, ring)
                out[f"r{rope_r}_q{q}"] = "ok" if (a is not None and p is not None) else "none"
            except Exception as e:           # noqa: BLE001
                out[f"r{rope_r}_q{q}"] = f"error: {type(e).__name__}: {e}"
            ok &= out[f"r{rope_r}_q{q}"] == "ok"
    # q_len + cr - 1 <= ring: 17 + 3 = 20 rows is the largest window a 20-row ring holds (R544 try 1 expected q17 to be
    # refused, which contradicts the guard); 18 must be refused
    try:
        a, p = _qsa_ring_graph_kernels(dev, 128, 4, 64, 1.0, 1e-6, 17, ring)
        out["q17"] = "ok" if (a is not None and p is not None) else "none"
    except Exception as e:           # noqa: BLE001
        out["q17"] = f"error: {type(e).__name__}: {e}"
    ok &= out["q17"] == "ok"
    try:
        _qsa_ring_graph_kernels(dev, 128, 4, 64, 1.0, 1e-6, 18, ring)
        out["q18"] = "not refused"
        ok = False
    except AssertionError:
        out["q18"] = "refused"
    out["pass"] = ok
    return out


# ---- 4. the real cache layer class -------------------------------------------------------------------------------

def run_cache_layer(dev):
    import exllamav3.cache.qsa as cq
    att = SimpleNamespace(num_kv_heads = 2, head_dim = 256, qsa_indexer = SimpleNamespace(head_dim = 128, compress_ratio = 4))
    out = {"default_selector": cq._rawk_ring}
    saved = cq._rawk_ring
    try:
        layers = {}
        for flag in (False, True):
            cq._rawk_ring = flag
            a = cq.CacheLayer_qsa_quant(None, att, 0, 8 * 256, 8, 8)
            a.alloc(torch.device(dev))
            b = cq.CacheLayer_qsa_quant(None, att, 0, 8 * 256, 8, 8)
            b.alloc(torch.device(dev))
            layers[flag] = (a, b)
            out[f"flag_{int(flag)}"] = {
                "ring": a.raw_ring_rows, "raw_shape": list(a.raw_k.shape), "storage": int(a.storage_size()),
            }
        a, b = layers[True]
        a.pooled.normal_()
        b.copy_page(a, 1, 2, 128)
        aligned = torch.equal(b.pooled[2, :32], a.pooled[1, :32])
        try:
            b.copy_page(a, 1, 3, 130)
            refused = False
        except RuntimeError:
            refused = True
        out["copy_block_aligned"] = aligned
        out["copy_misaligned_refused"] = refused
    finally:
        cq._rawk_ring = saved
    out["pass"] = (
        out["default_selector"] is False and out["flag_0"]["ring"] == 0 and out["flag_0"]["raw_shape"] == [8, 256, 128]
        and out["flag_1"]["ring"] == 20 and out["flag_1"]["raw_shape"] == [8, 20, 128]
        and out["flag_0"]["storage"] - out["flag_1"]["storage"] == 8 * 236 * 128 * 2
        and out["copy_block_aligned"] and out["copy_misaligned_refused"]
    )
    return out


# ---- 5. timing (informational) ------------------------------------------------------------------------------------

def run_bench(dev, K, ring):
    out = {}
    for label, length, bsz in (("eager_2048", 2048, 1), ("q1_bsz4", 1, 4), ("q4_bsz4", 4, 4)):
        for arm in ("served", "ring_eager", "ring_bc"):
            w = sc.World(K, dev, 64, ring, seed = 41)
            names = [f"s{i}" for i in range(bsz)]
            for n in names:
                w.new_seq(n, 12)
            pos0s = [1001 + 7 * i for i in range(bsz)]
            kraw = torch.randn((bsz, length, 128), device = dev).half()
            bt = w._bt(names)
            seqlens = torch.tensor(pos0s, dtype = torch.int32, device = dev)
            npr = bt.shape[1]

            def once():
                with torch.cuda.device(dev):
                    if arm == "served":
                        K.mla[(bsz * length,)](kraw, w.ref.raw, bt, seqlens, npr, length,
                                               page_size = 256, D = 128, DST_D = 0, DST_OFF = 0)
                        K.pool[(bsz, length // 4 + 1)](w.ref.raw.view(-1, 128), w.ref.pooled.view(-1, 128), w.k_norm_w,
                                                       w.inv_freq, bt, seqlens, npr, length, page_size = 256, P = 4,
                                                       D = 128, ROPE_R = 64, attn_factor = 1.0, eps = 1e-6, MAXPOOLS = 1)
                    elif arm == "ring_eager":
                        K.ring_plane_update(kraw, w.rg.raw, w.rg.pooled, w.k_norm_w, w.inv_freq, 1.0, 1e-6, bt, seqlens, 4)
                    else:
                        if length > 16:
                            return
                        K.ring_append[(bsz * length,)](kraw, w.rg.raw, bt, seqlens, npr, length,
                                                       page_size = 256, D = 128, RING = ring)
                        K.ring_bc_pool[(bsz, length // 4 + 1)](w.rg.raw.view(-1, 128), w.rg.pooled.view(-1, 128),
                                                               w.k_norm_w, w.inv_freq, bt, seqlens, npr, length,
                                                               page_size = 256, P = 4, D = 128, ROPE_R = 64,
                                                               attn_factor = 1.0, eps = 1e-6,
                                                               MAXPOOLS = length // 4 + 1, RING = ring)
            for _ in range(5):
                once()
            torch.cuda.synchronize(dev)
            n = 50
            t = time.perf_counter()
            for _ in range(n):
                once()
            torch.cuda.synchronize(dev)
            out[f"{label}_{arm}"] = round((time.perf_counter() - t) / n * 1e6, 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--json", default = None)
    ap.add_argument("--no-bench", action = "store_true")
    args = ap.parse_args()
    dev = args.device
    torch.cuda.set_device(dev)

    from exllamav3.cache.qsa import rawk_ring_rows
    ring = rawk_ring_rows(4)
    K = real_kernels()
    rec = {"device": dev, "gpu": torch.cuda.get_device_name(dev), "ring_rows": ring, "torch": torch.__version__}
    try:
        import triton
        rec["triton"] = triton.__version__
    except Exception:                        # noqa: BLE001
        pass

    rec["scenarios"] = [f(K, dev, ring) for f in sc.POSITIVE]
    rec["controls"] = [f(K, dev, ring) for f in sc.NEGATIVE]
    rec["update_planes"] = run_update_planes(dev, ring)
    rec["aot"] = run_aot(dev, ring)
    rec["cache_layer"] = run_cache_layer(dev)
    if not args.no_bench:
        rec["bench_us"] = run_bench(dev, K, ring)

    failures = []
    failures += [s["name"] for s in rec["scenarios"] if not s["pooled_equal"]]
    failures += [f"{c['name']} (control did not mismatch)" for c in rec["controls"] if c["pooled_equal"]]
    for key in ("update_planes", "aot", "cache_layer"):
        if not rec[key]["pass"]:
            failures.append(key)
    rec["failures"] = failures
    rec["pass"] = not failures

    text = json.dumps(rec, indent = 2, default = str)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)
    print("PASS" if rec["pass"] else f"FAIL: {failures}")
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
