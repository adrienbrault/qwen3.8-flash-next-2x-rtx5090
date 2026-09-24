#!/usr/bin/env python3
"""densegemm r2 GPU parity (r1's test; r2 changes only the gemm / mgemm twin internals, revision 2): the V2 dense kernels (EXL3_DENSE_V2 / EXL3_DENSE_ROWS32) against the served kernels,
bitwise, on the REAL weights of every dense K=4 projection of the served checkpoint, in-process, one card.

Entry points exactly as the model calls them (d0_microbench.py docstring, R682):
  gemm  : ext.exl3_gemm(x, trellis, y, suh, xh, svh, -1, False, True, 0)       GDN out_proj, attn o_proj,
          index_qk_proj, shared-expert down (and in_proj_qkv / in_proj_z standalone, the >32-row fallback)
  mgemm : ext.exl3_mgemm(...) sliced                                           GDN in_proj qkv+z (8 slices),
          attention q+k+v (26 slices)
          ext.exl3_mgemm(...) per matrix, one input                            shared-expert gate+up
Rows: 1, 2, 3, 4, 8, 12, 16 (served decode shapes are 4 = c1d3 and 16 = c4d3 / c8d1) and 17, 24, 32 (rows32).
Output dtypes: fp16 and fp32 for every family.

Per cell: the served kernel runs first (EXL3_DENSE_V2 off; this also fills the autotune cache the V2 dispatch reads),
then V2 (mode 1 + rows32 on), then V2 again; all three outputs must be bitwise equal. A cell counts as ENGAGED when
the V2 launch counters moved (rows 1-2 go to the int8 GEMV and are not engaged by design).

Also: 200-launch interleaved stress, a torch.cuda.CUDAGraph capture/replay of V2 launches against eager served
outputs, the workspace VRAM delta. Prints PARITY PASS / PARITY FAIL; --json writes every cell.

  python3 test_densegemm_parity.py --model /models/<ckpt> --json /results/parity.json [--layers 4]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "d0"))

HIDDEN = 2560
ROWS = [1, 2, 3, 4, 8, 12, 16, 17, 24, 32]
FAILS = []
CELLS = []
STATE = {"arm": None}
# (family) served at 4 and 16 rows (R680 inventory, ANALYSIS.md section 1); V2 must engage there. The standalone
# in_proj_qkv / in_proj_z gemm cells are not served calls (the served path is the sliced mgemm): log only
SERVED_FAMILIES = {"gemm/gdn.out_proj", "gemm/attn.o_proj", "gemm/attn.index_qk_proj", "gemm/shared.down",
                   "mgemm/gdn.in_proj_qkvz", "mgemm/attn.qkv", "mgemm/shared.gate_up"}


def guarded(fn, fam, name, rows, odt, *args):
    """A combination the served path itself rejects (TORCH_CHECK on the OFF arm) is skipped and listed;
    an exception on a V2 arm is a failure."""
    import torch
    try:
        fn(*args)
    except RuntimeError as e:
        torch.cuda.synchronize()
        arm = STATE["arm"]
        cell = dict(family=fam, name=name, rows=rows, dtype=str(odt).replace("torch.", ""), equal=arm == "off",
                    engaged=False, note=f"{'served path rejects' if arm == 'off' else 'V2 arm raised'}: {str(e)[:200]}")
        CELLS.append(cell)
        if arm != "off":
            FAILS.append(cell)
        log("EXCEPTION", arm, cell)


def log(*a):
    print("[parity]", *a, flush=True)


def bits(t):
    import torch
    return t.contiguous().view(torch.int16 if t.dtype == torch.half else torch.int32)


def same(a, b):
    import torch
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--json")
    ap.add_argument("--layers", type=int, default=0, help="limit layers per family (0 = all)")
    ap.add_argument("--mode", type=int, default=1, help="EXL3_DENSE_V2 value of the ON arm")
    a = ap.parse_args()

    import torch
    import exllamav3_ext as ext
    from d0_microbench import Checkpoint

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    assert ext.dense_v2_revision == 2
    log("modes at entry (env):", tuple(ext.dense_v2_modes()))
    ck = Checkpoint(a.model, dev)
    keys = set(ck.map)

    def has(prefix):
        return f"{prefix}.trellis" in keys

    def K_of(tr):
        return tr.shape[-1] // 16

    def lin(prefix):
        """(trellis, suh, svh) of a mul1 linear, or None (no mul1 codebook: V2 does not cover it)."""
        if f"{prefix}.mul1" not in keys:
            log(f"skip {prefix}: not mul1")
            return None
        return ck.linear(prefix)

    layers = list(range(48))
    gdn = [i for i in layers if i % 4 != 3]
    att = [i for i in layers if i % 4 == 3]
    if a.layers:
        gdn, att, layers = gdn[:a.layers], att[:a.layers], layers[:a.layers]
    P = "model.language_model.layers"
    extra_attn = ["mtp.layers.0.self_attn"] if has("mtp.layers.0.self_attn.q_proj") else []

    counts = lambda: tuple(ext.dense_v2_launch_counts())

    def set_off():
        ext.dense_v2_set_mode(0, 0)

    def set_on():
        ext.dense_v2_set_mode(a.mode, 1)

    free0 = torch.cuda.mem_get_info(dev)[0]

    def record(fam, name, rows, dtype, eq, engaged, note=""):
        cell = dict(family=fam, name=name, rows=rows, dtype=str(dtype).replace("torch.", ""), equal=eq,
                    engaged=engaged, note=note)
        CELLS.append(cell)
        if not eq:
            FAILS.append(cell)
            log("MISMATCH", cell)

    # ------------------------------------------------------------------------------------------ gemm
    def gemm_cell(fam, name, lin, rows, odt):
        tr, suh, svh = lin
        k, n = tr.shape[0] * 16, tr.shape[1] * 16
        g = torch.Generator(device=dev).manual_seed(hash((name, rows, str(odt))) & 0x7fffffff)
        x = (torch.randn((rows, k), device=dev, generator=g) * 0.5).half()
        outs = []
        for arm in ("off", "on", "on2"):
            STATE["arm"] = arm
            (set_off if arm == "off" else set_on)()
            y = torch.full((rows, n), float("nan"), dtype=odt, device=dev)
            xh = torch.empty_like(x)
            c0 = counts()
            ext.exl3_gemm(x, tr, y, suh, xh, svh, -1, False, True, 0)
            torch.cuda.synchronize()
            c1 = counts()
            outs.append((y, c1[0] - c0[0] + c1[1] - c0[1]))
        eq = same(outs[0][0], outs[1][0]) and same(outs[1][0], outs[2][0])
        record(fam, name, rows, odt, eq, outs[1][1] > 0)

    def gemm_family(fam, prefixes):
        n_lin = 0
        for pre in prefixes:
            if not has(pre):
                continue
            L = lin(pre)
            if L is None:
                continue
            if K_of(L[0]) != 4:
                log(f"skip {pre}: K={K_of(L[0])}")
                continue
            n_lin += 1
            for rows in ROWS:
                for odt in (torch.half, torch.float):
                    guarded(gemm_cell, fam, pre, rows, odt, fam, pre, L, rows, odt)
            del L
        log(f"{fam}: {n_lin} K4 linears")

    gemm_family("gemm/gdn.out_proj", [f"{P}.{i}.linear_attn.out_proj" for i in gdn])
    gemm_family("gemm/attn.o_proj", [f"{P}.{i}.self_attn.o_proj" for i in att] + [f"{p}.o_proj" for p in extra_attn])
    gemm_family("gemm/attn.index_qk_proj", [f"{P}.{i}.self_attn.indexer.index_qk_proj" for i in att])
    gemm_family("gemm/shared.down", [f"{P}.{i}.mlp.shared_expert.down_proj" for i in layers])
    gemm_family("gemm/gdn.in_proj_qkv", [f"{P}.{i}.linear_attn.in_proj_qkv" for i in gdn[:4]])
    gemm_family("gemm/gdn.in_proj_z", [f"{P}.{i}.linear_attn.in_proj_z" for i in gdn[:4]])

    # ------------------------------------------------------------------------------------------ sliced mgemm
    def sliced(srcs, rows, odt, name, fam):
        K = K_of(srcs[0][0])
        k = srcs[0][0].shape[0] * 16
        widths = [s[0].shape[1] * 16 for s in srcs]
        width = math.gcd(*widths)
        g = torch.Generator(device=dev).manual_seed(hash((name, rows, str(odt))) & 0x7fffffff)
        x3 = (torch.randn((1, rows, k), device=dev, generator=g) * 0.5).half()
        tp, sp, tg, off, st, hs = [], [], [], [], [], []
        for i, (tr, suh, svh) in enumerate(srcs):
            n_i = tr.shape[1] * 16
            for n0 in range(0, n_i, width):
                tp.append(tr.data_ptr() + (n0 // 16) * 16 * K * tr.element_size())
                sp.append(svh.data_ptr() + n0 * svh.element_size())
                tg.append(i); off.append(n0); st.append(n_i); hs.append(i)
        S = len(tg)
        T = lambda v, dt: torch.tensor(v, dtype=dt, device=dev)
        ptr_tr, ptr_svh = T(tp, torch.long), T(sp, torch.long)
        ptr_suh = T([s[1].data_ptr() for s in srcs], torch.long)
        size_n, n_stride, had_src = T([width] * S, torch.int32), T(st, torch.int32), T(hs, torch.int32)
        esz = 2 if odt == torch.half else 4
        outs = []
        for arm in ("off", "on", "on2"):
            STATE["arm"] = arm
            (set_off if arm == "off" else set_on)()
            o = [torch.full((rows, w), float("nan"), dtype=odt, device=dev) for w in widths]
            c_ptrs = T([o[tg[j]].data_ptr() + off[j] * esz for j in range(S)], torch.long)
            carrier = torch.zeros((S, 1, width), dtype=odt, device=dev).expand(S, rows, width)
            xh = torch.empty((len(srcs), rows, k), dtype=torch.half, device=dev)
            c0 = counts()
            ext.exl3_mgemm(x3, ptr_tr, carrier, ptr_suh, xh, ptr_svh, None, None, K, -1, False, True, -1, -1, 0, 1,
                           size_n, c_ptrs, n_stride, had_src, len(srcs))
            torch.cuda.synchronize()
            c1 = counts()
            outs.append((o, c1[0] - c0[0]))
        eq = all(same(p, q) and same(q, r) for p, q, r in zip(outs[0][0], outs[1][0], outs[2][0]))
        record(fam, name, rows, odt, eq, outs[1][1] > 0)

    def sliced_family(fam, groups):
        n_g = 0
        for name, prefixes in groups:
            if not all(has(p) for p in prefixes):
                continue
            srcs = [lin(p) for p in prefixes]
            if any(s is None for s in srcs):
                continue
            if any(K_of(s[0]) != 4 for s in srcs):
                log(f"skip {name}: K={[K_of(s[0]) for s in srcs]}")
                continue
            n_g += 1
            for rows in ROWS:
                for odt in (torch.float, torch.half):
                    guarded(sliced, fam, name, rows, odt, srcs, rows, odt, name, fam)
            del srcs
        log(f"{fam}: {n_g} groups")

    sliced_family("mgemm/gdn.in_proj_qkvz",
                  [(f"{P}.{i}.linear_attn.in_proj", [f"{P}.{i}.linear_attn.in_proj_qkv", f"{P}.{i}.linear_attn.in_proj_z"])
                   for i in gdn])
    sliced_family("mgemm/attn.qkv",
                  [(f"{p}", [f"{p}.q_proj", f"{p}.k_proj", f"{p}.v_proj"])
                   for p in [f"{P}.{i}.self_attn" for i in att] + extra_attn])

    # ------------------------------------------------------------------------------------------ per-matrix mgemm
    def gate_up(pre, rows, odt):
        gl, ul = lin(f"{pre}.gate_proj"), lin(f"{pre}.up_proj")
        if gl is None or ul is None or K_of(gl[0]) != 4 or K_of(ul[0]) != 4:
            return None
        k, n = gl[0].shape[0] * 16, gl[0].shape[1] * 16
        g = torch.Generator(device=dev).manual_seed(hash((pre, rows, str(odt))) & 0x7fffffff)
        x3 = (torch.randn((1, rows, k), device=dev, generator=g) * 0.5).half()
        T = lambda v: torch.tensor(v, dtype=torch.long, device=dev)
        ptr_tr, ptr_suh, ptr_svh = T([gl[0].data_ptr(), ul[0].data_ptr()]), T([gl[1].data_ptr(), ul[1].data_ptr()]), \
            T([gl[2].data_ptr(), ul[2].data_ptr()])
        outs = []
        for arm in ("off", "on", "on2"):
            STATE["arm"] = arm
            (set_off if arm == "off" else set_on)()
            c = torch.full((2, rows, n), float("nan"), dtype=odt, device=dev)
            xh = torch.empty((2, rows, k), dtype=torch.half, device=dev)
            c0 = counts()
            ext.exl3_mgemm(x3, ptr_tr, c, ptr_suh, xh, ptr_svh, None, None, 4, -1, False, True, -1, -1, 0)
            torch.cuda.synchronize()
            c1 = counts()
            outs.append((c, c1[0] - c0[0]))
        eq = same(outs[0][0], outs[1][0]) and same(outs[1][0], outs[2][0])
        record("mgemm/shared.gate_up", pre, rows, odt, eq, outs[1][1] > 0)
        return True

    n_su = 0
    for i in layers:
        pre = f"{P}.{i}.mlp.shared_expert"
        if not has(f"{pre}.gate_proj"):
            continue
        for rows in ROWS:
            for odt in (torch.half, torch.float):
                STATE["ok"] = False
                guarded(lambda: STATE.__setitem__("ok", bool(gate_up(pre, rows, odt))),
                        "mgemm/shared.gate_up", pre, rows, odt)
                n_su += STATE["ok"]
    log(f"mgemm/shared.gate_up: {n_su} cells")

    # ------------------------------------------------------------------------------------------ engagement summary
    fams = sorted({c["family"] for c in CELLS})
    for f in fams:
        cs = [c for c in CELLS if c["family"] == f]
        for rows in ROWS:
            rc = [c for c in cs if c["rows"] == rows]
            if not rc:
                continue
            n_eq = sum(c["equal"] for c in rc)
            n_en = sum(c["engaged"] for c in rc)
            log(f"{f:28s} rows {rows:2d}: {n_eq}/{len(rc)} equal, V2 engaged {n_en}/{len(rc)}")
            if rows in (4, 16) and n_en == 0:
                if f in SERVED_FAMILIES:
                    FAILS.append(dict(family=f, rows=rows, note="V2 never engaged at a served decode shape"))
                    log(f"NOT ENGAGED at served rows {rows}: {f}")
                else:
                    log(f"not engaged at rows {rows}: {f} (not a served call; bitwise equality still required)")

    # ------------------------------------------------------------------------------------------ stress
    set_on()
    lin_o = lin(f"{P}.{gdn[0]}.linear_attn.out_proj")
    srcs = [lin(f"{P}.{gdn[0]}.linear_attn.in_proj_qkv"), lin(f"{P}.{gdn[0]}.linear_attn.in_proj_z")]
    assert lin_o is not None and all(s_ is not None for s_ in srcs), "stress/graph need mul1 GDN linears"

    def mk_gemm(rows, odt, seed):
        tr, suh, svh = lin_o
        g = torch.Generator(device=dev).manual_seed(seed)
        x = (torch.randn((rows, tr.shape[0] * 16), device=dev, generator=g) * 0.5).half()
        y = torch.empty((rows, tr.shape[1] * 16), dtype=odt, device=dev)
        xh = torch.empty_like(x)
        return lambda: ext.exl3_gemm(x, tr, y, suh, xh, svh, -1, False, True, 0), y, x

    def mk_qkvz(rows, seed):
        K, k = 4, HIDDEN
        widths = [s[0].shape[1] * 16 for s in srcs]
        width = math.gcd(*widths)
        g = torch.Generator(device=dev).manual_seed(seed)
        x3 = (torch.randn((1, rows, k), device=dev, generator=g) * 0.5).half()
        tp, sp, tg, off, st = [], [], [], [], []
        for i, (tr, suh, svh) in enumerate(srcs):
            n_i = tr.shape[1] * 16
            for n0 in range(0, n_i, width):
                tp.append(tr.data_ptr() + (n0 // 16) * 16 * K * tr.element_size())
                sp.append(svh.data_ptr() + n0 * svh.element_size())
                tg.append(i); off.append(n0); st.append(n_i)
        S = len(tg)
        T = lambda v, dt: torch.tensor(v, dtype=dt, device=dev)
        o = [torch.empty((rows, w), dtype=torch.float, device=dev) for w in widths]
        args = (x3, T(tp, torch.long), torch.zeros((S, 1, width), dtype=torch.float, device=dev).expand(S, rows, width),
                T([s[1].data_ptr() for s in srcs], torch.long), torch.empty((2, rows, k), dtype=torch.half, device=dev),
                T(sp, torch.long), None, None, K, -1, False, True, -1, -1, 0, 1, T([width] * S, torch.int32),
                T([o[tg[j]].data_ptr() + off[j] * 4 for j in range(S)], torch.long), T(st, torch.int32),
                T([t_ for t_ in tg], torch.int32), 2)
        return lambda: ext.exl3_mgemm(*args), o, x3

    cases = [mk_gemm(16, torch.float, 11), mk_gemm(4, torch.float, 12), mk_gemm(16, torch.half, 13), mk_qkvz(16, 14),
             mk_qkvz(4, 15), mk_gemm(24, torch.float, 16)]
    refs = []
    set_off()
    for fn, out, _ in cases:
        fn(); torch.cuda.synchronize()
        refs.append([t.clone() for t in (out if isinstance(out, list) else [out])])
    set_on()
    bad = 0
    for it in range(200):
        j = (it * 7) % len(cases)
        fn, out, _ = cases[j]
        fn()
        if it % 10 == 9 or it >= 190:
            torch.cuda.synchronize()
            outs = out if isinstance(out, list) else [out]
            bad += sum(not same(p, q) for p, q in zip(outs, refs[j]))
    torch.cuda.synchronize()
    log(f"stress: 200 interleaved V2 launches over {len(cases)} cells, mismatches at checkpoints: {bad}")
    if bad:
        FAILS.append(dict(family="stress", note=f"{bad} mismatches"))

    # ------------------------------------------------------------------------------------------ CUDA graph
    set_on()
    gcases = [mk_gemm(16, torch.float, 21), mk_qkvz(16, 22), mk_gemm(4, torch.float, 23), mk_qkvz(4, 24)]
    for fn, _, _ in gcases:
        fn()    # eager warm-up: autotune entries + workspace exist before capture
    torch.cuda.synchronize()
    c0 = counts()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for fn, _, _ in gcases:
            fn()
    torch.cuda.synchronize()
    gbad = 0
    for rep in range(20):
        for _, _, x in gcases:
            x.copy_((torch.randn(x.shape, device=dev) * 0.5).half())
        graph.replay()
        torch.cuda.synchronize()
        got = [[t.clone() for t in (o if isinstance(o, list) else [o])] for _, o, _ in gcases]
        set_off()
        for fn, _, _ in gcases:
            fn()
        torch.cuda.synchronize()
        want = [[t.clone() for t in (o if isinstance(o, list) else [o])] for _, o, _ in gcases]
        set_on()
        gbad += sum(not same(p, q) for g_, w_ in zip(got, want) for p, q in zip(g_, w_))
    c1 = counts()
    log(f"graph: captured {len(gcases)} V2 launches (counters +{c1[0] - c0[0]} gemm/mgemm, +{c1[1] - c0[1]} gemv "
        f"incl. capture), 20 replays vs eager served: mismatches {gbad}")
    if gbad:
        FAILS.append(dict(family="graph", note=f"{gbad} mismatches"))
    del graph

    free1 = torch.cuda.mem_get_info(dev)[0]
    log(f"VRAM: free before {free0 / 2**20:.1f} MiB, after {free1 / 2**20:.1f} MiB (includes torch cache; the V2 "
        f"workspace is 8 MiB + 64 KiB with rows32 on, 4 MiB + 64 KiB without)")

    ok = not FAILS
    log(f"cells: {len(CELLS)}, equal {sum(c['equal'] for c in CELLS)}, engaged {sum(c['engaged'] for c in CELLS)}")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(cells=CELLS, fails=FAILS, mode=a.mode), indent=1) + "\n")
    print("PARITY PASS" if ok else f"PARITY FAIL ({len(FAILS)})", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
