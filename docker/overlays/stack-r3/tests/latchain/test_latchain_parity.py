#!/usr/bin/env python3
"""latchain r1 GPU parity (one card, no model): every kernel-level flag ON vs the served path, bitwise.

A. EXL3_LC_GDN_RR (register-resident recurrent kernel) vs the served cuda_recurrent_gated_delta_rule_kernel_128,
   through ext.cuda_recurrent_gated_delta_rule with the flag toggled per launch (the dispatcher reads it per launch):
   * the served geometry (16 k-heads, 48 v-heads, 128 x 128), bf16 state (served EXL3_GDN_STATE_BF16=1) and fp32;
   * (bsz, rows per job) over every served decode shape (c1d3 1x4, c4d3 4x4, c8d1 8x2, the d0 controls b x 1) plus
     neighbours (1x2, 1x3, 2x4, 3x3, 5x3, 6x2, 7x2, 8x4);
   * history on (MTP verify: token s writes history slot s + 1) and off; permuted recurrent slots, spare slots and
     every unwritten history slot filled with a sentinel, so a stray read or write shows in the full-buffer compare;
   * 12 chained steps per case (the state each step starts from is the previous step's output);
   * a CUDA-graph capture of the ON path, replayed 6 times with new inputs, against eager OFF;
   * the kernel that ran is checked by name (torch.profiler): ON must launch ..._kernel_128_rr, OFF must not.
   Compared: core_attn_out and the whole recurrent_state buffer, byte for byte.

B. EXL3_LC_QSA_SPLIT_STAGES / _COMBINE_STAGES / _DIV16 (QSA sparse-regime AOT compile options) vs the served options
   (split 2 stages, combine 1 stage, no hints): the two kernels are compiled through bc_attn._compile_kernel exactly
   as bc_attn._configure_qsa does (served signature/constexprs; the variant = the same with the flag's change) and
   launched through TritonKernel.launch_py (the C++ launcher the BC graphs use), on synthetic 8-bit packed K/V
   pages (random words, served row layout: 128 int32 + 16 fp16 group scales per token), a shuffled block table,
   index rows with -1 padding (full, partial and short selections), R = bsz * q_len rows per served shape.
   Compared: partial_o, partial_ml (split) and out (combine), byte for byte, for every variant in VARIANTS (the
   set the P0 bench may choose from).

The fork flags (EXL3_LC_QSA_FORK, EXL3_LC_GDN_FORK) change graph node dependencies only; they are checked at model
level (lc_model_parity.py: logits hashes, and the gate's P1 sequence hashes + fn_greedy).

  python3 test_latchain_parity.py [--json out.json] [--quick]
Exit 0 and "PARITY PASS" only if every comparison is equal and every flag demonstrably took effect.
"""
import argparse
import itertools
import json
import os
import sys
import time

import torch

STATS = {"compared": 0, "equal": 0, "fail": []}


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def record(ok: bool, what: str):
    STATS["compared"] += 1
    if ok:
        STATS["equal"] += 1
    else:
        STATS["fail"].append(what)
        print("  MISMATCH:", what, flush=True)


def set_flag(name: str, on: bool):
    os.environ[name] = "1" if on else "0"


# ---------------------------------------------------------------------------------------------
# A. GDN recurrent

NK, NV, HD = 16, 48, 128
GDN_SHAPES = [(1, 4), (4, 4), (8, 2), (1, 1), (4, 1), (8, 1),          # served: c1d3 c4d3 c8d1 + d0 controls
              (1, 2), (1, 3), (2, 4), (3, 3), (5, 3), (6, 2), (7, 2), (8, 4)]
SENTINEL = {torch.bfloat16: -1234.0, torch.float32: -98765.4321}


def gdn_inputs(gen, bsz, seqlen, dev):
    qkv_dim = 2 * NK * HD + NV * HD
    mixed = torch.randn(bsz, seqlen, qkv_dim, generator=gen, device=dev).bfloat16()
    g = -torch.nn.functional.softplus(torch.randn(bsz, seqlen, NV, generator=gen, device=dev) - 1.0)
    beta = torch.sigmoid(torch.randn(bsz, seqlen, NV, generator=gen, device=dev)).bfloat16()
    return mixed, g.float().contiguous(), beta


def gdn_state(gen, bsz, seqlen, history, dtype, dev):
    num_slots = bsz + 3
    hist = (max(seqlen, 4) + 1) if history else 1
    st = torch.full((num_slots, hist, NV, HD, HD), SENTINEL[dtype], dtype=dtype, device=dev)
    st[:, 0] = (torch.randn(num_slots, NV, HD, HD, generator=gen, device=dev) * 0.3).to(dtype)
    slots = torch.randperm(num_slots, generator=gen, device=dev)[:bsz].int()
    return st, slots


def gdn_call(on, mixed, g, beta, st, out, slots, history):
    import exllamav3_ext as ext
    set_flag("EXL3_LC_GDN_RR", on)
    ext.cuda_recurrent_gated_delta_rule(mixed, g, beta, st, out, NK, NV, HD, HD, slots, history)


def gdn_kernel_names(on, bsz, seqlen, history, dtype, dev):
    gen = torch.Generator(device=dev).manual_seed(7)
    st, slots = gdn_state(gen, bsz, seqlen, history, dtype, dev)
    mixed, g, beta = gdn_inputs(gen, bsz, seqlen, dev)
    out = torch.empty(bsz, seqlen, NV, HD, dtype=torch.bfloat16, device=dev)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        gdn_call(on, mixed, g, beta, st, out, slots, history)
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if "recurrent_gated_delta_rule" in e.name]


def part_a(dev, quick, steps=12, graph_replays=6):
    shapes = GDN_SHAPES[:6] if quick else GDN_SHAPES
    # the flag took effect (kernel names), per dtype / history
    for dtype, history in itertools.product((torch.bfloat16, torch.float32), (True, False)):
        on = gdn_kernel_names(True, 1, 4, history, dtype, dev)
        off = gdn_kernel_names(False, 1, 4, history, dtype, dev)
        ok = any("kernel_128_rr" in n for n in on) and off and not any("kernel_128_rr" in n for n in off)
        record(bool(ok), f"A: kernel selection dtype={dtype} history={history}: ON {on[:1]} OFF {off[:1]}")
    n0 = STATS["compared"]
    for (bsz, seqlen), dtype, history in itertools.product(shapes, (torch.bfloat16, torch.float32), (True, False)):
        gen = torch.Generator(device=dev).manual_seed(1000 * bsz + 10 * seqlen + (dtype == torch.float32) * 2 + history)
        st_a, slots = gdn_state(gen, bsz, seqlen, history, dtype, dev)
        st_b = st_a.clone()
        tag = f"A: bsz={bsz} seqlen={seqlen} state={str(dtype)[6:]} history={int(history)}"
        # eager chain
        for step in range(steps):
            mixed, g, beta = gdn_inputs(gen, bsz, seqlen, dev)
            out_a = torch.full((bsz, seqlen, NV, HD), 7.0, dtype=torch.bfloat16, device=dev)
            out_b = out_a.clone()
            gdn_call(False, mixed, g, beta, st_a, out_a, slots, history)
            gdn_call(True, mixed, g, beta, st_b, out_b, slots, history)
            torch.cuda.synchronize()
            ok = same_bytes(out_a, out_b) and same_bytes(st_a, st_b)
            if not ok or step == steps - 1:
                record(ok, f"{tag} eager step {step}")
            if not ok:
                break
        # graph: capture ON once, replay with fresh inputs, against eager OFF
        s_mixed, s_g, s_beta = gdn_inputs(gen, bsz, seqlen, dev)
        s_out = torch.zeros(bsz, seqlen, NV, HD, dtype=torch.bfloat16, device=dev)
        st_g = st_a.clone()
        st_e = st_a.clone()
        set_flag("EXL3_LC_GDN_RR", True)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):   # warm the extension outside capture (on a throwaway state)
            tmp = st_a.clone()
            gdn_call(True, s_mixed, s_g, s_beta, tmp, s_out, slots, history)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gdn_call(True, s_mixed, s_g, s_beta, st_g, s_out, slots, history)
        set_flag("EXL3_LC_GDN_RR", False)     # replay must keep the captured (rr) kernel
        for rep in range(graph_replays):
            mixed, g, beta = gdn_inputs(gen, bsz, seqlen, dev)
            s_mixed.copy_(mixed); s_g.copy_(g); s_beta.copy_(beta)
            graph.replay()
            out_e = torch.empty_like(s_out)
            gdn_call(False, mixed, g, beta, st_e, out_e, slots, history)
            torch.cuda.synchronize()
            ok = same_bytes(s_out, out_e) and same_bytes(st_g, st_e)
            if not ok or rep == graph_replays - 1:
                record(ok, f"{tag} graph replay {rep}")
            if not ok:
                break
        del graph
    print(f"[A] GDN recurrent: {STATS['compared'] - n0} case checks over {len(shapes)} shapes x 2 dtypes x "
          f"history on/off ({steps} eager steps + {graph_replays} graph replays each)", flush=True)


# ---------------------------------------------------------------------------------------------
# B. QSA sparse split / combine compile options

QH, KVH, QHD, BITS, PAGE = 24, 2, 256, 8, 256
BLOCK_H, BLOCK_N = 16, 32
# (split num_stages, combine num_stages, div16); the first entry is the served configuration
VARIANTS = [(2, 1, False)] + [v for v in itertools.product((2, 3, 4), (1, 2), (False, True)) if v != (2, 1, False)]
QSA_SHAPES = [(1, 4), (4, 4), (8, 2), (1, 1), (4, 1), (8, 1)]   # (bsz, q_len): c1d3 c4d3 c8d1 + d0 controls

SIG_SP = {"q": "*fp16", "k_cache": "*i32", "v_cache": "*i32", "block_table": "*i32",
          "indices": "*i32", "partial_o": "*fp32", "partial_ml": "*fp32",
          "k_len": "i32", "num_pages_per_seq": "i32", "num_splits": "i32",
          "split_len": "i32", "k_scales": "*fp16", "v_scales": "*fp16", "h32": "*fp16"}
SIG_CB = {"partial_o": "*fp32", "partial_ml": "*fp32", "out": "*fp16", "h32": "*fp16",
          "num_splits": "i32", "sinks": "*fp32"}
DIV16_SP = ("q", "k_cache", "v_cache", "indices", "partial_o", "partial_ml", "k_len", "split_len",
            "k_scales", "v_scales", "h32")
DIV16_CB = ("partial_o", "partial_ml", "out", "h32")


def qsa_geometry(rows, k_pad, sms):
    h_blocks = -(-(QH // KVH) // BLOCK_H)
    programs = rows * KVH * h_blocks
    splits = max(1, min((2 * sms) // programs, -(-k_pad // (4 * BLOCK_N)), 128))
    split_len = -(-(-(-k_pad // splits)) // BLOCK_N) * BLOCK_N
    return programs, splits, split_len


def qsa_kernels(dev, q_len, k_pad, scale, variant):
    from exllamav3.modules.attention_fn import bc_attn
    from exllamav3.modules.attention_fn.qsa_triton import _qsa_sparse_split_kernel
    from exllamav3.modules.attention_fn.triton_paged import _paged_attn_decode_combine_kernel
    sp_stages, cb_stages, div16 = variant
    sig_sp, sig_cb = dict(SIG_SP), dict(SIG_CB)
    if div16:
        sig_sp = bc_attn._lc_div16(sig_sp, DIV16_SP)
        sig_cb = bc_attn._lc_div16(sig_cb, DIV16_CB)
    k_sp = bc_attn._compile_kernel(dev, _qsa_sparse_split_kernel,
        sig_sp | {n: "constexpr" for n in (
            "n_q_heads", "n_kv_heads", "page_size", "head_dim", "K_pad", "scale",
            "BLOCK_H", "BLOCK_N", "PAGED", "QCK", "QCV", "SEQ")},
        dict(n_q_heads=QH, n_kv_heads=KVH, page_size=PAGE, head_dim=QHD, K_pad=k_pad,
             scale=float(scale), BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, PAGED=1, QCK=BITS, QCV=BITS, SEQ=q_len),
        4, sp_stages)
    k_cb = bc_attn._compile_kernel(dev, _paged_attn_decode_combine_kernel,
        sig_cb | {n: "constexpr" for n in (
            "QCV", "HAS_SINKS", "q_len", "n_q_heads", "n_kv_heads", "head_dim", "HD_PAD",
            "BLOCK_M", "BLOCK_H", "BLOCK_ROWS")},
        dict(QCV=BITS, HAS_SINKS=False, q_len=1, n_q_heads=QH, n_kv_heads=KVH, head_dim=QHD, HD_PAD=QHD,
             BLOCK_M=1, BLOCK_H=BLOCK_H, BLOCK_ROWS=BLOCK_H),
        4, cb_stages)
    return k_sp, k_cb


class QsaCase:
    """Synthetic served-layout inputs for one (bsz, q_len, context); copies > 1 = rotation copies (bench)."""

    def __init__(self, dev, bsz, q_len, context, k_pad, sel_frac, seed, copies=1):
        gen = torch.Generator(device=dev).manual_seed(seed)
        self.bsz, self.q_len, self.rows = bsz, q_len, bsz * q_len
        self.k_pad = k_pad
        pages = -(-(context + q_len) // PAGE)
        total_pages = bsz * pages + 3
        self.copies = []
        for _ in range(copies):
            kc = torch.randint(-2 ** 31, 2 ** 31 - 1, (total_pages * PAGE, 128), dtype=torch.int32, generator=gen, device=dev)
            vc = torch.randint(-2 ** 31, 2 ** 31 - 1, (total_pages * PAGE, 128), dtype=torch.int32, generator=gen, device=dev)
            ks = (torch.rand(total_pages * PAGE, 16, generator=gen, device=dev) * 0.02 + 0.002).half()
            vs = (torch.rand(total_pages * PAGE, 16, generator=gen, device=dev) * 0.02 + 0.002).half()
            self.copies.append((kc, vc, ks, vs))
        self.block_table = torch.randperm(total_pages, generator=gen, device=dev)[:bsz * pages].view(bsz, pages).int().contiguous()
        self.q = (torch.randn(self.rows, QH, QHD, generator=gen, device=dev)).half()
        idx = torch.full((self.rows, k_pad), -1, dtype=torch.int32, device=dev)
        for r in range(self.rows):
            limit = context + (r % q_len) + 1              # causal bound of this query row
            n = min(limit, max(1, int(k_pad * sel_frac[r % len(sel_frac)])))
            pos = torch.randperm(limit, generator=gen, device=dev)[:n].sort().values
            idx[r, :n] = pos.int()
        self.indices = idx.contiguous()


def qsa_run(case, k_sp, k_cb, sms, copy=0, bufs=None):
    import exllamav3_ext  # noqa: F401  (TritonKernel lives in the extension)
    from exllamav3.modules.attention_fn.triton_paged import _get_h32
    dev = case.q.device
    programs, splits, split_len = qsa_geometry(case.rows, case.k_pad, sms)
    if bufs is None:
        po = torch.full((programs * splits * BLOCK_H * QHD,), float("nan"), dtype=torch.float, device=dev)
        pml = torch.full((programs * splits * BLOCK_H * 2,), float("nan"), dtype=torch.float, device=dev)
        out = torch.full((case.rows, QH, QHD), 3.0, dtype=torch.half, device=dev)
    else:
        po, pml, out = bufs
    h32 = _get_h32(dev)
    kc, vc, ks, vs = case.copies[copy]
    k_sp.launch_py(programs, splits, 1, [
        case.q.data_ptr(), kc.data_ptr(), vc.data_ptr(), case.block_table.data_ptr(), case.indices.data_ptr(),
        po.data_ptr(), pml.data_ptr(), case.k_pad, case.block_table.size(1), splits, split_len,
        ks.data_ptr(), vs.data_ptr(), h32.data_ptr()])
    k_cb.launch_py(programs, 1, 1, [po.data_ptr(), pml.data_ptr(), out.data_ptr(), h32.data_ptr(), splits,
                                    case.q.data_ptr()])
    return po, pml, out


def part_b(dev, quick, k_pad, scale, contexts):
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    variants = VARIANTS[:4] if quick else VARIANTS
    shapes = QSA_SHAPES[:3] if quick else QSA_SHAPES
    n0 = STATS["compared"]
    for (bsz, q_len), context in itertools.product(shapes, contexts):
        # full selections, a partly filled row mix, and short (early-context-like) rows
        case = QsaCase(dev, bsz, q_len, context, k_pad, (1.0, 0.93, 0.5, 0.12), seed=bsz * 100 + q_len * 10 + context)
        ref = [t.clone() for t in qsa_run(case, *qsa_kernels(dev, q_len, k_pad, scale, VARIANTS[0]), sms)]
        again = qsa_run(case, *qsa_kernels(dev, q_len, k_pad, scale, VARIANTS[0]), sms)
        torch.cuda.synchronize()
        record(all(same_bytes(a, b) for a, b in zip(ref, again)),
               f"B: served run-to-run bsz={bsz} q_len={q_len} ctx={context}")
        nan_out = bool(torch.isnan(ref[2].float()).any())
        record(not nan_out, f"B: served output finite bsz={bsz} q_len={q_len} ctx={context}")
        for v in variants[1:]:
            got = qsa_run(case, *qsa_kernels(dev, q_len, k_pad, scale, v), sms)
            torch.cuda.synchronize()
            ok = all(same_bytes(a, b) for a, b in zip(ref, got))
            record(ok, f"B: variant split_stages={v[0]} combine_stages={v[1]} div16={int(v[2])} "
                       f"bsz={bsz} q_len={q_len} ctx={context}")
        programs, splits, split_len = qsa_geometry(case.rows, k_pad, sms)
        print(f"[B] bsz={bsz} q_len={q_len} ctx={context}: programs={programs} splits={splits} split_len={split_len}, "
              f"{len(variants) - 1} variants compared", flush=True)
    print(f"[B] QSA split/combine: {STATS['compared'] - n0} checks", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--quick", action="store_true", help="subset (smoke test)")
    ap.add_argument("--k-pad", type=int, default=2080, help="QSA K_pad (served: ceil32(512 * 4 + 3) = 2080)")
    ap.add_argument("--scale", type=float, default=256 ** -0.5, help="attention sm_scale (head_dim 256)")
    ap.add_argument("--contexts", type=int, nargs="+", default=[4096, 32768])
    ap.add_argument("--model", default=None, help="accepted for gate-script symmetry; unused (no model load)")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "GPU test"
    import exllamav3_ext as ext
    assert getattr(ext, "latchain_revision", None) == 1, "latchain r1 extension not installed"
    assert hasattr(ext.TritonKernel, "launch_py")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    t0 = time.time()
    part_a(dev, a.quick)
    part_b(dev, a.quick, a.k_pad, a.scale, a.contexts)
    ok = not STATS["fail"] and STATS["compared"] > 0
    print(f"comparisons equal: {STATS['equal']}/{STATS['compared']} ({time.time() - t0:.0f} s)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({**STATS, "pass": ok, "variants": VARIANTS, "gdn_shapes": GDN_SHAPES,
                       "qsa_shapes": QSA_SHAPES, "device": torch.cuda.get_device_name(dev)}, f, indent=1)
    print("PARITY PASS" if ok else f"PARITY FAIL ({len(STATS['fail'])} mismatches)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
