#!/usr/bin/env python3
"""latchain r1 P0 kernel microbenchmark (one card, no model; diagnostic, not a gate per OPERATIONS §16).

  GDN   served cuda_recurrent_gated_delta_rule_kernel_128 vs EXL3_LC_GDN_RR (..._128_rr), per served shape
        (c1d3 1x4, c4d3 4x4, c8d1 8x2 with history = MTP verify; d0 controls b x 1 without history), bf16 state.
  QSA   served sparse split (2 stages) + combine (1 stage) vs every VARIANT of test_latchain_parity.py (the
        parity test proves each one bitwise before the gate uses this choice), per served shape at 4k context.

Method: one CUDA graph per arm holding the kernel for `copies` distinct layer copies back to back (GDN: 36 copies of
state + inputs = the 36 GDN layers of a step; QSA: enough K/V copies that the touched rows exceed 2x the 96 MB L2),
so every launch reads cold memory like the served step does between layers. Arms are replayed round-robin, `reps`
rounds x `inner` replays; the reported number is the median over rounds of us per launch (graph time / copies).
The per-step estimate multiplies by the served launch count: 36 GDN layers / 12 QSA layers per target forward
(the MTP draft forward is not counted).

Prints "[p0] ..." lines and, last, the QSA compile choice for the gate:
  [p0] QT choice EXL3_LC_QSA_SPLIT_STAGES=a EXL3_LC_QSA_COMBINE_STAGES=b EXL3_LC_QSA_DIV16=c   (or "QT choice none")
The choice minimises the summed split+combine us over the three served drafting shapes and must beat the served
options there by >= 1 %, with no served shape more than 1 % slower. Writes <out>/p0.json.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_latchain_parity as tp  # noqa: E402

GDN_SERVED = [("c1d3", 1, 4, True), ("c4d3", 4, 4, True), ("c8d1", 8, 2, True),
              ("c1d0", 1, 1, False), ("c4d0", 4, 1, False), ("c8d0", 8, 1, False)]
QSA_SERVED = [("c1d3", 1, 4), ("c4d3", 4, 4), ("c8d1", 8, 2), ("c1d0", 1, 1), ("c4d0", 4, 1), ("c8d0", 8, 1)]
DRAFTING = ("c1d3", "c4d3", "c8d1")
L2_BYTES = 96 * 1024 * 1024


def capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    return g


def time_round_robin(graphs: dict, launches: int, reps: int, inner: int) -> dict:
    per = {k: [] for k in graphs}
    keys = list(graphs)
    for r in range(reps):
        order = keys[r % len(keys):] + keys[:r % len(keys)]     # rotate who goes first
        for k in order:
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(inner):
                graphs[k].replay()
            b.record()
            torch.cuda.synchronize()
            per[k].append(a.elapsed_time(b) * 1000.0 / inner / launches)
    return {k: {"median_us": st.median(v), "min_us": min(v), "max_us": max(v)} for k, v in per.items()}


def bench_gdn(dev, copies, reps, inner):
    res = {}
    for name, bsz, seqlen, history in GDN_SERVED:
        gen = torch.Generator(device=dev).manual_seed(bsz * 10 + seqlen)
        hist = seqlen if history else 1          # served: Cache(max_history = depth) -> depth + 1 = seqlen slots
        layers = []
        for _ in range(copies):
            st_ = (torch.randn(bsz, hist, tp.NV, tp.HD, tp.HD, generator=gen, device=dev) * 0.3).bfloat16()
            slots = torch.arange(bsz, dtype=torch.int32, device=dev)
            mixed, g, beta = tp.gdn_inputs(gen, bsz, seqlen, dev)
            out = torch.empty(bsz, seqlen, tp.NV, tp.HD, dtype=torch.bfloat16, device=dev)
            layers.append((mixed, g, beta, st_, out, slots))
        graphs = {}
        for arm, on in (("served", False), ("rr", True)):
            def fn(on=on):
                for (mixed, g, beta, st_, out, slots) in layers:
                    tp.gdn_call(on, mixed, g, beta, st_, out, slots, history)
            os.environ["EXL3_LC_GDN_RR"] = "1" if on else "0"
            graphs[arm] = capture(fn)
        os.environ["EXL3_LC_GDN_RR"] = "0"
        t = time_round_robin(graphs, copies, reps, inner)
        s, r = t["served"]["median_us"], t["rr"]["median_us"]
        res[name] = {"bsz": bsz, "seqlen": seqlen, "history": history, **{k: v for k, v in t.items()},
                     "delta_us_per_launch": r - s, "delta_us_per_step_36_layers": 36 * (r - s)}
        print(f"[p0] GDN {name} (bsz {bsz}, {seqlen} rows/job, history {int(history)}): served {s:.2f} us, "
              f"rr {r:.2f} us ({100 * (r - s) / s:+.1f} %), per step x36 {36 * (r - s):+.1f} us", flush=True)
        del graphs, layers
        torch.cuda.empty_cache()
    return res


def bench_qsa(dev, reps, inner, k_pad, scale, context, variants):
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    res = {}
    for name, bsz, q_len in QSA_SERVED:
        rows = bsz * q_len
        touched = rows * k_pad * (512 + 512 + 32 + 32)            # K + V words and scales per selected token
        copies = max(4, min(64, -(-2 * L2_BYTES // touched)))
        case = tp.QsaCase(dev, bsz, q_len, context, k_pad, (1.0,), seed=bsz * 7 + q_len, copies=copies)
        programs, splits, split_len = tp.qsa_geometry(rows, k_pad, sms)
        bufs = (torch.empty(programs * splits * tp.BLOCK_H * tp.QHD, dtype=torch.float, device=dev),
                torch.empty(programs * splits * tp.BLOCK_H * 2, dtype=torch.float, device=dev),
                torch.empty(rows, tp.QH, tp.QHD, dtype=torch.half, device=dev))
        graphs, meta = {}, {}
        for v in variants:
            k_sp, k_cb = tp.qsa_kernels(dev, q_len, k_pad, scale, v)
            key = f"s{v[0]}c{v[1]}d{int(v[2])}"
            meta[key] = {"split_stages": v[0], "combine_stages": v[1], "div16": v[2],
                         "split_shared_bytes": k_sp.shared_bytes, "combine_shared_bytes": k_cb.shared_bytes}

            def fn(k_sp=k_sp, k_cb=k_cb):
                for c in range(copies):
                    tp.qsa_run(case, k_sp, k_cb, sms, copy=c, bufs=bufs)
            graphs[key] = capture(fn)
        t = time_round_robin(graphs, copies, reps, inner)
        base = t["s2c1d0"]["median_us"]
        res[name] = {"bsz": bsz, "q_len": q_len, "programs": programs, "splits": splits, "split_len": split_len,
                     "copies": copies, "arms": {k: {**t[k], **meta[k]} for k in t}}
        line = " ".join(f"{k} {t[k]['median_us']:.2f}" for k in t)
        print(f"[p0] QSA {name} (R {rows}, {programs}x{splits} CTAs, {copies} copies): split+combine us: {line}", flush=True)
        best = min(t, key=lambda k: t[k]["median_us"])
        print(f"[p0] QSA {name} best {best} {t[best]['median_us']:.2f} us vs served {base:.2f} "
              f"({100 * (t[best]['median_us'] - base) / base:+.1f} %; x12 layers {12 * (t[best]['median_us'] - base):+.1f} us/step)",
              flush=True)
        del graphs, case
        torch.cuda.empty_cache()
    return res


def choose(qsa):
    keys = list(next(iter(qsa.values()))["arms"])
    tot = {k: sum(qsa[s]["arms"][k]["median_us"] for s in DRAFTING if s in qsa) for k in keys}
    base = tot["s2c1d0"]
    cands = []
    for k in keys:
        if k == "s2c1d0" or tot[k] > 0.99 * base:
            continue
        worst = max(qsa[s]["arms"][k]["median_us"] / qsa[s]["arms"]["s2c1d0"]["median_us"] for s in DRAFTING if s in qsa)
        if worst <= 1.01:
            cands.append((tot[k], k))
    if not cands:
        return None, tot
    return min(cands)[1], tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="p0")
    ap.add_argument("--model", default=None, help="unused; accepted for gate-script symmetry")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--gdn-copies", type=int, default=36)
    ap.add_argument("--k-pad", type=int, default=2080)
    ap.add_argument("--scale", type=float, default=256 ** -0.5)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--skip-gdn", action="store_true")
    ap.add_argument("--skip-qsa", action="store_true")
    a = ap.parse_args()
    import exllamav3_ext as ext
    assert getattr(ext, "latchain_revision", None) == 1
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = {"device": torch.cuda.get_device_name(dev), "reps": a.reps, "inner": a.inner}
    if not a.skip_gdn:
        result["gdn"] = bench_gdn(dev, a.gdn_copies, a.reps, a.inner)
    if not a.skip_qsa:
        result["qsa"] = bench_qsa(dev, a.reps, a.inner, a.k_pad, a.scale, a.context, tp.VARIANTS)
        pick, tot = choose(result["qsa"])
        result["qsa_totals_drafting_us"] = tot
        result["qsa_choice"] = pick
        if pick is None:
            print("[p0] QT choice none (served compile options fastest within 1 %)")
        else:
            m = result["qsa"]["c1d3"]["arms"][pick]
            print(f"[p0] QT summed drafting-shape us: served {tot['s2c1d0']:.2f}, {pick} {tot[pick]:.2f}")
            print(f"[p0] QT choice EXL3_LC_QSA_SPLIT_STAGES={m['split_stages']} "
                  f"EXL3_LC_QSA_COMBINE_STAGES={m['combine_stages']} EXL3_LC_QSA_DIV16={int(m['div16'])}")
    (out / "p0.json").write_text(json.dumps(result, indent=1))
    print(f"[p0] done in {time.time() - t0:.0f} s -> {out / 'p0.json'}")


if __name__ == "__main__":
    main()
