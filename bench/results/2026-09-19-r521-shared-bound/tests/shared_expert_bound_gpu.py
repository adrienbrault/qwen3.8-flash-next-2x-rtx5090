#!/usr/bin/env python3
"""GPU probe for the operator: how much wall time does the shared expert still cost per MoE layer, with the served
side-stream overlap on? That number is the ceiling of ANY shared-expert kernel work (fused, regridded, or merged into
the routed coop kernel). Runs in the served image as-is (tabbyapi:stack-r4-e3r2): no overlay, no rebuild.

Per selected decoder layer (real 2.50bpw weights, loaded alone onto its serving card) and per row count it runs:

  off      BC_BlockSparseMLP.run_bszN built with EXL3_SHARED_EXPERT_OVERLAP=0 (shared graph, then routed, one stream)
  on       the same built with EXL3_SHARED_EXPERT_OVERLAP=1 (served: shared graph on a side stream, joined before B)
  routed   ext.exl3_moe_coop with no shared output (rot + stage A + stage B only, the same routed kernels)
  shared   the shared expert's BC_GatedMLP graph alone
  route    the layer's router (routing_fn) alone

and reports  residual = on - routed  (what the shared expert still adds to the layer: join stall + contention),
overlap_saving = off - on, hidden = overlap_saving / shared. Correctness (torch.equal, rows 1..16):
  on == off (re-checks R490 on the 2.50 pack), eager/capture/replay calls equal, and
  exl3_moe_coop(sh_out = shared graph output, sh_gate_w) == on  (proves the "routed" arm runs the served kernels).

Timing: CUDA events around `iters` back-to-back calls issued behind a torch.cuda._sleep pre-fill so the host is
ahead of the GPU; arms interleaved over `rounds`; median reported. A row is flagged host_bound when the host enqueue
time per call reaches the GPU time (then the GPU number is a host number).

Environment: pass the daily's EXTRA_ENV (EXL3_MOE_COOP_V2=1 etc.); the script toggles only
EXL3_SHARED_EXPERT_OVERLAP. Point EXLLAMAV3_TUNE_CACHE at a COPY of the daily's tune cache so the kernels run the
served geometry and nothing is appended to the served file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab")
    ap.add_argument("--layers", default="0:0,29:0,30:1,47:1", help="decoder layer:cuda index pairs (served split [30,30])")
    ap.add_argument("--rows-equal", default=",".join(str(i) for i in range(1, 17)))
    ap.add_argument("--rows-bench", default="1,2,4,8,16")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--sleep-ms", type=float, default=15.0, help="GPU pre-fill so the host enqueues ahead")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="shared_expert_bound.json")
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    for k in ("EXL3_MOE_COOP_V2",):
        if os.environ.get(k) != "1":
            print(f"WARN: {k} is not 1 -- the daily serves with it on; pass the daily EXTRA_ENV", file=sys.stderr)
    os.environ["EXL3_SHARED_EXPERT_OVERLAP"] = "1"

    import torch
    from exllamav3 import Config, Model
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP, MAX_BSZN

    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    moe = {}
    for m in model:
        if isinstance(m, BlockSparseMLP) and m.shared_experts is not None:
            mm = re.search(r"\.layers\.(\d+)\.mlp$", m.key)
            if mm and not m.key.startswith("mtp"):
                moe[int(mm.group(1))] = m
    print(f"{len(moe)} MoE layers with a shared expert found")

    results = {"env": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "EXLLAMAV3_"))},
               "model": a.model, "layers": [], "equality": [], "timing": []}
    rows_equal = [int(r) for r in a.rows_equal.split(",")]
    rows_bench = [int(r) for r in a.rows_bench.split(",")]
    ok_all = True

    for spec in a.layers.split(","):
        li, di = (int(v) for v in spec.split(":"))
        mlp = moe[li]
        dev = torch.device(f"cuda:{di}")
        t0 = time.time()
        os.environ["EXL3_SHARED_EXPERT_OVERLAP"] = "1"
        mlp.load(dev)
        with torch.cuda.device(dev):
            ok_all &= run_layer(a, torch, ext, mlp, li, dev, rows_equal, rows_bench, results, MAX_BSZN)
        mlp.unload()
        torch.cuda.empty_cache()
        print(f"layer {li} on {dev} done in {time.time() - t0:.1f} s")

    summarize(results)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}; equality {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


def run_layer(a, torch, ext, mlp, li, dev, rows_equal, rows_bench, results, MAX_BSZN) -> bool:
    sh = mlp.shared_experts
    cfg = mlp.experts_cfg
    assert mlp.bc is not None and mlp.bc_sh_exp, "layer did not build the fused decode BC with an embedded shared expert"

    # ---- shapes / K actually served (answers the 2.50-pack K question) ----
    mgu = sh.multi_gu[0]
    info = {
        "layer": li, "device": str(dev),
        "hidden": mlp.hidden_size, "routed_interm": int(cfg.interm_u.shape[-1]),
        "num_experts": mlp.num_experts, "topk": mlp.num_experts_per_tok,
        "routed_K_gate_up_down": [mlp.multi_gate.K if mlp.gated else None, mlp.multi_up.K, mlp.multi_down.K],
        "routed_mul1": bool(mlp.multi_up.mul1), "routed_mcg": bool(mlp.multi_up.mcg),
        "shared_interm": sh.intermediate_size,
        "shared_K_gate_up_down": [sh.gates[0].inner.K, sh.ups[0].inner.K, sh.downs[0].inner.K],
        "shared_mul1": bool(sh.downs[0].inner.mul1), "shared_mcg": bool(sh.downs[0].inner.mcg),
        "shared_gate_up_fused_mgemm": mgu is not None,
        "shared_trellis_shapes": [list(sh.gates[0].inner.trellis.shape), list(sh.downs[0].inner.trellis.shape)],
        "shared_gate_proj": mlp.shared_gate is not None,
    }
    results["layers"].append(info)
    print(json.dumps(info))

    # ---- build OFF next to the served ON BC (the flag is read in the constructor, blocksparse_mlp.cpp:303) ----
    bc_on = mlp.bc
    os.environ["EXL3_SHARED_EXPERT_OVERLAP"] = "0"
    mlp.load_local()
    bc_off = mlp.bc
    os.environ["EXL3_SHARED_EXPERT_OVERLAP"] = "1"
    mlp.bc = bc_on
    sh_bc = sh.bc

    H = mlp.hidden_size
    Hi = int(cfg.yh.shape[-1])
    I = int(cfg.interm_u.shape[-1])
    Ho = int(cfg.out_d.shape[-1])
    bszn_rows = MAX_BSZN * mlp.num_experts_per_tok
    mg = mlp.multi_gate if mlp.gated else mlp.multi_up
    mu, md = mlp.multi_up, mlp.multi_down
    act = 3 if mlp.activation_fn == "swiglu_oai" else 1 if mlp.activation_fn == "gelu" else \
        2 if mlp.activation_fn == "relu2" else 0
    # private scratch for the functional ("routed") arm: same shapes as the served ones
    had_g = torch.empty_like(cfg.yh)
    had_u = torch.empty((bszn_rows, Hi), dtype=torch.half, device=dev)
    gu_g = torch.empty_like(cfg.interm_g)
    gu_u = torch.empty_like(cfg.interm_u)
    act_out = torch.empty_like(cfg.interm_a)
    d_out = torch.empty_like(cfg.out_d)
    ctr = torch.zeros((bszn_rows * (I // 128) + MAX_BSZN * (Ho // 128) + 2 + (bszn_rows + 1) + bszn_rows,),
                      dtype=torch.int, device=dev)
    out_f = torch.empty((MAX_BSZN, H), dtype=torch.float, device=dev)
    sh_buf = torch.empty((1, MAX_BSZN, H), dtype=torch.float, device=dev)
    sh_w = mlp.shared_gate.inner.weight if mlp.shared_gate is not None else None

    def coop(y, sel, rw, rows, with_shared):
        ext.exl3_moe_coop(
            y, sel, rw, cfg.min_expert, cfg.max_expert, Hi,
            mg.ptrs_trellis, mg.ptrs_suh, mg.ptrs_svh,
            mu.ptrs_trellis, mu.ptrs_suh, mu.ptrs_svh,
            md.ptrs_trellis, md.ptrs_suh, md.ptrs_svh,
            None, None, None,
            mg.K, mu.K, md.K, bool(mu.mcg), bool(mu.mul1), act, float(mlp.act_limit), bool(mlp.gated),
            had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out_f,
            sh_buf[0, :rows] if with_shared else None,
            sh_w if with_shared else None,
        )

    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed + li)
    x_all = torch.randn((MAX_BSZN, H), generator=gen, device=dev, dtype=torch.float).half()

    def routing(rows):
        y = x_all[:rows].contiguous()
        try:
            sel, rw = mlp.routing_fn(rows, mlp.routing_cfg, y, {})
            return y, sel.clone(), rw.clone(), True
        except Exception as e:  # fall back to uniform random distinct picks
            print(f"  routing_fn failed ({e!r}); using random routing", file=sys.stderr)
            sel = torch.stack([torch.randperm(mlp.num_experts, device=dev)[:mlp.num_experts_per_tok] for _ in range(rows)])
            rw = torch.softmax(torch.randn((rows, mlp.num_experts_per_tok), device=dev), -1).half()
            return y, sel.long().contiguous(), rw.contiguous(), False

    out_bszn = cfg.out_bszn
    ok = True

    # ---- equality: rows 1..16 ----
    for rows in rows_equal:
        y, sel, rw, _ = routing(rows)
        outs = {}
        for name, bc in (("off", bc_off), ("on", bc_on)):
            vals = []
            for _ in range(3):   # eager (autotune), capture, replay of the shared graph
                bc.run_bszN(y, sel, rw)
                torch.cuda.synchronize(dev)
                vals.append(out_bszn[:rows].clone())
            outs[name] = vals
        sh_bc.run_bszN(y.unsqueeze(0), sh_buf[:, :rows])
        coop(y, sel, rw, rows, True)
        torch.cuda.synchronize(dev)
        func = out_f[:rows].clone()
        rec = {
            "layer": li, "device": str(dev), "rows": rows,
            "on_eq_off": all(torch.equal(v, outs["off"][2]) for v in outs["on"]),
            "calls_eq": all(torch.equal(v, outs["on"][0]) for v in outs["on"] + outs["off"]),
            "functional_eq_on": torch.equal(func, outs["on"][2]),
            "max_abs_functional_vs_on": float((func - outs["on"][2]).abs().max()),
        }
        ok &= rec["on_eq_off"] and rec["calls_eq"] and rec["functional_eq_on"]
        results["equality"].append(rec)
        print(" ", rec)

    # ---- timing ----
    clock_khz = torch.cuda.get_device_properties(dev).clock_rate if hasattr(
        torch.cuda.get_device_properties(dev), "clock_rate") else 2_400_000
    sleep_cycles = int(a.sleep_ms * clock_khz)
    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)

    def time_arm(fn):
        torch.cuda.synchronize(dev)
        torch.cuda._sleep(sleep_cycles)
        t0 = time.perf_counter()
        s_ev.record()
        for _ in range(a.iters):
            fn()
        e_ev.record()
        host = (time.perf_counter() - t0) * 1e6 / a.iters
        e_ev.synchronize()
        return s_ev.elapsed_time(e_ev) * 1000.0 / a.iters, host

    for rows in rows_bench:
        y, sel, rw, real_routing = routing(rows)
        x3 = y.unsqueeze(0)
        sh_d = sh_buf[:, :rows]
        arms = {
            "off": lambda: bc_off.run_bszN(y, sel, rw),
            "on": lambda: bc_on.run_bszN(y, sel, rw),
            "routed": lambda: coop(y, sel, rw, rows, False),
            "shared": lambda: sh_bc.run_bszN(x3, sh_d),
            "route": lambda: mlp.routing_fn(rows, mlp.routing_cfg, y, {}),
        }
        if not real_routing:
            arms.pop("route")
        for fn in arms.values():      # warm: graphs captured, tuner settled
            for _ in range(5):
                fn()
        samples = {k: [] for k in arms}
        hosts = {k: [] for k in arms}
        order = list(arms)
        for r in range(a.rounds):
            for k in (order if r % 2 == 0 else order[::-1]):
                g, h = time_arm(arms[k])
                samples[k].append(g)
                hosts[k].append(h)
        med = {k: statistics.median(v) for k, v in samples.items()}
        hmed = {k: statistics.median(v) for k, v in hosts.items()}
        rec = {
            "layer": li, "device": str(dev), "rows": rows,
            "unique_experts": int(torch.unique(sel).numel()),
            **{f"{k}_us": round(v, 2) for k, v in med.items()},
            **{f"{k}_host_us": round(v, 2) for k, v in hmed.items()},
            "host_bound": [k for k in med if hmed[k] >= 0.95 * med[k]],
            "overlap_saving_us": round(med["off"] - med["on"], 2),
            "residual_us": round(med["on"] - med["routed"], 2),
            "hidden_frac": round((med["off"] - med["on"]) / med["shared"], 3) if med["shared"] > 0 else None,
        }
        results["timing"].append(rec)
        print(" ", rec)

    mlp.bc = None
    del bc_on, bc_off
    return ok


def summarize(results):
    # served shapes: verify rows 4 (c1 d3), 8 (c4 d1), 16 (c4 d3) at 48 target layers; draft block rows 1 / 4 x 3
    by_rows = {}
    for t in results["timing"]:
        by_rows.setdefault(t["rows"], []).append(t)
    summ = {}
    for rows, ts in sorted(by_rows.items()):
        res = statistics.mean(t["residual_us"] for t in ts)
        summ[rows] = {
            "residual_us_mean": round(res, 2),
            "residual_us_max": round(max(t["residual_us"] for t in ts), 2),
            "shared_us_mean": round(statistics.mean(t["shared_us"] for t in ts), 2),
            "overlap_saving_us_mean": round(statistics.mean(t["overlap_saving_us"] for t in ts), 2),
            "target_ms_per_step_if_48_layers": round(res * 48 / 1000.0, 3),
            "draft_ms_per_step_3_sites": round(res * 3 / 1000.0, 3),
            "any_host_bound": sorted({k for t in ts for k in t["host_bound"]}),
        }
    results["summary"] = summ
    print(json.dumps(summ, indent=1))
    worst = max((v["residual_us_max"] for r, v in summ.items() if r in (4, 8, 16)), default=None)
    if worst is not None:
        verdict = ("CLOSE: shared expert is off the critical path at the served verify shapes"
                   if worst < 2.0 else
                   "RESIDUAL: the shared expert still costs wall time; the follow-up is scheduling (impl-status §5), "
                   "ceiling = residual x 48 per step")
        print(f"decision (worst served-shape residual {worst} us/layer): {verdict}")
        results["decision"] = {"worst_residual_us": worst, "verdict": verdict}


if __name__ == "__main__":
    sys.exit(main())
