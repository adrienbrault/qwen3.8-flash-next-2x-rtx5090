#!/usr/bin/env python3
"""hcfast r2 P0 microbench: microseconds per HC boundary chain (hc_apply -> dots -> up), served vs r1 vs r2,
and the per-R choice of the r2 knobs that the P1 gate then measures.

Same sites, timing method and chain as r1's bench (forked from hcfuse r1's bench_hcfuse_p0.py). Every chain runs
the served hc_apply first; the variants differ in the mixer call (a spec = the gr_mix_v2_int8_v3 arguments
mode, dots_bmax, up_bmax, dots_j, dots_pf, up_q; None = the served binding):
  reference / reference#2   ext.gr_mix_v2_int8_statein (the served chain; #2 = drift check at the end)
  r1-d2                     (3, 2, 8, 4, 0, 2)  hcfast r1 at DOTS_B=2 UP_B=8: the R699 stack candidate
  r1-d8                     (3, 8, 8, 4, 0, 2)  hcfast r1 default
Per row count R, three stages, each variant timed once (graph replay, 24 cold sites):
  A  dots sweep: d{B}j{J}p{PF} = (15, B, 8, J, PF, 2) over every instantiated (B, J, PF)
  B  up sweep at A's best dots: u{B}q{Q} = (15, dB, B, dJ, dPF, Q) over B in 1/2/4/8, Q in 2/4
  C  confirmation, 5 interleaved rounds of [reference, r1-d2, pick, pick+pdl] (pick = A+B best; +pdl = mode 31):
     the per-R medians are the reported numbers. PDL is one switch (EXL3_HC_MIX_V3_PDL): on if pick+pdl is
     >= 1 % below pick at the largest R measured (16) and nowhere > 1 % above it.
The sweep chooses among close variants on single timings; stage C is what the reported ratios come from.

Timing: CUDA events around >= --reps chains cycling --sites HC sites (default 24 real sites = ~160 MB of int8
weights, above the 96 MB L2, so weights come from DRAM as in decode). graph = the chain sequence captured once
and replayed (GPU time per chain; the numbers used); eager = back-to-back Python launches (printed only).

Pre-registered (HOW-TO-VERIFY.md section 3):
  control  reference#2 within +-3 % of reference at every R (else the bench is not measuring kernels: stop)
  the P0 numbers explain and choose; they do not gate (OPERATIONS section 16). The gate reads the RECOMMEND
  line (EXL3_HC_MIX_V3=2 plus per-R tables) from --json and measures it in P1.

  docker run --rm --gpus '"device=0"' -v /srv/qwen5090/models:/models:ro -v $R:/results --entrypoint python3 \\
    tabbyapi:hcfast-r2 /opt/hcfast-r2/bench_hcfast_p0.py \\
    --model /models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab --json /results/p0.json

--ncu: one eager chain of each of --ncu-variants per row count after 3 warm chains, inside NVTX ranges
"<variant> R=<n>", for ncu --nvtx (specs by name: reference, r1-d2, r1-d8, or any d{B}j{J}p{PF} / full spec
"m,bd,bu,j,pf,q").
"""
import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

for k, v in {
    "EXL3_HC_MIX_V2": "1", "EXL3_HC_MIX_V2_MIN_R": "1", "EXL3_HC_MIX_V2_INT8": "1",
    "EXL3_GR_STATE_REGRID": "1", "EXL3_GR_STATE_IN_UP": "1", "EXL3_HC_APPLY_WARP1": "1",
}.items():
    os.environ[k] = v
for k in ("EXL3_HC_MIX_V3", "EXL3_HC_MIX_V3_DOTS_B", "EXL3_HC_MIX_V3_UP_B", "EXL3_HC_MIX_V3_DOTS_J",
          "EXL3_HC_MIX_V3_DOTS_PF", "EXL3_HC_MIX_V3_UP_Q", "EXL3_HC_MIX_V3_PDL"):
    os.environ.pop(k, None)

FIXED = {"reference": None, "r1-d2": (3, 2, 8, 4, 0, 2), "r1-d8": (3, 8, 8, 4, 0, 2)}
# Instantiated r2 dots forms (hc_mix_v3.cu resolve_dots_v4): B = 8 only J = 4, PF = 0; B = 4 PF <= 1 at J = 4,
# PF = 0 at J = 8.
DOTS_FORMS = [(b, j, pf) for b in (1, 2) for j in (4, 8) for pf in (0, 1, 2)] + [(4, 4, 0), (4, 4, 1), (4, 8, 0), (8, 4, 0)]
UP_FORMS = [(b, q) for b in (1, 2, 4, 8) for q in (2, 4)]
CONFIRM_ROUNDS = 5


def served_rows(R):
    return 1 if R == 1 else (2 if R == 2 else (4 if R <= 4 else 8))


def arguments():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default = "/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab")
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--rows", type = int, nargs = "+", default = [1, 4, 8, 16])
    ap.add_argument("--reps", type = int, default = 504, help = "chains timed per variant and row count (>= 500)")
    ap.add_argument("--sites", type = int, default = 24, help = "distinct HC sites cycled (L2 rotation)")
    ap.add_argument("--warmup", type = int, default = 50)
    ap.add_argument("--synthetic", action = "store_true", help = "random weights of the checkpoint shapes")
    ap.add_argument("--ncu", action = "store_true", help = "one eager chain per --ncu-variants and row count")
    ap.add_argument("--ncu-variants", nargs = "+", default = ["reference", "r1-d2", "d2j4p0", "d2j8p1"])
    ap.add_argument("--no-pdl", action = "store_true",
                    help = "skip pick+pdl (parity section 6 found PDL unsupported); RECOMMEND gets PDL=0")
    ap.add_argument("--json", type = Path)
    a = ap.parse_args()
    assert a.reps >= 500 or a.ncu, "at least 500 reps"
    assert all(1 <= r <= 32 for r in a.rows)
    return a


def parse_spec(name):
    if name in FIXED:
        return FIXED[name]
    if name.startswith("d") and "j" in name and "p" in name:
        b, rest = name[1:].split("j")
        j, pf = rest.split("p")
        return (15, int(b), 8, int(j), int(pf), 2)
    return tuple(int(x) for x in name.split(","))


def main():
    args = arguments()
    import torch
    from exllamav3.modules.hyperconnections import GatedResidual
    from exllamav3.ext import exllamav3_ext as ext
    assert getattr(ext, "hc_mix_v3_revision", None) == 2, "extension without the r2 gr_mix_v2_int8_v3"
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    cfg_json = json.loads((Path(args.model) / "config.json").read_text())
    tc = cfg_json.get("text_config", cfg_json)
    H, D, LR, eps = tc["hc_count"], tc["hidden_size"], tc["hc_lowrank"], tc["rms_norm_eps"]
    print(f"torch {torch.__version__} CUDA {torch.version.cuda} {torch.cuda.get_device_name(dev)}; "
          f"H={H} D={D} LR={LR}")

    sites = []
    if args.synthetic:
        gen = torch.Generator(device = dev).manual_seed(1)
        for i in range(args.sites):
            m = GatedResidual(None, f"syn{i}", H, D, eps)
            m.norm_w_raw = torch.randn(H * D, device = dev, generator = gen) * 0.05
            m._prepare((torch.randn(LR, H * D, device = dev, generator = gen) / 100).half(),
                       (torch.randn(H * D, LR, device = dev, generator = gen) / 20).half(),
                       (torch.randn(H, H * D, device = dev, generator = gen) / 100).half())
            sites.append(m)
    else:
        from exllamav3.model.config import Config
        cfg = Config.from_directory(args.model)
        n_layers = tc["num_hidden_layers"]
        for i in range(args.sites):
            layer, kind = (i // 2) % n_layers, ("attn", "mlp")[i % 2]
            m = GatedResidual(cfg, f"model.language_model.layers.{layer}.{kind}_hyper_connection", H, D, eps)
            m.load(dev)
            sites.append(m)
    for m in sites:
        assert m.fn_q is not None and m.upx_q is not None
        m.proj_h = m.up_h = m.down_h = m.inject_h = None     # only the int8 decode tables are used here
    torch.cuda.empty_cache()
    wbytes = sum(m.fn_q.numel() + m.upx_q.numel() for m in sites) / len(sites)
    print(f"{len(sites)} {'synthetic' if args.synthetic else 'real'} sites, "
          f"{wbytes / 1e6:.2f} MB int8 weights per site, {wbytes * len(sites) / 1e6:.0f} MB cycled")

    results, picks = {}, {}
    for R in args.rows:
        g = torch.Generator(device = dev).manual_seed(R)
        M = sites[0].fn_q.shape[0]
        x = [torch.randn((R, H, D), device = dev, generator = g) * 4.0 for _ in sites]
        y = [(torch.randn((R, D), device = dev, generator = g) * 1e-3).half() for _ in sites]
        pp = [2.0 * torch.sigmoid(torch.randn((R, H), device = dev, generator = g)) for _ in sites]
        dots = torch.empty((R, M + 1, H), device = dev)
        state = torch.empty((R, LR + H), device = dev)
        post = torch.empty((R, H), device = dev)
        mixed = torch.empty((R, D), device = dev, dtype = torch.half)

        def chain(spec, i):
            m = sites[i]
            ext.hc_apply(x[i], y[i], pp[i], None)
            if spec is None:
                ext.gr_mix_v2_int8_statein(x[i], m.fn_q, m.fn_s, m.upx_q, m.upx_s, m.w_h, m.rms_eps,
                                           dots, state, post, mixed)
            else:
                ext.gr_mix_v2_int8_v3(x[i], m.fn_q, m.fn_s, m.upx_q, m.upx_s, m.w_h, m.rms_eps,
                                      dots, state, post, mixed, *spec)

        if args.ncu:
            for name in args.ncu_variants:
                spec = parse_spec(name)
                for i in range(3):
                    chain(spec, i % len(sites))
                torch.cuda.synchronize(dev)
                torch.cuda.nvtx.range_push(f"{name} R={R}")
                chain(spec, 0)
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize(dev)
            print(f"R={R}: one chain per variant issued for ncu")
            continue

        n_sites = len(sites)
        replays = math.ceil(args.reps / n_sites)
        chains = replays * n_sites

        def time_graph(spec):
            for k in range(args.warmup):
                chain(spec, k % n_sites)
            torch.cuda.synchronize(dev)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(n_sites):
                    chain(spec, i)
            graph.replay()
            torch.cuda.synchronize(dev)
            e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            e0.record()
            for _ in range(replays):
                graph.replay()
            e1.record()
            torch.cuda.synchronize(dev)
            del graph
            return e0.elapsed_time(e1) * 1000.0 / chains

        row = {"reference": time_graph(None)}
        # A: dots sweep, up held at B = 8, Q = 2 (r1's up kernel) so only dots varies; stage B sweeps up
        sweep_a = {}
        for b, j, pf in DOTS_FORMS:
            if b > served_rows(R):
                continue                       # the cap cannot exceed the served tile: a duplicate
            sweep_a[f"d{b}j{j}p{pf}"] = time_graph((15, b, 8, j, pf, 2))
        best_d = min(sweep_a, key = sweep_a.get)
        db, rest = best_d[1:].split("j")
        dj, dpf = (int(v) for v in rest.split("p"))
        db = int(db)
        # B: up sweep at the best dots
        sweep_b = {}
        for b, q in UP_FORMS:
            if b > served_rows(R):
                continue
            sweep_b[f"u{b}q{q}"] = time_graph((15, db, b, dj, dpf, q))
        best_u = min(sweep_b, key = sweep_b.get)
        ub, uq = (int(v) for v in best_u[1:].split("q"))
        pick = (15, db, ub, dj, dpf, uq)
        pick_pdl = (31,) + pick[1:]
        # C: confirmation, interleaved
        conf = {"reference": [], "r1-d2": [], "pick": []} if args.no_pdl else \
            {"reference": [], "r1-d2": [], "pick": [], "pick+pdl": []}
        specs = {"reference": None, "r1-d2": FIXED["r1-d2"], "pick": pick, "pick+pdl": pick_pdl}
        for rnd in range(CONFIRM_ROUNDS):
            k = rnd % len(conf)
            order = list(conf)[k:] + list(conf)[:k]
            for name in order:
                conf[name].append(time_graph(specs[name]))
        med = {k: statistics.median(v) for k, v in conf.items()}
        use_pdl = (not args.no_pdl) and med["pick+pdl"] <= 0.99 * med["pick"]
        final = pick_pdl if use_pdl else pick
        row["reference#2"] = time_graph(None)
        row.update({"sweep_dots": sweep_a, "sweep_up": sweep_b, "confirm": conf, "confirm_median": med,
                    "pick": list(final)})
        results[R] = row
        picks[R] = final
        print(f"R={R:2d}  A dots: " + "  ".join(f"{k} {v:6.2f}" for k, v in sweep_a.items()))
        print(f"      B up @{best_d}: " + "  ".join(f"{k} {v:6.2f}" for k, v in sweep_b.items()))
        print(f"      C medians of {CONFIRM_ROUNDS}: " + "  ".join(
            f"{k} {v:6.2f} ({v / med['reference']:.3f}x)" for k, v in med.items())
            + f"   pick {pick} pdl {'yes' if use_pdl else 'no'}; reference {row['reference']:.2f} / #2 {row['reference#2']:.2f}")
        print(f"      (u8q2 at R >= 5 is r1's V3 up kernel, launched without PDL: hc_mix_v3.cu launch_up_v4)")

    if args.ncu:
        return 0
    print()
    verdicts = []
    for R, row in results.items():
        ref, ref2 = row["reference"], row["reference#2"]
        if abs(ref2 / ref - 1) > 0.03:
            verdicts.append(f"CONTROL FAIL R={R}: reference#2 {ref2:.2f} vs reference {ref:.2f} us (drift > 3 %)")
        med = row["confirm_median"]
        best = min(med["pick"], med.get("pick+pdl", float("inf")))
        verdicts.append(f"R={R}: r1-d2 / reference {med['r1-d2'] / med['reference']:.3f}, "
                        f"r2 pick / reference {med['pick'] / med['reference']:.3f}, "
                        + (f"r2 pick+pdl / reference {med['pick+pdl'] / med['reference']:.3f}, " if "pick+pdl" in med else "")
                        + f"r2 / r1-d2 {best / med['r1-d2']:.3f}")

    # Recommended per-R tables: R <= r_k uses the pick at r_k (ascending measured rows; the largest covers 32).
    rows_sorted = sorted(picks)
    def table(idx):
        vals = [(r if r != rows_sorted[-1] else 32, picks[r][idx]) for r in rows_sorted]
        merged = []
        for lim, v in vals:
            if merged and merged[-1][1] == v:
                merged[-1] = (lim, v)
            else:
                merged.append((lim, v))
        return str(merged[0][1]) if len(merged) == 1 else ",".join(f"{lim}:{v}" for lim, v in merged)
    if picks:
        # PDL is one switch for every R: on if it is >= 1 % faster than the same pick at the largest measured R
        # (R = 16: c4d3 / c8d1) and nowhere > 1 % slower.
        if args.no_pdl:
            ratio, pdl = {}, "0"
        else:
            ratio = {R: results[R]["confirm_median"]["pick+pdl"] / results[R]["confirm_median"]["pick"] for R in rows_sorted}
            pdl = "1" if ratio[rows_sorted[-1]] <= 0.99 and all(v <= 1.01 for v in ratio.values()) else "0"
        verdicts.append("PDL pick+pdl / pick: " + ", ".join(f"R={R} {v:.3f}" for R, v in ratio.items())
                        + f" -> EXL3_HC_MIX_V3_PDL={pdl}")
        env = (f"EXL3_HC_MIX_V3=2 EXL3_HC_MIX_V3_DOTS_B={table(1)} EXL3_HC_MIX_V3_UP_B={table(2)} "
               f"EXL3_HC_MIX_V3_DOTS_J={table(3)} EXL3_HC_MIX_V3_DOTS_PF={table(4)} EXL3_HC_MIX_V3_UP_Q={table(5)} "
               f"EXL3_HC_MIX_V3_PDL={pdl}")
        verdicts.append(f"RECOMMEND {env}")
    for vd in verdicts:
        print(vd)
    if args.json:
        args.json.write_text(json.dumps({"rows": results, "sites": len(sites), "verdicts": verdicts,
                                         "synthetic": args.synthetic,
                                         "recommend": env if picks else None}, indent = 2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
