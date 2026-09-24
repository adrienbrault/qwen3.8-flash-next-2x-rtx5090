#!/usr/bin/env python3
"""hcfast r1 P0 microbench: microseconds per HC boundary chain (hc_apply -> dots -> up), served vs V3.

Forked from hcfuse r1's bench_hcfuse_p0.py (same sites, same timing method, same chain), with the variants
replaced. All variants run the served hc_apply first; they differ in the mixer call:
  reference  ext.gr_mix_v2_int8_statein                      the served chain (bar baseline)
  copy       ext.gr_mix_v2_int8_v3(mode 0, caps 8/8)         served kernels copied into hc_mix_v3.cu (control)
  v3         ext.gr_mix_v2_int8_v3(mode 3, caps 8/8)         what EXL3_HC_MIX_V3=1 serves (the candidate)
  v3-dots    mode 1: V3 dots + copied served up              attribution
  v3-up      mode 2: copied served dots + V3 up              attribution
  v3-d4/d2/d1  mode 3, dots rows per CTA capped at 4/2/1     tile sweep (EXL3_HC_MIX_V3_DOTS_B)
  v3-u4/u2     mode 3, up rows per CTA capped at 4/2         tile sweep (EXL3_HC_MIX_V3_UP_B)
  v3-auto      mode 3, both caps 0 = one-wave auto tiling    (EXL3_HC_MIX_V3_DOTS_B=0 _UP_B=0)
  v3-dauto/uauto  auto on one launch only

Timing: CUDA events around >= --reps chains. The chains cycle through --sites different HC sites
(default 24 real sites = ~160 MB of int8 weights, above the 96 MB L2) so weights come from DRAM as in
decode; --sites 1 gives the L2-hot number. Two timings per variant:
  graph  the chain sequence captured once in a CUDA graph and replayed: GPU time per chain (the bars)
  eager  plain Python launches back to back: includes host launch cost
Variants run in the listed order, and then the reference again ("reference#2") as a drift check.

Pre-registered (HOW-TO-VERIFY.md section 3), graph timing, 24 cold sites:
  control  copy within +-3 % of reference at every R, and reference#2 within +-3 % of reference
           (else the bench is not measuring kernels: stop)
  PASS     v3 <= 0.75 x reference at R = 4 AND at R = 16
  KILL     v3 > 0.92 x reference at R = 4 AND at R = 16
  tile     a cap variant replaces the default caps only if it is >= 5 % below v3 at R = 4 and R = 16
           and not > 3 % above v3 at R = 1 and R = 8

  docker run --rm --gpus '"device=0"' -v /srv/qwen5090/models:/models:ro -v $R:/results --entrypoint python3 \\
    tabbyapi:hcfast-r1 /opt/hcfast-r1/bench_hcfast_p0.py \\
    --model /models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab --json /results/p0.json

--ncu: one eager chain of each listed variant per row count after 3 warm chains, inside NVTX ranges
"<variant> R=<n>", for ncu --nvtx.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

for k, v in {
    "EXL3_HC_MIX_V2": "1", "EXL3_HC_MIX_V2_MIN_R": "1", "EXL3_HC_MIX_V2_INT8": "1",
    "EXL3_GR_STATE_REGRID": "1", "EXL3_GR_STATE_IN_UP": "1", "EXL3_HC_APPLY_WARP1": "1",
}.items():
    os.environ[k] = v
os.environ.pop("EXL3_HC_MIX_V3", None)

# name -> (mode, dots cap, up cap); None = the served binding
VARIANTS = {
    "reference": None,
    "copy": (0, 8, 8),
    "v3": (3, 8, 8),
    "v3-dots": (1, 8, 8),
    "v3-up": (2, 8, 8),
    "v3-d4": (3, 4, 8),
    "v3-d2": (3, 2, 8),
    "v3-d1": (3, 1, 8),
    "v3-u4": (3, 8, 4),
    "v3-u2": (3, 8, 2),
    "v3-auto": (3, 0, 0),
    "v3-dauto": (3, 0, 8),
    "v3-uauto": (3, 8, 0),
    "reference#2": None,
}


def arguments():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default = "/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab")
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--rows", type = int, nargs = "+", default = [1, 4, 8, 16])
    ap.add_argument("--reps", type = int, default = 504, help = "chains timed per variant and row count (>= 500)")
    ap.add_argument("--sites", type = int, default = 24, help = "distinct HC sites cycled (L2 rotation)")
    ap.add_argument("--warmup", type = int, default = 50)
    ap.add_argument("--variants", nargs = "+", default = list(VARIANTS))
    ap.add_argument("--synthetic", action = "store_true", help = "random weights of the checkpoint shapes")
    ap.add_argument("--ncu", action = "store_true", help = "one eager chain per variant and row count, for ncu")
    ap.add_argument("--json", type = Path)
    a = ap.parse_args()
    assert a.reps >= 500 or a.ncu, "at least 500 reps"
    assert all(1 <= r <= 32 for r in a.rows)
    assert all(v in VARIANTS for v in a.variants), a.variants
    return a


def main():
    args = arguments()
    import torch
    from exllamav3.modules.hyperconnections import GatedResidual
    from exllamav3.ext import exllamav3_ext as ext
    assert getattr(ext, "hc_mix_v3_revision", None) == 1, "extension without gr_mix_v2_int8_v3"
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

    results = {}
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

        def chain(variant, i):
            m = sites[i]
            ext.hc_apply(x[i], y[i], pp[i], None)
            spec = VARIANTS[variant]
            if spec is None:
                ext.gr_mix_v2_int8_statein(x[i], m.fn_q, m.fn_s, m.upx_q, m.upx_s, m.w_h, m.rms_eps,
                                           dots, state, post, mixed)
            else:
                ext.gr_mix_v2_int8_v3(x[i], m.fn_q, m.fn_s, m.upx_q, m.upx_s, m.w_h, m.rms_eps,
                                      dots, state, post, mixed, *spec)

        if args.ncu:
            for variant in args.variants:
                for i in range(3):
                    chain(variant, i % len(sites))
                torch.cuda.synchronize(dev)
                torch.cuda.nvtx.range_push(f"{variant} R={R}")
                chain(variant, 0)
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize(dev)
            print(f"R={R}: one chain per variant issued for ncu")
            continue

        row = {}
        n_sites = len(sites)
        replays = math.ceil(args.reps / n_sites)
        chains = replays * n_sites
        for variant in args.variants:
            for k in range(args.warmup):
                chain(variant, k % n_sites)
            torch.cuda.synchronize(dev)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(n_sites):
                    chain(variant, i)
            graph.replay()
            torch.cuda.synchronize(dev)
            e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            e0.record()
            for _ in range(replays):
                graph.replay()
            e1.record()
            torch.cuda.synchronize(dev)
            us_graph = e0.elapsed_time(e1) * 1000.0 / chains
            del graph
            e0.record()
            for k in range(chains):
                chain(variant, k % n_sites)
            e1.record()
            torch.cuda.synchronize(dev)
            us_eager = e0.elapsed_time(e1) * 1000.0 / chains
            row[variant] = {"graph_us": us_graph, "eager_us": us_eager, "chains": chains}
        results[R] = row
        ref = row.get("reference", {}).get("graph_us")
        line = "  ".join(f"{v} {row[v]['graph_us']:6.2f}" + (f" ({row[v]['graph_us'] / ref:4.2f}x)" if ref else "")
                         for v in args.variants)
        print(f"R={R:2d}  graph us/chain: {line}   ({chains} chains each)")
        print(f"      eager us/chain: " + "  ".join(f"{v} {row[v]['eager_us']:6.2f}" for v in args.variants))

    if args.ncu:
        return 0
    print()
    verdicts = []

    def gus(R, v):
        return results.get(R, {}).get(v, {}).get("graph_us")

    for R in results:
        ref, cp, ref2 = gus(R, "reference"), gus(R, "copy"), gus(R, "reference#2")
        if ref and cp and abs(cp / ref - 1) > 0.03:
            verdicts.append(f"CONTROL FAIL R={R}: copy {cp:.2f} vs reference {ref:.2f} us (> 3 %)")
        if ref and ref2 and abs(ref2 / ref - 1) > 0.03:
            verdicts.append(f"CONTROL FAIL R={R}: reference#2 {ref2:.2f} vs reference {ref:.2f} us (drift > 3 %)")
    r4, r16 = (gus(4, "v3") / gus(4, "reference") if gus(4, "v3") and gus(4, "reference") else None,
               gus(16, "v3") / gus(16, "reference") if gus(16, "v3") and gus(16, "reference") else None)
    if r4 is not None and r16 is not None:
        verdicts.append(f"v3 / reference: R=4 {r4:.3f}, R=16 {r16:.3f}")
        if r4 <= 0.75 and r16 <= 0.75:
            verdicts.append("P0 PASS (<= 0.75 at R=4 and R=16)")
        elif r4 > 0.92 and r16 > 0.92:
            verdicts.append("P0 KILL (> 0.92 at R=4 and R=16)")
        else:
            verdicts.append("P0 MARGINAL (between the bars): run P1, it decides")
    for v in ("v3-d4", "v3-d2", "v3-d1", "v3-u4", "v3-u2", "v3-auto", "v3-dauto", "v3-uauto"):
        if all(gus(R, v) and gus(R, "v3") for R in (1, 4, 8, 16)):
            gain = [gus(R, v) / gus(R, "v3") for R in (1, 4, 8, 16)]
            if gain[1] <= 0.95 and gain[3] <= 0.95 and gain[0] <= 1.03 and gain[2] <= 1.03:
                verdicts.append(f"TILE: {v} beats default caps ({', '.join(f'{x:.3f}' for x in gain)} at R=1/4/8/16)")
    for vd in verdicts:
        print(vd)
    if args.json:
        args.json.write_text(json.dumps({"rows": results, "sites": len(sites), "verdicts": verdicts,
                                         "synthetic": args.synthetic, "variants": VARIANTS}, indent = 2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
