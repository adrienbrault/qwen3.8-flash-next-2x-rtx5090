#!/usr/bin/env python3
"""TP=2 bounding probe, part 3: the budget join and the GO / NO-GO decision. Pure Python (runs on the box after the
GPU parts, and in the CPU tests).

    projected TP step = sum_f s_f * K_f  +  n_AR_trunk * t_AR(trunk rows) + n_AR_draft * t_AR(draft rows)
                        + (E - K)                               <- host/gap, kept at the layer-split value (ASSUMPTION)
                        + (1 - s_MOE) * O                       <- shared-expert overlap correction (see below)

  K_f     R519 kernel ms/step of family f (sum over both cards; data/r519_<cfg>_kernels.json, profiler mode)
  s_f     measured t(half)/t(full) from tp_bound_gpu.py at the matching rows; the slower rank (s = max(rank0, rank1)
          / full) because the step waits for both cards
  E, K    R519 elapsed and kernel ms/step (profiler mode); E - K is everything that is not kernel time
  O       per-step time of shared-expert kernels HIDDEN under the routed kernels by the served side-stream overlap
          (O_layer = t_routed + t_shared - t_full, measured). K counts those kernels although they cost no wall time,
          so E - K = G - O. Under TP the hidden part shrinks with the MoE layer: G' - O' = (E - K) + (1 - s_MOE) * O.
  n_AR    2 per trunk layer (after attention/GDN out_proj, after MoE down + shared down) x 48, plus 2 per MTP block
          forward x 3 draft forwards (d3) = 96 + 6 = 102. t_AR = fastest VALIDATED all-reduce measured on this box at
          (rows, 2560, fp32) inside a CUDA graph (eager if no graph-capable method validated).

Families (first matching rule wins; every kernel above 0.5 % of kernel time must hit an explicit rule, the CPU test
enforces it on R519 c1d3 and c4d3):

  MOE_K2 / MOE_K3  coop V2 A/B kernels with K = 2 / 3 (routed experts), plus the rest of the MoE block (rot kernel,
                   shared-expert mgemm, act_mul, the MoE share of the split-o gemv) apportioned to K2/K3 by launches
  GDN              recurrent step, conv1d update, gdn_ba gemv, fused op, gated norm, qkvz mgemm, GDN share of split-o
  GDN_REWIND       batched state / conv rewind (per-head state; s = 0.5 ASSUMED, the state halves with the heads)
  ATTN             paged decode split/combine, QSA planes, rope, cache quant, q/k/v bundle, indexer gemv, attention
                   share of split-o
  MTP_BLOCK        the draft block on cuda:0 (K=4 coop, K5/K6 EXL3 on dev 0, the draft-block gemv); s from the measured
                   MTP attention + MTP MoE halves (time-weighted)
  HEAD             lm_head + pruned draft head (K5/K6 EXL3 on dev 1), argmax / fused sampler
  HC               gated-residual mixer V2 kernels + hc_apply (replicated in design v1: s = 1)
  ROUTER           router GEMM (cutlass 16x16 + splitK reduce), top-k (replicated: s = 1)
  OTHER            everything else (torch elementwise, copies, PLE, embedding, norms): s = 1

The split-o group (exl3_gemv/gemm <4, true, ...>, 96 launches/step = 36 GDN out_proj + 12 attention o_proj + 48 shared
down) is apportioned by weight bytes (6144x2560, 6144x2560, 640x2560 per launch; all K = 4).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

GO_THRESHOLD = 1.25
N_LAYERS, N_GDN, N_ATTN = 48, 36, 12
DRAFT_FORWARDS = 3                       # d3
AR_PER_BLOCK = 2
VOCAB, MTP_HEAD_N = 248320, 65536
REF_AR_US = {"R184 vLLM custom all-reduce, graph, 1 row": 4.9, "R185 pcie_ipc slab, 1 row": 1.94}

CONFIGS = {
    "c1d3": {"trunk_shape": "1x4", "moe_rows": 4, "draft_shape": "1x1", "draft_rows": 1, "lm_rows": 4},
    "c4d3": {"trunk_shape": "4x4", "moe_rows": 16, "draft_shape": "4x1", "draft_rows": 4, "lm_rows": 16},
}

# SPLIT_O weights (bytes per launch x launches per step)
SPLIT_O_SHARES = {"GDN": N_GDN * 6144 * 2560, "ATTN": N_ATTN * 6144 * 2560, "MOE": N_LAYERS * 640 * 2560}

_K56 = re.compile(r"exl3_(gemm|gemv|mgemm|gemv_int8_sq)_kernel<[56],")


def _match_any(name, pats):
    return any(p in name for p in pats)


def classify(name: str, device: str | None = None) -> str | None:
    """Family of one kernel (name = full R519 name). device: '0' / '1' when the caller splits by card."""
    n = name
    if "exl3_moe_coop_a_kernel<4," in n or "exl3_moe_coop_b_kernel<4," in n:
        return "MTP_BLOCK"
    if "exl3_moe_coop_a_kernel<2," in n or "exl3_moe_coop_b_kernel<2," in n:
        return "MOE_K2"
    if "exl3_moe_coop_a_kernel<3," in n or "exl3_moe_coop_b_kernel<3," in n:
        return "MOE_K3"
    if "exl3_moe_coop_rot_kernel" in n:
        return "MOE"
    if _match_any(n, ("recurrent_gated_delta_rule", "gdn_ba_gemv", "conv1d_update", "gated_delta_net_fused_op",
                      "gated_rms_norm", "exl3_mgemm_kernel<4, true,")):
        return "GDN"
    if _match_any(n, ("batched_state_rewind", "batched_conv_rewind")):
        return "GDN_REWIND"
    if _match_any(n, ("exl3_mgemm_kernel<4, false, 2, 16, 32, 128,", "act_mul_kernel")):
        return "MOE"
    if _match_any(n, ("exl3_mgemm_kernel<4, false, 2, 16, 32, 256,", "exl3_gemv_kernel<4, false, 2, 1, 0, false>",
                      "exl3_gemm_kernel<4, false, 2, 16, 32, 128,", "_paged_attn_decode", "_qsa_", "_mla_plane_update",
                      "rope_kernel", "quant_cache_paged", "deinterleave_qg", "mul_sigmoid")):
        return "ATTN"
    if _match_any(n, ("exl3_gemv_kernel<4, true, 2, 1, 0, false>", "exl3_gemm_kernel<4, true, 2, 16, 32, 128,")):
        return "SPLIT_O"
    if _K56.search(n):
        if device is None:
            return "K56"
        return "HEAD" if device == "1" else "MTP_BLOCK"
    if "exl3_gemv_kernel<4, false, 2, 0, 0, false>" in n:
        return "MTP_BLOCK"
    if _match_any(n, ("ArgMax", "fs_partial_argmax", "fs_finalize")):
        return "HEAD"
    if _match_any(n, ("cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_64x2", "splitKreduce", "routing_std_topk",
                      "routing_gemv")):
        return "ROUTER"
    if _match_any(n, ("gr_v2_", "hc_apply")):
        return "HC"
    return None


def family_totals(kern: dict) -> dict:
    """ms/step per family from an R519 kernels.json (per-device split for the K5/K6 kernels)."""
    it = kern["capture"]["iterations"]
    fam, unmatched, explicit = {}, [], 0.0
    for k in kern["kernels"]:
        for dev, v in k["devices"].items():
            ms = v["total_cuda_us"] / 1000.0 / it
            f = classify(k["name"], dev)
            if f is None:
                unmatched.append((k["name"], ms))
                f = "OTHER"
            else:
                explicit += ms
            fam[f] = fam.get(f, 0.0) + ms
    total = sum(fam.values())
    # apportion split-o
    so = fam.pop("SPLIT_O", 0.0)
    wsum = sum(SPLIT_O_SHARES.values())
    for f, w in SPLIT_O_SHARES.items():
        fam[f] = fam.get(f, 0.0) + so * w / wsum
    # the MoE-block remainder (rot, shared expert, act_mul, split-o share) goes to K2/K3 by coop launches
    moe = fam.pop("MOE", 0.0)
    l2 = _launches(kern, "exl3_moe_coop_a_kernel<2,")
    l3 = _launches(kern, "exl3_moe_coop_a_kernel<3,")
    if l2 + l3 > 0:
        fam["MOE_K2"] = fam.get("MOE_K2", 0.0) + moe * l2 / (l2 + l3)
        fam["MOE_K3"] = fam.get("MOE_K3", 0.0) + moe * l3 / (l2 + l3)
    else:
        fam["MOE_K2"] = fam.get("MOE_K2", 0.0) + moe
    unmatched.sort(key=lambda x: -x[1])
    return {"families": fam, "total_ms": total, "explicit_ms": explicit, "split_o_ms": so,
            "unmatched": unmatched, "moe_launches": {"K2": l2, "K3": l3}}


def _launches(kern, pat):
    it = kern["capture"]["iterations"]
    return sum(k["launches"] for k in kern["kernels"] if pat in k["name"]) / it


# ----------------------------------------------------------------------------------------------------------------------
# Probe results
# ----------------------------------------------------------------------------------------------------------------------

def _s(entry):
    return None if entry is None else entry.get("s")


def scaling_for(sc: dict, cfg: dict) -> dict:
    """{family: (s, provenance)} at the config's shapes. Missing primary families raise."""
    F = sc.get("families", {})
    out = {}

    def need(fam, key, sub):
        e = F.get(fam, {}).get(sub, {}).get(key)
        if e is None or e.get("s") is None:
            raise KeyError(f"scaling for {fam} at {sub}={key} missing from the probe results")
        return e

    g = need("GDN", cfg["trunk_shape"], "shapes")
    out["GDN"] = (g["s"], f"GDN {cfg['trunk_shape']}")
    a = need("ATTN", cfg["trunk_shape"], "shapes")
    out["ATTN"] = (a["s"], f"ATTN {cfg['trunk_shape']}")
    for k in ("MOE_K2", "MOE_K3"):
        m = need(k, str(cfg["moe_rows"]), "rows")
        out[k] = (m["s"], f"{k} rows {cfg['moe_rows']}")
    out["GDN_REWIND"] = (0.5, "ASSUMED (state halves with the heads)")
    # MTP block: time-weighted attention + MoE of the draft block; fall back to the trunk ATTN/MoE at draft shapes
    ma = F.get("MTP_ATTN", {}).get("shapes", {}).get(cfg["draft_shape"])
    mm = F.get("MTP_MOE", {}).get("rows", {}).get(str(cfg["draft_rows"]))
    if ma and mm and ma.get("s") and mm.get("s"):
        full = ma["full"]["median_us"] + mm["full"]["median_us"]
        half = max(ma[r]["median_us"] for r in ("rank0", "rank1")) + max(mm[r]["median_us"] for r in ("rank0", "rank1"))
        out["MTP_BLOCK"] = (half / full, f"MTP attn {cfg['draft_shape']} + MTP MoE rows {cfg['draft_rows']}")
    else:
        ta = F.get("ATTN", {}).get("shapes", {}).get(cfg["draft_shape"])
        if ta and ta.get("s"):
            out["MTP_BLOCK"] = (ta["s"], f"FALLBACK trunk ATTN {cfg['draft_shape']} (MTP arms missing)")
        else:
            out["MTP_BLOCK"] = (1.0, "FALLBACK s = 1 (no MTP measurement)")
    out["HC"] = (1.0, "replicated (design v1)")
    out["ROUTER"] = (1.0, "replicated")
    out["OTHER"] = (1.0, "replicated / unsplit")
    out["HEAD"] = (1.0, "replicated")
    return out


def overlap_ms(sc: dict, cfg: dict) -> float:
    """Per-step hidden shared-expert time O (ms): 48 trunk layers at moe_rows (K2/K3 by R519 launch counts is second
    order; the mean of the two layers is used) + 3 draft-block forwards at draft_rows."""
    F = sc.get("families", {})
    vals = [F[k]["overlap"][str(cfg["moe_rows"])]["hidden_us"] for k in ("MOE_K2", "MOE_K3")
            if str(cfg["moe_rows"]) in F.get(k, {}).get("overlap", {})]
    o = (statistics.mean(vals) if vals else 0.0) * N_LAYERS
    mo = F.get("MTP_MOE", {}).get("overlap", {}).get(str(cfg["draft_rows"]))
    if mo:
        o += mo["hidden_us"] * DRAFT_FORWARDS
    return o / 1000.0


def head_sharded(sc: dict, cfg: dict):
    """(s_lm, s_draft) of a vocab-sharded head, or None."""
    h = sc.get("families", {}).get("HEAD", {}).get("rows", {})
    lm, dr = h.get(f"lm_{cfg['lm_rows']}"), h.get(f"draft_{cfg['draft_rows']}")
    if not lm or not dr or not lm.get("s") or not dr.get("s"):
        return None
    return lm["s"], dr["s"]


def ar_candidates(ar_results: list, rows: int, width: int = 2560, dtype: str = "fp32") -> list:
    return [r for r in ar_results if r.get("valid") and r["rows"] == rows and r["width"] == width
            and r["dtype"] == dtype]


def pick_ar(ar_results: list, rows: int, width: int = 2560, dtype: str = "fp32"):
    """Fastest validated method: graph mode preferred (the decode layers run as CUDA graphs), eager otherwise."""
    c = ar_candidates(ar_results, rows, width, dtype)
    if not c:
        return None
    g = [r for r in c if r["mode"] == "graph"]
    best = min(g or c, key=lambda r: r["median_us"])
    return best


# ----------------------------------------------------------------------------------------------------------------------
# Budget
# ----------------------------------------------------------------------------------------------------------------------

def budget(kern: dict, s_map: dict, t_ar_trunk_us: float, t_ar_draft_us: float, overlap: float,
           s_moe_eff: float, e_ms: float | None = None) -> dict:
    ft = family_totals(kern)
    fam = ft["families"]
    it = kern["capture"]["iterations"]
    E = e_ms if e_ms is not None else kern["capture"]["elapsed_ms_per_step"]
    K = kern["total_kernel_ms"] / it
    gap = E - K
    rows = {}
    proj_k = 0.0
    for f, ms in sorted(fam.items(), key=lambda x: -x[1]):
        s, src = s_map.get(f, (1.0, "unmapped: s = 1"))
        rows[f] = {"r519_ms": ms, "s": s, "tp_ms": ms * s, "src": src}
        proj_k += ms * s
    n_trunk = AR_PER_BLOCK * N_LAYERS
    n_draft = AR_PER_BLOCK * DRAFT_FORWARDS
    ar_ms = (n_trunk * t_ar_trunk_us + n_draft * t_ar_draft_us) / 1000.0
    ov = (1.0 - s_moe_eff) * overlap
    step = proj_k + ar_ms + gap + ov
    return {"E_ms": E, "K_ms": K, "gap_ms": gap, "family_sum_ms": sum(fam.values()), "families": rows,
            "tp_kernel_ms": proj_k, "n_ar": n_trunk + n_draft, "n_ar_trunk": n_trunk, "n_ar_draft": n_draft,
            "t_ar_trunk_us": t_ar_trunk_us, "t_ar_draft_us": t_ar_draft_us, "ar_ms": ar_ms,
            "overlap_ms": overlap, "overlap_correction_ms": ov, "step_ms": step, "speedup": E / step,
            "unmatched_top": ft["unmatched"][:8], "explicit_share": ft["explicit_ms"] / ft["total_ms"]}


def s_moe_effective(s_map, fam):
    w2, w3 = fam.get("MOE_K2", 0.0), fam.get("MOE_K3", 0.0)
    if w2 + w3 == 0:
        return 1.0
    return (s_map["MOE_K2"][0] * w2 + s_map["MOE_K3"][0] * w3) / (w2 + w3)


def moe_split(sc: dict, cfg: dict) -> dict:
    """The MoE tensor split the probe's halves used (tp_bound_gpu.Probe.moe_split) and the arms behind s at the
    config's rows. s = max(rank0, rank1) / full is the critical (384-channel) rank; 0.6 is its structural floor."""
    out = {}
    for k in ("MOE_K2", "MOE_K3"):
        f = sc.get("families", {}).get(k, {})
        st = f.get("rows", {}).get(str(cfg["moe_rows"]), {})
        out[k] = {"split": f.get("info", {}).get("split"), "key": f.get("key"),
                  "us": {r: st[r]["median_us"] for r in ("full", "rank0", "rank1") if r in st}, "s": st.get("s")}
    return out


def fmt_moe_split(ms: dict, rows) -> list:
    L = []
    sp = next((v["split"] for v in ms.values() if v.get("split")), None)
    if sp:
        moe = ", ".join(f"{r} [{a}, {b})" for r, (a, b) in sp["moe"].items())
        sh = ", ".join(f"{r} [{a}, {b})" for r, (a, b) in sp.get("shared", {}).items())
        L.append(f"MoE tensor split ({sp['unit_channels']}-channel units): experts {sp['moe_intermediate']} -> {moe}"
                 + (f"; shared expert {sp.get('shared_intermediate')} -> {sh}" if sh else "")
                 + f"; critical {sp['critical_rank']} holds {sp['critical_share']:.2f} (structural floor of s_MOE)")
    for k, v in ms.items():
        u = v["us"]
        if u:
            L.append(f"  {k} rows {rows}: full {u.get('full', float('nan')):.1f} us, rank0 "
                     f"{u.get('rank0', float('nan')):.1f} us, rank1 {u.get('rank1', float('nan')):.1f} us -> s = "
                     f"max(rank0, rank1) / full = {v['s']}  ({v.get('key')})")
    return L


def evaluate(cfg_name: str, kern: dict, wall: dict | None, sc: dict, ar_results: list) -> dict:
    cfg = CONFIGS[cfg_name]
    s_map = scaling_for(sc, cfg)
    fam = family_totals(kern)["families"]
    s_moe = s_moe_effective(s_map, fam)
    ov = overlap_ms(sc, cfg)
    tr = pick_ar(ar_results, cfg["moe_rows"])
    dr = pick_ar(ar_results, cfg["draft_rows"])
    if tr is None or dr is None:
        raise KeyError(f"{cfg_name}: no validated all-reduce at rows {cfg['moe_rows']} / {cfg['draft_rows']}, fp32, "
                       f"width 2560")
    out = {"config": cfg_name, "shapes": cfg, "ar_trunk": tr, "ar_draft": dr, "s_moe_effective": s_moe,
           "ar_candidates_trunk": sorted(({"method": r["method"], "mode": r["mode"], "median_us": r["median_us"]}
                                          for r in ar_candidates(ar_results, cfg["moe_rows"])),
                                         key=lambda r: r["median_us"])}
    prim = budget(kern, s_map, tr["median_us"], dr["median_us"], ov, s_moe)
    out["primary"] = prim
    out["moe_split"] = moe_split(sc, cfg)
    # head: best of replicated (primary) and vocab-sharded (+ one small exchange per head call)
    hs = head_sharded(sc, cfg)
    if hs is not None:
        s2 = dict(s_map)
        head_ms = prim["families"].get("HEAD", {}).get("r519_ms", 0.0)
        # R519 does not separate the full head (1 call) from the pruned draft head (3 calls) by name at every shape;
        # weight the two s by the weight bytes each streams per step (memory-bound GEMV/GEMM): 1 x 248320 columns vs
        # 3 x EXL3_MTP_HEAD_N (65536) columns
        w_lm, w_dr = 1 * VOCAB, DRAFT_FORWARDS * MTP_HEAD_N
        s_head = (w_lm * hs[0] + w_dr * hs[1]) / (w_lm + w_dr)
        s2["HEAD"] = (s_head, f"vocab-sharded head (lm s {hs[0]:.3f}, draft s {hs[1]:.3f}, byte-weighted)")
        b = budget(kern, s2, tr["median_us"], dr["median_us"], ov, s_moe)
        b["step_ms"] += (tr["median_us"] + DRAFT_FORWARDS * dr["median_us"]) / 1000.0   # argmax gather per head call
        b["speedup"] = b["E_ms"] / b["step_ms"]
        out["head_sharded"] = b
        out["head_ms_r519"] = head_ms
    best = min([out["primary"]] + ([out["head_sharded"]] if "head_sharded" in out else []), key=lambda b: b["step_ms"])
    out["best"] = {"step_ms": best["step_ms"], "speedup": best["speedup"],
                   "variant": "head_sharded" if best is out.get("head_sharded") else "primary"}
    # HC sharded sensitivity: s_HC from the probe + TWO extra exchanges per site (2 sites/layer, 2 per draft block):
    # an all-reduce of the partial dots (R x (rank + H) floats, latency-bound) and an all-gather of the mixed output,
    # which the column-parallel projections need at full hidden; both costed at t_AR of the same rows
    hc = sc.get("families", {}).get("HC", {}).get("shapes", {}).get(cfg["trunk_shape"])
    if hc and hc.get("s"):
        s3 = dict(s_map)
        s3["HC"] = (hc["s"], f"HC split hidden {cfg['trunk_shape']} (sensitivity)")
        b = budget(kern, s3, tr["median_us"], dr["median_us"], ov, s_moe)
        b["step_ms"] += 2 * (2 * N_LAYERS * tr["median_us"] + 2 * DRAFT_FORWARDS * dr["median_us"]) / 1000.0
        b["speedup"] = b["E_ms"] / b["step_ms"]
        out["hc_sharded"] = b
    # bounds
    half = {f: (0.5, "perfect scaling") for f in list(fam) + ["GDN_REWIND"]}
    out["bound_perfect_s05_ar0"] = budget(kern, half, 0.0, 0.0, ov, 0.5)
    out["bound_perfect_s05_measured_ar"] = budget(kern, half, tr["median_us"], dr["median_us"], ov, 0.5)
    struct = dict(s_map)
    for f in ("GDN", "ATTN", "MTP_BLOCK"):
        struct[f] = (0.5, "structural (perfect split)")
    for f in ("MOE_K2", "MOE_K3"):
        struct[f] = (0.6, "structural (384/640 channels on the critical rank)")
    out["bound_structural_measured_ar"] = budget(kern, struct, tr["median_us"], dr["median_us"], ov, 0.6)
    out["bound_measured_s_ar0"] = budget(kern, s_map, 0.0, 0.0, ov, s_moe)
    for label, us in REF_AR_US.items():
        out.setdefault("ref_ar", {})[label] = budget(kern, s_map, us, us, ov, s_moe)["speedup"]
    if wall is not None:
        e_wall = wall["capture"]["elapsed_ms_per_step"]
        out["wall"] = {"E_wall_ms": e_wall, "projected_wall_ms": out["best"]["step_ms"] * e_wall / prim["E_ms"],
                       "assumption": "wall step scales like the profiler-mode step (ratio E_wall / E_profiler)"}
    out["go"] = out["best"]["speedup"] >= GO_THRESHOLD
    return out


def fmt_eval(ev: dict) -> str:
    L = []
    p = ev["primary"]
    L.append(f"=== {ev['config']} (trunk {ev['shapes']['trunk_shape']}, MoE rows {ev['shapes']['moe_rows']}, draft "
             f"{ev['shapes']['draft_shape']}) ===")
    L.append(f"R519 profiler step E = {p['E_ms']:.3f} ms, kernel K = {p['K_ms']:.3f} ms, host/gap E-K = "
             f"{p['gap_ms']:.3f} ms (kept as-is under TP: ASSUMPTION)")
    L.append(f"{'family':<12} {'R519 ms':>8} {'s':>6} {'TP ms':>8}  source")
    for f, r in p["families"].items():
        L.append(f"{f:<12} {r['r519_ms']:8.3f} {r['s']:6.3f} {r['tp_ms']:8.3f}  {r['src']}")
    L.append(f"{'kernels':<12} {p['family_sum_ms']:8.3f} {'':6} {p['tp_kernel_ms']:8.3f}")
    L += fmt_moe_split(ev.get("moe_split", {}), ev["shapes"]["moe_rows"])
    L.append(f"all-reduce: {p['n_ar_trunk']} x {p['t_ar_trunk_us']:.2f} us ({ev['ar_trunk']['method']} "
             f"{ev['ar_trunk']['mode']}, rows {ev['shapes']['moe_rows']}) + {p['n_ar_draft']} x "
             f"{p['t_ar_draft_us']:.2f} us (rows {ev['shapes']['draft_rows']}) = {p['ar_ms']:.3f} ms")
    L.append("  validated all-reduce candidates at the trunk rows (fp32 x 2560): " + ", ".join(
        f"{c['method']}/{c['mode']} {c['median_us']:.2f} us" for c in ev.get("ar_candidates_trunk", [])))
    L.append(f"host/gap {p['gap_ms']:.3f} ms + shared-overlap correction (1 - s_MOE {ev['s_moe_effective']:.3f}) x "
             f"O {p['overlap_ms']:.3f} = {p['overlap_correction_ms']:.3f} ms")
    L.append(f"PROJECTED TP step (head replicated) = {p['step_ms']:.3f} ms -> speedup {p['speedup']:.3f}x")
    if "head_sharded" in ev:
        h = ev["head_sharded"]
        L.append(f"  variant vocab-sharded head (+ argmax gather per head call) = {h['step_ms']:.3f} ms -> "
                 f"{h['speedup']:.3f}x")
    if "hc_sharded" in ev:
        h = ev["hc_sharded"]
        L.append(f"  sensitivity sharded HC mixer (+2 exchanges per site) = {h['step_ms']:.3f} ms -> {h['speedup']:.3f}x")
    L.append(f"BEST projection ({ev['best']['variant']}) = {ev['best']['step_ms']:.3f} ms -> "
             f"{ev['best']['speedup']:.3f}x")
    if "wall" in ev:
        w = ev["wall"]
        L.append(f"  wall-mode view: R519 wall {w['E_wall_ms']:.3f} ms -> projected {w['projected_wall_ms']:.3f} ms "
                 f"({w['assumption']})")
    b = ev["bound_perfect_s05_ar0"]
    L.append(f"ceiling s = 0.5 everywhere, AR = 0      : {b['step_ms']:.3f} ms -> {b['speedup']:.3f}x")
    b = ev["bound_perfect_s05_measured_ar"]
    L.append(f"ceiling s = 0.5 everywhere, measured AR : {b['step_ms']:.3f} ms -> {b['speedup']:.3f}x")
    b = ev["bound_structural_measured_ar"]
    L.append(f"structural (GDN/ATTN/MTP 0.5, MoE 0.6, replicated 1.0), measured AR: {b['step_ms']:.3f} ms -> "
             f"{b['speedup']:.3f}x")
    b = ev["bound_measured_s_ar0"]
    L.append(f"measured s, AR = 0 (free all-reduce)    : {b['step_ms']:.3f} ms -> {b['speedup']:.3f}x")
    for k, v in ev.get("ref_ar", {}).items():
        L.append(f"measured s with t_AR = {k}: {v:.3f}x")
    return "\n".join(L)


def decide(evals: dict) -> str:
    c1 = evals["c1d3"]
    line = (f"DECISION: {'GO' if c1['go'] else 'NO-GO'} -- projected c1 d3 speedup {c1['best']['speedup']:.3f}x "
            f"(threshold {GO_THRESHOLD:.2f}x)")
    if "c4d3" in evals:
        line += f"; c4 d3 {evals['c4d3']['best']['speedup']:.3f}x"
    if not c1["go"]:
        line += "; TP is closed for this model on this box"
    return line


def load_ar(paths, notes=None):
    """All-reduce results of every file; a file whose transport check failed (NCCL not over P2P) is excluded."""
    res = []
    for p in paths:
        if p and os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            if d.get("transport", {}).get("ok") is False:
                if notes is not None:
                    notes.append(f"{os.path.basename(p)} excluded: transport {d['transport'].get('kinds')} is not P2P")
                continue
            res += d.get("results", [])
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scaling", required=True)
    ap.add_argument("--ar", nargs="+", required=True, help="tp_bound_ar.json and/or tp_bound_p2p.json")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="tp_bound_budget.json")
    a = ap.parse_args(argv)
    with open(a.scaling) as f:
        sc = json.load(f)
    notes = []
    ar_results = load_ar(a.ar, notes)
    evals, errors = {}, []
    for cfg in CONFIGS:
        with open(os.path.join(a.data, f"r519_{cfg}_kernels.json")) as f:
            kern = json.load(f)
        wall = None
        wp = os.path.join(a.data, f"wall_r519_{cfg}.json")
        if os.path.exists(wp):
            with open(wp) as f:
                wall = json.load(f)
        try:
            evals[cfg] = evaluate(cfg, kern, wall, sc, ar_results)
        except KeyError as e:
            errors.append(f"{cfg}: {e}")
    text = "\n\n".join(fmt_eval(e) for e in evals.values())
    if "c1d3" in evals:
        text += "\n\n" + decide(evals)
    for n in notes:
        text += f"\nNOTE: {n}"
    for e in errors:
        text += f"\nERROR: {e}"
    if not sc.get("ok", False):
        text += "\nWARNING: the scaling probe reported problems (see tp_bound_scaling.json ok/errors); the decision " \
                "above uses whatever it measured"
        text += (f"\n  scaling probe: ok={sc.get('ok')} missing_primary_families={sc.get('missing_primary_families')} "
                 f"incomplete_steps={sc.get('incomplete_steps')} respawns={len(sc.get('respawns', []))} "
                 f"errors={len(sc.get('errors', []))}")
        for e in sc.get("errors", [])[:10]:
            text += "\n  probe error: " + str(e).splitlines()[0][:300]
    print(text)
    with open(a.out, "w") as f:
        json.dump({"evals": evals, "errors": errors, "text": text}, f, indent=1, default=str)
    with open(os.path.splitext(a.out)[0] + ".txt", "w") as f:
        f.write(text + "\n")
    return 0 if not errors and "c1d3" in evals else 1


if __name__ == "__main__":
    sys.exit(main())
