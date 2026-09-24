#!/usr/bin/env python3
"""rows32 r4 GPU parity (r3's test on the stack-r3 image), one card: 4b (routed MoE coop, one launch at 17-32 rows)
and 4c (shared expert BC_GatedMLP graphs at 17-32 rows), bitwise.

r4 changes: run under the A env (the stack-r3 union: EXL3_MOE_COOP_V3=3, EXL3_MOE_COOP_V3_MAP=2-4:2, EXL3_DENSE_V2=1,
...). The served reference is that env's 16-row call (mode 3 = moefast r3's r2 kernels at 5-16 rows), and it must equal
the mode-2 16-row call (precondition "4b-ref-served==mode2", R713's claim). Arms at 17-32 rows: V3 unset / 1 / 2 / 3,
the A env (MAP 2-4:2: rows 17-32 fall back from mode 3 to mode 2 in the launcher) and the B env (--b-map, default
the A MAP + ",17-32:2"). Dispatch adds: at 24 rows both envs launch the V3 kernels and no r2 kernel, at 16 rows the A
env launches the r2 kernels (mode 3 is live where the MAP says so). Stress, join and bench run the B env.

4b reference = the served path at 16 rows. A 17..32-row call (EXL3_MOE_COOP_ROWS32=1, 32-row scratch) is compared
with TWO served 16-row calls on the served scratch sizes (flag unset, 160-slot scratch): rows [0, 16) and rows
[R - 16, R). Row r < 16 must equal the first call's row r, row r >= 16 the second call's row r - (R - 16), for the
output rows (routed sum + gated shared-expert term, fp32) and, per active slot, the down input act_out (fp16) and
the down GEMV rows d_out (fp32). So every row of the single launch is bitwise the row a served 16-row call computes.
The rows both windows share (R - 16 .. 15) must also agree between the two reference calls: that is the served
kernel's own row independence, the premise of the comparison, reported as PRECONDITION if it fails.
  arms: EXL3_MOE_COOP_V3 unset (V2), 1, 2 (served), and 3 when the moefast r3 extension is present, each against
        the one served reference (V3=2, flag unset);
  rows 17..32; routings: served-like unions (D = 10, ~4.6 R as the R708 capture, 10 R), every row on the same 10
        experts (runs of 8 slots, R slots per expert), a 24-expert pool, inactive slots incl. an empty token row,
        every other row empty; with and without the shared-expert term; real experts K2 (layer 20), K3 (layer 3),
        K4 (MTP layer) and synthetic K2/K3/K4; scratch and output start from different sentinels per arm.
  <= 16 rows unchanged: rows 1..16 with the flag on and the 32-row scratch == flag unset and the served scratch.
  dispatch (torch.profiler kernel names): 24 rows with the flag on run the V3 kernels (rotation + v3 A + v3 B);
        with the flag unset they keep the generic V1 kernels; 16 rows run the same kernels either way;
        EXL3_MOE_COOP_ROWS32=true is refused at a 17+-row launch.
  stress: 200 back-to-back launches per cell (rows 18, 24, 32, served-like routing, shared term) alternating two
        inputs on one scratch set; every launch compared with its reference (catches ordering races and stale reads).
4c: BC_GatedMLP built from the checkpoint's shared expert exactly as GatedMLP.load does (fused gate/up MultiLinear
    pointer table, silu, BC_LinearEXL3 down), statics (2, 32, H) / (2, 32, I) / (1, 32, I) / (1, 32, I):
  rows 17..32: the eager first call, the capturing second call and graph replays on fresh inputs are bitwise equal
        to an eager call of a fresh instance on the same input (the graph patches the input pointer);
  with densegemm's EXL3_DENSE_ROWS32 on and off when the extension has it (dense_v2_set_mode): both paths eager ==
        graph, and ON == OFF (densegemm's claim, checked here because the gate serves them together);
  the served statics (16 rows) refuse a 17-row call (loud, not a clamped slice).
Join: the shared expert at R rows on a side stream (the served overlap block: input event, side stream waits, graph,
    done event), then the coop launch at R rows with its output as the shared term, 100 launches alternating two
    inputs == the serial order. With the moefast r3 extension the join is the served one (exl3_moe_coop_ev waits
    right before stage B); on stack-r2 the main stream waits before the launch (the served run_bszN join is the same
    event at the same place, exercised end to end by the gate's P1 and served legs).

--bench adds a diagnostic timing table (CUDA events, us per call, R708-like unions): 16-row call, split (two 16-row
calls) and single launch at 18/21/24/28/32 rows. Not a gate.

Exit 0 and "PARITY PASS" only if every comparison is equal.
"""
import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

TOPK, NE, H, I = 10, 512, 2560, 640
EXPERT_LAYERS = {2: "model.language_model.layers.20.mlp", 3: "model.language_model.layers.3.mlp", 4: "mtp.layers.0.mlp"}
SHARED_LAYERS = ["model.language_model.layers.3.mlp", "model.language_model.layers.20.mlp", "mtp.layers.0.mlp"]
FLAG = "EXL3_MOE_COOP_ROWS32"


class Checkpoint:
    """Tensors by key via safetensors (model.safetensors.index.json), one handle per file (as d0_microbench)."""

    def __init__(self, model_dir, device):
        self.dir = Path(model_dir)
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.device = device
        self.handles = {}

    def get(self, key):
        from safetensors import safe_open
        f = self.map[key]
        if f not in self.handles:
            self.handles[f] = safe_open(str(self.dir / f), framework="pt", device=str(self.device))
        return self.handles[f].get_tensor(key)

    def linear(self, prefix):
        tr, suh, svh = (self.get(f"{prefix}.{s}").contiguous() for s in ("trellis", "suh", "svh"))
        return tr, suh, svh, f"{prefix}.mcg" in self.map, f"{prefix}.mul1" in self.map


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--stress", type=int, default=200)
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--bench", action="store_true", help="diagnostic timing table only (no parity)")
    ap.add_argument("--b-map", default=None, help="B env EXL3_MOE_COOP_V3_MAP (default: the A MAP + ',17-32:2')")
    a = ap.parse_args()

    assert os.environ.get("EXL3_MOE_COOP_V2") == "1", "run with the served env (EXL3_MOE_COOP_V2=1)"
    assert not os.environ.get("EXL3_MOE_COOP_DBG"), "EXL3_MOE_COOP_DBG bypasses V2/V3"
    served_v3 = os.environ.get("EXL3_MOE_COOP_V3", "")
    served_map = os.environ.get("EXL3_MOE_COOP_V3_MAP", "")
    b_map = a.b_map if a.b_map is not None else (served_map + "," if served_map else "") + "17-32:2"
    for k in ("EXL3_MOE_COOP_V3", "EXL3_MOE_COOP_V3_MAP", "EXL3_MOE_COOP_V3_L2EF", "EXL3_MOE_COOP_V3_HEAD", FLAG):
        os.environ.pop(k, None)
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    assert getattr(ext, "moe_rows32_revision", 0) >= 2, "needs the rows32 r4 extension"
    r3 = getattr(ext, "moe_coop_v3_revision", 0) >= 3
    has_dense = hasattr(ext, "dense_v2_set_mode")
    dev = torch.device("cuda:0")
    assert torch.cuda.get_device_capability(dev) == (12, 0), "V2/V3 are sm_120 only"
    torch.manual_seed(0)
    print(f"[parity] {torch.cuda.get_device_name(dev)}; served EXL3_MOE_COOP_V3={served_v3 or 'unset'} "
          f"MAP={served_map or 'unset'}; B MAP={b_map}; moefast revision {getattr(ext, 'moe_coop_v3_revision', 0)}; "
          f"densegemm {'yes' if has_dense else 'no'}", flush=True)
    # an arm is an int (EXL3_MOE_COOP_V3 alone) or a (V3, MAP) pair
    SERVED_MODE = (served_v3, served_map)                 # the A env: the reference and the <= 16-row arm
    B_ARM = (served_v3, b_map)                            # the B env's MoE part
    MODES = [0, 1, 2] + ([3] if r3 else []) + [SERVED_MODE, B_ARM]
    MODE_LABEL = {0: "V2", 1: "V3=1", 2: "V3=2", 3: "V3=3", SERVED_MODE: f"A-env({served_v3}|{served_map})",
                  B_ARM: f"B-env({served_v3}|{b_map})"}

    def scr_shapes(maxb):
        s = maxb * TOPK
        return dict(had_g=(s, H, torch.half), had_u=(s, H, torch.half), gu_g=(s, I, torch.half),
                    gu_u=(s, I, torch.half), act=(s, I, torch.half), d_out=(s, H, torch.float),
                    out=(maxb, H, torch.float)), s * (I // 128) + maxb * (H // 128) + 2 * s + 3

    def fresh(maxb, sentinel):
        shp, clen = scr_shapes(maxb)
        s = {k: torch.full((r, c), sentinel, dtype=dt, device=dev) for k, (r, c, dt) in shp.items()}
        s["ctr"] = torch.zeros((clen,), dtype=torch.int, device=dev)
        return s

    def set_env(mode, rows32):
        os.environ.pop("EXL3_MOE_COOP_V3", None)
        os.environ.pop("EXL3_MOE_COOP_V3_MAP", None)
        os.environ.pop(FLAG, None)
        v3, mp = mode if isinstance(mode, tuple) else (str(mode) if mode else "", "")
        if v3 and v3 != "0":
            os.environ["EXL3_MOE_COOP_V3"] = v3
        if mp:
            os.environ["EXL3_MOE_COOP_V3_MAP"] = mp
        if rows32:
            os.environ[FLAG] = "1" if rows32 is True else rows32

    def coop(tab, x, sel, rw, sh, gate_w, s, mode, rows32, sync=True):
        set_env(mode, rows32)
        try:
            ext.exl3_moe_coop(x, sel, rw, -1, -1, H,
                              tab["gate_trellis"], tab["gate_suh"], tab["gate_svh"],
                              tab["up_trellis"], tab["up_suh"], tab["up_svh"],
                              tab["down_trellis"], tab["down_suh"], tab["down_svh"],
                              None, None, None, tab["K"], tab["K"], tab["Kd"], tab["mcg"], tab["mul1"], 0, 0.0, True,
                              s["had_g"], s["had_u"], s["gu_g"], s["gu_u"], s["act"], s["d_out"], s["ctr"], s["out"],
                              sh, gate_w)
            if sync:
                torch.cuda.synchronize()
        finally:
            set_env(0, False)
        return s

    def bits_equal(u, v):
        if u.dtype == torch.half:
            return torch.equal(u.view(torch.int16), v.view(torch.int16))
        return torch.equal(u.view(torch.int32), v.view(torch.int32))

    def routings(rows, rng):
        pats = []
        for D in sorted({TOPK, min(NE, max(TOPK, round(4.6 * rows))), min(NE, TOPK * rows)}):
            perm = rng.sample(range(NE), D)
            pats.append((f"served-D{D}", [[perm[(r * TOPK + j) % D] for j in range(TOPK)] for r in range(rows)], None))
        same = rng.sample(range(NE), TOPK)
        pats.append(("same-top10", [list(same) for _ in range(rows)], None))
        pool = rng.sample(range(NE), 24)
        pats.append(("pool24", [rng.sample(pool, TOPK) for _ in range(rows)], None))
        sel = [rng.sample(range(NE), TOPK) for _ in range(rows)]
        mask = [[rng.random() < 0.3 for _ in range(TOPK)] for _ in range(rows)]
        mask[-1] = [True] * TOPK
        pats.append(("inactive", sel, mask))
        sel = [rng.sample(range(NE), TOPK) for _ in range(rows)]
        pats.append(("empty-rows", sel, [[r % 2 == 1] * TOPK for r in range(rows)]))
        return pats

    def inputs(rows, sel_l, mask, with_sh):
        sel = torch.tensor(sel_l, dtype=torch.long, device=dev)
        w = torch.rand((rows, TOPK), device=dev) + 0.1
        rw = (w / w.sum(-1, keepdim=True)).half()
        if mask is not None:
            rw[torch.tensor(mask, device=dev)] = 0.0
        rw = rw.contiguous()
        x = (torch.randn((rows, H), device=dev) * 0.5).half().contiguous()
        sh = (torch.randn((rows, H), device=dev) * 0.1).float().contiguous() if with_sh else None
        gate_w = (torch.randn((H,), device=dev) * 0.02).half().contiguous() if with_sh else None
        return x, sel, rw, sh, gate_w

    results, fails = [], 0

    def record(rec):
        nonlocal fails
        results.append(rec)
        if not rec["ok"]:
            fails += 1
            print(f"[parity] FAIL {rec}", flush=True)

    def reference(tab, x, sel, rw, sh, gate_w, R):
        """Two served 16-row calls: rows [0,16) and [R-16,R). Returns out (R,H), per-(row,k) act / d_out rows, and
        the precondition (the shared rows of both windows agree)."""
        lo, hi = 16, R - 16
        A = coop(tab, x[:16], sel[:16], rw[:16], None if sh is None else sh[:16], gate_w, fresh(16, -7.0), SERVED_MODE, False)
        B = coop(tab, x[hi:], sel[hi:], rw[hi:], None if sh is None else sh[hi:], gate_w, fresh(16, -9.0), SERVED_MODE, False)
        out = torch.cat([A["out"][:16], B["out"][16 - (R - 16):16]], 0)
        act = torch.cat([A["act"][:160], B["act"][(16 - (R - 16)) * TOPK:160]], 0)
        dout = torch.cat([A["d_out"][:160], B["d_out"][(16 - (R - 16)) * TOPK:160]], 0)
        pre = True
        if R < 32:
            ov = 32 - R                                  # rows hi..15 are in both windows
            active = (rw[hi:16].flatten() != 0).nonzero().flatten()
            pre = bits_equal(A["out"][hi:16], B["out"][:ov]) and \
                bits_equal(A["act"][hi * TOPK:160][active], B["act"][:ov * TOPK][active]) and \
                bits_equal(A["d_out"][hi * TOPK:160][active], B["d_out"][:ov * TOPK][active])
        if SERVED_MODE != (2, "") and SERVED_MODE != ("2", ""):
            # the served 16-row call (mode 3 at 5-16 rows under the stack-r3 MAP) == the mode-2 16-row call
            A2 = coop(tab, x[:16], sel[:16], rw[:16], None if sh is None else sh[:16], gate_w, fresh(16, -5.0), 2, False)
            pre = pre and bits_equal(A["out"][:16], A2["out"][:16])
        return out, act, dout, pre

    def parity_4b(tab, label, seeds):
        n0 = len(results)
        for R in range(17, 33):
            for seed in range(seeds):
                rng = random.Random(7919 * R + 104729 * seed + tab["K"])
                for name, sel_l, mask in routings(R, rng):
                    with_sh = (seed + R) % 2 == 0
                    x, sel, rw, sh, gate_w = inputs(R, sel_l, mask, with_sh)
                    out, act, dout, pre = reference(tab, x, sel, rw, sh, gate_w, R)
                    active = (rw.flatten() != 0).nonzero().flatten()
                    if not pre:
                        record(dict(kind="4b-precondition", label=label, K=tab["K"], rows=R, pattern=name, ok=False))
                    for mi, mode in enumerate(MODES):
                        got = coop(tab, x, sel, rw, sh, gate_w, fresh(32, 3.0 + mi), mode, True)
                        chk = {"out": bits_equal(out, got["out"][:R]),
                               "act": bits_equal(act[active], got["act"][:R * TOPK][active]),
                               "d_out": bits_equal(dout[active], got["d_out"][:R * TOPK][active])}
                        rec = dict(kind="4b", label=label, K=tab["K"], Kd=tab["Kd"], rows=R, pattern=name,
                                   shared=with_sh, mode=MODE_LABEL[mode], ok=all(chk.values()),
                                   **{f"eq_{k}": v for k, v in chk.items()})
                        if not rec["ok"]:
                            rec["max_abs_out"] = (out - got["out"][:R]).abs().max().item()
                        record(rec)
        rs = results[n0:]
        print(f"[parity] 4b {label} K{tab['K']}/{tab['Kd']}: {sum(r['ok'] for r in rs)}/{len(rs)} equal "
              f"(rows 17-32, arms {[MODE_LABEL[m] for m in MODES]}, vs two served 16-row calls)", flush=True)

    def unchanged_le16(tab, label):
        n0 = len(results)
        for R in range(1, 17):
            rng = random.Random(31 * R + tab["K"])
            for name, sel_l, mask in routings(R, rng)[:4]:
                x, sel, rw, sh, gate_w = inputs(R, sel_l, mask, R % 2 == 0)
                ref = coop(tab, x, sel, rw, sh, gate_w, fresh(16, -7.0), SERVED_MODE, False)
                got = coop(tab, x, sel, rw, sh, gate_w, fresh(32, 5.0), SERVED_MODE, True)
                record(dict(kind="le16-unchanged", label=label, K=tab["K"], rows=R, pattern=name,
                            ok=bits_equal(ref["out"][:R], got["out"][:R])))
        rs = results[n0:]
        print(f"[parity] <=16 rows, flag on + 32-row scratch vs served: {sum(r['ok'] for r in rs)}/{len(rs)} equal ({label})",
              flush=True)

    def dispatch(tab):
        from torch.profiler import profile, ProfilerActivity
        names = {}
        cells = [(24, True, SERVED_MODE), (24, False, SERVED_MODE), (16, True, SERVED_MODE), (16, False, SERVED_MODE),
                 (24, True, B_ARM), (16, True, B_ARM), (18, True, B_ARM), (32, True, B_ARM)]
        for R, rows32, arm in cells:
            sel = torch.tensor([[(r * TOPK + j) % 97 for j in range(TOPK)] for r in range(R)], dtype=torch.long, device=dev)
            rw = torch.full((R, TOPK), 0.1, dtype=torch.half, device=dev)
            x = (torch.randn((R, H), device=dev) * 0.5).half()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                coop(tab, x, sel, rw, None, None, fresh(32, 0.0), arm, rows32)
            names[(R, rows32, arm)] = sorted({e.name for e in prof.events() if "moe_coop" in e.name})
            print(f"[parity] dispatch rows {R} flag {'on' if rows32 else 'off'} {MODE_LABEL[arm]}: "
                  f"{names[(R, rows32, arm)]}", flush=True)
        on24, off24 = names[(24, True, SERVED_MODE)], names[(24, False, SERVED_MODE)]
        v3_only = lambda ks: (any("v3_a_kernel" in k for k in ks) and any("v3_b_kernel" in k for k in ks)
                              and any("rot_kernel" in k for k in ks) and not any("r2_" in k for k in ks))
        ok = (v3_only(on24) and all(v3_only(names[(R, True, B_ARM)]) for R in (18, 24, 32))
              and names[(24, True, B_ARM)] == on24
              and not any("v3_" in k or "v2_" in k or "r2_" in k for k in off24) and len(off24) > 0
              and names[(16, True, SERVED_MODE)] == names[(16, False, SERVED_MODE)] == names[(16, True, B_ARM)])
        if r3 and str(served_v3) == "3":
            ok = ok and any("r2_a_kernel" in k for k in names[(16, True, SERVED_MODE)])   # mode 3 live at 16 rows
        refused = False
        try:
            sel = torch.zeros((20, TOPK), dtype=torch.long, device=dev)
            coop(tab, torch.zeros((20, H), dtype=torch.half, device=dev), sel,
                 torch.full((20, TOPK), 0.1, dtype=torch.half, device=dev), None, None, fresh(32, 0.0), SERVED_MODE, "true")
        except RuntimeError as e:
            refused = "must be 0 or 1" in str(e)
        record(dict(kind="dispatch", ok=ok and refused, refused_true=refused,
                    names={f"{k[0]}-{k[1]}-{MODE_LABEL[k[2]]}": v for k, v in names.items()}))

    def stress(tab, label, n):
        for R, D in ((18, 83), (24, 110), (32, 138)):
            rng = random.Random(1009 * R)
            perm = rng.sample(range(NE), D)
            sel_l = [[perm[(r * TOPK + j) % D] for j in range(TOPK)] for r in range(R)]
            ins = [inputs(R, sel_l, None, True) for _ in range(2)]
            refs = [reference(tab, *i, R)[0] for i in ins]
            s = fresh(32, 5.0)
            outs = []
            set_env(B_ARM, True)
            try:
                for i in range(n):
                    x, sel, rw, sh, gate_w = ins[i % 2]
                    ext.exl3_moe_coop(x, sel, rw, -1, -1, H,
                                      tab["gate_trellis"], tab["gate_suh"], tab["gate_svh"],
                                      tab["up_trellis"], tab["up_suh"], tab["up_svh"],
                                      tab["down_trellis"], tab["down_suh"], tab["down_svh"],
                                      None, None, None, tab["K"], tab["K"], tab["Kd"], tab["mcg"], tab["mul1"], 0, 0.0, True,
                                      s["had_g"], s["had_u"], s["gu_g"], s["gu_u"], s["act"], s["d_out"], s["ctr"], s["out"],
                                      sh, gate_w)
                    outs.append(s["out"][:R].clone())
                torch.cuda.synchronize()
            finally:
                set_env(0, False)
            bad = sum(not bits_equal(refs[i % 2], o) for i, o in enumerate(outs))
            record(dict(kind="stress", label=label, K=tab["K"], rows=R, D=D, n=n, bad=bad, ok=bad == 0))
            print(f"[parity] stress {label} K{tab['K']} rows {R} D{D}: {n - bad}/{n} launches equal", flush=True)

    # ------------------------------------------------------------------ expert tables
    ckpt = Checkpoint(a.model, dev)
    tabs = []

    def make_tab(experts, label):
        tab = {}
        for p in ("gate", "up", "down"):
            for j, s in enumerate(("trellis", "suh", "svh")):
                tab[f"{p}_{s}"] = torch.tensor([e[p][j].data_ptr() for e in experts], dtype=torch.long, device=dev)
        tab["K"] = experts[0]["up"][0].shape[-1] // 16
        tab["Kd"] = experts[0]["down"][0].shape[-1] // 16
        tab["mcg"], tab["mul1"] = experts[0]["up"][3], experts[0]["up"][4]
        tab["keep"] = experts
        tab["label"] = label
        return tab

    for K, pfx in EXPERT_LAYERS.items():
        try:
            experts = [{p: ckpt.linear(f"{pfx}.experts.{e}.{p}_proj") for p in ("gate", "up", "down")} for e in range(NE)]
        except KeyError as e:
            print(f"[parity] real experts {pfx}: missing {e}, skipped", flush=True)
            continue
        tabs.append(make_tab(experts, f"real:{pfx}"))
    if not a.skip_synth:
        for K in (2, 3, 4):
            def syn(k, n):
                tr = torch.randint(-32768, 32767, (NE, k // 16, n // 16, 16 * K), dtype=torch.int16, device=dev)
                suh = (torch.randint(0, 2, (NE, k), device=dev) * 2 - 1).half()
                svh = ((torch.randint(0, 2, (NE, n), device=dev) * 2 - 1) * 0.01).half()
                return tr, suh, svh
            g, u, d = syn(H, I), syn(H, I), syn(I, H)
            experts = [{"gate": (g[0][e], g[1][e], g[2][e], False, True), "up": (u[0][e], u[1][e], u[2][e], False, True),
                        "down": (d[0][e], d[1][e], d[2][e], False, True)} for e in range(NE)]
            tabs.append(make_tab(experts, f"synth:K{K}"))
    assert tabs, "no expert tables"

    if a.bench:
        bench(torch, ext, tabs, coop, fresh, inputs, B_ARM)
        return 0

    # ------------------------------------------------------------------ 4b
    for tab in tabs:
        parity_4b(tab, tab["label"], a.seeds)
        unchanged_le16(tab, tab["label"])
    dispatch(tabs[0])
    for tab in tabs[:2]:
        stress(tab, tab["label"], a.stress)

    # ------------------------------------------------------------------ 4c: shared expert graphs to 32 rows
    shared = build_shared(torch, ext, ckpt, dev)
    for e in shared["entries"]:
        parity_4c(torch, ext, shared, e, has_dense, record, bits_equal)
    join_check(torch, ext, shared, tabs[0], coop, fresh, inputs, record, bits_equal, r3, B_ARM)

    n_ok = sum(r["ok"] for r in results)
    print(f"[parity] {n_ok}/{len(results)} checks equal; fails {fails}", flush=True)
    if a.json:
        a.json.write_text(json.dumps(dict(results=results, fails=fails, modes=[MODE_LABEL[m] for m in MODES], r3=r3,
                                          dense=has_dense),
                                     indent=1, default=str))
    print("PARITY PASS" if fails == 0 else "PARITY FAIL", flush=True)
    return 0 if fails == 0 else 1


def build_shared(torch, ext, ckpt, dev, maxb=32):
    """BC_GatedMLP per shared expert as GatedMLP.load builds it (modules/mlp.py: fused gate/up pointer table ->
    exl3_mgemm, silu, down BC_LinearEXL3), statics as bsz1_pa_args at MAX_BSZN = maxb. After moefast r3's
    moefast_r3_side.SharedSet."""
    entries = []
    I_sh = None
    for pfx in SHARED_LAYERS:
        try:
            g = ckpt.linear(f"{pfx}.shared_expert.gate_proj")
            u = ckpt.linear(f"{pfx}.shared_expert.up_proj")
            dn = ckpt.linear(f"{pfx}.shared_expert.down_proj")
        except KeyError as e:
            print(f"[parity] shared expert {pfx}: missing {e}, skipped", flush=True)
            continue
        K = g[0].shape[-1] // 16
        I_sh = g[0].shape[1] * 16
        entries.append(dict(pfx=pfx, K=K, g=g, u=u, dn=dn))
    assert entries, "no shared expert in the checkpoint"

    def statics(mb):
        return (torch.empty((2, mb, H), dtype=torch.half, device=dev), torch.empty((2, mb, I_sh), dtype=torch.half, device=dev),
                torch.empty((1, mb, I_sh), dtype=torch.half, device=dev), torch.empty((1, mb, I_sh), dtype=torch.half, device=dev))

    lin_xh = torch.empty((1, I_sh), dtype=torch.half, device=dev)
    st32, st16 = statics(maxb), statics(16)
    for e in entries:
        g, u, dn = e["g"], e["u"], e["dn"]
        e["ptrs"] = tuple(torch.tensor([g[j].data_ptr(), u[j].data_ptr()], dtype=torch.long, device=dev) for j in range(3))
        e["down_bc"] = ext.BC_LinearEXL3(dn[0], dn[1], dn[2], dn[0].shape[-1] // 16, None, dn[3], dn[4], lin_xh)

        def mk(st, e=e):
            return ext.BC_GatedMLP(*st, e["ptrs"][0], e["ptrs"][1], e["ptrs"][2], e["K"], e["g"][3], e["g"][4],
                                   True, False, False, None, None, e["down_bc"], 0.0)
        e["mk32"] = lambda e=e, mk=mk: mk(st32)
        e["mk16"] = lambda e=e, mk=mk: mk(st16)
    print(f"[parity] shared experts: {[e['pfx'] + ' K' + str(e['K']) for e in entries]}, I={I_sh}", flush=True)
    return dict(entries=entries, I=I_sh, lin_xh=lin_xh, st32=st32, st16=st16)


def parity_4c(torch, ext, shared, e, has_dense, record, bits_equal):
    dev = shared["lin_xh"].device
    dense_modes = [None]
    if has_dense:
        dense_modes = [0, 1]
    outs_by_dense = {}
    for dm in dense_modes:
        if dm is not None:
            ext.dense_v2_set_mode(-1, dm)
        n0 = 0
        bad = []
        for R in range(17, 33):
            xs = [(torch.randn((1, R, H), device=dev) * 0.5).half() for _ in range(3)]
            # eager references from fresh instances (first call per row count is eager)
            ref = []
            for x in xs:
                bc = e["mk32"]()
                d = torch.full((1, R, H), -3.0, dtype=torch.float, device=dev)
                bc.run_bszN(x, d)
                torch.cuda.synchronize()
                ref.append(d.clone())
            bc = e["mk32"]()
            got = []
            for i, x in enumerate([xs[0], xs[1], xs[2], xs[1]]):     # eager, capture + replay, replay, replay
                d = torch.full((1, R, H), 7.0 + i, dtype=torch.float, device=dev)
                bc.run_bszN(x, d)
                torch.cuda.synchronize()
                got.append(d.clone())
            ok = all(bits_equal(g_, ref[j]) for g_, j in zip(got, (0, 1, 2, 1)))
            n0 += 1
            if not ok:
                bad.append(R)
            outs_by_dense.setdefault(R, {})[dm] = (xs[0], ref[0])
        record(dict(kind="4c-graph", pfx=e["pfx"], dense_rows32=dm, ok=not bad, bad_rows=bad))
        print(f"[parity] 4c {e['pfx']} dense_rows32={dm}: eager == graph (capture + 3 replays, fresh inputs) at "
              f"{n0 - len(bad)}/{n0} row counts 17-32", flush=True)
    if has_dense:
        ext.dense_v2_set_mode(-1, 1)
        bad = []
        for R, m in outs_by_dense.items():
            x0, r0 = m[0]
            bc = e["mk32"]()
            d = torch.empty((1, R, H), dtype=torch.float, device=dev)
            bc.run_bszN(x0, d)
            torch.cuda.synchronize()
            if not bits_equal(d, r0):
                bad.append(R)
        ext.dense_v2_set_mode(-1, 0)
        record(dict(kind="4c-dense-on-vs-off", pfx=e["pfx"], ok=not bad, bad_rows=bad))
        print(f"[parity] 4c {e['pfx']}: EXL3_DENSE_ROWS32 on == off at {16 - len(bad)}/16 row counts", flush=True)
    # the served statics refuse 17 rows
    refused = False
    try:
        bc = e["mk16"]()
        bc.run_bszN((torch.randn((1, 17, H), device=dev) * 0.5).half(), torch.empty((1, 17, H), dtype=torch.float, device=dev))
    except RuntimeError as ex:
        refused = "out of supported range" in str(ex)
    record(dict(kind="4c-refuse-17-on-16-statics", pfx=e["pfx"], ok=refused))
    # <= 16 rows: 32-row statics == 16-row statics
    bad = []
    for R in range(1, 17):
        x = (torch.randn((1, R, H), device=dev) * 0.5).half()
        d16 = torch.empty((1, R, H), dtype=torch.float, device=dev)
        d32 = torch.empty((1, R, H), dtype=torch.float, device=dev)
        e["mk16"]().run_bszN(x, d16)
        e["mk32"]().run_bszN(x, d32)
        torch.cuda.synchronize()
        if not bits_equal(d16, d32):
            bad.append(R)
    record(dict(kind="4c-le16-unchanged", pfx=e["pfx"], ok=not bad, bad_rows=bad))
    print(f"[parity] 4c {e['pfx']}: refuse 17 rows on 16-row statics {refused}; <=16 rows 32- == 16-row statics at "
          f"{16 - len(bad)}/16", flush=True)


def join_check(torch, ext, shared, tab, coop, fresh, inputs, record, bits_equal, r3, served_mode):
    """Side-stream shared expert at R rows joined into the coop launch at R rows == serial, 100 alternating launches."""
    dev = shared["lin_xh"].device
    e = shared["entries"][0]
    side = torch.cuda.Stream(device=dev)
    ev_in, ev_done = torch.cuda.Event(), torch.cuda.Event()
    ev_in.record(); ev_done.record(); torch.cuda.synchronize()
    gate_w = (torch.randn((H,), device=dev) * 0.02).half().contiguous()
    for R in (18, 24, 32):
        bc = e["mk32"]()
        rng = random.Random(R)
        D = round(4.6 * R)
        perm = rng.sample(range(NE), D)
        sel_l = [[perm[(r * TOPK + j) % D] for j in range(TOPK)] for r in range(R)]
        ins = [inputs(R, sel_l, None, False)[:3] for _ in range(2)]
        sh_buf = torch.zeros((1, 32, H), dtype=torch.float, device=dev)
        # serial references: shared on the main stream, then coop with its output (fp32 staging as BlockSparseMLP)
        refs = []
        for x, sel, rw in ins:
            bc.run_bszN(x.unsqueeze(0), sh_buf[:, :R])
            refs.append(coop(tab, x, sel, rw, sh_buf[0, :R].contiguous().clone(), gate_w, fresh(32, 1.0),
                             served_mode, True)["out"][:R].clone())
        s = fresh(32, 2.0)
        bad = 0
        v3_, map_ = served_mode if isinstance(served_mode, tuple) else (str(served_mode), "")
        if v3_ and v3_ != "0":
            os.environ["EXL3_MOE_COOP_V3"] = v3_
        if map_:
            os.environ["EXL3_MOE_COOP_V3_MAP"] = map_
        os.environ[FLAG] = "1"
        try:
            outs = []
            for i in range(100):
                x, sel, rw = ins[i % 2]
                ev_in.record(torch.cuda.current_stream())
                side.wait_event(ev_in)
                with torch.cuda.stream(side):
                    bc.run_bszN(x.unsqueeze(0), sh_buf[:, :R])
                ev_done.record(side)
                args = (x, sel, rw, -1, -1, H, tab["gate_trellis"], tab["gate_suh"], tab["gate_svh"],
                        tab["up_trellis"], tab["up_suh"], tab["up_svh"], tab["down_trellis"], tab["down_suh"],
                        tab["down_svh"], None, None, None, tab["K"], tab["K"], tab["Kd"], tab["mcg"], tab["mul1"],
                        0, 0.0, True, s["had_g"], s["had_u"], s["gu_g"], s["gu_u"], s["act"], s["d_out"], s["ctr"],
                        s["out"], sh_buf[0, :R], gate_w)
                if r3:
                    ext.exl3_moe_coop_ev(*args, ev_done.cuda_event)            # served join: wait right before B
                else:
                    torch.cuda.current_stream().wait_event(ev_done)          # coarser: wait before the launch
                    ext.exl3_moe_coop(*args)
                outs.append(s["out"][:R].clone())
            torch.cuda.synchronize()
        finally:
            os.environ.pop("EXL3_MOE_COOP_V3", None)
            os.environ.pop("EXL3_MOE_COOP_V3_MAP", None)
            os.environ.pop(FLAG, None)
        bad = sum(not bits_equal(refs[i % 2], o) for i, o in enumerate(outs))
        record(dict(kind="join", rows=R, served_join=r3, n=100, bad=bad, ok=bad == 0))
        print(f"[parity] join rows {R} ({'wait before B, exl3_moe_coop_ev' if r3 else 'wait before the launch'}): "
              f"{100 - bad}/100 equal to serial", flush=True)


def bench(torch, ext, tabs, coop, fresh, inputs, served_mode):
    """Diagnostic: us per call. 16 rows (D 95), split = two 16-row calls, single = one launch. D per R708 files."""
    D_OF = {16: 95, 18: 98, 21: 111, 24: 119, 28: 129, 32: 138}
    print("[bench] us per call (median of 50 after 10 warm-up; routing sets rotate over the 512 experts)")
    print("[bench] expert   R    D   16-row   split   single   single/split")
    for tab in tabs[:2]:
        t16 = None
        for R, D in D_OF.items():
            rng = random.Random(R)
            sets = []
            for c in range(8):
                perm = rng.sample(range(NE), D)
                sets.append([[perm[(r * TOPK + j) % D] for j in range(TOPK)] for r in range(R)])
            ins = [inputs(R, s_, None, True) for s_ in sets]
            s32, s16a, s16b = fresh(32, 0.0), fresh(16, 0.0), fresh(16, 0.0)

            def timeit(fn):
                ts = []
                for i in range(60):
                    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    st.record(); fn(i); en.record(); torch.cuda.synchronize()
                    if i >= 10:
                        ts.append(st.elapsed_time(en) * 1000)
                return statistics.median(ts)

            def single(i):
                x, sel, rw, sh, gw = ins[i % 8]
                coop(tab, x, sel, rw, sh, gw, s32, served_mode, True, sync=False)

            def split(i):
                x, sel, rw, sh, gw = ins[i % 8]
                coop(tab, x[:16], sel[:16], rw[:16], sh[:16], gw, s16a, served_mode, False, sync=False)
                coop(tab, x[16:], sel[16:], rw[16:], sh[16:], gw, s16b, served_mode, False, sync=False)
            if R == 16:
                t16 = timeit(single)
                print(f"[bench] {tab['label'][-24:]:24s} 16 {D:4d} {t16:8.1f}")
                continue
            ts, tp = timeit(single), timeit(split)
            print(f"[bench] {tab['label'][-24:]:24s} {R:2d} {D:4d} {t16:8.1f} {tp:7.1f} {ts:8.1f}   {ts / tp:.2f}")


if __name__ == "__main__":
    sys.exit(main())
