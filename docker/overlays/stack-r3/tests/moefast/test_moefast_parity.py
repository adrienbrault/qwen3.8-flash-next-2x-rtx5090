#!/usr/bin/env python3
"""moefast r3 GPU parity: EXL3_MOE_COOP_V3 = 1, 2 (r1) and 3 (r2) against the served V2 path, bitwise, plus the
r3 overlap levers (early fork, high-priority side stream, head kernel) through the served side-stream join.

r3 additions (on top of r2's suite below, which is unchanged except for the row list):
  rows 1-16 (r2 skipped 6, 7, 9-11, 14, 15; the served verify rows include 14 = c7d1 and 15 = c5d2);
  SIDE: for real K2/K3 at rows 2, 4, 8, 12, 14, 15, 16, the full served layer call (router kernels, the real
    shared expert BC_GatedMLP graph, coop launch; moefast_r3_side.make_call) in every arm -- serial, late fork
    (served), early fork (r3), early fork on a high-priority side stream, head kernel (EXL3_MOE_COOP_V3_HEAD=1 and
    2000) -- in modes 0 / 2 / 3, all bitwise against the serial V2 call; the shared expert output itself too;
  JOIN STRESS: 200 back-to-back early-fork + high-priority calls per mode (2, 3) at rows 4 and 16, inputs
    alternating between two x so a stage B that did not wait for the shared expert would read the other input's
    shared output; every launch compared;
  DISPATCH: the head kernel launches in mode 3 with EXL3_MOE_COOP_V3_HEAD and in no other case.
The early fork inside BlockSparseMLP.forward (python guard) is covered by test_moefast_r3_cpu.py and P1 hashes.

Runs in the tabbyapi:stack-moefast-r3 image on one card. The reference is the served launch (EXL3_MOE_COOP_V2=1,
EXL3_MOE_COOP_V3 unset); the flag is read per launch by the C++ launcher, so all arms run in one process on
the same inputs and the same weights. Entry point: ext.exl3_moe_coop, the launcher BC_BlockSparseMLP::run_bszN
uses (quant/exl3_moe_coop.cu), here WITH the shared-expert term (sh_out + sh_gate_w) as served, and without.

Compared, storage-bit equal (torch.equal on int32 views): the output rows (bsz, 2560) and, for every active
slot, the down-input activation act_out (slot, 640) and the down GEMV rows d_out (slot, 2560). Scratch and
output start from different sentinels in each arm, so a skipped write cannot pass.

Cases (real K=2 layer 20, K=3 layer 3 experts; synthetic K=2/3/4):
  rows 2, 3, 4, 5, 8, 12 (narrow down stage: mode 2 merges it), 13, 16 (wide down stage);
  routings: served-like (D distinct experts, as d0_microbench), all distinct, all rows the same top-10 (runs of
  8 rows at 16 rows), random picks from a 24-expert pool, inactive slots (rw = 0) incl. one empty token row,
  every other token row empty (rows > 2);
  with and without the shared-expert term. rows 1 is included as a control (V3 must not engage: same path).
Plus: V3 run twice is deterministic; the dispatch check (torch.profiler kernel names) proves V3 ran, that
mode 3 launches the r2 kernels without the rotation kernel, and that EXL3_MOE_COOP_V3_MAP picks per row count.
Mode 3 is also checked with EXL3_MOE_COOP_V3_L2EF=0 (weight copies without the evict_first policy). The
same-top10 pattern at 12-16 rows makes 8-row runs, which r2's kernel A stages in two passes (7 or 6 rows fit).
Stress (--stress N, default 200): V3 replaces V2's block-wide __threadfence() pairs around the completion
counters with one atom.add.acq_rel.gpu per counting thread. A missing ordering shows up as an intermittent
wrong partial, not as a deterministic mismatch, so served-like cells (real K2/K3, rows 4/12/16) are
launched N times per mode back to back on one scratch set and every launch is compared to the reference.

Exit 0 and "PARITY PASS" only if every comparison is equal. Split-k: EXL3_MOE_COOP_KSPLIT is cached once per
process by the launcher, so run the file a second time with EXL3_MOE_COOP_KSPLIT=2 exported to cover it.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE / "d0", Path("/probe/d0")):
    if (p / "d0_microbench.py").is_file():
        sys.path.insert(0, str(p))
        break
import d0_microbench as d0  # noqa: E402  (Checkpoint, synth_linear, constants)
sys.path.insert(0, str(HERE))
import moefast_r3_side as side_mod  # noqa: E402

TOPK, NE, H, I, MAXB = d0.TOPK, d0.N_EXPERTS, d0.HIDDEN, d0.MOE_I, d0.MAX_BSZN


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--seeds", type=int, default=3, help="random routings per (rows, pattern)")
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--stress", type=int, default=200, help="back-to-back V3 launches per stress cell and mode")
    ap.add_argument("--no-side", action="store_true", help="skip the r3 side-stream section")
    a = ap.parse_args()

    assert os.environ.get("EXL3_MOE_COOP_V2") == "1", "run with the served env (EXL3_MOE_COOP_V2=1)"
    assert not os.environ.get("EXL3_MOE_COOP_DBG"), "EXL3_MOE_COOP_DBG bypasses V2/V3"
    for k_ in ("EXL3_MOE_COOP_V3", "EXL3_MOE_COOP_V3_MAP", "EXL3_MOE_COOP_V3_L2EF", "EXL3_MOE_COOP_V3_HEAD"):
        os.environ.pop(k_, None)
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    assert getattr(ext, "moe_coop_v3_revision", 0) >= 3, "needs the moefast r3 extension"
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    cap = torch.cuda.get_device_capability(dev)
    assert cap == (12, 0), f"V3 is sm_120 only, got {cap}"
    ks_env = os.environ.get("EXL3_MOE_COOP_KSPLIT", "")
    print(f"[parity] device {torch.cuda.get_device_name(dev)}; KSPLIT={ks_env or 'default'}", flush=True)

    S = MAXB * TOPK
    scr_shapes = dict(had_g=(S, H, torch.half), had_u=(S, H, torch.half), gu_g=(S, I, torch.half),
                      gu_u=(S, I, torch.half), act=(S, I, torch.half), d_out=(S, H, torch.float),
                      out=(MAXB, H, torch.float))
    ctr_len = S * (I // 128) + MAXB * (H // 128) + 2 * S + 3

    def fresh_scratch(sentinel):
        s = {k: torch.full((r, c), sentinel, dtype=dt, device=dev) for k, (r, c, dt) in scr_shapes.items()}
        s["ctr"] = torch.zeros((ctr_len,), dtype=torch.int, device=dev)
        return s

    def run(tab, K, Kd, x, sel, rw, sh, gate_w, mode, sentinel):
        os.environ.pop("EXL3_MOE_COOP_V3_L2EF", None)
        if mode == 0:
            os.environ.pop("EXL3_MOE_COOP_V3", None)
        elif mode == "3ef0":
            os.environ["EXL3_MOE_COOP_V3"] = "3"
            os.environ["EXL3_MOE_COOP_V3_L2EF"] = "0"
        else:
            os.environ["EXL3_MOE_COOP_V3"] = str(mode)
        s = fresh_scratch(sentinel)
        ext.exl3_moe_coop(x, sel, rw, -1, -1, H,
                          tab["gate_trellis"], tab["gate_suh"], tab["gate_svh"],
                          tab["up_trellis"], tab["up_suh"], tab["up_svh"],
                          tab["down_trellis"], tab["down_suh"], tab["down_svh"],
                          None, None, None, K, K, Kd, False, True, 0, 0.0, True,
                          s["had_g"], s["had_u"], s["gu_g"], s["gu_u"], s["act"], s["d_out"],
                          s["ctr"], s["out"], sh, gate_w)
        torch.cuda.synchronize()
        os.environ.pop("EXL3_MOE_COOP_V3", None)
        os.environ.pop("EXL3_MOE_COOP_V3_L2EF", None)
        return s

    def bits_equal(u, v):
        if u.dtype == torch.half:
            return torch.equal(u.view(torch.int16), v.view(torch.int16))
        return torch.equal(u.view(torch.int32), v.view(torch.int32))

    def routings(rows, rng):
        pats = []
        # served-like: D distinct experts, row r takes pool[(10 r + j) % D] (d0_microbench.Cells.moe)
        for D in sorted({TOPK, min(TOPK * rows, max(TOPK, round(7 * rows))), TOPK * rows}):
            perm = rng.sample(range(NE), D)
            pats.append((f"served-D{D}", [[perm[(r * TOPK + j) % D] for j in range(TOPK)] for r in range(rows)], None))
        same = rng.sample(range(NE), TOPK)
        pats.append(("same-top10", [list(same) for _ in range(rows)], None))
        pool = rng.sample(range(NE), 24)
        pats.append(("pool24", [rng.sample(pool, TOPK) for _ in range(rows)], None))
        # inactive slots: rw = 0 on ~30 % of slots; the last token row fully inactive (rows > 1)
        sel = [rng.sample(range(NE), TOPK) for _ in range(rows)]
        mask = [[rng.random() < 0.3 for _ in range(TOPK)] for _ in range(rows)]
        if rows > 1:
            mask[-1] = [True] * TOPK
        pats.append(("inactive", sel, mask))
        # several empty token rows (every other row fully inactive): r2's grid-strided empty-row loop indexes a
        # per-block list, so every block must build it in the same order
        if rows > 2:
            sel = [rng.sample(range(NE), TOPK) for _ in range(rows)]
            mask = [[r % 2 == 1] * TOPK for r in range(rows)]
            pats.append(("empty-rows", sel, mask))
        return pats

    results = []
    fails = 0

    def case(tab, K, Kd, label, rows, name, sel_l, mask, with_sh, rng):
        nonlocal fails
        sel = torch.tensor(sel_l, dtype=torch.long, device=dev)
        w = torch.rand((rows, TOPK), device=dev) + 0.1
        rw = (w / w.sum(-1, keepdim=True)).half()
        if mask is not None:
            rw[torch.tensor(mask, device=dev)] = 0.0
        rw = rw.contiguous()
        x = (torch.randn((rows, H), device=dev) * 0.5).half().contiguous()
        sh = (torch.randn((rows, H), device=dev) * 0.1).float().contiguous() if with_sh else None
        gate_w = (torch.randn((H,), device=dev) * 0.02).half().contiguous() if with_sh else None
        ref = run(tab, K, Kd, x, sel, rw, sh, gate_w, 0, -7.0)
        active = (rw.flatten() != 0).nonzero().flatten()
        slots = rows * TOPK
        for mode in (1, 2, 3, "3ef0"):
            for rep in range(2 if mode in (2, 3) else 1):
                got = run(tab, K, Kd, x, sel, rw, sh, gate_w, mode, 3.0 + rep)
                checks = {
                    "out": bits_equal(ref["out"][:rows], got["out"][:rows]),
                    "act": bits_equal(ref["act"][:slots][active], got["act"][:slots][active]),
                    "d_out": bits_equal(ref["d_out"][:slots][active], got["d_out"][:slots][active]),
                }
                ok = all(checks.values())
                rec = dict(label=label, K=K, rows=rows, pattern=name, shared=with_sh, mode=mode, rep=rep,
                           ok=ok, **{f"eq_{k}": v for k, v in checks.items()})
                if not ok:
                    fails += 1
                    d = (ref["out"][:rows] - got["out"][:rows]).abs().max().item()
                    rec["max_abs_out"] = d
                    print(f"[parity] FAIL {label} K{K} rows {rows} {name} shared={with_sh} mode {mode} rep {rep}: "
                          f"{checks} max|dout| {d:.3e}", flush=True)
                results.append(rec)

    def suite(tab, K, Kd, label, seeds):
        n0 = len(results)
        for rows in range(1, MAXB + 1):
            for seed in range(seeds):
                rng = random.Random(7919 * rows + 104729 * seed + K)
                for name, sel_l, mask in routings(rows, rng):
                    case(tab, K, Kd, label, rows, name, sel_l, mask, (seed + rows) % 2 == 0, rng)
        n = len(results) - n0
        bad = sum(not r["ok"] for r in results[n0:])
        print(f"[parity] {label} K{K}: {n - bad}/{n} equal", flush=True)

    # ---------------------------------------------------------------- memory-ordering stress
    def stress(tab, K, Kd, label, n):
        nonlocal fails
        for rows, D in ((4, 28), (12, 60), (16, 77)):
            rng = random.Random(31 * rows + K)
            perm = rng.sample(range(NE), D)
            sel = torch.tensor([[perm[(r * TOPK + j) % D] for j in range(TOPK)] for r in range(rows)],
                               dtype=torch.long, device=dev)
            w = torch.rand((rows, TOPK), device=dev) + 0.1
            rw = (w / w.sum(-1, keepdim=True)).half().contiguous()
            x = (torch.randn((rows, H), device=dev) * 0.5).half().contiguous()
            sh = (torch.randn((rows, H), device=dev) * 0.1).float().contiguous()
            gate_w = (torch.randn((H,), device=dev) * 0.02).half().contiguous()
            ref = run(tab, K, Kd, x, sel, rw, sh, gate_w, 0, -7.0)["out"][:rows].clone()
            for mode in (1, 2, 3):
                os.environ["EXL3_MOE_COOP_V3"] = str(mode)
                s = fresh_scratch(5.0)
                outs = []
                for i in range(n):          # counters: the kernels reset them, as in serving
                    ext.exl3_moe_coop(x, sel, rw, -1, -1, H,
                                      tab["gate_trellis"], tab["gate_suh"], tab["gate_svh"],
                                      tab["up_trellis"], tab["up_suh"], tab["up_svh"],
                                      tab["down_trellis"], tab["down_suh"], tab["down_svh"],
                                      None, None, None, K, K, Kd, False, True, 0, 0.0, True,
                                      s["had_g"], s["had_u"], s["gu_g"], s["gu_u"], s["act"], s["d_out"],
                                      s["ctr"], s["out"], sh, gate_w)
                    outs.append(s["out"][:rows].clone())
                torch.cuda.synchronize()
                os.environ.pop("EXL3_MOE_COOP_V3", None)
                bad = sum(not bits_equal(ref, o) for o in outs)
                results.append(dict(label=label, K=K, rows=rows, pattern=f"stress-D{D}", shared=True, mode=mode,
                                    rep=n, ok=bad == 0, stress_bad=bad))
                print(f"[parity] stress {label} K{K} rows {rows} D{D} mode {mode}: {n - bad}/{n} launches equal",
                      flush=True)
                if bad:
                    fails += 1

    # ---------------------------------------------------------------- dispatch check: V3 really runs
    def dispatch_check(tab, K, Kd):
        from torch.profiler import profile, ProfilerActivity
        rows = 4
        sel = torch.tensor([[(r * TOPK + j) % 28 for j in range(TOPK)] for r in range(rows)], dtype=torch.long, device=dev)
        rw = torch.full((rows, TOPK), 0.1, dtype=torch.half, device=dev)
        x = (torch.randn((rows, H), device=dev) * 0.5).half()
        names = {}
        for mode in (0, 1, 2, 3):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                run(tab, K, Kd, x, sel, rw, None, None, mode, 0.0)
            ks = sorted({e.name for e in prof.events() if "moe_coop" in e.name})
            names[mode] = ks
            print(f"[parity] dispatch K{K} rows 4 mode {mode}: {ks}", flush=True)
        ok = (all("v3_" not in k and "r2_" not in k for k in names[0]) and any("v3_a_kernel" in k for k in names[1])
              and any("v3_b_kernel" in k for k in names[1]) and any("v3_b_kernel" in k and "true>" in k.split(",")[-1] for k in names[2])
              and any("r2_a_kernel" in k for k in names[3]) and any("r2_b_kernel" in k for k in names[3])
              and all("rot_kernel" not in k for k in names[3]))
        # EXL3_MOE_COOP_V3_MAP: V2 up to 4 rows, r2 above (the flag itself unset)
        os.environ["EXL3_MOE_COOP_V3_MAP"] = "2-4:0,5-16:3"
        for r_ in (4, 8):
            sel_ = torch.tensor([[(r * TOPK + j) % 40 for j in range(TOPK)] for r in range(r_)], dtype=torch.long, device=dev)
            rw_ = torch.full((r_, TOPK), 0.1, dtype=torch.half, device=dev)
            x_ = (torch.randn((r_, H), device=dev) * 0.5).half()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                run(tab, K, Kd, x_, sel_, rw_, None, None, 0, 0.0)
            ks = sorted({e.name for e in prof.events() if "moe_coop" in e.name})
            names[f"map_r{r_}"] = ks
            print(f"[parity] dispatch K{K} rows {r_} MAP 2-4:0,5-16:3: {ks}", flush=True)
        del os.environ["EXL3_MOE_COOP_V3_MAP"]
        ok = ok and all("r2_" not in k and "v3_" not in k for k in names["map_r4"]) \
            and any("r2_a_kernel" in k for k in names["map_r8"])
        # r3 head kernel: mode 3 + EXL3_MOE_COOP_V3_HEAD only
        ok = ok and all("head_kernel" not in k for m_ in (0, 1, 2, 3) for k in names[m_])
        for m_, head in ((3, "1"), (3, "2000"), (2, "1"), (0, "1")):
            os.environ["EXL3_MOE_COOP_V3_HEAD"] = head
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                run(tab, K, Kd, x, sel, rw, None, None, m_, 0.0)
            del os.environ["EXL3_MOE_COOP_V3_HEAD"]
            ks = sorted({e.name for e in prof.events() if "moe_coop" in e.name})
            names[f"head{head}_m{m_}"] = ks
            has = any("head_kernel" in k for k in ks)
            print(f"[parity] dispatch K{K} rows 4 mode {m_} HEAD={head}: {ks}", flush=True)
            ok = ok and has == (m_ == 3)
        return ok, names

    tabs = []
    ckpt = d0.Checkpoint(a.model, dev)
    for K, pfx in ((2, d0.EXPERT_LAYERS[2]), (3, d0.EXPERT_LAYERS[3])):
        experts = [{p: ckpt.linear(f"{pfx}.experts.{e}.{p}_proj") for p in ("gate", "up", "down")} for e in range(NE)]
        tabs.append((f"real:{pfx}", experts))
    if not a.skip_synth:
        for K in (2, 3, 4):
            g = d0.synth_linear(torch, H, I, K, dev, copies=NE)
            u = d0.synth_linear(torch, H, I, K, dev, copies=NE)
            dd = d0.synth_linear(torch, I, H, K, dev, copies=NE)
            experts = [{"gate": (g[0][e], g[1][e], g[2][e]), "up": (u[0][e], u[1][e], u[2][e]),
                        "down": (dd[0][e], dd[1][e], dd[2][e])} for e in range(NE)]
            tabs.append((f"synth:K{K}", experts))

    dispatch = {}
    for label, experts in tabs:
        tab = {}
        for p in ("gate", "up", "down"):
            for j, s in enumerate(("trellis", "suh", "svh")):
                tab[f"{p}_{s}"] = torch.tensor([e[p][j].data_ptr() for e in experts], dtype=torch.long, device=dev)
        K = experts[0]["up"][0].shape[-1] // 16
        Kd = experts[0]["down"][0].shape[-1] // 16
        if label.startswith("real"):
            ok, names = dispatch_check(tab, K, Kd)
            dispatch[label] = dict(ok=ok, names={str(k): v for k, v in names.items()})
            if not ok:
                fails += 1
                print(f"[parity] FAIL dispatch {label}: V3 kernels not selected as expected", flush=True)
        suite(tab, K, Kd, label, a.seeds)
        if label.startswith("real") and a.stress:
            stress(tab, K, Kd, label, a.stress)

    # ---------------------------------------------------------------- r3: served side-stream layer call
    if not a.no_side:
        side = side_mod.SharedSet(torch, ext, d0, ckpt, dev, log=lambda *s_: print("[parity]", *s_, flush=True))
        side_rows = (2, 4, 8, 12, 14, 15, 16)
        side.warm(side_rows)
        ovs = {None: side_mod.Overlap(torch, dev, None), "high": side_mod.Overlap(torch, dev, "high")}
        print(f"[parity] side streams: priority {ovs[None].priority} / {ovs['high'].priority}", flush=True)
        S_ARMS = [("serial-V2", 0, "serial", None, None), ("late-V2", 0, "late", None, None),
                  ("late-m2", 2, "late", None, None), ("late-m3", 3, "late", None, None),
                  ("early-m3", 3, "early", None, None), ("earlyP-m3", 3, "early", "high", None),
                  ("early-m2", 2, "early", None, None), ("earlyP-m2", 2, "early", "high", None),
                  ("lateP-m3", 3, "late", "high", None), ("head1-m3", 3, "late", None, "1"),
                  ("head2000-m3", 3, "late", None, "2000"), ("serial-m3", 3, "serial", None, None)]

        def side_call(tab, K, Kd, x, sel, rw, j, mode, fork, prio, head, sentinel):
            s_ = fresh_scratch(sentinel)
            side.sh_out.fill_(-3.0)
            if mode:
                os.environ["EXL3_MOE_COOP_V3"] = str(mode)
            if head:
                os.environ["EXL3_MOE_COOP_V3_HEAD"] = head
            side_mod.make_call(torch, ext, side, ovs[prio], tab, K, Kd, s_, H, x, sel, rw, j, fork)()
            torch.cuda.synchronize()
            os.environ.pop("EXL3_MOE_COOP_V3", None)
            os.environ.pop("EXL3_MOE_COOP_V3_HEAD", None)
            return s_["out"].clone(), side.sh_out.clone()

        for label, experts in tabs[:2]:
            tab = {}
            for p in ("gate", "up", "down"):
                for j_, s in enumerate(("trellis", "suh", "svh")):
                    tab[f"{p}_{s}"] = torch.tensor([e[p][j_].data_ptr() for e in experts], dtype=torch.long, device=dev)
            K = experts[0]["up"][0].shape[-1] // 16
            Kd = experts[0]["down"][0].shape[-1] // 16
            n_ok = n_all = 0
            for rows in side_rows:
                for seed in range(a.seeds):
                    rng = random.Random(7 * rows + 131 * seed + K)
                    D = min(TOPK * rows, max(TOPK, round(7 * rows)))
                    perm = rng.sample(range(NE), D)
                    sel = torch.tensor([[perm[(r * TOPK + j) % D] for j in range(TOPK)] for r in range(rows)],
                                       dtype=torch.long, device=dev)
                    w = torch.rand((rows, TOPK), device=dev) + 0.1
                    rw = (w / w.sum(-1, keepdim=True)).half().contiguous()
                    x = (torch.randn((rows, H), device=dev) * 0.5).half().contiguous()
                    j = rng.randrange(len(side.entries))
                    ref_out, ref_sh = side_call(tab, K, Kd, x, sel, rw, j, 0, "serial", None, None, -7.0)
                    for name, mode, fork, prio, head in S_ARMS[1:]:
                        got_out, got_sh = side_call(tab, K, Kd, x, sel, rw, j, mode, fork, prio, head, 3.0)
                        ok_ = bits_equal(ref_out[:rows], got_out[:rows]) and bits_equal(ref_sh[:, :rows], got_sh[:, :rows])
                        n_all += 1
                        n_ok += ok_
                        results.append(dict(label=label, K=K, rows=rows, pattern=f"side-{name}", shared=True,
                                            mode=mode, rep=seed, ok=ok_, layer=side.entries[j]["layer"]))
                        if not ok_:
                            fails += 1
                            print(f"[parity] FAIL side {label} K{K} rows {rows} seed {seed} {name}: max|dout| "
                                  f"{(ref_out[:rows] - got_out[:rows]).abs().max().item():.3e}", flush=True)
            print(f"[parity] side {label} K{K}: {n_ok}/{n_all} equal (arms {[x_[0] for x_ in S_ARMS[1:]]})", flush=True)

            # join stress: back-to-back early-fork calls, x alternating so a missed join reads the other x's sh
            if a.stress:
                for rows, D in ((4, 28), (16, 77)):
                    rng = random.Random(97 * rows + K)
                    perm = rng.sample(range(NE), D)
                    sel = torch.tensor([[perm[(r * TOPK + j) % D] for j in range(TOPK)] for r in range(rows)],
                                       dtype=torch.long, device=dev)
                    w = torch.rand((rows, TOPK), device=dev) + 0.1
                    rw = (w / w.sum(-1, keepdim=True)).half().contiguous()
                    xs = [(torch.randn((rows, H), device=dev) * 0.5).half().contiguous() for _ in range(2)]
                    refs = [side_call(tab, K, Kd, xs[i], sel, rw, 5, 0, "serial", None, None, -7.0)[0][:rows].clone()
                            for i in range(2)]
                    for mode in (2, 3):
                        s_ = fresh_scratch(5.0)
                        calls = [side_mod.make_call(torch, ext, side, ovs["high"], tab, K, Kd, s_, H, xs[i], sel, rw, 5,
                                                    "early") for i in range(2)]
                        os.environ["EXL3_MOE_COOP_V3"] = str(mode)
                        outs = []
                        for i in range(a.stress):
                            calls[i % 2]()
                            outs.append(s_["out"][:rows].clone())
                        torch.cuda.synchronize()
                        os.environ.pop("EXL3_MOE_COOP_V3", None)
                        bad = sum(not bits_equal(refs[i % 2], o) for i, o in enumerate(outs))
                        results.append(dict(label=label, K=K, rows=rows, pattern=f"join-stress-D{D}", shared=True,
                                            mode=mode, rep=a.stress, ok=bad == 0, stress_bad=bad))
                        print(f"[parity] join stress {label} K{K} rows {rows} mode {mode}: {a.stress - bad}/{a.stress} "
                              f"launches equal", flush=True)
                        if bad:
                            fails += 1

    total = len(results)
    summary = dict(total=total, failed=fails, ksplit=ks_env or "default", dispatch=dispatch)
    if a.json:
        a.json.write_text(json.dumps(dict(summary=summary, cases=results), indent=1))
    print(f"[parity] {total - sum(not r['ok'] for r in results)}/{total} comparisons equal; dispatch "
          f"{'ok' if all(d['ok'] for d in dispatch.values()) else 'FAIL'}", flush=True)
    print("PARITY PASS" if fails == 0 else f"PARITY FAIL ({fails})", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
