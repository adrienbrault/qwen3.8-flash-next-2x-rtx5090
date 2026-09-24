#!/usr/bin/env python3
"""moefast r3 P0: the routed MoE layer WITH the served side-stream shared expert and router, per arm.

r2's P0 passed the shared output precomputed into the down kernel: no side stream, no router, no join. It therefore
could not see the regression that decided R703b (nsys: the shared expert starts after r2's kernel A instead of
before it, B waits on it; A->B gap 1.9 -> 14.1 us at c1d3). This P0 runs each call as BC_BlockSparseMLP::run_bszN
does (moefast_r3_side.py): the real router kernels (ext.routing_std on the layer's router weight) on the main
stream, the real shared expert (BC_GatedMLP graph of the layer's shared_expert, all 48 layers rotated) on a side
stream between an input and a done event, and the coop launch that waits on the done event before stage B
(ext.exl3_moe_coop_ev). Routed experts: d0 cells (K=2 layer 20, K=3 layer 3), 64 routings walking a random
permutation of the 512 experts (DRAM, as R682).

Arms (EXL3_MOE_COOP_V3 and EXL3_MOE_COOP_V3_HEAD are read per launch, so all arms run in one process, interleaved):
  ref   V2, late fork (the served overlap before moefast)           m2    r1 mode 2, late fork (SERVED stack-r2)
  m3    r2, late fork (R703b's R2)                                   m3H   r2 + head kernel (EXL3_MOE_COOP_V3_HEAD=1)
  m3H2  r2 + 2 us head kernel (HEAD=2000; r1's rot window)           m3P   r2, late fork, high-priority side stream
  m3E   r2, early fork (EXL3_SHARED_EXPERT_EARLY)                    m3EP  r2, early fork, high priority
  m2E   r1 mode 2, early fork                                        m2EP  r1 mode 2, early fork, high priority
  ser   r2, shared expert serial on the main stream (no overlap)     m3N   r2, no shared expert at all (r2's P0 cell)
Per cell and arm: us per call (CUDA events around the whole call on the main stream, behind torch.cuda._sleep; the
median over --rounds rotated rounds of each round's median) and, from a torch.profiler pass, per-call medians of
the A->B gap, shared start - A start, shared end - A end, shared span and the MoE span (first head/rot/A kernel to
B end).

Diagnostics printed and written to p0-summary.json:
  INSTRUMENT  at r4/D28 (both K): gap(m3) - gap(m2) >= 5 us reproduces R703b (else "NOT REPRODUCED", loud)
  48-layer MoE per arm and shape (25 K2 + 23 K3 layers)
  PICK        pre-registered rule for the gate's R3 arm (see pick()): lever E or E+P, kernel mode per row class
              -> "R3_ENV=<env>" (the gate reads this line)
--l2probe: r2's L2-policy question (fixed: torch.sum with dim). --ncu: one call per cell/arm under ncu (no side).
--no-side: r2's P0 cells (shared output precomputed, no router, no side stream).
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE / "d0", Path("/probe/d0")):
    if (p / "d0_microbench.py").is_file():
        sys.path.insert(0, str(p))
        break
sys.path.insert(0, str(HERE))
import d0_microbench as d0  # noqa: E402
import moefast_r3_side as side_mod  # noqa: E402

# name: (EXL3_MOE_COOP_V3, fork, side priority, EXL3_MOE_COOP_V3_HEAD)
ARMS = {
    "ref": (0, "late", None, 0), "m2": (2, "late", None, 0), "m3": (3, "late", None, 0),
    "m3H": (3, "late", None, 1), "m3H2": (3, "late", None, 2000), "m3P": (3, "late", "high", 0),
    "m3E": (3, "early", None, 0), "m3EP": (3, "early", "high", 0),
    "m2E": (2, "early", None, 0), "m2EP": (2, "early", "high", 0),
    "ser": (3, "serial", None, 0), "m3N": (3, "none", None, 0),
}
DEFAULT_ARMS = ["ref", "m2", "m3", "m3H", "m3H2", "m3P", "m3E", "m3EP", "m2E", "m2EP", "ser"]
SERVED = [(1, 10), (4, 28), (16, 77)]                     # rows, distinct experts (R682 Q2 / R619)
EXTRA = [(2, 20), (3, 28), (4, 10), (4, 20), (4, 40), (8, 40), (12, 60), (16, 40), (16, 160)]
N_LAYERS = {2: 25, 3: 23}                                  # K=2 layers 12-36, K=3 layers 0-11 and 37-47


def set_arm(mode, head):
    for k, v in (("EXL3_MOE_COOP_V3", mode), ("EXL3_MOE_COOP_V3_HEAD", head)):
        if v:
            os.environ[k] = str(v)
        else:
            os.environ.pop(k, None)


def routing_sets(torch, dev, rows, D, nsets=64, seed=0):
    import random
    rng = random.Random(1000 * rows + D + seed)
    perm = list(range(d0.N_EXPERTS))
    rng.shuffle(perm)
    sets = []
    for c in range(nsets):
        pool = [perm[(c * D + i) % d0.N_EXPERTS] for i in range(D)]
        rng.shuffle(pool)
        sets.append(torch.tensor([[pool[(r * d0.TOPK + j) % D] for j in range(d0.TOPK)] for r in range(rows)],
                                 dtype=torch.long, device=dev))
    return sets


def legacy_fns(torch, ext, dev, scr, tab, K, Kd, rows, D, shared=True):
    """r2's P0 call: ext.exl3_moe_coop with the shared output precomputed (no side stream)."""
    sets = routing_sets(torch, dev, rows, D)
    w = torch.rand((rows, d0.TOPK), device=dev) + 0.1
    rw = (w / w.sum(-1, keepdim=True)).half().contiguous()
    x = (torch.randn((rows, d0.HIDDEN), device=dev) * 0.5).half().contiguous()
    sh = (torch.randn((rows, d0.HIDDEN), device=dev) * 0.1).float().contiguous() if shared else None
    gate_w = (torch.randn((d0.HIDDEN,), device=dev) * 0.02).half().contiguous() if shared else None
    args = side_mod.coop_args(tab, K, Kd, scr, d0.HIDDEN)

    def mk(sel):
        def f():
            ext.exl3_moe_coop(x, sel, rw, -1, -1, d0.HIDDEN, *args, sh, gate_w)
        return f
    return [mk(s) for s in sets]


def l2probe(torch, fns, arms, victim_mb, reps=9):
    """Victim re-read time (us) after one MoE call, per arm; plus hot and cold references."""
    dev = torch.device("cuda:0")
    victim = torch.ones(victim_mb * (1 << 20) // 2, dtype=torch.half, device=dev)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    sink = torch.empty((), dtype=torch.float, device=dev)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def read():
        torch.sum(victim, dim=0, dtype=torch.float, out=sink)      # R703: without dim this overload does not exist

    def read_timed():
        e0.record(); read(); e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1e3

    out = {}
    hot, cold = [], []
    for i in range(reps):
        read(); hot.append(read_timed())
        flush.zero_(); cold.append(read_timed())
    out["hot"], out["cold"] = st.median(hot), st.median(cold)
    k = 0
    for arm in arms:
        set_arm(ARMS[arm][0], 0)
        fns[k % len(fns)](); torch.cuda.synchronize(); k += 1
        v = []
        for i in range(reps):
            flush.zero_()
            read()
            fns[k % len(fns)](); k += 1
            v.append(read_timed())
        set_arm(0, 0)
        out[arm] = st.median(v)
    del victim, flush
    torch.cuda.empty_cache()
    return out


def pick(summary, lever_margin_us=0.0):
    """Pre-registered rule (HOW-TO-VERIFY): the gate's R3 arm from the side-stream P0.
    lever: E+P if the 48-layer total of m3EP is below m3E at BOTH r4/D28 and r16/D77, else E.
    kernel mode per row class under that lever, the lower 48-layer total of m2<L> vs m3<L>:
      2-4 rows <- r4/D28;  5-12 rows <- r8/D40 + r12/D60 (if measured, else follows 2-4);  13-16 rows <- r16/D77.
    R3_ENV = EXL3_MOE_COOP_V3=3 [+ EXL3_MOE_COOP_V3_MAP for classes that pick mode 2] + EXL3_SHARED_EXPERT_EARLY=1
    [+ EXL3_SHARED_EXPERT_PRIO=1]; every class on mode 2 -> EXL3_MOE_COOP_V3=2 + the lever."""
    def tot(arm, rows, D):
        v = 0.0
        for K, n in N_LAYERS.items():
            s = summary.get(f"K{K}/r{rows}/D{D}")
            if not s or arm not in s["us"]:
                return None
            v += n * s["us"][arm]
        return v
    need = [("m3E", 4, 28), ("m3E", 16, 77), ("m2E", 4, 28), ("m2E", 16, 77)]
    if any(tot(a, r, D) is None for a, r, D in need):
        return None, "missing cells/arms for the rule"
    ep = all(tot("m3EP", r, D) is not None and tot("m3EP", r, D) + lever_margin_us < tot("m3E", r, D)
             for r, D in ((4, 28), (16, 77)))
    L = "EP" if ep else "E"
    classes = {"2-4": [(4, 28)], "5-12": [(8, 40), (12, 60)], "13-16": [(16, 77)]}
    modes, why = {}, []
    for cls, cells in classes.items():
        c2 = [tot("m2" + L, r, D) for r, D in cells]
        c3 = [tot("m3" + L, r, D) for r, D in cells]
        if any(v is None for v in c2 + c3):
            if cls != "5-12":
                return None, f"class {cls}: m2{L}/m3{L} not measured at {cells}"
            modes[cls] = None
            continue
        modes[cls] = 2 if sum(c2) < sum(c3) else 3
        why.append(f"{cls}: m2{L} {sum(c2) / 1e3:.3f} vs m3{L} {sum(c3) / 1e3:.3f} ms -> mode {modes[cls]}")
    if modes["5-12"] is None:
        modes["5-12"] = modes["2-4"]
        why.append("5-12: not measured, follows 2-4")
    env = []
    if all(m == 2 for m in modes.values()):
        env.append("EXL3_MOE_COOP_V3=2")
    else:
        env.append("EXL3_MOE_COOP_V3=3")
        mp = ",".join(f"{c}:2" for c, m in modes.items() if m == 2)
        if mp:
            env.append(f"EXL3_MOE_COOP_V3_MAP={mp}")
    env.append("EXL3_SHARED_EXPERT_EARLY=1")
    if L == "EP":
        env.append("EXL3_SHARED_EXPERT_PRIO=1")
    return " ".join(env), f"lever {L}; " + "; ".join(why)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--extra", action="store_true", help="also the non-served D sweep (EXTRA)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=128)
    ap.add_argument("--warm", type=int, default=16)
    ap.add_argument("--prof-calls", type=int, default=64)
    ap.add_argument("--no-side", action="store_true", help="r2's P0 cells: shared output precomputed, no router/side stream")
    ap.add_argument("--ncu", action="store_true", help="profile one call per served cell and arm (no side stream)")
    ap.add_argument("--l2probe", action="store_true", help="L2 retention probe instead of timing (no side stream)")
    ap.add_argument("--arms", nargs="+", default=None, choices=list(ARMS))
    a = ap.parse_args()

    assert os.environ.get("EXL3_MOE_COOP_V2") == "1", "run with the served env (EXL3_MOE_COOP_V2=1)"
    for k in ("EXL3_MOE_COOP_V3", "EXL3_MOE_COOP_V3_MAP", "EXL3_MOE_COOP_V3_HEAD"):
        os.environ.pop(k, None)
    legacy = a.no_side or a.ncu or a.l2probe
    arms = a.arms or (["ref", "m2", "m3"] if legacy else DEFAULT_ARMS)
    if legacy:
        assert all(ARMS[x][1] == "late" and not ARMS[x][2] and not ARMS[x][3] for x in arms), \
            "--no-side / --ncu / --l2probe take kernel-mode arms only (ref, m2, m3)"
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from torch.profiler import profile, ProfilerActivity
    assert getattr(ext, "moe_coop_v3_revision", 0) >= 3, "needs the moefast r3 extension (exl3_moe_coop_ev)"
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    l2 = torch.cuda.get_device_properties(dev).L2_cache_size
    a.out.mkdir(parents=True, exist_ok=True)
    jl = (a.out / ("p0-l2probe.jsonl" if a.l2probe else "p0.jsonl")).open("w")
    print(f"[p0] arms {arms}; {'legacy (no side stream)' if legacy else 'served side stream + router'}; "
          f"revision {ext.moe_coop_v3_revision}", flush=True)

    def emit(rec):
        jl.write(json.dumps(rec) + "\n")
        jl.flush()

    ckpt = d0.Checkpoint(a.model, dev)
    C = d0.Cells(torch, ext, dev, l2, ckpt, False)
    timer = None if (a.ncu or a.l2probe) else d0.Timer(torch, a.reps, a.warm)
    cells = SERVED + (EXTRA if a.extra else [])
    if a.l2probe:
        cells = [(4, 28), (16, 77)]
    side = None
    ovs = {}
    if not legacy:
        side = side_mod.SharedSet(torch, ext, d0, ckpt, dev, log=lambda *s: print(*s, flush=True))
        side.warm(sorted({r for r, _ in cells}))
        ovs = {None: side_mod.Overlap(torch, dev, None), "high": side_mod.Overlap(torch, dev, "high")}
        print(f"[p0] side streams: served priority {ovs[None].priority}, high {ovs['high'].priority} "
              f"(range {torch.cuda.Stream.priority_range()})", flush=True)
        if ovs[None].priority == ovs["high"].priority:
            print("[p0] WARNING: high-priority side stream has the default priority; the P arms equal the plain ones",
                  flush=True)
    summary = {}

    for K in a.k:
        pfx = d0.EXPERT_LAYERS[K]
        t0 = time.time()
        experts = [{p: ckpt.linear(f"{pfx}.experts.{e}.{p}_proj") for p in ("gate", "up", "down")}
                   for e in range(d0.N_EXPERTS)]
        tab, K_, Kd, eb = C.moe_tables(experts)
        assert K_ == K and Kd == K
        print(f"[p0] K={K} {pfx}: {eb} B/expert, loaded in {time.time() - t0:.1f}s", flush=True)
        for rows, D in cells:
            label = f"K{K}/r{rows}/D{D}"
            if legacy:
                fns = legacy_fns(torch, ext, dev, C.moe_scr, tab, K, Kd, rows, D)
                fns_of = {arm: fns for arm in arms}          # arms differ only by the per-launch env
            else:
                sets = routing_sets(torch, dev, rows, D)
                w = torch.rand((rows, d0.TOPK), device=dev) + 0.1
                rw = (w / w.sum(-1, keepdim=True)).half().contiguous()
                x = (torch.randn((rows, d0.HIDDEN), device=dev) * 0.5).half().contiguous()
                fns_of = {arm: [side_mod.make_call(torch, ext, side, ovs[ARMS[arm][2]], tab, K, Kd, C.moe_scr,
                                                   d0.HIDDEN, x, sel, rw, j, ARMS[arm][1])
                                for j, sel in enumerate(sets)] for arm in arms}
            if a.l2probe:
                for vmb in (32, 64):
                    r = l2probe(torch, fns_of[arms[0]], arms, vmb)
                    emit(dict(kind="l2probe", cell=label, victim_mb=vmb, us=r))
                    print(f"[p0] l2probe {label} victim {vmb} MB: hot {r['hot']:.1f} cold {r['cold']:.1f} us | "
                          + "  ".join(f"{arm} {r[arm]:.1f}" for arm in arms), flush=True)
                continue
            if a.ncu:
                for j, arm in enumerate(arms):
                    fns = fns_of[arm]
                    set_arm(ARMS[arm][0], ARMS[arm][3])
                    for i in range(3):
                        fns[(4 * j + 1 + i) % len(fns)]()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push(f"moefast/{arm}/{label}")
                    fns[(4 * j) % len(fns)]()
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                    set_arm(0, 0)
                    emit(dict(kind="ncu", cell=label, arm=arm))
                    print(f"[p0] ncu {arm} {label}", flush=True)
                continue
            per = {arm: [] for arm in arms}
            batch = {arm: [] for arm in arms}
            for rnd in range(a.rounds):
                for arm in arms[rnd % len(arms):] + arms[:rnd % len(arms)]:
                    set_arm(ARMS[arm][0], ARMS[arm][3])
                    r = timer.run(fns_of[arm])
                    set_arm(0, 0)
                    per[arm].append(r["us_each"])
                    batch[arm].append(r["us_batch"])
                    emit(dict(kind="time", cell=label, K=K, rows=rows, D=D, arm=arm, round=rnd, bytes=D * eb,
                              side=not legacy, **r))
            tl, kern = {}, {}
            for arm in arms:
                fns = fns_of[arm]
                set_arm(ARMS[arm][0], ARMS[arm][3])
                for i in range(8):
                    fns[i % len(fns)]()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for i in range(a.prof_calls):
                        fns[i % len(fns)]()
                    torch.cuda.synchronize()
                set_arm(0, 0)
                ks = [(e.time_range.start, e.time_range.end, side_mod.kernel_role(e.name)) for e in prof.events()
                      if e.device_type == torch.autograd.DeviceType.CUDA]
                durs = {}
                for s0, s1, role in ks:
                    durs.setdefault(role, []).append(s1 - s0)
                kern[arm] = {c: st.median(v) for c, v in durs.items()}
                tl[arm] = side_mod.timeline(ks)
                emit(dict(kind="timeline", cell=label, K=K, rows=rows, D=D, arm=arm, kernels_us=kern[arm], **tl[arm]))
            med = {arm: st.median(v) for arm, v in per.items()}
            ref = med.get("ref")
            rec = dict(cell=label, K=K, rows=rows, D=D, us=med, us_batch={k: st.median(v) for k, v in batch.items()},
                       spread={arm: (min(v), max(v)) for arm, v in per.items()}, kernels=kern, timeline=tl,
                       ratio={arm: med[arm] / ref for arm in med} if ref else {}, side=not legacy)
            summary[label] = rec
            print(f"[p0] {label:14s} " + "  ".join(f"{arm} {med[arm]:7.2f}" for arm in arms), flush=True)
            fmt = lambda v: "   -  " if v is None else f"{v:6.1f}"
            for arm in arms:
                t = tl[arm]
                print(f"[p0]   {arm:5s} gap {fmt(t['gap'])}  sh-A0 {fmt(t['sh_start'])}  sh-A1 {fmt(t['sh_end'])}  "
                      f"sh span {fmt(t['sh_span'])}  moe span {fmt(t['moe_span'])}  call {fmt(t['call_span'])}  "
                      f"head/rot/A/B " + "/".join(f"{kern[arm].get(c, 0):.1f}" for c in ("head", "rot", "A", "B")),
                      flush=True)
        del experts, tab
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if a.ncu or a.l2probe:
        return 0
    rep = report(summary, legacy)
    (a.out / "p0-summary.json").write_text(json.dumps(dict(cells=summary, report=rep), indent=1))
    return 0


def report(summary, legacy):
    """Diagnostic (OPERATIONS §16: P0 explains and picks the per-shape dispatch; P1 gates)."""
    out = {}
    print("\n[p0] ---- summary ----")
    ctl_arms = ("ref", "m2", "m3", "m3H", "m3H2")      # at 1 row every one of these is V2, late fork
    for c in [c for c in summary if "/r1/" in c]:
        r = {k: v for k, v in summary[c]["ratio"].items() if k in ctl_arms}
        flag = "ok" if r and max(abs(v - 1) for v in r.values()) <= 0.03 else "CONTROL DRIFT (>3 %): rerun"
        print(f"[p0] control {c}: {r}  {flag}")
        out.setdefault("control", {})[c] = flag
    shapes = sorted({(v["rows"], v["D"]) for v in summary.values()})
    tots = {}
    for rows, D in shapes:
        tot = {}
        for K, n_layers in N_LAYERS.items():
            s = summary.get(f"K{K}/r{rows}/D{D}")
            if not s:
                continue
            for arm, us in s["us"].items():
                tot[arm] = tot.get(arm, 0.0) + n_layers * us
        if not tot:
            continue
        tots[f"r{rows}/D{D}"] = tot
        base = tot.get("m2", tot.get("ref"))
        best = min(tot, key=tot.get)
        print(f"[p0] r{rows} D{D}: 48-layer MoE " + "  ".join(f"{arm} {v / 1000:.3f}" for arm, v in tot.items())
              + f" ms  -> best {best} ({(tot[best] - base) / 1000:+.3f} ms vs m2)")
    out["totals_us"] = tots
    if legacy:
        return out
    # instrument acceptance: does the side-stream P0 reproduce R703b's r2 overlap loss?
    ok = True
    for K in N_LAYERS:
        s = summary.get(f"K{K}/r4/D28")
        if not s:
            ok = False
            continue
        g2, g3 = s["timeline"]["m2"]["gap"], s["timeline"]["m3"]["gap"]
        d = None if g2 is None or g3 is None else g3 - g2
        print(f"[p0] INSTRUMENT K{K}/r4/D28: A->B gap m2 {g2} m3 {g3} us (m3 - m2 {d}; R703b nsys c1d3: 1.9 -> 14.1)")
        ok = ok and d is not None and d >= 5.0
    out["instrument"] = "OK" if ok else "NOT REPRODUCED"
    print(f"[p0] INSTRUMENT {'OK: reproduces the r2 overlap loss' if ok else 'NOT REPRODUCED: P0 does not show the r2 gap; the PICK below is not trustworthy'}")
    env, why = pick(summary)
    out["pick"] = dict(env=env, why=why)
    if env:
        print(f"[p0] PICK {why}")
        print(f"[p0] R3_ENV={env}")
    else:
        print(f"[p0] PICK FAILED: {why}")
    return out


if __name__ == "__main__":
    sys.exit(main())
