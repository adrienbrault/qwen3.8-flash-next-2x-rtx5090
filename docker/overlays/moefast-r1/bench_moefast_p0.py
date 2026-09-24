#!/usr/bin/env python3
"""moefast r1 P0: served routed-expert cells, reference (V2) vs V3 mode 1 vs V3 mode 2, µs per call and per kernel.

Harness: d0_microbench (R682) cells, unchanged: ext.exl3_moe_coop on real experts (K=2 layer 20, K=3 layer 3),
64 routings walking a random permutation of the 512 experts (consecutive calls touch disjoint experts, so
weights come from DRAM), CUDA-event timing behind torch.cuda._sleep. EXL3_MOE_COOP_V3 is read per launch, so
the three arms run in one process on identical inputs, interleaved: for each cell, `--rounds` rounds of
(ref, m1, m2) in rotating order; the arm's number is the median over rounds of each round's median.

Per-kernel times (rot / a / b) come from a torch.profiler (CUPTI) pass per cell and arm: `--prof-calls` calls,
median duration per kernel. They are what the pre-registered per-kernel predictions are checked against.

--ncu: no timing; for each served cell and arm j, 3 warm-up calls on routings 4j+1..4j+3, then ONE call on
routing 4j between cudaProfilerStart/Stop inside NVTX range "moefast/<arm>/K<K>/r<rows>/D<D>" (run under
ncu --profile-from-start off --cache-control all; HOW-TO-VERIFY §1). Each arm profiles its own routing,
whose experts its warm-ups do not touch (consecutive routings use disjoint experts). Use --cache-control all:
ncu replays each kernel once per section pass, and with 'none' the later passes would find the r4 weights
(35-52 MB) warm in the 96 MB L2. 'all' makes weights and activations cold in both arms.

Output: <out>/p0.jsonl (one record per cell x arm x round, plus per-kernel records), <out>/p0-summary.json,
and a table + the pre-registered verdict (HOW-TO-VERIFY.md §3) on stdout.
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
import d0_microbench as d0  # noqa: E402

ARMS = {"ref": 0, "m1": 1, "m2": 2}
SERVED = [(1, 10), (4, 28), (16, 77)]                     # rows, distinct experts (R682 Q2 / R619)
EXTRA = [(4, 10), (4, 20), (4, 40), (8, 40), (12, 60), (16, 40), (16, 160)]


def set_arm(mode):
    if mode:
        os.environ["EXL3_MOE_COOP_V3"] = str(mode)
    else:
        os.environ.pop("EXL3_MOE_COOP_V3", None)


def kernel_class(name):
    if "rot_kernel" in name:
        return "rot"
    if "_a_kernel" in name:
        return "a"
    if "_b_kernel" in name:
        return "b"
    return None


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
    ap.add_argument("--ncu", action="store_true", help="profile one call per served cell and arm (under ncu)")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    a = ap.parse_args()

    assert os.environ.get("EXL3_MOE_COOP_V2") == "1", "run with the served env (EXL3_MOE_COOP_V2=1)"
    os.environ.pop("EXL3_MOE_COOP_V3", None)
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from torch.profiler import profile, ProfilerActivity
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    l2 = torch.cuda.get_device_properties(dev).L2_cache_size
    a.out.mkdir(parents=True, exist_ok=True)
    jl = (a.out / "p0.jsonl").open("w")

    def emit(rec):
        jl.write(json.dumps(rec) + "\n")
        jl.flush()

    ckpt = d0.Checkpoint(a.model, dev)
    C = d0.Cells(torch, ext, dev, l2, ckpt, False)          # 64 routings also in --ncu mode (see docstring)
    timer = None if a.ncu else d0.Timer(torch, a.reps, a.warm)
    cells = SERVED + (EXTRA if a.extra else [])
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
            fns = C.moe(tab, K, Kd, rows, D)
            label = f"K{K}/r{rows}/D{D}"
            if a.ncu:
                for j, arm in enumerate(a.arms):
                    set_arm(ARMS[arm])
                    for i in range(3):
                        fns[(4 * j + 1 + i) % len(fns)]()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push(f"moefast/{arm}/{label}")
                    fns[(4 * j) % len(fns)]()
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                    set_arm(0)
                    emit(dict(kind="ncu", cell=label, arm=arm))
                    print(f"[p0] ncu {arm} {label}", flush=True)
                continue
            per = {arm: [] for arm in a.arms}
            batch = {arm: [] for arm in a.arms}
            order = list(a.arms)
            for rnd in range(a.rounds):
                for arm in order[rnd % len(order):] + order[:rnd % len(order)]:
                    set_arm(ARMS[arm])
                    r = timer.run(fns)
                    set_arm(0)
                    per[arm].append(r["us_each"])
                    batch[arm].append(r["us_batch"])
                    emit(dict(kind="time", cell=label, K=K, rows=rows, D=D, arm=arm, round=rnd, bytes=D * eb, **r))
            # per-kernel pass
            kern = {}
            for arm in a.arms:
                set_arm(ARMS[arm])
                for i in range(8):
                    fns[i % len(fns)]()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for i in range(a.prof_calls):
                        fns[i % len(fns)]()
                    torch.cuda.synchronize()
                set_arm(0)
                durs, names = {}, {}
                for e in prof.events():
                    c = kernel_class(e.name) if "moe_coop" in e.name else None
                    if c and e.device_type == torch.autograd.DeviceType.CUDA:
                        durs.setdefault(c, []).append(e.time_range.end - e.time_range.start)   # device interval, µs
                        names[c] = e.name
                kern[arm] = {c: st.median(v) for c, v in durs.items()}
                emit(dict(kind="kernels", cell=label, K=K, rows=rows, D=D, arm=arm,
                          us=kern[arm], names=names, n={c: len(v) for c, v in durs.items()}))
            med = {arm: st.median(v) for arm, v in per.items()}
            rec = dict(cell=label, K=K, rows=rows, D=D, us=med, us_batch={k: st.median(v) for k, v in batch.items()},
                       spread={arm: (min(v), max(v)) for arm, v in per.items()}, kernels=kern,
                       ratio={arm: med[arm] / med["ref"] for arm in med}, GBps_ref=D * eb / med["ref"] / 1e3)
            summary[label] = rec
            kr = " ".join(f"{arm}:" + "/".join(f"{kern[arm].get(c, 0):.1f}" for c in ("rot", "a", "b")) for arm in kern)
            print(f"[p0] {label:14s} " + "  ".join(f"{arm} {med[arm]:7.2f} ({med[arm] / med['ref']:.3f})" for arm in med)
                  + f"   rot/a/b µs {kr}", flush=True)
        del experts, tab
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if a.ncu:
        return
    (a.out / "p0-summary.json").write_text(json.dumps(summary, indent=1))
    verdict(summary)


def verdict(summary):
    """Pre-registered bars (HOW-TO-VERIFY.md §3). best = the faster of m1 / m2 per cell."""
    print("\n[p0] ---- verdict (pre-registered) ----")
    dec = [c for c in summary if c.split("/")[1] in ("r4", "r16") and
           (c.endswith("/r4/D28") or c.endswith("/r16/D77"))]
    best = {c: min(v for k, v in summary[c]["ratio"].items() if k != "ref") for c in dec}
    for c in dec:
        print(f"[p0] {c:14s} best ratio {best[c]:.3f}  ({summary[c]['ratio']})")
    ctrl = [c for c in summary if "/r1/" in c]
    for c in ctrl:
        r = summary[c]["ratio"]
        flag = "ok" if max(abs(v - 1) for v in r.values()) <= 0.03 else "CONTROL DRIFT (>3 %): rerun"
        print(f"[p0] control {c}: {r}  {flag}")
    if not dec:
        print("[p0] no decision cells")
        return
    if all(b <= 0.80 for b in best.values()):
        print("[p0] PASS: >= 20 % faster at every decision cell (K2/K3 r4 D28 and r16 D77) -> run P1")
    elif all(b > 0.95 for b in best.values()):
        print("[p0] KILL: < 5 % at every decision cell -> reject V3")
    else:
        print("[p0] MIXED: see HOW-TO-VERIFY.md §3 (P1 runs if the step prediction >= 0.4 ms at c1d3 or c8d1)")
    # step prediction: 48 layers per step; K3 on 23 layers, K2 on 25 (the pack's split)
    for rows, D, shape in ((4, 28, "c1d3"), (16, 77, "c8d1 / c4d3")):
        k2, k3 = summary.get(f"K2/r{rows}/D{D}"), summary.get(f"K3/r{rows}/D{D}")
        if k2 and k3:
            for arm in [x for x in k2["us"] if x != "ref"]:
                saved = 25 * (k2["us"]["ref"] - k2["us"][arm]) + 23 * (k3["us"]["ref"] - k3["us"][arm])
                print(f"[p0] {shape} {arm}: predicted step saving {saved / 1000:.2f} ms (48 layers, events medians)")


if __name__ == "__main__":
    main()
