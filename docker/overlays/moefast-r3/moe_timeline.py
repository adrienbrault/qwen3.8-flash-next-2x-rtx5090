#!/usr/bin/env python3
"""moefast r2/r3: per-layer MoE timeline and per-step kernel totals from nsys sqlite exports, per arm.

r3: with EXL3_SHARED_EXPERT_EARLY the shared expert starts before kernel A (during the router), so "shared start -
A start" goes negative by up to the router time; the 60 us look-back window below covers it. --gap-target prints
one "GAP <tag> <median> OK|HIGH" line per arm (r3 target: A->B gap back to M2's ~2 us). The ms/step columns divide
by --layers-per-step over multi-row A launches of both cards and the MTP draft (R703b review, defect 8): estimates.

Usage: moe_timeline.py [--tail-frac 0.5] [--layers-per-step 48] TAG=trace.sqlite [TAG=trace.sqlite ...]

For every routed-MoE layer launched with more than one token row (kernel A of V2 / r1 / r2 at grid 340, or any
V3 / r2 kernel), in the last --tail-frac of the trace (steady decode: the harness's settle + capture steps):
  rot, A, B     device durations (us); rot is absent under r2 (mode 3)
  A->B gap      B start - A end: the wait on the shared-expert event plus the launch gap
  shared        side-stream kernels (the shared expert, EXL3_SHARED_EXPERT_OVERLAP) between the layer's first
                kernel and B: their span, start relative to A start, end relative to A end
  layer         first MoE kernel start -> B end
and, over the same window, every kernel name's device time per decode step (step count = multi-row layers /
--layers-per-step), so a slowdown outside the MoE kernels (e.g. L2 pollution by r1's weight copies) shows up
as a per-name delta between arms. All numbers are medians over layers unless marked.
"""
import argparse
import re
import sqlite3
import statistics as st
from collections import defaultdict


def short(name):
    name = name.split("(")[0]
    name = re.sub(r"<.*", "", name)
    return name.split("::")[-1][-60:]


def load(path):
    c = sqlite3.connect(path)
    rows = c.execute("""select k.start, k.end, k.deviceId, k.streamId, k.gridX, s.value
                        from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.demangledName
                        order by k.start""").fetchall()
    return rows


def is_moe(n, part):
    return "moe_coop" in n and part in n


def multi_row_a(k):
    n = k[5]
    return is_moe(n, "_a_kernel") and ("v3_ns" in n or "r2_ns" in n or k[4] == 340)


def analyse(path, tail_frac, lps):
    K = load(path)
    a_all = [k for k in K if multi_row_a(k)]
    if not a_all:
        return None
    t_lo = a_all[int(len(a_all) * (1 - tail_frac))][0]
    t_hi = a_all[-1][1] + 5_000_000
    W = [k for k in K if t_lo <= k[0] <= t_hi]
    per = defaultdict(list)
    by_dev = defaultdict(list)
    for k in W:
        by_dev[k[2]].append(k)
    layers = 0
    for dev, ks in by_dev.items():
        idx_a = [i for i, k in enumerate(ks) if multi_row_a(k)]
        for i in idx_a:
            a = ks[i]
            b = next((k for k in ks[i + 1:i + 60] if is_moe(k[5], "_b_kernel") and k[3] == a[3]), None)
            if b is None:
                continue
            rot = next((k for k in reversed(ks[max(0, i - 40):i]) if "rot_kernel" in k[5] and k[3] == a[3]
                        and a[0] - k[1] < 50_000), None)
            first = rot[0] if rot else a[0]
            side = [k for k in ks if k[3] != a[3] and first - 60_000 <= k[0] <= b[0] and k[1] >= first - 60_000]
            layers += 1
            per["rot"].append((rot[1] - rot[0]) / 1e3 if rot else 0.0)
            per["A"].append((a[1] - a[0]) / 1e3)
            per["B"].append((b[1] - b[0]) / 1e3)
            per["A->B gap"].append((b[0] - a[1]) / 1e3)
            per["layer"].append((b[1] - first) / 1e3)
            if side:
                s0, s1 = min(k[0] for k in side), max(k[1] for k in side)
                per["shared span"].append((s1 - s0) / 1e3)
                per["shared start - A start"].append((s0 - a[0]) / 1e3)
                per["shared end - A end"].append((s1 - a[1]) / 1e3)
    steps = layers / lps if lps else 1
    tot = defaultdict(float)
    cnt = defaultdict(int)
    for k in W:
        n = short(k[5])
        tot[n] += (k[1] - k[0]) / 1e6
        cnt[n] += 1
    return dict(layers=layers, steps=steps, per={k: st.median(v) for k, v in per.items() if v},
                p90={k: sorted(v)[int(0.9 * (len(v) - 1))] for k, v in per.items() if v},
                ms_per_step={n: v / steps for n, v in tot.items()}, launches={n: c / steps for n, c in cnt.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="TAG=path.sqlite")
    ap.add_argument("--tail-frac", type=float, default=0.5)
    ap.add_argument("--layers-per-step", type=int, default=48)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--gap-target", type=float, default=None, help="print GAP <tag> <median us> OK|HIGH per arm")
    a = ap.parse_args()
    res = {}
    for t in a.traces:
        tag, path = t.split("=", 1)
        res[tag] = analyse(path, a.tail_frac, a.layers_per_step)
        if res[tag] is None:
            print(f"{tag}: no multi-row MoE layers found")
    tags = [t for t in res if res[t]]
    if not tags:
        return
    print("per-layer medians (us); p90 in brackets")
    keys = ["rot", "A", "B", "A->B gap", "layer", "shared span", "shared start - A start", "shared end - A end"]
    print(f"{'':26s}" + "".join(f"{t:>20s}" for t in tags))
    print(f"{'layers / est. steps':26s}" + "".join(f"{res[t]['layers']:>11d} / {res[t]['steps']:6.1f}" for t in tags))
    for k in keys:
        print(f"{k:26s}" + "".join(f"{res[t]['per'].get(k, float('nan')):>11.2f} [{res[t]['p90'].get(k, float('nan')):6.2f}]"
                                   for t in tags))
    base = tags[0]
    names = sorted(set().union(*[res[t]["ms_per_step"] for t in tags]),
                   key=lambda n: -max(res[t]["ms_per_step"].get(n, 0) for t in tags))[:a.top]
    print(f"\nkernel device ms per decode step (window), deltas vs {base}")
    print(f"{'kernel':62s}" + "".join(f"{t:>12s}" for t in tags) + "".join(f"{'d ' + t:>12s}" for t in tags[1:]))
    for n in names:
        v = [res[t]["ms_per_step"].get(n, 0.0) for t in tags]
        print(f"{n:62s}" + "".join(f"{x:12.4f}" for x in v) + "".join(f"{x - v[0]:+12.4f}" for x in v[1:]))
    if a.gap_target is not None:
        for t in tags:
            g = res[t]["per"].get("A->B gap")
            gs = "n/a" if g is None else f"{g:.2f}"
            print(f"GAP {t} {gs} {'OK' if g is not None and g <= a.gap_target else 'HIGH'} (target <= {a.gap_target} us)")
    for t in tags:
        moe = sum(v for n, v in res[t]["ms_per_step"].items() if "moe_coop" in n)
        other = sum(v for n, v in res[t]["ms_per_step"].items() if "moe_coop" not in n)
        print(f"{t}: MoE kernels {moe:.3f} ms/step, all other kernels {other:.3f} ms/step")


if __name__ == "__main__":
    main()
