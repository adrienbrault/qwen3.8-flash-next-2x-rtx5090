#!/usr/bin/env python3
"""stack-r3 served-boot analysis (pre-registered bars), reading the gate's per-boot artifacts in R:

  boot-<tag>.log       the launcher's output; its "UP on <port> | warmup <s> | VRAM free MiB <a>/<b>/" line is the
                       headroom read (after warm-up, before any request: the R712 review's clean read)
  container-<tag>.log  docker logs of the boot at its end: OOM, TORCH_CHECK / c10 errors, tracebacks, lcguard lines
  ramp-<tag>.jsonl     fn_bench ramp c1..c8 rows (ok / not ok)
  gate-<tag>/bench-{A,B}.jsonl   canonical fn_gate RUNS=3 rows
  env-<tag>.txt        `docker exec flashnext env` of the boot (EXL3_* lines)

  served_stack.py --results R --a "A1 A2 A3" --b "B1 B2 B3" [--extra "C"] --union "<UNION>" [--expect-lcguard]
                  [--json out.json]
Bars:
  headroom  per card: B boot UP-line free >= min(A boots' UP-line free) - 32 MiB (both cards, every B boot)
  safety    0 OOM lines, 0 TORCH_CHECK / c10::Error, 0 tracebacks in every boot; every ramp row ok; leg B all ok
  env       B boots: every UNION key present with its value, no duplicate EXL3 keys; A boots: no UNION-only key
  lcguard   with --expect-lcguard (QF on and a dense mode on): every B container log shows the guard's side-path
            line (else the guard never engaged in the served capture and the co-existence claim is untested)
  fn_gate   3 ABAB pairs; per cell the mean over pairs of ON/OFF (median per-stream decode t/s per boot and cell):
            no cell < 0.99, except leg-A c1 >= 0.98 (§16 c1 tolerance); aggregate (mean over cells) > 1.000
Prints HEADROOM / SAFETY / ENV / LCGUARD / RATIO / MEAN / AGGREGATE lines and "SERVED-VERDICT PASS|FAIL: ...".
"""
import argparse
import json
import os
import re
import statistics as st
import sys

UP = re.compile(r"UP on \d+ \| warmup \S+ \| VRAM free MiB ([0-9/ ]+)")
OOM = re.compile(r"OutOfMemoryError|out of memory|CUDA error: out of memory|graph\.cu")
TCHK = re.compile(r"TORCH_CHECK|c10::Error|Exception raised from")
LCG = re.compile(r"densegemm lcguard: device (\d+)")


def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return None


def up_free(R, tag):
    t = read(f"{R}/boot-{tag}.log")
    if not t:
        return None
    m = UP.findall(t)
    if not m:
        return None
    vals = [int(x) for x in m[-1].replace(" ", "").split("/") if x]
    return vals if len(vals) >= 2 else None


def cells(R, tag, leg):
    p = f"{R}/gate-{tag}/bench-{leg}.jsonl"
    out, ok, n = {}, 0, 0
    if not os.path.exists(p):
        return out, 0, 0
    for ln in open(p):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        n += 1
        if not d.get("ok"):
            continue
        ok += 1
        c, v = d.get("conc"), d.get("decode_tps")
        if c is not None and v:
            out.setdefault(c, []).append(v)
    return {c: st.median(v) for c, v in out.items()}, ok, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--union", default="")
    ap.add_argument("--expect-lcguard", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    R, A, B, X = a.results, a.a.split(), a.b.split(), a.extra.split()
    why, out = [], {}
    # headroom
    fa = {t: up_free(R, t) for t in A}
    fb = {t: up_free(R, t) for t in B + X}
    for t, f in {**fa, **fb}.items():
        print(f"HEADROOM {t}: UP-line free MiB {f if f else 'MISSING'}")
    good_a = [f for f in fa.values() if f]
    if not good_a:
        why.append("no UP line in any A boot (no headroom reference)")
    else:
        ref = [min(f[i] for f in good_a) for i in range(2)]
        print(f"HEADROOM reference (min over A boots): cuda:0 {ref[0]} cuda:1 {ref[1]} MiB; bar = ref - 32")
        for t in B + X:
            f = fb.get(t)
            if not f:
                why.append(f"{t}: no UP line"); continue
            short = [f"cuda:{i} {f[i]} < {ref[i] - 32}" for i in range(2) if f[i] < ref[i] - 32]
            print(f"HEADROOM {t}: " + ("ok" if not short else "FAIL " + ", ".join(short)) +
                  f" (delta vs ref {f[0] - ref[0]:+d} / {f[1] - ref[1]:+d} MiB)")
            if short:
                why.append(f"headroom {t}: " + ", ".join(short))
        out["headroom_ref"] = ref
    # safety + lcguard
    for t in A + B + X:
        c = read(f"{R}/container-{t}.log") or ""
        lines = c.splitlines()
        oom, tchk, tb = (sum(1 for l in lines if rx.search(l)) for rx in (OOM, TCHK, re.compile("Traceback")))
        devs = sorted(set(LCG.findall(c)))
        rp = f"{R}/ramp-{t}.jsonl"
        ramp_bad = ramp_n = 0
        if os.path.exists(rp):
            for ln in open(rp):
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                ramp_n += 1; ramp_bad += not d.get("ok")
        _, okb, nb = cells(R, t, "B")
        print(f"SAFETY {t}: OOM {oom}, TORCH_CHECK/c10 {tchk}, tracebacks {tb}, ramp {ramp_n - ramp_bad}/{ramp_n} ok, "
              f"leg B {okb}/{nb} ok, lcguard devices {devs or '-'}" + ("" if c else " (NO CONTAINER LOG)"))
        if not c:
            why.append(f"{t}: container log missing")
        if oom or tchk or tb:
            why.append(f"{t}: OOM {oom} / TORCH_CHECK {tchk} / tracebacks {tb}")
        if ramp_bad or (t in B and ramp_n == 0):
            why.append(f"{t}: ramp {ramp_bad} failed of {ramp_n}")
        if nb and okb != nb:
            why.append(f"{t}: leg B {okb}/{nb}")
        if a.expect_lcguard and t in B and not devs:
            why.append(f"{t}: lcguard never logged (QF + dense mode on, but no side-branch dense call reached the guard)")
    # env
    union = dict(kv.split("=", 1) for kv in a.union.split()) if a.union else {}
    for t in A + B + X:
        e = read(f"{R}/env-{t}.txt")
        if e is None:
            why.append(f"{t}: env dump missing"); continue
        kv = [l.strip() for l in e.splitlines() if l.startswith("EXL3_")]
        keys = [l.split("=", 1)[0] for l in kv]
        dup = sorted({k for k in keys if keys.count(k) > 1})
        env = dict(l.split("=", 1) for l in kv)
        if t in B:
            miss = [f"{k}={v}" for k, v in union.items() if env.get(k) != v]
        else:
            miss = [k for k in union if k in env and env[k] != "0" and k.startswith(("EXL3_LC_", "EXL3_DENSE_", "EXL3_SHARED_EXPERT_EARLY", "EXL3_SHARED_EXPERT_PRIO"))]
        print(f"ENV {t}: {len(kv)} EXL3 keys; duplicates {dup or '-'}; " + (f"union keys missing/wrong {miss or '-'}" if t in B else f"stack-r3-only keys present {miss or '-'}"))
        if dup or miss:
            why.append(f"env {t}: duplicates {dup} / {'missing' if t in B else 'unexpected'} {miss}")
    # fn_gate ratios
    means = []
    per = {}
    for leg in ("A", "B"):
        for i, (ta, tb) in enumerate(zip(A, B), 1):
            ca, _, _ = cells(R, ta, leg)
            cb, _, _ = cells(R, tb, leg)
            for c in sorted(set(ca) | set(cb)):
                r = cb[c] / ca[c] if ca.get(c) and cb.get(c) else float("nan")
                per.setdefault((leg, c), []).append(r)
                print(f"RATIO leg {leg} pair {i} c{c}: OFF {ca.get(c, float('nan')):.1f} ON {cb.get(c, float('nan')):.1f} ON/OFF {r:.4f}")
    for (leg, c), rs in sorted(per.items()):
        rs = [r for r in rs if r == r]
        if len(rs) < len(A):
            why.append(f"fn_gate leg {leg} c{c}: {len(rs)} valid pairs of {len(A)}")
        if not rs:
            continue
        m = st.mean(rs)
        means.append(m)
        bar = 0.98 if (leg == "A" and c == 1) else 0.99
        flag = "" if m >= bar else f"  < {bar} FAIL"
        print(f"MEAN leg {leg} c{c}: {m:.4f} (bar {bar}; pairs {' '.join(f'{r:.3f}' for r in rs)}){flag}")
        if m < bar:
            why.append(f"fn_gate leg {leg} c{c} mean {m:.4f} < {bar}")
    agg = st.mean(means) if means else float("nan")
    print(f"AGGREGATE mean ON/OFF over cells: {agg:.4f}" + ("" if agg > 1.0 else "  <= 1.000 FAIL"))
    if not (agg > 1.0):
        why.append(f"aggregate {agg:.4f} <= 1.000")
    out.update({"aggregate": agg, "cells": {f"{l}c{c}": st.mean([r for r in v if r == r]) if [r for r in v if r == r] else None
                                            for (l, c), v in per.items()}, "why": why})
    print("SERVED-VERDICT " + ("PASS" if not why else "FAIL: " + "; ".join(why)))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
