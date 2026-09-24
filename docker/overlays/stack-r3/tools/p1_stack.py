#!/usr/bin/env python3
"""stack-r3 in-process P1 analysis (OPERATIONS §16 + the R705 / R709c / R712 review rules), pre-registered.

Input: one results dir with runs p1-<shape>-<arm><round>/ctx4096_b<b>_d<d>/{kernels.txt, events.json,
sequence-hashes.json} for shapes c1d3 (b1 d3), c4d3 (b4 d3), c8d1 (b8 d1) and their draft-0 cells (d0).

Three reads of every paired delta (arm - reference, same round, ms/iterate; negative = faster):
  raw     the harness mean (kernels.txt "ms/iterate")
  med     per-run median of events.json iterate_call_ms, differenced
  excl    stall-excluded paired mean: iterates where EITHER run of the pair is > its own median + 8 ms are dropped
          from both runs (R712 review: ~1/3 of runs carry one ~34 ms iterate at a fixed index), dropped count reported
Sign of a read over n pairs: GAIN = all < 0, REGRESSION = all > 0, else flat.

Comparisons and verdicts:
  U vs OFF       the union. PASS when: hashes identical everywhere (below); no REGRESSION at c4d3 or c8d1 on the excl
                 OR med read; c1d3: a REGRESSION on excl or med passes only if both its excl-mean and med-mean are
                 <= 2 % of OFF and a c4d3/c8d1 GAIN (excl) is larger in ms (§16 c1 exception, recorded as c1 cost);
                 gain clause: at least one shape GAIN on the excl read. Raw is reported, not decisive (the stall).
  OFF2 vs OFF    in-run A/A: reported (mean, sign); a GAIN or REGRESSION here says the rotation/position is biased
  U<no X> vs OFF the fallback union without X, judged like the union ("P1-VERDICT UNION-<X>")
  U vs U<no X>   component X's marginal inside the union, for every optional X. KEEP unless a REGRESSION (excl or med)
                 at c4d3/c8d1, or a c1d3 REGRESSION outside the c1 bound; the shapes where it GAINs are listed. A flat
                 marginal is KEEP (the served gate admits carried candidates; §16 R712 review Q5).
Identity: every arm's sequence-hashes.json equals the round's OFF in the d and d0 cell of every round and shape, and
OFF's own hashes are identical across rounds. Any DIFFER or MISSING fails the arm (and the union if the arm is U).

  p1_stack.py --results R --arms "OFF U UnoQT UnoQF UnoDG OFF2" --rounds 6 [--json out.json]
  (components: every arm named Uno<X> is the union without X)
Exit 0 always (the gate reads the verdict lines / JSON); 2 on unusable input.
"""
import argparse
import json
import os
import re
import statistics as st
import sys

SHAPES = {"c1d3": (1, 3), "c4d3": (4, 3), "c8d1": (8, 1)}
STALL_MS = 8.0


def load_run(R, shape, arm, rnd, d):
    b = SHAPES[shape][0]
    p = f"{R}/p1-{shape}-{arm}{rnd}/ctx4096_b{b}_d{d}"
    out = {"path": p}
    try:
        t = open(p + "/kernels.txt").read()
        out["raw"] = float(re.search(r"([0-9.]+) ms/iterate", t).group(1))
    except Exception:
        out["raw"] = None
    try:
        ev = json.load(open(p + "/events.json"))["iterations"]
        out["it"] = [float(x["iterate_call_ms"]) for x in ev]
    except Exception:
        out["it"] = None
    try:
        out["hash"] = open(p + "/sequence-hashes.json").read()
    except Exception:
        out["hash"] = None
    return out


def sign(ds):
    if not ds:
        return "none"
    return "GAIN" if all(v < 0 for v in ds) else "REGRESSION" if all(v > 0 for v in ds) else "flat"


def paired(runs, arm, ref, shape, d, rounds):
    raw, med, excl, dropped, base_raw, base_med = [], [], [], 0, [], []
    for r in range(1, rounds + 1):
        x, y = runs.get((shape, arm, r, d)), runs.get((shape, ref, r, d))
        if not x or not y:
            continue
        if x["raw"] is not None and y["raw"] is not None:
            raw.append(x["raw"] - y["raw"]); base_raw.append(y["raw"])
        if x["it"] and y["it"] and len(x["it"]) == len(y["it"]):
            mx, my = st.median(x["it"]), st.median(y["it"])
            med.append(mx - my); base_med.append(my)
            keep = [i for i in range(len(x["it"])) if x["it"][i] <= mx + STALL_MS and y["it"][i] <= my + STALL_MS]
            dropped += len(x["it"]) - len(keep)
            excl.append(st.mean(x["it"][i] for i in keep) - st.mean(y["it"][i] for i in keep))

    def summ(ds, base):
        if not ds:
            return {"n": 0, "sign": "none"}
        m = st.mean(ds)
        return {"n": len(ds), "deltas": [round(v, 4) for v in ds], "mean": m, "median": st.median(ds),
                "pct": 100 * m / st.mean(base) if base else None, "neg": sum(v < 0 for v in ds), "sign": sign(ds)}
    return {"raw": summ(raw, base_raw), "med": summ(med, base_med), "excl": summ(excl, base_med), "dropped": dropped}


def fmt(c):
    def one(k):
        s = c[k]
        if not s["n"]:
            return f"{k} n/a"
        return f"{k} {s['mean']:+.3f} ({s['pct']:+.2f} %) {s['neg']}/{s['n']} {s['sign']}"
    return f"{one('excl')} | {one('med')} | {one('raw')} | dropped {c['dropped']}"


def c1_ok(S, gains_ms):
    """§16 c1 exception on the excl AND med reads."""
    c1 = S["c1d3"]
    for k in ("excl", "med"):
        if c1[k]["sign"] == "REGRESSION" and not (c1[k]["pct"] is not None and c1[k]["pct"] <= 2.0 and gains_ms > c1[k]["mean"]):
            return False, f"c1d3 {k} regression {c1[k]['pct']:+.2f} % ({c1[k]['mean']:+.3f} ms) not within 2 % and covered by c4/c8 gains ({gains_ms:.3f} ms)"
    cost = [f"{k} {c1[k]['pct']:+.2f} %" for k in ("excl", "med") if c1[k]["sign"] == "REGRESSION"]
    return True, ("c1d3 cost " + ", ".join(cost) + " (record in STACK.md)") if cost else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    R, arms, N = a.results, a.arms.split(), a.rounds
    if "OFF" not in arms or "U" not in arms:
        print("P1: arms must include OFF and U"); sys.exit(2)
    runs = {}
    for sh, (b, d) in SHAPES.items():
        for arm in arms:
            for r in range(1, N + 1):
                for dd in (d, 0):
                    runs[(sh, arm, r, dd)] = load_run(R, sh, arm, r, dd)
    # identity
    ident = {}
    for arm in arms:
        if arm == "OFF":
            continue
        same = differ = missing = 0
        for sh, (b, d) in SHAPES.items():
            for r in range(1, N + 1):
                for dd in (d, 0):
                    h0, h1 = runs[(sh, "OFF", r, dd)]["hash"], runs[(sh, arm, r, dd)]["hash"]
                    if h0 is None or h1 is None:
                        missing += 1
                    elif h0 == h1:
                        same += 1
                    else:
                        differ += 1
        ident[arm] = {"identical": same, "differ": differ, "missing": missing, "ok": differ == 0 and missing == 0 and same == 6 * N}
        print(f"P1-HASH {arm} vs OFF: {same} identical, {differ} DIFFER, {missing} missing (need {6 * N})")
    off_det = all(len({runs[(sh, 'OFF', r, dd)]['hash'] for r in range(1, N + 1)} - {None}) == 1
                  for sh, (b, d) in SHAPES.items() for dd in (d, 0))
    print(f"P1-HASH OFF across rounds: {'deterministic' if off_det else 'NOT deterministic'}")

    out = {"identity": ident, "off_deterministic": off_det, "comparisons": {}, "verdicts": {}}
    comps = ([("U", "OFF")] + ([("OFF2", "OFF")] if "OFF2" in arms else []) + [("U", x) for x in arms if x.startswith("Uno")]
             + [(x, "OFF") for x in arms if x.startswith("Uno")])   # the fallback unions (UNION minus one component)
    for arm, ref in comps:
        key = f"{arm}-{ref}"
        S = {}
        for sh, (b, d) in SHAPES.items():
            S[sh] = paired(runs, arm, ref, sh, d, N)
            S[sh + "_d0"] = paired(runs, arm, ref, sh, 0, N)
            print(f"P1 {key} {sh}: {fmt(S[sh])}")
            print(f"P1 {key} {sh} d0(b{b}): {fmt(S[sh + '_d0'])}")
        out["comparisons"][key] = S
        # verdicts
        why, gains = [], [sh for sh in SHAPES if S[sh]["excl"]["sign"] == "GAIN"]
        if any(S[sh]["excl"]["n"] < 5 for sh in SHAPES):
            why.append("fewer than 5 pairs at some shape: " + ", ".join(f"{sh} n={S[sh]['excl']['n']}" for sh in SHAPES))
        for sh in ("c4d3", "c8d1"):
            for k in ("excl", "med"):
                if S[sh][k]["sign"] == "REGRESSION":
                    why.append(f"{sh} same-sign regression on {k} ({S[sh][k]['mean']:+.3f} ms, {S[sh][k]['pct']:+.2f} %)")
        gain_ms = -sum(S[sh]["excl"]["mean"] for sh in ("c4d3", "c8d1") if S[sh]["excl"]["sign"] == "GAIN")
        ok1, c1note = c1_ok(S, gain_ms)
        if not ok1:
            why.append(c1note)
        if arm == "OFF2":
            v = "A/A " + ("clean (flat at every shape)" if not gains and not any(S[sh][k]["sign"] == "REGRESSION" for sh in SHAPES for k in ("excl", "med"))
                          else "BIASED: " + ", ".join(f"{sh} {S[sh]['excl']['sign']}" for sh in SHAPES if S[sh]["excl"]["sign"] != "flat"))
            out["verdicts"]["AA"] = v
            print(f"P1-VERDICT A/A (OFF2 - OFF): {v}")
            continue
        if ref == "OFF":
            if not ident[arm]["ok"]:
                why.append(f"hashes: {ident[arm]['differ']} DIFFER, {ident[arm]['missing']} missing")
            if not off_det:
                why.append("OFF hashes differ across rounds (harness nondeterministic)")
            if not gains:
                v = "FLAT (no shape GAINs on the stall-excluded read)" + (f"; also {'; '.join(why)}" if why else "")
                verdict = "FAIL"
            elif why:
                v, verdict = "; ".join(why), "FAIL"
            else:
                v, verdict = "gain at " + ", ".join(gains) + (f"; {c1note}" if c1note else ""), "PASS"
            name = "U" if arm == "U" else arm            # Uno<X>: the union without X, vs OFF
            out["verdicts"][name] = {"verdict": verdict, "why": v, "gains": gains, "c1": c1note}
            print(f"P1-VERDICT {'UNION' if arm == 'U' else 'UNION-' + arm[3:]}: {verdict}: {v}")
        else:
            X = ref[3:]
            if not ident[ref]["ok"]:
                why.append(f"hashes of {ref}: {ident[ref]['differ']} DIFFER, {ident[ref]['missing']} missing")
            verdict = "DROP" if why else "KEEP"
            v = "; ".join(why) if why else ("marginal gain at " + ", ".join(gains) if gains else "flat marginal (no same-sign regression)") + (f"; {c1note}" if c1note else "")
            out["verdicts"][X] = {"verdict": verdict, "why": v, "gains": gains, "c1": c1note}
            print(f"P1-VERDICT {X} (U - Uno{X}): {verdict}: {v}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
