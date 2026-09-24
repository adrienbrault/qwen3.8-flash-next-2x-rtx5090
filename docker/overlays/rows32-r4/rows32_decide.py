#!/usr/bin/env python3
"""rows32 r4 gate analysis (pre-registered; rows32-r4-gate.sh calls it at the end, CPU only).

Reads the gate's results dir R:
  p1-verdicts.json        stack-r3's tools/p1_stack.py on the §16 shapes, arms OFF (= A env) and U (= B flags, B MAP)
  p1-<cN>-<ARM><r>/...    the rows32 shapes (c6 / c7 / c8): SRV (A env, depth 1) vs R32 (B env, depth 2), ON1
                          (B env, depth 1), the c8 depth-2 A/A matrix R32noQF / R32noDR / R32noMAP, OFF2 (A env, depth 2)
  records.jsonl           fn_bench (R707 instrument, --distinct) per boot: tags <boot>-c<N>-<kind>
  boot-<tag>.log          the launcher's "UP on ... | VRAM free MiB a/b/" line (headroom) and its "policy '...'" line
  env-<tag>.txt           the container's EXL3_* env
  container-<tag>.log     OOM / TORCH_CHECK / tracebacks / lcguard lines
  greedy-compare.txt      fn_greedy.py --compare (ref = A1; B1 and the A/A A2)
  gc-compare.txt          greedy_conc.py --compare ... --verdict (condition 2)
  free.txt                "<tag> <free MiB per card>" after the ramp (reported)

DECISION rows32 r4 (all must hold; the unit never promotes):
  condition 1  every kernel parity PASS (the gate stops before this script otherwise; re-checked from parity-*.log)
  condition 2  GREEDYC-VERDICT PASS: B's c8-vs-c1 divergences <= the worse A half's, early ones too, no errors
  condition 3  served per-stream decode B/A (median per boot and cell, mean of the two ABBA pairs):
                 c6 and c8, code and prose: mean >= 1.03 AND both pairs > 1.00 (same sign in both pairs)
                 c2-c5 (and c7), code and prose: mean >= 0.985;  c1: mean >= 0.98 (c1 noise, §16 c1 bound)
  usual gates  §16 P1 at c4d3 / c8d1 / c1d3: hashes identical in every pair and d0 cell, OFF deterministic, no
               same-sign regression at c4d3 / c8d1 on the stall-excluded or per-iterate-median read, c1d3 within the
               §16 c1 bound (the flags are inert at <= 16 rows: FLAT is the expected result, not a failure);
               rows32 P1: ON1 == SRV (<= 16 rows identical), R32 deterministic across rounds, the c8d2 A/A matrix
               (QF off, EXL3_DENSE_ROWS32 off, MAP without the 17-32 entry) identical to R32 in the same round;
               fn_greedy B1 vs ref 0 divergences (VOID, not REJECT, if the A/A A2 itself diverges);
               headroom: per card, min UP-line free over the B boots >= min over the A boots - 32 MiB;
               0 OOM / TORCH_CHECK / tracebacks in any boot; every boot's env and policy as intended.
Prints the tables, then "DECISION rows32 r4: ACCEPT|REJECT|VOID ..." as the last line. Exit 0 always (2 on unusable
input). --selftest builds synthetic results dirs and checks the verdict logic.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys
import tempfile

UP = re.compile(r"UP on \d+ \| warmup \S+ \| VRAM free MiB ([0-9/ ]+)")
POL = re.compile(r"policy '([^']*)'")
KINDS = ("code", "prose")


def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return None


def up_free(R, tag):
    m = UP.findall(read(f"{R}/boot-{tag}.log") or "")
    if not m:
        return None
    v = [int(x) for x in m[-1].replace(" ", "").split("/") if x]
    return v if len(v) >= 2 else None


def served(R, pairs):
    dec = {}
    for ln in open(f"{R}/records.jsonl"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("ok") and r.get("decode_tps") and r.get("tag"):
            dec.setdefault(r["tag"], []).append(r["decode_tps"])
    med = lambda t: st.median(dec[t]) if dec.get(t) else None
    table, missing = {}, []
    for kind in KINDS:
        for c in range(1, 9):
            vals = [(med(f"{a}-c{c}-{kind}"), med(f"{b}-c{c}-{kind}")) for a, b in pairs]
            if any(x is None or y is None for x, y in vals):
                missing.append(f"c{c} {kind}")
                continue
            pr = [y / x for x, y in vals]
            table[(c, kind)] = dict(a=[x for x, _ in vals], b=[y for _, y in vals], pairs=pr, mean=st.mean(pr))
    return table, missing


def served_verdict(table, missing, gain_cells=(6, 8), gain_bar=1.03, keep_cells=(2, 3, 4, 5, 7), keep_bar=0.985,
                   c1_bar=0.98):
    fails = [f"served cells missing: {', '.join(missing)}"] if missing else []
    for (c, kind), t in sorted(table.items()):
        if c in gain_cells:
            if t["mean"] < gain_bar or min(t["pairs"]) <= 1.0:
                fails.append(f"c{c} {kind} B/A {t['mean']:.3f} (pairs {' '.join(f'{p:.3f}' for p in t['pairs'])}): "
                             f"needs mean >= {gain_bar} and both pairs > 1")
        elif c in keep_cells:
            if t["mean"] < keep_bar:
                fails.append(f"c{c} {kind} B/A {t['mean']:.3f} < {keep_bar}")
        elif c == 1 and t["mean"] < c1_bar:
            fails.append(f"c1 {kind} B/A {t['mean']:.3f} < {c1_bar} (c1 noise ~20 %: read the P1 c1d3 bound)")
    for c in gain_cells:
        for kind in KINDS:
            if (c, kind) not in table and f"c{c} {kind}" not in " ".join(missing):
                fails.append(f"gain cell c{c} {kind} missing")
    return fails


def p1_16(R):
    """§16 shapes from p1_stack.py's JSON: identity + no regression; FLAT expected (flags inert at <= 16 rows)."""
    try:
        j = json.load(open(f"{R}/p1-verdicts.json"))
    except Exception as e:
        return [f"§16 P1: no p1-verdicts.json ({e})"], "missing"
    fails = []
    idt = j.get("identity", {}).get("U", {})
    if not idt.get("ok"):
        fails.append(f"§16 P1 hashes: {idt.get('differ')} DIFFER, {idt.get('missing')} missing")
    if not j.get("off_deterministic"):
        fails.append("§16 P1: OFF hashes differ across rounds")
    S = j.get("comparisons", {}).get("U-OFF", {})
    notes = []
    for sh in ("c4d3", "c8d1", "c1d3"):
        c = S.get(sh)
        if not c or c.get("excl", {}).get("n", 0) < 5:
            fails.append(f"§16 P1 {sh}: fewer than 5 pairs")
            continue
        notes.append(f"{sh} excl {c['excl']['mean']:+.3f} ms ({c['excl']['pct']:+.2f} %) {c['excl']['sign']}, "
                     f"med {c['med']['sign']}")
        for k in ("excl", "med"):
            if c[k]["sign"] == "REGRESSION":
                if sh in ("c4d3", "c8d1"):
                    fails.append(f"§16 P1 {sh} same-sign regression on {k} ({c[k]['pct']:+.2f} %)")
                elif c[k]["pct"] is None or c[k]["pct"] > 2.0:
                    fails.append(f"§16 P1 c1d3 regression on {k} {c[k]['pct']:+.2f} % > 2 % (§16 c1 bound)")
    return fails, "; ".join(notes)


def hashes(R, tag, b, d):
    return read(f"{R}/p1-{tag}/ctx4096_b{b}_d{d}/sequence-hashes.json")


def ms(R, tag, b, d):
    t = read(f"{R}/p1-{tag}/ctx4096_b{b}_d{d}/kernels.txt") or ""
    m = re.search(r"([0-9.]+) ms/iterate", t)
    return float(m.group(1)) if m else None


def tps(R, tag, b, d):
    try:
        k = json.load(open(f"{R}/p1-{tag}/ctx4096_b{b}_d{d}/kernels.json"))["capture"]
        return st.mean(k["streamed_during_per_job"]) / (k["elapsed_ms"] / 1000)
    except Exception:
        return None


def p1_rows32(R, rounds, aa_rounds=(1,)):
    fails, lines = [], []
    for b in (6, 7, 8):
        rat, step = [], []
        for r in range(1, rounds + 1):
            s, x = tps(R, f"c{b}-SRV{r}", b, 1), tps(R, f"c{b}-R32{r}", b, 2)
            if s and x:
                rat.append(x / s)
            m1, m2 = ms(R, f"c{b}-SRV{r}", b, 1), ms(R, f"c{b}-R32{r}", b, 2)
            if m1 and m2:
                step.append(m2 / m1)
            h1, hr = hashes(R, f"c{b}-R32{r}", b, 2), hashes(R, f"c{b}-R321", b, 2)
            if h1 is None:
                fails.append(f"c{b}d2 R32 round {r}: no hashes")
            elif r > 1 and h1 != hr:
                fails.append(f"c{b}d2 R32 round {r} != round 1 (not deterministic above 16 rows)")
        for r in (1, 3):
            if r <= rounds:
                a_, o_ = hashes(R, f"c{b}-SRV{r}", b, 1), hashes(R, f"c{b}-ON1{r}", b, 1)
                if a_ is None or o_ is None or a_ != o_:
                    fails.append(f"c{b}d1 ON1 != SRV round {r}")
        lines.append(f"P1 rows32 c{b}: per-stream tok/s R32(d2)/SRV(d1) " + " ".join(f"{v:.3f}" for v in rat)
                     + (f" median {st.median(rat):.3f}" if rat else "") + "; step ms d2/d1 "
                     + " ".join(f"{v:.3f}" for v in step) + (f" median {st.median(step):.3f}" if step else "")
                     + " (diagnostic: harness prompt, not production acceptance)")
    for r in aa_rounds:
        ref = hashes(R, f"c8-R32{r}", 8, 2)
        for arm, what in (("R32noQF", "QF off (the lcguard side path at 24 rows)"),
                          ("R32noDR", "EXL3_DENSE_ROWS32 off (the dense rows32 twins at model level)"),
                          ("R32noMAP", "MAP without 17-32:2 (mode 3 falls back to 2 by itself)")):
            h = hashes(R, f"c8-{arm}{r}", 8, 2)
            ok = h is not None and ref is not None and h == ref
            lines.append(f"P1 rows32 c8d2 A/A round {r} {arm} vs R32: {'IDENTICAL' if ok else 'DIFFER/MISSING'} ({what})")
            if not ok:
                fails.append(f"c8d2 {arm} != R32 round {r} ({what})")
    o2, r2 = ms(R, "c8-OFF21", 8, 2), ms(R, "c8-R321", 8, 2)
    if o2 and r2:
        lines.append(f"P1 rows32 c8d2 mechanism: flags off (generic > 16-row path) {o2:.2f} ms/iterate vs on {r2:.2f} "
                     f"({r2 / o2:.3f})")
    return fails, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--r32-rounds", type=int, default=4)
    ap.add_argument("--a", default="A1 A2")
    ap.add_argument("--b", default="B1 B2")
    ap.add_argument("--policy-a", default="[[4, 3], [5, 2], [8, 1]]")
    ap.add_argument("--policy-b", default="[[4, 3], [8, 2]]")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    R = a.results
    A, B = a.a.split(), a.b.split()
    fails, void = [], []

    # condition 1: parity logs (the gate stops on a FAIL; re-read so the decision line carries it)
    # each suite has its own success line (R717 review: cpuflags ends "CPU TESTS PASS", never "PARITY PASS")
    for p in sorted(glob.glob(f"{R}/parity-*.log")):
        t = read(p) or ""
        want = "CPU TESTS PASS" if os.path.basename(p) == "parity-cpuflags.log" else "PARITY PASS"
        if want not in t:
            fails.append(f"{os.path.basename(p)}: no {want}")
    # usual: §16 P1 and rows32 P1
    f16, n16 = p1_16(R)
    print(f"P1-S16: {n16}")
    fails += f16
    f32, l32 = p1_rows32(R, a.r32_rounds)
    for ln in l32:
        print(ln)
    fails += f32
    # condition 3: served
    table, missing = served(R, list(zip(A, B)))
    for kind in KINDS:
        for c in range(1, 9):
            t = table.get((c, kind))
            if t:
                print(f"SERVED c{c} {kind:5s}: A " + " ".join(f"{v:6.1f}" for v in t["a"]) + "  B "
                      + " ".join(f"{v:6.1f}" for v in t["b"]) + "  B/A pairs " + " ".join(f"{p:.3f}" for p in t["pairs"])
                      + f"  mean {t['mean']:.3f}")
    fails += served_verdict(table, missing)
    # headroom (UP line) + post-ramp report
    ups = {t: up_free(R, t) for t in A + B}
    if any(v is None for v in ups.values()):
        fails.append(f"headroom: UP line missing for {[t for t, v in ups.items() if v is None]}")
    else:
        for g in range(len(ups[A[0]])):
            amin, bmin = min(ups[t][g] for t in A), min(ups[t][g] for t in B)
            print(f"HEADROOM card {g}: UP-line free min A {amin} MiB, min B {bmin} MiB, B - A {bmin - amin:+d} MiB "
                  f"(bar: >= -32)")
            if bmin < amin - 32:
                fails.append(f"headroom card {g}: B {bmin} < A {amin} - 32 MiB")
    fr = {}
    for ln in (read(f"{R}/free.txt") or "").splitlines():
        p = ln.split()
        if len(p) > 1 and p[1] != "NOBOOT":
            fr[p[0]] = [int(x) for x in p[1:] if x.lstrip("-").isdigit()]
    if all(t in fr for t in A + B) and fr[A[0]]:
        print("POSTRAMP free MiB (report): " + ", ".join(f"{t} {fr[t]}" for t in A + B))
    # safety + env + policy
    for t in A + B:
        log = read(f"{R}/container-{t}.log")
        if log is None:
            fails.append(f"container {t}: no log")
            continue
        n_oom = len(re.findall(r"OutOfMemoryError|out of memory", log))
        n_tc = len(re.findall(r"TORCH_CHECK|c10::Error", log))
        n_tb = log.count("Traceback")
        side = re.findall(r"densegemm lcguard: device \d+: side call .*", log)
        print(f"CONTAINER {t}: OOM {n_oom}, TORCH_CHECK {n_tc}, tracebacks {n_tb}, lcguard lines "
              f"{len(re.findall(r'densegemm lcguard: device', log))}, side-slot refusals {len(side)}"
              + (f" ({side[0][:160]})" if side else ""))
        if n_oom or n_tc or n_tb:
            fails.append(f"container {t}: OOM {n_oom}, TORCH_CHECK {n_tc}, tracebacks {n_tb}")
        env = (read(f"{R}/env-{t}.txt") or "")
        r32 = [k for k in ("EXL3_DENSE_ROWS32=1", "EXL3_MOE_COOP_ROWS32=1", "EXL3_SHARED_EXPERT_ROWS32=1")
               if re.search("^" + re.escape(k) + "$", env, re.M)]
        want = 3 if t in B else 0
        if len(r32) != want:
            fails.append(f"env {t}: {len(r32)} rows32 flags (want {want})")
        if re.search(r"^EXL3_NVME_TIER=", env, re.M):
            fails.append(f"env {t}: the NVMe tier is on (NVME_TIER= did not reach the launcher)")
        pol = POL.findall(read(f"{R}/boot-{t}.log") or "")
        wantp = a.policy_b if t in B else a.policy_a
        if not pol or pol[-1] != wantp:
            fails.append(f"boot {t}: policy {pol[-1] if pol else 'missing'} != {wantp}")
    # fn_greedy (c1 identity)
    g = read(f"{R}/greedy-compare.txt") or ""
    gl = {m.group(1): int(m.group(2)) if m.group(2) else 0
          for m in re.finditer(r"^GREEDY (\S+) vs ref: \d+ identical, (?:IDENTICAL|(\d+) DIVERGENT)", g, re.M)}
    print("FN_GREEDY: " + ", ".join(f"{k} {v} divergent" for k, v in sorted(gl.items())))
    if "B1" not in gl:
        fails.append("fn_greedy B1 missing")
    elif gl["B1"]:
        if gl.get("A2", 0):
            void.append(f"fn_greedy: B1 {gl['B1']} and the A/A A2 {gl['A2']} diverge from ref (instrument)")
        else:
            fails.append(f"fn_greedy B1 vs ref: {gl['B1']} divergent (A/A A2 identical)")
    # condition 2
    gc = read(f"{R}/gc-compare.txt") or ""
    for ln in gc.splitlines():
        if ln.startswith("GREEDYC"):
            print(ln[:400])
    v = re.findall(r"^GREEDYC-VERDICT (.*)$", gc, re.M)
    if not v or not v[-1].startswith("PASS"):
        fails.append("condition 2 (greedy_conc c8-vs-c1): " + (v[-1][:300] if v else "no verdict"))
    if fails:
        print("DECISION rows32 r4: REJECT: " + "; ".join(fails))
    elif void:
        print("DECISION rows32 r4: VOID (re-run the served steps): " + "; ".join(void))
    else:
        c6 = " ".join(f"{k} {table[(6, k)]['mean']:.3f}" for k in KINDS)
        c8 = " ".join(f"{k} {table[(8, k)]['mean']:.3f}" for k in KINDS)
        print(f"DECISION rows32 r4: ACCEPT (conditions 1-3 and the usual gates hold; c6 {c6}, c8 {c8}); promotion is "
              f"the operator's step (the unit never promotes)")
    return 0


def selftest():
    ok = True

    def mk(R, bvals, greedy_b1=0, gc_pass=True, up_b=(100, 900), pol_b="[[4, 3], [8, 2]]"):
        os.makedirs(R, exist_ok=True)
        with open(f"{R}/records.jsonl", "w") as f:
            for tag, scale in (("A1", 1.0), ("A2", 1.0), ("B1", None), ("B2", None)):
                for c in range(1, 9):
                    for k in KINDS:
                        v = 100.0 * (bvals.get(c, 1.0) if scale is None else scale)
                        for _ in range(3):
                            f.write(json.dumps(dict(tag=f"{tag}-c{c}-{k}", ok=True, decode_tps=v)) + "\n")
        for t in ("A1", "A2", "B1", "B2"):
            up = up_b if t.startswith("B") else (110, 900)
            pol = pol_b if t.startswith("B") else "[[4, 3], [5, 2], [8, 1]]"
            open(f"{R}/boot-{t}.log", "w").write(f"x policy '{pol}' y\nUP on 8022 | warmup 30s | VRAM free MiB {up[0]}/{up[1]}/\n")
            open(f"{R}/container-{t}.log", "w").write("ok\n")
            env = "EXL3_MOE_COOP_V2=1\n" + ("EXL3_DENSE_ROWS32=1\nEXL3_MOE_COOP_ROWS32=1\nEXL3_SHARED_EXPERT_ROWS32=1\n"
                                            if t.startswith("B") else "")
            open(f"{R}/env-{t}.txt", "w").write(env)
        open(f"{R}/greedy-compare.txt", "w").write(
            f"GREEDY A2 vs ref: 6 identical, IDENTICAL\nGREEDY B1 vs ref: {6 - greedy_b1} identical, "
            + ("IDENTICAL" if not greedy_b1 else f"{greedy_b1} DIVERGENT") + "\n")
        open(f"{R}/gc-compare.txt", "w").write("GREEDYC-VERDICT " + ("PASS | x" if gc_pass else "FAIL: B divergences 9/64 > worst A half 2 x 2 | x") + "\n")
        S = {sh: {"excl": {"n": 6, "mean": 0.01, "pct": 0.05, "sign": "flat"},
                  "med": {"n": 6, "mean": 0.0, "pct": 0.0, "sign": "flat"}} for sh in ("c4d3", "c8d1", "c1d3")}
        json.dump({"identity": {"U": {"ok": True}}, "off_deterministic": True, "comparisons": {"U-OFF": S}},
                  open(f"{R}/p1-verdicts.json", "w"))
        for b in (6, 7, 8):
            for r in range(1, 5):
                for arm, d in (("SRV", 1), ("R32", 2), ("ON1", 1)):
                    p = f"{R}/p1-c{b}-{arm}{r}/ctx4096_b{b}_d{d}"
                    os.makedirs(p, exist_ok=True)
                    open(f"{p}/sequence-hashes.json", "w").write(f"h-{b}-{d}")
                    open(f"{p}/kernels.txt", "w").write("20.0 ms/iterate")
        for arm in ("R32noQF", "R32noDR", "R32noMAP"):
            p = f"{R}/p1-c8-{arm}1/ctx4096_b8_d2"
            os.makedirs(p, exist_ok=True)
            open(f"{p}/sequence-hashes.json", "w").write("h-8-2")
        open(f"{R}/parity-rows32.log", "w").write("PARITY PASS\n")

    def run(R):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sys.argv = ["x", "--results", R]
            main()
        return buf.getvalue().strip().splitlines()[-1]

    good = {6: 1.06, 7: 1.04, 8: 1.05}
    cases = [("pass", dict(bvals=good), "ACCEPT"),
             ("c8 below bar", dict(bvals={6: 1.06, 8: 1.02}), "REJECT"),
             ("c3 loss", dict(bvals={**good, 3: 0.98}), "REJECT"),
             ("c1 small loss ok", dict(bvals={**good, 1: 0.985}), "ACCEPT"),
             ("headroom", dict(bvals=good, up_b=(70, 900)), "REJECT"),
             ("greedy_conc", dict(bvals=good, gc_pass=False), "REJECT"),
             ("fn_greedy", dict(bvals=good, greedy_b1=1), "REJECT"),
             ("policy", dict(bvals=good, pol_b="[[4, 3], [5, 2], [8, 1]]"), "REJECT")]
    with tempfile.TemporaryDirectory() as td:
        for name, kw, want in cases:
            R = os.path.join(td, name.replace(" ", "_"))
            mk(R, **kw)
            got = run(R)
            good_ = got.startswith(f"DECISION rows32 r4: {want}")
            ok &= good_
            print(f"selftest {name:18s}: {'ok' if good_ else 'WRONG'}  {got[:150]}")
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
