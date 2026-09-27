#!/usr/bin/env python3
"""R784 decision (pre-registered; this file is the rule, docs/PROMOTION.md "A rebase onto a new upstream engine" its prose).
Derived from r2/r741_decide.py: the P rule with the pool clause moved to the user's floor (884,736), median cold
prefill over 3 salted samples per ctx, the S1/S2 A/A band and the row counts as VOID triggers, and the tiled-OFF fallback
(PNT) read as CANDIDATE-NOTILED. The EP rule is gone (R784 has no U arms).

  r784_decide.py --dir <results dir> [--floor 884736] [--lpool 983040] [--ref-up 1125/1573] [--band 0.009]
  r784_decide.py --selftest

Inputs written by r784-rebase-dev-r3.sh:
  boots.tsv          one row per boot (measured or search): tag, arm, pool, up_free, headroom, gen, oom, void, measured,
                     tb (tracebacks), restarts, mp_n / mp_fail (mp_decode records / failed), ...
  cmp-S-P.json       mp_decode.py compare --a S --b P --summary-json (paired geo-mean P/S per kind x conc, bootstrap CI)
  cmp-SA-SB.json     the same for S2 vs S1 (the unit relabels S1 -> SA1, S2 -> SB1: mp_decode pairs arms by tag prefix)
  prefill.jsonl      fn_bench cold prefill records, tag pf-<boot tag>-<ctx>, 3 per ctx per measured boot
  greedy-compare.txt fn_greedy.py --compare --ref S1 (line "GREEDY S2 vs S1: 6 identical, IDENTICAL")

Rule (P = tabbyapi:rebase-dev-r3, tiled HC prefill ON; S = the served image; ABBA S1 P1 P2 S2):
  pool     FOUND (P's measured pool) >= floor 884,736, P boots' per-card free at UP >= S1 - 32 MiB (headroom OK on both
           cards; the unit marks a search boot with an OOM / Traceback line after its 12k layout probe as no-fit)
  decode   code AND prose, c4 AND c8: paired geo-mean P/S >= 0.99 and CI lower bound >= 0.98 (c1 reported only)
  prefill  median P / median S >= 0.95 at 30k and at 120k (expectation >= 1.10 at 120k, reported)
  health   every S and P measured boot GEN_SANE, 0 OOM, 0 tracebacks, 0 restarts
  VOID     S2 vs S1 beyond 0.9 % (pooled code + prose) at c4, or S2 greedy not 6/6 identical to S1; S1 UP free outside the
           reference +- 32 MiB on either card; power not stock or clock offsets not core 0 / memory +4500 (boots.tsv void);
           mp_decode rows != 144 (24 code + 24 prose x c1/c4/c8) or any failed, or cold prefill rows != 3 per ctx
  -> CANDIDATE                every clause met (R785 promotes only this)
     CANDIDATE-NOTILED        the tiled-ON route fails the pool clause only, and PNT (tiled OFF) fits >= floor with
                              headroom OK, healthy, and its prefill >= 0.95x S at both ctx. Its decode is NOT measured
                              (PNT is a prefill-only arm): a follow-up must run the ABBA decode on it; R785 aborts on it
     NOT-A-CANDIDATE (<clause>)
     INCOMPLETE (missing ... | VOID: ...)   the plan's VOID is reported inside INCOMPLETE: the four outcomes are the contract
"""
import argparse
import csv
import json
import os
import statistics as st
import tempfile

MP_ROWS = 144          # 24 code + 24 prose prompts x concs 1 4 8
PF_ROWS = 3            # salted cold prefills per ctx per measured boot
CTXS = (30000, 120000)


def load_boots(d):
    p = os.path.join(d, "boots.tsv")
    if not os.path.isfile(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f, delimiter = "\t"))


def load_cmp(d, a, b):
    p = os.path.join(d, f"cmp-{a}-{b}.json")
    if not os.path.isfile(p):
        return None
    return json.load(open(p))


def prefill_rates(d):
    """(boot tag, ctx) -> [tok/s ...] over the successful cold prefill records"""
    out = {}
    p = os.path.join(d, "prefill.jsonl")
    if not os.path.isfile(p):
        return out
    for line in open(p):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        tag = r.get("tag", "")
        if not tag.startswith("pf-") or not r.get("ttft_s") or not r.get("prompt_tokens"):
            continue
        boot, ctx = tag[3:].rsplit("-", 1)
        out.setdefault((boot, int(ctx)), []).append(r["prompt_tokens"] / r["ttft_s"])
    return out


def arm_of(tag):
    return tag.rstrip("0123456789")


def row(cmp, kind, conc):
    for r in cmp or []:
        if r["kind"] == kind and r["conc"] == conc:
            return r
    return None


def fmt(r):
    if r is None:
        return "n/a"
    return f"{(r['ratio'] - 1) * 100:+.2f} % [{(r['ci_lo'] - 1) * 100:+.2f}, {(r['ci_hi'] - 1) * 100:+.2f}] (n {r['n']})"


def blank(v):
    return v in (None, "", "-")


def free_pair(s):
    try:
        a, b = s.split("/")
        return int(a), int(b)
    except (AttributeError, ValueError):
        return None


def read_first(d, name, prefix):
    p = os.path.join(d, name)
    if not os.path.isfile(p):
        return None
    for line in open(p, errors = "replace"):
        if line.startswith(prefix):
            return line.rstrip("\n")
    return None


def decide(d, floor, lpool, ref_up, band):
    lines = []
    boots = load_boots(d)
    measured = [b for b in boots if b.get("measured") == "1"]
    by_arm = {}
    for b in measured:
        by_arm.setdefault(b["arm"], []).append(b)
    S, P, PNT = by_arm.get("S", []), by_arm.get("P", []), by_arm.get("PNT", [])
    lines.append("boots: " + ", ".join(
        f"{b['tag']} pool {b['pool']} [{b.get('split', '-')}] free {b['up_free']} post {b.get('post_free', '-')} headroom "
        f"{b['headroom']} {b['gen']} oom {b['oom']} tb {b.get('tb', '?')}{' VOID ' + b['void'] if not blank(b['void']) else ''}"
        for b in measured))
    why, missing, void = [], [], []
    pf = prefill_rates(d)

    # ---------------- VOID ----------------
    for b in S + P + PNT:
        if not blank(b["void"]):
            void.append(f"{b['tag']}: {b['void']}")
    s1 = next((b for b in S if b["tag"] == "S1"), None)
    if s1 is not None:
        up, ref = free_pair(s1["up_free"]), free_pair(ref_up)
        if up is None:
            void.append(f"S1 UP free unreadable ({s1['up_free']})")
        elif ref is not None and (abs(up[0] - ref[0]) > 32 or abs(up[1] - ref[1]) > 32):
            void.append(f"S1 UP free {up[0]}/{up[1]} outside the reference {ref[0]}/{ref[1]} +- 32 MiB")
    if any(b["tag"] == "S2" for b in S):
        aa = load_cmp(d, "SA", "SB")
        r = row(aa, "both", 4)
        if r is None:
            missing.append("cmp-SA-SB.json (S2 vs S1 A/A)")
        else:
            lines.append(f"A/A S2 vs S1 c4 (both): {fmt(r)}; band +-{band * 100:.1f} %")
            if abs(r["ratio"] - 1) > band:
                void.append(f"S2 vs S1 c4 {fmt(r)} beyond the +-{band * 100:.1f} % band")
        g = read_first(d, "greedy-compare.txt", "GREEDY S2 vs S1:")
        if g is None:
            missing.append("greedy S2 vs S1")
        elif not g.endswith("6 identical, IDENTICAL"):
            void.append(f"S2 greedy not 6/6 identical to S1 ({g})")
    for b in S + P:
        if b.get("mp_n") != str(MP_ROWS) or b.get("mp_fail") not in ("0",):
            void.append(f"{b['tag']} mp_decode rows {b.get('mp_n')} (failed {b.get('mp_fail')}), want {MP_ROWS} ok")
    for b in S + P + PNT:
        for ctx in CTXS:
            n = len(pf.get((b["tag"], ctx), []))
            if n != PF_ROWS:
                void.append(f"{b['tag']} cold prefill {ctx // 1000}k rows {n}, want {PF_ROWS}")

    # ---------------- pool ----------------
    pool_fail = []
    if P:
        found = int(P[0]["pool"])
        lines.append(f"P pool FOUND {found:,} [{P[0].get('split', '-')}] vs served {lpool:,} ({(found - lpool) // 16384:+d} steps, "
                     f"{(found - lpool) / lpool * 100:+.1f} %); floor {floor:,}")
        if found < floor:
            pool_fail.append(f"pool {found:,} < floor {floor:,}")
        for b in P:
            if b["headroom"] != "OK":
                pool_fail.append(f"{b['tag']} headroom {b['headroom']}")
    else:
        tried = [b for b in boots if b["arm"] == "P" and b.get("measured") != "1" and int(b["pool"]) >= floor
                 and not b["headroom"].startswith("COLD")]
        func = [b for b in tried if b["gen"] not in ("NO_BOOT", "GEN_SANE", "DIED")]
        if func:   # the unit stops the search on a functional failure (psearch rc 2): not a VRAM verdict
            why.append("P search: functional failure (" + ", ".join(f"{b['tag']} {b['gen']}" for b in func) + ")")
        elif tried and not any(b["headroom"] == "OK" and b["gen"] == "GEN_SANE" for b in tried):
            pool_fail.append(f"pool: no P boot fits at >= floor {floor:,} ({len(tried)} tried: "
                             + ", ".join(f"{b['tag']} {b['headroom'] if not blank(b['headroom']) else b['gen']}" for b in tried) + ")")
        else:
            missing.append("P pool (no measured P boot)")
    why += pool_fail

    # ---------------- decode ----------------
    cmp = load_cmp(d, "S", "P")
    if len(S) < 2 or len(P) < 2:
        missing.append(f"boots S {len(S)} / P {len(P)} (need 2 each, ABBA)")
    if cmp is None:
        missing.append("cmp-S-P.json")
    for conc in (1, 4, 8):
        lines.append(f"P vs S c{conc}: code {fmt(row(cmp, 'code', conc))}; prose {fmt(row(cmp, 'prose', conc))}; "
                     f"both {fmt(row(cmp, 'both', conc))}{'   (report only)' if conc == 1 else ''}")
    for conc in (4, 8):
        for kind in ("code", "prose"):
            r = row(cmp, kind, conc)
            if r is None:
                if cmp is not None:
                    missing.append(f"{kind} c{conc}")
            elif r["ratio"] < 0.99 or r["ci_lo"] < 0.98:
                why.append(f"decode {kind} c{conc} {fmt(r)} (bar >= -1 %, CI >= -2 %)")

    # ---------------- prefill ----------------
    def med(arm, ctx):
        v = [x for (b, c), xs in pf.items() if c == ctx and arm_of(b) == arm and b in {r["tag"] for r in by_arm.get(arm, [])}
             for x in xs]
        return (st.median(v), len(v)) if v else (None, 0)
    pnt_pf_ok = bool(PNT)
    for ctx in CTXS:
        ms, ns = med("S", ctx)
        mp, np_ = med("P", ctx)
        if ms is None:
            missing.append(f"prefill S {ctx // 1000}k")
            pnt_pf_ok = False
            continue
        if mp is not None:
            ratio = mp / ms
            exp = "   (expectation >= 1.10: " + ("met" if ratio >= 1.10 else "NOT met, reported") + ")" if ctx == 120000 else ""
            lines.append(f"P vs S cold prefill {ctx // 1000}k: median S {ms:.0f} / P {mp:.0f} tok/s = {ratio:.3f}x "
                         f"(S {ns} / P {np_} samples){exp}")
            if ratio < 0.95:
                why.append(f"prefill {ctx // 1000}k {ratio:.3f}x (bar 0.95x)")
        elif P:
            missing.append(f"prefill P {ctx // 1000}k")
        mn, nn = med("PNT", ctx)
        if mn is not None:
            lines.append(f"PNT (tiled OFF) vs S cold prefill {ctx // 1000}k: median {mn:.0f} tok/s = {mn / ms:.3f}x (report / NOTILED rule)")
            pnt_pf_ok = pnt_pf_ok and mn / ms >= 0.95
        else:
            pnt_pf_ok = False

    # ---------------- health ----------------
    for b in S + P:
        bad = []
        if b["gen"] != "GEN_SANE":
            bad.append(b["gen"])
        for k in ("oom", "tb", "restarts"):
            if b.get(k) not in ("0",):
                bad.append(f"{k} {b.get(k)}")
        if bad:
            why.append(f"{b['tag']} health: {', '.join(bad)}")

    # ---------------- report only ----------------
    for b in S + P + PNT:
        if not blank(b.get("post_free")):
            lines.append(f"report {b['tag']}: free after the 120k prefill {b['post_free']} MiB (UP {b['up_free']})")
    m = read_first(d, "margins-compare.txt", "MARGIN-SUMMARY")
    if m:
        lines.append(f"report F5 lp_margin (first server run, report only): {m}")
    f6 = os.path.join(d, "fid-compare.txt")
    if os.path.isfile(f6):
        lines.append("report F6 corpus fidelity (report only, bars UNCALIBRATED): see fid-compare.txt")

    # ---------------- verdict ----------------
    notiled = None
    if PNT:
        b = PNT[0]
        ok = (int(b["pool"]) >= floor and b["headroom"] == "OK" and b["gen"] == "GEN_SANE" and b["oom"] == "0"
              and b.get("tb") == "0" and pnt_pf_ok)
        lines.append(f"PNT pool {int(b['pool']):,} [{b.get('split', '-')}] headroom {b['headroom']} {b['gen']} "
                     f"-> NOTILED rule {'met' if ok else 'not met'} (decode not measured)")
        notiled = ok
    if void:
        verdict = "INCOMPLETE (VOID: " + "; ".join(void) + (("; also failing: " + "; ".join(why)) if why else "") + ")"
    elif why:
        only_pool = pool_fail and len(why) == len(pool_fail)
        if only_pool and notiled:
            verdict = ("CANDIDATE-NOTILED (tiled ON fails the pool only: " + "; ".join(pool_fail) + "; tiled OFF fits "
                       f"{int(PNT[0]['pool']):,} with prefill >= 0.95x S; decode NOT measured: a follow-up ABBA is needed, "
                       "R785 does not promote this)")
        else:
            verdict = "NOT-A-CANDIDATE (" + "; ".join(why) + ")"
    elif missing:
        verdict = "INCOMPLETE (missing " + ", ".join(missing) + ")"
    else:
        verdict = "CANDIDATE (every pre-registered clause met; R785 gates and promotes)"
    lines.append(f"RULE {verdict}")
    lines.append(f"DECISION: {verdict}")
    return lines


# ---------------------------------------------------------------- selftest
COLS = ["tag", "arm", "image", "pool", "tp", "env_n", "up_free", "post_free", "headroom", "power", "gpc", "mem", "gen", "oom",
        "void", "measured", "split", "layout", "tb", "restarts", "mp_n", "mp_fail"]


def _write_boots(d, rows):
    with open(os.path.join(d, "boots.tsv"), "w") as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows:
            base = {"image": "img", "tp": "false", "env_n": "41", "post_free": "300/800", "power": "600 575", "gpc": "0 0",
                    "mem": "4500 4500", "gen": "GEN_SANE", "oom": "0", "void": "-", "split": "30,30", "layout": "-",
                    "tb": "0", "restarts": "0", "mp_n": "144", "mp_fail": "0"}
            base.update(r)
            f.write("\t".join(str(base.get(c, "-")) for c in COLS) + "\n")


def _cmp(d, a, b, ratios, ci = 0.008):
    out = []
    for conc, rr in ratios.items():
        for kind in ("code", "prose", "both"):
            r = rr[kind] if isinstance(rr, dict) else rr
            out.append({"a": a, "b": b, "kind": kind, "conc": conc, "n": 24, "tps_a": 100, "tps_b": 100 * r,
                        "ratio": r, "ci_lo": r - ci, "ci_hi": r + ci, "short_a": 0, "short_b": 0})
    json.dump(out, open(os.path.join(d, f"cmp-{a}-{b}.json"), "w"))


def _prefill(d, rates, n = PF_ROWS):
    with open(os.path.join(d, "prefill.jsonl"), "w") as f:
        for boot, (r30, r120) in rates.items():
            for ctx, rate in ((30000, r30), (120000, r120)):
                for k in range(n):
                    f.write(json.dumps({"tag": f"pf-{boot}-{ctx}", "prompt_tokens": ctx, "ttft_s": ctx / (rate + k)}) + "\n")


def _greedy(d, ok = True):
    open(os.path.join(d, "greedy-compare.txt"), "w").write(
        "GREEDY P1 vs S1: 4 identical, 2 DIVERGENT\n" + ("GREEDY S2 vs S1: 6 identical, IDENTICAL\n" if ok else "GREEDY S2 vs S1: 5 identical, 1 DIVERGENT\n"))


def _abba(d, found = 901120):
    _write_boots(d, [
        {"tag": "S1", "arm": "S", "pool": 983040, "up_free": "1125/1573", "headroom": "REF", "measured": "1"},
        {"tag": "P@917504-s30-30", "arm": "P", "pool": 917504, "up_free": "1050/1600", "headroom": "BELOW(1050/1600 vs 1125/1573)",
         "measured": "0", "env_n": "42", "mp_n": "-", "mp_fail": "-", "post_free": "-"},
        {"tag": "P1", "arm": "P", "pool": found, "up_free": "1130/1600", "headroom": "OK", "measured": "1", "env_n": "42"},
        {"tag": "P2", "arm": "P", "pool": found, "up_free": "1130/1600", "headroom": "OK", "measured": "1", "env_n": "42"},
        {"tag": "S2", "arm": "S", "pool": 983040, "up_free": "1125/1573", "headroom": "OK", "measured": "1"}])
    _cmp(d, "S", "P", {1: 0.97, 4: 0.995, 8: 0.992})
    _cmp(d, "SA", "SB", {1: 1.01, 4: 1.004, 8: 0.999})
    _prefill(d, {"S1": (9000, 7000), "S2": (9100, 7050), "P1": (9900, 7900), "P2": (10000, 8000)})
    _greedy(d)


def _dec(d, **kw):
    a = dict(floor = 884736, lpool = 983040, ref_up = "1125/1573", band = 0.009)
    a.update(kw)
    return "\n".join(decide(d, **a))


def selftest():
    d = tempfile.mkdtemp()
    _abba(d)
    out = _dec(d)
    assert "DECISION: CANDIDATE (" in out, out
    assert "expectation >= 1.10: met" in out, out
    # decode miss at c4 code only
    _cmp(d, "S", "P", {1: 0.97, 4: {"code": 0.985, "prose": 0.995, "both": 0.99}, 8: 0.992})
    out = _dec(d)
    assert "DECISION: NOT-A-CANDIDATE (decode code c4" in out and "decode prose c4" not in out, out
    # CI lower bound below 0.98 with a passing mean
    _cmp(d, "S", "P", {1: 0.97, 4: 0.995, 8: 0.992}, ci = 0.02)
    out = _dec(d)
    assert "DECISION: NOT-A-CANDIDATE (decode code c4" in out, out
    _cmp(d, "S", "P", {1: 0.97, 4: 0.995, 8: 0.992})
    # prefill miss at 30k
    _prefill(d, {"S1": (9000, 7000), "S2": (9100, 7050), "P1": (8000, 7900), "P2": (8100, 8000)})
    out = _dec(d)
    assert "DECISION: NOT-A-CANDIDATE (prefill 30k" in out, out
    _prefill(d, {"S1": (9000, 7000), "S2": (9100, 7050), "P1": (9900, 7900), "P2": (10000, 8000)})
    # measured pool below the floor
    _abba(d, found = 868352)
    out = _dec(d)
    assert "DECISION: NOT-A-CANDIDATE (pool 868,352 < floor 884,736" in out, out
    # VOID: A/A beyond the band, S2 greedy divergent, S1 UP outside the reference, short rows
    _abba(d)
    _cmp(d, "SA", "SB", {1: 1.0, 4: 1.012, 8: 1.0})
    out = _dec(d)
    assert "DECISION: INCOMPLETE (VOID: S2 vs S1 c4" in out, out
    _abba(d)
    _greedy(d, ok = False)
    assert "S2 greedy not 6/6" in _dec(d)
    _abba(d)
    assert "S1 UP free 1125/1573 outside" in _dec(d, ref_up = "1200/1573")
    _abba(d)
    _prefill(d, {"S1": (9000, 7000), "S2": (9100, 7050), "P1": (9900, 7900)})
    _prefill(d, {"S1": (9000, 7000), "S2": (9100, 7050), "P1": (9900, 7900), "P2": (10000, 8000)}, n = 2)
    out = _dec(d)
    assert "DECISION: INCOMPLETE (VOID:" in out and "cold prefill 30k rows 2" in out, out
    _abba(d)
    rows = load_boots(d)
    rows[2]["mp_n"] = "140"
    _write_boots(d, rows)
    assert "P1 mp_decode rows 140" in _dec(d)
    # a void keeps failing clauses visible
    _abba(d, found = 868352)
    _cmp(d, "SA", "SB", {1: 1.0, 4: 1.02, 8: 1.0})
    out = _dec(d)
    assert "DECISION: INCOMPLETE (VOID:" in out and "also failing: pool 868,352" in out, out
    # missing: no cmp-S-P
    _abba(d)
    os.remove(os.path.join(d, "cmp-S-P.json"))
    assert "DECISION: INCOMPLETE (missing cmp-S-P.json" in _dec(d)
    # health: a traceback on P2
    _abba(d)
    rows = load_boots(d)
    rows[3]["tb"] = "1"
    _write_boots(d, rows)
    assert "DECISION: NOT-A-CANDIDATE (P2 health: tb 1)" in _dec(d)

    # tiled-ON search fails the floor, PNT (tiled OFF) fits: CANDIDATE-NOTILED; only S1 was measured
    d2 = tempfile.mkdtemp()
    _write_boots(d2, [
        {"tag": "S1", "arm": "S", "pool": 983040, "up_free": "1125/1573", "headroom": "REF", "measured": "1"},
        {"tag": "P@917504-s30-30", "arm": "P", "pool": 917504, "up_free": "900/1600", "headroom": "COLD(warmup 40.1s)",
         "measured": "0", "mp_n": "-", "mp_fail": "-"},
        {"tag": "P@917504-s30-30+w", "arm": "P", "pool": 917504, "up_free": "1000/1600", "headroom": "BELOW(1000/1600 vs 1125/1573)",
         "measured": "0", "mp_n": "-", "mp_fail": "-"},
        {"tag": "P@901120-s30-30", "arm": "P", "pool": 901120, "up_free": "1080/1600", "headroom": "OOM(2)", "measured": "0",
         "mp_n": "-", "mp_fail": "-"},
        {"tag": "P@884736-s30-30", "arm": "P", "pool": 884736, "up_free": "-", "headroom": "-", "gen": "NO_BOOT", "measured": "0",
         "mp_n": "-", "mp_fail": "-"},
        {"tag": "PNT1", "arm": "PNT", "pool": 950272, "up_free": "1130/1580", "headroom": "OK", "measured": "1",
         "mp_n": "-", "mp_fail": "-"}])
    _prefill(d2, {"S1": (9000, 7000), "PNT1": (9000, 7000)})
    out = _dec(d2)
    assert "DECISION: CANDIDATE-NOTILED (tiled ON fails the pool only: pool: no P boot fits at >= floor 884,736 (3 tried" in out, out
    _prefill(d2, {"S1": (9000, 7000), "PNT1": (8000, 7000)})   # PNT's prefill below 0.95x: plain NOT-A-CANDIDATE on pool
    out = _dec(d2)
    assert "DECISION: NOT-A-CANDIDATE (pool: no P boot fits" in out, out
    # a functional stop (wrong generation) is named as such, not as a pool verdict, and does not open NOTILED
    rows = load_boots(d2)
    rows[4]["gen"] = "GEN_GARBAGE"
    _write_boots(d2, rows)
    _prefill(d2, {"S1": (9000, 7000), "PNT1": (9000, 7000)})
    out = _dec(d2)
    assert "DECISION: NOT-A-CANDIDATE (P search: functional failure (P@884736-s30-30 GEN_GARBAGE)" in out, out
    # a fitting search row at the floor, not yet measured -> still open
    d3 = tempfile.mkdtemp()
    _write_boots(d3, [
        {"tag": "S1", "arm": "S", "pool": 983040, "up_free": "1125/1573", "headroom": "REF", "measured": "1"},
        {"tag": "P@884736-s30-30", "arm": "P", "pool": 884736, "up_free": "1130/1600", "headroom": "OK", "measured": "0",
         "mp_n": "-", "mp_fail": "-"}])
    _prefill(d3, {"S1": (9000, 7000)})
    out = _dec(d3)
    assert "DECISION: INCOMPLETE (missing" in out, out
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--floor", type = int, default = 884736)
    ap.add_argument("--lpool", type = int, default = 983040)
    ap.add_argument("--ref-up", default = "1125/1573", help = "S1's expected UP free per card (MiB), VOID outside +- 32")
    ap.add_argument("--band", type = float, default = 0.009, help = "S2 vs S1 c4 A/A band (docs/PROMOTION.md)")
    ap.add_argument("--selftest", action = "store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    for line in decide(a.dir, a.floor, a.lpool, a.ref_up, a.band):
        print(line)


if __name__ == "__main__":
    main()
