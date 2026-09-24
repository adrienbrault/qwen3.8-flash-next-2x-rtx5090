#!/usr/bin/env python3
"""stack-r3 multi-stream greedy identity probe (served path at c2 / c4 / c8), after rows32-r3's greedy_conc.py.

fn_greedy.py sends one request at a time, i.e. the served c1 shape. The stack's levers also run at 8 verify rows
(c2 d3: densegemm's gemv twin engages at <= 8 rows, R710b review) and 16 rows (c4 d3, c8 d1), where the served
process captures its own graphs. This probe runs CONC temperature-0 requests at once and records every full text:

  greedy_streams.py --url http://127.0.0.1:8022 --tag A1 --conc 2 4 8 --out gs.jsonl
  greedy_streams.py --compare --out gs.jsonl --ref A1 --aa A2 A3 --arms B1 B2 B3 --controls C

Prompts (fn_bench --unique --distinct style): each (conc, round, stream) gets its own salted ~1,500-token passage
plus a per-stream suffix, so no two requests of a run, and no two concurrency levels, share a cached prefix; every
arm sends byte-identical prompt sets. `min_tokens` = `max_tokens` = TOKENS keeps every stream in the batch for the
whole run (no stream leaves early and changes the batch composition). All CONC requests start behind a barrier.
Run it first after a boot (fresh page cache: R709c found served greedy is not self-reproducing under prefix reuse).

Comparison, per concurrency level L (rule rewritten 2026-09-24 after the R716b review; the old rule judged each arm
against the max of three A/A divergences from one reference boot and false-alarmed on 64 % of role assignments):
  A      the daily boots (--ref + --aa), all pairwise; rate = mean A/A divergent streams / n
  arms   the candidate boots (--arms), pooled as one group; --controls (flags-off image boots) are reported with the
         same statistics but never decide
  rate == 0         strict: every arm and control stream identical to the reference, no errors
  rate > --inconclusive (default 0.25)
                    INCONCLUSIVE, non-blocking: served greedy cannot adjudicate identity at L; the gate must take
                    identity at L from in-process OFF/U/OFF2 hash cells at every row count the draft policy reaches
  otherwise         pooled test: stat = mean(arm-A) - mean(A-A) pair divergence, exact label-permutation p over all
                    splits of A+arms; FAIL iff p <= 0.05 AND the arms' mean novel-output count (streams matching no
                    A boot, at equal reference count) exceeds the A boots' max; errors always FAIL
A boot with errors makes the level VOID. Last line: "GSTREAM-SUMMARY PASS|FAIL|INCONCLUSIVE|VOID ..."; exit 0 / 1 / 2 / 3.
INCONCLUSIVE is not a PASS: a gate must then check its in-process cells (R716c) before accepting.
"""
import argparse
import hashlib
import json
import random
import sys
import threading
import urllib.request

WORDS = ("the of and to in is was for on that with as by at from his her an were are which this be or had not but "
         "river castle engine signal garden harbor lantern market orbit pattern quarry ribbon saddle timber valley "
         "window yarrow zephyr anchor bridge cipher delta ember falcon glacier hollow island jasper kernel lumen "
         "meadow nectar oracle prism quiver raven summit thistle umber vessel willow").split()
TASKS = [
    "Summarise the passage above in five numbered points, then explain which point matters most and why.",
    "Write a Python function that parses the passage above into sentences and counts the words in each, with tests.",
    "Rewrite the passage above as a short formal report with a title, an introduction and a conclusion.",
    "List every place name you can find in the passage above and invent a one-line history for each.",
]


def prompt(conc, rnd, idx, salt):
    rng = random.Random(salt * 1_000_003 + conc * 10_007 + rnd * 101 + idx)
    body = " ".join(rng.choice(WORDS) for _ in range(1100))
    return (body + "\n\n" + TASKS[(idx + rnd) % len(TASKS)]
            + f"\n\n[variant c{conc}-r{rnd}-s{idx}: treat this document as section {idx} of series {conc}.{rnd}.]\n")


def gen(url, text, n, timeout):
    body = json.dumps(dict(model="flashnext", prompt=text, max_tokens=n, min_tokens=n, temperature=0.0, seed=0,
                           stream=False)).encode()
    req = urllib.request.Request(url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    c = d["choices"][0]
    return c.get("text", ""), c.get("finish_reason"), (d.get("usage") or {}).get("completion_tokens")


def run(a):
    out = open(a.out, "a")
    for conc in a.conc:
        for rnd in range(a.rounds):
            res = {}
            bar = threading.Barrier(conc)

            def one(i):
                p = prompt(conc, rnd, i, a.salt)
                try:
                    bar.wait(timeout=60)
                    res[i] = gen(a.url, p, a.tokens, a.timeout) + (None,)
                except Exception as e:  # noqa: BLE001 -- recorded; --compare counts it as a divergence
                    res[i] = (None, None, None, repr(e)[:200])
            th = [threading.Thread(target=one, args=(i,)) for i in range(conc)]
            for t in th:
                t.start()
            for t in th:
                t.join()
            for i in range(conc):
                text, fin, ntok, err = res[i]
                out.write(json.dumps(dict(tag=a.tag, conc=conc, round=rnd, stream=i, salt=a.salt, text=text,
                                          sha=hashlib.sha256(text.encode()).hexdigest() if text is not None else None,
                                          finish=fin, tokens=ntok, error=err)) + "\n")
            out.flush()
            ok = sum(1 for i in range(conc) if res[i][3] is None)
            print(f"[gstream] {a.tag} c{conc} round {rnd}: {ok}/{conc} ok, tokens "
                  f"{[res[i][2] for i in range(conc)]}", flush=True)


def compare(a):
    import itertools
    recs = {}
    for ln in open(a.out):
        d = json.loads(ln)
        recs.setdefault(d["tag"], {})[(d["conc"], d["round"], d["stream"])] = d
    if a.ref not in recs:
        print(f"GSTREAM-SUMMARY FAIL: reference {a.ref!r} not in {a.out} (have {sorted(recs)})")
        return 1
    ref = recs[a.ref]
    concs = sorted({k[0] for k in ref})
    A = [a.ref] + [t for t in a.aa if t in recs]
    missing = [t for t in a.aa + a.arms + a.controls if t not in recs]
    fail = bool(missing)
    if missing:
        print(f"GSTREAM missing boots {missing} -> FAIL")
    arms = [t for t in a.arms if t in recs]
    ctrls = [t for t in a.controls if t in recs]

    def text(tag, k):
        o = recs.get(tag, {}).get(k)
        return None if o is None else o.get("text")

    inconclusive, void = [], []
    for conc in concs:
        keys = sorted(k for k in ref if k[0] == conc)
        n = len(keys)
        errs = {t: sum(text(t, k) is None for k in keys) for t in A + arms + ctrls}
        div = lambda x, y: sum(text(x, k) is None or text(x, k) != text(y, k) for k in keys)
        def novel(t, refs):
            return sum(text(t, k) not in {text(r, k) for r in refs} for k in keys)
        def mean(v):
            return sum(v) / len(v) if v else 0.0
        aa = [div(x, y) for x, y in itertools.combinations(A, 2)]
        rate = mean(aa) / n if n else 0.0
        # novel outputs at equal reference count: each A vs the other A's; each arm vs every (len(A)-1)-subset of A
        nA = {t: novel(t, [r for r in A if r != t]) for t in A}
        sub = list(itertools.combinations(A, len(A) - 1)) if len(A) > 1 else [tuple(A)]
        nX = {t: mean([novel(t, s) for s in sub]) for t in arms + ctrls}
        def perm(group):
            pool = A + group
            def stat(g):
                h = [t for t in pool if t not in g]
                w = [div(x, y) for x, y in itertools.combinations(h, 2)] + [div(x, y) for x, y in itertools.combinations(g, 2)]
                return mean([div(x, y) for x in h for y in g]) - mean(w)
            obs = stat(group)
            splits = list(itertools.combinations(pool, len(group)))
            return obs, sum(stat(g) >= obs - 1e-9 for g in splits), len(splits)
        print(f"GSTREAM c{conc} n={n}: A/A pairs {aa} rate {rate:.2f}; novel A {nA}; novel arms/controls "
              f"{ {t: round(v, 2) for t, v in nX.items()} }")
        for t in arms + ctrls:
            print(f"GSTREAM {t} c{conc}: div vs A {[div(t, r) for r in A]} (A/A mean {mean(aa):.2f}) errors {errs[t]}"
                  f"{' [control]' if t in ctrls else ''}")
        err = any(errs[t] for t in arms)
        if any(errs[t] for t in A):
            void.append(conc); print(f"GSTREAM-LEVEL c{conc}: A boot errors {[t for t in A if errs[t]]} -> VOID")
        if any(errs[t] for t in ctrls):
            print(f"GSTREAM c{conc}: control errors {[t for t in ctrls if errs[t]]} (reported, not decisive)")
        if rate == 0:
            bad = [t for t in arms if div(t, a.ref) > 0]
            cbad = [t for t in ctrls if div(t, a.ref) > 0]
            if cbad:
                print(f"GSTREAM c{conc}: WARNING controls {cbad} diverge where the daily is deterministic (image-level effect?)")
            ok = not bad and not err
            print(f"GSTREAM-LEVEL c{conc}: strict (A/A identical) -> {'PASS' if ok else 'FAIL'}"
                  f"{' divergent ' + str(bad) if bad else ''}{' errors' if err else ''}")
        elif rate > a.inconclusive:
            ok = not err
            inconclusive.append(conc)
            extra = ""
            if arms and len(A) >= 2:
                o, k, m = perm(arms)
                extra = f"; pooled arms stat {o:+.2f} p {k}/{m}"
                if k / m <= 0.05:
                    extra += " WARNING p <= 0.05"
                if ctrls:
                    o, k, m = perm(ctrls)
                    extra += f"; controls stat {o:+.2f} p {k}/{m}"
            print(f"GSTREAM-LEVEL c{conc}: INCONCLUSIVE (A/A rate {rate:.2f} > {a.inconclusive}; identity must come from "
                  f"in-process cells at every row count reached){extra}{' -> FAIL (errors)' if err else ''}")
        else:
            o, k, m = perm(arms) if arms else (0.0, 1, 1)
            p = k / m
            shift = mean([nX[t] for t in arms]) > max(nA.values())
            ok = not err and not (p <= 0.05 and shift)
            c = ""
            if ctrls:
                co, ck, cm = perm(ctrls)
                c = f"; controls stat {co:+.2f} p {ck}/{cm}"
            print(f"GSTREAM-LEVEL c{conc}: POOLED (A/A rate {rate:.2f}) arms stat {o:+.2f} p {k}/{m}, novel shift "
                  f"{'yes' if shift else 'no'}{c} -> {'PASS' if ok else 'FAIL'}{' (errors)' if err else ''}")
        fail |= not ok
    verdict = "FAIL" if fail else "VOID" if void else "INCONCLUSIVE" if inconclusive else "PASS"
    print(f"GSTREAM-SUMMARY {verdict} ref {a.ref} aa {' '.join(a.aa) or '-'} arms {' '.join(a.arms)} "
          f"controls {' '.join(a.controls) or '-'} concs {concs} inconclusive {inconclusive} void {void}"
          + (" (identity at the inconclusive levels must come from in-process OFF/U/OFF2 cells at every row count)"
             if verdict == "INCONCLUSIVE" else ""))
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2, "VOID": 3}[verdict]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--tag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--rounds", type=int, default=2, help="prompt sets per concurrency level")
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--salt", type=int, default=20260924, help="fixed: every arm must send the same prompts")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--ref", default="A1")
    ap.add_argument("--aa", nargs="*", default=[])
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--controls", nargs="*", default=[], help="flags-off image boots: reported, never decide")
    ap.add_argument("--inconclusive", type=float, default=0.25, help="A/A divergence rate above which a level is INCONCLUSIVE")
    a = ap.parse_args()
    if a.compare:
        return compare(a)
    if not (a.url and a.tag):
        ap.error("--url and --tag are required unless --compare")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
