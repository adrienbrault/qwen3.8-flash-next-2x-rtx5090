#!/usr/bin/env python3
"""Concurrent greedy probe, rows32 r4 (r3's greedy_conc.py, rebuilt for the user's condition 2).

Condition 2 (user-approved promotion condition for rows32): the c8-vs-c1 greedy divergence rate of B (rows32 flags +
policy [[4, 3], [8, 2]]: 24 verify rows at 8 streams) must be no worse than the served config's own c8-vs-c1
divergence rate, measured the same way. Batching changes the floating-point order of many kernels, so an 8-stream
continuation may legitimately part from the 1-stream one at a near-tie late in the text; a broken 17-32-row path
parts early and everywhere (R670: acceptance 1.11 against 1.84 with garbage verify outputs).

Measured the same way for A and B:
  * two disjoint prompt sets P and Q, 32 prompts each (16 natural prompts, Q's carrying a fixed framing line, plus 16
    salted ~1,400-token passages with a task, per-set salt): no prompt of one set shares a prefix with the other;
  * every set is sent ONCE per boot, as the first requests after the boot (fresh cache: R709c found served greedy is
    not self-reproducing under prefix reuse), at one concurrency: batches of CONC started behind a barrier, with
    min_tokens = max_tokens = TOKENS so no stream leaves the batch early and the verify batch stays at CONC streams;
  * the c8 run of one boot is compared with the c1 run of ANOTHER boot on the same set (the c1 path is the same code
    in A and B: depth 3 at <= 4 streams, the rows32 flags inert at <= 16 rows; fn_greedy checks that separately).
The served A/B boots A1 B1 B2 A2 run:  A1: c1 P, c8 Q   B1: c8 P   B2: c8 Q   A2: c8 P, c1 Q   (first after boot)
  A rate = A2-c8-P vs A1-c1-P  and  A1-c8-Q vs A2-c1-Q     (the daily's own c8-vs-c1 variation; two halves = A/A)
  B rate = B1-c8-P vs A1-c1-P  and  B2-c8-Q vs A2-c1-Q

  greedy_conc.py --url http://127.0.0.1:8022 --set P --conc 1 --tag A1-c1-P --out gc.jsonl
  greedy_conc.py --compare --out gc.jsonl --pair A_P=A2-c8-P:A1-c1-P --pair A_Q=A1-c8-Q:A2-c1-Q \
                 --pair B_P=B1-c8-P:A1-c1-P --pair B_Q=B2-c8-Q:A2-c1-Q --verdict A_P,A_Q B_P,B_Q
Pre-registered verdict (--verdict A_PAIRS B_PAIRS): with n prompts per pair and the worst A pair's divergences
dA_max:
  PASS iff  no errors in any pair
            and  sum of B divergences <= dA_max x (number of B pairs)        (B's rate <= the worse A half's rate)
            (early divergences, first differing character < --early, are reported as GREEDYC-NOTE and never decide:
             R717 review, the zero-margin sub-rule fired 21-27 % of the time on identical configs)
Output lines: "GREEDYC <pair>: ..." per pair, "GREEDYC-AA ..." (the A halves side by side), and the last line
"GREEDYC-VERDICT PASS|FAIL ...". Exit 0 on PASS or when no verdict is asked.
"""
import argparse
import json
import random
import sys
import threading
import urllib.request

NATURAL = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "Q: A train leaves at 3pm travelling 60 km/h. How far in 2.5 hours?\nA: Let me work through it.",
    "List the first eight prime numbers, separated by commas:",
    "Explain in two sentences why the sky appears blue.",
    "Write a Python function that parses an ISO 8601 date string and returns a datetime, with error handling:\n",
    "#include <stdio.h>\n\n/* Reverse a singly linked list in place. */\nstruct node { int v; struct node* next; };\n",
    "Summarise the causes of the French Revolution in five bullet points.\n",
    "SELECT customers.name, COUNT(orders.id) AS n\nFROM customers\n",
    "Translate into French: 'The meeting was moved to Thursday because the room was booked.'\n",
    "fn main() {\n    // Read lines from stdin and print them sorted, without duplicates\n",
    "A short story about a lighthouse keeper who finds a message in a bottle. Once upon a time,",
    "Explain the difference between TCP and UDP to a new engineer, with one example each.\n",
    "import numpy as np\n\ndef softmax(x, axis=-1):\n",
    "The three laws of thermodynamics are",
    "Give a step-by-step recipe for a basic sourdough loaf.\n1.",
]
FRAMING = {"P": "", "Q": "Answer as a careful expert would, continuing directly from the text below.\n\n"}
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
SALT = {"P": 20260924, "Q": 20260925}


def prompts(which):
    out = [FRAMING[which] + p for p in NATURAL]
    for i in range(16):
        rng = random.Random(SALT[which] * 1_000_003 + i)
        body = " ".join(rng.choice(WORDS) for _ in range(1000))
        out.append(body + "\n\n" + TASKS[i % len(TASKS)] + f"\n\n[document {which}{i}]\n")
    return out


def gen(url, prompt, n, timeout):
    body = json.dumps(dict(model="flashnext", prompt=prompt, max_tokens=n, min_tokens=n, temperature=0.0, seed=0,
                           stream=False)).encode()
    req = urllib.request.Request(url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    c = d["choices"][0]
    return c.get("text", ""), c.get("finish_reason"), (d.get("usage") or {}).get("completion_tokens")


def run(a):
    ps = prompts(a.set)
    out = open(a.out, "a")
    for start in range(0, len(ps), a.conc):
        batch = list(range(start, min(start + a.conc, len(ps))))
        res = {}
        barrier = threading.Barrier(len(batch))

        def one(pid):
            barrier.wait()
            try:
                res[pid] = gen(a.url, ps[pid], a.tokens, a.timeout) + (None,)
            except Exception as e:  # noqa: BLE001 -- recorded, counted as an error by --compare
                res[pid] = (None, None, None, repr(e))
        th = [threading.Thread(target=one, args=(p,)) for p in batch]
        for t in th:
            t.start()
        for t in th:
            t.join()
        for pid in batch:
            text, fin, ntok, err = res[pid]
            out.write(json.dumps(dict(tag=a.tag, set=a.set, pid=pid, conc=a.conc, tokens=a.tokens, text=text,
                                      finish=fin, completion_tokens=ntok, error=err)) + "\n")
        out.flush()
        print(f"[greedyc] {a.tag} set {a.set} c{a.conc}: prompts {batch[0]}..{batch[-1]} done "
              f"({sum(res[p][3] is None for p in batch)} ok)", flush=True)


def first_diff(u, v):
    for i, (x, y) in enumerate(zip(u, v)):
        if x != y:
            return i
    return None if len(u) == len(v) else min(len(u), len(v))


def load(path):
    recs = {}
    for ln in open(path):
        d = json.loads(ln)
        recs.setdefault(d["tag"], {})[d["pid"]] = d
    return recs


def summary(recs, tag, ref, early):
    x, r = recs.get(tag), recs.get(ref)
    if not x or not r:
        return dict(missing=[t for t in (tag, ref) if not recs.get(t)])
    sets = {d["set"] for d in x.values()} | {d["set"] for d in r.values()}
    n = max(len(r), len(x), 32)
    same = div = early_n = errs = 0
    pos = []
    for pid in range(n):
        o, q = x.get(pid), r.get(pid)
        if o is None or q is None or o.get("text") is None or q.get("text") is None:
            errs += 1
            continue
        fd = first_diff(q["text"], o["text"])
        if fd is None:
            same += 1
        else:
            div += 1
            pos.append(fd)
            early_n += fd < early
    return dict(n=n, sets=sorted(sets), identical=same, div=div, early=early_n, errors=errs, first_diff=sorted(pos))


def compare(a):
    recs = load(a.out)
    pairs = {}
    for spec in a.pair:
        name, rest = spec.split("=", 1)
        tag, ref = rest.split(":", 1)
        s = summary(recs, tag, ref, a.early)
        pairs[name] = s
        if "missing" in s:
            print(f"GREEDYC {name}: {tag} vs {ref}: MISSING {s['missing']}")
            continue
        flag = "" if len(s["sets"]) == 1 else f" SET MISMATCH {s['sets']}"
        print(f"GREEDYC {name}: {tag} vs {ref} set {'/'.join(s['sets'])}: identical {s['identical']}/{s['n']} "
              f"divergent {s['div']} early {s['early']} errors {s['errors']} first-diff {s['first_diff']}{flag}")
    if not a.verdict:
        return 0
    an, bn = a.verdict[0].split(","), a.verdict[1].split(",")
    why = []
    for nm in an + bn:
        s = pairs.get(nm)
        if s is None or "missing" in s:
            why.append(f"{nm} missing")
        elif len(s["sets"]) != 1:
            why.append(f"{nm} compares different prompt sets")
    if why:
        print("GREEDYC-VERDICT FAIL (unusable: " + "; ".join(why) + ")")
        return 1
    A = [pairs[x] for x in an]
    B = [pairs[x] for x in bn]
    dmax = max(s["div"] for s in A)
    emax = max(s["early"] for s in A)
    bdiv, bearly, berr = sum(s["div"] for s in B), sum(s["early"] for s in B), sum(s["errors"] for s in B)
    aerr = sum(s["errors"] for s in A)
    na, nb = sum(s["n"] for s in A), sum(s["n"] for s in B)
    print("GREEDYC-AA A halves: " + ", ".join(f"{x} {pairs[x]['div']}/{pairs[x]['n']}" for x in an)
          + f" (spread {max(s['div'] for s in A) - min(s['div'] for s in A)} prompts)")
    fails = []
    if berr:
        fails.append(f"B errors {berr}")
    if aerr:
        fails.append(f"A errors {aerr} (reference unusable)")
    if bdiv > dmax * len(B):
        fails.append(f"B divergences {bdiv}/{nb} > worst A half {dmax} x {len(B)}")
    # early divergences are reported, not judged (R717 review: the zero-margin max-of-halves sub-rule fires 21-27 % on
    # identical configs, and the user's condition 2 is the divergence RATE)
    if bearly > emax * len(B):
        print(f"GREEDYC-NOTE B early divergences {bearly} > worst A half {emax} x {len(B)} (reported, not a FAIL)")
    rates = (f"A {sum(s['div'] for s in A)}/{na} ({100 * sum(s['div'] for s in A) / max(na, 1):.1f} %), worst half "
             f"{dmax}; B {bdiv}/{nb} ({100 * bdiv / max(nb, 1):.1f} %); early A max {emax}, B {bearly}")
    print("GREEDYC-VERDICT " + ("FAIL: " + "; ".join(fails) + " | " if fails else "PASS | ") + rates)
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--tag")
    ap.add_argument("--set", choices=["P", "Q"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--pair", action="append", default=[], help="NAME=TAG:REFTAG (same prompt set)")
    ap.add_argument("--verdict", nargs=2, metavar=("A_PAIRS", "B_PAIRS"), help="comma-separated pair names")
    ap.add_argument("--early", type=int, default=64, help="a first difference before this character is 'early'")
    a = ap.parse_args()
    if a.compare:
        return compare(a)
    if not (a.url and a.tag and a.set):
        ap.error("--url, --tag and --set are required unless --compare")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
