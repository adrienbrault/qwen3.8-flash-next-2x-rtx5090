#!/usr/bin/env python3
"""Multi-prompt decode benchmark for arms whose greedy text differs (R542/R545).

A single fixed prompt cannot rank two arms that decode different text: MTP acceptance depends on the content, and one prompt
moved c1 by -13 % .. +11 % between numerically different arms (R542, R545). This probe decodes many distinct prompts per arm
and pairs them by prompt id, so the content effect averages out and the comparison is per prompt.

  run      one boot: c1 over every prompt, then c4 in groups of four distinct prompts; one JSONL line per request
  compare  pair two arms by (kind, conc, prompt id), mean over boots per arm, geometric-mean ratio B/A with a bootstrap CI

Requests reuse fn_bench.one (forced length via min_tokens, streaming, decode_tps = (n - 1) / decode window).
"""
import argparse
import collections
import importlib.util
import json
import math
import random
import statistics as st
import sys
import threading
from pathlib import Path

CODE = [
    "Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests.",
    "Implement an LRU cache class in Python with get/put in O(1), plus unit tests.",
    "Write a Rust function that merges overlapping intervals and returns them sorted, with tests.",
    "Implement a thread-safe bounded blocking queue in Java with put/take and a timeout variant.",
    "Write a TypeScript function that deep-merges two JSON objects, arrays concatenated, with Jest tests.",
    "Implement Dijkstra's shortest path in Go over an adjacency list, returning distances and the path.",
    "Write a SQL schema and queries for a library system: books, members, loans, overdue report.",
    "Implement a tokenizer and recursive-descent parser for arithmetic expressions in Python, with tests.",
    "Write a C function that reverses a singly linked list in place and a small test harness.",
    "Implement a rate limiter (token bucket) in Python usable as a decorator, with tests using a fake clock.",
    "Write a bash script that rotates log files older than N days into gzip archives, with argument parsing.",
    "Implement a trie in Kotlin with insert, search, prefix listing and deletion.",
    "Write a Python asyncio web crawler limited to one domain with a concurrency cap and robots.txt support.",
    "Implement matrix multiplication with cache blocking in C++ and a benchmark harness.",
    "Write a React component for a paginated, sortable table with a search box, plus tests.",
    "Implement a consistent-hashing ring with virtual nodes in Python, with rebalancing tests.",
    "Write a Python CLI (argparse) that diffs two CSV files by key column and prints changed rows.",
    "Implement a simple regex engine supporting . * + ? and character classes in Python.",
    "Write a Go HTTP middleware that logs request duration and status, with a table-driven test.",
    "Implement an interval tree in Python supporting insert, delete and overlap queries.",
    "Write a Python function that validates and normalizes international phone numbers without libraries.",
    "Implement a Bloom filter in Rust with configurable false-positive rate and tests.",
    "Write a Dockerfile and docker-compose file for a Flask app with Postgres and Redis, with health checks.",
    "Implement topological sort with cycle detection in TypeScript, returning the cycle when one exists.",
]
PROSE = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how vaccines train the immune system, for a curious teenager.",
    "Describe a walk through a night market in a city you invent, with sounds and smells.",
    "Write a persuasive essay on why cities should plant more trees.",
    "Explain the causes of the French Revolution in plain language.",
    "Write a letter from an astronaut on Mars to their younger self.",
    "Describe how bread is made, from wheat field to bakery, as a narrative.",
    "Write a travel guide section for a fictional mountain village.",
    "Explain how a refrigerator works, step by step, without equations.",
    "Write a fable about a fox, a crow and a river that changes course.",
    "Summarize the history of the printing press and its effects on Europe.",
    "Write a product review of a fictional pair of hiking boots after a month-long trek.",
    "Explain the rules and strategy of chess openings to a beginner.",
    "Write a diary entry of a lighthouse cat during a storm.",
    "Describe the life cycle of a star, from nebula to remnant.",
    "Write a speech for a school graduation about curiosity.",
    "Explain how interest rates affect housing prices.",
    "Write a mystery scene in a train compartment with three suspects.",
    "Describe the migration of monarch butterflies and why it matters.",
    "Write a dialogue between a chef and a food critic about a failed dish.",
    "Explain what causes the seasons on Earth, with common misconceptions.",
    "Write a poem-like prose piece about the first snowfall in a city.",
    "Describe how a bill becomes law in a parliamentary system.",
    "Write a short biography of an invented inventor from the 1800s.",
]


def load_fn_bench(path):
    spec = importlib.util.spec_from_file_location("fn_bench_mp", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fn_bench_mp"] = mod
    spec.loader.exec_module(mod)
    return mod


def cmd_run(a):
    fb = load_fn_bench(a.fn_bench)
    fb.UNIQUE[0] = False
    out = open(a.out, "a")
    for kind, prompts in (("code", CODE), ("prose", PROSE)):
        prompts = prompts[: a.n]
        for i, p in enumerate(prompts):
            sink = []
            fb.one(i, a.url, a.model, p, a.tokens, True, sink, a.timeout)
            for r in sink:
                out.write(json.dumps({"tag": a.tag, "kind": kind, "conc": 1, "pid": i, **r}) + "\n")
            out.flush()
        for g in range(0, len(prompts) - len(prompts) % 4, 4):
            sink = []
            ths = [threading.Thread(target=fb.one, args=(g + j, a.url, a.model, prompts[g + j], a.tokens, True, sink, a.timeout))
                   for j in range(4)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            for r in sink:
                out.write(json.dumps({"tag": a.tag, "kind": kind, "conc": 4, "pid": r["i"], **r}) + "\n")
            out.flush()
    out.close()


def cmd_compare(a):
    rows = [json.loads(l) for f in a.files for l in open(f)]
    val = collections.defaultdict(list)
    short = collections.Counter()
    for r in rows:
        if not r.get("ok") or not r.get("decode_tps"):
            continue
        arm = r["tag"].rstrip("0123456789")
        val[(arm, r["kind"], r["conc"], r["pid"])].append(r["decode_tps"])
        short[(arm, r["kind"], r["conc"])] += (r["completion_tokens"] or 0) < r["min_tokens"]
    rng = random.Random(7)
    for kind in ("code", "prose"):
        for conc in (1, 4):
            pids = sorted({k[3] for k in val if k[1] == kind and k[2] == conc and (a.a, kind, conc, k[3]) in val and (a.b, kind, conc, k[3]) in val})
            if not pids:
                continue
            lr = [math.log(st.mean(val[(a.b, kind, conc, p)]) / st.mean(val[(a.a, kind, conc, p)])) for p in pids]
            boots = sorted(st.mean(rng.choice(lr) for _ in lr) for _ in range(4000))
            g = math.exp(st.mean(lr)); lo, hi = math.exp(boots[100]), math.exp(boots[3899])
            ma = st.mean(st.mean(val[(a.a, kind, conc, p)]) for p in pids); mb = st.mean(st.mean(val[(a.b, kind, conc, p)]) for p in pids)
            print(f"{kind} c{conc}: {len(pids)} prompts, per-request decode_tps {a.a} {ma:.1f} / {a.b} {mb:.1f}; "
                  f"paired geo-mean {a.b}/{a.a} {(g - 1) * 100:+.2f} %, 95 % CI [{(lo - 1) * 100:+.2f}, {(hi - 1) * 100:+.2f}] %; "
                  f"per-prompt range {(math.exp(min(lr)) - 1) * 100:+.1f} .. {(math.exp(max(lr)) - 1) * 100:+.1f} %; "
                  f"short {a.a} {short[(a.a, kind, conc)]} / {a.b} {short[(a.b, kind, conc)]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest = "cmd", required = True)
    r = sub.add_parser("run")
    r.add_argument("--url", required = True)
    r.add_argument("--model", required = True)
    r.add_argument("--tag", required = True, help = "arm letter(s) + boot number, e.g. A1, B2")
    r.add_argument("--tokens", type = int, default = 512)
    r.add_argument("--n", type = int, default = 24)
    r.add_argument("--timeout", type = float, default = 900)
    r.add_argument("--fn-bench", default = str(Path(__file__).with_name("probe.py")))
    r.add_argument("--out", required = True)
    c = sub.add_parser("compare")
    c.add_argument("--a", default = "A")
    c.add_argument("--b", default = "B")
    c.add_argument("files", nargs = "+")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    main()
