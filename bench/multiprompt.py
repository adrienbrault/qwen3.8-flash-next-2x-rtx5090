#!/usr/bin/env python3
"""Multi-prompt decode probe: N distinct prompts per kind, sampled (temp 0.6, top_p 0.95, top_k 20, per-prompt seed), forced
length, at concurrency c. Averages out the content luck of a single greedy sample (MTP acceptance depends on the text).
Prints one summary line per (kind, conc) and writes every request to --out as JSONL."""
import argparse, json, time, threading, urllib.request, statistics
CODE = ["Write a Python function that merges overlapping intervals, with tests.",
        "Implement an LRU cache class in Python without functools, with tests.",
        "Write a Rust function that parses a CSV line with quoted fields, with unit tests.",
        "Implement Dijkstra's shortest path in TypeScript with a binary heap.",
        "Write a Go HTTP handler that streams server-sent events with a heartbeat.",
        "Implement a trie with insert, search and prefix listing in Java.",
        "Write a Python asyncio rate limiter (token bucket) with tests.",
        "Write a SQL schema and queries for a library loan system with overdue reports.",
        "Implement matrix multiplication with loop tiling in C, with a benchmark main.",
        "Write a bash script that rotates log files by size and keeps N gzip archives.",
        "Implement a JSON pretty-printer in Python without the json module.",
        "Write a React hook that debounces a value and cancels on unmount, with tests."]
PROSE = ["Write an essay on how tides work and why there are two a day.",
         "Describe a day in the life of a lighthouse keeper in 1890.",
         "Explain the causes of the 1929 stock market crash to a teenager.",
         "Write a short story about a cartographer who maps a city that keeps changing.",
         "Compare the philosophies of Stoicism and Epicureanism.",
         "Write a travel guide to the Faroe Islands for a first-time visitor.",
         "Explain how vaccines train the immune system.",
         "Write a letter from a retired sailor to his granddaughter about the sea.",
         "Describe the history of the printing press and its effects on Europe.",
         "Write a review of an imaginary jazz album recorded in a cave.",
         "Explain why the sky is blue and sunsets are red.",
         "Write a fable about a fox, a crow and a broken clock."]
def one(url, model, prompt, seed, ntok, sink):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": ntok, "min_tokens": ntok,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": seed, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            try: c = json.loads(d)
            except Exception: continue
            if c.get("usage"): usage = c["usage"]
            for ch in c.get("choices") or []:
                dd = ch.get("delta") or {}
                if tf is None and (dd.get("content") or dd.get("reasoning_content") or dd.get("reasoning")): tf = time.time()
    t1 = time.time(); n = (usage or {}).get("completion_tokens", 0)
    sink.append({"seed": seed, "tokens": n, "wall": t1 - t0, "decode_tps": (n - 1) / (t1 - tf) if tf and t1 > tf else None})
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--tokens", type=int, default=1024); ap.add_argument("--conc", type=int, nargs="+", default=[1, 4])
ap.add_argument("--out", required=True)
a = ap.parse_args()
log = open(a.out, "a")
for kind, prompts in (("code", CODE), ("prose", PROSE)):
    for c in a.conc:
        sink = []; t0 = time.time()
        for i in range(0, len(prompts), c):
            th = [threading.Thread(target=one, args=(a.url, a.model, p, 1000 + i + j, a.tokens, sink)) for j, p in enumerate(prompts[i:i + c])]
            [t.start() for t in th]; [t.join() for t in th]
        wall = time.time() - t0
        for s in sink: s.update({"tag": a.tag, "kind": kind, "conc": c}); log.write(json.dumps(s) + "\n")
        tot = sum(s["tokens"] for s in sink); dt = [s["decode_tps"] for s in sink if s["decode_tps"]]
        print(f"[{a.tag}] {kind} c{c}: {len(sink)} prompts, aggregate {tot / wall:.1f} t/s, per-request decode median {statistics.median(dt):.1f} (min {min(dt):.1f}, max {max(dt):.1f})", flush=True)
