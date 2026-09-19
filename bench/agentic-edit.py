#!/usr/bin/env python3
"""Served agentic-edit decode probe (R501, prompt lookup): the answer repeats its context.

Each prompt carries one real source file (the box's own probe scripts, first --lines lines) and asks for the whole file
back with a small edit, the shape of a coding agent's file-rewrite turn, where prompt lookup has material to copy.
Streams each request (TTFT and decode t/s from the first-token timestamp), at concurrency c, greedy or sampled
(0.6 / 0.95 / 20, seed per prompt). Prints one summary line per (mode, conc) and writes every request to --out (JSONL).
"""
import argparse, glob, json, os, statistics, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--files", default="/srv/qwen5090/probes/*.py"); ap.add_argument("--n", type=int, default=6)
ap.add_argument("--lines", type=int, default=160); ap.add_argument("--tokens", type=int, default=3072)
ap.add_argument("--conc", type=int, nargs="+", default=[1, 4]); ap.add_argument("--modes", nargs="+", default=["greedy", "sampled"])
ap.add_argument("--out", required=True)
a = ap.parse_args()

files = sorted(f for f in glob.glob(a.files) if os.path.getsize(f) > 4000)[: a.n]
EDITS = ["rename every local variable called `r` to `record`", "add a module docstring line and type hints to every function signature",
         "replace every f-string with str.format", "rename the function `main` to `run` and update its call site",
         "sort the import lines alphabetically", "add a `# noqa` comment to every line that defines a function"]
prompts = []
for i, f in enumerate(files):
    body = "".join(open(f).readlines()[: a.lines])
    prompts.append(f"Here is the file `{os.path.basename(f)}`:\n\n```python\n{body}```\n\nReturn the COMPLETE file with one change: "
                   f"{EDITS[i % len(EDITS)]}. Output only the code in one ```python block, no explanation.")

def one(prompt, mode, seed, rec):
    body = {"model": a.model, "stream": True, "max_tokens": a.tokens, "messages": [{"role": "user", "content": prompt}],
            "stream_options": {"include_usage": True}}
    body.update({"temperature": 0} if mode == "greedy" else {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": seed})
    t0 = time.time(); first = None; n = None
    req = urllib.request.Request(a.url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                d = json.loads(line[5:])
                if d.get("usage"): n = d["usage"].get("completion_tokens")
                ch = d.get("choices") or []
                if ch and first is None and ((ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning_content")):
                    first = time.time()
        t1 = time.time()
        rec.update(ok=True, ttft_s=(first or t1) - t0, completion_tokens=n,
                   decode_tps=(n - 1) / (t1 - first) if (n and first and t1 > first) else None, wall_s=t1 - t0)
    except Exception as e:
        rec.update(ok=False, error=str(e)[:200])

with open(a.out, "a") as out:
    for mode in a.modes:
        for c in a.conc:
            recs = [dict(tag=a.tag, mode=mode, conc=c, prompt=i, file=os.path.basename(files[i])) for i in range(len(prompts))]
            t0 = time.time()
            for start in range(0, len(prompts), c):
                ths = [threading.Thread(target=one, args=(prompts[i], mode, 5000 + i, recs[i])) for i in range(start, min(start + c, len(prompts)))]
                [t.start() for t in ths]; [t.join() for t in ths]
            wall = time.time() - t0
            for r in recs: out.write(json.dumps(r) + "\n")
            ok = [r for r in recs if r.get("ok") and r.get("decode_tps")]
            toks = sum(r.get("completion_tokens") or 0 for r in recs)
            med = statistics.median(r["decode_tps"] for r in ok) if ok else float("nan")
            print(f"[{a.tag}] agentic-edit {mode} c{c}: {len(ok)}/{len(recs)} ok, {toks} tokens, per-request decode median {med:.1f} t/s "
                  f"(min {min((r['decode_tps'] for r in ok), default=float('nan')):.1f}, max {max((r['decode_tps'] for r in ok), default=float('nan')):.1f}), "
                  f"aggregate {toks / wall:.1f} t/s", flush=True)
            # the whole-run aggregate mixes waves: with 6 files at c4 the second wave runs only 2 streams. Report the first
            # full wave on its own (all c streams busy): aggregate over its slowest request, and the per-stream mean.
            w = [r for r in recs[:c] if r.get("ok") and r.get("wall_s")]
            if len(prompts) >= c and len(w) == c:
                print(f"[{a.tag}] agentic-edit {mode} c{c} first full wave: aggregate "
                      f"{sum(r['completion_tokens'] for r in w) / max(r['wall_s'] for r in w):.1f} t/s, per stream "
                      f"{statistics.mean(r['completion_tokens'] / r['wall_s'] for r in w):.1f} t/s", flush=True)
