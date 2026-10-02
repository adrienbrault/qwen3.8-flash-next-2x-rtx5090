#!/usr/bin/env python3
"""Greedy-equivalence probe: does --spec-draft-n-max change the OUTPUT?

Speculative decoding is output-equivalent by construction. The target model verifies every drafted token,
so at temperature 0 the emitted text must be byte-identical whatever the draft length is; ns is supposed to
buy speed and nothing else. If two ns values disagree on one greedy continuation, verification is not
holding, and any throughput gained is gained by skipping work that mattered.

This is a sharper instrument than the needle probe that first caught it (R292c: ns7 answered 2/5 where ns3
answers 10/10). A needle miss could be a hard question; a byte difference at temperature 0 cannot.

Prompts are FIXED, not seeded per invocation -- the opposite of depth_ss.py/fn_needle.py, where a fresh
filler per run is what stops the prefix cache serving a previous run. Here every arm must see exactly the
same bytes or the comparison means nothing. The long prompt is built from a fixed seed for the same reason.

  fn_greedy.py --url http://127.0.0.1:8031 --tag ns7 --out greedy.jsonl
  fn_greedy.py --compare greedy.jsonl --ref ns3

--stream (R806): the same requests streamed (SSE, stream_options.include_usage); the row is the concatenation of
choices[0].text over the chunks, finish = the last non-null finish_reason, pred_n/prompt_n from the usage chunk. Same row
schema as the non-streamed mode, so --compare works across modes. The streamed /v1/completions path is the one that runs
TabbyAPI's pre-stream check_context_length (router.py), which the non-streamed path skips.
"""
import argparse, json, random, sys, urllib.request

SHORT = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "Q: A train leaves at 3pm travelling 60 km/h. How far in 2.5 hours?\nA: Let me work through it.",
    "List the first eight prime numbers, separated by commas:",
    "Explain in two sentences why the sky appears blue.",
]
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar "
         "papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()

def long_prompt(approx_tokens=100000):
    rng = random.Random(20260912)          # FIXED seed: every arm must see identical bytes
    body = " ".join(rng.choice(WORDS) for _ in range(int(approx_tokens * 0.78)))
    return body + "\n\nSummarise the above in exactly one sentence.\nSummary:"

def gen(url, prompt, n, timeout):
    body = json.dumps(dict(model="flashnext", prompt=prompt, max_tokens=n,
                           temperature=0.0, seed=0, stream=False)).encode()
    req = urllib.request.Request(url + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    c = d["choices"][0]
    t = (d.get("timings") or {})
    return c.get("text", ""), c.get("finish_reason"), t.get("predicted_n"), t.get("prompt_n")

def parse_stream(lines):
    """SSE lines (bytes or str) of a streamed /v1/completions -> (text, finish, completion_tokens, prompt_tokens)."""
    text, finish, usage = [], None, {}
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        d = json.loads(data)
        for c in d.get("choices") or []:
            if c.get("index", 0) != 0:
                continue
            text.append(c.get("text") or "")
            finish = c.get("finish_reason") or finish
        usage = d.get("usage") or usage
    return "".join(text), finish, usage.get("completion_tokens"), usage.get("prompt_tokens")

def gen_stream(url, prompt, n, timeout):
    body = json.dumps(dict(model="flashnext", prompt=prompt, max_tokens=n, temperature=0.0, seed=0, stream=True,
                           stream_options={"include_usage": True})).encode()
    req = urllib.request.Request(url + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return parse_stream(r)

def compare(path, ref):
    recs = {}
    for ln in open(path):
        d = json.loads(ln); recs.setdefault(d["tag"], {})[d["pid"]] = d
    if ref not in recs:
        print(f"reference arm {ref!r} not in {path}; have {sorted(recs)}"); return 1
    bad = 0
    for tag in sorted(recs):
        if tag == ref: continue
        same = diff = 0
        for pid, r in sorted(recs[ref].items()):
            o = recs[tag].get(pid)
            if o is None:   # the arm errored on this prompt (no record written): that is not agreement (R683 review)
                diff += 1
                print(f"  MISSING {tag} prompt={pid} (errored or not run; counted as a divergence)")
                continue
            if o["text"] == r["text"]: same += 1
            else:
                diff += 1
                print(f"  DIVERGE {tag} vs {ref} prompt={pid}")
                print(f"     {ref}: {r['text'][:90]!r} (finish={r['finish']}, n={r['pred_n']})")
                print(f"     {tag}: {o['text'][:90]!r} (finish={o['finish']}, n={o['pred_n']})")
        verdict = "IDENTICAL" if diff == 0 else f"{diff} DIVERGENT"
        print(f"GREEDY {tag} vs {ref}: {same} identical, {verdict}")
        bad += diff
    print(f"GREEDY-SUMMARY reference={ref} divergences={bad}")
    return 0 if bad == 0 else 1

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url"); p.add_argument("--tag"); p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=int, default=128); p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--skip-long", action="store_true")
    p.add_argument("--stream", action="store_true", help="stream the requests (R806; same row schema)")
    p.add_argument("--compare", action="store_true"); p.add_argument("--ref", default="ns3")
    a = p.parse_args()
    if a.compare:
        sys.exit(compare(a.out, a.ref))
    prompts = [(f"short{i}", s) for i, s in enumerate(SHORT)]
    if not a.skip_long:
        prompts.append(("long100k", long_prompt()))
    with open(a.out, "a") as fh:
        for pid, pr in prompts:
            try:
                text, finish, pn, promptn = (gen_stream if a.stream else gen)(a.url, pr, a.tokens, a.timeout)
            except Exception as e:
                print(f"GREEDY {a.tag} {pid} ERROR {e}"); continue
            fh.write(json.dumps(dict(tag=a.tag, pid=pid, text=text, finish=finish,
                                     pred_n=pn, prompt_n=promptn, mode="stream" if a.stream else "json")) + "\n")
            print(f"GREEDY {a.tag} {pid} depth={promptn} n={pn} finish={finish} {text[:60]!r}")
