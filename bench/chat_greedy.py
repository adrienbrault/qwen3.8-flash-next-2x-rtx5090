#!/usr/bin/env python3
"""Greedy /v1/chat/completions identity check (R783): the path fn_greedy.py (/v1/completions) does not exercise.

Sequential (c1, deterministic on the Flash-Next daily), temperature 0, thinking on, non-streamed. Five prompts: code,
arithmetic, prose, a tool call (one web_search tool) and a two-turn conversation. Records reasoning, content and tool
calls per prompt; --compare reports per-prompt identity of every field between two tags.
  chat_greedy.py --url http://127.0.0.1:8022/v1 --model M --tag OLD --out f.jsonl
  chat_greedy.py --compare --ref OLD --cand NEW --out f.jsonl      (exit 1 unless all identical)
--long (R785) adds one more prompt, pid "long", after the five: a long technical answer at T=0 capped at --long-tokens
(default 5000), so the output crosses the 2,048-token requeue boundary (max_rq_tokens) at least once; its row also records
"crossed_requeue" (completion_tokens > 2048). TabbyAPI returns "usage": null on non-streamed chat (R785 G2 false fail),
so when usage is missing completion_tokens is counted with /v1/token/encode over reasoning + content ("tokens_source":
"encode"; approximate, the think tags are not counted). The five prompts and their rows are unchanged; --compare includes "long"
only when --long is passed.
Stdlib only.
"""
import argparse, hashlib, json, sys, urllib.request

TOOL = {"type": "function", "function": {"name": "web_search", "description": "Search the web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
PROMPTS = {
    "code": [{"role": "user", "content": "Write a Python function that merges two sorted lists in linear time, with a short docstring."}],
    "math": [{"role": "user", "content": "A train leaves at 09:40 and arrives at 13:05. How long is the trip in minutes? Show the steps briefly."}],
    "prose": [{"role": "user", "content": "Describe a rainy morning in a harbour town in four sentences."}],
    "tool": [{"role": "user", "content": "Search the web for the current population of Lyon."}],
    "multi": [{"role": "user", "content": "Name three prime numbers above 50."},
              {"role": "assistant", "content": "53, 59 and 61."},
              {"role": "user", "content": "Which of them is closest to 60? Answer in one sentence."}],
}
LONG = {
    "long": [{"role": "user", "content": "Write a complete, detailed guide to implementing a B-tree in Python. First explain "
              "the invariants and why they hold, then give the full implementation (search, insertion with node splits, "
              "deletion with borrowing and merging), then a pytest test suite, then walk through inserting the keys 1 to 30 "
              "into an empty tree of minimum degree 3, showing the tree after every split. Aim for about 3,000 words."}],
}
REQUEUE = 2048


def encode_len(url, text, timeout):
    req = urllib.request.Request(url + "/token/encode", json.dumps({"text": text}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))["length"]


def ask(url, model, pid, n, timeout):
    body = {"model": model, "messages": {**PROMPTS, **LONG}[pid], "temperature": 0, "max_tokens": n}
    if pid == "tool":
        body["tools"] = [TOOL]
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    c = d["choices"][0]; m = c["message"]
    r = {"finish": c.get("finish_reason"), "reasoning": m.get("reasoning_content") or "", "content": m.get("content") or "",
         "tool_calls": [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []],
         "completion_tokens": (d.get("usage") or {}).get("completion_tokens"), "tokens_source": "usage"}
    if r["completion_tokens"] is None:
        r["completion_tokens"] = encode_len(url, r["reasoning"] + r["content"], timeout); r["tokens_source"] = "encode"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url"); ap.add_argument("--model"); ap.add_argument("--tag"); ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=2048); ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--compare", action="store_true"); ap.add_argument("--ref"); ap.add_argument("--cand")
    ap.add_argument("--long", action="store_true"); ap.add_argument("--long-tokens", type=int, default=5000)
    a = ap.parse_args()
    pids = list(PROMPTS) + (list(LONG) if a.long else [])
    if a.compare:
        rows = [json.loads(l) for l in open(a.out)]
        ref = {r["pid"]: r for r in rows if r["tag"] == a.ref}
        cand = {r["pid"]: r for r in rows if r["tag"] == a.cand}
        bad = 0
        for pid in pids:
            if pid not in ref or pid not in cand:
                print(f"CHAT-GREEDY {pid}: MISSING"); bad += 1; continue
            diff = [k for k in ("finish", "reasoning", "content", "tool_calls") if ref[pid][k] != cand[pid][k]]
            print(f"CHAT-GREEDY {pid}: {'IDENTICAL' if not diff else 'DIVERGE ' + ','.join(diff)} sha {cand[pid]['sha'][:16]}")
            bad += bool(diff)
        print(f"CHAT-GREEDY {a.cand} vs {a.ref}: {len(pids) - bad}/{len(pids)} identical")
        sys.exit(1 if bad else 0)
    with open(a.out, "a") as f:
        for pid in pids:
            r = ask(a.url, a.model, pid, a.long_tokens if pid in LONG else a.tokens, a.timeout)
            r.update(tag=a.tag, pid=pid)
            if pid in LONG:
                r["crossed_requeue"] = (r["completion_tokens"] or 0) > REQUEUE
            r["sha"] = hashlib.sha256(json.dumps([r["finish"], r["reasoning"], r["content"], r["tool_calls"]]).encode()).hexdigest()
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"{a.tag} {pid}: finish={r['finish']} tok={r['completion_tokens']} sha {r['sha'][:16]}", flush=True)


if __name__ == "__main__":
    main()
