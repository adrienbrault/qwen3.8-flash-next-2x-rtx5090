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
--stream (R806): the same requests streamed (SSE, stream_options.include_usage) -- the path that runs TabbyAPI's
pre-stream check_context_length and, with tokenize-offloop r1, reuses its ids. Rows have the same schema and sha as the
non-streamed mode (so --compare works across modes): reasoning / content are the concatenated delta.reasoning_content /
delta.content, tool_calls come from the final chunk's delta.tool_calls, finish is the last non-null finish_reason (the
usage chunk carries it when include_usage is set, loop-think r5), completion_tokens from the usage chunk. The
non-streamed collector returns content None when it is whitespace only (chat_completion.py `has_content`); the streamed
row applies the same rule, so a "\n\n" before a tool call does not read as a divergence.
--n2 (with --stream): one more request, the "code" prompt with n=2, rows pid "code@n2.0" / "code@n2.1";
--check-n2 --cand TAG then requires both choices identical to each other and to TAG's "code" row (exit 1 otherwise).
--pids P [P ...] runs only those prompt ids (R808: a second non-streamed "long" right after the first, tagged apart).
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


def parse_stream(lines):
    """SSE lines (bytes or str) of a streamed chat completion -> {choice index: row fields} (finish, reasoning, content,
    tool_calls, completion_tokens, tokens_source), the non-streamed row's schema."""
    acc, usage = {}, None
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
            r = acc.setdefault(c.get("index", 0), {"finish": None, "reasoning": [], "content": [], "tool_calls": []})
            dl = c.get("delta") or {}
            r["reasoning"].append(dl.get("reasoning_content") or "")
            r["content"].append(dl.get("content") or "")
            for t in dl.get("tool_calls") or []:
                r["tool_calls"].append((t["function"]["name"], t["function"]["arguments"]))
            r["finish"] = c.get("finish_reason") or r["finish"]
        usage = d.get("usage") or usage
    out = {}
    for i, r in acc.items():
        content = "".join(r["content"])
        out[i] = {"finish": r["finish"], "reasoning": "".join(r["reasoning"]),
                  "content": content if content.strip() else "",   # the non-streamed collector's has_content rule
                  "tool_calls": r["tool_calls"], "completion_tokens": None, "tokens_source": "usage"}
    if usage and len(out) == 1:
        next(iter(out.values()))["completion_tokens"] = usage.get("completion_tokens")
    return out


def ask_stream(url, model, pid, n, timeout, choices=1):
    body = {"model": model, "messages": {**PROMPTS, **LONG}[pid], "temperature": 0, "max_tokens": n, "stream": True,
            "stream_options": {"include_usage": True}}
    if choices > 1:
        body["n"] = choices
    if pid == "tool":
        body["tools"] = [TOOL]
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        rows = parse_stream(resp)
    for r in rows.values():
        if r["completion_tokens"] is None:
            r["completion_tokens"] = encode_len(url, r["reasoning"] + r["content"], timeout); r["tokens_source"] = "encode"
    return rows


def row_sha(r):
    return hashlib.sha256(json.dumps([r["finish"], r["reasoning"], r["content"], r["tool_calls"]]).encode()).hexdigest()


def check_n2(rows, cand):
    by = {r["pid"]: r for r in rows if r["tag"] == cand}
    base = by.get("code")
    got = [by.get(f"code@n2.{i}") for i in range(2)]
    if base is None or None in got:
        print(f"CHAT-N2 {cand}: MISSING (code {'ok' if base else 'missing'}, n2 rows {sum(g is not None for g in got)}/2)")
        return 1
    same = sum(1 for g in got if all(g[k] == base[k] for k in ("finish", "reasoning", "content", "tool_calls")))
    print(f"CHAT-N2 {cand}: {same}/2 identical to code (n=1) sha {base['sha'][:16]}; the two choices "
          f"{'identical' if got[0]['sha'] == got[1]['sha'] else 'DIFFER'}")
    return 0 if same == 2 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url"); ap.add_argument("--model"); ap.add_argument("--tag"); ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=2048); ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--compare", action="store_true"); ap.add_argument("--ref"); ap.add_argument("--cand")
    ap.add_argument("--long", action="store_true"); ap.add_argument("--long-tokens", type=int, default=5000)
    ap.add_argument("--stream", action="store_true"); ap.add_argument("--n2", action="store_true")
    ap.add_argument("--check-n2", action="store_true")
    ap.add_argument("--pids", nargs="+", help="run only these prompt ids (R808: 'long' again, the requeue discriminator)")
    a = ap.parse_args()
    if a.check_n2:
        sys.exit(check_n2([json.loads(l) for l in open(a.out)], a.cand))
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
    if a.pids:
        unknown = [p for p in a.pids if p not in pids]
        if unknown:
            sys.exit(f"--pids {unknown}: not in {pids} (add --long for 'long')")
        pids = [p for p in pids if p in a.pids]
    with open(a.out, "a") as f:
        jobs = [(pid, pid, 1) for pid in pids] + ([("code", "code@n2", 2)] if a.stream and a.n2 else [])
        for pid, rid, nch in jobs:
            ntok = a.long_tokens if pid in LONG else a.tokens
            if a.stream:
                got = ask_stream(a.url, a.model, pid, ntok, a.timeout, nch)
                rows = [(rid if nch == 1 else f"{rid}.{i}", got[i]) for i in sorted(got)]
            else:
                rows = [(rid, ask(a.url, a.model, pid, ntok, a.timeout))]
            for row_pid, r in rows:
                r.update(tag=a.tag, pid=row_pid, mode="stream" if a.stream else "json")
                if pid in LONG:
                    r["crossed_requeue"] = (r["completion_tokens"] or 0) > REQUEUE
                r["sha"] = row_sha(r)
                f.write(json.dumps(r) + "\n"); f.flush()
                print(f"{a.tag} {row_pid}: finish={r['finish']} tok={r['completion_tokens']} sha {r['sha'][:16]}", flush=True)


if __name__ == "__main__":
    main()
