#!/usr/bin/env python3
"""Agent-shaped serving probe: replay recorded mini-SWE-agent conversations against an OpenAI-compatible endpoint.

Each of --agents workers takes conversations from a fixed, hash-ordered list and walks them call by call: the prompt for call k
is the recorded messages before the k-th assistant turn (system, user, assistant turns with their reasoning and tool calls, tool
results), sent with the mini-SWE-agent bash tool, greedy, max_tokens = the recorded turn's length (estimated, capped). The
server's own generation is discarded, so call k+1 shares a prefix with call k up to the end of call k's prompt -- the cache
reuse an agent gets when its re-rendered turn differs from what the server generated. --tool-gap seconds of sleep between calls
stand in for tool execution. Same conversations, same order, same budget in every arm.

--echo instead sends the server's own previous answers back, as an agent loop does: call k+1's history is call k's history,
the recorded messages between the two assistant turns (tool results re-keyed to the server's tool-call ids; if the server made
no call, the recorded tool output goes in as a user message), and the server's answer (content, reasoning, tool calls as
returned). The conversation drifts from the recording after the first call; what it exercises is prefix reuse of generated
tokens (recurrent tip checkpoints).

Writes one JSON line per call and a summary line; the TabbyAPI log (prompt N tokens, X cached, Y new) gives the cache side.
"""
import argparse
import glob
import hashlib
import json
import os
import queue
import statistics
import threading
import time
import urllib.request

BASH_TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
                   "required": ["command"]}}}


def load(path):
    d = json.load(open(path))
    msgs, calls = [], []
    for m in d["messages"]:
        role = m.get("role")
        if role in ("system", "user"):
            msgs.append({"role": role, "content": m.get("content") or ""})
        elif role == "assistant":
            calls.append(len(msgs))
            a = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("reasoning_content"):
                a["reasoning_content"] = m["reasoning_content"]
            tcs = [{"id": t.get("id"), "type": "function",
                    "function": {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}}
                   for t in (m.get("tool_calls") or [])]
            if tcs:
                a["tool_calls"] = tcs
            est = len(a["content"]) + len(a.get("reasoning_content", "")) + sum(len(t["function"]["arguments"]) for t in tcs)
            a["_est_tokens"] = est // 3
            msgs.append(a)
        elif role == "tool":
            msgs.append({"role": "tool", "tool_call_id": m.get("tool_call_id"), "content": m.get("content") or ""})
    return d.get("instance_id") or os.path.basename(os.path.dirname(path)), msgs, calls


def clean(msgs):
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in msgs]


def between(msgs, calls, k):
    return msgs[(calls[k - 1] + 1) if k else 0:calls[k]]


def adapt(seg, prev):
    """Recorded messages that follow an assistant turn, re-keyed to the server's own answer `prev`."""
    ids = [t.get("id") for t in (prev or {}).get("tool_calls") or []]
    tools = [m for m in seg if m["role"] == "tool"]
    rest = [m for m in seg if m["role"] != "tool"]
    if prev is None:
        return seg
    if not ids:
        body = "\n\n".join(m["content"] for m in tools)
        return ([{"role": "user", "content": "Tool output:\n" + body}] if tools else []) + rest
    out = [{"role": "tool", "tool_call_id": ids[i], "content": tools[i]["content"] if i < len(tools) else "(no output)"}
           for i in range(len(ids))]
    return out + rest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--trajs", nargs="+", required=True, help="globs of *.traj.json; later globs override earlier per instance")
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--convs", type=int, default=16)
    p.add_argument("--calls", type=int, default=24, help="first N calls of each conversation")
    p.add_argument("--cap", type=int, default=2048, help="max_tokens cap per call")
    p.add_argument("--tool-gap", type=float, default=2.0)
    p.add_argument("--echo", action="store_true", help="send the server's own answers back (agent loop), not the recording")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    files = {}
    for g in a.trajs:
        for f in sorted(glob.glob(g)):
            files[os.path.basename(os.path.dirname(f))] = f
    order = sorted(files, key=lambda i: hashlib.sha1(i.encode()).hexdigest())[:a.convs]
    work = queue.Queue()
    for i in order:
        work.put(load(files[i]))
    lock = threading.Lock()
    rows = []
    out = open(a.out, "a")

    def agent(n):
        while True:
            try:
                iid, msgs, calls = work.get_nowait()
            except queue.Empty:
                return
            hist, prev = [], None
            for k, idx in enumerate(calls[:a.calls]):
                max_tokens = max(64, min(a.cap, msgs[idx]["_est_tokens"]))
                if a.echo:
                    hist += adapt(clean(between(msgs, calls, k)), prev)
                sent = hist if a.echo else clean(msgs[:idx])
                body = {"model": a.model, "messages": sent, "tools": [BASH_TOOL], "temperature": 0,
                        "max_tokens": max_tokens}
                t0 = time.time()
                err = None
                try:
                    r = json.loads(urllib.request.urlopen(urllib.request.Request(
                        a.url + "/chat/completions", data=json.dumps(body).encode(),
                        headers={"Content-Type": "application/json"}), timeout=1800).read())
                    u = r.get("usage") or {}
                    fin = r["choices"][0].get("finish_reason")
                    m = r["choices"][0]["message"]
                    prev = {"role": "assistant", "content": m.get("content") or ""}
                    if m.get("reasoning_content"):
                        prev["reasoning_content"] = m["reasoning_content"]
                    if m.get("tool_calls"):
                        prev["tool_calls"] = [{"id": t.get("id"), "type": "function", "function": {
                            "name": t["function"]["name"], "arguments": t["function"]["arguments"]}} for t in m["tool_calls"]]
                except Exception as e:  # keep walking: a failed call is a result
                    u, fin, err = {}, None, repr(e)[:200]
                    prev = {"role": "assistant", "content": ""}
                t1 = time.time()
                if a.echo:
                    hist.append(prev)
                row = {"tag": a.tag, "agent": n, "instance": iid, "call": k, "t0": t0, "latency_s": round(t1 - t0, 3),
                       "max_tokens": max_tokens, "prompt_tokens": u.get("prompt_tokens"),
                       "completion_tokens": u.get("completion_tokens"), "finish": fin, "error": err,
                       "tool_calls": len((prev or {}).get("tool_calls") or []) if a.echo else None}
                with lock:
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                time.sleep(a.tool_gap)

    start = time.time()
    threads = [threading.Thread(target=agent, args=(n,)) for n in range(a.agents)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - start
    lat = sorted(r["latency_s"] for r in rows if not r["error"])
    comp = sum(r["completion_tokens"] or 0 for r in rows)
    prm = sum(r["prompt_tokens"] or 0 for r in rows)
    q = lambda f: lat[min(len(lat) - 1, int(f * (len(lat) - 1)))] if lat else None
    s = {"tag": a.tag, "summary": True, "convs": len(order), "calls": len(rows), "errors": sum(1 for r in rows if r["error"]),
         "wall_s": round(wall, 1), "completion_tokens": comp, "prompt_tokens": prm, "gen_tps": round(comp / wall, 1),
         "calls_per_min": round(len(rows) / wall * 60, 1), "lat_p50": q(0.5), "lat_p90": q(0.9),
         "lat_mean": round(statistics.mean(lat), 2) if lat else None}
    out.write(json.dumps(s) + "\n")
    print(json.dumps(s))


if __name__ == "__main__":
    main()
