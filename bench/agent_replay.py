#!/usr/bin/env python3
"""Agent-shaped serving probe: replay recorded mini-SWE-agent conversations against an OpenAI-compatible endpoint.

Each of --agents workers takes conversations from a fixed, hash-ordered list and walks them call by call: the prompt for call k
is the recorded messages before the k-th assistant turn (system, user, assistant turns with their reasoning and tool calls, tool
results), sent with the mini-SWE-agent bash tool, greedy, max_tokens = the recorded turn's length (estimated, capped). The
server's own generation is discarded, so call k+1 shares a prefix with call k up to the end of call k's prompt -- the cache
reuse an agent gets when its re-rendered turn differs from what the server generated. --tool-gap seconds of sleep between calls
stand in for tool execution. Same conversations, same order, same budget in every arm.

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
            for k, idx in enumerate(calls[:a.calls]):
                max_tokens = max(64, min(a.cap, msgs[idx]["_est_tokens"]))
                body = {"model": a.model, "messages": clean(msgs[:idx]), "tools": [BASH_TOOL], "temperature": 0,
                        "max_tokens": max_tokens}
                t0 = time.time()
                err = None
                try:
                    r = json.loads(urllib.request.urlopen(urllib.request.Request(
                        a.url + "/chat/completions", data=json.dumps(body).encode(),
                        headers={"Content-Type": "application/json"}), timeout=1800).read())
                    u = r.get("usage") or {}
                    fin = r["choices"][0].get("finish_reason")
                except Exception as e:  # keep walking: a failed call is a result
                    u, fin, err = {}, None, repr(e)[:200]
                t1 = time.time()
                row = {"tag": a.tag, "agent": n, "instance": iid, "call": k, "t0": t0, "latency_s": round(t1 - t0, 3),
                       "max_tokens": max_tokens, "prompt_tokens": u.get("prompt_tokens"),
                       "completion_tokens": u.get("completion_tokens"), "finish": fin, "error": err}
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
