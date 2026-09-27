#!/usr/bin/env python3
"""Replay one Hermes (webui) turn against an OpenAI-compatible endpoint, N times, and classify each response.

Rebuilds the request Hermes sent for message index --upto of a webui session: the session's system prompt (state.db,
by hash), messages[0:upto] (user api_content, assistant content + tool_calls + reasoning_content, tool results) and
Hermes' tool schemas (--tools, a JSON file with a "tools" list, e.g. samples-r779.json). Streams like Hermes and sends
no sampler, so the server's preset applies. Per run: finish_reason, reasoning/content lengths, tool calls, and the
failure class R781 is about: finish "stop" with no content and no tool call ("reasoning-only stop").
A --forced-loop mode instead sends one prompt that asks for a repeated line inside the thinking and prefills the
thinking with 40 copies of it (TabbyAPI response_prefix), a mechanism check for the loop detector. --prefix-file
replaces that prefill (either mode), --prompt the forced-mode question, --no-think sends enable_thinking false.
Usage (completion_tokens) is requested with stream_options and recorded per run.
Stdlib only. Writes one JSON line per run (full reasoning and content) to --out.
"""
import argparse, concurrent.futures as cf, json, sqlite3, time, urllib.request


def build_messages(session_json, db, upto):
    d = json.load(open(session_json))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    h = con.execute("select system_prompt_hash from sessions where id=?", (d["session_id"],)).fetchone()[0]
    system = con.execute("select prompt from system_prompts where hash=?", (h,)).fetchone()[0]
    out = [{"role": "system", "content": system}]
    for m in d["messages"][:upto]:
        r = m["role"]
        if r == "user":
            out.append({"role": "user", "content": m.get("api_content") or m["content"]})
        elif r == "assistant":
            a = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("reasoning_content"):
                a["reasoning_content"] = m["reasoning_content"]
            if m.get("tool_calls"):
                a["tool_calls"] = [{"id": t["id"], "type": "function",
                                    "function": {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}}
                                   for t in m["tool_calls"]]
            out.append(a)
        elif r == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
    return out, d["messages"][upto] if upto < len(d["messages"]) else None


def stream(url, body, timeout):
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); reasoning = []; content = []; tools = {}; finish = None; ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            ch = json.loads(line[5:])
            usage = ch.get("usage") or usage
            for c in ch.get("choices", []):
                dl = c.get("delta") or {}
                if dl.get("reasoning_content"):
                    reasoning.append(dl["reasoning_content"]); ttft = ttft or time.time() - t0
                if dl.get("content"):
                    content.append(dl["content"]); ttft = ttft or time.time() - t0
                for t in dl.get("tool_calls") or []:
                    e = tools.setdefault(t.get("index", 0), {"name": "", "arguments": ""})
                    f = t.get("function") or {}
                    e["name"] += f.get("name") or ""; e["arguments"] += f.get("arguments") or ""
                finish = c.get("finish_reason") or finish
    return {"finish": finish, "reasoning": "".join(reasoning), "content": "".join(content),
            "tool_calls": list(tools.values()), "wall": round(time.time() - t0, 2), "ttft": ttft,
            "completion_tokens": (usage or {}).get("completion_tokens")}


def classify(r):
    if r.get("error"):
        return "error"
    if r["tool_calls"]:
        return "tool_call"
    if r["content"].strip():
        return "content"
    return "reasoning_only_" + str(r["finish"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--session"); ap.add_argument("--db", default="/srv/hermes/state/state.db")
    ap.add_argument("--upto", type=int); ap.add_argument("--tools")
    ap.add_argument("--forced-loop", action="store_true")
    ap.add_argument("--prompt"); ap.add_argument("--prefix-file"); ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32768); ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--tag", default="run"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.forced_loop:
        msgs = [{"role": "user", "content": a.prompt or "Inside your thinking, before anything else, write the exact line "
                 "'the quick brown fox jumps over the lazy dog' 400 times, one per line, without skipping any. "
                 "Then give the final answer to: what is 17*23? Reply with the number only."}]
        # The model deliberates instead of complying (R781 control: 0/2 looped), so the thinking is prefilled with
        # 40 copies of the line through TabbyAPI's response_prefix; in-context copying then keeps the loop going.
        line = "the quick brown fox jumps over the lazy dog\n"
        body = {"model": a.model, "messages": msgs, "stream": True, "max_tokens": a.max_tokens,
                "response_prefix": "I will write the line 400 times first.\n\n" + line * 40}
    else:
        msgs, orig = build_messages(a.session, a.db, a.upto)
        body = {"model": a.model, "messages": msgs, "stream": True, "max_tokens": a.max_tokens,
                "tools": json.load(open(a.tools))["tools"], "reasoning_effort": "medium"}
        print(f"replay: {len(msgs)} messages, original reply finish={orig and orig.get('finish_reason')}")

    body["stream_options"] = {"include_usage": True}
    if a.prefix_file:
        body["response_prefix"] = open(a.prefix_file).read()
    if a.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    def one(i):
        try:
            r = stream(a.url, body, a.timeout)
        except Exception as e:
            r = {"error": repr(e)}
        r["i"] = i; r["class"] = classify(r); r["tag"] = a.tag
        return r

    rows = []
    with cf.ThreadPoolExecutor(a.conc) as ex, open(a.out, "a") as f:
        for r in ex.map(one, range(a.n)):
            rows.append(r); f.write(json.dumps(r) + "\n"); f.flush()
            print(f"{a.tag} #{r['i']}: {r['class']} finish={r.get('finish')} R={len(r.get('reasoning', ''))} "
                  f"C={len(r.get('content', ''))} tok={r.get('completion_tokens')} tools={[t['name'] for t in r.get('tool_calls', [])]} wall={r.get('wall')} "
                  f"| C: {r.get('content', '')[:80]!r}", flush=True)
    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print(f"RESULT {a.tag}: n={len(rows)} {counts}")


if __name__ == "__main__":
    main()
