#!/usr/bin/env python3
"""
tool-choice-r1 box probe: sends the box-ab-spec.md request matrix to one
TabbyAPI server and writes a JSON report. Standard library only.

  python3 box_probe.py --url http://127.0.0.1:PORT --salt S --out unpatched.json --identity-only
  python3 box_probe.py --url http://127.0.0.1:PORT --salt S --out patched.json
  python3 box_probe.py --compare unpatched.json patched.json

Use the same --salt for both servers and a new one per A/B session, boot both
servers fresh and send nothing else to either before the probe: generation on
this stack depends on the server's history (see box-ab-spec.md), so the A/B is
only meaningful for identical request sequences since boot. Gate it with a
same-image control (A vs A') first.

Sections:
  identity    tool_choice absent/"auto"/"none", greedy, stream and not, thinking
              on and off. Every request kind carries its own salt at the start
              of the prompt (first tool's description) so its first send is
              cold on either server: no prefix shared with any other request.
              Each kind is sent twice; the A/B compares the cold first sends,
              and the warm repeat is reported as within-server repeatability.
              The body of every request is a pure function of (salt, key) and
              its sha256 is stored per row; --compare refuses rows whose
              hashes differ.
  greedy      thinking off, logprobs + top_logprobs 2 on fresh salts: every
              content token must be the top-1 token (argmax sampling reached
              the sampler); the smallest top-1/top-2 logprob gap is reported
              (near ties are where history-dependent numerics flip a greedy
              output). Runs after identity so it does not change its history.
  required    tool_choice "required": every response must carry a parseable call;
              content is allowed before the call, never after it
  named       named function: exactly one call, to that function
  answer      second turn after the tool result, tool_choice "required": reports
              how often the answer (56) is surfaced in content next to the call
  validation  requests that must be rejected with HTTP 400
  cap         required with a small reasoning budget (phase-2 continuation)
  concurrent  4 simultaneous required streams plus 4 mixed (2 auto, 2 required)
"""

import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request

PROMPT = "What is 7 times 8?"


def tool(name, description, **props):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": v, "description": k} for k, v in props.items()},
                "required": list(props),
            },
        },
    }


TOOLS = [
    tool("calculator", "Evaluate an arithmetic expression", expression="string"),
    tool("get_weather", "Current weather for a city", city="string"),
    tool("search_web", "Search the web", query="string"),
    tool("read_file", "Read a file", path="string"),
    tool("write_file", "Write a file", path="string", content="string"),
    tool("send_email", "Send an email", to="string", subject="string", body="string"),
    tool("get_time", "Current time in a timezone", timezone="string"),
    tool("translate", "Translate text", text="string", target_language="string"),
    tool("create_event", "Create a calendar event", title="string", start="string"),
    tool("list_files", "List files in a directory", directory="string"),
    tool("run_code", "Run Python code", code="string"),
    tool("get_stock_price", "Current stock price", symbol="string"),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def salted_tools(salt):
    tools = json.loads(json.dumps(TOOLS))
    tools[0]["function"]["description"] += f" [probe {salt}]"
    return tools


def body(choice="absent", stream=False, thinking=True, tools=True, salt=None, messages=None, **extra):
    user = PROMPT if salt is None else f"{PROMPT} [probe {salt}]"
    msgs = messages or [{"role": "user", "content": user}]
    b = {"messages": msgs, "stream": stream, "max_tokens": 2048}
    if tools:
        b["tools"] = TOOLS if salt is None else salted_tools(salt)
    if choice != "absent":
        b["tool_choice"] = choice
    if not thinking:
        b["chat_template_kwargs"] = {"enable_thinking": False}
    if stream:
        b["stream_options"] = {"include_usage": True}
    b.update(extra)
    return b


def body_sha256(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def post(url, payload, timeout=900):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if payload.get("stream"):
                return _read_stream(resp, t0)
            data = json.loads(resp.read())
            return {"status": resp.status, "json": data, "elapsed": time.time() - t0}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": e.read().decode(errors="replace"), "elapsed": time.time() - t0}


def _read_stream(resp, t0):
    chunks, first_token = [], None
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            chunks.append("[DONE]")
            break
        chunk = json.loads(data)
        if first_token is None and chunk.get("choices") and chunk["choices"][0].get("delta"):
            first_token = time.time() - t0
        chunks.append(chunk)
    return {"status": resp.status, "chunks": chunks, "elapsed": time.time() - t0, "ttft": first_token}


def summarize(result):
    """Message-level view of a streamed or plain response."""
    if result.get("status") != 200:
        return {"status": result.get("status"), "error": result.get("error")}
    if "json" in result:
        choice = result["json"]["choices"][0]
        msg = choice["message"]
        return {
            "status": 200,
            "finish_reason": choice.get("finish_reason"),
            "eos_reason": choice.get("eos_reason"),
            "stop_str": choice.get("stop_str"),
            "content": msg.get("content"),
            "reasoning": msg.get("reasoning_content"),
            "tool_calls": [c["function"] for c in msg.get("tool_calls") or []],
            "usage": result["json"].get("usage"),
        }
    content, reasoning, calls, finish, order, usage, eos = "", "", [], [], [], None, None
    for c in result["chunks"]:
        if c == "[DONE]":
            order.append("done")
            continue
        if c.get("usage"):
            usage = c["usage"]
        for ch in c.get("choices", []):
            d = ch.get("delta") or {}
            if d.get("reasoning_content"):
                reasoning += d["reasoning_content"]
                order.append("r")
            if d.get("content"):
                content += d["content"]
                order.append("c")
            if d.get("tool_calls"):
                calls += [t["function"] for t in d["tool_calls"]]
                order.append("t")
            if ch.get("finish_reason"):
                finish.append(ch["finish_reason"])
                eos = ch.get("eos_reason", eos)
    # Collapse runs: e.g. r,r,r,t,done -> r t done
    shape = [x for i, x in enumerate(order) if i == 0 or order[i - 1] != x]
    return {
        "status": 200,
        "finish_reason": finish[0] if finish else None,
        "eos_reason": eos,
        "finish_count": len(finish),
        "content": content or None,
        "reasoning": reasoning or None,
        "tool_calls": calls,
        "shape": " ".join(shape),
        "usage": usage,
    }


def normalized(result):
    """What must be byte-identical between images: the full payload minus ids/times."""
    if result.get("status") != 200:
        return {"status": result.get("status"), "error": result.get("error")}

    def scrub(d):
        d = json.loads(json.dumps(d))
        for k in ("id", "created"):
            d.pop(k, None)
        usage = d.get("usage")
        if usage:
            d["usage"] = {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        for ch in d.get("choices", []):
            for part in (ch.get("message") or {}, ch.get("delta") or {}):
                for tc in part.get("tool_calls") or []:
                    tc.pop("id", None)
        return d

    if "json" in result:
        return scrub(result["json"])
    return [c if c == "[DONE]" else scrub(c) for c in result["chunks"]]


def call_ok(summary, allowed):
    """A forced turn: one finish, at least one parseable call to an allowed tool,
    content (if any) only before the call."""
    calls = summary.get("tool_calls") or []
    if summary.get("status") != 200 or not calls or summary.get("finish_reason") != "tool_calls":
        return False
    for fn in calls:
        if fn["name"] not in allowed:
            return False
        try:
            json.loads(fn["arguments"])
        except (TypeError, ValueError):
            return False
    shape = summary.get("shape")
    if shape is not None:
        # Streaming: content deltas may precede the call chunk, never follow it
        kinds = shape.split()
        if "t" in kinds and "c" in kinds[kinds.index("t") + 1:]:
            return False
    return summary.get("finish_count", 1) == 1


def identity_salt(salt, choice, stream, thinking):
    """The per-key prompt salt: a pure function of (salt, key), no counters."""
    return f"{salt}:{choice}:{int(stream)}:{int(thinking)}"


def greedy_row(i, payload, result):
    """Every content token must be the top-1 token of its position."""
    row = {"key": i, "prompt_sha256": body_sha256(payload), "status": result.get("status")}
    if result.get("status") != 200:
        row.update(ok=False, error=result.get("error"))
        return row
    choice = result["json"]["choices"][0]
    entries = ((choice.get("logprobs") or {}).get("content")) or []
    not_top1, gaps = [], []
    for n, e in enumerate(entries):
        tops = e.get("top_logprobs") or []
        if not tops:
            continue
        best = max(tops, key=lambda t: t["logprob"])
        if e.get("token_id") != best.get("token_id") and e["logprob"] < best["logprob"]:
            not_top1.append(n)
        if len(tops) >= 2:
            ranked = sorted((t["logprob"] for t in tops), reverse=True)
            gaps.append(ranked[0] - ranked[1])
    row.update(
        ok=bool(entries) and not not_top1,
        tokens=len(entries),
        not_top1=not_top1,
        min_top_gap=min(gaps) if gaps else None,
        content=choice["message"].get("content"),
        finish_reason=choice.get("finish_reason"),
    )
    return row


def run(url, trials, identity_only, salt):
    report = {"url": url, "salt": salt, "identity": [], "required": [], "named": [], "answer": [],
              "validation": [], "cap": [], "concurrent": []}
    # adaptive_target 1.0: a preset's (unforced) adaptive-P would otherwise
    # sample even at temperature 0 (backends/exllamav3/sampler.py build())
    greedy = {"temperature": 0, "top_k": 1, "seed": 1234, "adaptive_target": 1.0}
    report["greedy"] = []
    seq = 0

    for choice in ("absent", "auto", "none"):
        for stream in (False, True):
            for thinking in (True, False):
                key_salt = identity_salt(salt, choice, stream, thinking)
                for rep in (1, 2):
                    payload = body(choice, stream, thinking, salt=key_salt, **greedy)
                    r = post(url, payload)
                    seq += 1
                    report["identity"].append(
                        {"key": [choice, stream, thinking, rep], "seq": seq,
                         "prompt_sha256": body_sha256(payload), "normalized": normalized(r),
                         "summary": summarize(r), "elapsed": r.get("elapsed"), "ttft": r.get("ttft")}
                    )

    for i in (1, 2):
        payload = body("auto", False, False, salt=f"{salt}:greedy:{i}", logprobs=True, top_logprobs=2, **greedy)
        r = post(url, payload)
        report["greedy"].append(greedy_row(i, payload, r))
    if identity_only:
        return report

    for stream in (False, True):
        for thinking in (True, False):
            for i in range(trials):
                r = post(url, body("required", stream, thinking))
                s = summarize(r)
                report["required"].append(
                    {"key": [stream, thinking, i], "ok": call_ok(s, TOOL_NAMES), "summary": s,
                     "elapsed": r.get("elapsed"), "ttft": r.get("ttft")}
                )

    named = {"type": "function", "function": {"name": "get_weather"}}
    for stream in (False, True):
        for thinking in (True, False):
            r = post(url, body(named, stream, thinking))
            s = summarize(r)
            ok = call_ok(s, {"get_weather"}) and len(s["tool_calls"]) == 1
            report["named"].append({"key": [stream, thinking], "ok": ok, "summary": s})

    # The tool-eval TC-45 shape: required on every turn, so the turn after the
    # tool result must carry the answer as content next to its (forced) call
    second_turn = [
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_probe_1", "type": "function",
             "function": {"name": "calculator", "arguments": json.dumps({"expression": "7 * 8"})}}]},
        {"role": "tool", "tool_call_id": "call_probe_1", "content": "56"},
    ]
    for stream in (False, True):
        for i in range(trials):
            r = post(url, body("required", stream, True, messages=second_turn))
            s = summarize(r)
            report["answer"].append({
                "key": [stream, i],
                "ok": call_ok(s, TOOL_NAMES),
                "answer_in_content": "56" in (s.get("content") or ""),
                "summary": s,
            })

    cases = {
        "required_without_tools": body("required", tools=False),
        "named_without_tools": body(named, tools=False),
        "named_unknown": body({"type": "function", "function": {"name": "nope"}}),
        "required_with_json_schema": body(
            "required", response_format={"type": "json_schema", "json_schema": {"type": "object"}}
        ),
    }
    for name, payload in cases.items():
        r = post(url, payload)
        report["validation"].append({"case": name, "ok": r.get("status") == 400, "status": r.get("status")})

    for stream in (False, True):
        r = post(url, body("required", stream, True, reasoning_budget_tokens=64))
        s = summarize(r)
        report["cap"].append({"stream": stream, "ok": call_ok(s, TOOL_NAMES), "summary": s,
                              "elapsed": r.get("elapsed"), "ttft": r.get("ttft")})

    def burst(choices):
        out = [None] * len(choices)

        def worker(i, c):
            r = post(url, body(c, True, True))
            out[i] = {"choice": c, "summary": summarize(r), "elapsed": r.get("elapsed")}

        threads = [threading.Thread(target=worker, args=(i, c)) for i, c in enumerate(choices)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for o in out:
            o["ok"] = call_ok(o["summary"], TOOL_NAMES) if o["choice"] == "required" else o["summary"]["status"] == 200
        return out

    report["concurrent"] = burst(["required"] * 4) + burst(["auto", "required", "auto", "required"])
    return report


def print_summary(report):
    for section in ("greedy", "required", "named", "answer", "validation", "cap", "concurrent"):
        rows = report.get(section) or []
        if rows:
            ok = sum(1 for r in rows if r["ok"])
            print(f"{section:11s} {ok}/{len(rows)} ok")
    answers = report.get("answer") or []
    if answers:
        n = sum(1 for r in answers if r["answer_in_content"])
        print(f"answer      56 in content {n}/{len(answers)} (informational)")
    for g in report.get("greedy") or []:
        print(f"greedy #{g['key']}   {g.get('tokens')} content tokens, not top-1 at {g.get('not_top1')}, "
              f"smallest top-1/top-2 gap {g.get('min_top_gap')}")
    ident = report.get("identity") or []
    if ident:
        by_key = {}
        for row in ident:
            by_key.setdefault(tuple(row["key"][:3]), []).append(row["normalized"])
        stable = sum(1 for v in by_key.values() if len(v) == 2 and v[0] == v[1])
        print(f"identity    warm repeat == cold first send for {stable}/{len(by_key)} request kinds "
              "(within-server repeatability, not part of the A/B)")


def compare(path_a, path_b):
    """
    A/B of the identity rows. The cold first sends (rep 1) are the verdict; the
    warm repeats (rep 2) are compared too and reported separately. Rows must
    have been sent with byte-identical bodies at the same position of the
    request sequence, or the comparison is refused.
    """
    a, b = (json.load(open(p)) for p in (path_a, path_b))
    if a.get("salt") is None or a.get("salt") != b.get("salt"):
        print(f"salts differ ({a.get('salt')} vs {b.get('salt')}): prompts are not the same, no comparison")
        return 2
    rows_a = {tuple(r["key"]): r for r in a["identity"]}
    rows_b = {tuple(r["key"]): r for r in b["identity"]}
    if set(rows_a) != set(rows_b):
        print("the two reports hold different identity keys, no comparison")
        return 2
    for key, ra in rows_a.items():
        rb = rows_b[key]
        if ra.get("prompt_sha256") is None or ra.get("prompt_sha256") != rb.get("prompt_sha256"):
            print(f"request body differs for {list(key)} ({ra.get('prompt_sha256')} vs {rb.get('prompt_sha256')}), no comparison")
            return 2
        if ra.get("seq") != rb.get("seq"):
            print(f"{list(key)} was request {ra.get('seq')} on A but {rb.get('seq')} on B, no comparison")
            return 2
    result = 0
    for rep, what in ((1, "cold first sends (verdict)"), (2, "warm repeats (informational)")):
        same = diff = 0
        for key in sorted(k for k in rows_a if k[3] == rep):
            if rows_a[key]["normalized"] == rows_b[key]["normalized"]:
                same += 1
            else:
                diff += 1
                cross = [k2 for k2 in rows_b if k2[:3] == key[:3] and rows_b[k2]["normalized"] == rows_a[key]["normalized"]]
                note = f" (A's output equals B's rep {cross[0][3]})" if cross else ""
                print(f"DIFF {list(key)} sha256 {rows_a[key]['prompt_sha256'][:12]}{note}")
        print(f"identity A/B, {what}: {same} identical, {diff} different")
        if rep == 1 and diff:
            result = 1
    ga = {g["key"]: g for g in a.get("greedy") or []}
    gb = {g["key"]: g for g in b.get("greedy") or []}
    for k in sorted(set(ga) & set(gb)):
        same = ga[k].get("content") == gb[k].get("content")
        print(f"greedy #{k}: A ok={ga[k].get('ok')} B ok={gb[k].get('ok')}, content {'identical' if same else 'DIFFERENT'}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--out")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--identity-only", action="store_true")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--salt", help="same value for A and B; default: a new one per run (printed)")
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    salt = args.salt or f"{int(time.time() * 1000) % 100000000:08d}"
    print(f"salt {salt}")
    report = run(args.url, args.trials, args.identity_only, salt)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print_summary(report)


if __name__ == "__main__":
    main()
