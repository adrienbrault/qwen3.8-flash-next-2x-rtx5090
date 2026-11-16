#!/usr/bin/env python3
"""Where the decode curve's tokens go: thinking channel or answer.

Sends the decode curve's code and prose requests exactly as fn_bench.py builds them (chat, thinking at the template
default, greedy, min_tokens = max_tokens = 1,024, the --distinct suffix of request 0) and records, per request, the
streamed frames and characters in the thinking channel (`reasoning_content`) and in the answer (`content`), the
finish reason and the server's completion count, plus the first 300 characters of each channel.

    python3 thinking_split.py --url http://127.0.0.1:8023/v1 --model <id> --fn-bench fn_bench.py > thinking-split.jsonl
"""
import argparse
import importlib.util
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--fn-bench", required=True)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("fn_bench", a.fn_bench)
fb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fb)
# Match fn_bench request 0 at ctx=0: the task instruction, then the distinct-stream suffix.
# CODE/PROSE are context filler, not the curve's ctx=0 request; their earlier split is withdrawn.
for kind, ask in (("code", fb.ASK_CODE), ("prose", fb.ASK_PROSE)):
    prompt = ask.strip() + " Begin now." + "\n\n[variant 0: treat this document as section 0 of a series.]"
    body = {"model": a.model, "max_tokens": 1024, "min_tokens": 1024, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    rec = dict(kind=kind, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), reasoning_frames=0, reasoning_chars=0,
               content_frames=0, content_chars=0, finish_reason=None, completion_tokens=None, reasoning_head="", content_head="")
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                rec["completion_tokens"] = d["usage"].get("completion_tokens")
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                for key, name in (("reasoning_content", "reasoning"), ("reasoning", "reasoning"), ("content", "content")):
                    t = delta.get(key)
                    if t:
                        rec[name + "_frames"] += 1
                        rec[name + "_chars"] += len(t)
                        if len(rec[name + "_head"]) < 300:
                            rec[name + "_head"] = (rec[name + "_head"] + t)[:300]
                if ch.get("finish_reason"):
                    rec["finish_reason"] = ch["finish_reason"]
    print(json.dumps(rec), flush=True)
