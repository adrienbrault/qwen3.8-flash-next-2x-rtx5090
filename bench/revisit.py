#!/usr/bin/env python3
"""Prefix-revisit probe: N distinct long sessions sent one after another (so the first is evicted from a VRAM pool smaller
than N x ctx), then session 0 is sent again. Greedy, forced output length. Reports TTFT per request and whether the
revisit's output equals the first answer (a restored KV page must give the same greedy continuation)."""
import argparse, json, random, time, urllib.request, hashlib
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--sessions", type=int, default=5); ap.add_argument("--words", type=int, default=68000)
ap.add_argument("--tokens", type=int, default=64); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--nonce", type=int, default=int(time.time()) % 100000)
a = ap.parse_args()
W = "harbor lantern quartz meadow copper violet summit ember glacier falcon orchard tundra saffron cobalt willow basalt canyon juniper".split()
def prompt(k):
    rng = random.Random(a.nonce * 1000 + k)
    body = " ".join(rng.choice(W) for _ in range(a.words))
    return f"Session {a.nonce}-{k}. Archive document follows.\n\n{body}\n\nSummarize the document above in two sentences."
def ask(p):
    body = {"model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": a.tokens, "min_tokens": a.tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; text = ""; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
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
                t = (dd.get("content") or "") + (dd.get("reasoning_content") or dd.get("reasoning") or "")
                if t and tf is None: tf = time.time()
                text += t
    return {"ttft_s": (tf or time.time()) - t0, "prompt_tokens": (usage or {}).get("prompt_tokens"), "sha": hashlib.sha256(text.encode()).hexdigest()[:16]}
log = open(a.out, "a")
first = None
order = list(range(a.sessions)) + [0, 1]
for i, k in enumerate(order):
    r = ask(prompt(k)); r.update({"tag": a.tag, "step": i, "session": k, "revisit": i >= a.sessions})
    if i == 0: first = r["sha"]
    if i == a.sessions: r["same_as_first"] = (r["sha"] == first)
    log.write(json.dumps(r) + "\n"); log.flush()
    print(f"[{a.tag}] step {i} session {k}{' (revisit)' if r['revisit'] else ''}: {r['prompt_tokens']} prompt tokens, TTFT {r['ttft_s']:.2f} s"
          + (f", output same as first: {r['same_as_first']}" if 'same_as_first' in r else ""), flush=True)
