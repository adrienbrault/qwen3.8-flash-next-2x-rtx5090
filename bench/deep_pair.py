#!/usr/bin/env python3
"""Two concurrent greedy requests with an identical deep context, texts kept for comparison.

Exists because bench/probe.py deliberately discards text: it measures throughput. The QSA patch changes attention
at bsz>1, so the arm gate is whether concurrent greedy decoding still produces the same words — within an arm
(determinism under batching) and across arms (the patch changes nothing).

usage: deep_pair.py --url ... --model ... --ctx 120000 --tokens 256 --conc 2 --out-prefix results/greedy-control
"""
import argparse, json, sys, threading, time, urllib.request

FILLER = ("The vault passphrase is harbour lantern cipher sevenths paradox. "
          "Interval buffer kernel tensor latency scheduler prefill decode cache page slot quantisation "
          "attention recurrent window batch token stream fusion pipeline shard expert router gating norm rope. ")
ASK = "\n\nNow write the complete source of a production-quality Python module implementing the interfaces " \
      "described above: every class, every method, full type hints, no commentary. Do not stop early."


def build(ctx_tokens):
    reps = int(ctx_tokens / 1.3 / 26) + 1
    return (FILLER * reps)[: int(ctx_tokens * 4.0)] + ASK


def one(idx, url, model, prompt, ntok, sink, timeout):
    body = json.dumps({"model": model, "max_tokens": ntok, "min_tokens": ntok, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    rec = {"i": idx, "text": "", "usage": None, "err": None}
    try:
        req = urllib.request.Request(url + "/chat/completions", body, {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    c = json.loads(data)
                except Exception:
                    continue
                if c.get("usage"):
                    rec["usage"] = c["usage"]
                for ch in c.get("choices") or []:
                    d = ch.get("delta") or {}
                    # Reasoning first, then content: a capture that keeps only content records nothing while the
                    # model is still thinking, and two empty captures compare equal.
                    rec["text"] += d.get("reasoning_content") or ""
                    rec["text"] += d.get("content") or ""
        rec["wall_s"] = round(time.time() - t0, 2)
        rec["decode_tps"] = round((rec["usage"] or {}).get("completion_tokens", 0) / max(rec["wall_s"], 1e-9), 1)
    except Exception as e:
        rec["err"] = str(e)[:200]
    sink.append(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=120000)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--conc", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()

    prompt = build(a.ctx)
    sink = []
    ths = [threading.Thread(target=one, args=(i, a.url, a.model, prompt, a.tokens, sink, a.timeout))
           for i in range(a.conc)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = round(time.time() - t0, 2)
    for rec in sorted(sink, key=lambda r: r["i"]):
        path = f"{a.out_prefix}.{rec['i']}.txt"
        if len(rec["text"]) < 200:
            print(f"  WARNING: captured text for conc{rec['i']} is {len(rec['text'])} bytes; "
                  f"the equality gate would be vacuous", flush=True)
        open(path, "w").write(rec["text"])
        u = rec["usage"] or {}
        print(f"  conc{rec['i']}: prompt_tokens={u.get('prompt_tokens')} n={u.get('completion_tokens')} "
              f"wall={rec.get('wall_s')}s -> {path}"
              + (f" ERR {rec['err']}" if rec["err"] else ""), flush=True)
    print(f"  round wall {wall}s, {len([r for r in sink if not r['err']])}/{a.conc} ok", flush=True)
    json.dump(sink, open(f"{a.out_prefix}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
