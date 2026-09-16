#!/usr/bin/env python3
"""Long-context retrieval gate for the Flash-Next TabbyAPI endpoint (stdlib only).

Why not the existing probes: `fn_needle.py` on the box is a llama.cpp instrument (pooled-key cache era) and reads
llama.cpp response fields. This one speaks the OpenAI chat API only.

What it does: builds a deterministic document of approximately `--ctx-tokens` tokens, plants one unique fact (a
six-word passphrase) at a chosen fraction of the document, asks for it by name in a chat turn, and checks the
answer. Repeats at several fractions so a pass cannot come from the fact landing near the prompt's end, where any
positional shortcut would score. The server's `prompt_tokens` is recorded next to the requested depth, so a
"100k" arm that actually sent 40k tokens is visible rather than assumed.

This measures retrieval, not decode speed: the answer is short and the run is not a throughput sample.

usage:
  needle.py --url http://127.0.0.1:8022/v1 --model qwen3.8-flash-next-exl3-3.05bpw \
            --ctx-tokens 131072 --fracs 0.08 0.3 0.55 0.8 0.96 --out r340.jsonl
"""
import argparse, json, random, time, urllib.request

WORDS = ("interval buffer kernel tensor latency scheduler prefill decode cache page slot quantisation attention "
         "recurrent window batch token stream kernel fusion pipeline shard expert router gating norm rope").split()
PASSPHRASE = "harbour lantern cipher sevenths paradox"


def build_doc(ctx_tokens, frac, seed):
    """Deterministic filler with the passphrase planted once, at `frac` of the document."""
    rnd = random.Random(seed)
    n_words = int(ctx_tokens * 0.75)
    words = [rnd.choice(WORDS) for _ in range(n_words)]
    at = max(1, min(len(words) - 1, int(len(words) * frac)))
    words.insert(at, f"The vault passphrase is {PASSPHRASE}. ")
    return " ".join(words), at


def ask(url, model, doc, timeout):
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True},
            "temperature": 0, "max_tokens": 64,
            "messages": [{"role": "user", "content":
                          doc + "\n\nQuestion: what is the vault passphrase? Answer with the passphrase only."}]}
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                {"Content-Type": "application/json"})
    t0 = time.time()
    text, usage, finish = "", None, None
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
                usage = c["usage"]
            for ch in c.get("choices") or []:
                d = ch.get("delta") or {}
                text += d.get("content") or ""
                text += d.get("reasoning_content") or ""
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return text, usage, round(time.time() - t0, 2), finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="needle")
    ap.add_argument("--ctx-tokens", type=int, nargs="+", default=[131072])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.08, 0.3, 0.55, 0.8, 0.96])
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log = open(a.out, "a", buffering=1)
    for ctx in a.ctx_tokens:
        hits = 0
        for frac in a.fracs:
            doc, at = build_doc(ctx, frac, seed=7)
            try:
                text, usage, wall, finish = ask(a.url, a.model, doc, a.timeout)
                hits += PASSPHRASE in text
                rec = {"tag": a.tag, "ctx_requested": ctx, "frac": frac, "planted_at_word": at,
                       "prompt_tokens": (usage or {}).get("prompt_tokens"),
                       "completion_tokens": (usage or {}).get("completion_tokens"),
                       "wall_s": wall, "finish_reason": finish, "hit": PASSPHRASE in text,
                       "answer": text.strip()[:200]}
            except Exception as e:
                rec = {"tag": a.tag, "ctx_requested": ctx, "frac": frac, "hit": False,
                       "err": str(e)[:200]}
            log.write(json.dumps(rec) + "\n")
            print(f"  ctx~{ctx} frac={frac}: {'HIT ' if rec.get('hit') else 'MISS'} "
                  f"prompt_tokens={rec.get('prompt_tokens')} wall={rec.get('wall_s')}s "
                  f"{rec.get('err', '')}", flush=True)
        print(f"  {a.tag} ctx~{ctx}: {hits}/{len(a.fracs)} retrieved", flush=True)
    log.close()


if __name__ == "__main__":
    main()
