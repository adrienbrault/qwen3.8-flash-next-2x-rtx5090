#!/usr/bin/env python3
"""Decode, concurrency and depth instrument for the Flash-Next TabbyAPI endpoint (stdlib only).

WHY THIS EXISTS, given probes/oai_conc.py and probes/depth_ss.py already exist:

  * oai_conc.py cannot measure a LONG generation. It posts a bare /completions prompt at temperature 0 and lets
    the model stop when it wants to. On this checkpoint that is an EOS after 1-430 tokens (server log, R339), so
    a `--tokens 8192` row measured 8 tokens of work divided by a wall that included prefill and reported
    9.0 t/s. The row looked like a measurement and was arithmetic over a stopping condition.
  * oai_conc.py's summary carries no token counts, so that could only be found in the server log.
  * depth_ss.py reads llama.cpp's `timings` block. TabbyAPI returns none, so every decode rate from it is zero.

The fix for length is `min_tokens`, not `ignore_eos`: `ban_eos_token`/`ignore_eos` is accepted by TabbyAPI and
IGNORED by the exllamav3 backend (common/sampling.py UNSUPPORTED_PARAMS), whereas min_tokens reaches
`Job(min_new_tokens=...)` and suppresses EOS until the count is reached (exllamav3/generator/job.py:459).

Every request is recorded as one JSONL line with its own completed-token count, wall, TTFT and decode window, so
no summary can hide the work done. The server's `usage.completion_tokens` is recorded alongside the client's own
frame count: R338 was an accounting bug in the engine's requeue path, and this is what catches its return.

Decode rate is measured two ways, both from the SSE timestamps:
  wall_tps  = completion_tokens / wall            (includes prefill and queueing; what a user feels)
  decode_tps= (completion_tokens - 1) / (t_last - t_first)   (steady-state decode; comparable to engine numbers)

usage:
  probe.py --url http://127.0.0.1:8022/v1 --model qwen3.8-flash-next-exl3-3.05bpw \
           --conc 1 2 4 8 --tokens 256 --runs 2 --out r339.jsonl
  probe.py ... --tokens 8192 --ctx 30000 --kind code --runs 1     # deep-context, forced-length decode
"""
import argparse, json, statistics, sys, threading, time, urllib.request

PROSE = ("The tide is a standing wave driven by the moon's gravity, and its phase at any port is set by the "
         "shape of the basin, the Coriolis force, and the depth of the water column. ")
CODE = ("def merge(intervals):\n    intervals.sort()\n    out = []\n    for start, end in intervals:\n"
        "        if out and start <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], end)\n"
        "        else:\n            out.append([start, end])\n    return out\n")

ASK_PROSE = ("\n\nUsing only the material above, write an exhaustive technical analysis of it. Do not stop early; "
             "keep the prose dense and specific and continue for as long as you can.")
ASK_CODE = ("\n\nNow write the complete source of a production-quality Python module that implements the "
            "interfaces described above: every class, every method, full type hints, no commentary and no "
            "placeholders. Do not stop until the module is complete and importable.")


NOFORCE = [False]
UNIQUE = [False]
SALT = [0]
# Sampling temperature. 0 (the default) is greedy, which is what every published number in this repo was measured
# at. Real client traffic is sampled -- the desktop harness sends 0.6 -- and MTP draft acceptance is not the same
# under sampling as under argmax, so a greedy-only suite cannot see a sampling-dependent slowdown (R583).
TEMP = [0.0]


def filler(ctx_tokens, kind, salt=None):
    """Deterministic padded passage of roughly ctx_tokens tokens (~1.3 tokens per word)."""
    unit = CODE if kind == "code" else PROSE
    if ctx_tokens <= 0:
        return ""
    if salt is not None:
        # A per-request salt rebuilds the passage from a different RNG stream, so two requests share no token
        # sequence at all. Without it, concurrent requests share the filler and the pool test measures prefix
        # reuse instead of independent context footprint.
        import random
        rnd = random.Random(salt)
        words = [rnd.choice(unit.split()) for _ in range(int(ctx_tokens / 1.6))]
        return " ".join(words) + " "
    reps = int(ctx_tokens / 1.3 / max(len(unit.split()), 1)) + 1
    return (unit * reps)[: int(ctx_tokens * 4.2)]


def one(idx, url, model, prompt, ntok, chat, sink, timeout, distinct=False, no_force=False):
    if UNIQUE[0] and prompt.startswith(PREFIX[0]):
        # REPLACE the shared filler, do not add to it: prepending the per-request passage to the shared one made a
        # "120k" request carry 315,253 tokens and the server rejected it with a 400 (R345, arm 3). The unique
        # passage goes where the shared one was, and the tail (the actual instruction) is kept.
        prompt = filler(CTX[0], KIND[0], salt=1000 + idx + SALT[0]) + prompt[len(PREFIX[0]):]
    if distinct:
        # Threads must not share a prefix: with the paged cache, identical prompts collapse onto the same pages
        # and an admission test then measures prefix reuse instead of concurrent context footprint.
        prompt = prompt + f"\n\n[variant {idx}: treat this document as section {idx} of a series.]"
    body = {"model": model, "max_tokens": ntok, "temperature": TEMP[0], "stream": True,
            "stream_options": {"include_usage": True}}
    # Length is forced here. Without min_tokens this measures whatever length the model felt like producing.
    if not no_force:
        body["min_tokens"] = ntok
    if chat:
        body["messages"] = [{"role": "user", "content": prompt}]
        path = "/chat/completions"
    else:
        body["prompt"] = prompt
        path = "/completions"
    req = urllib.request.Request(url + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    rec = {"i": idx, "ok": False, "max_tokens": ntok, "min_tokens": ntok, "temperature": TEMP[0]}
    t0 = time.time()
    t_first = t_last = None
    frames = 0
    usage = None
    finish = None
    try:
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
                    # THREE shapes exist in the wild and an instrument that knows only one of them reports a
                    # request with no text, which looks like an engine failure: SSE delta (the normal streaming
                    # case), a /completions-style `text`, and a COMPLETE chat completion wrapped in a single SSE
                    # frame as `message` (vLLM does this when it stops streaming mid-request). R342/R349 recorded
                    # seven such responses as "usage, no text, no error" for exactly this reason.
                    d = ch.get("delta") or ch.get("message") or {}
                    # `reasoning_content` is TabbyAPI's name for the thinking channel; vLLM's OpenAI server here
                    # sends the same thing as `reasoning`. Reading only one of the two makes a request whose
                    # entire forced length went into thinking look like a request that returned nothing.
                    txt = (ch.get("text") or d.get("content") or d.get("reasoning_content")
                           or d.get("reasoning"))
                    if txt:
                        if ch.get("message") and not ch.get("delta"):
                            rec["shape"] = "single-message-frame"
                        now = time.time()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        frames += 1
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except Exception as e:
        rec["err"] = str(e)[:200]
        rec["wall_s"] = round(time.time() - t0, 3)
        sink.append(rec)
        return

    wall = time.time() - t0
    n_server = (usage or {}).get("completion_tokens")
    n = n_server if n_server else frames
    rec.update({
        "ok": bool(n) and n > 0 and t_first is not None,
        "wall_s": round(wall, 3),
        "ttft_s": round(t_first - t0, 3) if t_first else None,
        "decode_window_s": round(t_last - t_first, 3) if (t_first and t_last) else None,
        # absolute epoch times (R704): lets an analysis measure how far concurrent streams' decode windows overlap
        "t_start_abs": round(t0, 3),
        "t_first_abs": round(t_first, 3) if t_first else None,
        "t_last_abs": round(t_last, 3) if t_last else None,
        "completion_tokens": n,
        "server_completion_tokens": n_server,
        "client_frames": frames,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "finish_reason": finish,
    })
    rec["wall_tps"] = round(n / wall, 2) if wall else None
    if t_first and t_last and n > 1:
        rec["decode_tps"] = round((n - 1) / (t_last - t_first), 2)
    # Frames are SSE deltas, not tokens: with MTP one frame carries several accepted tokens, so frames must
    # never exceed the token count. Only the direction that cannot happen is worth flagging -- R338 was the
    # engine under-reporting its own generation.
    if n_server is not None and frames and n_server < frames:
        rec["accounting_note"] = f"server reports {n_server} tokens but {frames} frames arrived"
    sink.append(rec)


def round_run(url, model, conc, ntok, chat, prompt, timeout, sink, log, distinct=False):
    ths = [threading.Thread(target=one, args=(i, url, model, prompt, ntok, chat, sink, timeout, distinct, NOFORCE[0]))
           for i in range(conc)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return time.time() - t0


PREFIX = [""]
CTX = [0]
KIND = ["code"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--conc", type=int, nargs="+", default=[1])
    ap.add_argument("--tokens", type=int, default=256, help="forced completion length (min_tokens = this)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--warmup-runs", type=int, default=0,
                    help="full-length rounds per shape run first and not recorded. R520 (2026-09-19): the first measured round"
                         " after a boot read up to 4 %% low, enough to flip the sign of a 0.5 %% A/B")
    ap.add_argument("--ctx", type=int, nargs="+", default=[0],
                    help="filler tokens prepended; a list runs the depth ladder (e.g. --ctx 0 30000 120000)")
    ap.add_argument("--kind", choices=["prose", "code"], default="prose")
    ap.add_argument("--completions", action="store_true", help="use /completions instead of /chat/completions")
    ap.add_argument("--no-force", action="store_true", help="omit min_tokens (reproduces the EOS trap)")
    ap.add_argument("--unique", action="store_true",
                    help="rebuild the filler per request from a different RNG stream: no shared token sequence")
    ap.add_argument("--distinct", action="store_true",
                    help="give every concurrent request a unique suffix so they cannot share cached pages")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--salt", type=int, default=0,
                    help="offset for the --unique filler seed. The seed is otherwise 1000 + request index, identical on every run and"
                         " invocation, so a repeated 'cold' prefill hits the prefix cache and a longer context shares the shorter"
                         " one's opening (R507, 2026-09-18). Give each cold measurement its own salt.")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="sampling temperature; 0 (default) is greedy, which is what every published number here "
                         "was measured at. Pass the client's real value (the desktop harness uses 0.6) to measure "
                         "the regime the user actually runs in.")
    ap.add_argument("--out", required=True, help="JSONL: one line per request, never a summary")
    a = ap.parse_args()
    NOFORCE[0] = a.no_force
    UNIQUE[0] = a.unique
    SALT[0] = a.salt
    TEMP[0] = a.temp

    fill = filler(a.ctx[0], a.kind)
    ask = ASK_CODE if a.kind == "code" else ASK_PROSE
    prompt = (fill + ("\n\n" + ask if fill else ask.strip() + " Begin now."))
    log = open(a.out, "a", buffering=1)
    print(f"tag={a.tag} url={a.url} conc={a.conc} forced_tokens={a.tokens} ctx~{a.ctx} kind={a.kind} "
          f"endpoint={'completions' if a.completions else 'chat'} runs={a.runs} temp={a.temp}", flush=True)

    for ctx in a.ctx:
        fill = filler(ctx, a.kind)
        prompt = (fill + ("\n\n" + ask if fill else ask.strip() + " Begin now."))
        CTX[0] = ctx
        KIND[0] = a.kind
        PREFIX[0] = fill
        if len(a.ctx) > 1:
            print(f"  ### ctx~{ctx}", flush=True)
        for c in a.conc:
            # Warm the batch shape: the engine compiles kernels per shape, an unwarmed round can read 10x low.
            w = []
            round_run(a.url, a.model, c, 8, not a.completions, "Warmup.", a.timeout, w, log, a.distinct)
            ok_w = sum(1 for x in w if x["ok"])
            for _ in range(a.warmup_runs):
                ww = round_run(a.url, a.model, c, a.tokens, not a.completions, prompt, a.timeout, [], log, a.distinct)
                print(f"  c={c} warm-up round ctx~{ctx}: {ww:.1f} s (not recorded)", flush=True)
            for r in range(a.runs):
                sink = []
                wall = round_run(a.url, a.model, c, a.tokens, not a.completions, prompt, a.timeout, sink,
                                 log, a.distinct)
                good = [s2 for s2 in sink if s2["ok"]]
                for s2 in sink:
                    s2.update({"conc": c, "run": r, "round_wall_s": round(wall, 3), "tag": a.tag,
                               "ctx_requested": ctx})
                    log.write(json.dumps(s2) + "\n")
                if not good:
                    print(f"  c={c} run{r} ctx~{ctx}: FAILED all {len(sink)} requests "
                          f"({sink[0].get('err') if sink else 'no records'})", flush=True)
                    continue
                row = {
                    "conc": c, "ctx_requested": ctx, "n_ok": len(good),
                    "tokens_median": statistics.median(s2["completion_tokens"] for s2 in good),
                    "prompt_tokens": good[0].get("prompt_tokens"),
                    "aggregate_tps": round(sum(s2["completion_tokens"] for s2 in good) / wall, 1),
                    "per_stream_wall_tps": round(statistics.median(s2["wall_tps"] for s2 in good), 1),
                    "decode_tps": round(statistics.median(s2["decode_tps"] for s2 in good), 1)
                    if any(s2.get("decode_tps") for s2 in good) else None,
                    "ttft_s": round(statistics.median(s2["ttft_s"] for s2 in good), 3)
                    if any(s2.get("ttft_s") for s2 in good) else None,
                }
                print(f"  c={c} run{r} ctx~{ctx}: " + json.dumps(row), flush=True)
            print(f"  warmup c={c} ctx~{ctx}: {ok_w}/{c} ok", flush=True)
    log.close()


if __name__ == "__main__":
    main()
