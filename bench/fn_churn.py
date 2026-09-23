#!/usr/bin/env python3
"""Cancel/churn driver for the Flash-Next TabbyAPI endpoint (stdlib only).

WHY THIS EXISTS: the recurrent-state slot pool (num_slots = max_batch_size) is exercised by job
teardown paths that ordinary benchmarks never produce -- a client disconnect mid-stream, a job
requeued past max_rq_tokens (output_chunking => 2048 tokens) through the stash/restore path, a
prefix revived from the page cache after other traffic. A gate that only measures throughput
cannot see a pool that leaks one slot per such event until exhaustion (R586c, 2,380 x
"Cannot create new state" in 27 min).

Worker shapes per iteration (weighted):
  short   : salted prompt, forced 128-512 tokens, read to completion.
  long    : salted prompt, forced 2049-8192 tokens -> at least one requeue, each requeue
            stashes the recurrent state and the resume restores it via new_from_stashed.
  prefix  : the SAME ~8K-token prefix across workers + 256 forced tokens -> prompt-cache
            page reuse and stash lookup on a shared chain.
  cancel  : any of the above read for a few frames, then the socket closed -> the server
            sees a client disconnect and cancels the job mid-decode or mid-prefill.

Every attempt appends one JSONL record; the summary counts by shape/outcome. The pool's own
health is server-side evidence (docker logs: "no available slots", "returned slot",
"released twice", restart count); this probe produces the load and the client-side taxonomy.

usage:
  fn_churn.py --url http://127.0.0.1:8022/v1 --model NAME --workers 12 --minutes 20 \
      --out churn.jsonl [--cancel-rate 0.25] [--seed 7]
"""
import argparse, json, random, statistics, sys, threading, time, urllib.request

PROSE = ("The tide is a standing wave driven by the moon's gravity, and its phase at any port is set by the "
         "shape of the basin, the Coriolis force, and the depth of the water column. ")

def filler(ctx_tokens, salt):
    rnd = random.Random(salt)
    words = [rnd.choice(PROSE.split()) for _ in range(int(ctx_tokens / 1.6))]
    return " ".join(words) + " "

SHARED_PREFIX = filler(8000, salt=1) + "\n\nSummarize the material above in a few sentences."
ASK = ("\n\nContinue this passage in the same register, in detail, for as long as you can.")


def post_stream(url, model, prompt, ntok, stop_after_frames=None, timeout=180):
    """One streaming request. Returns a record dict; stop_after_frames closes the socket early
    (a client disconnect -- the server's cancel path)."""
    body = {"model": model, "prompt": prompt, "max_tokens": ntok, "min_tokens": ntok,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    rec = {"ok": False, "max_tokens": ntok}
    t0 = time.time()
    frames = 0
    usage = None
    cancelled = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                if stop_after_frames is not None and frames >= stop_after_frames:
                    cancelled = True
                    r.close()
                    break
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
                    if (ch.get("text") or ch.get("delta", {}).get("content")):
                        frames += 1
    except Exception as e:
        rec["err"] = str(e)[:200]
    rec.update({
        "ok": bool(usage) and not cancelled and "err" not in rec,
        "cancelled": cancelled,
        "wall_s": round(time.time() - t0, 3),
        "frames": frames,
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "http_err": rec.get("err"),
    })
    return rec


def worker(w, url, model, deadline, cancel_rate, sink, seed):
    rnd = random.Random(seed * 7919 + w)
    while time.time() < deadline:
        roll = rnd.random()
        if roll < 0.55:
            shape, ntok, prompt = "short", rnd.choice([128, 256, 384, 512]), filler(rnd.choice([200, 400, 800]), seed + w) + ASK
        elif roll < 0.75:
            # >2048 forced tokens guarantees at least one requeue (max_rq_tokens = chunk_size = 2048):
            # each requeue stashes the recurrent state and the resume restores it.
            shape, ntok, prompt = "long", rnd.choice([2049, 3072, 4096, 6144, 8192]), filler(rnd.choice([200, 800, 2000]), seed * 3 + w) + ASK
        else:
            shape, ntok, prompt = "prefix", 256, SHARED_PREFIX
        do_cancel = rnd.random() < cancel_rate
        stop = rnd.randint(2, 8) if do_cancel else None
        if do_cancel and shape == "short":
            shape = "cancel-short"
        elif do_cancel:
            shape = "cancel-" + shape
        rec = post_stream(url, model, prompt, ntok, stop_after_frames=stop)
        rec["w"] = w
        rec["shape"] = shape
        rec["t"] = round(time.time(), 1)
        sink.append(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--cancel-rate", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    url = a.url.rstrip("/")

    lock = threading.Lock()
    class Sink(list):
        def append(self, x):
            with lock:
                super().append(x)
    sink = Sink()
    deadline = time.time() + a.minutes * 60
    threads = [threading.Thread(target=worker, args=(i, url, a.model, deadline, a.cancel_rate, sink, a.seed), daemon=True)
               for i in range(a.workers)]
    for t in threads: t.start()
    for t in threads: t.join()

    with open(a.out, "w") as f:
        for r in sorted(sink, key=lambda r: r.get("t", 0)):
            f.write(json.dumps(r) + "\n")

    by = {}
    for r in sink:
        k = (r["shape"], "ok" if r.get("ok") else ("cancelled" if r.get("cancelled") else "err"))
        by[k] = by.get(k, 0) + 1
    for (shape, outcome), n in sorted(by.items()):
        print(f"  {shape:14s} {outcome:9s} {n}")
    errs = [r for r in sink if r.get("err")]
    if errs:
        from collections import Counter
        for e, n in Counter(r["err"][:80] for r in errs).most_common(8):
            print(f"  err x{n}: {e}")
    print(f"total {len(sink)} | ok {sum(1 for r in sink if r.get('ok'))} | "
          f"cancelled {sum(1 for r in sink if r.get('cancelled'))} | err {len(errs)}")


if __name__ == "__main__":
    main()
