#!/usr/bin/env python3
"""Hot-slot TTFT / stall probe (R549): "at c1..c4, does TTFT stay flat, or does the 8-bit KV pool stall once all four
slots are hot?" Stdlib only; prompt filler from fn_bench.filler (salted, so no prefix sharing between requests).

  A  short prompts, c1..c4 simultaneous: TTFT, per-stream decode, aggregate
  B  ~30k-token prompts, c1..c4 simultaneous (all cold, distinct): TTFT spread = prefill queueing
  C  late arrival with N = 0..3 slots already decoding ~150k contexts: TTFT + decode of a short request and TTFT of a
     30k request; the hot streams' longest inter-frame gap and frame rate before vs during the arrival's prefill
  D  full pool: 4 x ~190k simultaneous: TTFT each, per-stream decode while all four decode, longest gap, errors

One JSON line per phase result in --out; a readable summary on stdout.
"""
import argparse
import importlib.util
import json
import statistics as st
import sys
import threading
import time
import urllib.request
from pathlib import Path

SHORT = "Write a detailed explanation of how a hash map handles collisions, with examples."
TAIL = "\n\nSummarize the passage above in three sentences, then continue writing an original story inspired by it."


def load_filler(path):
    spec = importlib.util.spec_from_file_location("fn_bench_hs", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fn_bench_hs"] = mod
    spec.loader.exec_module(mod)
    return mod.filler


class Stream:
    """One streamed chat request; records the time of every content frame."""

    def __init__(self, url, model, prompt, ntok, stop = None):
        self.url, self.model, self.prompt, self.ntok = url, model, prompt, ntok
        self.stop = stop or threading.Event()
        self.t0 = self.t_first = None
        self.frames = []
        self.usage = self.err = self.finish = None
        self.th = threading.Thread(target = self.run, daemon = True)

    def start(self):
        self.th.start()
        return self

    def run(self):
        body = {"model": self.model, "max_tokens": self.ntok, "min_tokens": self.ntok, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": self.prompt}]}
        req = urllib.request.Request(self.url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        self.t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout = 3600) as r:
                for raw in r:
                    if self.stop.is_set():
                        self.finish = "client-abort"
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
                        self.usage = c["usage"]
                    for ch in c.get("choices") or []:
                        d = ch.get("delta") or ch.get("message") or {}
                        if ch.get("text") or d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                            now = time.time()
                            if self.t_first is None:
                                self.t_first = now
                            self.frames.append(now)
                        if ch.get("finish_reason"):
                            self.finish = ch["finish_reason"]
        except Exception as e:
            self.err = str(e)[:200]

    def join(self, timeout = None):
        self.th.join(timeout)

    @property
    def ttft(self):
        return round(self.t_first - self.t0, 3) if self.t_first else None

    def decode_tps(self):
        n = (self.usage or {}).get("completion_tokens")
        if not n or len(self.frames) < 2:
            return None
        return round((n - 1) / (self.frames[-1] - self.frames[0]), 1)

    def rate(self, a, b):
        """frames/s inside [a, b]"""
        k = sum(1 for t in self.frames if a <= t <= b)
        return k / (b - a) if b > a else None

    def max_gap(self, a, b):
        fs = [t for t in self.frames if a <= t <= b]
        return round(max((y - x for x, y in zip(fs, fs[1:])), default = 0.0), 3)


def wait_first(streams, timeout = 1800):
    end = time.time() + timeout
    while time.time() < end:
        if all(s.t_first or s.err for s in streams):
            return True
        time.sleep(0.05)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required = True)
    ap.add_argument("--model", required = True)
    ap.add_argument("--pool", type = int, required = True)
    ap.add_argument("--fn-bench", default = str(Path(__file__).with_name("probe.py")), help = "fn_bench (bench/probe.py), for its prompt filler")
    ap.add_argument("--salt", type = int, default = int(time.time()) % 100000)
    ap.add_argument("--hot-ctx", type = int, default = 150000)
    ap.add_argument("--full-ctx", type = int, default = 190000)
    ap.add_argument("--phases", default = "ABCD")
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    filler = load_filler(a.fn_bench)
    out = open(a.out, "a")
    seq = [a.salt * 100]

    def ctx_prompt(n):
        seq[0] += 1
        return filler(n, "prose", salt = seq[0]) + TAIL

    def short_prompt():
        seq[0] += 1
        return f"[request {seq[0]}] " + SHORT

    def emit(rec):
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(json.dumps(rec), flush = True)

    if "A" in a.phases:
        for conc in (1, 2, 3, 4):
            for run in range(2):
                ss = [Stream(a.url, a.model, short_prompt(), 1024).start() for _ in range(conc)]
                for s in ss: s.join()
                wall = max(s.frames[-1] for s in ss if s.frames) - min(s.t0 for s in ss)
                ntok = sum((s.usage or {}).get("completion_tokens", 0) for s in ss)
                emit({"phase": "A", "conc": conc, "run": run, "ttft": [s.ttft for s in ss], "decode_tps": [s.decode_tps() for s in ss],
                      "aggregate_tps": round(ntok / wall, 1), "prompt_tokens": [(s.usage or {}).get("prompt_tokens") for s in ss],
                      "errors": [s.err for s in ss if s.err]})
    if "B" in a.phases:
        for conc in (1, 2, 3, 4):
            ss = [Stream(a.url, a.model, ctx_prompt(30000), 256).start() for _ in range(conc)]
            for s in ss: s.join()
            emit({"phase": "B", "conc": conc, "ttft": sorted(s.ttft or -1 for s in ss), "decode_tps": [s.decode_tps() for s in ss],
                  "prompt_tokens": [(s.usage or {}).get("prompt_tokens") for s in ss], "errors": [s.err for s in ss if s.err]})
    if "C" in a.phases:
        hot_ctx = min(a.hot_ctx, int(a.pool * 0.92 / 4))
        for nhot in (0, 1, 2, 3):
            stop = threading.Event()
            hot = []
            for _ in range(nhot):   # one at a time: each hot stream is decoding before the next one is sent
                s = Stream(a.url, a.model, ctx_prompt(hot_ctx), 8192, stop).start(); hot.append(s)
                wait_first([s])
            time.sleep(5)
            t_arr = time.time()
            p = Stream(a.url, a.model, short_prompt(), 512).start(); p.join()
            time.sleep(3)
            q = Stream(a.url, a.model, ctx_prompt(30000), 64).start(); q.join()
            t_end = time.time()
            rec = {"phase": "C", "hot": nhot, "hot_ctx": hot_ctx, "short_ttft": p.ttft, "short_decode_tps": p.decode_tps(),
                   "ctx30k_ttft": q.ttft, "ctx30k_prompt_tokens": (q.usage or {}).get("prompt_tokens"),
                   "errors": [s.err for s in hot + [p, q] if s.err]}
            if hot:
                q_pref = (q.t0, q.t_first or t_end)   # the 30k arrival's prefill window
                rec.update({
                    "hot_rate_before": [round(s.rate(t_arr - 5, t_arr), 2) for s in hot],
                    "hot_rate_during_short": [round(s.rate(p.t0, p.frames[-1] if p.frames else t_end) or 0, 2) for s in hot],
                    "hot_rate_during_30k_prefill": [round(s.rate(*q_pref) or 0, 2) for s in hot],
                    "hot_max_gap_during_30k_prefill": [s.max_gap(q_pref[0] - 1, q_pref[1] + 1) for s in hot],
                    "hot_max_gap_before": [s.max_gap(t_arr - 5, t_arr) for s in hot]})
            stop.set()
            for s in hot: s.join(60)
            time.sleep(8)   # let aborted jobs leave the slots
            emit(rec)
    if "D" in a.phases:
        # --full-ctx is in fn_bench filler units (~0.75 server tokens each, R549); cap the server-token total at 93 % of the pool
        full_ctx = min(a.full_ctx, int(a.pool * 0.93 / 4 / 0.75))
        ss = [Stream(a.url, a.model, ctx_prompt(full_ctx), 1024).start() for _ in range(4)]
        for s in ss: s.join()
        firsts = [s.t_first for s in ss if s.t_first]
        ends = [s.frames[-1] for s in ss if s.frames]
        w = (max(firsts), min(ends)) if len(firsts) == 4 and len(ends) == 4 else None
        rec = {"phase": "D", "ctx_each": full_ctx, "prompt_tokens": [(s.usage or {}).get("prompt_tokens") for s in ss],
               "ttft": sorted(s.ttft or -1 for s in ss), "decode_tps_whole": [s.decode_tps() for s in ss],
               "finish": [s.finish for s in ss], "errors": [s.err for s in ss if s.err]}
        if w and w[1] > w[0]:
            tpf = [((s.usage or {}).get("completion_tokens") or 0) / max(len(s.frames), 1) for s in ss]
            rec.update({"all_hot_window_s": round(w[1] - w[0], 2),
                        "all_hot_decode_tps": [round(s.rate(*w) * t, 1) for s, t in zip(ss, tpf)],
                        "all_hot_max_gap": [s.max_gap(*w) for s in ss]})
        emit(rec)
    out.close()


if __name__ == "__main__":
    main()
