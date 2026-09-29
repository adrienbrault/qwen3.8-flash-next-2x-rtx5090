#!/usr/bin/env python3
"""GIL release: while the patched exllamav3 Tokenizer encodes a long prompt on one thread, a concurrent pure-Python
thread keeps >= 80 % of its idle rate (probes/gil_tokenizer_test.py's measure, but through Tokenizer.encode as TabbyAPI
calls it: encode_special_tokens=True, add_bos). Sensitivity control: the same measure with EXL3_TOKENIZE_OFFLOOP=0 (the
served encode(), which holds the GIL) must stay <= 60 %, or the test could not see the difference.

  test_gil.py [--tokenizer-dir DIR ...]      (synthetic tokenizer always; real tokenizers as given)
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _load  # noqa: E402

ON_MIN = 0.80
OFF_MAX = 0.60
FAILS = []


def busy(stop, out):
    c = 0
    while not stop.is_set():
        c += 1
    out.append(c)


def idle_rate():
    stop, out = threading.Event(), []
    th = threading.Thread(target=busy, args=(stop, out))
    th.start()
    time.sleep(0.4)
    stop.set()
    th.join()
    return out[0] / 0.4


def rate_while(fn):
    stop, out = threading.Event(), []
    th = threading.Thread(target=busy, args=(stop, out))
    th.start()
    time.sleep(0.005)
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    stop.set()
    th.join()
    return dt, out[0] / (dt + 0.005)


def run(model_dir, label, chars):
    src = _load.load_tokenizer_module("src")
    tok = src.Tokenizer(_load.make_config(model_dir))
    text = _load.long_text(chars)
    res = {}
    for knob in ("1", "0"):
        os.environ["EXL3_TOKENIZE_OFFLOOP"] = knob
        fn = lambda: tok.encode(text, add_bos=True, encode_special_tokens=True)  # noqa: E731
        n = fn().shape[-1]
        base = idle_rate()
        runs = sorted(rate_while(fn) for _ in range(3))
        dt, rate = runs[1][0], sorted(r[1] for r in runs)[1]
        res[knob] = rate / base
        print(f"{label}: EXL3_TOKENIZE_OFFLOOP={knob}: {n} tokens in {dt * 1000:.1f} ms; concurrent Python thread at "
              f"{rate / base * 100:.0f} % of its idle rate")
    os.environ.pop("EXL3_TOKENIZE_OFFLOOP", None)
    if res["1"] < ON_MIN:
        FAILS.append(f"{label}: patched encode leaves the other thread {res['1'] * 100:.0f} % < {ON_MIN * 100:.0f} %")
    if res["0"] > OFF_MAX:
        FAILS.append(f"{label}: control (served encode) leaves {res['0'] * 100:.0f} % > {OFF_MAX * 100:.0f} %: the "
                     f"measure cannot tell a held GIL from a released one here")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", action="append", default=[])
    ap.add_argument("--chars", type=int, default=400_000)
    a = ap.parse_args()
    run(_load.build_synth_tokenizer(), "synthetic", a.chars)
    for d in a.tokenizer_dir:
        run(d, os.path.basename(os.path.normpath(d)) or d, a.chars)
    for f in FAILS:
        print("FAIL:", f)
    n = 2 * (1 + len(a.tokenizer_dir))
    print(f"test_gil: {n - len(FAILS)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
