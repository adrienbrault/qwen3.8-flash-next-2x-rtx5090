#!/usr/bin/env python3
"""The encode lock (exllamav3 tokenizer.py `with self._encode_lock:`) is what keeps concurrent encodes in opposite
special-token modes correct; this test fails when it is removed.

One patched Tokenizer, two threads: thread 1 encodes chat markup with encode_special_tokens=True (the prompt path:
<|im_start|> -> one special id), thread 2 the same text with encode_special_tokens=False (the loop thread's short
encodes: the chunk fallback, banned strings, /v1/token/encode with the flag off -> <|im_start|> as plain text). Every
result is compared to its single-threaded reference.

The race window is widened DETERMINISTICALLY in both legs: the Tokenizer's HF tokenizer is wrapped so that every
encode / encode_batch call sleeps --delay-ms (default 0.5 ms, GIL released) before the real call, i.e. between
encode_part_base's flag write and the encode that depends on it. Without the lock the other thread then almost surely
flips the flag inside that window; with the lock it cannot. (R808b: with only a 1-us switch interval, the unlocked leg
saw no mismatch in 2 s inside a docker build on a busy CPU although it had on the same image 90 min earlier: the plain
race was timing-dependent.)
  lock ON           : --seconds (default 2 s), 0 mismatches required and >= 50 encodes per thread. HARD gate everywhere.
  lock replaced by a no-op context manager (the mutation): runs until the FIRST mismatch, up to --max-seconds (default
                    20 s); >= 1 mismatch required, i.e. the test can see the race. If none is seen: a hard FAIL on the build
                    host (tests/run_offline.sh), a WARN "sensitivity not demonstrated" inside the image build (this file
                    under /opt/tokenize-offloop-*: a busy build CPU cannot prove the lock's absence matters, and that
                    leg says nothing about the lock that ships). --sensitivity fail|warn overrides the auto choice.

  test_lock.py [--tokenizer-dir DIR ...] [--seconds 2] [--max-seconds 20] [--delay-ms 0.5] [--sensitivity auto|fail|warn]
"""
import argparse
import contextlib
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _load  # noqa: E402

FAILS = []
WARNS = []


class SlowHF:
    """Forwards everything to the HF tokenizer (the encode_special_tokens flag included); encode / encode_batch sleep
    first, which widens the gap between the wrapper's flag write and its encode."""

    def __init__(self, real, delay):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_delay", delay)

    def __getattr__(self, k):
        return getattr(self._real, k)

    def __setattr__(self, k, v):
        setattr(self._real, k, v)

    def encode(self, *a, **k):
        time.sleep(self._delay)
        return self._real.encode(*a, **k)

    def encode_batch(self, *a, **k):
        time.sleep(self._delay)
        return self._real.encode_batch(*a, **k)


def race(tok, text, seconds, refs, until_mismatch=False):
    stop = threading.Event()
    stats = {True: [0, 0], False: [0, 0]}   # special -> [encodes, mismatches]

    def worker(special):
        ref = refs[special]
        while not stop.is_set():
            ids = tok.encode(text, encode_special_tokens=special)
            stats[special][0] += 1
            if ids.shape[-1] != ref.shape[-1] or not bool((ids == ref).all()):
                stats[special][1] += 1
                if until_mismatch:
                    stop.set()
    ths = [threading.Thread(target=worker, args=(sp,)) for sp in (True, False)]
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    t0 = time.perf_counter()
    try:
        for t in ths:
            t.start()
        stop.wait(seconds)
        stop.set()
        for t in ths:
            t.join()
    finally:
        sys.setswitchinterval(old)
    return stats, time.perf_counter() - t0


def run(model_dir, label, a, sensitivity):
    os.environ.pop("EXL3_TOKENIZE_OFFLOOP", None)
    src = _load.load_tokenizer_module("src")
    tok = src.Tokenizer(_load.make_config(model_dir))
    text = "<|im_start|>user\nHi <think>x</think><|im_end|>\n<|im_start|>assistant\n" * 8
    refs = {sp: tok.encode(text, encode_special_tokens=sp) for sp in (True, False)}
    if refs[True].shape[-1] == refs[False].shape[-1] and bool((refs[True] == refs[False]).all()):
        FAILS.append(f"{label}: the two modes give the same ids on the test text: the test cannot see a race")
        return
    tok.tokenizer = SlowHF(tok.tokenizer, a.delay_ms / 1000)
    on, t_on = race(tok, text, a.seconds, refs)
    real = tok._encode_lock
    tok._encode_lock = contextlib.nullcontext()
    try:
        off, t_off = race(tok, text, a.max_seconds, refs, until_mismatch=True)
    finally:
        tok._encode_lock = real
    fmt = lambda st: ", ".join(f"special={sp}: {n} encodes, {m} wrong" for sp, (n, m) in st.items())  # noqa: E731
    print(f"{label}: lock ON  -> {fmt(on)} in {t_on:.1f} s ({a.delay_ms:g} ms window)")
    print(f"{label}: lock OFF -> {fmt(off)} in {t_off:.1f} s (mutation; stops at the first mismatch, cap {a.max_seconds:g} s)")
    if any(m for _, m in on.values()) or min(n for n, _ in on.values()) < 50:
        FAILS.append(f"{label}: with the lock: {fmt(on)}")
    if not any(m for _, m in off.values()):
        msg = f"{label}: without the lock no mismatch in {t_off:.1f} s: sensitivity not demonstrated"
        (WARNS if sensitivity == "warn" else FAILS).append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", action="append", default=[])
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--delay-ms", type=float, default=0.5)
    ap.add_argument("--sensitivity", choices=["auto", "fail", "warn"], default="auto")
    a = ap.parse_args()
    sens = a.sensitivity
    if sens == "auto":
        sens = "warn" if os.path.abspath(__file__).startswith("/opt/tokenize-offloop-") else "fail"
    run(_load.build_synth_tokenizer(), "synthetic", a, sens)
    for d in a.tokenizer_dir:
        run(d, os.path.basename(os.path.normpath(d)) or d, a, sens)
    for w in WARNS:
        print("WARN:", w)
    for f in FAILS:
        print("FAIL:", f)
    n = 2 * (1 + len(a.tokenizer_dir))
    print(f"test_lock: {n - len(FAILS) - len(WARNS)} passed, {len(FAILS)} failed"
          + (f", {len(WARNS)} warned (sensitivity not demonstrated; the lock-ON leg is the gate)" if WARNS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
