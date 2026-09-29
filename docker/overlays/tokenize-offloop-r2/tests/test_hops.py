#!/usr/bin/env python3
"""tokenize-offloop r2: a short prompt takes ZERO executor hops; a long one hops once (streamed) and never blocks the loop.

REVIEW-R806 sec. 5: every run_in_executor hop cost the arriving request ~30-40 ms of TTFT under 4-stream decode (class
(iii), ~85-token prompts: BOTH 255.9 vs NONE 214.3 ms), because the loop picks the worker's result up only between two
synchronous iterate() calls. r2 hops only above EXL3_TOKENIZE_OFFLOOP_MIN_CHARS (default 12000). Through the patched
request path (test_event_loop's harness: router step A = run_tokenize(prompt_chars(prompt), check_context_length, ...),
step B = stream_generate -> generate_gen up to a stub AsyncJob), counting common.tokenize_offloop.hops and the thread
every Tokenizer.encode runs on:
  1. default knobs, short prompts (chat markup, 300 and 11,999 chars, and exactly MIN_CHARS): streamed A + B = 0 hops,
     1 encode, on the loop thread; non-streamed B alone = 0 hops, 1 encode, on the loop thread; ids == the served encode
  2. MIN_CHARS + 1 chars: streamed = 1 hop (A, on the worker) and B reuses the ids (0 more hops, 1 encode in all);
     non-streamed = 1 hop (B on the worker)
  3. a list prompt (/v1/completions) is measured by its total length (prompt_chars)
  4. EXL3_TOKENIZE_OFFLOOP_MIN_CHARS=0: a short prompt hops (r1's behaviour); '', 'abc', '-5', '1_000', ' 12000 ',
     '+50', non-ASCII digits -> the default 12000;
     =50: a 300-char prompt hops
  5. EXL3_TOKENIZE_OFFLOOP=0: 0 hops at any length (the knobs-off control), encodes on the loop thread
  6. busy worker: while a long encode runs on the worker, a short prompt's check ALSO goes to the worker (1 hop) instead of
     waiting inline on the exllamav3 encode lock the worker holds, and the 1-ms ticker's largest gap stays small; once
     the worker is idle again the next short prompt is inline (0 hops); a queued call cancelled before it ran and a call
     that raised both leave the busy count at 0
  7. ids identical to the served tokenizer.py in every case above

  test_hops.py [--tokenizer-dir DIR ...]
"""
import argparse
import asyncio
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_event_loop as TE  # noqa: E402  (the patched-app harness: stubs, container, steps, ticker)
import _load  # noqa: E402

TO, CM, TM, TB = TE.TO, TE.CM, TE.TM, TE.TB
check = TE.check
MAIN = threading.get_ident()


def count_threads(tok):
    """Wrap Tokenizer.encode on this instance: record the thread of every call."""
    inner = type(tok).encode.__get__(tok)
    calls = []

    def encode(*a, **k):
        calls.append(threading.get_ident())
        return inner(*a, **k)
    tok.encode = encode
    return calls


def text_of(n, seed=""):
    """Exactly n characters of chat-shaped text (markup included) ending without a partial special token."""
    unit = f"<|im_start|>user\nNote {seed}: the storage engine flushes pages to disk.<|im_end|>\n"
    s = (unit * (n // len(unit) + 2))
    s = s[:n]
    return s


def clear_env():
    for k in ("TABBY_ENCODE_ONCE", "EXL3_TOKENIZE_OFFLOOP", "EXL3_TOKENIZE_OFFLOOP_MIN_CHARS"):
        os.environ.pop(k, None)


async def path(c, calls, prompt, streamed, ref_tok, label):
    """-> (hops, encodes, encode threads) for one request; checks the ids AsyncJob receives."""
    params = TE.BaseSamplerRequest(max_tokens=16)
    h0, n0 = TO.hops, len(calls)
    if streamed:
        r = await TE.step_a(prompt, params)
        check(not isinstance(r, Exception), f"{label}: check raised {r!r}")
    rb = await TE.step_b(c, prompt, params)
    ok = isinstance(rb, dict) and rb.get("input_ids") is not None
    check(ok, f"{label}: generate_gen did not reach AsyncJob")
    if ok:
        ref = ref_tok.encode(prompt, add_bos=False, encode_special_tokens=True)
        ids = rb["input_ids"][0]
        check(tuple(ids.shape) == tuple(ref.shape) and bool((ids == ref).all()), f"{label}: ids != the served encode")
    return TO.hops - h0, len(calls) - n0, calls[n0:]


async def suite(model_dir, label):
    clear_env()
    tok = TM.Tokenizer(_load.make_config(model_dir))
    ref_tok = TB.Tokenizer(_load.make_config(model_dir))
    c = TE.make_container(tok)
    CM.container = c
    calls = count_threads(tok)
    M = TO.DEFAULT_MIN_CHARS
    check(TO.offload_min_chars() == M == 12000, f"{label}: default MIN_CHARS {TO.offload_min_chars()} (want 12000)")

    # 1. short prompts: zero hops, encoded inline on the loop thread
    for n in (300, M - 1, M):
        p = text_of(n, f"{label}-s{n}")
        check(len(p) == n, f"{label}: text_of({n}) has {len(p)} chars")
        for streamed in (True, False):
            h, e, th = await path(c, calls, p, streamed, ref_tok, f"{label} {n} chars streamed={streamed}")
            check(h == 0 and e == 1 and th == [MAIN], f"{label}: {n}-char prompt streamed={streamed}: {h} hops, {e} "
                                                    f"encodes, on {'the loop thread' if th == [MAIN] else 'another thread'}"
                                                    f" (want 0 hops, 1 encode, loop thread)")
    print(f"  {label}: 300 / {M - 1} / {M} chars, streamed and non-streamed: 0 executor hops, 1 encode on the loop thread")

    # 2. one char over the threshold: one hop (the check on the worker; generate_gen reuses its ids)
    p = text_of(M + 1, f"{label}-l")
    h, e, th = await path(c, calls, p, True, ref_tok, f"{label} {M + 1} streamed")
    check(h == 1 and e == 1 and th and th[0] != MAIN, f"{label}: {M + 1} chars streamed: {h} hops, {e} encodes "
                                                      f"(want 1 hop, 1 encode on the worker)")
    h, e, th = await path(c, calls, p, False, ref_tok, f"{label} {M + 1} non-streamed")
    check(h == 1 and e == 1 and th and th[0] != MAIN, f"{label}: {M + 1} chars non-streamed: {h} hops, {e} encodes")
    print(f"  {label}: {M + 1} chars: streamed 1 hop (check on the worker, ids reused), non-streamed 1 hop")

    # 3. list prompts count their total length
    check(TO.prompt_chars(["a" * 7000, "b" * 6000]) == 13000 and TO.prompt_chars("xyz") == 3
          and TO.prompt_chars([1, "ab"]) == 2 and TO.prompt_chars(None) == 0, f"{label}: prompt_chars")
    h0 = TO.hops
    await TO.run_tokenize(TO.prompt_chars(["a" * 7000, "b" * 6000]), lambda: None)
    await TO.run_tokenize(TO.prompt_chars(["a" * 7000, "b" * 4000]), lambda: None)
    check(TO.hops - h0 == 1, f"{label}: list prompts 13,000 / 11,000 chars -> {TO.hops - h0} hops (want 1)")

    # 4. the threshold knob
    short = text_of(300, f"{label}-k")
    os.environ["EXL3_TOKENIZE_OFFLOOP_MIN_CHARS"] = "0"
    h, e, th = await path(c, calls, short, True, ref_tok, f"{label} MIN_CHARS=0")
    check(h == 1 and e == 1 and th[0] != MAIN, f"{label}: MIN_CHARS=0: {h} hops (want 1: r1's behaviour)")
    for bad in ("", "abc", "-5", "1.5", "1_000", " 12000 ", "12000\n", "+50", "\u0661\u0662"):
        os.environ["EXL3_TOKENIZE_OFFLOOP_MIN_CHARS"] = bad
        check(TO.offload_min_chars() == M, f"{label}: MIN_CHARS={bad!r} -> {TO.offload_min_chars()} (want the default)")
    os.environ["EXL3_TOKENIZE_OFFLOOP_MIN_CHARS"] = "50"
    h, e, th = await path(c, calls, short, False, ref_tok, f"{label} MIN_CHARS=50")
    check(h == 1 and e == 1, f"{label}: MIN_CHARS=50, 300 chars non-streamed: {h} hops (want 1)")
    os.environ.pop("EXL3_TOKENIZE_OFFLOOP_MIN_CHARS")

    # 5. knobs off: never a hop
    os.environ["EXL3_TOKENIZE_OFFLOOP"] = "0"
    for n in (300, 4 * M):
        p = text_of(n, f"{label}-o{n}")
        for streamed in (True, False):
            h, e, th = await path(c, calls, p, streamed, ref_tok, f"{label} OFFLOOP=0 {n}")
            check(h == 0 and set(th) == {MAIN}, f"{label}: EXL3_TOKENIZE_OFFLOOP=0 {n} chars streamed={streamed}: {h} hops")
    os.environ.pop("EXL3_TOKENIZE_OFFLOOP")
    print(f"  {label}: MIN_CHARS=0 hops a 300-char prompt; bad values -> {M}; EXL3_TOKENIZE_OFFLOOP=0: 0 hops at any length")

    # 6. busy worker: a short prompt queues behind the long encode instead of blocking the loop on the encode lock
    long_p = _load.long_text(600_000)
    t0 = time.perf_counter()
    tok.encode(long_p, add_bos=False, encode_special_tokens=True)
    T = time.perf_counter() - t0
    small = max(0.015, 0.3 * T)

    async def busy_then_short():
        pl = TE.BaseSamplerRequest(max_tokens=16)
        ps = TE.BaseSamplerRequest(max_tokens=16)
        h0 = TO.hops
        tl = asyncio.create_task(TE.step_a(long_p, pl))
        await asyncio.sleep(0.005)                         # the long check is on the worker now
        busy = TO._pending
        await TE.step_a(text_of(300, f"{label}-b"), ps)
        await tl
        return TO.hops - h0, busy, ps
    (hb, busy, ps), gap = await TE.with_ticker(busy_then_short)
    check(busy >= 1 and hb == 2, f"{label}: busy worker: pending {busy}, hops {hb} (want >= 1 and 2: the short check "
                                 f"queued on the worker)")
    check(gap <= small, f"{label}: busy worker: loop gap {gap * 1000:.1f} ms > {small * 1000:.1f} ms (the short check "
                        f"blocked the loop)")
    check(TO._pending == 0, f"{label}: pending {TO._pending} after the busy test (want 0)")
    h, e, th = await path(c, calls, text_of(300, f"{label}-a"), True, ref_tok, f"{label} after busy")
    check(h == 0 and th == [MAIN], f"{label}: idle again: a short prompt took {h} hops (want 0)")
    print(f"  {label}: busy worker ({T * 1000:.0f} ms encode): the short check queued on the worker (hops {hb}), "
          f"max loop gap {gap * 1000:.1f} ms; idle again -> inline")

    # cancelled-before-running and raising calls release the busy count
    async def cancel_queued():
        slow = asyncio.create_task(TO.run_tokenize(TE.BIG, time.sleep, 0.05))
        await asyncio.sleep(0.005)
        q = asyncio.create_task(TO.run_tokenize(TE.BIG, time.sleep, 0.05))
        await asyncio.sleep(0.001)
        q.cancel()
        try:
            await q
        except asyncio.CancelledError:
            pass
        await slow

        def boom():
            raise RuntimeError("x")
        try:
            await TO.run_tokenize(TE.BIG, boom)
        except RuntimeError:
            pass
    await cancel_queued()
    check(TO._pending == 0, f"{label}: pending {TO._pending} after a cancelled queued call and a raising call (want 0)")
    clear_env()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", action="append", default=[])
    a = ap.parse_args()
    n0 = TE.NCHECK[0]
    asyncio.run(suite(_load.build_synth_tokenizer(), "synthetic"))
    for d in a.tokenizer_dir:
        asyncio.run(suite(d, os.path.basename(os.path.normpath(d)) or d))
    n = TE.NCHECK[0] - n0
    print(f"test_hops: {n - len(TE.FAILS)} passed, {len(TE.FAILS)} failed")
    return 1 if TE.FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
