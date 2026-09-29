#!/usr/bin/env python3
"""Event loop + encode-once through the patched TabbyAPI request path (CPU; the model, sampler and job are stubbed).

A ticker coroutine sleeps 1 ms at a time on the event loop (a stand-in for AsyncGenerator._run_iteration's per-turn
iterate()) while a long prompt goes through the two request-path steps exactly as the router and the collector call
them:
  A  await run_tokenize(prompt_chars(prompt), common.model.check_context_length, prompt, params)   (router.py, streamed)
  B  ExllamaV3Container.stream_generate(..., request.model_copy(deep=True)) -> generate_gen up to AsyncJob(...)
     (the chat path's per-choice deep copy and entry point; AsyncJob is a stub that captures input_ids and stops)
with the patched exllamav3 Tokenizer (synthetic Qwen-shaped tokenizer; plus --tokenizer-dir). Per knob arm
(TABBY_ENCODE_ONCE x EXL3_TOKENIZE_OFFLOOP):
  - the ids AsyncJob receives == the served tokenizer.py's encode of the prompt (dtype, shape, values);
  - Tokenizer.encode calls over A + B: 1 with ENCODE_ONCE on, 2 off;
  - the ticker's largest gap in each step: small (<= max(15 ms, 0.3 x T), T = the solo encode time) where the arm
    moves or removes that step's encode, >= 0.7 x T where the arm leaves it inline (the served path: the control that
    shows the measure sees the stall).
Plus: a worker thread with a GIL-holding encode (exllamav3's encode() forced) still stalls the loop (why encode_batch
is needed); the reuse keys (another prompt string, add_bos, multimodal content, a deep-copied request) miss or hit as
designed; an oversized prompt raises the same exception type and message under every arm; run_tokenize keeps
submission order, copies contextvars, propagates exceptions and survives a cancelled await; the router source awaits
both check_context_length calls through run_tokenize and has no bare one left.

  test_event_loop.py [--tokenizer-dir DIR ...] [--chars N]
"""
import argparse
import ast
import asyncio
import contextvars
import copy
import gc
import os
import sys
import time
import types
import weakref

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _load  # noqa: E402

FAILS = []
NCHECK = [0]


def check(cond, what):
    NCHECK[0] += 1
    if not cond:
        FAILS.append(what)
        print("FAIL:", what)
    return cond


# ------------------------------------------------------------------------------------------------ setup
APP = _load.app_dir()
os.chdir(APP)
sys.path.insert(0, APP)
_pkg = _load.install_exl3_stubs()
TM = _load.load_tokenizer_module("src")
TB = _load.load_tokenizer_module("base")
_pkg.Tokenizer = TM.Tokenizer
import backends.exllamav3.model as M  # noqa: E402
import common.model as CM  # noqa: E402
import common.tokenize_offloop as TO  # noqa: E402
from common.errors import ContextLengthExceededError  # noqa: E402,F401
from common.sampling import BaseSamplerRequest  # noqa: E402


class Reached(Exception):
    pass


class FakeJob:
    """Stands in for exllamav3's AsyncJob: records the input ids generate_gen built, then stops the generator."""
    last = None

    def __init__(self, generator, **kw):
        FakeJob.last = kw
        raise Reached()


class FakeSamplerBuilder:
    settings = []

    @classmethod
    def from_params(cls, params, tokenizer, max_seq_len):
        return cls()

    def build(self, greedy):
        return None


M.AsyncJob = FakeJob
M.ExllamaV3SamplerBuilder = FakeSamplerBuilder


def make_container(tok, max_seq_len=1 << 20):
    c = M.ExllamaV3Container()
    c.tokenizer = tok
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    c.hf_model = types.SimpleNamespace(add_bos_token=lambda: False, eos_tokens=lambda: [eos])
    c.config = types.SimpleNamespace(eos_token_id_list=[eos])
    c.max_seq_len = max_seq_len
    c.max_rq_tokens = 2048
    c.cache = types.SimpleNamespace(max_num_tokens=1 << 22)
    c.generator = types.SimpleNamespace(generator=types.SimpleNamespace(recurrent_checkpoint_interval=2048,
                                                                          recurrent_cache=None))
    c.loaded = True
    return c


def count_encodes(tok):
    """Wrap the wrapper's top-level encode on this instance (validate_context_length and _encode_prompt both look it
    up as self.tokenizer.encode at call time)."""
    inner = type(tok).encode.__get__(tok)
    n = [0]

    def encode(*a, **k):
        n[0] += 1
        return inner(*a, **k)
    tok.encode = encode
    return n


def set_knobs(once, offloop):
    os.environ["TABBY_ENCODE_ONCE"] = once
    os.environ["EXL3_TOKENIZE_OFFLOOP"] = offloop


async def with_ticker(coro_fn):
    """Runs coro_fn() on the loop while a 1-ms ticker runs; returns (result or exception, max gap in s)."""
    stop = asyncio.Event()
    gaps = []

    async def tick():
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.001)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now
    t = asyncio.create_task(tick())
    await asyncio.sleep(0.02)
    gaps.clear()
    try:
        res = await coro_fn()
    except Exception as e:  # noqa: BLE001
        res = e
    await asyncio.sleep(0.01)
    stop.set()
    await t
    return res, max(gaps) if gaps else 0.0


async def step_a(prompt, params):
    return await TO.run_tokenize(TO.prompt_chars(prompt), CM.check_context_length, prompt, params)


async def step_b(c, prompt, params, mm=None):
    """As production reaches generate_gen: stream_generate_chat_completion deep-copies the request per choice
    (chat_completion.py:1009), the collector calls mc.stream_generate (:838), which iterates generate_gen."""
    FakeJob.last = None
    agen = c.stream_generate("rid-test", prompt, params.model_copy(deep=True), None, mm)
    try:
        await agen.__anext__()
    except Reached:
        pass
    return FakeJob.last


# ------------------------------------------------------------------------------------------------ the arms
async def run_arms(model_dir, label, chars):
    tok = TM.Tokenizer(_load.make_config(model_dir))
    ref_tok = TB.Tokenizer(_load.make_config(model_dir))
    prompt = _load.long_text(chars)
    ref = ref_tok.encode(prompt, add_bos=False, encode_special_tokens=True)
    t0 = time.perf_counter()
    tok.encode(prompt, add_bos=False, encode_special_tokens=True)
    T = time.perf_counter() - t0
    small = max(0.015, 0.3 * T)
    print(f"{label}: prompt {len(prompt):,} chars = {ref.shape[-1]:,} tokens, solo encode {T * 1000:.1f} ms; "
          f"small-gap bar {small * 1000:.1f} ms, stall bar {0.7 * T * 1000:.1f} ms")
    check(T >= 0.04, f"{label}: solo encode {T * 1000:.1f} ms < 40 ms: prompt too short to see a stall (raise --chars)")
    c = make_container(tok)
    CM.container = c
    n = count_encodes(tok)
    for once in ("1", "0"):
        for off in ("1", "0"):
            set_knobs(once, off)
            params = BaseSamplerRequest(max_tokens=16)
            n[0] = 0
            ra, ga = await with_ticker(lambda: step_a(prompt, params))
            check(not isinstance(ra, Exception), f"{label} once={once} off={off}: check_context_length raised {ra!r}")
            rb, gb = await with_ticker(lambda: step_b(c, prompt, params))
            ok = isinstance(rb, dict) and rb.get("input_ids") is not None
            check(ok, f"{label} once={once} off={off}: generate_gen did not reach AsyncJob ({rb!r})")
            if ok:
                ids = rb["input_ids"][0]
                check(ids.dtype == ref.dtype and tuple(ids.shape) == tuple(ref.shape) and bool((ids == ref).all()),
                      f"{label} once={once} off={off}: AsyncJob ids != the served encode")
            want = 1 if once == "1" else 2
            check(n[0] == want, f"{label} once={once} off={off}: {n[0]} prompt encodes, want {want}")
            # step A: the check's encode is inline unless off-loop
            if off == "1":
                check(ga <= small, f"{label} once={once} off={off}: step A max gap {ga * 1000:.1f} ms > {small * 1000:.1f}")
            else:
                check(ga >= 0.7 * T, f"{label} once={once} off={off}: control step A gap {ga * 1000:.1f} ms < 0.7 T "
                                     f"(the measure should see the inline encode)")
            # step B: no encode at all with once, off-loop with off, inline otherwise
            if once == "1" or off == "1":
                check(gb <= small, f"{label} once={once} off={off}: step B max gap {gb * 1000:.1f} ms > {small * 1000:.1f}")
            else:
                check(gb >= 0.7 * T, f"{label} once={once} off={off}: control step B gap {gb * 1000:.1f} ms < 0.7 T")
            print(f"  {label} TABBY_ENCODE_ONCE={once} EXL3_TOKENIZE_OFFLOOP={off}: encodes {n[0]}, max loop gap "
                  f"A {ga * 1000:6.1f} ms  B {gb * 1000:6.1f} ms")
            if once == "1":
                check(params._prompt_ids is not None and len(params._prompt_ids) == 1, f"{label}: ids not stored")
            else:
                check(params._prompt_ids is None, f"{label}: ids stored with TABBY_ENCODE_ONCE=0")

    # worker thread + GIL-holding encode: the loop still stalls (why the exllamav3 half is needed)
    set_knobs("1", "1")
    real = TM.tokenize_offloop_enabled
    TM.tokenize_offloop_enabled = lambda: False
    try:
        params = BaseSamplerRequest(max_tokens=16)
        _, g = await with_ticker(lambda: step_a(prompt, params))
    finally:
        TM.tokenize_offloop_enabled = real
    print(f"  {label} worker thread with the served encode() (GIL held): max loop gap {g * 1000:.1f} ms")
    check(g >= 0.5 * T, f"{label}: off-loop with a GIL-holding encode gap {g * 1000:.1f} ms < 0.5 T: the GIL "
                        f"measure is blind")

    # reuse keys
    set_knobs("1", "1")
    params = BaseSamplerRequest(max_tokens=16)
    n[0] = 0
    await step_a(prompt, params)
    stored = params._prompt_ids[(prompt, False)]
    check(isinstance(stored, tuple) and stored[0]() is tok and bool((stored[1] == ref).all()),
          f"{label}: stored entry is not (weakref to this tokenizer, ids)")
    p2 = params.model_copy(deep=True)                       # what stream_generate_chat_completion hands each choice
    e2 = p2._prompt_ids[(prompt, False)]
    check(p2._prompt_ids is not params._prompt_ids and e2[0]() is tok and e2[1] is not stored[1]
          and bool((e2[1] == stored[1]).all()), f"{label}: model_copy(deep=True) did not carry (weakref, ids copy)")
    # never serialized (generate_gen's debug dump and log_generation_params dump the request)
    check("_prompt_ids" not in p2.model_dump() and "_prompt_ids" not in p2.model_dump_json()
          and "prompt_ids" not in p2.model_dump_json(), f"{label}: _prompt_ids leaks into model_dump")
    rb = await step_b(c, prompt, p2)
    check(n[0] == 1 and rb["input_ids"][0] is not e2[1]
          and bool((rb["input_ids"][0] == ref).all()), f"{label}: deep-copied request: hit + clone")
    # the key includes the tokenizer: another container (another Tokenizer instance, same text) must miss and encode
    tok_b = TM.Tokenizer(_load.make_config(model_dir))
    n_b = count_encodes(tok_b)
    c_b = make_container(tok_b)
    rb = await step_b(c_b, prompt, p2)
    check(n_b[0] == 1 and bool((rb["input_ids"][0] == ref).all()), f"{label}: another tokenizer must miss and encode")
    p5 = BaseSamplerRequest(max_tokens=16)
    p5._prompt_ids = {(prompt, False): (weakref.ref(tok_b), stored[1])}
    del c_b, tok_b, n_b
    gc.collect()
    check(p5._prompt_ids[(prompt, False)][0]() is None, f"{label}: collected tokenizer still referenced")
    n0 = n[0]
    rb = await step_b(c, prompt, p5)
    check(n[0] == n0 + 1, f"{label}: a dead tokenizer reference must miss and encode")
    other = prompt[:-7] + "changed"                          # forced_tool_generation builds a different prompt
    n0 = n[0]
    rb = await step_b(c, other, p2)
    check(n[0] == n0 + 1 and bool((rb["input_ids"][0] == ref_tok.encode(other, encode_special_tokens=True)).all()),
          f"{label}: another prompt string must miss and encode")
    same_text = "".join(list(prompt))                        # equal text, another str object: a hit
    rb = await step_b(c, same_text, p2)
    check(n[0] == n0 + 1 and bool((rb["input_ids"][0] == ref).all()), f"{label}: equal text (other object) must hit")
    p3 = BaseSamplerRequest(max_tokens=16, add_bos_token=True)
    await step_a(prompt, p3)
    p3._prompt_ids = {(prompt, False): stored}               # ids stored for add_bos False only
    rb = await step_b(c, prompt, p3)
    check(n[0] == n0 + 3 and bool((rb["input_ids"][0] == ref_tok.encode(prompt, add_bos=True, encode_special_tokens=True)).all()),
          f"{label}: add_bos mismatch must miss and encode")
    fake_mm = types.SimpleNamespace(content=[types.SimpleNamespace(text_alias="<|fake_alias_not_in_text|>")])
    p4 = BaseSamplerRequest(max_tokens=16)
    n0 = n[0]
    await TO.run_tokenize(TO.prompt_chars(prompt), CM.check_context_length, prompt, p4, fake_mm)
    check(p4._prompt_ids is None, f"{label}: multimodal request must not store ids")
    rb = await step_b(c, prompt, p4, fake_mm)
    check(n[0] == n0 + 2 and bool((rb["input_ids"][0] == ref).all()), f"{label}: multimodal request encodes in both steps")

    # oversized prompt: same exception type and message under every arm, inline or through the worker
    small_c = make_container(tok, max_seq_len=ref.shape[-1] // 2)
    CM.container = small_c
    errs = {}
    for once in ("1", "0"):
        for off in ("1", "0"):
            set_knobs(once, off)
            try:
                await step_a(prompt, BaseSamplerRequest(max_tokens=16))
                errs[(once, off)] = None
            except Exception as e:  # noqa: BLE001
                errs[(once, off)] = (type(e).__name__, str(e), getattr(e, "status_code", None))
    vals = set(errs.values())
    check(len(vals) == 1 and None not in vals, f"{label}: oversized prompt errors differ across arms: {errs}")
    print(f"  {label} oversized prompt, every arm: {next(iter(vals))}")
    CM.container = c
    os.environ.pop("TABBY_ENCODE_ONCE", None)
    os.environ.pop("EXL3_TOKENIZE_OFFLOOP", None)


# ------------------------------------------------------------------------------------------------ run_tokenize itself
CV = contextvars.ContextVar("cv", default="unset")
BIG = 1 << 40                                                # a length above any hop threshold: always the worker


async def run_tokenize_semantics():
    os.environ.pop("EXL3_TOKENIZE_OFFLOOP", None)
    order = []

    def work(i, dt):
        time.sleep(dt)
        order.append(i)
        return i, CV.get()
    CV.set("request-ctx")
    res = await asyncio.gather(*(TO.run_tokenize(BIG, work, i, 0.02 if i == 0 else 0.001) for i in range(5)))
    check(order == list(range(5)), f"run_tokenize order {order} != submission order")
    check([r[0] for r in res] == list(range(5)) and all(r[1] == "request-ctx" for r in res),
          f"run_tokenize results/contextvars {res}")

    def boom():
        raise ValueError("boom")
    try:
        await TO.run_tokenize(BIG, boom)
        check(False, "run_tokenize swallowed an exception")
    except ValueError as e:
        check(str(e) == "boom", "run_tokenize exception message")
    finished = []

    def slow():
        time.sleep(0.05)
        finished.append(1)
        return 1
    t = asyncio.create_task(TO.run_tokenize(BIG, slow))
    await asyncio.sleep(0.01)
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    r = await TO.run_tokenize(BIG, lambda: 42)
    check(r == 42 and finished == [1], f"after a cancelled await: worker finished {finished}, next call {r}")
    os.environ["EXL3_TOKENIZE_OFFLOOP"] = "0"
    import threading
    tid = await TO.run_tokenize(BIG, threading.get_ident)
    check(tid == threading.get_ident(), "EXL3_TOKENIZE_OFFLOOP=0 must call inline on the loop thread")
    os.environ.pop("EXL3_TOKENIZE_OFFLOOP", None)
    check(TO.offloop_enabled() and TO.encode_once_enabled(), "knobs default ON")


def router_static():
    src = open(os.path.join(APP, "endpoints", "OAI", "router.py")).read()
    tree = ast.parse(src)
    wrapped = bare = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
            f = node.value.func
            if isinstance(f, ast.Name) and f.id == "run_tokenize" and len(node.value.args) >= 2:
                a0, a1 = node.value.args[0], node.value.args[1]
                if (isinstance(a1, ast.Attribute) and a1.attr == "check_context_length"
                        and isinstance(a0, ast.Call) and getattr(a0.func, "id", None) == "prompt_chars"
                        and len(a0.args) == 1 and isinstance(a0.args[0], ast.Name) and a0.args[0].id == "prompt"
                        and len(node.value.args) >= 3 and isinstance(node.value.args[2], ast.Name)
                        and node.value.args[2].id == "prompt"):
                    wrapped += 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "check_context_length":
            bare += 1
    check(wrapped == 2 and bare == 0, f"router.py: {wrapped} awaited run_tokenize(prompt_chars(prompt), check_context_length, prompt, ...) (want 2), "
                                      f"{bare} bare calls (want 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", action="append", default=[])
    ap.add_argument("--chars", type=int, default=600_000)
    a = ap.parse_args()
    router_static()
    asyncio.run(run_tokenize_semantics())
    asyncio.run(run_arms(_load.build_synth_tokenizer(), "synthetic", a.chars))
    for d in a.tokenizer_dir:
        asyncio.run(run_arms(d, os.path.basename(os.path.normpath(d)) or d, a.chars))
    print(f"test_event_loop: {NCHECK[0] - len(FAILS)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
