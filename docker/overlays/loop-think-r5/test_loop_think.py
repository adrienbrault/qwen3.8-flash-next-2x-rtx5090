"""Offline check of loop-think r5 (no GPU): drives _chat_stream_collector with a fake backend.

Sections 1-13 are r4's checks, unchanged. 14-19 are r5's: the long-period ladder on the R791 loop bodies
(fixtures/r791-rows.json.gz), false-positive texts, r4-identical firing on r4's periods, the CPU cost of the
ladder (report only), and finish_reason when a tool call is cut by max_tokens.

Run inside the image with the patched chat_completion.py in place:
  docker run --rm -v $PWD:/t -w /app --entrypoint python3 IMG /t/test_loop_think.py
or on a CPU-only host against a TabbyAPI tree: python3 run_offline.py APP_DIR (see run_offline.py).
"""
import asyncio
import sys

from common import model
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils import chat_completion as cc

END = "</think>"


class FakeContainer:
    max_rq_tokens = None
    harmony = False
    muse_glimmer = False
    tool_format = None
    reasoning = True
    reasoning_start_token = "<think>"
    reasoning_end_token = END
    tool_calls_in_reasoning = False
    reasoning_budget_tokens = None
    reasoning_budget_message = None

    def __init__(self, tokens, answer="The answer.", chunk=4):
        self.tokens = tokens
        self.answer = answer
        self.chunk = chunk
        self.injections = []
        self.params_seen = None

    def constrain_generation_output(self, request_id, text):
        self.injections.append(text)
        return True

    async def stream_generate(self, request_id, prompt, params, disconnect_handler, mm, filter_trigger=None, label=None):
        self.params_seen = params
        i = 0
        while i < len(self.tokens):
            if self.injections:
                # the forced text arrives, then the model answers
                yield {"text": self.injections[0], "token_ids": [1] * 20}
                yield {"text": self.answer, "token_ids": [2] * 3, "finish_reason": "stop"}
                return
            part = self.tokens[i:i + self.chunk]
            i += self.chunk
            yield {"text": " ".join(f"t{t}" for t in part) + " ", "token_ids": part}
        # engine loop detector / natural end with no content
        yield {"text": "", "token_ids": [], "finish_reason": "stop"}


def run(mc, start_in_reasoning=True, **req):
    model.container = mc
    params = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **req)
    return asyncio.run(cc._chat_stream_collector(0, None, "rid", "prompt", params, start_in_reasoning, streaming_mode=False)), params


fresh = list(range(1000, 1300))          # 300 distinct tokens
loop = list(range(50)) * 40              # period-50 loop, 2,000 tokens
fails = 0


def check(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    fails += not cond


# 1. loop in reasoning -> one injection, content present, engine window doubled
mc = FakeContainer(fresh + loop)
g, p = run(mc)
check("reasoning loop injects once", len(mc.injections) == 1, f"{len(mc.injections)}")
check("injection = message + end tag", mc.injections and mc.injections[0] == cc.LOOP_THINK_MESSAGE + END)
check("content non-empty", bool(g.get("content")), repr(g.get("content")))
check("engine backstop (2W, 4)", mc.params_seen.get_stop_on_loop() == (1600, 4), f"{mc.params_seen.get_stop_on_loop()}")
check("client-visible window untouched", mc.params_seen.loop_detect_window == 800 and "loop_detect_window" not in mc.params_seen.model_fields_set)

# 2. no loop -> no injection
mc = FakeContainer(list(range(5000, 7000)))
g, _ = run(mc)
check("no loop, no injection", mc.injections == [])

# 3. loop only in content (no reasoning phase) -> untouched
mc = FakeContainer(fresh + loop)
g, _ = run(mc, start_in_reasoning=False)
check("content-phase loop not injected, engine keeps (W, 2)", mc.injections == [] and mc.params_seen.get_stop_on_loop() == (800, 2),
      f"{mc.params_seen.get_stop_on_loop()}")

# 4. client disables loop detection -> untouched, window stays 0
mc = FakeContainer(fresh + loop)
g, _ = run(mc, loop_detect_window=0)
check("loop_detect_window 0 disables", mc.injections == [] and mc.params_seen.loop_detect_window == 0)

# 5. constrained generation -> untouched
mc = FakeContainer(fresh + loop)
g, _ = run(mc, regex_pattern="a+")
check("regex request not injected", mc.injections == [])

# 6. custom window 300 -> fires, engine 600
mc = FakeContainer(fresh + loop)
g, _ = run(mc, loop_detect_window=300)
check("window 300 injects, engine (600, 4)", len(mc.injections) == 1 and mc.params_seen.get_stop_on_loop() == (600, 4))

# 7. reasoning budget fires first -> loop detector disabled, one injection only
mc = FakeContainer(fresh + loop)
mc.reasoning_budget_tokens = 100
g, _ = run(mc)
check("budget first, single injection", len(mc.injections) == 1 and mc.injections[0] != cc.LOOP_THINK_MESSAGE + END)

# 8. review issue 1: a period-500 loop is out of reach of the (800, 2) detector the engine used before; the backstop
#    must not newly catch it (r1's (1600, 2) did), and the collector must not inject on it either
from exllamav3.generator.loop_detect import LoopDetector
p500 = list(range(3000, 3500)) * 6
def fires(w, reps):
    return bool(LoopDetector(w, w // reps).feed_many(fresh + p500))
check("period 500: old engine (800,2) silent", not fires(800, 2))
check("period 500: r1 backstop (1600,2) fires (the r1 regression)", fires(1600, 2))
check("period 500: r2 backstop (1600,4) silent", not fires(1600, 4))
mc = FakeContainer(fresh + p500)
g, _ = run(mc)
check("period 500 x6 reasoning loop: injected by the long detector (r4)", len(mc.injections) == 1)
# 9. period 400 loop: collector fires, backstop (1600, 4) would too (only later)
p400 = list(range(4000, 4400)) * 6
check("period 400: r2 backstop still catches it", bool(LoopDetector(1600, 400).feed_many(fresh + p400)))
mc = FakeContainer(fresh + p400)
g, _ = run(mc)
check("period 400 reasoning loop: injected", len(mc.injections) == 1)
# 10. detection on the final chunk: no injection
class Last(FakeContainer):
    async def stream_generate(self, *a, **k):
        self.params_seen = a[2]
        toks = self.tokens
        yield {"text": "x ", "token_ids": toks[:-1]}
        yield {"text": "y ", "token_ids": toks[-1:], "finish_reason": "stop"}
mc = Last(fresh + loop[:800])
g, _ = run(mc)
check("detection on the finishing chunk: no injection", mc.injections == [])

# 11. R782 review issue 1: 2W beyond the output chunk (engine detector rebuilt per requeue) -> not armed
mc = FakeContainer(fresh + loop)
mc.max_rq_tokens = 2048
g, _ = run(mc, loop_detect_window=1100)
check("W 1100 with 2048-token chunks: not armed, engine (W, 2)", mc.injections == [] and mc.params_seen.get_stop_on_loop() == (1100, 2),
      f"{mc.params_seen.get_stop_on_loop()}")
mc = FakeContainer(fresh + loop)
mc.max_rq_tokens = 2048
g, _ = run(mc)
check("W 800 with 2048-token chunks: armed", len(mc.injections) == 1 and mc.params_seen.get_stop_on_loop() == (1600, 4))
mc = FakeContainer(fresh + loop)
mc.max_rq_tokens = None
g, _ = run(mc, loop_detect_window=1100)
check("W 1100 without chunking: armed", len(mc.injections) == 1 and mc.params_seen.get_stop_on_loop() == (2200, 4))
# 12. n>1 isolation: the override lands on the per-choice copy only
from endpoints.OAI.types.chat_completion import ChatCompletionRequest as CCR
base = CCR(messages=[{"role": "user", "content": "hi"}])
c1 = base.model_copy(deep=True); c1._loop_backstop = (1600, 4)
c2 = base.model_copy(deep=True)
check("per-choice copies isolated", base.get_stop_on_loop() == (800, 2) and c2.get_stop_on_loop() == (800, 2) and c1.get_stop_on_loop() == (1600, 4))

# 13. r4 long-period detector: (3000, 1000), >= 3 copies
def mk(period, copies, base=20000):
    return fresh + list(range(base, base + period)) * copies
mc = FakeContainer(mk(710, 6)); g, _ = run(mc)
check("period 710 x6 (R783 p107 #1 shape): injected", len(mc.injections) == 1)
mc = FakeContainer(mk(710, 2) + list(range(40000, 41000))); g, _ = run(mc)
check("period 710 x2 then new text: not injected", mc.injections == [])
mc = FakeContainer(mk(600, 2) + list(range(40000, 42000))); g, _ = run(mc)
check("a 600-token block restated once: not injected", mc.injections == [])
mc = FakeContainer(mk(1100, 5)); g, _ = run(mc)
check("period 1100 x5: beyond 1.25W, not injected", mc.injections == [])
mc = FakeContainer(mk(710, 6)); g, _ = run(mc, loop_detect_window=0)
check("window 0 disables the long detector too", mc.injections == [])

# ---------------------------------------------------------------------------------------------------------------
# r5 (flan R791 review): long-period ladder + finish_reason of a tool call cut by max_tokens
# ---------------------------------------------------------------------------------------------------------------
import gzip
import json
import os
import random
import time

# fixtures/r791-rows.json.gz (public copy): synthetic ids. The u* rows keep only the length and the long-equality
# structure of R791's rows (every run of x[i] == x[i-p], p <= 4000, length >= 400, listed per row as "structure");
# every other id is fresh (seeded, see the README). The syn_* rows are tokenized template text. The private original
# held the re-tokenized reasoning; every check line below is identical on both.
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "r791-rows.json.gz")
with gzip.open(FIX, "rt") as fh:
    rows = {r["name"]: r for r in json.load(fh)["rows"]}

R4_SET = [(800, 400), (3000, 1000)]          # r4 collector at W = 800: (W, W/2) + (3L, L), L = 1.25W
LADDER = [(6000, 2000), (12000, 4000)]       # r5 rungs (6L, 2L), (12L, 4L)
R5_SET = R4_SET + LADDER


def first_fire(ids, rungs):
    """(index of the token at which the first rung fires, that rung) over a detector set, or (None, None)"""
    best = (None, None)
    for rung in rungs:
        f = LoopDetector(*rung).feed_many(ids)
        if f and (best[0] is None or f[0] < best[0]):
            best = (f[0], rung)
    return best


class Counting(FakeContainer):
    """FakeContainer that records how many tokens had been streamed when the injection arrived."""

    max_rq_tokens = 2048  # the daily's chunk

    def constrain_generation_output(self, request_id, text):
        self.fired_at = self.sent
        return super().constrain_generation_output(request_id, text)

    async def stream_generate(self, request_id, prompt, params, disconnect_handler, mm, filter_trigger=None, label=None):
        self.params_seen = params
        self.sent = 0
        self.fired_at = None
        i = 0
        while i < len(self.tokens):
            if self.injections:
                yield {"text": self.injections[0], "token_ids": [1] * 20}
                yield {"text": self.answer, "token_ids": [2] * 3, "finish_reason": "stop"}
                return
            part = self.tokens[i:i + self.chunk]
            i += self.chunk
            self.sent += len(part)
            yield {"text": " ".join(f"t{t}" for t in part) + " ", "token_ids": part}
        yield {"text": "", "token_ids": [], "finish_reason": "stop"}


# 14. the two R791 loop bodies fire at the expected rung; the collector injects once, where the rung fires
for name, rung, lo, hi, period in [
    ("u16_t06_len", (6000, 2000), 7000, 9000, 1199),     # 3,575-char block x25 -> 1,199-token period
    ("u10_t06_len", (12000, 4000), 19000, 21500, 3713),  # 9,999-char block x6  -> 3,713-token period
]:
    ids = rows[name]["ids"]
    p = next(q for q in range(1, 8000) if ids[-q:] == ids[-2 * q:-q])
    check(f"{name}: exact token tail period {period}", p == period, f"{p}")
    at4, _ = first_fire(ids, R4_SET)
    check(f"{name}: r4 detectors silent over all {len(ids)} tokens", at4 is None, f"{at4}")
    at, got = first_fire(ids, R5_SET)
    check(f"{name}: fires at rung {rung}", got == rung, f"{got} at token {at}")
    check(f"{name}: fires within [{lo}, {hi}] reasoning tokens", at is not None and lo <= at <= hi, f"{at}")
    mc = Counting(ids)
    g, _ = run(mc)
    check(f"{name}: collector injects once, at the rung's token", len(mc.injections) == 1 and mc.fired_at is not None
          and 0 <= mc.fired_at - (at + 1) < mc.chunk, f"injections {len(mc.injections)}, fired_at {mc.fired_at}")
    check(f"{name}: then answers", g.get("content") == "The answer.", repr(g.get("content")))

# 15. no false positives: the 1.0 near-exact repetition (left by the model itself), plan restatement, and
#     legitimately repetitive text whose lines differ in a number or a name
for name in ["u10_t10_max", "u1_t10_max", "syn_numbered_same", "syn_numbered_varied", "syn_table", "syn_code"]:
    ids = rows[name]["ids"]
    at, got = first_fire(ids, R5_SET)
    check(f"{name} ({len(ids)} tokens): no rung fires", at is None, f"{got} at {at}")
    mc = Counting(ids)
    g, _ = run(mc)
    check(f"{name}: collector does not inject", mc.injections == [])

# 16. r4 behaviour kept for the periods it already catches: on single-block loops of period <= L = 1,000, the r5
#     set fires exactly when r4's does (the ladder's windows are longer, so they can only fire later)
rng = random.Random(791)
same, fired, collector_runs, bad = 0, 0, 0, []
for trial in range(120):
    p = rng.choice([1, 2, 7, 50, 399, 400, 401, 500, 710, 999, 1000] + [rng.randint(1, 1000) for _ in range(5)])
    copies = rng.randint(1, 12)
    block = [rng.randrange(10000, 240000) for _ in range(p)]
    stream = [rng.randrange(10000, 240000) for _ in range(rng.randint(0, 3000))] + block * copies
    stream += [rng.randrange(10000, 240000) for _ in range(rng.randint(0, 2000))]
    a4, _ = first_fire(stream, R4_SET)
    a5, _ = first_fire(stream, R5_SET)
    same += a4 == a5
    fired += a4 is not None
    if a4 != a5:
        bad.append((p, copies, a4, a5))
    if collector_runs < 12 and a4 is not None:
        collector_runs += 1
        mc = Counting(stream)
        run(mc)
        if not (len(mc.injections) == 1 and 0 <= mc.fired_at - (a4 + 1) < mc.chunk):
            bad.append(("collector", p, copies, a4, mc.fired_at))
check(f"r4 periods: r5 fires at r4's token in 120/120 random loops ({fired} fire; collector checked on {collector_runs})",
      same == 120 and collector_runs == 12 and not bad, f"{bad[:3]}")

# 17. ladder coverage and its limits
mc = Counting(mk(1100, 7)); g, _ = run(mc)
check("period 1100 x7: caught by (6000, 2000)", len(mc.injections) == 1)
mc = Counting(mk(2000, 3)); g, _ = run(mc)
check("period 2000 x3 (3-copy floor of rung 2L): injected", len(mc.injections) == 1)
mc = Counting(mk(1500, 2) + list(range(40000, 46000))); g, _ = run(mc)
check("a 1,500-token block restated once: not injected", mc.injections == [])
mc = Counting(mk(4000, 2) + list(range(40000, 50000))); g, _ = run(mc)
check("a 4,000-token block restated once: not injected", mc.injections == [])
mc = Counting(mk(4000, 3)); g, _ = run(mc)
check("period 4000 x3 (3-copy floor of rung 4L): injected", len(mc.injections) == 1)
mc = Counting(mk(4100, 5)); g, _ = run(mc)
check("period 4100 x5: beyond 4L, not injected (gap)", mc.injections == [])
ab = list(range(50000, 50700)) + list(range(60000, 60700))
mc = Counting(fresh + ab * 5); g, _ = run(mc)
check("two alternating 700-token blocks x5 (period 1,400): r4 silent, r5 injects",
      first_fire(fresh + ab * 5, R4_SET)[0] is None and len(mc.injections) == 1)
mc = Counting(mk(1199, 25)); g, _ = run(mc, loop_detect_window=0)
check("window 0 disables the ladder too", mc.injections == [])

# 18. CPU cost of the detector set per reasoning token (report only: the image's CPU differs from the build host)
def cost(rungs, stream, chunk=4, reps=3):
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        dets = [LoopDetector(*r) for r in rungs]
        t1 = time.perf_counter()
        worst = 0.0
        for i in range(0, len(stream), chunk):
            c = stream[i:i + chunk]
            t = time.perf_counter()
            hit = any([d.feed_many(c) for d in dets])
            worst = max(worst, time.perf_counter() - t)
            if hit:
                break
        t2 = time.perf_counter()
        r = ((t1 - t0) * 1e3, (t2 - t1) / min(len(stream), i + chunk) * 1e6, worst * 1e3)
        best = r if best is None or r[1] < best[1] else best
    return best


zr = random.Random(0)
streams = {
    "zipf 30k (seeded, no loop)": [min(int(zr.paretovariate(1.1)), 248000) for _ in range(30000)],
    "R791 text 30k (u10 1.0 #2 + u10 0.6 #3 pre-loop)": (rows["u10_t10_max"]["ids"] + rows["u10_t06_len"]["ids"][:8121])[:30000],
    "syn_table (28.6k, repetitive, no loop)": rows["syn_table"]["ids"],
    "syn_code (18.4k, repetitive, no loop)": rows["syn_code"]["ids"],
}
# worst case for partial streaks: 300-token runs of one token straddling the 4L rung's synchronized wakes (12k, 24k)
wake = list(streams["zipf 30k (seeded, no loop)"])
for at in (11850, 23850):
    wake[at:at + 300] = [7] * 300
streams["zipf 30k + one-token runs of 300 at the 12k/24k wakes"] = wake
for sname, s in streams.items():
    for lname, rungs in [("r4", R4_SET), ("r5", R5_SET)]:
        init, us, worst = cost(rungs, s)
        print(f"INFO cpu {lname} {sname}: init {init:.2f} ms, {us:.2f} us/token, worst 4-token chunk {worst:.2f} ms")

# 19. finish_reason: a max_tokens stop inside an unparsed tool call reports "length" (base: "tool_calls" with no call);
#     every other end keeps the base's "tool_calls", with or without a parsed call (review F1: a reasoning-only
#     "stop" is the shape Hermes shows as the reply, R781)
class ToolFake(FakeContainer):
    tool_format = "qwen3_coder"

    def __init__(self, parts):
        super().__init__([])
        self.parts = parts

    async def stream_generate(self, request_id, prompt, params, disconnect_handler, mm, filter_trigger=None, label=None):
        self.params_seen = params
        for part in self.parts:
            yield dict(part)


def run_stream(mc, **req):
    model.container = mc
    params = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **req)

    async def go():
        q = asyncio.Queue()
        await cc._chat_stream_collector(0, q, "rid", "prompt", params, True, streaming_mode=True)
        out = []
        while not q.empty():
            out.append(q.get_nowait())
        return out

    out = asyncio.run(go())
    errs = [o for o in out if isinstance(o, Exception)]
    assert not errs, errs
    last = out[-1]
    data = cc._compose_serialize_stream_chunk("rid", last)[1]
    return last, data["choices"][0], "".join(o.get("delta_content") or "" for o in out)


PRE = "\n\nWriting it now.\n\n"
THINK = [{"text": "Plan the file.", "token_ids": [5, 6]}, {"text": "</think>" + PRE, "token_ids": [7]}]
BARE = [{"text": "Plan the file.", "token_ids": [5, 6]}, {"text": "</think>", "token_ids": [7]}]  # no preamble
CALL_OPEN = "<tool_call>\n<function=write_file>\n<parameter=path>\na.js\n</parameter>\n<parameter=content>\nconst x = 1;\n"
CALL_DONE = "</parameter>\n</function>\n</tool_call>"
cut = [{"text": CALL_OPEN, "token_ids": [8]},
       {"text": "const y", "token_ids": [9], "finish_reason": "length", "eos_reason": "max_new_tokens"}]
def unparsed(eos):
    return [{"text": CALL_OPEN, "token_ids": [8]}, {"text": "const y", "token_ids": [9], "finish_reason": "stop", "eos_reason": eos}]
cases = {  # name: (parts, finish, calls, non-streaming content)
    "cut inside a tool call by max_tokens": (THINK + cut, "length", 0, PRE),
    "cut inside a tool call by max_tokens, no preamble": (BARE + cut, "length", 0, None),
    "complete tool call": (
        THINK + [{"text": CALL_OPEN, "token_ids": [8]},
                 {"text": CALL_DONE, "token_ids": [9], "finish_reason": "stop", "eos_reason": "stop_token"}],
        "tool_calls", 1, PRE),
    "one complete call, a second cut by max_tokens (accepted edge: the parsed call is emitted)": (
        THINK + [{"text": CALL_OPEN + CALL_DONE + "\n" + CALL_OPEN, "token_ids": [8]},
                 {"text": "const y", "token_ids": [9], "finish_reason": "length", "eos_reason": "max_new_tokens"}],
        "tool_calls", 1, PRE + "\n"),
    "no tool text, max_tokens": (
        THINK + [{"text": "Some prose", "token_ids": [8], "finish_reason": "length", "eos_reason": "max_new_tokens"}],
        "length", 0, PRE + "Some prose"),
    "unparseable tool text, stop_token (base kept)": (BARE + unparsed("stop_token"), "tool_calls", 0, None),
    "unparseable tool text, loop_detected (base kept)": (BARE + unparsed("loop_detected"), "tool_calls", 0, None),
    "unparseable tool text, stop_string, with preamble (base kept)": (THINK + unparsed("stop_string"), "tool_calls", 0, PRE),
}
for cname, (parts, want, ncalls, content) in cases.items():
    g, _ = run(ToolFake(parts))
    calls = g.get("tool_calls") or []
    check(f"non-streaming, {cname}: finish {want}, {ncalls} call(s), content {content!r}",
          g.get("finish_reason") == want and len(calls) == ncalls and g.get("content") == content,
          f"{g.get('finish_reason')}, {len(calls)} calls, content {g.get('content')!r}")
    last, choice, streamed = run_stream(ToolFake(parts))
    scalls = choice["delta"].get("tool_calls") or []
    check(f"streaming, {cname}: finish {want}, {ncalls} call(s)",
          choice["finish_reason"] == want and len(scalls) == ncalls and streamed == (content or ""),
          f"{choice['finish_reason']}, {len(scalls)} calls, content {streamed!r}")
g, _ = run(ToolFake(cases["complete tool call"][0]))
check("complete tool call parses as write_file(path=a.js)", g["tool_calls"][0]["function"]["name"] == "write_file"
      and json.loads(g["tool_calls"][0]["function"]["arguments"]).get("path") == "a.js")


# 20. content and tool calls are never watched: only reasoning tokens outside a tool call feed the detectors
blk = list(range(20000, 21199)) * 10  # period 1,199 x10: fires in reasoning (see the control)
END_STOP = {"text": "", "token_ids": [], "finish_reason": "stop", "eos_reason": "stop_token"}


def chunked(ids, n=4):
    return [{"text": "x ", "token_ids": ids[i:i + n]} for i in range(0, len(ids), n)]


mc = ToolFake([{"text": "plan ", "token_ids": [5]}] + chunked(blk) + [END_STOP])
run(mc)
check("control: period 1199 x10 in reasoning is injected", len(mc.injections) == 1)
mc = ToolFake(BARE + chunked(blk) + [END_STOP])
run(mc)
check("period 1199 x10 after </think> (content): no injection", mc.injections == [])
mc = ToolFake([{"text": "plan <tool_call>\n<function=write_file>\n<parameter=content>\n", "token_ids": [5]}] + chunked(blk) + [END_STOP])
mc.tool_calls_in_reasoning = True
g, _ = run(mc)
check("period 1199 x10 inside a <tool_call> in reasoning: no injection (and the parser saw the tool span)",
      mc.injections == [] and g.get("finish_reason") == "tool_calls", f"{len(mc.injections)}, {g.get('finish_reason')}")

# 21. through the endpoint functions: the include_usage usage chunk carries the finish, and n=2 choices get theirs
#     independently
import pathlib
from types import SimpleNamespace


class DH:
    async def cleanup(self):
        pass


class PerChoice(ToolFake):
    start_in_reasoning = "always"

    def __init__(self, by_idx):
        super().__init__([])
        self.by_idx = by_idx

    async def stream_generate(self, request_id, prompt, params, disconnect_handler, mm, filter_trigger=None, label=None):
        idx = int(request_id.rsplit("-", 1)[1]) if "-" in request_id else 0
        for part in self.by_idx[idx]:
            yield dict(part)


def endpoint_stream(by_idx, **req):
    model.container = PerChoice(by_idx)
    data = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], stream=True, **req)

    async def go():
        gen = cc.stream_generate_chat_completion("prompt", None, data, SimpleNamespace(state=SimpleNamespace(id="rid0", serial=1)),
                                                 pathlib.Path("/models/x"), DH())
        return [c async for c in gen]

    return [json.loads(c) for c in asyncio.run(go()) if c != "[DONE]"]


for cname, parts, want in [("cut inside a call", BARE + cut, "length"),
                           ("complete call", cases["complete tool call"][0], "tool_calls")]:
    js = endpoint_stream({0: parts}, stream_options={"include_usage": True})
    usage = [j for j in js if "usage" in j]
    others = [j for j in js if "usage" not in j]
    check(f"stream + include_usage, {cname}: usage chunk finish {want}, earlier finishes suppressed",
          len(usage) == 1 and usage[0]["choices"][0]["finish_reason"] == want
          and all(j["choices"][0]["finish_reason"] is None for j in others),
          f"{[j['choices'][0]['finish_reason'] for j in js]}")
    js = endpoint_stream({0: parts})
    check(f"stream without usage, {cname}: last chunk finish {want}", js[-1]["choices"][0]["finish_reason"] == want,
          f"{[j['choices'][0]['finish_reason'] for j in js]}")

model.container = PerChoice({0: BARE + cut, 1: cases["complete tool call"][0]})
resp = asyncio.run(cc.generate_chat_completion(
    "prompt", None, ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], n=2),
    SimpleNamespace(state=SimpleNamespace(id="rid0", serial=1)), pathlib.Path("/models/x"), DH()))
got = {c.index: (c.finish_reason, len(c.message.tool_calls or [])) for c in resp.choices}
check("n=2 non-streaming: cut call -> length, complete call -> tool_calls", got == {0: ("length", 0), 1: ("tool_calls", 1)}, f"{got}")

# 22. the ladder through streaming_mode=True, and with merged 64-token chunks (MTP / requeue-sized deltas)
for name in ["u16_t06_len", "u10_t06_len"]:
    ids = rows[name]["ids"]
    at, _ = first_fire(ids, R5_SET)
    mc = Counting(ids)
    last, choice, streamed = run_stream(mc)
    check(f"{name} streaming: one injection at the rung's token, then the answer",
          len(mc.injections) == 1 and 0 <= mc.fired_at - (at + 1) < mc.chunk and streamed == "The answer."
          and choice["finish_reason"] == "stop", f"{len(mc.injections)}, {mc.fired_at}, {streamed!r}, {choice['finish_reason']}")
    mc = Counting(ids, chunk=64)
    g, _ = run(mc)
    check(f"{name} in 64-token chunks: one injection within one chunk of the rung's token",
          len(mc.injections) == 1 and 0 <= mc.fired_at - (at + 1) < 64, f"{len(mc.injections)}, {mc.fired_at} vs {at + 1}")

print("FAILS", fails)
sys.exit(1 if fails else 0)
