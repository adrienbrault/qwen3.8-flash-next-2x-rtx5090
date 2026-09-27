"""Offline check of loop-think r4 (no GPU): drives _chat_stream_collector with a fake backend.

Run inside the image with the patched chat_completion.py in place:
  docker run --rm -v $PWD:/t -w /app --entrypoint python3 IMG /t/test_loop_think.py
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

print("FAILS", fails)
sys.exit(1 if fails else 0)
