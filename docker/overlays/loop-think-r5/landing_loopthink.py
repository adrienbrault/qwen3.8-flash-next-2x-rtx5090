# Import-time proof that loop-think r5 landed in /app (fails the build otherwise).
import sys
sys.path.insert(0, "/app")
from endpoints.OAI.utils import chat_completion as cc
import inspect
assert cc.LoopDetector is not None, "exllamav3 LoopDetector not importable"
src = inspect.getsource(cc._chat_stream_collector)
assert "window * 2 <= max_rq" in src and "loop_injection" in src and "params._loop_backstop = (window * 2, min_reps * 2)" in src, "collector not patched"
# r5 change 1: long-period ladder (3kL, kL), k = 1, 2, 4
assert "LoopDetector(3 * rung * long_period, rung * long_period)" in src and "for rung in LOOP_THINK_LADDER" in src, "ladder not patched"
assert cc.LOOP_THINK_LADDER == (1, 2, 4), f"unexpected ladder {cc.LOOP_THINK_LADDER}"
p = list(range(1000, 2199))  # the exllamav3 detector must still see a 1,199-token period x7 at rung (6000, 2000) (R791 u16)
assert cc.LoopDetector(6000, 2000).feed_many(list(range(5000, 5300)) + p * 7), "LoopDetector(6000, 2000) misses period 1199"
# r5 change 2: a max_tokens stop inside an unparsed tool call reports "length"; every other end keeps the base's
# "tool_calls" (streaming and non-streaming share the helper)
assert src.count("_finish_with_tool_calls(generation, generation[") == 2 and '"finish_reason"] = "tool_calls"' not in src, "finish fix not patched"
for fin, eos, calls, want in [
    ("length", "max_new_tokens", [], "length"),
    ("length", "max_new_tokens", [{"function": {"name": "f"}}], "tool_calls"),
    ("stop", "stop_token", [], "tool_calls"),
    ("stop", "stop_string", [], "tool_calls"),
    ("stop", "loop_detected", [], "tool_calls"),
    ("stop", "stop_token", [{"function": {"name": "f"}}], "tool_calls"),
]:
    g = {"finish_reason": fin, "eos_reason": eos}
    cc._finish_with_tool_calls(g, calls, "landing")
    assert g["finish_reason"] == want, (fin, eos, len(calls), g)
from common.sampling import BaseSamplerRequest
assert "_loop_backstop" in inspect.getsource(BaseSamplerRequest.get_stop_on_loop), "sampling not patched"
print("loop-think r5 landed:", repr(cc.LOOP_THINK_MESSAGE), "ladder", cc.LOOP_THINK_LADDER)
