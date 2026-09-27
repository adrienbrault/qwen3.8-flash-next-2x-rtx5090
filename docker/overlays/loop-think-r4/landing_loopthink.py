# Import-time proof that loop-think r4 landed in /app (fails the build otherwise).
import sys
sys.path.insert(0, "/app")
from endpoints.OAI.utils import chat_completion as cc
import inspect
assert cc.LoopDetector is not None, "exllamav3 LoopDetector not importable"
src = inspect.getsource(cc._chat_stream_collector)
assert "LoopDetector(3 * long_period, long_period)" in src and "window * 2 <= max_rq" in src and "loop_injection" in src and "params._loop_backstop = (window * 2, min_reps * 2)" in src, "collector not patched"
from common.sampling import BaseSamplerRequest
assert "_loop_backstop" in inspect.getsource(BaseSamplerRequest.get_stop_on_loop), "sampling not patched"
print("loop-think r4 landed:", repr(cc.LOOP_THINK_MESSAGE))
