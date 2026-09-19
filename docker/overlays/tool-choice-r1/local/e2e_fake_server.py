"""
Local end-to-end harness (not shipped): serve a TabbyAPI tree's real OAI router
(validation, templating, collector, SSE serialisation) over HTTP with a scripted
fake model container in place of exllamav3, so box_probe.py can be exercised
locally against both the unpatched and the patched tree.

  python e2e_fake_server.py --tree PATH --port 18080

The fake "model" honours the grammar the way the engine's filter does: with a
tool call grammar armed by the reasoning end (or active from the start) it
emits a tool call, otherwise it answers in content. With a reasoning budget in
effect it never ends its reasoning, so required requests exercise the phase-2
continuation. Tag strings are built from escapes, never written out literally.
"""

import argparse
import asyncio
import pathlib
import re
import sys
from types import SimpleNamespace

THINK_START = "\u003c" + "think" + "\u003e"
THINK_END = "\u003c/" + "think" + "\u003e"


def build_app(tree):
    sys.path.insert(0, tree)
    import common.model  # noqa: F401
    from fastapi import FastAPI, Depends
    from common import model
    from common.auth import check_api_key
    from common.networking import add_request_id
    from common.templating import PromptTemplate
    from endpoints.OAI import router as oai_router
    from endpoints.OAI.utils.tools import get_toolcall_tags

    tool_start, tool_end = get_toolcall_tags("qwen3_coder")

    template = (
        "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
        "{% if tools %}tools: {% for t in tools %}{{ t.function.name }} {% endfor %}\n{% endif %}"
        "assistant:\n"
        "{% if enable_thinking is defined and not enable_thinking %}"
        + THINK_START + "\n\n" + THINK_END + "\n\n"
        "{% else %}" + THINK_START + "\n{% endif %}"
    )

    def forced_names(grammar):
        return re.findall(r'"<function=" "([^"]+)"', grammar or "")

    def pieces(call):
        grammar, prompt = call["grammar"], call["prompt"]

        def tool_call():
            names = forced_names(grammar)
            name = "calculator" if "calculator" in names else names[0]
            if "start: WS?" in grammar:  # leading whitespace allowed
                yield "\n\n"
            yield tool_start
            yield f"\n<function={name}>\n<parameter=x>\n"
            yield "7 * 8"
            yield "\n</parameter>\n</function>\n"
            yield tool_end

        if prompt.endswith(THINK_END):
            yield from tool_call()
            return
        if prompt.endswith(THINK_START + "\n"):
            if call["budget"] is not None:
                while True:
                    yield "hmm "
            yield from ["I ", "could ", "use ", "a ", "tool."]
            yield "\n" + THINK_END
        if grammar is not None and call["filter_trigger"] in (THINK_END, None):
            if "content?" in grammar and "\ntool: " in prompt:
                # Second turn after a tool result: answer as content and end
                # the turn; the server continues it with a call-only job
                if "start: WS?" in grammar:
                    yield "\n\n"
                yield from ["7 ", "x ", "8 ", "= ", "56."]
                return
            yield from tool_call()
        else:
            yield "\n\n"
            yield from ["7 ", "x ", "8 ", "= ", "56"]

    class Job:
        def __init__(self):
            self.cancelled = False

        async def cancel(self):
            self.cancelled = True

    class Fake:
        harmony = False
        muse_glimmer = False
        tool_format = "qwen3_coder"
        reasoning = True
        reasoning_start_token = THINK_START
        reasoning_end_token = THINK_END
        tool_calls_in_reasoning = True
        start_in_reasoning = "auto"
        reasoning_budget_tokens = None
        reasoning_budget_message = None
        use_vision = False
        template_vars_default = {}
        template_vars_force = {}
        model_dir = pathlib.Path("fake-flash-next")
        prompt_template = PromptTemplate("fake", template)
        hf_model = SimpleNamespace(add_bos_token=lambda: False)
        loaded = True

        def __init__(self):
            self.active_job_ids = {}
            self.tokenizer = SimpleNamespace(single_id={tool_start: 1, tool_end: 2, THINK_END: 3}.get)

        def get_special_tokens(self):
            return {}

        def constrain_generation_output(self, request_id, text):
            job = self.active_job_ids.get(request_id)
            if job is None:
                return False
            job.injected = text
            return True

        async def stream_generate(self, request_id, prompt, params, disconnect_handler=None,
                                  mm_embeddings=None, filter_trigger=None, label=None):
            call = {"prompt": prompt, "grammar": params.grammar_string, "filter_trigger": filter_trigger,
                    "budget": params.reasoning_budget_tokens}
            job = Job()
            job.injected = None
            self.active_job_ids[request_id] = job
            n = 0
            try:
                source = pieces(call)
                while True:
                    await asyncio.sleep(0.001)
                    if job.cancelled:
                        return
                    if job.injected is not None:
                        # Output injection (reasoning budget for auto): the
                        # injected text, then the model answers in content
                        text, job.injected = job.injected, None
                        source = iter([text, "\n\n", "56"])
                    piece = next(source, None)
                    if piece is None:
                        break
                    n += 1
                    yield {"text": piece, "token_ids": [n], "prompt_tokens": 50, "generated_tokens": n}
                    if params.max_tokens and n >= params.max_tokens:
                        yield {"prompt_tokens": 50, "gen_tokens": n, "cached_tokens": 0, "prompt_time": 0.1,
                               "gen_time": 0.1, "total_time": 0.2, "finish_reason": "length",
                               "eos_reason": "max_new_tokens"}
                        return
                yield {"prompt_tokens": 50, "gen_tokens": n, "cached_tokens": 0, "prompt_time": 0.1,
                       "gen_time": 0.1, "total_time": 0.2, "finish_reason": "stop", "eos_reason": "stop_token"}
            finally:
                del self.active_job_ids[request_id]

    model.container = Fake()
    model.check_context_length = lambda *a, **k: None
    app = FastAPI(dependencies=[Depends(add_request_id)])
    app.dependency_overrides[check_api_key] = lambda: None
    app.include_router(oai_router.router)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--port", type=int, default=18080)
    args = ap.parse_args()
    import uvicorn

    uvicorn.run(build_app(args.tree), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
