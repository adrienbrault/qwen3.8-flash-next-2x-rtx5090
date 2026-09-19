"""
Local differential check (not shipped in the image): run the chat completion
collector of two TabbyAPI trees (unpatched ref and patched overlay) against the
same scripted fake backend for tool_choice None/"auto"/"none", and compare
everything that leaves the collector: the arguments each backend call receives
(prompt, full sampler params, filter trigger), output injections, and the
emitted stream chunks / final responses.

Usage:
  python auto_none_identity.py --tree REF --out ref.json
  python auto_none_identity.py --tree PATCHED --out patched.json
  python auto_none_identity.py --compare ref.json patched.json

Tag strings are built from escapes, never written out literally.
"""

import argparse
import asyncio
import json
import sys
from types import SimpleNamespace

THINK_START = "\u003c" + "think" + "\u003e"
THINK_END = "\u003c/" + "think" + "\u003e"
PROMPT_THINKING = "user: What is 7 times 8?\nassistant:\n" + THINK_START + "\n"
PROMPT_NO_THINKING = PROMPT_THINKING + "\n" + THINK_END + "\n\n"


def scenarios():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "calc",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
            },
        }
    ]
    for choice in (None, "auto", "none"):
        for streaming in (False, True):
            for prompt_name in ("thinking", "no_thinking"):
                for behaviour in ("answer", "tool", "eos", "endless_max", "endless_budget"):
                    for n in (1, 2):
                        yield {
                            "choice": choice,
                            "streaming": streaming,
                            "prompt": prompt_name,
                            "behaviour": behaviour,
                            "n": n,
                            "tools": tools,
                        }


def run_tree(tree):
    sys.path.insert(0, tree)
    import common.model  # noqa: F401
    from common import model
    from endpoints.OAI.types.chat_completion import ChatCompletionRequest
    from endpoints.OAI.utils import chat_completion as cc
    from endpoints.OAI.utils.tools import get_toolcall_tags

    tool_start, tool_end = get_toolcall_tags("qwen3_coder")

    def pieces(behaviour, prompt):
        if prompt.endswith(THINK_START + "\n"):
            if behaviour.startswith("endless"):
                while True:
                    yield "hmm "
            yield from ["I ", "think ", "so."]
            if behaviour == "eos":
                yield ("stop", "stop_token")
                return
            yield "\n" + THINK_END
        yield "\n\n"
        if behaviour == "tool":
            yield tool_start
            yield "\n<function=calculator>\n<parameter=expression>\n7*8\n</parameter>\n</function>\n"
            yield tool_end
        else:
            yield from ["7 ", "x ", "8 ", "= ", "56"]
        yield ("stop", "stop_token")

    class Job:
        async def cancel(self):
            pass

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

        def __init__(self, behaviour):
            self.behaviour = behaviour
            self.calls = []
            self.injections = []
            self.active_job_ids = {}
            self.tokenizer = SimpleNamespace(single_id={tool_start: 1, tool_end: 2}.get)

        def constrain_generation_output(self, request_id, text):
            self.injections.append([request_id, text])
            return True

        async def stream_generate(
            self, request_id, prompt, params, disconnect_handler=None, mm_embeddings=None,
            filter_trigger=None, label=None,
        ):
            self.calls.append(
                {
                    "request_id": request_id,
                    "prompt": prompt,
                    "params": params.model_dump(mode="json"),
                    "filter_trigger": filter_trigger,
                    "label": label,
                }
            )
            self.active_job_ids[request_id] = Job()
            generated = 0
            try:
                for piece in pieces(self.behaviour, prompt):
                    await asyncio.sleep(0)
                    if isinstance(piece, tuple):
                        yield {"prompt_tokens": 10, "gen_tokens": generated, "cached_tokens": 3,
                               "finish_reason": piece[0], "eos_reason": piece[1]}
                        return
                    generated += 1
                    yield {"text": piece, "token_ids": [generated], "prompt_tokens": 10,
                           "generated_tokens": generated}
                    if params.max_tokens and generated >= params.max_tokens:
                        yield {"prompt_tokens": 10, "gen_tokens": generated, "cached_tokens": 3,
                               "finish_reason": "length", "eos_reason": "max_new_tokens"}
                        return
            finally:
                del self.active_job_ids[request_id]

    def strip(chunk_json):
        d = json.loads(chunk_json)
        d.pop("created", None)
        for choice in d["choices"]:
            for tc in choice["delta"].get("tool_calls") or []:
                tc.pop("id", None)  # random per call
        return d

    async def one(sc):
        kwargs = {
            "messages": [{"role": "user", "content": "What is 7 times 8?"}],
            "tools": sc["tools"],
            "n": sc["n"],
        }
        if sc["choice"] is not None:
            kwargs["tool_choice"] = sc["choice"]
        if sc["behaviour"] == "endless_max":
            kwargs["max_tokens"] = 12
        if sc["behaviour"] == "endless_budget":
            kwargs["max_tokens"] = 20
            kwargs["reasoning_budget_tokens"] = 5
            kwargs["reasoning_budget_message"] = " stop."
        data = ChatCompletionRequest(**kwargs)
        cc.validate_tool_choice(data) if hasattr(cc, "validate_tool_choice") else None
        prompt = PROMPT_THINKING if sc["prompt"] == "thinking" else PROMPT_NO_THINKING
        fake = Fake(sc["behaviour"])
        model.container = fake
        start = cc._resolve_start_in_reasoning(prompt, data)
        out = []
        for idx in range(data.n):
            params = data.model_copy(deep=True)
            if sc["streaming"]:
                q = asyncio.Queue()
                await cc._chat_stream_collector(idx, q, f"r{idx}", prompt, params, start,
                                                streaming_mode=True, label="L")
                while not q.empty():
                    g = q.get_nowait()
                    s, _, finish, empty = cc._compose_serialize_stream_chunk("rid", g, "m")
                    out.append({"chunk": strip(s), "empty": empty})
            else:
                g = await cc._chat_stream_collector(idx, None, f"r{idx}", prompt, params, start,
                                                    streaming_mode=False, label="L")
                gens = [g]
                resp = cc._compose_response("rid", gens, "m", True)
                d = json.loads(resp.model_dump_json())
                d.pop("created", None)
                d.pop("id", None)
                for choice in d["choices"]:
                    for tc in choice["message"].get("tool_calls") or []:
                        tc.pop("id", None)
                out.append(d)
        return {"calls": fake.calls, "injections": fake.injections, "out": out}

    async def main():
        results = []
        for sc in scenarios():
            key = {k: v for k, v in sc.items() if k != "tools"}
            results.append({"scenario": key, "result": await one(sc)})
        return results

    return asyncio.run(main())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        a, b = (json.load(open(p)) for p in args.compare)
        assert len(a) == len(b), (len(a), len(b))
        diffs = [x["scenario"] for x, y in zip(a, b) if x != y]
        print(f"{len(a)} scenarios compared, {len(diffs)} differ")
        for d in diffs:
            print("DIFF", d)
        sys.exit(1 if diffs else 0)
    results = run_tree(args.tree)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, sort_keys=True)
    print(f"{len(results)} scenarios written to {args.out}")


if __name__ == "__main__":
    main()
