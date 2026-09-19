"""
Chat completion collector tests for tool_choice "required"/named, driven by a
fake model container whose "model" honours the grammar the way the engine's
filter does: with a tool call grammar active after its reasoning ends it emits
a tool call, otherwise it answers in content.

Covers the request -> generation -> parsing path in both streaming and
non-streaming mode, the phase-2 continuation for a model that never ends its
reasoning (bounded by the reasoning cap, or by max_tokens), and that "auto" and
"none" reach the backend exactly as before (no grammar, same trigger).

Tag strings are built from escapes or taken from the served tool format, never
written out literally.
"""

import asyncio
import os
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import common.model  # noqa: F401 - resolve import cycle ordering
from common import model
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils.chat_completion import (
    _chat_stream_collector,
    _resolve_start_in_reasoning,
)
from endpoints.OAI.utils.tool_choice import REASONING_CAP_ENV, ToolChoiceNotHonoured
from endpoints.OAI.utils.tools import get_toolcall_tags

TOOL_START, TOOL_END = get_toolcall_tags("qwen3_coder")
THINK_START = "\u003c" + "think" + "\u003e"
THINK_END = "\u003c/" + "think" + "\u003e"
TOKEN_IDS = {TOOL_START: 9001, TOOL_END: 9002, THINK_START: 9003, THINK_END: 9004}

PROMPT_THINKING = "user: What is 7 times 8?\nassistant:\n" + THINK_START + "\n"
PROMPT_NO_THINKING = PROMPT_THINKING + "\n" + THINK_END + "\n\n"
PROMPT_TOKENS = 100

REASONING = ["I ", "could ", "use ", "the ", "calculator, ", "but ", "7x8 ", "is ", "easy."]
CONTENT = ["7 ", "x ", "8 ", "= ", "**56**"]
# The parser holds the whitespace after the reasoning end and emits it with the content
ANSWER = "\n\n" + "".join(CONTENT)


def tool(name, properties):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {p: {"type": "string"} for p in properties}},
        },
    }


TOOLS = [tool("calculator", ["expression"]), tool("get_weather", ["city"])]


def forced_names(grammar):
    return re.findall(r'"<function=" "([^"]+)"', grammar or "")


class FakeJob:
    def __init__(self):
        self.cancelled = False

    async def cancel(self):
        self.cancelled = True


class FakeModel:
    """
    Scripted model. behaviour:
      "answer"   - reasons, ends reasoning, then answers in content unless a
                   tool call grammar is active (then calls a tool it is allowed to)
      "endless"  - reasons forever
      "eos"      - reasons, then stops (EOS) without ending the reasoning
      "call_in_reasoning" - reasons, calls a tool without ending the
                   reasoning, then stops
      "answer_then_stop" - with a content grammar active, writes its answer
                   as content and stops (EOS), as a model that would rather
                   answer does
      "content_loop" - with a grammar active, writes content until the loop
                   detector ends the job
      "answer_then_call" - like "answer", but with a grammar active it writes
                   its answer as content first and then calls the tool (the
                   grammar admits content before the call)
    """

    def __init__(self, behaviour="answer", preferred="calculator", ignores_grammar=False):
        self.behaviour = behaviour
        self.preferred = preferred
        # Stands in for a backend that dropped the filter (grammar failed to compile)
        self.ignores_grammar = ignores_grammar

    def pieces(self, call):
        grammar = call["grammar"]
        continuing = call["prompt"].endswith(THINK_END)

        def tool_call():
            names = forced_names(grammar)
            name = self.preferred if self.preferred in names or self.ignores_grammar else names[0]
            param = "expression" if name == "calculator" else "city"
            if "start: WS?" in grammar:  # leading whitespace allowed
                yield "\n\n"
            yield TOOL_START
            yield f"\n<function={name}>\n<parameter={param}>\n"
            yield "7 * 8" if name == "calculator" else "Paris"
            yield "\n</parameter>\n</function>\n"
            yield TOOL_END

        if continuing:
            # Phase 2: grammar active from the first token
            yield from tool_call()
            yield ("stop", "stop_token")
            return

        if call["prompt"].endswith(THINK_START + "\n"):
            if self.behaviour == "endless":
                while True:
                    yield "hmm "
            yield from REASONING
            if self.behaviour == "call_in_reasoning":
                yield from tool_call()
                yield ("stop", "stop_token")
                return
            if self.behaviour == "eos":
                yield ("stop", "stop_token")
                return
            yield "\n" + THINK_END

        # The grammar is active here if it was armed by the reasoning end or
        # has no trigger at all
        grammar_active = grammar is not None and call["filter_trigger"] in (THINK_END, None)
        if grammar_active and "content?" in grammar and self.behaviour == "answer_then_stop":
            yield "\n\n"
            yield from CONTENT
            yield "\n\n"
            yield ("stop", "stop_token")
            return
        if grammar_active and self.behaviour == "content_loop":
            yield "\n\n"
            yield from CONTENT
            yield ("stop", "loop_detected")
            return
        if grammar_active and self.behaviour == "answer_then_call":
            assert "content?" in grammar
            yield "\n\n"
            yield from CONTENT
            yield from tool_call()  # opens with the whitespace before TOOL_START
        elif grammar_active and not (self.ignores_grammar and self.behaviour == "answer"):
            yield from tool_call()
        else:
            yield "\n\n"
            yield from CONTENT
        yield ("stop", "stop_token")


class FakeContainer:
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

    def __init__(self, fake_model):
        self.fake_model = fake_model
        self.calls = []
        self.jobs = []
        self.active_job_ids = {}
        self.injections = []
        self.tokenizer = SimpleNamespace(single_id=TOKEN_IDS.get)

    def constrain_generation_output(self, request_id, text):
        self.injections.append(text)
        return True

    async def stream_generate(
        self,
        request_id,
        prompt,
        params,
        disconnect_handler=None,
        mm_embeddings=None,
        filter_trigger=None,
        label=None,
    ):
        call = {
            "request_id": request_id,
            "prompt": prompt,
            "grammar": params.grammar_string,
            "filter_trigger": filter_trigger,
            "max_tokens": params.max_tokens,
        }
        self.calls.append(call)
        job = FakeJob()
        self.jobs.append(job)
        self.active_job_ids[request_id] = job
        generated = 0
        prompt_tokens = PROMPT_TOKENS + (len(prompt) - len(PROMPT_THINKING))

        def finish(finish_reason, eos_reason):
            return {
                "request_id": request_id,
                "prompt_tokens": prompt_tokens,
                "gen_tokens": generated,
                "cached_tokens": prompt_tokens - 1,
                "finish_reason": finish_reason,
                "eos_reason": eos_reason,
            }

        try:
            for piece in self.fake_model.pieces(call):
                await asyncio.sleep(0)
                if job.cancelled:
                    return
                if isinstance(piece, tuple):
                    yield finish(*piece)
                    return
                generated += 1
                yield {
                    "request_id": request_id,
                    "text": piece,
                    "token_ids": [generated],
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated,
                }
                if params.max_tokens and generated >= params.max_tokens:
                    yield finish("length", "max_new_tokens")
                    return
        finally:
            del self.active_job_ids[request_id]


def request(**kwargs):
    kwargs.setdefault("messages", [{"role": "user", "content": "What is 7 times 8?"}])
    kwargs.setdefault("tools", TOOLS)
    return ChatCompletionRequest(**kwargs)


class CollectorTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._env = os.environ.pop(REASONING_CAP_ENV, None)

    def tearDown(self):
        os.environ.pop(REASONING_CAP_ENV, None)
        if self._env is not None:
            os.environ[REASONING_CAP_ENV] = self._env

    async def run_collector(self, data, fake, prompt=PROMPT_THINKING, streaming=False):
        with patch.object(model, "container", fake):
            start = _resolve_start_in_reasoning(prompt, data)
            params = data.model_copy(deep=True)
            if not streaming:
                result = await asyncio.wait_for(
                    _chat_stream_collector(0, None, "req", prompt, params, start, streaming_mode=False),
                    timeout=10,
                )
                if isinstance(result, Exception):
                    raise result
                return result
            queue = asyncio.Queue()
            await asyncio.wait_for(
                _chat_stream_collector(0, queue, "req", prompt, params, start, streaming_mode=True),
                timeout=10,
            )
            items = []
            while not queue.empty():
                item = queue.get_nowait()
                if isinstance(item, Exception):
                    raise item
                items.append(item)
            return items

    def assert_stream_shape(self, items):
        """Reasoning deltas only before the end, no content, one finish chunk carrying the calls."""
        finishes = [i for i in items if i.get("finish_reason")]
        self.assertEqual(len(finishes), 1)
        self.assertIs(items[-1], finishes[0])
        self.assertFalse(any(i.get("delta_content") for i in items))
        self.assertTrue(items[0].get("delta_reasoning_content"))
        return finishes[0]


class AutoNoneTests(CollectorTestBase):
    async def test_auto_is_unconstrained(self):
        for choice in (None, "auto"):
            fake = FakeContainer(FakeModel())
            result = await self.run_collector(request(tool_choice=choice), fake)
            self.assertEqual(len(fake.calls), 1)
            self.assertIsNone(fake.calls[0]["grammar"])
            self.assertEqual(fake.calls[0]["filter_trigger"], THINK_END)
            self.assertEqual(result["content"], ANSWER)
            self.assertEqual(result["tool_calls"], [])
            self.assertEqual(result["finish_reason"], "stop")

    async def test_none_is_unconstrained(self):
        fake = FakeContainer(FakeModel())
        result = await self.run_collector(request(tool_choice="none"), fake)
        self.assertIsNone(fake.calls[0]["grammar"])
        self.assertEqual(result["content"], ANSWER)

    async def test_auto_keeps_budget_injection(self):
        fake = FakeContainer(FakeModel("endless"))
        data = request(tool_choice="auto", reasoning_budget_tokens=5, max_tokens=30)
        result = await self.run_collector(data, fake)
        self.assertEqual(fake.injections, [THINK_END])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(result["finish_reason"], "length")


class RequiredTests(CollectorTestBase):
    async def test_required_non_streaming(self):
        fake = FakeContainer(FakeModel())
        result = await self.run_collector(request(tool_choice="required"), fake)
        self.assertEqual(len(fake.calls), 1)
        call = fake.calls[0]
        self.assertEqual(call["filter_trigger"], THINK_END)
        self.assertIn("<[9001]>", call["grammar"])
        self.assertEqual(forced_names(call["grammar"]), ["calculator", "get_weather"])
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertIsNone(result["content"])
        self.assertEqual(result["reasoning_content"], "".join(REASONING) + "\n")
        self.assertEqual([c["function"]["name"] for c in result["tool_calls"]], ["calculator"])
        self.assertEqual(result["tool_calls"][0]["function"]["arguments"], '{"expression": "7 * 8"}')

    async def test_required_streaming_order(self):
        fake = FakeContainer(FakeModel())
        items = await self.run_collector(request(tool_choice="required"), fake, streaming=True)
        final = self.assert_stream_shape(items)
        self.assertEqual(final["finish_reason"], "tool_calls")
        self.assertEqual([c["function"]["name"] for c in final["delta_tool_calls"]], ["calculator"])
        reasoning = "".join(i.get("delta_reasoning_content") or "" for i in items)
        self.assertEqual(reasoning, "".join(REASONING) + "\n")

    async def test_named_calls_the_named_function(self):
        fake = FakeContainer(FakeModel(preferred="calculator"))
        data = request(tool_choice={"type": "function", "function": {"name": "get_weather"}})
        result = await self.run_collector(data, fake)
        self.assertEqual(forced_names(fake.calls[0]["grammar"]), ["get_weather"])
        self.assertIn("start: WS? content? call WS?", fake.calls[0]["grammar"])
        self.assertEqual([c["function"]["name"] for c in result["tool_calls"]], ["get_weather"])

    async def test_required_without_thinking_constrains_from_first_token(self):
        fake = FakeContainer(FakeModel())
        result = await self.run_collector(
            request(tool_choice="required"), fake, prompt=PROMPT_NO_THINKING
        )
        self.assertEqual(len(fake.calls), 1)
        self.assertIsNone(fake.calls[0]["filter_trigger"])
        self.assertIn("start: content? call", fake.calls[0]["grammar"])
        self.assertEqual(result["finish_reason"], "tool_calls")

    async def test_required_without_thinking_streams_no_content(self):
        fake = FakeContainer(FakeModel())
        items = await self.run_collector(
            request(tool_choice="required"), fake, prompt=PROMPT_NO_THINKING, streaming=True
        )
        finishes = [i for i in items if i.get("finish_reason")]
        self.assertEqual(len(finishes), 1)
        self.assertFalse(any(i.get("delta_content") for i in items))
        self.assertEqual(len(finishes[0]["delta_tool_calls"]), 1)

    async def test_unsupported_format_served_as_auto(self):
        fake = FakeContainer(FakeModel())
        fake.tool_format = "glm4_5"
        result = await self.run_collector(request(tool_choice="required"), fake)
        self.assertIsNone(fake.calls[0]["grammar"])
        self.assertEqual(result["content"], ANSWER)


class NotHonouredTests(CollectorTestBase):
    """A forced request that ends on its own without the forced call fails (503)."""

    async def test_content_answer_fails_non_streaming(self):
        fake = FakeContainer(FakeModel(ignores_grammar=True))
        with self.assertRaises(ToolChoiceNotHonoured):
            await self.run_collector(request(tool_choice="required"), fake)

    async def test_content_answer_fails_streaming(self):
        fake = FakeContainer(FakeModel(ignores_grammar=True))
        with self.assertRaises(ToolChoiceNotHonoured):
            await self.run_collector(request(tool_choice="required"), fake, streaming=True)

    async def test_wrong_function_fails_named(self):
        fake = FakeContainer(FakeModel("tool_anyway", preferred="calculator", ignores_grammar=True))
        data = request(tool_choice={"type": "function", "function": {"name": "get_weather"}})
        with self.assertRaises(ToolChoiceNotHonoured):
            await self.run_collector(data, fake)

    async def test_auto_content_answer_is_fine(self):
        fake = FakeContainer(FakeModel(ignores_grammar=True))
        result = await self.run_collector(request(tool_choice="auto"), fake)
        self.assertEqual(result["content"], ANSWER)


class ContentWithCallTests(CollectorTestBase):
    """A forced turn may carry the answer as content before the call."""

    async def test_non_streaming_returns_content_and_call(self):
        fake = FakeContainer(FakeModel("answer_then_call"))
        result = await self.run_collector(request(tool_choice="required"), fake)
        self.assertIn("content? call", fake.calls[0]["grammar"])
        self.assertEqual(result["content"], ANSWER + "\n\n")
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual([c["function"]["name"] for c in result["tool_calls"]], ["calculator"])
        self.assertEqual(result["reasoning_content"], "".join(REASONING) + "\n")

    async def test_streaming_order_reasoning_content_call(self):
        fake = FakeContainer(FakeModel("answer_then_call"))
        items = await self.run_collector(request(tool_choice="required"), fake, streaming=True)
        finishes = [i for i in items if i.get("finish_reason")]
        self.assertEqual(len(finishes), 1)
        self.assertIs(items[-1], finishes[0])
        kinds = []
        for i in items:
            if i.get("delta_reasoning_content"):
                kinds.append("r")
            if i.get("delta_content"):
                kinds.append("c")
            if i.get("delta_tool_calls"):
                kinds.append("t")
        shape = [k for n, k in enumerate(kinds) if n == 0 or kinds[n - 1] != k]
        self.assertEqual(shape, ["r", "c", "t"])
        self.assertEqual("".join(i.get("delta_content") or "" for i in items), ANSWER + "\n\n")
        self.assertEqual(finishes[0]["finish_reason"], "tool_calls")

    async def test_named_with_content(self):
        fake = FakeContainer(FakeModel("answer_then_call"))
        data = request(tool_choice={"type": "function", "function": {"name": "get_weather"}})
        result = await self.run_collector(data, fake)
        self.assertEqual(result["content"], ANSWER + "\n\n")
        self.assertEqual([c["function"]["name"] for c in result["tool_calls"]], ["get_weather"])


class ContentThenCallContinuationTests(CollectorTestBase):
    """A forced turn that ends in content is continued by a call-only job."""

    async def test_non_streaming(self):
        fake = FakeContainer(FakeModel("answer_then_stop"))
        result = await self.run_collector(request(tool_choice="required"), fake)
        self.assertEqual(len(fake.calls), 2)
        first, cont = fake.calls
        self.assertEqual(cont["request_id"], "req-tc2")
        self.assertIsNone(cont["filter_trigger"])
        self.assertNotIn("content", cont["grammar"])
        self.assertTrue(cont["grammar"].startswith("start: call"))
        self.assertEqual(
            cont["prompt"],
            PROMPT_THINKING + "".join(REASONING) + "\n" + THINK_END + ANSWER + "\n\n",
        )
        self.assertEqual(result["content"], ANSWER + "\n\n")
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual([c["function"]["name"] for c in result["tool_calls"]], ["calculator"])
        # Usage: client prompt; completion tokens of both jobs
        self.assertEqual(result["prompt_tokens"], PROMPT_TOKENS)
        self.assertEqual(result["gen_tokens"], len(REASONING) + 1 + 1 + len(CONTENT) + 1 + 5)

    async def test_streaming_single_finish(self):
        fake = FakeContainer(FakeModel("answer_then_stop"))
        items = await self.run_collector(request(tool_choice="required"), fake, streaming=True)
        finishes = [i for i in items if i.get("finish_reason")]
        self.assertEqual(len(finishes), 1)
        self.assertIs(items[-1], finishes[0])
        kinds = []
        for i in items:
            for key, k in (("delta_reasoning_content", "r"), ("delta_content", "c"), ("delta_tool_calls", "t")):
                if i.get(key):
                    kinds.append(k)
        self.assertEqual([k for n, k in enumerate(kinds) if n == 0 or kinds[n - 1] != k], ["r", "c", "t"])

    async def test_without_thinking(self):
        fake = FakeContainer(FakeModel("answer_then_stop"))
        result = await self.run_collector(request(tool_choice="required"), fake, prompt=PROMPT_NO_THINKING)
        self.assertEqual(len(fake.calls), 2)
        self.assertIsNone(fake.calls[0]["filter_trigger"])
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertTrue(result["content"].strip())

    async def test_named(self):
        fake = FakeContainer(FakeModel("answer_then_stop"))
        data = request(tool_choice={"type": "function", "function": {"name": "get_weather"}})
        result = await self.run_collector(data, fake)
        self.assertEqual(forced_names(fake.calls[1]["grammar"]), ["get_weather"])
        self.assertEqual([c["function"]["name"] for c in result["tool_calls"]], ["get_weather"])

    async def test_content_loop_fails_loud(self):
        # R523 round 2: a content loop ended by the loop detector returned 200
        # with no call; any end other than a client limit must fail loud
        fake = FakeContainer(FakeModel("content_loop"))
        with self.assertRaises(ToolChoiceNotHonoured):
            await self.run_collector(request(tool_choice="required"), fake)

    async def test_content_disabled_keeps_single_job(self):
        from endpoints.OAI.utils.tool_choice import CONTENT_MAX_TOKENS_ENV

        os.environ[CONTENT_MAX_TOKENS_ENV] = "0"
        try:
            fake = FakeContainer(FakeModel())
            result = await self.run_collector(request(tool_choice="required"), fake)
        finally:
            os.environ.pop(CONTENT_MAX_TOKENS_ENV, None)
        self.assertEqual(len(fake.calls), 1)
        self.assertNotIn("content", fake.calls[0]["grammar"])
        self.assertEqual(result["finish_reason"], "tool_calls")


class EndlessReasoningTests(CollectorTestBase):
    async def test_cap_forces_phase_two(self):
        os.environ[REASONING_CAP_ENV] = "20"
        fake = FakeContainer(FakeModel("endless"))
        result = await self.run_collector(request(tool_choice="required"), fake)
        self.assertEqual(len(fake.calls), 2)
        p1, p2 = fake.calls
        self.assertTrue(fake.jobs[0].cancelled)
        self.assertEqual(p1["filter_trigger"], THINK_END)
        self.assertIsNone(p2["filter_trigger"])
        self.assertEqual(p2["grammar"], p1["grammar"])
        self.assertEqual(p2["request_id"], "req-tc2")
        self.assertEqual(p2["prompt"], PROMPT_THINKING + "hmm " * 20 + THINK_END)
        self.assertEqual(fake.injections, [])
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual([c["function"]["name"] for c in result["tool_calls"]], ["calculator"])
        self.assertEqual(result["reasoning_content"], "hmm " * 20)
        self.assertIsNone(result["content"])
        # Usage: the client's prompt, all generated tokens of both phases
        self.assertEqual(result["prompt_tokens"], PROMPT_TOKENS)
        self.assertEqual(result["gen_tokens"], 20 + 6)
        self.assertLessEqual(result["cached_tokens"], PROMPT_TOKENS)

    async def test_cap_streaming_single_finish(self):
        os.environ[REASONING_CAP_ENV] = "12"
        fake = FakeContainer(FakeModel("endless"))
        items = await self.run_collector(request(tool_choice="required"), fake, streaming=True)
        final = self.assert_stream_shape(items)
        self.assertEqual(final["finish_reason"], "tool_calls")
        self.assertEqual(len(final["delta_tool_calls"]), 1)
        reasoning = "".join(i.get("delta_reasoning_content") or "" for i in items)
        self.assertEqual(reasoning, "hmm " * 12)

    async def test_budget_is_the_cap(self):
        fake = FakeContainer(FakeModel("endless"))
        data = request(
            tool_choice="required", reasoning_budget_tokens=7, reasoning_budget_message=" Time is up."
        )
        result = await self.run_collector(data, fake)
        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(fake.calls[1]["prompt"].endswith("hmm " * 7 + " Time is up." + THINK_END))
        self.assertEqual(fake.injections, [])  # no output injection: it would disable the filter
        self.assertEqual(result["reasoning_content"], "hmm " * 7 + " Time is up.")
        self.assertEqual(result["finish_reason"], "tool_calls")

    async def test_max_tokens_bounds_reasoning_without_cap(self):
        os.environ[REASONING_CAP_ENV] = "0"
        fake = FakeContainer(FakeModel("endless"))
        result = await self.run_collector(request(tool_choice="required", max_tokens=15), fake)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["tool_calls"], [])

    async def test_phase_two_gets_the_remaining_max_tokens(self):
        os.environ[REASONING_CAP_ENV] = "10"
        fake = FakeContainer(FakeModel("endless"))
        await self.run_collector(request(tool_choice="required", max_tokens=100), fake)
        self.assertEqual(fake.calls[1]["max_tokens"], 90)

    async def test_eos_inside_reasoning_forces_phase_two(self):
        fake = FakeContainer(FakeModel("eos"))
        result = await self.run_collector(request(tool_choice="required"), fake)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[1]["prompt"], PROMPT_THINKING + "".join(REASONING) + THINK_END)
        self.assertEqual(result["finish_reason"], "tool_calls")

    async def test_call_without_reasoning_end_is_not_repeated(self):
        # A call already emitted (inside the reasoning, or by a filter that
        # constrained from the first token) must not trigger phase 2
        fake = FakeContainer(FakeModel("call_in_reasoning"))
        result = await self.run_collector(request(tool_choice="required"), fake)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual(len(result["tool_calls"]), 1)

    async def test_eos_inside_reasoning_auto_unchanged(self):
        fake = FakeContainer(FakeModel("eos"))
        result = await self.run_collector(request(tool_choice="auto"), fake)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(result["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
