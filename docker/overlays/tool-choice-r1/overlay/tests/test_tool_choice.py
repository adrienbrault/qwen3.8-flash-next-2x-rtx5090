"""
Unit tests for tool_choice "required"/named enforcement: request validation,
mode selection and the tool call grammar.

The grammar tests run the grammar through llguidance (the matcher behind
exllamav3's LLGuidanceFilter) with a small synthetic byte-level tokenizer, so
they need llguidance and tokenizers but no model files. Set
TABBY_TOOLCHOICE_TOKENIZER to a model's tokenizer.json to repeat them with a
real vocabulary.

Tag strings are built from escapes or taken from the served tool format, never
written out literally.
"""

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils.tool_choice import (
    CONTENT_MAX_TOKENS_ENV,
    DEFAULT_REASONING_CAP,
    REASONING_CAP_ENV,
    build_qwen3_coder_grammar,
    forces_tool_call,
    prepare_tool_choice_forcing,
    resolve_reasoning_cap,
    resolve_tool_choice_forcing,
    supports_forcing,
    validate_tool_choice,
)
from endpoints.OAI.utils.tools import get_toolcall_tags

try:
    from llguidance import LLMatcher, LLTokenizer, grammar_from
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
    import numpy as np

    HAVE_LLG = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_LLG = False

TOOL_START, TOOL_END = get_toolcall_tags("qwen3_coder")
THINK_START = "\u003c" + "think" + "\u003e"
THINK_END = "\u003c/" + "think" + "\u003e"
EOS_TEXT = "\u003c|" + "im_end" + "|\u003e"


def tool(name, properties=None):
    parameters = {"type": "object"}
    if properties is not None:
        parameters["properties"] = {p: {"type": "string"} for p in properties}
    return {
        "type": "function",
        "function": {"name": name, "description": f"{name} tool", "parameters": parameters},
    }


TOOLS = [
    tool("calculator", ["expression"]),
    tool("get_weather", ["city", "unit"]),
    tool("free_form"),  # no declared properties
]


def request(**kwargs):
    kwargs.setdefault("messages", [{"role": "user", "content": "What is 7 times 8?"}])
    return ChatCompletionRequest(**kwargs)


def call_text(name, params=None):
    body = "".join(
        f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in (params or {}).items()
    )
    return f"{TOOL_START}\n<function={name}>\n{body}</function>\n{TOOL_END}"


class ValidationTests(unittest.TestCase):
    def assert_400(self, **kwargs):
        with self.assertRaises(HTTPException) as ctx:
            validate_tool_choice(request(**kwargs))
        self.assertEqual(ctx.exception.status_code, 400)
        return ctx.exception

    def test_required_without_tools(self):
        self.assert_400(tool_choice="required")

    def test_required_with_empty_tools(self):
        self.assert_400(tool_choice="required", tools=[])

    def test_named_without_tools(self):
        self.assert_400(tool_choice={"type": "function", "function": {"name": "calculator"}})

    def test_named_unknown_function(self):
        exc = self.assert_400(
            tools=TOOLS, tool_choice={"type": "function", "function": {"name": "nope"}}
        )
        self.assertIn("nope", exc.detail)

    def test_required_with_client_constraint(self):
        self.assert_400(tools=TOOLS, tool_choice="required", json_schema={"type": "object"})
        self.assert_400(tools=TOOLS, tool_choice="required", regex_pattern="a+")
        self.assert_400(tools=TOOLS, tool_choice="required", grammar_string='start: "a"')
        self.assert_400(
            tools=TOOLS,
            tool_choice="required",
            response_format={"type": "json_schema", "json_schema": {"type": "object"}},
        )

    def test_valid_requests_pass(self):
        validate_tool_choice(request(tools=TOOLS, tool_choice="required"))
        validate_tool_choice(
            request(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "calculator"}})
        )

    def test_auto_none_absent_never_rejected(self):
        # Including combinations that would be rejected for required
        for choice in (None, "auto", "none"):
            validate_tool_choice(request(tool_choice=choice))
            validate_tool_choice(request(tool_choice=choice, tools=[]))
            validate_tool_choice(
                request(tool_choice=choice, tools=TOOLS, json_schema={"type": "object"})
            )


class ModeSelectionTests(unittest.TestCase):
    def test_auto_none_absent_select_nothing(self):
        for choice in (None, "auto", "none"):
            data = request(tools=TOOLS, tool_choice=choice)
            self.assertFalse(forces_tool_call(data))
            self.assertIsNone(resolve_tool_choice_forcing(data))

    def test_required(self):
        forcing = resolve_tool_choice_forcing(request(tools=TOOLS, tool_choice="required"))
        self.assertEqual((forcing.mode, forcing.name, forcing.single_call), ("required", None, False))

    def test_required_without_parallel_calls_is_single(self):
        forcing = resolve_tool_choice_forcing(
            request(tools=TOOLS, tool_choice="required", parallel_tool_calls=False)
        )
        self.assertTrue(forcing.single_call)

    def test_named(self):
        data = request(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "get_weather"}})
        self.assertTrue(forces_tool_call(data))
        forcing = resolve_tool_choice_forcing(data)
        self.assertEqual((forcing.mode, forcing.name, forcing.single_call), ("named", "get_weather", True))

    def test_supported_formats(self):
        for fmt in ("qwen3_coder", "qwen3_5", "step3_5", "step3_7"):
            self.assertTrue(supports_forcing(fmt), fmt)
        for fmt in (None, "", "harmony", "glm4_5", "mistral", "unknown"):
            self.assertFalse(supports_forcing(fmt), fmt)

    def test_unsupported_format_serves_as_auto(self):
        data = request(tools=TOOLS, tool_choice="required")
        self.assertIsNone(prepare_tool_choice_forcing(data, "glm4_5", lambda _: None, "t"))
        self.assertIsNone(prepare_tool_choice_forcing(data, None, lambda _: None, "t"))

    def test_reasoning_cap(self):
        self.assertEqual(resolve_reasoning_cap(100), 100)
        self.assertEqual(resolve_reasoning_cap(0), 0)
        old = os.environ.pop(REASONING_CAP_ENV, None)
        try:
            self.assertEqual(resolve_reasoning_cap(None), DEFAULT_REASONING_CAP)
            os.environ[REASONING_CAP_ENV] = "512"
            self.assertEqual(resolve_reasoning_cap(None), 512)
            os.environ[REASONING_CAP_ENV] = "0"
            self.assertIsNone(resolve_reasoning_cap(None))
        finally:
            os.environ.pop(REASONING_CAP_ENV, None)
            if old is not None:
                os.environ[REASONING_CAP_ENV] = old


class GrammarTextTests(unittest.TestCase):
    def build(self, choice="required", ids=None, tools=TOOLS, **kwargs):
        data = request(tools=tools, tool_choice=choice, **kwargs)
        forcing = resolve_tool_choice_forcing(data)
        ids = ids or {}
        return build_qwen3_coder_grammar(
            forcing, data.tools, TOOL_START, TOOL_END, lambda t: ids.get(t)
        )

    def test_single_token_tags_referenced_by_id(self):
        g = self.build(ids={TOOL_START: 1001, TOOL_END: 1002})
        self.assertIn("<[1001]>", g)
        self.assertIn("<[1002]>", g)
        self.assertNotIn(TOOL_START, g)

    def test_multi_token_tags_fall_back_to_text(self):
        g = self.build()
        self.assertIn('"' + TOOL_START + '"', g)

    def test_content_before_call_only_with_single_token_tool_start(self):
        data = request(tools=TOOLS, tool_choice="required")
        forcing = resolve_tool_choice_forcing(data)
        ids = {TOOL_START: 1001, TOOL_END: 1002}.get
        g = build_qwen3_coder_grammar(forcing, data.tools, TOOL_START, TOOL_END, ids, content_max_tokens=64)
        self.assertIn("start: WS? content? call (WS? call)* WS? | WS? content\n", g)
        self.assertIn("content[max_tokens=64]", g)
        # Off by default in the builder, and off when TOOL_START is text (the
        # content lexeme could swallow a spelled-out tag)
        self.assertNotIn("content", build_qwen3_coder_grammar(forcing, data.tools, TOOL_START, TOOL_END, ids))
        text_tags = build_qwen3_coder_grammar(
            forcing, data.tools, TOOL_START, TOOL_END, lambda _: None, content_max_tokens=64
        )
        self.assertNotIn("content", text_tags)

    def test_content_env(self):
        data = request(tools=TOOLS, tool_choice="required")
        ids = {TOOL_START: 1001, TOOL_END: 1002}.get
        with patch.dict(os.environ, {CONTENT_MAX_TOKENS_ENV: "0"}):
            forcing = prepare_tool_choice_forcing(data, "qwen3_coder", ids, "t")
            self.assertNotIn("content", forcing.grammar)
            self.assertIsNone(forcing.call_grammar)
        with patch.dict(os.environ, {CONTENT_MAX_TOKENS_ENV: "7"}):
            forcing = prepare_tool_choice_forcing(data, "qwen3_coder", ids, "t")
            self.assertIn("content[max_tokens=7]", forcing.grammar)
            self.assertIn("start: call (WS? call)*\n", forcing.call_grammar)
            self.assertNotIn("content", forcing.call_grammar)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(CONTENT_MAX_TOKENS_ENV, None)
            self.assertIn("content[max_tokens=1024]", prepare_tool_choice_forcing(data, "qwen3_coder", ids, "t").grammar)

    def test_named_grammar_lists_only_that_function(self):
        g = self.build(choice={"type": "function", "function": {"name": "get_weather"}})
        self.assertIn('"get_weather"', g)
        self.assertNotIn('"calculator"', g)
        self.assertIn("start: WS? call WS?", g)

    def test_leading_whitespace_only_when_armed_by_reasoning_end(self):
        data = request(tools=TOOLS, tool_choice="required")
        armed = prepare_tool_choice_forcing(data, "qwen3_coder", lambda _: None, "t")
        unarmed = prepare_tool_choice_forcing(
            data, "qwen3_coder", lambda _: None, "t", armed_by_reasoning_end=False
        )
        self.assertIn("start: WS? call (WS? call)* WS?\n", armed.grammar)
        self.assertIn("start: call (WS? call)*\n", unarmed.grammar)

    def test_required_lists_all_functions(self):
        g = self.build()
        for name in ("calculator", "get_weather", "free_form"):
            self.assertIn(f'"{name}"', g)
        self.assertIn("(WS? call)*", g)

    def test_is_lark_not_gbnf(self):
        # The backend switches to GBNF when it sees the GBNF rule operator
        self.assertNotIn("::=", self.build())
        self.assertIsNone(self.build(tools=[tool("a::=b", ["x"])]))

    def test_unparseable_names_dropped(self):
        self.assertIsNone(self.build(tools=[tool("has space")]))
        g = self.build(tools=[tool("has space"), tool("ok")])
        self.assertIn('"ok"', g)
        self.assertNotIn("has space", g)


def _synthetic_tokenizer():
    corpus = [
        "What is 7 times 8? The answer is 56.",
        "<function=calculator>\n<parameter=expression>\n7 * 8\n</parameter>\n</function>\n",
        "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n",
        "Let me think about which tool to call. a < b and c > d",
    ] * 4
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
    )
    tok.train_from_iterator(corpus, trainer)
    tok.add_tokens(
        [AddedToken(t, special=False, normalized=False) for t in (TOOL_START, TOOL_END, THINK_START, THINK_END)]
    )
    tok.add_special_tokens([AddedToken(EOS_TEXT, special=True, normalized=False)])
    return tok


class _GrammarHarness:
    """Runs a tool call grammar through llguidance over a given HF tokenizer."""

    def __init__(self, hf_tokenizer, eos_ids):
        self.tok = hf_tokenizer
        self.eos = eos_ids[0]
        self.ll = LLTokenizer(hf_tokenizer.to_str(), eos_token=eos_ids)

    def grammar(self, choice="required", tools=TOOLS, armed=True, content_max=1024, **kwargs):
        data = request(tools=tools, tool_choice=choice, **kwargs)
        with patch.dict(os.environ, {CONTENT_MAX_TOKENS_ENV: str(content_max)}):
            forcing = prepare_tool_choice_forcing(
                data, "qwen3_coder", self.tok.token_to_id, "test", armed_by_reasoning_end=armed
            )
        grammar = grammar_from("lark", forcing.grammar)
        err = LLMatcher.validate_grammar(grammar, self.ll)
        assert not err, err
        return grammar

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def allowed(self, matcher, token_id):
        bm = np.zeros(((self.ll.vocab_size + 31) // 32,), dtype=np.int32)
        matcher.unsafe_compute_mask_ptr(bm.ctypes.data, bm.size * bm.itemsize)
        return bool((int(bm.view(np.uint32)[token_id >> 5]) >> (token_id & 31)) & 1)

    def feed(self, grammar, text):
        """Consume text; returns (matcher, number of tokens accepted before a rejection)."""
        m = LLMatcher(self.ll, grammar, log_level=0)
        ids = self.ids(text)
        for i, t in enumerate(ids):
            if not self.allowed(m, t) or not m.consume_token(t):
                return m, i
        return m, len(ids)

    def accepts(self, grammar, text):
        m, n = self.feed(grammar, text)
        return n == len(self.ids(text)) and not m.is_error()

    def accepts_then_eos(self, grammar, text):
        m, n = self.feed(grammar, text)
        return n == len(self.ids(text)) and self.allowed(m, self.eos)


class _GrammarBehaviour:
    """Shared grammar checks; subclasses provide self.h (a _GrammarHarness)."""

    def test_tags_are_single_tokens(self):
        # TOOL_START/TOOL_END are referenced by id in the grammar; THINK_END is
        # the filter trigger, which the backend drops (grammar active from the
        # first token, no reasoning) if it is not a single token
        for tag in (TOOL_START, TOOL_END, THINK_END):
            self.assertEqual(len(self.h.ids(tag)), 1, tag)
            self.assertIsNotNone(self.h.tok.token_to_id(tag), tag)

    def test_complete_call_accepted_then_eos(self):
        g = self.h.grammar()
        text = "\n\n" + call_text("calculator", {"expression": "7 * 8"})
        self.assertTrue(self.h.accepts_then_eos(g, text))

    def test_eos_not_allowed_before_a_call(self):
        g = self.h.grammar()
        m = LLMatcher(self.h.ll, g, log_level=0)
        self.assertFalse(self.h.allowed(m, self.h.eos))
        self.assertTrue(self.h.allowed(m, self.h.tok.token_to_id(TOOL_START)))
        m, _ = self.h.feed(g, TOOL_START + "\n<function=calculator>\n")
        self.assertFalse(self.h.allowed(m, self.h.eos))

    def test_content_may_end_the_turn(self):
        # Ending the content is allowed (the server continues the turn with a
        # call-only job); masking it made the model loop in its content
        g = self.h.grammar()
        m, n = self.h.feed(g, "\n\n7 x 8 = 56.")
        self.assertEqual(n, len(self.h.ids("\n\n7 x 8 = 56.")))
        self.assertTrue(self.h.allowed(m, self.h.eos))
        self.assertTrue(self.h.allowed(m, self.h.tok.token_to_id(TOOL_START)))

    def test_eos_at_once_after_reasoning_is_masked(self):
        g = self.h.grammar()
        m, _ = self.h.feed(g, "\n\n")
        self.assertFalse(self.h.allowed(m, self.h.eos))

    def test_call_grammar_starts_with_the_call(self):
        data = request(tools=TOOLS, tool_choice="required")
        forcing = prepare_tool_choice_forcing(data, "qwen3_coder", self.h.tok.token_to_id, "t")
        g = grammar_from("lark", forcing.call_grammar)
        self.assertFalse(LLMatcher.validate_grammar(g, self.h.ll))
        m = LLMatcher(self.h.ll, g, log_level=0)
        self.assertFalse(self.h.allowed(m, self.h.eos))
        self.assertTrue(self.h.allowed(m, self.h.tok.token_to_id(TOOL_START)))
        _, n = self.h.feed(g, "Sure. " + call_text("calculator"))
        self.assertEqual(n, 0)
        self.assertTrue(self.h.accepts_then_eos(g, call_text("calculator", {"expression": "7 * 8"})))

    def test_content_then_call_accepted(self):
        g = self.h.grammar()
        text = "\n\n7 x 8 = **56**. Checking with the tool.\n\n" + call_text("calculator", {"expression": "7 * 8"})
        self.assertTrue(self.h.accepts_then_eos(g, text))

    def test_content_after_call_rejected(self):
        g = self.h.grammar()
        self.assertFalse(self.h.accepts(g, call_text("calculator", {"expression": "1"}) + "\nThe answer is 56."))

    def test_content_is_bounded(self):
        g = self.h.grammar(content_max=3)
        m, _ = self.h.feed(g, "\n\n" + "word " * 3)
        self.assertTrue(self.h.allowed(m, self.h.tok.token_to_id(TOOL_START)))
        self.assertTrue(self.h.allowed(m, self.h.eos))
        self.assertFalse(self.h.accepts(g, "\n\n" + "word " * 12))

    def test_content_disabled(self):
        g = self.h.grammar(content_max=0)
        _, n = self.h.feed(g, "7 x 8 = 56")
        self.assertEqual(n, 0)

    def test_second_reasoning_block_rejected(self):
        g = self.h.grammar()
        _, n = self.h.feed(g, THINK_START + "more")
        self.assertEqual(n, 0)

    def test_unknown_function_rejected(self):
        g = self.h.grammar()
        self.assertFalse(self.h.accepts(g, call_text("rm_rf")))

    def test_undeclared_parameter_rejected(self):
        g = self.h.grammar()
        self.assertFalse(self.h.accepts(g, call_text("calculator", {"nope": "1"})))

    def test_free_parameter_names_when_schema_lists_none(self):
        g = self.h.grammar()
        self.assertTrue(self.h.accepts_then_eos(g, call_text("free_form", {"anything": "x"})))

    def test_values_are_free_text(self):
        g = self.h.grammar()
        value = 'a < b && c > d, "quoted", {"json": [1, 2]}\nsecond line'
        self.assertTrue(self.h.accepts_then_eos(g, call_text("get_weather", {"city": value})))

    def test_zero_parameters(self):
        g = self.h.grammar()
        self.assertTrue(self.h.accepts_then_eos(g, call_text("calculator")))

    def test_parallel_calls(self):
        g = self.h.grammar()
        two = call_text("calculator", {"expression": "1"}) + "\n" + call_text("get_weather", {"city": "x"})
        self.assertTrue(self.h.accepts_then_eos(g, two))

    def test_single_call_when_parallel_disabled(self):
        g = self.h.grammar(parallel_tool_calls=False)
        one = call_text("calculator", {"expression": "1"})
        self.assertTrue(self.h.accepts_then_eos(g, one))
        m, _ = self.h.feed(g, one + "\n")
        self.assertFalse(self.h.allowed(m, self.h.tok.token_to_id(TOOL_START)))

    def test_named_function_only(self):
        g = self.h.grammar(choice={"type": "function", "function": {"name": "get_weather"}})
        self.assertTrue(self.h.accepts_then_eos(g, call_text("get_weather", {"city": "Paris"})))
        self.assertFalse(self.h.accepts(g, call_text("calculator", {"expression": "1"})))
        m, _ = self.h.feed(g, call_text("get_weather", {"city": "Paris"}) + "\n")
        self.assertFalse(self.h.allowed(m, self.h.tok.token_to_id(TOOL_START)))

    def test_unarmed_grammar_has_no_outer_whitespace(self):
        g = self.h.grammar(armed=False)
        one = call_text("calculator", {"expression": "1"})
        self.assertTrue(self.h.accepts_then_eos(g, one))
        _, n = self.h.feed(g, "\n\n" + call_text("calculator"))
        self.assertEqual(n, 0)
        self.assertFalse(self.h.accepts_then_eos(g, one + "\n"))

    def test_whitespace_is_bounded(self):
        g = self.h.grammar()
        self.assertTrue(self.h.accepts(g, " " * 8))
        self.assertFalse(self.h.accepts(g, " " * 9))

    def test_parser_reads_back_the_forced_call(self):
        from endpoints.OAI.utils.tools import parse_toolcalls

        text = call_text("get_weather", {"city": "Paris", "unit": "C"})
        self.assertTrue(self.h.accepts_then_eos(self.h.grammar(), text))
        calls = parse_toolcalls(text, "qwen3_coder")
        self.assertEqual([c.function.name for c in calls], ["get_weather"])
        self.assertEqual(calls[0].function.arguments, '{"city": "Paris", "unit": "C"}')


class LLGuidanceAvailableTest(unittest.TestCase):
    """The image build sets TABBY_TOOLCHOICE_REQUIRE_LLG: without llguidance the
    backend silently drops the grammar filter and required degrades to auto."""

    @unittest.skipUnless(os.environ.get("TABBY_TOOLCHOICE_REQUIRE_LLG"), "not required here")
    def test_llguidance_available(self):
        self.assertTrue(HAVE_LLG, "llguidance, tokenizers and numpy must be importable")


@unittest.skipUnless(HAVE_LLG, "llguidance/tokenizers not installed")
class SyntheticTokenizerGrammarTests(_GrammarBehaviour, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = _synthetic_tokenizer()
        cls.h = _GrammarHarness(tok, [tok.token_to_id(EOS_TEXT)])


@unittest.skipUnless(
    HAVE_LLG and os.environ.get("TABBY_TOOLCHOICE_TOKENIZER"),
    "set TABBY_TOOLCHOICE_TOKENIZER to a tokenizer.json to run against a real vocabulary",
)
class RealTokenizerGrammarTests(_GrammarBehaviour, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = Tokenizer.from_file(os.environ["TABBY_TOOLCHOICE_TOKENIZER"])
        eos = [tok.token_to_id(EOS_TEXT)]
        cls.h = _GrammarHarness(tok, eos)


@unittest.skipUnless(
    os.environ.get("TABBY_TOOLCHOICE_CHAT_TEMPLATE"),
    "set TABBY_TOOLCHOICE_CHAT_TEMPLATE to the model's chat template to check the round trip",
)
class ChatTemplateRoundTripTests(unittest.IsolatedAsyncioTestCase):
    """A turn carrying content and a tool call renders back as content, then the call."""

    async def test_content_and_call_render_back(self):
        from common.templating import PromptTemplate

        with open(os.environ["TABBY_TOOLCHOICE_CHAT_TEMPLATE"]) as f:
            template = PromptTemplate("model", f.read())
        # Same shapes format_messages_with_template produces (arguments as a dict)
        messages = [
            {"role": "user", "content": "What is 7 times 8?"},
            {
                "role": "assistant",
                "content": "\n\n7 x 8 = **56**. Let me confirm.\n\n",
                "reasoning_content": "Easy, but a call is required.",
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "calculator", "arguments": {"expression": "7 * 8"}}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "56"},
        ]
        prompt = await template.render(
            {"messages": messages, "tools": TOOLS, "add_generation_prompt": True}
        )
        turn = prompt[prompt.index("Let me confirm.") - 40:]
        content_at = turn.index("7 x 8 = **56**. Let me confirm.")
        call_at = turn.index(TOOL_START)
        self.assertLess(content_at, call_at)
        self.assertIn("<function=calculator>", turn[call_at:])
        self.assertIn("7 * 8", turn[call_at:])
        # The rendered turn is what the grammar admits: content, whitespace, call
        between = turn[content_at + len("7 x 8 = **56**. Let me confirm."):call_at]
        self.assertEqual(between.strip(), "")


if __name__ == "__main__":
    unittest.main()
