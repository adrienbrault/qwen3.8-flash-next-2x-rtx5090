"""
Enforcement of tool_choice "required" and named tool choice for chat completions.

"auto", "none" and an absent tool_choice never reach this module's forcing
path: the chat completion collector only calls in here when
resolve_tool_choice_forcing() returns a forcing spec, so those requests take
exactly the code path they took before tool_choice enforcement existed.

Mechanism: the tool call is enforced by a Lark grammar (llguidance filter)
passed to the backend through the request's grammar_string. When generation
starts inside a reasoning block, the filter is armed with the reasoning end
token as its trigger (the existing filter_trigger plumbing), so reasoning is
unconstrained; after the reasoning end the model either calls at once, or
writes assistant content (bounded; OpenAI allows content alongside
tool_calls) and then calls or ends the content. Nothing but further calls may
follow a call. A turn that ends in content is continued by a call-only job
(the end of the content is the transition to the call, instead of a masked
EOS that would make the model loop in its content). The constraint is applied in the engine at
sampling time, so it holds for every verified token, including speculative
(MTP) draft verification.

A model that never ends its reasoning is bounded by a phase-2 continuation:
once the reasoning phase exceeds a cap (the request's reasoning budget, else
TABBY_TOOL_CHOICE_REASONING_CAP tokens), or the model stops inside its
reasoning, the first job is cancelled and a second job continues from
prompt + reasoning + reasoning end, with the grammar active from its first
token.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import AsyncIterator, Callable, List, Optional

from fastapi import HTTPException

from common.logger import xlogger
from common.networking import handle_request_error
from endpoints.OAI.types.tools import NamedToolChoice, ToolSpec
from endpoints.OAI.utils.toolcall_formats import qwen3_coder
from endpoints.OAI.utils.tools import ALL_TOOLCALL_FORMATS, get_toolcall_tags

# Default cap on reasoning tokens before a required/named tool call is forced
# by a phase-2 continuation, when the request resolves no reasoning budget.
# 0 disables the cap (the reasoning phase is then bounded by max_tokens only).
REASONING_CAP_ENV = "TABBY_TOOL_CHOICE_REASONING_CAP"
DEFAULT_REASONING_CAP = 16384

# Assistant content allowed before the forced call, in tokens. A turn may
# carry content and tool_calls together (clients that force a call on every
# turn need the answer to go somewhere); the bound keeps a model that would
# rather answer than call from writing until max_tokens. 0 disallows content.
CONTENT_MAX_TOKENS_ENV = "TABBY_TOOL_CHOICE_CONTENT_MAX_TOKENS"
DEFAULT_CONTENT_MAX_TOKENS = 1024

# Longest whitespace run the grammar accepts at any position. Bounded so a
# model that resists the tool call cannot stall in whitespace until max_tokens.
_MAX_WS = 8

# Names the qwen3_coder parser can read back: `<function=([^>\s]+)...>`
_PARSEABLE_NAME = re.compile(r"[^>\s]+")


@dataclass
class ToolChoiceForcing:
    """What a required/named tool_choice request forces."""

    mode: str  # "required" or "named"
    name: Optional[str]  # function name for named choice
    single_call: bool  # exactly one call (named, or parallel_tool_calls false)
    grammar: Optional[str] = None
    # Call-only grammar for the continuation after a turn that ended in
    # content (None when the main grammar does not admit content)
    call_grammar: Optional[str] = None


def _request_error(message: str) -> HTTPException:
    error_message = handle_request_error(message, exc_info=False).error.message
    return HTTPException(400, error_message)


def _tool_names(tools: Optional[List[ToolSpec]]) -> List[str]:
    return [tool.function.name for tool in tools or []]


def validate_tool_choice(data) -> None:
    """
    Reject tool_choice combinations OpenAI rejects, and ones this server cannot
    honour. Only "required" and named choices are checked; "auto", "none" and
    an absent tool_choice pass through untouched.
    """

    choice = data.tool_choice
    if choice != "required" and not isinstance(choice, NamedToolChoice):
        return

    label = "required" if choice == "required" else "a named function"
    if not data.tools:
        raise _request_error(f"tool_choice is {label}, but no tools were provided.")

    if isinstance(choice, NamedToolChoice):
        name = choice.function.name
        if name not in _tool_names(data.tools):
            raise _request_error(
                f"tool_choice names function '{name}', which is not in the provided tools."
            )

    # The tool call is enforced with a grammar; a second, client-supplied
    # constraint would be intersected with it and could leave no valid token
    response_format = getattr(data, "response_format", None)
    constrained = (
        data.json_schema
        or data.regex_pattern
        or data.grammar_string
        or (response_format is not None and response_format.type in ("json", "json_schema"))
    )
    if constrained:
        raise _request_error(
            f"tool_choice is {label}; it cannot be combined with json_schema, "
            "regex_pattern, grammar_string or a JSON response_format."
        )


def forces_tool_call(data) -> bool:
    """True for tool_choice "required" or a named function."""

    return data.tool_choice == "required" or isinstance(data.tool_choice, NamedToolChoice)


def resolve_tool_choice_forcing(data) -> Optional[ToolChoiceForcing]:
    """Forcing spec for a request, or None for auto/none/absent tool_choice."""

    choice = data.tool_choice
    if isinstance(choice, NamedToolChoice):
        return ToolChoiceForcing(mode="named", name=choice.function.name, single_call=True)
    if choice == "required":
        return ToolChoiceForcing(
            mode="required",
            name=None,
            single_call=data.parallel_tool_calls is False,
        )
    return None


def supports_forcing(tool_format: Optional[str]) -> bool:
    """Tool formats with a grammar builder."""

    return bool(tool_format) and ALL_TOOLCALL_FORMATS.get(tool_format) is qwen3_coder


def _lark_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _tag_literal(tag: str, single_id: Callable[[str], Optional[int]]) -> str:
    """
    Grammar literal for a tool tag. llguidance treats added tokens as special
    (their text form does not match the token), so a tag that is a single
    token must be referenced by ID, or the grammar would force the model to
    spell the tag out of ordinary text tokens.
    """

    token_id = single_id(tag)
    if token_id is not None:
        return f"<[{int(token_id)}]>"
    return _lark_string(tag)


def _param_names(tool: ToolSpec) -> Optional[List[str]]:
    """Declared parameter names, or None when the schema does not list any."""

    properties = (tool.function.parameters or {}).get("properties")
    if not isinstance(properties, dict) or not properties:
        return None
    names = list(properties.keys())
    if not all(isinstance(n, str) and _PARSEABLE_NAME.fullmatch(n) for n in names):
        return None
    return names


def build_qwen3_coder_grammar(
    forcing: ToolChoiceForcing,
    tools: List[ToolSpec],
    tool_start: str,
    tool_end: str,
    single_id: Callable[[str], Optional[int]],
    outer_ws: bool = True,
    content_max_tokens: int = 0,
) -> Optional[str]:
    """
    Lark grammar (llguidance dialect) for one or more qwen3_coder tool calls:
    optional whitespace (only with outer_ws), optional assistant content (up to
    content_max_tokens tokens, starting with a non-whitespace character; only
    when TOOL_START is a single token, since the content lexeme ends at a
    special token), then TOOL_START, `<function=NAME>`, zero or more
    `<parameter=P>VALUE</parameter>`, `</function>`, TOOL_END. NAME is one of
    the provided tools (only the named one for a named choice), P one of that
    tool's declared parameters (any name if the schema lists none), VALUE free
    text. The grammar admits EOS after a complete call, or (with content) at
    the end of the content: the caller continues such a turn with a call-only
    grammar. Masking EOS after content made the model loop in its content
    (R523 round 2: repeated sign-offs until the loop detector stopped it).

    outer_ws (whitespace before the first and after the last call) is for a
    grammar armed by the reasoning end, where the chat template puts whitespace
    between the reasoning block and the call and the stream parser holds
    whitespace after the reasoning end back. A grammar active from the first
    token follows a prompt that already ends the reasoning block and its
    whitespace; outer whitespace there would only stream as a whitespace-only
    content delta.
    """

    tools_by_name = {}
    for tool in tools:
        tools_by_name.setdefault(tool.function.name, tool)

    names = [forcing.name] if forcing.mode == "named" else list(tools_by_name)
    names = [n for n in names if n in tools_by_name and _PARSEABLE_NAME.fullmatch(n)]
    if not names:
        return None

    lead, trail = (" WS? ", " WS?") if outer_ws else (" ", "")
    tool_start_lit = _tag_literal(tool_start, single_id)
    allow_content = content_max_tokens > 0 and tool_start_lit.startswith("<[")
    calls = "call" if forcing.single_call else "call (WS? call)*"
    lines = []
    if allow_content:
        # Content lexeme absorbs the whitespace between content and call
        lines.append(f"start:{lead}content? {calls}{trail} |{lead}content")
    else:
        lines.append(f"start:{lead}{calls}{trail}")

    fn_rules = [f"fn_{i}" for i in range(len(names))]
    lines.append(
        f"call: {tool_start_lit} WS? ({' | '.join(fn_rules)}) "
        f"WS? {_tag_literal(tool_end, single_id)}"
    )
    if allow_content:
        # Free text; `.` never matches a special token, so the lexeme ends at
        # TOOL_START (or any other special token, which the grammar rejects)
        lines.append(
            f"content[max_tokens={int(content_max_tokens)}]: " + r"/(?s:[^ \t\r\n].*)/"
        )

    for i, name in enumerate(names):
        param_names = _param_names(tools_by_name[name])
        if param_names is None:
            pname = "PNAME_ANY"
        else:
            pname = f"pname_{i}"
            lines.append(f"{pname}: " + " | ".join(_lark_string(p) for p in param_names))
        lines.append(
            f'fn_{i}: "<function=" {_lark_string(name)} ">" WS? '
            f'(param_{i} WS?)* "</function>"'
        )
        lines.append(f'param_{i}: "<parameter=" {pname} ">" pvalue')

    # A lazy lexeme ends at the first closing tag, so values are free text
    # that cannot contain the closing parameter tag (the parser splits there)
    lines.append(r"pvalue[lazy]: /(?s:.*)<\/parameter>/")
    lines.append(r"PNAME_ANY: /[^>\s]+/")
    lines.append(r"WS: /[ \t\r\n]{1," + str(_MAX_WS) + r"}/")

    grammar = "\n".join(lines) + "\n"

    # The backend picks GBNF over Lark when it sees the GBNF rule operator
    if "::=" in grammar:
        return None
    return grammar


def prepare_tool_choice_forcing(
    data,
    tool_format: Optional[str],
    single_id: Callable[[str], Optional[int]],
    label: str,
    armed_by_reasoning_end: bool = True,
) -> Optional[ToolChoiceForcing]:
    """
    Forcing spec with its grammar for a required/named request, or None when
    the request does not force a tool call or the model's tool format has no
    grammar builder (the request then behaves as tool_choice "auto").
    """

    forcing = resolve_tool_choice_forcing(data)
    if forcing is None:
        return None

    tool_start, tool_end = get_toolcall_tags(tool_format)
    if not supports_forcing(tool_format) or not tool_start or not tool_end:
        xlogger.warning(
            f"{label}: tool_choice {forcing.mode} is not enforced for tool format "
            f"'{tool_format}'; the request is served as tool_choice auto."
        )
        return None

    content_max_tokens = resolve_content_max_tokens()
    forcing.grammar = build_qwen3_coder_grammar(
        forcing,
        data.tools or [],
        tool_start,
        tool_end,
        single_id,
        outer_ws=armed_by_reasoning_end,
        content_max_tokens=content_max_tokens,
    )
    if forcing.grammar is not None and "content" in forcing.grammar.split("\n", 1)[0]:
        # The continuation prompt ends with the content and the template's
        # separator, so the call starts at once
        forcing.call_grammar = build_qwen3_coder_grammar(
            forcing, data.tools or [], tool_start, tool_end, single_id, outer_ws=False
        )
    if forcing.grammar is None:
        xlogger.warning(
            f"{label}: no tool call grammar could be built for tool_choice "
            f"{forcing.mode} (tool names the parser cannot read back); the request "
            "is served as tool_choice auto."
        )
        return None

    return forcing


def resolve_content_max_tokens() -> int:
    """Content tokens allowed before the forced call (0 = none)."""

    try:
        value = int(os.environ.get(CONTENT_MAX_TOKENS_ENV, DEFAULT_CONTENT_MAX_TOKENS))
    except ValueError:
        value = DEFAULT_CONTENT_MAX_TOKENS
    return max(value, 0)


def resolve_reasoning_cap(budget: Optional[int]) -> Optional[int]:
    """Reasoning tokens allowed before a forced call: the budget, else the default cap."""

    if budget is not None:
        return budget
    try:
        cap = int(os.environ.get(REASONING_CAP_ENV, DEFAULT_REASONING_CAP))
    except ValueError:
        cap = DEFAULT_REASONING_CAP
    return cap if cap > 0 else None


class ToolChoiceNotHonoured(RuntimeError):
    """A required/named request finished on its own without the forced call."""


def check_forced_tool_calls(forcing: ToolChoiceForcing, finish: dict, calls: list, label: str):
    """
    Fail the request (503 through the existing error path) when a forced
    request stopped on its own without the call it was forced to make, e.g.
    because the backend could not compile the grammar and dropped the filter.
    Limits the client set (max_tokens, stop strings) end the request as usual;
    every other end (stop token, filter end, the loop detector, ...) without
    the call is a failure. R523 round 2: a content loop ended by the loop
    detector (eos_reason loop_detected) slipped past a check that only looked
    at stop_token/end_filter.
    """

    if finish.get("eos_reason") in ("max_new_tokens", "stop_string"):
        return
    names = [c["function"]["name"] for c in calls or []]
    if names and (forcing.mode != "named" or all(n == forcing.name for n in names)):
        return
    message = (
        f"{label}: tool_choice {forcing.mode} was not honoured: "
        f"generation ended with tool calls {names}"
        + (f", expected '{forcing.name}'" if forcing.mode == "named" else "")
        + ". Check the log for a grammar that failed to compile."
    )
    xlogger.error(message)
    raise ToolChoiceNotHonoured(message)


class _TagWatch:
    """Detects tags in a stream of text chunks, including tags split across chunks."""

    def __init__(self, *tags: str):
        self.tags = [t for t in tags if t]
        self.keep = max((len(t) - 1 for t in self.tags), default=0)
        self.tail = ""
        self.seen = set()

    def feed(self, text: str):
        window = self.tail + text
        self.seen.update(t for t in self.tags if t in window)
        self.tail = window[-self.keep :] if self.keep else ""


async def forced_tool_generation(
    mc,
    request_id: str,
    prompt: str,
    params,
    disconnect_handler,
    mm_embeddings,
    reasoning_end: Optional[str],
    tool_start: str,
    reasoning_cap: Optional[int],
    reasoning_message: str,
    call_grammar: Optional[str],
    label: str,
) -> AsyncIterator[dict]:
    """
    Generation stream for a required/named request. params carries the tool
    call grammar; reasoning_end is the filter trigger when generation starts
    inside a reasoning block, else None (grammar active from the first token).

    The first job's chunks pass through unchanged, so a turn that calls within
    the reasoning cap is a single job, identical in shape to any other
    constrained request. Two ends start a continuation job instead of ending
    the request:

    - The reasoning phase reaches reasoning_cap tokens (the job is cancelled
      and whatever it still had queued is discarded, never forwarded), or the
      job stops on a stop token without ever ending its reasoning: continue
      from prompt + forwarded text + reasoning_message + reasoning end with the
      same grammar, active from its first token. The injected text is yielded
      as a regular chunk, as the reasoning budget injection is, so downstream
      parsing sees an ordinary end of reasoning.
    - The job ends after its reasoning without a tool call start (the grammar
      admits ending the content): continue from prompt + forwarded text with
      the content's trailing whitespace replaced by the template's separator,
      under call_grammar, which starts with the call.

    Only the final job's finish chunk is yielded, with usage covering every job.
    """

    # Pristine copy for continuations (the backend extends params.stop in place)
    base_params = params.model_copy(deep=True)
    job_prompt = prompt
    job_params = params
    job_trigger = reasoning_end
    job_id = request_id
    job_label = label
    phase = 1

    forwarded_text = ""
    forwarded_tokens = 0
    reasoning_tokens = 0
    prompt_tokens = None
    # Reasoning is open until its end tag, or a tool call start (a call made
    # inside the reasoning, or a filter that could not be armed by the
    # reasoning end and constrains from the first token), shows up
    reasoning_watch = _TagWatch(reasoning_end, tool_start) if reasoning_end else None
    tool_watch = _TagWatch(tool_start)
    reasoning_ended_by_injection = False
    continued_to_call = False
    last_step = None

    def reasoning_open():
        return reasoning_watch is not None and not reasoning_watch.seen and not reasoning_ended_by_injection

    while True:
        next_params = base_params.model_copy(deep=True)
        generation = mc.stream_generate(
            job_id,
            job_prompt,
            job_params,
            disconnect_handler,
            mm_embeddings,
            filter_trigger=job_trigger,
            label=job_label,
        )

        step = None
        # Completion tokens of the earlier jobs (the finish chunk of this one
        # counts its own)
        earlier_tokens = forwarded_tokens
        async for chunk in generation:
            if prompt_tokens is None:
                prompt_tokens = chunk.get("prompt_tokens")
            if "finish_reason" in chunk:
                eos_reason = chunk.get("eos_reason")
                if reasoning_open() and eos_reason == "stop_token":
                    step = ("end_reasoning", "the model stopped inside its reasoning")
                elif (
                    not reasoning_open()
                    and not tool_watch.seen
                    and call_grammar is not None
                    and not continued_to_call
                    and eos_reason in ("stop_token", "end_filter")
                ):
                    step = ("call", "the turn ended in content without a tool call")
                else:
                    if phase > 1:
                        _report_continued_usage(chunk, prompt_tokens, earlier_tokens, last_step)
                    yield chunk
                    return
                break

            text = chunk.get("text", "")
            token_ids = chunk.get("token_ids") or []
            forwarded_text += text
            forwarded_tokens += len(token_ids)
            tool_watch.feed(text)
            if reasoning_open():
                reasoning_watch.feed(text)
                if reasoning_open():
                    reasoning_tokens += len(token_ids)
            yield chunk

            if reasoning_open() and reasoning_cap is not None and reasoning_tokens >= reasoning_cap:
                step = ("end_reasoning", f"reasoning reached {reasoning_tokens} tokens (cap {reasoning_cap})")
                job = mc.active_job_ids.get(job_id)
                if job is not None:
                    await job.cancel()
                # Drain the cancelled job; nothing it still had queued is forwarded
                async for _ in generation:
                    pass
                break

        if step is None or getattr(disconnect_handler, "disconnected", False):
            # The job ended without a finish chunk (cancelled from elsewhere),
            # or the client is gone
            return

        if next_params.max_tokens and next_params.max_tokens > 0:
            remaining = next_params.max_tokens - forwarded_tokens
            if remaining <= 0:
                yield {
                    "finish_reason": "length",
                    "eos_reason": "max_new_tokens",
                    "prompt_tokens": prompt_tokens or 0,
                    "gen_tokens": forwarded_tokens,
                }
                return
            next_params.max_tokens = remaining

        kind, reason = step
        phase += 1
        last_step = {"trigger": reason, "phase": phase}
        if kind == "end_reasoning":
            injection = reasoning_message + reasoning_end
            xlogger.info(
                f"{label}: tool_choice forcing a tool call: {reason}; "
                "continuing after the reasoning end with the tool call grammar active.",
                {"request_id": request_id, "reasoning_tokens": reasoning_tokens},
            )
            yield {"text": injection, "token_ids": [], "prompt_tokens": prompt_tokens}
            forwarded_text += injection
            reasoning_ended_by_injection = True
            job_prompt = prompt + forwarded_text
        else:
            xlogger.info(
                f"{label}: tool_choice forcing a tool call: {reason}; "
                "continuing after the content with a call-only grammar.",
                {"request_id": request_id},
            )
            next_params.grammar_string = call_grammar
            continued_to_call = True
            # The template renders a content + call turn as content, a blank
            # line, then the call; the whitespace already streamed as content
            # is not repeated
            job_prompt = prompt + forwarded_text.rstrip() + "\n\n"
        job_params = next_params
        job_trigger = None
        job_id = f"{request_id}-tc{phase}"
        job_label = f"{label} (tool_choice phase {phase})"


def _report_continued_usage(chunk: dict, prompt_tokens, earlier_tokens: int, step_info: dict):
    """
    Usage of a continued request against the client's prompt: a continuation's
    prompt includes earlier output, which is completion tokens. The last job's
    own numbers stay in the chunk under tool_choice_continuation (and in the
    server log).
    """

    job_prompt_tokens = chunk.get("prompt_tokens") or 0
    client_prompt_tokens = prompt_tokens if prompt_tokens is not None else job_prompt_tokens
    chunk["tool_choice_continuation"] = {
        **step_info,
        "prompt_tokens": job_prompt_tokens,
        "cached_tokens": chunk.get("cached_tokens"),
        "gen_tokens": chunk.get("gen_tokens"),
    }
    chunk["prompt_tokens"] = client_prompt_tokens
    chunk["cached_tokens"] = min(chunk.get("cached_tokens") or 0, client_prompt_tokens)
    chunk["gen_tokens"] = earlier_tokens + (chunk.get("gen_tokens") or 0)
