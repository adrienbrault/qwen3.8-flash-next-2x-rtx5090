# tool-choice-r1: implementation status

Round 1 of making the Flash-Next TabbyAPI daily enforce `tool_choice: "required"` and named tool choice. Tags are written as THINK_START, THINK_END, TOOL_START and TOOL_END throughout. Line references prefixed `ref:` point into `ref/served-src/tabbyapi` (the served `/app`), `exl3:` into `ref/served-src/exllamav3`, and `overlay:` into `out/tool-choice-r1/overlay`.

## 1. How the served code handles tools today

### Request and validation
- `ref:endpoints/OAI/types/chat_completion.py:157` accepts `tool_choice: "none" | "auto" | "required" | NamedToolChoice` (default `None`). `:158` has `parallel_tool_calls` (default `True`). Nothing validates the value against `tools`.
- `ref:endpoints/OAI/router.py:114-165` is the `/v1/chat/completions` route. It calls `apply_chat_template` (`:139`). Only after that does it map `response_format` onto `data.json_schema` (`:142-148`).
- `ref:endpoints/OAI/utils/chat_completion.py:478-532` (`apply_chat_template`) passes `tools` and `functions` to the Jinja template as template vars (`:487-497`). `tool_choice` never reaches the template.

### Generation
- `ref:endpoints/OAI/utils/chat_completion.py:622` (`_chat_stream_collector`) serves both streaming and non-streaming requests, one collector per choice. `params` is a deep copy for each choice (`:848`, `:950`).
- Parser selection is at `:653-678`. Harmony and Glimmer use their own channel parsers. Every other model uses `TagStreamParser` with the reasoning tags from the model config and the tool tags from `get_toolcall_tags(tool_format)` (`ref:endpoints/OAI/utils/tools.py:59`).
- **This line is the whole of today's `tool_choice` handling:** `use_tool = params.tool_choice != "none" and bool(t_tool_start)` at `:667`. `"none"` turns off tool-tag parsing. `"required"` and a named function behave the same as `"auto"`.
- The reasoning budget is set up at `:684-700` and injected at `:744-753`. When the budget runs out, `mc.constrain_generation_output` forces the budget message plus THINK_END into the output. It is disabled when the request carries `json_schema`, `regex_pattern` or `grammar_string` (`:692-699`), because injecting output permanently disables a job's filters.
- The backend call is at `:706-716`. It already passes `filter_trigger=THINK_END` whenever generation starts inside a reasoning block, so a client-supplied constraint only takes effect after the reasoning ends.

### Detecting and parsing tool calls
- `ref:endpoints/OAI/utils/stream_parser.py:35` (`TagStreamParser`) splits the text into reasoning, content and tool channels, including tags split across chunks. After THINK_END it holds whitespace until real content arrives, and drops it if none does (`:163`). With `tool_calls_in_reasoning` (the default), tool tags inside reasoning also open the tool channel (`:166`).
- Tool text is accumulated and parsed once, at finish. For streaming this happens at `chat_completion.py:782-786` (the last chunk carries every call and sets `finish_reason: tool_calls`). For non-streaming it is at `:794-803`.
- Flash-Next uses the `qwen3_coder` format, which covers `qwen3_coder`, `qwen3_5`, `step3_5` and `step3_7` (`tools.py:21-45`). Its tags are TOOL_START and TOOL_END (`toolcall_formats/qwen3_coder.py:28-29`). The body is `<function=NAME>` followed by `<parameter=P>VALUE</parameter>` elements and `</function>`, parsed with regexes (`:31-33`). Values are JSON-coerced when possible (`toolcall_formats/common.py`).

### Constrained decoding that already exists
- `ref:common/sampling.py:259-270` defines the `json_schema`, `regex_pattern` and `grammar_string` sampler fields.
- `ref:backends/exllamav3/model.py:1439-1461` builds one exllamav3 filter per field and resolves `filter_trigger` to a single token id (`:1441`). `ref:backends/exllamav3/grammar.py:76-106` wraps `LLGuidanceFilter`. It picks GBNF when the grammar contains `::=` and Lark otherwise (`:88`). If the grammar fails to compile, it logs the error and skips the filter (`:96-106`).
- `exl3:generator/filter/filter.py:50-68`: a filter with a trigger stays inactive until the trigger token is sampled. It then resets and constrains every following token. On completion it ends the job when `eos_after_completed` is set, which Tabby always sets. `exl3:generator/job.py:1563-1572` arms filters when a job starts.
- `exl3:generator/filter/llguidance.py` is the llguidance matcher. `create_ll_tokenizer` is cached per process (`:17`) and costs about 0.8 s on a 248k vocabulary. Masks are packed bitmasks.
- MTP / speculative decoding: masks for active filters are launched before the forward pass and applied at sampling (`exl3:generator/generator.py:1073-1081`, `:1110-1124`). Draft verification consumes positions one at a time. After each accepted draft token it advances the filters and rebuilds the mask for the next position (`:1328-1347`). The filter is fed every sampled token in `receive_sample` (`exl3:generator/job.py:629-631`). The constraint therefore applies to every verified token, and a trigger accepted in the middle of a window takes effect at the next position of the same window. The opt-in batched greedy verify path (`EXL3_DECODE_OVERLAP=1`) skips any job that has filters (`exl3:generator/draft_overlap.py:26-28`).
- Output injection: `constrain_output_now` (`exl3:generator/job.py:497-543`) forces tokens and permanently suspends filters (`:531`). It lands late: tokens already sampled and queued for the consumer come before it (`exl3:generator/async_generator.py:167-173`).

## 2. Mechanism chosen

**(b) A grammar filter armed by THINK_END, plus a bounded phase-2 continuation (a) as the fallback.**

### Common case: one job
For a `required` or named request on a `qwen3_coder`-family model, the collector builds a Lark grammar and puts it in the per-choice `params.grammar_string`. The existing plumbing then arms it with `filter_trigger = THINK_END` (`overlay:endpoints/OAI/utils/chat_completion.py:713-733`, `:739-764`). This gives:
- Reasoning is free, and the grammar activates on the THINK_END token.
- After THINK_END the model may write up to 8 whitespace characters, then optional assistant content (round 2, see below), then TOOL_START. It then writes `<function=` with one of the provided tool names (only the named tool for a named choice), then zero or more `<parameter=P>` elements. `P` is limited to the tool's declared `properties` (any name if the schema lists none). The value is free text ending at the first `</parameter>`. The call closes with `</function>` and TOOL_END.
- EOS is masked until at least one call is complete. `parallel_tool_calls: false` and named choice allow exactly one call. `required` allows one or more.
- The grammar builder is `build_qwen3_coder_grammar` (`overlay:endpoints/OAI/utils/tool_choice.py:176-261`).
- There is no extra prefill, no extra tokens and no second job.

### Fallback: phase-2 continuation
`forced_tool_generation` (`overlay:tool_choice.py:373-510`) handles a model that never ends its reasoning. It triggers when either of these happens:
- The reasoning phase reaches a cap: the resolved reasoning budget, else `TABBY_TOOL_CHOICE_REASONING_CAP`, default 16384, where 0 disables it.
- The model stops on EOS inside its reasoning.

It then cancels phase 1 (`AsyncJob.cancel`; a second cancel at request cleanup is a no-op, `exl3:generator/generator.py:462-485`) and discards anything phase 1 still had queued. It then starts phase 2 from `prompt + forwarded text + budget message + THINK_END`, with the grammar active from the first token. The injected text is yielded as an ordinary chunk, the same way the budget injection is, so the parser sees a normal end of reasoning.

Only the final phase's finish chunk reaches the client. Usage reports the client's prompt tokens and the completion tokens of both phases. Phase 2 receives `max_tokens` minus what phase 1 produced. A `length` stop or a client stop string in phase 1 is passed through untouched.

The collector's reasoning-budget injection is turned off for forced requests, because it would disable the filter. The wrapper enforces the budget as its cap instead.

### Why this mechanism and not the alternatives
- **(c) Prefix forcing / output injection.** Rejected. The injection lands late: tokens the engine sampled after THINK_END can already be queued, so part of a content answer would reach the client before TOOL_START. It also suspends all filters, so the call body is unconstrained and may not parse. It cannot force a specific function name either.
- **(a) Phase-2 continuation for every request.** Rejected as the main path. Every required request would pay a second job and a re-prefill of its reasoning. On Flash-Next's hybrid GDN layers that means restoring recurrent state from the nearest stash and replaying from there. The client would see two jobs' worth of timing, and phase 2 would still need a grammar to guarantee a parseable call. It remains the fallback for the only case (b) cannot bound.
- **(b)** reuses the constraint path that `json_schema` already takes with reasoning models (`chat_completion.py:712-714`, `model.py:1439-1461`). The engine applies the constraint synchronously at sampling time, on the verified token. It is per job, so the 4 slots are independent. It costs one grammar compile per request (1.6 ms measured locally for 12 tools) and a mask per constrained token (median 1.7 µs, p95 10 µs, first mask 2.5 ms, measured with llguidance 1.8.0 on the Qwen3.6 vocabulary on an M-series Mac).

### Tag literals must be token ids
llguidance treats added tokens as special tokens: their text form does not match the token. The Qwen tags are `special: false` added tokens, but llguidance still does this. A grammar that writes TOOL_START as a text string rejects the TOOL_START token and would force the model to spell the tag out of ordinary text tokens. This was observed in the prototype and is guarded by a test. The grammar therefore references single-token tags by id (`<[id]>`), resolved through the container tokenizer's `single_id` (`overlay:tool_choice.py:150-161`).

### Outer whitespace
Whitespace before the first call and after the last call is allowed only when the grammar is armed by THINK_END. The template puts whitespace there, and after THINK_END the parser holds whitespace back. With thinking off, the grammar is active from token 0 and the prompt already ends with THINK_END and its whitespace. Outer whitespace there would stream as a whitespace-only content delta, which the local end-to-end probe caught.

### A forced request that ends without its call fails
`check_forced_tool_calls` (`overlay:tool_choice.py:331-355`) runs at the final finish chunk, both streaming and non-streaming (`overlay:chat_completion.py:835-839`, `:855-856`). It raises `ToolChoiceNotHonoured` in two cases:
- a forced request stopped on its own (`eos_reason` of `stop_token` or `end_filter`) without a parsed call
- a named request called a different function

The existing error path turns this into a 503 for non-streaming requests and an SSE error for streaming ones (`ref:chat_completion.py:915-917`, `:989-996`). It logs the reason. The case it guards is the backend dropping a grammar that failed to compile (`ref:backends/exllamav3/grammar.py:96-106`), which would otherwise silently turn `required` back into `auto`. Limits the client set, `max_tokens` and stop strings, still end the request normally.

## 3. What changed

| File | Change |
|---|---|
| `endpoints/OAI/utils/chat_completion.py` | Import (`overlay:47-54`). `validate_tool_choice(data)` at the top of `apply_chat_template` (`overlay:494-495`). The forcing block after the budget setup (`overlay:713-733`). The backend call now picks between `forced_tool_generation` and the unchanged `mc.stream_generate(...)` call (`overlay:739-764`). The forced-call checks at finish (`overlay:835-839`, `:855-856`). |
| `endpoints/OAI/utils/tool_choice.py` (new) | Validation, mode selection, the grammar builder, the reasoning cap, the forced-call check, and the phase-2 wrapper. |
| `tests/test_tool_choice.py` (new) | Validation, mode selection and grammar text tests. llguidance behaviour tests use a synthetic byte-level tokenizer, or a real `tokenizer.json` via `TABBY_TOOLCHOICE_TOKENIZER`. |
| `tests/test_tool_choice_collector.py` (new) | Collector tests with a fake container whose model honours the grammar. |

### Validation (400, OpenAI-style)
A request is rejected in these cases:
- `required` or named with `tools` missing or empty.
- A named function that is not in `tools`.
- `required` or named combined with `json_schema`, `regex_pattern`, `grammar_string` or a JSON `response_format`. Two grammars on one job are intersected and can leave no valid token.

`auto`, `none` and an absent `tool_choice` are never checked.

The check lives in `apply_chat_template`, so `/v1/apply-template` returns the same 400s for these requests. That is a behaviour change on a second endpoint.

### Formats without a grammar builder
These are `harmony`, `glm4_5`, `mistral` and the other non-`qwen3_coder` formats. `required` and named requests on them are served as `auto` and log a warning. That matches their behaviour today.

## 4. Verified locally (CPU, `.venv`, Python 3.13, llguidance 1.8.0)

- **New tests: 75 pass, 0 skipped.** This is `tests.test_tool_choice` plus `tests.test_tool_choice_collector`. They were run with `TABBY_TOOLCHOICE_REQUIRE_LLG=1` and with `TABBY_TOOLCHOICE_TOKENIZER` pointed at the Qwen3.6-35B-A3B `tokenizer.json` from the local HF cache (a stand-in for Flash-Next's vocabulary), so the real-tokenizer grammar tests ran too. Without those env vars, 17 are skipped as designed.
- **Mutation check.** Switching the tag literals back to text makes 21 grammar tests fail.
- **Existing tests.** The complete served test suite passes on the patched tree: 224 tests, with the 149 existing tests unchanged. The one error is `tests.test_generator_recovery`, which imports `pytest`. It fails the same way on the unpatched tree.
- **`auto`/`none` byte identity, collector level.** `local/auto_none_identity.py` runs the unpatched and patched collectors against the same scripted backend. The matrix covers:
  - `tool_choice` absent, `auto` or `none`
  - streaming and non-streaming
  - thinking on and off
  - a model that answers, calls a tool voluntarily, stops inside its reasoning, reasons until `max_tokens`, or triggers the reasoning-budget injection
  - `n` of 1 and 2

  Across 120 scenarios, zero differ in backend call arguments (prompt, full sampler params dump, filter trigger, label), output injections, serialized stream chunks or final responses. No backend call carried a grammar.
- **`auto`/`none` byte identity, HTTP level.** `local/e2e_fake_server.py` serves each tree's real OAI router with a fake container, and `box_probe.py` probes both over HTTP. The results:
  - identity 24/24 identical between the unpatched and patched tree
  - patched: required 12/12, named 4/4, validation 4/4, cap 2/2, concurrent 8/8
  - unpatched: required 0/12, as expected
- **Packaging.**
  - `served-source.patch` applies with `patch -p1` to a copy of `ref/served-src/tabbyapi`, and the result is identical to the `install.py` result.
  - `install.py` passes on a fresh copy, passes again when re-run (already applied), and refuses a tree whose baseline file has drifted. In that case it writes nothing.
  - Baseline hashes were computed from `ref/served-src/tabbyapi`.

## 5. Only the box can verify
- **The build itself.** The image must build with the legacy builder on `tabbyapi:stack-r4-e3r2`, with baselines matching. The build runs the pure test module with `TABBY_TOOLCHOICE_REQUIRE_LLG=1`, so the image's llguidance must support `[lazy]` rules and `<[id]>` token literals.
- **The Flash-Next tokenizer.** TOOL_START, TOOL_END and THINK_END must be single tokens and the grammar must accept the model's call format. THINK_END is the filter trigger: if it is not a single token, the backend warns and constrains from the first token, so reasoning would be lost. Run the tests with `TABBY_TOOLCHOICE_TOKENIZER` set to the model's `tokenizer.json`.
- **Real model behaviour.** TC-45 should go from 0/12 to 12/12. Every required response should carry a parseable call, and named requests should call the named tool. No regression on the full 69 x 4 tool-eval run.
- **`auto`/`none` byte identity against the unpatched image**, on greedy, c1, on the real model.
- **MTP.** No errors with constrained verification. Draft acceptance during the tool-call phase is a watch item, not a gate.
- **Concurrency.** 4 slots with mixed required and auto requests. If the daily sets `EXL3_DECODE_OVERLAP=1`, one live required request moves the whole batch onto the serial verify path for its lifetime (`draft_overlap.py:28` checks `job.filters`, including a filter that is not yet active). Measure the c4 decode delta.
- **Phase-2 cost.** Check the phase-2 `cached_tokens` against its prompt tokens in the metrics log. Phase 2 should re-prefill at most the reasoning, less whatever pages and recurrent stashes it reuses.

## 6. Risks and known limits
- **Reasoning cap.** A forced request that reasons past 16384 tokens gets its reasoning cut and a call forced (`TABBY_TOOL_CHOICE_REASONING_CAP=0` disables this). For forced requests, the budget ends reasoning by phase 2 instead of by output injection.
- **Retokenization in phase 2.** Phase 2 re-encodes the reasoning text. If the model produced a non-canonical tokenization, prefix-cache reuse stops at that point and the model sees the canonical tokens.
- **Duplicate calls.** A tool call made inside the reasoning (`tool_calls_in_reasoning`) is followed by a second, forced call after THINK_END.
- **Stop inside reasoning.** An EOS inside the reasoning starts phase 2 for forced requests. For `auto` it still ends the request.
- **Usage after phase 2.** `completion_tokens` excludes the injected message and THINK_END tokens. Timing and draft stats cover phase 2 only. `cached_tokens` is clipped to the client prompt, and the phase-2 values are in the server log.
- **First forced request after boot.** It builds the llguidance tokenizer (about 0.8 s) on the event loop, the same way the first `json_schema` request does today.
- **Parameter names.** They are restricted to the declared `properties`, so a schema that lists some properties but allows extra ones cannot receive the extras. Values are not type-checked.
- **Unreadable names.** Tool names the parser cannot read back (whitespace or `>`) are left out of the grammar. If no name survives, the request is served as `auto` with a warning.
- **`response_format` conflict.** `required` or named combined with a JSON `response_format` returns 400. OpenAI would ignore the format when a call is required.
- **Failed grammar compile.** If the grammar failed to compile in the image, the backend would log "Skipping because the grammar couldn't be parsed" and serve the request unconstrained. The request would then fail with 503 or an SSE error, not return a content answer. The build-time test and the TC-45 re-run cover this.

## 7. Round 2 (after the R523 box run)

### What R523 showed
Every TC-45 trial scored 1/2. tool-eval-bench sends `tool_choice="required"` on every turn (`runner/orchestrator.py`), and `_tc45_eval` also needs 56 in the final answer. The round-1 grammar rejected any content after THINK_END, so a forced turn could never carry the answer, and the run looped on calculator calls until it hit `max_turns`.

The identity A/B was confounded. Repeat 2 of each request hit the prefix cache, and so did kinds that share the 12-tool prefix. A was not self-identical (2/12), so the A/B said nothing either way.

### Change 1: content before the forced call
OpenAI allows a message to carry `content` and `tool_calls` together, and so does the grammar now:

```
start: WS? content? call (WS? call)* WS?
```

Without thinking, the rule is `start: content? call (WS? call)*`.

- **What content is.** A lexeme of free text that starts with a non-whitespace character. It ends at TOOL_START: llguidance's `.` never matches a special token, and TOOL_START is referenced by id.
- **Bound.** At most `TABBY_TOOL_CHOICE_CONTENT_MAX_TOKENS` tokens (default 1024; 0 disables content), set with llguidance's `[max_tokens=N]` rule attribute. A model that would rather answer than call cannot write until `max_tokens`.
- **Unchanged.** EOS is still masked until a call is complete, and content after the call is still rejected.
- **Where it applies.** Content is only allowed when TOOL_START is a single token. As spelled-out text, the free-text lexeme could swallow the tag.
- **Code:**
  - `overlay:endpoints/OAI/utils/tool_choice.py:51-52` (knob)
  - `:183` (builder parameter)
  - `:215-218` and `:230-235` (rule)
  - `:296` (wiring)
  - `:309-316` (`resolve_content_max_tokens`)
  - `chat_completion.py` is unchanged in round 2.

**Parser and stream shape.** These needed no change. `TagStreamParser` routes the text before TOOL_START to content and the call to the tool channel.
- Non-streaming returns `content` and `tool_calls` together.
- Streaming emits reasoning deltas, then content deltas, then one finish chunk carrying the calls.
- The whitespace held after THINK_END is prepended to the content, as it is today for `auto`.

**Chat template round trip.** I checked this against the Qwen3.6 template, used as a stand-in. An assistant turn with content and a tool call renders back as `content` (trimmed), `\n\n`, then TOOL_START and `<function=...>`. That is exactly the order the grammar admits.

**Verified locally with llguidance 1.8.0 and the Qwen3.6 tokenizer:**
- Content followed by a call is accepted, with EOS allowed afterwards.
- Content alone cannot end the turn: EOS is masked, and TOOL_START is allowed.
- Content after the call is rejected.
- The content bound holds.
- Disabling content restores the round-1 behaviour.

**New tests:**
- `tests/test_tool_choice.py`
  - `:197`, `:212` (grammar text and env)
  - `:346-372` (llguidance behaviour)
  - `:474-510` (`ChatTemplateRoundTripTests`, which runs when `TABBY_TOOLCHOICE_CHAT_TEMPLATE` points at the model's chat template)
- `tests/test_tool_choice_collector.py`
  - `:389-425` (`ContentWithCallTests`: non-streaming content plus calls, streaming shape `r c t` with a single finish, named with content)

### Change 2: a meaningful identity A/B
`box_probe.py` salts every identity request kind: absent, auto or none, crossed with stream and thinking.
- The salt goes into the first tool's description, the first variable text in the Qwen template, and into the user message (`box_probe.py:72-91`, `:230`).
- The first send of a kind therefore shares no cache page with anything, on either server.
- `--salt` must be the same for A and B. `--compare` refuses when the salts differ.
- The A/B compares only those cold first sends (`box_probe.py:336-353`).
- The warm repeat is reported separately as within-server repeatability, and is not part of the A/B.

### Change 3: probe expectations
- **Forced turns.** `call_ok` (`box_probe.py:200-219`) now accepts content before the call. For streaming it rejects any content delta after the call chunk.
- **New `answer` section.** It sends the TC-45 second turn: user, assistant calculator call, tool result 56, `tool_choice` required. It reports `ok` and, as information, how often 56 appears in the content (`box_probe.py:258-276`).

### Local results (round 2)
- **New tests:** 89 pass, 0 skipped, with the Qwen3.6 tokenizer and template env vars set. Without them, 22 are skipped as designed.
- **Complete served suite:** 238 tests. The only error is the pre-existing `test_generator_recovery`, which needs `pytest`.
- **Collector identity:** auto/none against the unpatched collector, 120 scenarios, 0 differences.
- **HTTP end-to-end** with the fake model:
  - cold A/B 12/12 identical
  - required 12/12, named 4/4, answer 6/6 (56 in content 6/6), validation 4/4, cap 2/2, concurrent 8/8
- **Packaging:** the patch still produces the same tree as `install.py`.

### What only the box can tell for round 2
- TC-45 again. This probably will not be 12/12 even now. Every turn is forced, so the last turn of the conversation must carry a call as well, and the score depends on whether the model puts 56 in the content it writes before that call, and on how the bench picks `final_answer`. Read the `answer` section's 56-in-content count next to it.
- The full 69 x 4 run as the regression gate.
- The cold identity A/B on the real model.

## 8. Round 3 (after the R523 round-2 box run)

Line references in this section are to the round-3 overlay (`overlay/...`) and to `ref/served-src/...`.

### What round 2 showed
- required 36/48, named 3/4, answer 11/24 (56 in content 24/24), cap 1/2, concurrent 3/8, TC-45 10/12.
- Every failing forced row had status 200, `finish_reason: "stop"`, no `tool_calls`, and content that repeated a closing phrase. There was no 503, and `ToolChoiceNotHonoured` never appeared in the log.
- Identity A/B: 3 identical, 9 different. A's rep-2 output of a key equalled B's rep-1 output of the same key, and the other way round.

### Root cause 1: the turn was ended by the loop detector
The round-2 grammar (`start: WS? content? call ...`, section 7) masked EOS after the content, until a call was complete. A model that wanted to end its turn after answering could only continue writing. It repeated its closing phrase, and the exllamav3 loop detector ended the job before the 1024-token content bound was reached:
- The loop window defaults to 800 tokens and 2 repetitions (`ref:tabbyapi/common/sampling.py:316-317`, `:344-350`).
- The job ends with `eos_reason: "loop_detected"` (`ref:exllamav3/generator/job.py:1013-1016`).
- Tabby maps every eos reason other than `max_new_tokens` to `finish_reason: "stop"`, and logs "generation stopped because a token loop was detected" as a warning (`ref:tabbyapi/backends/exllamav3/model.py:1221-1245`).

The 1024-token content cap would not have saved the turn. llguidance leaves only TOOL_START allowed when a `[max_tokens]` rule hits its bound (verified locally), but the loop detector fires first. A smaller cap would force a call in the middle of the model's sentence. It would also still leave the model fighting the mask.

That is also why TC-45's two zero-point trials failed: the forced turn ended in a content loop with no call.

### Root cause 2: the fail-loud check skipped this kind of finish
The round-2 `check_forced_tool_calls` only raised for finishes that end on a stop token or a completed grammar. A `loop_detected` finish passed through as a plain 200 `"stop"` with no call.

It now returns early only for the two ends the client asked for: `max_new_tokens` (the client's `max_tokens`) and `stop_string` (the client's `stop`). Every other end without a call raises `ToolChoiceNotHonoured`, which becomes the 503 or SSE error (`overlay:endpoints/OAI/utils/tool_choice.py:351-375`).

### Fix: ending the content becomes the transition to the forced call
This is the first design option the coordinator suggested. The model's attempt to end its answer is no longer masked. It is used as the point where the server starts the forced call:

1. **Grammar.** The first job's grammar allows the turn to end after content:

```
start: WS? content? call (WS? call)* WS? | WS? content
```

   (`tool_choice.py:225-232`). EOS directly after the reasoning is still masked, because content must start with a non-whitespace character. Content after a call is still rejected.
2. **Continuation.** When the first job ends on EOS (`stop_token` or `end_filter`) after the reasoning, with no TOOL_START seen, `forced_tool_generation` starts a continuation job (`tool_choice.py:476-487`, `:548-558`):
   - **Prompt.** The client prompt plus everything already streamed, with the content's trailing whitespace replaced by `\n\n`. That is exactly how the chat template renders an assistant turn with content and a call: content trimmed, a blank line, then TOOL_START.
   - **Grammar.** `forcing.call_grammar`, which starts with the call and has no content and no outer whitespace (`tool_choice.py:308-313`). EOS is masked until the call is complete.
   - **Trigger.** None, so the filter is active from the first token.
   - **Request id and label.** `-tc{phase}` and `(tool_choice phase N)`, as for the reasoning-cap continuation.
   - **Log.** "tool_choice forcing a tool call: the turn ended in content without a tool call; continuing after the content with a call-only grammar."
3. **What the client sees.** Reasoning, then the content, then the forced call, with one finish (`finish_reason: tool_calls`). Streaming shape is `r c t done`. Usage reports the client's prompt tokens, and the completion tokens of every job (`tool_choice.py:565-583`). The continuation's own numbers are kept in the finish chunk under `tool_choice_continuation`, replacing the round-2 `tool_choice_phase2` key.
4. **Limits.**
   - At most one content-to-call continuation per request.
   - The continuation's EOS-without-call, a loop, or any other end without a call raises `ToolChoiceNotHonoured`.
   - The client's `max_tokens` covers all jobs: the continuation gets what is left, and if nothing is left the request ends with `length`.

**Why this design.**
- **The model is never asked to write against a masked EOS.** Masked EOS is the classic constrained-decoding loop, and it is what R523 hit.
- **The prompt is the model's own format.** The call is generated from a prompt that is exactly what the template renders for a content-plus-call turn, so the model writes the call in its own format instead of being pushed into it mid-sentence.
- **The cost is one extra prefill.** Only the turns that answer first pay it. The continuation's prompt extends the first job's prompt and output, so the pages the first job wrote are reused. Its uncached part is at most the content plus one page.
- **The common case is unchanged.** A model that calls directly still runs one job.

**The same loop now covers every forced request.** The wrapper also runs when generation does not start inside a reasoning block, with thinking off or an unarmed trigger (`chat_completion.py:739-754`, with `reasoning_end=None`). A thinking-off turn that answers and stops is continued the same way. `auto` and `none` never enter the wrapper: that `else` branch is the unchanged original call.

### Root cause 3: the identity A/B, the probe and the server
- **The probe was not the cause.** The per-key prompt was already a pure function of (salt, key): the salt was `f"{salt}:{choice}:{stream}:{thinking}"`, with no counter, and the identity section runs first and in the same order in `--identity-only` and in the full run. Rep 1 and rep 2 of a key sent byte-identical bodies, and the same bodies went to A and B.
- **Why that rules out the probe.** One body returned output X on A's first send and Y on its second, and Y then X on B. So on each server the output of one prompt varied between sends. That is history-dependent numerics in the engine, and the tool_choice patch cannot cause it: `auto` and `none` take the unchanged branch, which the 120-scenario differential test confirms at the collector level.
- **Likely mechanisms,** all history-dependent:
  - **The prefix cache.** The warm repeat computes the last prompt position from a cached page, and a hybrid model restores recurrent state from a stash, where the cold send runs the full prefill.
  - **The online draft calibrator.** It is process-wide: `DraftConfidenceCalibrator` is created once per generator (`ref:exllamav3/generator/generator.py:304-307`) and updated after every verify round (`:1369-1380`). When `dynamic_draft` is on (`ref:tabbyapi/backends/exllamav3/model.py:267`, `:843`), the draft length, and so the verify batch shape, depends on every request served since boot.
  - **Different verify shapes give bit-different logits.** A greedy near-tie then flips. The reasoning text was different from early on, which is where such a near-tie tips.
  - **B's history may also have differed from A's.** If B served the daily's c1 fingerprint check before the probe and A did not, the two servers started the probe with different calibrator and cache state.
- **Greedy is what reaches the sampler.** `temperature 0` builds `SS_Argmax` (`ref:tabbyapi/backends/exllamav3/model.py:1364-1368`, `sampler.py:212-223`). There is one exception: an adaptive-P preset (`adaptive_target < 1.0`) samples even at temperature 0 (`sampler.py:147-153`, `:215-217`). The probe now sends `adaptive_target: 1.0` too. The request-start log line reports `temperature: 0, greedy (req)` for a greedy request (`sampler.py:152-153`, `common/gen_logging.py`).
- **The sampling RNG is not the cause either.** Each job seeds its own RNG from the request's `seed` (1234 in the probe), `ref:exllamav3/generator/job.py:222-224`, and draws from it per step (`draft_overlap.py:46-49`). There is no global counter, so even a sampled request is a fixed function of its logits. The shipped sample preset leaves adaptive-P off (`ref:tabbyapi/sampler_overrides/sample_preset.yml:149-154`). The daily's preset is not in `ref/`: the box check is the request-start line, which must read `temperature: 0, greedy (req)` and must not list `adaptive_target`.
- **Why not a within-server "same cold key twice" check.** On one server the second send of a body is always warm: the prefix cache serves it, and the last position is computed with different kernel shapes. So rep 1 == rep 2 is not a greediness test on this engine. The unpatched image scored 2/12 on it in round 1. What an A/B needs is boot-to-boot determinism of one image under one request sequence, which the A vs A2 control tests (box-ab-spec section 2). Greediness is tested directly by the logprob top-1 check. B's c1 fingerprint matching the daily's canonical `ae890c45` is already evidence of greedy `auto`-path identity on the real model from a fresh boot.

### Probe changes (`box_probe.py`)
- **Identity rows** carry `seq` (position in the request sequence) and `prompt_sha256` (sha256 of the exact request body, `:107-108`, `:285-299`). The salt function is `identity_salt` (`:243-245`).
- **`--compare`** refuses when keys, body hashes or sequence positions differ (`:407-451`). It reports the cold verdict and the warm repeats separately. It also notes when A's output equals B's other rep, which is the history signature.
- **A new `greedy` section** runs after identity, so it does not change identity's history. It sends 2 fresh-salt thinking-off requests with `logprobs` and `top_logprobs: 2`, and checks that every content token is the top-1 of its position. It reports the smallest top-1/top-2 logprob gap (`:248-276`, `:301-304`). Probabilities are a softmax of the raw logits (`ref:exllamav3/generator/job.py:587-597`), so this checks argmax sampling directly.
- **Summaries carry `eos_reason`** (and `stop_str` for non-streaming), so a `loop_detected` end is visible per row.

### Tests (round 3)
- **`tests/test_tool_choice.py`:**
  - `test_content_may_end_the_turn` (`:351`)
  - `test_eos_at_once_after_reasoning_is_masked` (`:360`)
  - `test_call_grammar_starts_with_the_call` (`:365`): the call grammar forbids content and EOS, allows TOOL_START, and accepts a call
  - `test_content_is_bounded` (`:386`): after the bound, TOOL_START or EOS
  - the grammar text and env tests check `call_grammar`
- **`tests/test_tool_choice_collector.py`, `ContentThenCallContinuationTests` (`:444-513`):**
  - Non-streaming: two jobs; the continuation id is `-tc2`, with no trigger, the call-only grammar, and the prompt `PROMPT + reasoning + THINK_END + content.rstrip() + "\n\n"`. The result carries content and the call. Usage covers both jobs.
  - Streaming: `r c t`, single finish.
  - Thinking off.
  - Named.
  - A `loop_detected` content loop raises `ToolChoiceNotHonoured`.
  - Content disabled keeps one job.
- **The usage double count fixed on the way.** Continued usage now adds the earlier jobs' tokens to the last job's own count (`tool_choice.py:470-472`), so the last job's tokens are no longer counted twice.

### Local results (round 3)
- **New tests:** 99 pass, 0 skipped, with `TABBY_TOOLCHOICE_TOKENIZER` and `TABBY_TOOLCHOICE_CHAT_TEMPLATE` pointing at the Qwen3.6 files and `TABBY_TOOLCHOICE_REQUIRE_LLG=1`. Without them, 24 are skipped as designed.
- **Complete served suite on the patched tree:** 248 tests. The only error is the pre-existing `test_generator_recovery`, which needs `pytest`.
- **Collector identity,** auto/none against the unpatched collector: 120 scenarios, 0 differences.
- **HTTP end-to-end** with the fake model, salt 777, 3 trials:
  - Cold A/B 12/12 identical, warm 12/12.
  - Patched: required 12/12, named 4/4, answer 6/6 (all through the content-to-call continuation, 6 log lines), validation 4/4, cap 2/2, concurrent 8/8.
  - Unpatched: required 0/12.
  - The `greedy` section reads 0/2 against the fake, which returns no logprobs. `greedy_row` and the compare refusal were checked on synthetic input.
- **Packaging.** `served-source.patch` gives the same tree as `install.py`, and `install.py` is idempotent.

### Risks (round 3)
- **Continuation rate on the real model.** A turn that answers and then stops pays one extra prefill of its uncached tail. The box's count of "continuing after the content" log lines against forced requests gives the rate.
- **What the call looks like after a finished answer.** The continuation asks for a call after a complete answer, so the model may pick a redundant tool (for example calculator again on the TC-45 second turn). `required` asks for exactly that; tool-eval scores it.
- **Stop strings.** `stop_string` ends are accepted without a call, because the chat path adds no template stop strings, so they come from the client's `stop`. A sampler preset that adds `stop` strings could end a forced turn early with a 200 and no call.
- **Re-tokenisation.** Continuation prompts are text: the streamed output is re-tokenised, which can cost prefix-cache reuse at the seam. It does not affect correctness.
- **Identity.** If the A vs A2 control fails, the output-level A/B cannot gate `auto`/`none`. The CPU differential test and a deterministic-engine A/B remain.
