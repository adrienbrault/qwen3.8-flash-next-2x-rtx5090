# tool-choice-r1: box A/B spec

- **A** = the unpatched daily image, `tabbyapi:stack-r4-e3r2`.
- **B** = the same image with this overlay.

Both images run the same model, config and launch flags, on a candidate port, never the daily's port. Tags are written as THINK_START, THINK_END, TOOL_START and TOOL_END.

## 0. Build (legacy builder, context = `out/tool-choice-r1/`)

```
docker build -f Dockerfile.box -t tabbyapi:stack-r4-e3r2-tc1 out/tool-choice-r1/
# any other daily image:  --build-arg BASE=tabbyapi:decode-kernels-r4
```

The build must print all of the following:
- `installed:` for 4 files, then `tool-choice-r1 overlay verified: 4 files under /app`.
- The llguidance version.
- `OK` for `tests.test_tool_choice`, run with `TABBY_TOOLCHOICE_REQUIRE_LLG=1`. That run proves the image's llguidance compiles the grammar with `[lazy]` rules and `<[id]>` token literals.

A baseline mismatch fails the build before anything is written. Record the llguidance version.

## 1. In-container tests against the real tokenizer (GPU container, before serving)

```
docker run --rm --gpus all --entrypoint python3 -w /app \
  -v <flash-next model dir>:/model:ro \
  -e TABBY_TOOLCHOICE_TOKENIZER=/model/tokenizer.json \
  -e TABBY_TOOLCHOICE_REQUIRE_LLG=1 \
  tabbyapi:stack-r4-e3r2-tc1 -m unittest -v tests.test_tool_choice tests.test_tool_choice_collector
```

Must hold:
- All pass, with 0 skipped.
- The `RealTokenizerGrammarTests` class passes. It checks that TOOL_START, TOOL_END and THINK_END (the filter trigger) are single tokens in the Flash-Next vocabulary and that the grammar accepts the model's call format. It also checks that EOS is masked right after the reasoning, that content may end the turn (the server then continues with the call-only grammar), that the call-only grammar starts with TOOL_START and masks EOS until the call is complete, that a second reasoning block and content after a call are rejected, and that named, single-call and whitespace bounds hold.

The collector tests import `common.model`, which imports exllamav3, so run them in the GPU container, not at build time.

## 2. Request matrix: `box_probe.py`

`box_probe.py` uses the standard library only. Run it from any host that can reach the port.

Generation on this stack depends on the server's history since boot (prefix cache, recurrent stashes, and the process-wide MTP draft calibrator when `dynamic_draft` is on; see impl-status section 8). The identity A/B is therefore only meaningful between servers that were booted fresh and received exactly the same requests in the same order. Nothing else may reach either server before the probe: no warm-up, no fingerprint check, no health request that generates. Run the probe as the first thing after boot.

Three boots, one salt:

```
SALT=$(date +%s)
# 1. fresh boot of A (unpatched image)
python3 box_probe.py --url http://127.0.0.1:<port> --salt $SALT --out A.json --identity-only
# 2. restart A, fresh boot of the same unpatched image (control)
python3 box_probe.py --url http://127.0.0.1:<port> --salt $SALT --out A2.json --identity-only
python3 box_probe.py --compare A.json A2.json       # determinism gate
# 3. fresh boot of B (patched image)
python3 box_probe.py --url http://127.0.0.1:<port> --salt $SALT --out B-id.json --identity-only
python3 box_probe.py --compare A.json B-id.json     # the A/B
# 4. then, on the same B boot, the full matrix with a new salt
python3 box_probe.py --url http://127.0.0.1:<port> --salt $((SALT+1)) --out B.json --trials 12
```

- **The control gates the A/B.** If A vs A2 is not 12/12 on the cold sends, the engine is not deterministic across boots under this config. The A/B then cannot judge byte identity from outputs, and the verdict for `auto`/`none` rests on the CPU differential test (120 scenarios, 0 differences) and the log checks below. Report the control's numbers either way.
- **Optional deterministic run.** If the control fails, repeat steps 1 to 3 with MTP drafting fixed or off (no `dynamic_draft`). The patch does not touch the backend, and the `auto`/`none` branch of the collector is the unchanged original call, so an A/B under a deterministic engine config is a valid identity check.

The prompt is "What is 7 times 8?", sent with 12 tools (calculator, get_weather and 10 others). Each section below lists what must hold.

### identity
The requests are `tool_choice` absent, `"auto"` and `"none"`, each non-streaming and streaming, and each with thinking on and off (`chat_template_kwargs.enable_thinking=false`). They use greedy sampling (`temperature 0, top_k 1, seed 1234`), and each request is sent twice.

- **Cold A/B** (`--compare`). Each request kind carries its own salt in the first tool's description and in the user message, so its first send shares no prefix with anything on either server. Use the same `--salt` for A and B.
  - The request body is a pure function of (salt, key). Each row stores its `prompt_sha256` and its position `seq` in the request sequence. The compare refuses mismatched salts, keys, body hashes or positions.
  - The cold first sends are the verdict, and all 12 must be byte-identical. The warm repeats are compared too, and reported separately.
  - A `DIFF` line that says "A's output equals B's rep N" is the history signature: the same body gave both outputs on each server.
  - The comparison covers the whole response JSON and every SSE chunk, minus `id`, `created`, timing fields and tool call ids.
- **Within-server repeatability** (the warm repeat 2 against the cold send). This is reported separately as `identity warm repeat == cold ...` and is not a gate. R523 saw 2/12 on the unpatched image, which is prefix-cache and MTP numerics.
- Run at c1, sequentially, with nothing else hitting either server.
- On B, the request-start log line for these requests must not list `grammar_string`.
- On both, it must show `temperature: 0, greedy (req)`. The probe also sends `adaptive_target: 1.0`, because an adaptive-P preset samples even at temperature 0.

### greedy
Runs right after identity, in both modes. It sends 2 fresh-salt `auto` requests, with thinking off, `logprobs` and `top_logprobs: 2`.

- Each row must be `ok`: every content token is the top-1 token of its position. This checks that argmax sampling reached the sampler.
- The report prints the smallest top-1/top-2 logprob gap. A gap near 0 is a near tie, which is where history-dependent numerics flip a greedy output.
- If the rows show 0 content tokens, the server returned no logprobs for this request, for example because drafted tokens carry none. Record that and rely on the log line above.

### required
`tool_choice: "required"`, streaming and not, thinking on and off, 12 trials each (48 requests) at the server's default sampling.

- 48/48 `ok`:
  - HTTP 200 with `finish_reason: tool_calls` and a single finish.
  - Content is allowed only before the call.
  - At least one call, every name in the tools, and `arguments` valid JSON.
- No row may end with `finish_reason: stop` and no calls. Each row's `summary.eos_reason` is recorded. `loop_detected` on a forced row is now a 503, never a 200.
- Streaming shape: `r t done` or `r c t done` with thinking on, and `t done` or `c t done` with thinking off. Content deltas may come before the call chunk, never after it; the probe checks this.

### answer (TC-45 second turn)
The request is user, then an assistant `calculator(7 * 8)` call, then the tool result 56, with `tool_choice: "required"` and thinking on, streaming and not.

- Every row must be `ok`: a forced call, with any content only before it.
- `56 in content N/M` is informational: how often the model surfaces the answer next to its forced call. This is what TC-45 scores. Record it with the TC-45 result.
- A turn that answers first and then stops is continued by the server with a call-only grammar. For each such request, B's log shows "tool_choice forcing a tool call: the turn ended in content without a tool call; continuing after the content with a call-only grammar", then a request-start line labelled `(tool_choice phase 2)`. Count these lines and report the count next to the `answer` numbers.
- No request may end in a 503 or SSE error from `ToolChoiceNotHonoured`. That error means the model finished without the forced call, for example because the image dropped the grammar, or because the continuation also ended without a call. Grep the log for it, for "Skipping because the grammar", and for "token loop was detected".

### named
`{"type":"function","function":{"name":"get_weather"}}` on the arithmetic prompt, streaming and not, thinking on and off.

- 4/4 `ok`: exactly one call, to `get_weather`.

### validation
All four must return HTTP 400:
- `required` without tools
- named without tools
- named `nope`
- `required` with a `json_schema` `response_format`

On A, all four return 200.

### cap
`required` with `reasoning_budget_tokens: 64`, streaming and not.

- 2/2 `ok`.
- B's log shows `tool_choice forcing a tool call: reasoning reached N tokens (cap 64)`, where N is at least 64 because MTP merges chunks. It also shows a second request-start line labelled `(tool_choice phase 2)`.
- Record the phase-2 metrics line: prompt tokens, cached tokens and prefill time. Phase-2 uncached tokens (prompt minus cached) must be at most the reasoning length plus one cache page. This is the "at most one extra prefill of the reasoning" latency bound, and the aim is that most of it is cached.
- Also run once with the server started with `TABBY_TOOL_CHOICE_REASONING_CAP=256` and no budget. It must behave the same.

### concurrent
4 simultaneous required streams, then 2 auto and 2 required.

- 8/8 `ok`, with no errors in the server log.

## 3. Latency

- **Required, single-job path.** Compare TTFT and total time for required against auto on the same prompt, both on B with 12 trials. The expected extra cost is the grammar compile (about 2 ms) and masks during the call (µs per token). Requests that call directly run one job, with no extra prefill. Requests the server continues (the content-to-call or reasoning-cap log lines) pay one extra prefill of the uncached tail: at most the content or the reasoning plus one page. Report their latency separately, with the count of continued requests.
- **First forced request after boot.** It builds the llguidance tokenizer (about 1 s, once). Note this number separately.
- **c4 decode with one required stream live.** Run 3 auto streams alongside 1 required stream, then 4 auto streams, and compare the aggregate decode t/s of the auto streams.

  Check `EXL3_DECODE_OVERLAP` inside the container. If it is `1`, a job that has filters moves the whole batch to the serial verify path while it runs (`exllamav3/generator/draft_overlap.py:28`), so expect a delta and report it. If it is `0` or unset, the delta should be within noise.

## 4. tool-eval-bench

1. TC-45 "tool_choice=required Compliance" on A and B. B must pass 12/12 (A fails 12/12 today).
2. The full 69 x 4 run on B is the regression gate. Its score must be at least A's same-day score, with no scenario newly failing outside run-to-run noise. Keep the raw outputs of both runs.

## 5. Pass criteria

- The build is green.
- The in-container tests are green, with 0 skipped.
- The A vs A2 control is recorded. If it is 12/12 on the cold sends, the cold A/B must be 12/12 too. If the control is not 12/12, the A/B is reported but not gated; the deterministic run, when it is done, gates instead. The warm numbers are recorded, not gated.
- Greedy rows: every content token is the top-1 token.
- No forced row ends in `stop` without calls. No `ToolChoiceNotHonoured` and no "token loop was detected" warning on forced requests; if either appears, report the row.
- The probe on B is all green: required, named, validation, cap and concurrent.
- TC-45 is 12/12.
- The full tool-eval is no worse than A.
- The phase-2 uncached prefill is at most the reasoning length plus one page.
- The c4 delta is reported.
