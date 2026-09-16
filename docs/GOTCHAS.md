# GOTCHAS — what it looks like vs what it is

Every entry here was paid for with a wrong number or a dead session on this box. Dates are when it bit.

## 1. A `--tokens 8192` run that measured 8 tokens (2026-09-16)

**Looks like:** c1 aggregate throughput of 9.0 t/s, where the same config reads 157–177 t/s at c1 elsewhere.
It reads as a collapse at long generation.

**Is:** the probe posted a bare `/completions` prompt at `temperature 0` and let the model stop when it wanted.
On this checkpoint it emits EOS after 8 tokens, so the row divided 8 tokens by a wall that included prefill. The
server log shows it plainly: `#168 completions (stream): 8 tokens generated`, `#177 … 1 tokens generated`. A
`--tokens 8192` arm never generated 8192 tokens.

**Fix:** force the length with **`min_tokens`** (see #2), and record the finish reason.

## 2. `ignore_eos` is accepted and ignored (2026-09-16)

`ban_eos_token` / `ignore_eos` is a recognised request field, validated, and listed in TabbyAPI's
`UNSUPPORTED_PARAMS` for the exllamav3 backend: a request that sets it gets a warning and the parameter is
dropped. `min_tokens` is the field that works — it is plumbed to `Job(min_new_tokens=...)`, which suppresses EOS
until the count is reached (`exllamav3/generator/job.py:459`). Verified: `min_tokens: 4096` produced exactly
4,096 tokens with `finish_reason: length`.

## 3. Forced length still stops early: the loop detector (2026-09-16)

A forced 16,384-token greedy code generation ended at **7,107 tokens** with the server warning
`generation stopped because a token loop was detected`. exllamav3 has a loop detector and it will end a long
greedy run. Any throughput sample whose `finish_reason` is not `length` is not a throughput sample.

## 4. The engine under-reported its own generation by ~5× (2026-09-16, fixed)

**Looks like:** 32.7 T/s logged for a generation the model's own tokenizer counts as 17,544 tokens, i.e. a
server 5× slower than it is; a client's `usage` block that disagrees with the text it received.

**Is:** `exllamav3/generator/job.py`, `prepare_for_requeue`, carried `"rq_new_tokens": self.new_tokens` — the
current physical segment only — while the final result reports `rq_new_tokens + new_tokens`. With the 2048-token
requeue budget, any generation longer than that is reported as at most its last two segments (~4k), whatever its
true length. It affects `usage.completion_tokens` and the logged T/s; token *limits* are unaffected, because
`max_new_tokens` is carried from the sequence.

**Fix:** image `tabbyapi:53da7919-rqcount` (one-line patch in the build, asserted at build time). Verified:
18,739 reported for a generation whose reasoning tokenizes to 18,630. Only generations past ~2045 tokens were
ever affected, which is why the 128/256/512-token probes of R329–R336 are clean.

## 5. Every client that sends no sampler was served at temperature 1.0, untruncated (2026-09-16, fixed)

**Looks like:** a reasoning model that breaks off mid-thought and then thinks in the output channel; multilingual
word salad in the visible answer; a DSH session that produces no tool call at all.

**Is:** TabbyAPI has no sampling fallbacks unless `sampling.override_preset` names one — it warns about this at
boot, and the warning was in the log four minutes before the failing request. DSH sends `max_tokens` and no
sampler, so every one of its requests sampled a 3.05 bpw checkpoint at raw T=1.0 with no truncation and no
penalty. **Fix:** the `qwen38_thinking` preset (0.6 / top_k 20 / top_p .95, `force: false` fallbacks), mounted in
by the launcher.

## 6. `depth_ss.py` reports zero decode against TabbyAPI

It reads llama.cpp's `timings` block for `predicted_per_second`; TabbyAPI returns no such block, so every rate it
prints is a zero that looks like a measurement — the same trap `oai_conc.py` documents for `usage: null`. Use
`bench/probe.py` here, or the client-side rate from SSE timestamps.

## 7. Decode rate is content-dependent by ~2× on this checkpoint

Same config, same box, same day: `/completions` code at 176.3–187.1 t/s with 62–72 % draft acceptance, versus a
chat prose analysis at 94.1 t/s with 46 %. MTP acceptance tracks how predictable the continuation is, and prose
analysis is not predictable. A decode number without its kind is not comparable to another one.
