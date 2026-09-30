# GOTCHAS — what it looks like vs what it is

Each entry records a wrong number or a lost session on this box. The date is when it happened.

## 1. A `--tokens 8192` run that measured 8 tokens (2026-09-16)

**Looks like:** c1 aggregate throughput of 9.0 t/s, where the same config reads 157–177 t/s at c1 elsewhere.
It reads as a collapse at long generation.

**Is:** the probe posted a bare `/completions` prompt at `temperature 0` and let the model stop on its own. On this
checkpoint it emits EOS after 8 tokens, so the row divided 8 tokens by a wall that included prefill. The server log
records it: `#168 completions (stream): 8 tokens generated`, `#177 … 1 tokens generated`. A
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

Since 2026-09-27 ([R783](../bench/results/r783-loopthink.md)) a chat request that starts in the thinking ends a loop
there with a forced `</think>` instead of a stop, and the engine's detector on such a request has a 1,600-token window,
so a content-phase loop runs up to about twice as long before it is ended (not measured on the GPU; #25).
`/v1/completions` requests and chat requests with thinking off keep the 800-token window.

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

**Lost, 2026-09-16 to 2026-09-26:** the fix lived in `Dockerfile.tabbyapi` only. `Dockerfile.tabbyapi-qsa-cid`
installs TabbyAPI and ExLlamaV3 afresh, so every image served from `qsa-cid-pr337` to `stack-r3-rows32` counted
generations longer than ~4,096 tokens short (9,000 / 13,000 / 20,000 forced tokens reported as 2,969 / 2,888 / 3,714).
Restored as the `tokcount-r1` layer and served since R747 ([bench/results/r747-tokcount.md](../bench/results/r747-tokcount.md)).
Any figure taken from the server's per-request log in that window is suspect; the client-timed decode figures are not.

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

## 8. A CPU-bound build next to a GPU measurement perturbs it (2026-09-16)

**Looks like:** stamina decay. A c4 soak that held 62–66 t/s per stream for five rounds dropped to 40.5 t/s at
round six and stayed there, which is the shape a thermal or fragmentation cause would produce.

**Is:** a native extension rebuild running on the same box. Compiling exllamav3's extension starts 487 compiler
processes and took the load average to 18.6; the decode path needs host CPU for sampling, launching and the PLE
gather, so the GPU starved while it was "idle" at 17 % utilisation. Same configuration, same server, no restart
between the clean and dirty rounds — the only variable was the build.

**Fix:** nothing runs on the box during a measurement except the measurement. That includes builds, and it is the
reason the A/B scripts in `bench/` take the GPU lock even when they only probe.

## 10. The two servers name the thinking channel differently, and one probe read only one name (2026-09-16)

**Looks like:** a vLLM server on the same box refusing concurrent deep-context work. Eight 38,283-token requests:
one or two complete, the rest come back with a `usage` block reporting 512 completion tokens, **no text at all** and
no error, each after ~16 s — while that server's own log shows `200 OK` for every one of them. Read at face value
that is "the server drops 6 of 8 deep-context requests", and it was twice reproduced before being questioned.

**Is:** two OpenAI-compatible servers naming the same thing differently, and an instrument that knew one name.

- **vLLM's** OpenAI server streams the model's thinking as `delta.reasoning` — captured raw:
  `data: {…"delta":{"reasoning":"The"}…}`, with the delta keys observed as `content, reasoning, role`.
- **TabbyAPI** streams it as `delta.reasoning_content`.

`bench/probe.py` collected text from `delta.content` and `delta.reasoning_content` only. Against a vLLM server, a
request whose entire forced length went into thinking therefore looked like a request that returned nothing — and
with `min_tokens: 512` forcing exactly 512 tokens, `content` is *legitimately* empty for a request that never
finished thinking. The counts agree with that reading: 157–179 frames arrive per request, none of them content.

It now reads `content`, `reasoning_content` **and** `reasoning`, in the probe, the pair probe and the capability
gate.

**The same mistake in the same investigation:** the raw-capture script passed a 250 KB JSON body as a `curl` argv
element and got `OSError: [Errno 7] Argument list too long`. Eight empty capture files, and the `200 OK` lines the
server logged belonged to other clients. Bodies now go through a file (`--data-binary @body.json`), and a capture
that writes zero bytes prints its curl exit status and stderr instead of looking like a server that said nothing.

**The rule this produced:** a probe that reports "no text" must also report *why* — frame shapes and field names
seen, the raw bytes for at least one request, curl's exit status, or the server's own log. Both of the 2026-09-16
false readings were caught only because a second instrument or a raw capture disagreed.

## 9. Decode rate is content-dependent by ~2× on this checkpoint

Same config, same box, same day: `/completions` code at 176.3–187.1 t/s with 62–72 % draft acceptance, versus a
chat prose analysis at 94.1 t/s with 46 %. MTP acceptance tracks how predictable the continuation is, and prose
analysis is not predictable. A decode number without its kind is not comparable to another one.

## 11. A GPU unit stalled behind its own child, and the diagnostic said nobody held the lock (2026-09-16, fixed)

`r363-enable.sh` takes the GPU-exclusive lock and then runs `r362-pr337.sh` as its child. r362 took the lock too. A
child that **re-opens the lock path** gets a *second* lock on the same file, so it waited for the parent — which was
waiting for it. Nothing ran, the GPU sat idle, and seven other units queued behind the pair.

The diagnostic was wrong in two ways:

1. `flock` holders are **not** reported in `/proc/PID/fdinfo` — that field is for POSIX record locks. The correct
   source is `/proc/locks`, whose inode field is **decimal**; grepping a hex-translated inode returns nothing, which
   was read as "nobody holds it".
2. "No process is building or probing" is not evidence that no unit is running. Both processes were alive and idle by
   construction: each was blocked on the other.

Fixed in `flan/lib/gpu-queue.sh`: `gpu_lock()` reuses fd 9 when it is already the lock file, so an inherited descriptor
shares the holder's open file description and `flock` succeeds at once. r362 calls it. The general rule for this host:
**a script that the lock-holder invokes must not take the lock itself** — inline `exec 9>` + `flock` is only safe at
the top level, which is what every unit except this pair is.

The queue was also rebuilt as a single chain ([`scripts/r370-chain.sh`](../scripts/r370-chain.sh)) rather than eight units sharing one flock: the
order in which waiters acquire a flock is not defined, so "queued" never meant "ordered".

## 12. A validation that passes on an empty file (2026-09-16)

Copying a script to the host and checking it with `bash -n` reported success on a **0-byte file**: an empty script is
valid bash. The transfer had produced an empty file, the check confirmed nothing, and the unit was launched, exited
immediately with status 0, and read as "ran successfully". It was the third passing check that session which
measured the absence of a thing rather than the thing.

What the check should have been, and now is: `wc -c`, plus a grep for a string that must be present (`greedy_same`),
plus comparing hashes of both copies. Size and content are different claims; a syntax checker answers neither.

The general rule: **a check must be able to fail.** `cmp` with a glob that matches one file, `bash -n` on an empty
file, a probe that reads one channel of four, a build whose verification step imports without its library, a lock
holder read from the wrong field of `/proc/locks` — each returned success and each measured nothing.

**The same condition recurred an hour later.** Pushing four updated scripts to the host used a loop with no input
redirection —

    for f in a b c; do ssh flan "sudo cat > /srv/qwen5090/$f && chmod +x /srv/qwen5090/$f"; done

— so `cat >` truncated each target and read nothing. **Three scripts became 0-byte files**, the check I ran was
`bash -n` (which an empty script passes), and the chain then executed them as `### DONE r366-ourkernel in 0s`,
`r364-hotvocab in 0s`, `r367-slots in 0s`. Three experiments completed in the log and did no work at all, and
"completed in 0s" was the only signal, which does not distinguish fast from empty.

The fix is a check that can fail, applied to every copy: byte count, a string that must be present, and matching
hashes on both ends. `r372-chain2.sh` now refuses to start if any step it was asked to run is under 500 bytes, so the
condition cannot recur silently. **A step that reports success without doing work is indistinguishable from a step
that did the work quickly, unless something counts.**

## 13. Two bugs in the identity helper, one of which made its own tests vacuous (2026-09-16)

An external review found the first; investigating it found the second. Both were in `lib/greedy-compare.sh`, the
helper every identity gate uses.

1. **Two empty directories compared EQUAL.** `greedy_hash` hashed an empty file listing into
   `e3b0c44298fc1c14` — the SHA-256 prefix of the empty string, a *non-empty* string — and `greedy_same` only required
   a non-empty, equal hash. So if both arms' captures silently failed, **the identity gate passed**, and that gate
   guarded every identity result quoted that day. The helper's own test covered directories that did not *exist*
   (which correctly differ) and never directories that existed and were *empty*.
2. **The hash could not be computed on macOS at all.** The file listing used `find -printf`, a GNU-only primary. On
   macOS that pipeline printed an error and hashed nothing, returning the same empty-input digest for *any* pair of
   directories, so local runs of this helper measured nothing while the ones run over ssh on the Linux host were
   valid. `greedy_hash` now includes the file count, lists files portably, picks `sha256sum` or `shasum` at
   runtime, and returns empty rather than the empty-input digest when it cannot compute.

Consequences, both applied: every hash the old function produced is superseded (`18e30f17883a38eb` ->
`8179222fec8df3b8` for the served configuration), and the verdicts were re-derived with the fixed helper against the
existing captures rather than assumed to survive — #290's paired re-capture is identical across all three arms, and
#246 still reads control == feature-off, control != feature-on.

**The test of a check is whether it can fail**, and a check that cannot compute must refuse rather than return a
plausible value.

## 14. GSM8K measured lm-eval's stop strings, not arithmetic (2026-09-18)

**What it looks like:** the 2.50 bpw pack scores 0.804 on GSM8K against 0.914 for the 3.05 bpw pack, and most wrong answers are empty.
**What it is:** the model's reasoning restates the problem as "Question: …", and TabbyAPI applies lm-eval's request-level stop list (`Question:`, `</s>`, `<|im_end|>`) to the reasoning text. The reply ends mid-thought with empty content. Without the stop list both packs score 0.980 and 0.978. GSM8K runs go through [`bench/nostop_proxy.py`](../bench/nostop_proxy.py) ([R509](../bench/results/r509-gsm8k-nostop.md)).

## 15. A `RUN` heredoc in a Dockerfile runs on empty input on the box (2026-09-18)

**What it looks like:** an image that should patch `generator.py` builds cleanly, and every arm of the experiment behaves like the default.
**What it is:** the box builds with Docker's legacy builder, which drops the body of a `RUN <<EOF` heredoc, so `python3 - <<'PY'` runs on empty stdin. Image recipes here copy scripts in and run them, and assert the change with `grep` or an import at build time ([R497](../bench/results/r497-draft-confidence.md)).

## 16. A "cold" prefill that was a prefix-cache hit (2026-09-18)

**What it looks like:** a second cold prefill run of the same length finishes in 0.19 s at 30k tokens, or a 120k prefill reads 11,070 t/s.
**What it is:** `fn_bench --unique` seeded its filler text identically on every invocation, so later runs and longer contexts shared cached prefixes with earlier ones. `fn_bench` takes `--salt`; cold prefill is measured one invocation per length with a random salt ([R507](../bench/results/r507-prefill-e3.md)).

## 17. One greedy prompt per kind is a content-sensitive number (2026-09-18)

**What it looks like:** a change that moves one layer to the other card reads −17 % on code at c1.
**What it is:** any change in rounding changes the greedy text, and the draft acceptance of that text with it. On 12 sampled prompts per kind the same change reads +1.4 % at c1. Changes that alter numerics are judged on [`bench/multiprompt.py`](../bench/multiprompt.py) ([R487 and R488](../bench/results/r487-pool-393k.md)).

## 18. A second request with the same 30k prompt returns a different fingerprint (2026-09-18)

**What it looks like:** the 30k greedy fingerprint differs between two requests in one boot.
**What it is:** the second request reuses the first one's cached prefix, which changes the prefill shape. Gates take the first 30k request of a fresh boot only ([R511](../bench/results/r511-promote-2p50.md)).

## Free VRAM at boot does not tell you a decode graph will fit

A page pool that clears the boot-time headroom rule on both cards can still abort later with
`GPU assert: out of memory exllamav3_ext/graph.cu 51`. CUDA graphs are captured per batch size, the first time that
batch size decodes, and capture needs free VRAM at that moment. A round that measures 1, 4 and 8 streams never
captures the 5-stream graph, so a gate that runs 5 concurrent requests is where it fails — which is what happened at
1,015,808 tokens with 901 MiB free on cuda:0 (2026-09-19, [R575](../bench/results/r575-promote-mtp-kv-window.md)).

Ladder each candidate pool with a full 1-to-8-stream ramp, one request per batch size, before trusting it.

## 19. The first ~1,024 generated tokens read fast (2026-09-22)

**What it looks like:** a length effect — acceptance 3.710 tokens per verify step at 256 forced tokens falling to 2.562 at 3,072, written up as acceptance decaying with generation length.
**What it is:** a start-of-generation transient — the warm first steps amortise over a short generation, so a 256-token row reads high. The same table's own middle points said so: 1,024 → 2.586 against 3,072 → 2.562 is −0.9 %. Gates that feed published numbers force at least 1,024 tokens per request ([`bench/fn_gate.sh`](../bench/fn_gate.sh) refuses less); 256-token rows are legal only for same-shape A/B screening, where the transient inflates both arms equally.

## 20. `EXTRA_ENV=<one flag>` boots a different stack, not a flag flip (2026-09-22)

**What it looks like:** `EXTRA_ENV=EXL3_SOMETHING=1 ./scripts/launch-flashnext.sh` turns one feature on for an arm.
**What it is:** a full override — the launcher expands `${EXTRA_ENV:-<the tuned set>}` only when the variable is unset, so the boot drops every tuned key. The arm measures a different stack and reports it as a flag flip. `EXTRA_ENV_ADD=` appends to the tuned set, and the launcher logs the resolved set as `env keys (N): ...` — assert the count; `assert_env_keys` in [`scripts/lib/serve-ctl.sh`](../scripts/lib/serve-ctl.sh) does.

## 21. A restart-looping container survives every name-grep wait (2026-09-22)

**What it looks like:** a boot wait on `docker ps` sees the container vanish when the config fails.
**What it is:** `docker ps` lists a `Restarting` container as present, so under `--restart unless-stopped` a config the schema rejects in 0.4 s keeps the wait alive for the whole timeout while nothing serves. The launcher validates the generated config inside the image (`config.load()`, the same path boot uses) before the old container is stopped, and the wait reads `.State.Status`, which fails in seconds on `restarting`, `exited` or `dead`. `DRAFT_MODE` emits the whole draft block from one knob so the illegal `disabled` + policy combination cannot be written.

## 22. A mean over a bimodal population is a wrong denominator (2026-09-22)

**What it looks like:** "72.8 t/s per stream" as the production decode rate.
**What it is:** the mean over requests of wildly different weights — sub-50 t/s requests were 21 % of the tokens but 59 % of the decode-seconds, and the median request ran 85–95 t/s. Per-request rates are summarised as medians, or time-weighted as total tokens over decode-busy seconds; `fn_gate.sh` reports per-request medians and the finish-reason histogram so a tail cannot hide.


## 23. An env flag being set is not evidence the code path runs (2026-09-22)

**What it looks like:** `EXL3_BATCH_VERIFY=1` in the resolved env keys, so the batched draft verify is on.

**What it is:** the eligibility veto compared `reqs_past_ids` aggregated over the sampler's *input* step stack — before `alt()` turned the frontend's unconditionally-appended neutral penalty steps into no-ops. Every request reported `reqs_past_ids=True` and took the serial per-token `.cpu()` accept loop (43 % of c4 wall time in a py-spy profile). A second veto on `device_logit_mask` covered every `min_tokens` request. The fix is `verifybatch-r1` ([R646](../bench/results/r646-verifybatch.md)).

**The fix for the measurement habit:** verify the path itself, not the flag — a profiler sample under its sync point (`ready.synchronize()` was absent from 27,498 samples), or a counter it increments.

## 24. A clock offset applied once at host boot does not stay applied (2026-09-25)

**What it looks like:** the memory clock offset is +4500 because the host's boot-time service applies it and the host has not rebooted.
**What it is:** the offset read 0 on both cards on 2026-09-25 after the service had applied it on 2026-09-02, with no reboot in between; it was recorded intact on 2026-09-03 and absent in this model's logs from 2026-09-19 on, and the cause is not determined. Every number measured in that span ran at the stock memory clock, about 1.7 % below +4500 in decode at 1 stream ([R726](../bench/results/r726-memoc.md)). The launcher now sets the offset before every engine boot and logs the readback, so every results directory records the state it was measured in.

## 25. An empty `stop` after a loop in the thinking shows the thoughts as the reply (2026-09-26, fixed for periods up to 400 tokens, up to 1,000 since 2026-09-27, up to 4,000 since 2026-09-28)

**What it looks like:** an agent client shows the model's thinking as its answer, often ending mid-sentence or repeating one paragraph; the stream carried reasoning deltas and no content, and `finish_reason` is `"stop"`.

**What it is:** the model looped inside its thinking and ExLlamaV3's loop detector ended the job (#3; TabbyAPI's `loop_detect_window`, default 800 tokens with 2 repetitions, catches periods up to 400 tokens). TabbyAPI reports that end as `"stop"`, so the response has reasoning and no content, the same shape as a model that ends its turn with an empty answer. A client that promotes the reasoning to the answer on an empty `stop` (Hermes Agent logs it as "Reasoning-only clean stop") shows the looping thoughts as the reply. On this configuration it happened on 2026-09-26 in two agent turns of 18,125 and 9,218 output tokens.

**Fix:** `loop-think-r3`, served since 2026-09-27 11:49 CEST: on a request that starts in the reasoning phase, TabbyAPI's chat collector detects the loop at the same window and forces a one-line message and `</think>` into the stream, and the model answers or calls a tool; the engine's detector moves to `(2W, 4)` on those requests ([R783](../bench/results/r783-loopthink.md), [CONFIG](CONFIG.md)). A request that sets `loop_detect_window` above 1,024 is not watched, because the engine's 2W window would not fit the 2,048-token output chunk, and keeps the old behaviour.

Since 2026-09-27 15:01 CEST the served image carries `loop-think-r4` ([R785](../bench/results/r785-promote-rebase-r3.md)): the collector runs a second detector, `(3L, L)` with L = 1.25 W (1,000 tokens at the default), which catches a reasoning loop with a period of up to 1,000 tokens once the model has generated three copies of it; prefilled copies do not count. It has not fired on the GPU yet: in R785 the model left every prefilled 688-token loop after one more copy. A loop with a period above 1,000 tokens is caught by no detector and runs until the model leaves it or reaches `max_tokens`, where `finish_reason` is `length`; from 401 to 1,000 tokens that was the behaviour until R785.

Since 2026-09-28 09:49 CEST the served image carries `loop-think-r5` ([R792](../bench/results/r792-promote-loopthink5.md)): two more collector rungs, (6000, 2000) and (12000, 4000), catch periods of up to 4,000 tokens after three copies at a rung's top period. [R791](../bench/results/r791-temp-incidence.md) had seen two such loops, with periods of about 1,200 and 3,700 tokens, run to `max_tokens` on ordinary agent turns at temperature 0.6. A period above 4,000 tokens, or a loop whose copies are not token-exact, is still caught by no detector.

## 26. A non-streamed chat completion has `"usage": null` (2026-09-27)

**What it looks like:** a probe reads `usage.completion_tokens` from a non-streamed `/v1/chat/completions` response and gets `None`; a check built on it fails, or passes without testing anything. R785's second run rolled back at its greedy gate on this: the gate required the long chat answer to exceed 2,048 tokens, and the count was `None` although the answer ran to its 5,000-token limit ([R785](../bench/results/r785-promote-rebase-r3.md)).

**What it is:** TabbyAPI returns `"usage": null` on non-streamed chat completions, on the previous image as on the new one; R783's and R784's chat greedy records carry the same null. Streamed requests that set `stream_options.include_usage` receive the counts in their last frame, so `replay_hermes_turn.py` and `agent_replay.py` were never affected. [`bench/chat_greedy.py`](../bench/chat_greedy.py) counts through `/v1/token/encode` over reasoning and content when usage is null and records `tokens_source: encode`; the think tags are not in that count.

## 27. An lm-eval samples file holds one line per question per filter (2026-09-27)

**What it looks like:** `samples_gsm8k_*.jsonl` of a 500-question run has 1,000 lines. A gate that requires 500 lines marks a complete run as incomplete: R785's third run passed every gate, read GSM8K 0.982, and rolled back as VOID on that count ([R785](../bench/results/r785-promote-rebase-r3.md)).

**What it is:** lm-eval writes one sample line per document per filter, and GSM8K has two filters, `strict-match` and `flexible-extract`. Count unique `doc_id`, or read `n-samples` in `results_*.json`. R728's GSM8K directory has the same shape; no earlier gate counted lines.

## 28. A per-cell decode rule over 24 prompts cannot resolve 1 % between two engines whose output differs (2026-09-27)

**What it looks like:** a candidate whose point estimates are all within ±1.6 % fails a rule that asks, per cell of 4 and 8 streams by code and prose, for a geometric mean of at least −1 % and a 95 % interval lower bound of at least −2 %: prose at 8 streams read −0.44 % with an interval of −2.30 to +2.40 % ([R784](../bench/results/r784-rebase-dev-r3.md)).

**What it is:** the interval is a bootstrap over the 24 prompts of the cell. When the two engines' greedy outputs differ, MTP acceptance moves per prompt (one prompt read +31 % because acceptance went from 0.18 to 0.74), and that sets the width. Replaying R784's per-prompt spread with a candidate of equal speed, the four cells all pass 20 to 27 % of the time; the A/A pair of the same run, read as candidate against reference, fails all four. More boots on the same prompts do not narrow the interval, and the bootstrap does not carry boot-to-boot variance. The review of R784 proposes gating on code and prose pooled (48 prompts) at 4 and at 8 streams, with the lower bound taken at the 1.25 % percentile. The prompts are also one-line requests: they measure decode at short context only. The same image at long context is [R786](../bench/results/r786-replay-abba.md)'s replay, 1.015× the previous image per stream (95 % interval 0.980 to 1.053).

## 29. An agent-replay per-stream rate reads about 10 % above the replay's steady state (2026-09-27)

**What it looks like:** `agent_replay.py --drain` against the served configuration reads 130 to 141 t/s per stream ([R786](../bench/results/r786-replay-abba.md)), while its requests read 116 to 126 t/s per stream before the 600 s deadline.

**What it is:** `--drain` lets the sessions still running at the deadline finish, and no new ones start. Those requests run as concurrency falls, at 195 to 257 t/s per stream, and hold 21 to 25 % of the scored tokens. Every replay per-stream figure in this repository ([R722](../bench/results/r728-promote-window-off.md), R728, R785, R786) includes that tail. The share varies by arm, and a faster arm starts more session instances before the deadline, which finish in the tail and raise its score further: R786's mean new-to-old ratio reads 1.029 as scored and 1.015 on the requests common to all arms. Compare replays on a fixed request set, or cut at the deadline, and quote the replay rate with the tail stated.

## 30. A single agent-replay run fails a fixed bar when one session falls into low MTP acceptance (2026-09-27)

**What it looks like:** a replay run reads 4 to 5 % below the previous one on the same image and flags, with the step time unchanged: R786's N2 read 127.9 t/s per stream against 137.6 and 140.6 on N1 and N3 ([R786](../bench/results/r786-replay-abba.md)).

**What it is:** at temperature 0.6 a session instance can enter a trajectory where MTP acceptance stays low for its remaining steps, because each reply becomes part of the next prompt: N2's instances (8, 0) at 0.40 over 19,591 tokens and (9, 1) at 0.56 over 15,650, the same instances at 0.87 to 0.96 in every other arm. It happens on both images: 4 of the 12 replay runs reviewed contain one (R722, R728, R785, R786). A gate on one run, such as R785's G7 at 0.98 × R722, then fails without an engine change. Print tokens per verify step and ms per verify step next to the rate, so a trajectory event shows as low tokens per step at normal step time, or score without the session instances whose acceptance is below 0.5.

## 31. A tool call cut by `max_tokens` arrives as `finish_reason: "tool_calls"` with no tool call (2026-09-28, fixed by loop-think r5)

**What it looks like:** a chat response with `finish_reason: "tool_calls"` and an empty `tool_calls` list, sometimes with a short preamble in `content`. A probe that reads `tool_calls` as "the model called a tool" counts it as an answer; [R791](../bench/results/r791-temp-incidence.md)'s first summary read no-answer 0 of 32 at temperature 1.0, where one row had reached 32,768 tokens inside a `write_file` call. An agent client takes its empty-tool-call path instead of its truncation recovery.

**What it is:** TabbyAPI `53da7919` sets `finish_reason` to `tool_calls` whenever tool-call text was generated (`if finish_reason and full_tool` when streaming, `if full_tool` when not), also when `_parse_tool_calls` returns nothing. A tool call cut by `max_tokens` never parses, so the `length` the backend reported is overwritten. The server log shows it: "max_tokens reached" and no "parsed … tool call" line for that request. `loop-think-r5`, served since R792, keeps `length` when tool text was generated, no call parsed and the backend's reason is `max_new_tokens`, and logs a WARNING whenever tool text does not parse; other ends with unparsed tool text (a stop token, a stop string, the engine's loop stop) still report `tool_calls` with no call, as in the base. Probes read `tool_calls` as an answer only when the list is not empty.

## 32. A gate probe that sends no temperature runs at the server's sampler fallback, so changing the fallback changes the gate (2026-09-28)

**What it looks like:** a promotion unit fails its loop-think gate on a change that cannot affect it: [R789](../bench/results/r789-promote-dropkeys.md)'s first run dropped three unused engine keys, reproduced the reference's greedy output byte for byte, and failed G5 because 1 of 2 forced-loop answers repeated the looped line.

**What it is:** the G5 probe sent no temperature. TabbyAPI applies the launcher's sampler preset to requests that omit a value (`force: false`), and the preset's temperature had moved from 0.6 to 1.0 the evening before; the reference run (R785c) had used 0.6. At 1.0 the unchanged configuration repeats the looped line in 7 of 16 forced rows (0 of 16 at 0.6), so a 2-row gate fails it 68 % of the time. Every gate probe sends its sampler explicitly, at the reference's values; a probe at the served fallback runs as a separate, report-only arm with its own n. The server log does not settle it: in R790 it printed `temperature: 0.6 (req)` for the explicit 0.6 and nothing for either an explicit 1.0 or the 1.0 fallback.

## 33. Moving a tokenizer `encode()` to a worker thread does not free the event loop (2026-09-29, fixed by tokenize-offloop r2)

**What it looks like:** every decoding stream stalls for about 53 ms each time a streamed 30k-token prompt arrives, before the server logs the prompt ([R805](../bench/results/r805-r808-tokenize-offloop.md)). Running the encode as `asyncio.to_thread(tokenizer.encode, prompt)` leaves the stall in place.

**What it is:** TabbyAPI encoded the prompt twice on the thread that runs `generator.iterate()`, once for the context-length check and once for the job, about 25 ms each. Hugging Face `tokenizers`' `Tokenizer.encode()` holds the GIL for the whole encode, so a worker thread running it still blocks the loop thread. `encode_batch([text])` runs the Rust encode with the GIL released and returns the same ids for one input with no padding or truncation configured. The served fix encodes once and, for long prompts, calls `encode_batch` on a one-thread worker, under a lock that also covers the tokenizer's shared `encode_special_tokens` flag ([`tokenize-offloop-r2`](../docker/overlays/tokenize-offloop-r2/README.md)). A test that runs the worker with `encode()` forced is the control that shows the loop gap measure sees the difference.

## 34. An executor hop costs a short prompt 1 to 2 decode steps of time to the first token (2026-09-29)

**What it looks like:** after moving every prompt encode to a worker thread, a ~98-token prompt arriving during 4-stream decode gets its first token 41.6 ms later (255.9 against 214.3 ms, [R806](../bench/results/r805-r808-tokenize-offloop.md)), although its encode takes well under 1 ms.

**What it is:** the loop picks the worker's result up only between two synchronous `iterate()` calls, about 17 ms apart at 4 streams, and the worker needs the GIL that `iterate()` mostly holds, so each hop waits 1 to 2 decode steps. A hop pays off only when the encode it moves costs more than that: the served round hops above 12,000 characters (about 4,200 tokens, about 3.8 ms inline) and encodes shorter prompts on the loop. The short-prompt time to the first token is then −2.9 ms [−6.2, +0.3] against no patch ([R808](../bench/results/r805-r808-tokenize-offloop.md)). A gate on the long prompt's stall alone does not see this cost.

## 35. A restore check that compares one-token outputs measures nothing (2026-09-29)

**What it looks like:** an NVMe tier crash-restart check passes with the restored output byte-identical to the warm and the cold outputs ([R809](../bench/results/r803-r810-prefill-merge.md)), and all three carry the hash `e3b0c44298fc1c14…`.

**What it is:** that hash is SHA-256 of the empty string. Each request generated one token, `<|im_end|>`, because the fill prompt's first token is a near-tie between `\n\n` and the end of the turn; the three outputs compared a single argmax, so the restored state's equality to the warm state was not measured. The check that measures it generates at least 32 tokens and compares the first token's top-5 logprobs: in R809t the restored output equals the warm one over 64 tokens with a first-token KL of 0.0. A capture hash of an empty text is never a valid identity; a gate that compares outputs fails closed on empty or one-token text.

## 36. A worker thread cannot write into tensors made under `torch.inference_mode` (2026-09-29, fixed in prefill-merge r1)

**What it looks like:** the first asynchronous recurrent stash fails in its worker thread with `Inplace update to inference tensor outside InferenceMode is not allowed`, and the error resurfaces on the generator thread when a reader waits for the stash ([R803](../bench/results/r803-r810-prefill-merge.md) step 0).

**What it is:** `torch.inference_mode` is thread-local. The generator runs under it, so the host tensors the stash allocates on the generator thread are inference tensors, as the served synchronous `.cpu()` stash's are; the worker thread runs outside the mode and may not copy into them. The fix passes `torch.is_inference_mode_enabled()` with each task and fills under `torch.inference_mode(<that value>)` on the worker, and allocates the reusable pinned staging slabs under `torch.inference_mode(False)`, so a slab first allocated inside the mode can still be written outside it. A CPU test that makes the stash under both modes fails with the same error without either half of the fix ([`prefill-merge-r1`](../docker/overlays/prefill-merge-r1/README.md)).

## 37. A change to how a prompt's prefill is split into forwards moves the recurrent state as much as the change under test: gate numerics against a partition null, not against zero (2026-09-29)

**What it looks like:** a prefill change that is exact by construction inside each layer still fails a bound on the recurrent stash: merged against served, 0.058 to 0.126 relative L2 over the whole stash, against a pre-registered bound of 0.05 ([R803](../bench/results/r803-r810-prefill-merge.md) step 0), and 34 of 80 greedy outputs diverge from the reference within 64 tokens.

**What it is:** every layer above the recurrent ones sees a forward of a different row count, and the kernels are chosen by row count, so the inputs to the recurrent layers move in the last bits and the difference grows with depth (0.0004 to 0.0034 at the first GDN layer, 0.02 to 0.05 by layers 12 to 14). The served configuration already produces the same class of variation: the same prompt prefilled at a 256-row chunk size moves the whole stash by 0.06 to 0.14, and the same prompt behind a cached prefix 512 tokens shorter diverges from the reference on 29 of 80 prompts. A gate that compares a partition-changing lever against bitwise identity, or against a bound set without a null, fails on variation that the served path has; the gate that separates a defect from rounding compares the candidate's divergence, confident flips and KL against a null arm that reaches the same prompts through a different served partition ([R809](../bench/results/r803-r810-prefill-merge.md)). A capture error shows at the first layer at full size; rounding starts small and grows with depth.

## 38. A log line that one of two code paths prints reads as the feature never running (2026-09-30)

**What it looks like:** none of the 16 served-image container logs of [R811](../bench/results/r811-r813-std-bench-merge.md) has `EXL3_STASH_ASYNC on: first asynchronous stash`, although the image serves with `EXL3_STASH_ASYNC=1`, so the asynchronous recurrent stash looks unused on ShareGPT and Spec-Bench, and a follow-up A/B was designed on the premise that it never ran.

**What it is:** the line is printed only by the stash of a prefill that does not merge. A merged forward stores its last-page stash through another path that also finishes on the worker thread and prints nothing, and the counter line that would show it is printed every 256 events, above the 157 to 166 merged prompts per cell. Every prompt over 257 tokens in both samples merges and none ends on a page boundary, so the path that prints never ran while the asynchronous path ran on every merged prompt. Before reading a missing log line as a feature that did not run, find every code path that does the work and check which of them print; a counter logged at the end of each cell settles it.

## 39. A step-time difference between two arms can be a speed-up one arm acquires mid-run at one request, not a cost per step (2026-09-30)

**What it looks like:** in [R811 and R812b](../bench/results/r811-r813-std-bench-merge.md) the median decode step at 1 and 2 streams is longer with `EXL3_PREFILL_MERGE=1` than without in every paired cell (0.8 to 3.8 % against the previous image, 2.5 to 3.1 % for the merge key alone), with equal τ and also on prompts too short to merge, which reads as a per-step cost of the merge, although nothing the merge adds runs in a decode step.

**What it is:** ordered by send time, the arms' steps are within about 2 % of each other until one request (ShareGPT request 45, a 786-token prompt; Spec-Bench request 29, 788 tokens). From that request on, every knob-off process's median step is 2.1 to 3.6 % shorter to the end of the run, and no merge-on process's is. The difference is a state the knob-off process enters, not work the merge-on process adds; its mechanism is not known. A whole-run median cannot tell these apart, and a second instrument whose prompts never reach the trigger (the decode curve's 106- and 118-token prompts) reads no difference under either explanation. Before attributing a step-time difference, split each run by send order and compare the arms before and after any change point.

## 40. A fresh process decodes about 3 % slower at 1 and 2 streams until it has run two new small row counts, and a benchmark's own warm-up can hide it (2026-09-30)

**What it looks like:** boots of one configuration agree on `fn_bench` at 1 stream, but a process that has served traffic decodes 2.3 to 3.4 % faster per step at 1 and 2 streams with byte-identical output; the prefill merge, which removes short prefill forwards, reads as a per-step cost at 1 and 2 streams (#39); and a launcher change that speeds up 1 and 2 streams reads as no gain at 8 streams.

**What it is:** every process starts at 10.10 to 10.15 ms per decode step (`fn_bench`, 1 stream, code prompt) and moves to 9.80 to 9.88 ms, where it stays for the life of the process, once the routed-MoE decode path for up to 32 rows (`run_bszN`) has run at two row counts the process has not run before. Those first calls launch the shared expert's kernels eagerly on its side stream; the state they leave is below the CUDA API and was not traced ([R815 to R820](../bench/results/r815-r820-fast-state.md)). Two short prefill forwards of up to 32 rows each, or the first episode of 4 or more concurrent streams, set it; the prefill merge folds away the short forwards that used to set it. A benchmark that warms each cell with c concurrent requests switches every arm before an 8-stream cell (measured with an 8-way warm-up) and none before a 2-stream cell (measured with a 2-way warm-up), so its 8-stream cells compare the fast state with itself and its 1- and 2-stream cells compare whatever state each arm is in; at 4 and 8 streams the step is the same in both states. The served launcher sends three raw completions of 20, 29 and 11 tokens at boot (`FASTWARM`, R818p). Before comparing two arms at 1 or 2 streams, record the short forwards and concurrency episodes each process has run since boot, and read `fn_bench` at 1 stream against the two levels above.
