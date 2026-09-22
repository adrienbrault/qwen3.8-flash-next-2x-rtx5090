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

