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

## 8. A CPU-bound build next to a GPU measurement perturbs it (2026-09-16)

**Looks like:** stamina decay. A c4 soak that held 62–66 t/s per stream for five rounds dropped to 40.5 t/s at
round six and stayed there — exactly the shape of a thermal or fragmentation story.

**Is:** a native extension rebuild running on the same box. Compiling exllamav3's extension starts 487 compiler
processes and took the load average to 18.6; the decode path needs host CPU for sampling, launching and the PLE
gather, so the GPU starved while it was "idle" at 17 % utilisation. Same configuration, same server, no restart
between the clean and dirty rounds — the only variable was the build.

**Fix:** nothing runs on the box during a measurement except the measurement. That includes builds, and it is the
reason the A/B scripts in `bench/` take the GPU lock even when they only probe.

## 10. The two servers name the thinking channel differently, and one probe read only one name (2026-09-16)

**Looks like:** the vLLM 27B daily refusing concurrent deep-context work. Eight 38,283-token requests: one or two
complete, the rest come back with a `usage` block reporting 512 completion tokens, **no text at all** and no error,
each after ~16 s — while the daily's own log shows `200 OK` for every one of them. Read at face value that is "the
incumbent drops 6 of 8 deep-context requests", and it was twice reproduced before being questioned.

**Is:** two OpenAI-compatible servers naming the same thing differently, and an instrument that knew one name.

- **vLLM's** OpenAI server streams the model's thinking as `delta.reasoning` — captured raw:
  `data: {…"delta":{"reasoning":"The"}…}`, with the delta keys observed as `content, reasoning, role`. This repo's
  own daily documentation says "API serves reasoning as message.reasoning".
- **TabbyAPI** streams it as `delta.reasoning_content`.

`bench/probe.py` collected text from `delta.content` and `delta.reasoning_content` only. On the daily, a request
whose entire forced length went into thinking therefore looked like a request that returned nothing — and with
`min_tokens: 512` forcing exactly 512 tokens, `content` is *legitimately* empty for a request that never finished
thinking. The counts agree with that reading: 157–179 frames arrive per request, none of them content.

It now reads `content`, `reasoning_content` **and** `reasoning`, in the probe, the pair probe and the capability
gate. The re-measurement is `2026-09-16-r353-daily-admit2`.

**Second, dumber version of the same mistake, in the same hunt:** the raw-capture script passed a 250 KB JSON body
as a `curl` argv element and got `OSError: [Errno 7] Argument list too long`. Eight empty capture files, and the
`200 OK` lines the server logged belonged to other clients. Bodies now go through a file
(`--data-binary @body.json`), and a capture that writes zero bytes prints its curl exit status and stderr instead
of looking like a server that said nothing.

**The lesson, since this is the third instrument defect of the day:** a probe that reports "no text" must also
report *why* — frame shapes and field names seen, the raw bytes for at least one request, curl's exit status, or the
server's own log. Every one of today's false alarms was one unread field away from a wrong conclusion about an
engine, and the two that mattered were caught only because a second instrument or a raw capture disagreed.

## 9. Decode rate is content-dependent by ~2× on this checkpoint

Same config, same box, same day: `/completions` code at 176.3–187.1 t/s with 62–72 % draft acceptance, versus a
chat prose analysis at 94.1 t/s with 46 %. MTP acceptance tracks how predictable the continuation is, and prose
analysis is not predictable. A decode number without its kind is not comparable to another one.

## 11. A GPU unit stalled behind its own child, and the diagnostic said nobody held the lock (2026-09-16, fixed)

`r363-enable.sh` takes the GPU-exclusive lock and then runs `r362-pr337.sh` as its child. r362 took the lock too. A
child that **re-opens the lock path** gets a *second* lock on the same file, so it waited for the parent — which was
waiting for it. Nothing ran, the GPU sat idle, and seven other units queued behind the pair.

The diagnostic was wrong twice over, and the wrongness is the lesson:

1. `flock` holders are **not** reported in `/proc/PID/fdinfo` — that field is for POSIX record locks. The right source
   is `/proc/locks`, whose inode field is **decimal**; grepping a hex-translated inode returns nothing, which I read as
   "nobody holds it" and briefly believed the kernel had lost a lock.
2. "No process is building or probing" is not evidence that no unit is running. Both processes were alive and idle by
   construction: each was blocked on the other.

Fixed in `flan/lib/gpu-queue.sh`: `gpu_lock()` reuses fd 9 when it is already the lock file, so an inherited descriptor
shares the holder's open file description and `flock` succeeds at once. r362 calls it. The general rule for this host:
**a script that the lock-holder invokes must not take the lock itself** — inline `exec 9>` + `flock` is only safe at
the top level, which is what every unit except this pair is.

The queue was also rebuilt as a single chain (`bench/r370-chain.sh`) rather than eight units sharing one flock: the
order in which waiters acquire a flock is not defined, so "queued" never meant "ordered".

## 12. A validation that passes on an empty file (2026-09-16)

Copying a script to the host and checking it with `bash -n` reported success on a **0-byte file**: an empty script is
valid bash. The transfer had produced an empty file, the check confirmed nothing, and the unit was launched, exited
immediately with status 0, and read as "ran successfully" — the third time this session that a passing check measured
the absence of a thing rather than the thing.

What the check should have been, and now is: `wc -c`, plus a grep for a string that must be present (`greedy_same`),
plus comparing hashes of both copies. Size and content are different claims; a syntax checker answers neither.

The general rule this session keeps re-learning: **a check must be able to fail.** `cmp` with a glob that matches one
file, `bash -n` on an empty file, a probe that reads one channel of four, a build whose verification step imports
without its library, a lock holder read from the wrong field of `/proc/locks` — each looked like a green light and
each was measuring nothing.

**It happened again, worse, an hour later.** Pushing four updated scripts to the host used a loop with no input
redirection —

    for f in a b c; do ssh flan "sudo cat > /srv/qwen5090/$f && chmod +x /srv/qwen5090/$f"; done

— so `cat >` truncated each target and read nothing. **Three scripts became 0-byte files**, the check I ran was
`bash -n` (which an empty script passes), and the chain then executed them as `### DONE r366-ourkernel in 0s`,
`r364-hotvocab in 0s`, `r367-slots in 0s`. Three experiments completed in the log and did no work at all, and
"completed in 0s" was the only signal — which is easy to read as "fast" rather than "empty".

The fix is a check that can fail, applied to every copy: byte count, a string that must be present, and matching
hashes on both ends. `r372-chain2.sh` now refuses to start if any step it was asked to run is under 500 bytes, so the
condition cannot recur silently. The lesson generalises past this host: **a step that reports success without doing
work is indistinguishable from a step that did the work quickly, unless something counts.**
