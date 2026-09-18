# MEASUREMENTS

Every number here was measured on the `flan` box, on the date given, with the served configuration as it stood
that day, and the raw records are named next to it. Decode rates are steady-state (first token to last) unless
the column says wall.

## The served configuration these numbers describe (since 2026-09-16)

`qwen3.8-flash-next-exl3-3.05bpw` on TabbyAPI + ExLlamaV3 v1.5.0, port 8022, image `tabbyapi:qsa-cid-pr337`,
262,144-token window and cache, 8-bit KV, 8 slots, layer-split across both cards (TP is not implemented for this
architecture), MTP draft depth 3 with concurrency-indexed depth `[[2, 3], [8, 1]]`, vision on, `qwen3_coder` tools,
sampler preset `qwen38_thinking` (T=0.6, top_k 20, top_p 0.95 as **fallbacks**). See `docs/CONFIG.md` for why each
value.

**Identity of the served configuration, and the two fingerprints that describe it.** They are produced by different
tools and are NOT interchangeable — comparing one against the other is the same error as comparing throughput from two
instruments, and I made it once:

| fingerprint | method | value |
| --- | --- | --- |
| `r363`/`r340` gate | sha256 of the single-file greedy capture, first 16 | `750e1459e177c47e` (1,989 bytes) |
| `greedy_hash()` (`lib/greedy-compare.sh`) | directory hash over the 4-prompt capture: count + sorted (relative name, content) pairs | `8179222fec8df3b8` |

An earlier `18e30f17883a38eb` from this function is superseded: the function was fixed to include the file count,
to list files portably and to refuse the empty-input digest (see `docs/GOTCHAS.md` #13), and every hash it produced
before that is superseded. The directory fingerprint is the one restores check (`bench/r373-restore.sh`), and it
prints the measured value when no reference is pinned rather than judging against a number from another method.
`8179222fec8df3b8` is also the value the #290 re-captures and the #246 feature-off arm produced, which is consistent
with those variants being output-identical to the served configuration, as their own gates showed.

Both fingerprints are recorded so a future A/B can establish that it started from the same configuration. The primary
identity is **behavioural**: greedy output on the forced-length probe is `750e1459e177c47e` (1,989 bytes), and the
probe scripts compare against it. The generated config `/srv/qwen5090/flashnext-config.yml` (mounted read-only at
the container's `/app/config.yml`, byte-identical inside and out) was `sha256 12252e838eaa…` at that moment, but
**that hash moves when a comment in the launcher's heredoc moves** — the file carries its own explanation inline.
Verified 2026-09-16: after correcting six comments, the rendered config differed from the served one in comment lines
only, with no differing setting. Read the hash as a provenance marker, not as the contract; grep the file for the
keys when it matters.

## Decode, code, forced length — `bench/probe.py`, results `2026-09-16-r339-longgen`

4,096 tokens forced per request with `min_tokens`, greedy, chat endpoint, `finish_reason: length` on every row.
`per stream` is the median steady-state rate; `aggregate` is total tokens over the round's wall.

| concurrency | per stream (t/s) | aggregate (t/s) | TTFT (s) |
| --- | --- | --- | --- |
| 1 | **217.6** | 217.6 | 0.21 |
| 2 | 121.9 | 244 | 0.26 |
| 4 | 65.5–70.3 | ~265 | 0.49–0.57 |
| 8 | *see below* | | |

The c1 figure is the one interactive use sees: **217.6 t/s steady-state on code**, from a 3.05 bpw checkpoint that
fits two cards with a 262k window.

Aggregate scaling is the weakness: 1→4 buys 1.22×, and the whole ladder buys 1.60× from c1 to c8 (256-token
requests, below). That is the cost of layer splitting: with `tensor_parallel: false` the two cards take turns, so
each card is busy only while its own layers run (measured 44–47 % utilisation, 236/225 W against 600/575 W limits,
during a request).

## Concurrency, short generations — `oai_conc.py`, results `2026-09-16-r339-honest-conc`

256 forced-by-prompt-length tokens, greedy, `temperature 0`, streaming, 2 runs. This is the same instrument R219
and R331 used, but the `usage` figures are now correct: the engine under-reported any generation past ~2048 tokens
by ~5× until the R338 image patch, and these rows never crossed that boundary, so they are unaffected either way.

| concurrency | aggregate (t/s) | per stream (t/s) | TTFT (s) |
| --- | --- | --- | --- |
| 1 | 177.6 | 177.6 | 0.103 |
| 2 | 232.8 | 129.0 | 0.18 |
| 4 | 252.7 | 70.2 | 0.33 |
| 8 | 283.5 | 42.9 | 0.51 |

## Long-context retrieval — `bench/needle.py`, results `2026-09-16-r339-gates`

One unique passphrase planted at five positions (8 %, 30 %, 55 %, 80 %, 96 %) of a deterministic document, asked
for by name. Requested depths are labels: the filler's estimate runs ~28 % high, so the actual `prompt_tokens` the
server saw is quoted.

| requested depth | prompt tokens seen | retrieved |
| --- | --- | --- |
| 32,768 | 26,518 | **5 / 5** |
| 131,072 | 105,680 | **5 / 5** |
| 196,608 | 158,452 | **5 / 5** |

A pass at every planted position, not only near the end, is what makes this a gate rather than a demonstration.
`bench/needle.py` speaks the OpenAI chat API, so it runs unchanged against any OpenAI-compatible server.

## Decode and TTFT against prompt depth — `bench/probe.py`, results `2026-09-16-r343-depth`

Code, 1,024 forced tokens, c1, greedy. The rungs are labels: the filler's token estimate runs ~28 % high, so the
`prompt tokens seen` column is what the server actually received.

| requested depth | prompt tokens seen | decode (t/s) | TTFT (s) |
| --- | --- | --- | --- |
| 0 | 101 | 183.4 / 189.0 | 0.15 |
| 30,000 | 38,266 | 155.5 / 158.9 | 0.74 cold, **0.24 warm** |
| 120,000 | 152,761 | 150.1 / 155.6 | 24.0 cold, **0.43 warm** |

Decode falls 18 % from a 101-token prompt to a 152,761-token one. The second TTFT column is the larger effect: a
152k prompt costs 24 s cold and **0.43 s on a repeat**, a 56× reduction from the paged prefix cache. An agent that
resends a long conversation every step pays the warm figure, not the cold one.

At c4 the same rungs read 50.3–53.7 t/s per stream at 38,283 prompt tokens (TTFT 0.73–1.01 s) and 46.9–49.9 t/s at
152,778 (TTFT 1.62–1.67 s). **Those c4 rows share their filler prefix** — only a short suffix differs between the
four requests — so they measure shared-prefix concurrency, which is what a fan-out of agents on one harness
actually sends, not four independent contexts. Independent contexts are measured in `2026-09-16-r345-pool`.

## Admission of deep contexts — results `2026-09-16-r339-gates` and `2026-09-16-r345-pool`

Eight concurrent requests, each carrying 38,283 prompt tokens and 512 forced output tokens.

| | value |
| --- | --- |
| admitted / completed | **8 / 8** |
| TTFT | 59.4–62.4 s (all eight within 3 s of each other) |
| decode | 28.5–33.6 t/s per stream, ~256 t/s aggregate |

All eight arrived together and prefilled concurrently, so the ~60 s is what it costs to put 8 × 38k tokens through
this box at once — an effective prefill of ~5,100 t/s aggregate. **Caveat that the first run did not establish:**
those eight requests shared one 38k-token filler and differed only in a short suffix, so this measures the
realistic same-harness fan-out, not independent context footprint against the 262,144-token pool. `r345-pool`
repeats it with a per-request RNG stream so the requests share no token sequence, and captures the server's own
`cached_tokens` figures as evidence rather than inferring sharing from timings.

## Decode at the requeue boundary — 2,048 forced tokens, results `2026-09-16-r339-gates`

2,048 is TabbyAPI's requeue budget on this config (`chunk_size 2048`, `output_chunking: true`), so these rows sit
exactly at the boundary where the engine's own token accounting used to break.

| kind | concurrency | decode per stream (t/s) | aggregate (t/s) | TTFT (s) |
| --- | --- | --- | --- | --- |
| code | 1 | 210.7–210.8 | 207.7–207.8 | 0.145 |
| code | 4 | 64.4 | 252.4 | 0.511 |
| prose | 1 | 162.4–167.2 | 160.6–165.3 | 0.14–0.15 |

## Concurrency-indexed draft depth: the A/B — results `2026-09-16-r340-ci-depth`

Three arms, one variable each, 2,048 forced code tokens, greedy, chat endpoint. Control is the served baseline;
parity is the patched engine with the policy unset; treatment is the patched engine with
`draft_num_tokens_by_batch: [[2, 3], [8, 1]]` — depth 3 up to two decoding jobs, depth 1 above.

| arm | c1 decode | c4 decode/stream | c4 aggregate | c8 decode/stream | c8 aggregate |
| --- | --- | --- | --- | --- | --- |
| control (unpatched) | 209.2–213.2 | 64.4–64.8 | 252–258 † | 40.8–42.6 | 316–320 |
| parity (patched, no policy) | 209.2–211.6 | 64.3–65.7 | 252–258 | 40.8–42.6 | 316–320 |
| **treatment (policy on)** | 209.2–214.1 | **87.1–89.4** | **338–347** | 39.6 | 311 |

† the control rows carry two attempts' records in the same file (an aborted first run and this one), so
`bench/summarize.py` prints AMBIG rather than a sum of both; the decode figures are per request and unaffected.

**+35 % aggregate at c4** from one load-time policy, with no change at c1 (where it deliberately keeps depth 3) and
no gain at c8. The correctness gate passed on all three arms: the greedy capture — reasoning and content together,
2,003 bytes — is byte-identical (`sha256` `95726ace17d5…`) across control, parity and treatment, so the patch alone
changes nothing and enabling the policy does not change what the model says.

Two requests across the arms stopped before the forced length (`finish_reason: stop` — the engine's loop detector);
they are flagged and excluded from the rates above.

## QSA sparse multi-job: the A/B — results `2026-09-16-r341-qsa`

Both arms built from the same CUDA devel base, same pip resolution, same native rebuild — the only difference is
`APPLY_QSA`. Deep context (152,761 prompt tokens), 1,024 forced code tokens, greedy.

| arm | c2 decode/stream | c2 aggregate | c4 decode/stream | c4 aggregate |
| --- | --- | --- | --- | --- |
| control (unpatched) | 99.2 | 27.4 | 51.7 | 191.6 |
| **treatment (QSA multi-job)** | **125.8** | 28.2 | **72.5** | **257.9** |

**+27 % at c2 and +40 % per stream at c4**, at the depth where the captured QSA path used to fall back to eager for
bsz>1 — the case the patch exists for. The c2 aggregate is unchanged because those two requests queue on a 262k
pool against 2 × 152,761 tokens of context; the per-stream figures are the ones to read.

The correctness gate passed on all three checks at 74,796-token contexts with bsz=2: the two concurrent greedy
outputs match within each arm, and control == treatment byte for byte (`sha256` `2e0c2a563342…`, 1,365 bytes). An
attention patch that changes what the model says is not a win, and this one does not.

## The pool, measured with independent contexts — results `2026-09-16-r345-pool`

Three arms; the third failed, and its failure is recorded below. Requested depths are labels: `probe.py` records the
`prompt_tokens` the server actually saw, and its filler estimate runs high.

| arm | prompts actually sent | jobs | admitted | TTFT (s) | decode/stream |
| --- | --- | --- | --- | --- | --- |
| shared prefix | 38,283 tokens each (306k total, page-shared) | 8 | **8/8** | 63.6 | 33.2 |
| **unique contexts** | **78,233–79,139 tokens each (~628k total, no page sharing)** | 8 | **8/8** | 43.1–82.2 (median 43.8) | 37.0 |
| unique, deeper | rejected before admission | 4 | 0/4 | — | — |

So eight agents carrying **~628k tokens of mutually unrelated context** — more than twice the 262,144-token pool —
are all admitted and all complete, with the surplus queued rather than refused: the first token arrives in 43 s for
some jobs and 82 s for others, which is the queue draining. The pool schedules; it does not reject.

The third arm is the guard, not a failure: the server answered `400 Prompt length 315,253 exceeds the available
context size of 262,144 tokens` for each request. **That run had a bug in the instrument** — `--unique` prepended
the per-request passage to the shared filler instead of replacing it, so a "120k" request carried 315k tokens. It
is fixed in `bench/probe.py`; the 400 is the server behaving correctly, and it is recorded because a request that
does not fit is refused with a reason rather than silently truncated.

`r345` also captured the server's own log into the results directory, so the cache and length figures above come
from the server rather than from timings.

## The promoted configuration, validated on the real request — results `2026-09-16-r356-promoted`

`IMG=tabbyapi:qsa-cid DRAFT_POLICY='[[2, 3], [8, 1]]'` booted as the served configuration, then the **exact agent
request that failed** — the DSH payload rebuilt from the session archive, 36 tool schemas and a 10,481-token
prompt — and the retrieval gate at depth.

| check | result on the promoted configuration |
| --- | --- |
| the original failing agent request | `finish_reason=tool_calls`, tool calls parsed: **`skill`, `write`** — it plans, loads a skill and writes the file |
| reasoning channel | 49,417 chars, 0.05 % non-Latin (the pre-fix run of the same request: 19,477 chars of multilingual salad in `content`) |
| needle at 105,680 prompt tokens | **5/5 retrieved** |

So the configuration this repository recommends is the configuration that has been run against the workload, not
only against the instrument. The baseline was restored afterwards: promoting it to the served default is the
operator's call.

## PR #337, the layer-split device context — results `2026-09-16-r362-pr337`

The patch keeps the process-wide CUDA current device on the module's device across `forward_ls`/`prefill_ls`. Its
author found the bug via an out-of-tree kernel whose fault was misattributed to autotune; stock wrappers self-guard,
so the expectation stated before the run ranged from no effect to "removes accidental P2P traffic". Because it moves device placement,
**its correctness gate ran first**: greedy output, 1,989 bytes, byte-identical between arms (`greedy-baseline.txt`
vs `greedy-pr337.txt`).

Aggregate decode, both arms as served (`tabbyapi:53da7919-rqcount` with and without the patch), one run each:

| arm | c1 | c4 | c8 | c4 at 152k-token prompts |
| --- | --- | --- | --- | --- |
| baseline | 208.2 | 254.8 | 319.5 | 181.7 |
| **PR #337** | 208.9 | 256.0 | 318.7 | **207.5** |
| delta | +0.3 % | +0.5 % | −0.3 % | **+14.2 %** |

Six columns are flat within run-to-run noise. The seventh is the deep-context arm, and it is the only place the
patch's mechanism predicts an effect: 152k-token prompts are where accidental cross-card traffic in a layer-split
forward would show. **One run per arm, so +14 % is consistent with the mechanism and not established.** What the run
establishes is no regression at any measured shape, a deep-context gain where the mechanism predicts one, and
byte-identical output. The patch is carried into the served image on those grounds rather than for the +14 %.


## GSM8K at n=1319 — results `2026-09-16-r368-gsm8k-1319`

The full test split, with the same harness parameters as the n=200 arm recorded below, at 6.6× the sample. It
tightens the interval from ±0.019 to **±0.0077** and lands at 0.9158 flexible-extract / 0.9151 strict-match, inside
the n=200 reading's own interval: the earlier figure was under-powered rather than wrong. This is the GSM8K figure to
quote for the served configuration.

Measured on this seat's own configuration, so it is the served article's number rather than a proxy for it. One check
before using the run: lm-eval reads `message.content`, and this seat serves `reasoning: true`, so the score is only
meaningful if answers land in `content` rather than in the thinking channel. Verified against the live server: a
GSM8K item returned 126 chars of reasoning and 131 chars of content ending "Answer: 72 clips". <!-- prose-ok: quoted model output -->

## Upstream #246 changes numerics for a prefill gain within noise — results `2026-09-16-r365-kernels`

Its assessment said the claim is prefill-only and that ordinary c1/c4/c8 decode cannot use it, so the columns to read
were TTFT at depth and output identity. Both are now measured, and they point the same way.

| arm | output vs control | TTFT ctx 0 | TTFT ctx 30k |
| --- | --- | --- | --- |
| control (unpatched) | — | 0.146 s | 1.428 s |
| patched, feature off | **byte-identical** | 0.146 s | — |
| patched, `EXL3_MOE_ROUTE_PACKED=1` | **all six responses differ** | 0.146 s | 1.403 s (−1.8 %) |

Two readings. First, the disabled path is identical to control, which is what makes the arms a valid comparison.
Second, the enabled path **reorders numerics** and buys, at most, 1.8 % TTFT at 30k in a single run — inside
run-to-run noise on this box, and against outputs that this repository fingerprints. Rejected: a feature that changes
what the model says needs a much larger gain than that to be worth re-validating quality behind it.

The in-run gate line said `FAIL/NOT-RUN control vs patched-off` and was wrong — the same glob-`cmp` defect that
reported a false negative for #290, in a script copy that predated the fix. The gate was recomputed from the captures
with `lib/greedy-compare.sh`; the numbers above are the recomputation.

## Upstream #290's memory fix is output-neutral — results `2026-09-16-r365-kernels`, gate in `r371-290-identity`

Three arms, one native rebuild each from the same v1.5.0 source with a different patch set applied: `unpatched`,
`+OOB fix`, `+OOB fix +reduction`. The question its assessment left open was whether a memory-safety fix in an API
this model does not route through changes anything; the gate is therefore **output identity**, not speed.

| arm | extension sha256 (first 16) | captured | dirhash |
| --- | --- | --- | --- |
| unpatched | `ec7270959395df30` | 6 responses | `18e30f17883a38eb` |
| +fix | `294407485b5b792c` | 6 responses | `18e30f17883a38eb` |
| +fix+reduction | `fb0f3690b357c436` | 6 responses | `18e30f17883a38eb` |

**All three identical.** The three extension hashes differ, which is the precondition that makes the identity
mean anything: three arms sharing one binary would have been an identity result about nothing. Aggregate decode,
fix vs unpatched: c1 220.5 vs 222.3, c4 381.8 vs 381.8, c8 319.3 vs 319.7, i.e. within run-to-run noise. So the fix
can be adopted on correctness grounds with no behavioural or throughput cost.

**Two caveats, both about what these numbers are not.** First, the arms are comparable *to each other* and not to the
served figures: `kernel290:*` is a CUDA-devel image built from v1.5.0 plus the port, which is a different image from
the served `tabbyapi:qsa-cid-pr337`, and the c4 column here (381.8) sits above the served configuration's 250 without
a policy and 338 with one — a cross-image comparison would be a category error. Second, this run's first attempt at
the gate reported `FAIL/NOT-RUN` for both arms and was **wrong**: the control arm captured one response and the
treatments six, because the control ran before a capture bug was fixed, and the comparison then used `cmp` on
mismatched file sets. The gate was recomputed from paired captures rather than read from the log line — see GOTCHAS 11
and `bench/r371-290-identity.sh`.

## #303 MTP hot vocabulary is inapplicable on this box, by construction — `2026-09-16-r377-hotvocab-on`

The port was rebased, built, and booted, and the engine refused it with its own guard:

```
File "exllamav3/architecture/qwen4_exp_mtp.py", line 178, in attach_to
    raise ValueError("MTP hot vocabulary requires target and draft on the same single GPU")
```

The served configuration is a **two-card layer split** (`gpu_split: [30, 30]`, `tensor_parallel: false`), and the feature
requires the target model and the MTP draft head on one GPU. So this is not a build problem, a patch problem, or an
experiment that needs better parameters: **the lever does not exist for this serving configuration.** The only
configuration in which it could be measured is single-GPU serving, which leaves the second card idle.

Two consequences of this arm. The first is that the failure surfaced as `docker run FAILED` and nothing
else for three separate attempts, because the launcher's run command ended in `>/dev/null 2>&1`; the error was captured
and printed only after that was fixed, and it named the cause immediately. The second is that the arm's container then
crash-looped under `--restart unless-stopped` while the launcher waited for readiness, and the runner's TERM trap
restored nothing — leaving the box on an experiment image until a restore was run by hand. That is the same lifecycle
gap the review's F3 describes, and it is the reason the last two experiments this session both ended with an explicit
restore rather than with the arm's own cleanup.

## Our own 32-row MoE decode envelope: correct, and no effect — results `2026-09-16-r366-ourkernel`

The hypothesis was that c4/c8 decode falls into a slow fallback path in the MoE dispatch, and that an envelope written
for exactly these shapes would recover it. The change is dispatch-only, so the gate is output identity — **PASS,
byte-identical** — and then the columns:

| | c1 | c2 | c4 | c8 |
| --- | --- | --- | --- | --- |
| control (`tabbyapi:qsa-devel`) | 0.152 / 0.146 / 0.146 | ~0.255 | ~0.48–0.58 | 0.485–1.246 |
| candidate (`tabbyapi:ourkernel`) | 0.153 / 0.145 / 0.145 | ~0.255 | ~0.48–0.58 | 0.485–1.238 |

Three runs per arm: TTFT agrees to within 0.002 s at c1/c2/c4 and 0.01 s at c8. Aggregate decode reads 248–252 at c4
and 288–294 at c8 on both arms. **The envelope is correct and buys nothing measurable**, which moves the c4/c8 question
away from dispatch and back to where the earlier analysis put it: the layer-split duty cycle.

### The measurement that came out of the control arm

24 c8 TTFT samples per arm give a spread the single-run comparisons never showed:

```
control    min 0.485  p50 0.844  max 1.246  ->  2.57x within ONE configuration
candidate  min 0.485  p50 0.858  max 1.238  ->  2.55x within ONE configuration
```

**c8 TTFT varies by 2.6x inside a single configuration**, so any single-run c8 TTFT comparison smaller than that is
noise. That includes the #246 result recorded above: control 1.249 s against feature-on 0.635 s is a ratio of 1.97,
*inside* the within-configuration spread. "The feature halves c8 TTFT" is therefore not supported; the prefill
columns in the same run moved 1.3-1.6 %, which is the size of the effect #246 produced here.

This is a metric-defect finding rather than an engine finding: **TTFT at high concurrency is dominated by queueing, so
it needs many runs or a median-and-spread report, never one sample per arm.** The r375 arms were rewritten around that
before they ran: two arms instead of three (feature-off is byte-identical to control, so it *is* the control), 512
forced tokens because the reading is TTFT, and four runs per concurrency with the spread printed.

## The slot ladder and 12-agent admission — results `2026-09-16-r367-slots`

`max_batch_size` is the last untested *config* lever on the weakest axis. TabbyAPI derives 4 slots for a recurrent
model and this seat is served at 8, so the arms below measure whether more slots admit more real work.

**More slots are not reachable here at all.** Both arms above the served value fail to boot:

```
max_batch_size 12 -> RuntimeError: Insufficient VRAM in split for model and cache
max_batch_size 16 -> RuntimeError: Insufficient VRAM in split for model and cache
```

That is the same wall the 393,216-token cache hits, and it has the same cause: the manual layer split has to hold the
weights *and* the whole cache, and the recurrent-state planes grow with the slot count. The ladder therefore does not
rank 8 against 12 or 16: **8 is the maximum this cache size supports**, and any future argument for more slots has to
be an argument for a smaller cache.

With the served 8 slots, offering more concurrency buys nothing (the probe's own aggregate, 512 forced tokens, two runs
each):

| offered concurrency | aggregate t/s | TTFT |
| --- | --- | --- |
| c1 | 163.7 / 167.5 | 0.15 s |
| c4 | 313.0 / 315.1 | 0.46 s |
| c8 | 301.3 / 320.7 | 0.74–0.79 s |
| c12 | 309.6 / 319.0 | 0.97–1.01 s |
| c16 | 314.7 / 313.5 | **7.21 s** |
| 12 agents at ~25.6k prompt tokens | **97.7** | **42.99 s** |

Two readings. **The aggregate is flat from c4 to c16** — roughly 310–320 t/s, which is the ceiling of eight slots at
about 40 t/s each, not a limit the offered concurrency can push. Above 8 the requests queue rather than parallelise,
and c16 pays for it in TTFT: 7.2 s, two waves. **And real agent contexts collapse it**: twelve jobs at ~25k tokens
each return 97.7 t/s aggregate with a 43-second TTFT, which is what "admission" means here — the box does not host
twelve deep agents, it queues them.

The c1 column reads 163.7 against the 207 t/s this repository quotes elsewhere; that is `docs/GOTCHAS.md` #9
(content and generation-length dependence), not a regression: 512 forced tokens of code versus 2,048.

## Quality: GSM8K as served — `bench/r355-fn-gsm8k.sh`, results `2026-09-16-r355-fn-gsm8k`

GSM8K, lm-eval, as served: `gsm8k`, 5-shot, `--apply_chat_template`, temperature 0, `max_gen_toks 8192`, limit 200,
`num_concurrent 4`, thinking on.

| arm | exact_match (flexible-extract) | conditions |
| --- | --- | --- |
| **as served here** | **0.925** (strict-match 0.920, stderr ±0.019) | this run, thinking on by configuration |
| **as served here, n=1319** | **0.9158** flexible-extract, **0.9151** strict-match, stderr **±0.0077** | `2026-09-16-r368-gsm8k-1319`, same parameters at 6.6× the sample |

The n=1319 row is the figure to quote; the n=200 row is the same measurement at a wider interval. The vllm-exl3
route's GSM8K figure on the same checkpoint is in the vLLM-route section below.

Thinking is on in this arm and can be seen doing so: a hand-checked item returned 126 chars of
`reasoning_content` plus the answer, and the per-request completion lengths across the 201 requests were
min 13 / p50 232 / p90 449 / max 1,660 tokens. This checkpoint reasons **briefly** — that is its character, not a
template flag left off.

## GPU duty cycle — measured under a live agentic load

`nvidia-smi` sampled at 1 Hz for 20 s while eight SWE-bench agents were running against the seat:

| | mean | min | max | mean power | limit |
| --- | --- | --- | --- | --- | --- |
| GPU 0 | **45 %** | 22 % | 84 % | 226 W | 600 W |
| GPU 1 | **38 %** | 20 % | 53 % | 195 W | 575 W |

This is the layer-split signature, and it is *not* a tuning miss: with `tensor_parallel: false` and
`gpu_split: [30, 30]` the two cards take turns over their own layers, so each is busy only while its own layers
execute. Under a real eight-agent load it is lower than the 44–47 % measured during a single request, because agent
turns are short and bursty: the cards are idle waiting for the next request as well as for each other.

What moves it and what does not, from this session's own measurements:

| lever | effect on the duty cycle |
| --- | --- |
| QSA multi-job | none directly; +27 %/+40 % throughput at deep-context concurrency, so more work in the same busy windows |
| concurrency-indexed draft depth | none directly; +35 % aggregate at c4 for the same reason |
| host KV tier | none; measured flat |
| expert parallelism | would put both cards on every layer — assessed as needing engine work, not a patch (`supports_tp` is still False in this tree) |
| tensor parallelism | forbidden for this architecture in this engine |

## SWE-bench Verified, four subsets — results `2026-09-16-r359-swebench`

Agentic coding, which neither GSM8K nor tool-eval measures. Harness: mini-SWE-agent 2.4.6, the builtin
`benchmarks/swebench.yaml` (the leaderboard's bash-only setting, step_limit 250), `--subset verified --split test`,
scored by the official swebench harness in the official task images. The sampler reaches the server through this
seat's preset, because mini-swe sends only `max_tokens`.

**How the subsets were chosen, and what that forbids.** Each subset is drawn from a prior 500-instance scored run on
this box and stratified by that run's per-instance outcomes; none of them is a random sample of SWE-bench Verified.
The rates below therefore describe these instances only, and must not be read against a full-run rate. **A first
attempt read the "first 10" from `preds.json`, which is ordered by completion rather than by dataset order, and would
have executed the wrong instances.** The list each run executed is recorded in its results directory.

**Result — `2026-09-16-r359-swebench-10`, the dataset's first ten instances (all astropy):** resolved **10 of 10**.
All ten trajectories ended `Submitted` with a non-empty patch, at 45–163 steps; the 250-step limit was never reached,
so nothing was truncated by the harness. n=10 and one repository, so the subset cannot resolve a few points.

**Result — `2026-09-16-r360-swebench-strat`, 18 instances across six repositories, officially scored:** resolved
**17/18**. All 18 trajectories ended `Submitted` with a non-empty patch, 34–143 steps.

| repo | resolved |
| --- | --- |
| django | 2/3 |
| matplotlib | 3/3 |
| pydata | 3/3 |
| scikit-learn | 3/3 |
| sphinx-doc | 3/3 |
| sympy | 3/3 |
| **total** | **17/18** |

**Result — `2026-09-16-r361-swebench-failed`, ten instances selected because the prior scored run failed them after
submitting a patch** (drawn from its 113 unresolved, filtered to the 109 that ended `Submitted`, so the selection is
on capability failures rather than budget ones)**:** resolved **9/10**. All ten trajectories ended `Submitted`,
0 errors, 0 empty patches.

GSM8K, tool-eval and SWE-bench measure three different things on this stack; only SWE-bench measures agentic coding.

### All four subsets, de-duplicated

The subsets are **not disjoint**, and adding their totals gives the wrong count. Four runs cover 68 instance-runs but
only **49 unique instances**: the 30-instance subset re-ran all 18 of the stratified subset and one of the ten
selected from prior failures.

| subset | n | resolved | configuration |
| --- | --- | --- | --- |
| dataset's first ten (astropy) | 10 | **10** | pre-enablement |
| stratified, six repositories | 18 | **17** | pre-enablement |
| ten selected from prior failures | 10 | **9** | pre-enablement |
| 30-instance stratified, six repos | 30 | **28** | enabled (`tabbyapi:qsa-cid` + policy) |
| **unique instances** | **49** | **46** | mixed, see below |

**46 of 49 unique instances resolved.** Every subset was selected on a prior run's outcomes, so this is a descriptive
rate on these 49 instances and not an estimate of a rate on SWE-bench Verified. An earlier version of this section
attached a p-value to it: a sign test assumes selection independent of outcome, which does not hold here. An
inferential claim needs a held-out subset selected independently of any engine's results, which has not been run.

**The 19 repeated instances are the control for the enablement.** The stratified 18 were run before the enablement
and again inside the 30, and the one overlap with the failure-selected ten likewise: resolved 17 → 17 and 1 → 1,
**zero outcomes changed**. The configuration now served was therefore measured, rather than assumed, to be
quality-neutral on those 19 instances, independent of the byte-identity gates. That is weaker than a general
equivalence claim and is all it shows.

## The host KV tier is not a lever — results `2026-09-16-r358-hostkv`

`sysmem_kv_cache` was 0 in every measurement above; this boots it at 4096 MiB and runs the three shapes that could
plausibly notice, against the same shapes with no tier. The expectation was written down before the run: a host tier
cannot make more than the 262,144-token VRAM pool fit, so it should not change admission; what it could change is
recomputation.

| shape | tier 0 (served) | tier 4096 MiB |
| --- | --- | --- |
| 152,761-token prompt, first send | decode 150.1, TTFT **32.36 s** | decode 150.9, TTFT **32.52 s** |
| the same prompt again | decode 161.4, TTFT **0.431 s** | decode 161.0, TTFT **0.441 s** |
| 8 × unique ~40k prompts at once | **8/8**, 48.4 agg, TTFT 17.7–116.1 s | **8/8**, 48.4 agg, TTFT 18.5–113.0 s |
| 4 × shared 152,761-token prompts | 51.4 per stream, 190.7 agg, TTFT 1.58 s | 48.8 per stream, 179.4 agg, TTFT 1.74 s |

Nothing moves. The reused prefix already stays in VRAM (0.43 s on the repeat, either way), and the pool is never
spilled to host under these shapes. **A negative result: the last configuration knob that could have addressed the
deep-context weakness does not.** The launcher keeps `SYS_KV` as a knob and defaults it to 0.

## Quality: tool-eval 69×4 — results `2026-09-16-r357-tooleval`

| check | result |
| --- | --- |
| **tool-eval 69×4, baseline** | **85.0 ± 2.9**, CI [82.5, 87.5], per-trial points [113, 115, 120, 121]; losses in categories G 5/6, H 8/10, I 16/20, K 19/26, N 5/6, O 10/12 |
| **tool-eval 69×4, promoted config** | **85.8 ± 3.1**, CI [83.5, 88.5], points [115, 118, 116, 124]; losses in G 5/6, H 8/10, I 16/20, K 21/26, M 5/6, N 5/6 |

**The promoted configuration does not cost quality.** 85.8 against 85.0 with overlapping intervals, on a
tool-calling benchmark, while measuring +35 % at short-context c4 and +78 % at deep-context c4. The greedy
byte-equality gates establish that the *decoding* is unchanged; this establishes that the *task behaviour* is.

Invocation: `tool-eval-bench --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8`. The harness sends
its own sampler parameters, so the server's preset fallbacks are not part of this measurement on either arm, which
is what makes the two arms comparable.

## Both levers together — results `2026-09-16-r354-combined`

Image `tabbyapi:qsa-cid` (QSA multi-job + concurrency-indexed draft depth, built from the same devel base, same
pip resolution, same native rebuild) against the served baseline, with `DRAFT_POLICY='[[2, 3], [8, 1]]'`.

| shape | baseline | combined | change |
| --- | --- | --- | --- |
| short context, c1 | 207.9 | 207.3 | unchanged, by design |
| short context, c4 | 64.0 / 251.1 | **87.3 / 338.0** | **+35 %** |
| short context, c8 | 39.7 / 307.5 | 39.6 / 310.7 | unchanged |
| deep context, c2 | 99.2 | 125.5 | **+27 %** |
| **deep context, c4** | 51.4 / 190.4 | **91.4 / 323.6** | **+78 %** |

The two levers compose and the deep-context c4 cell exceeds either alone (QSA alone read 72.5 there, the draft
policy alone is measured at short context): at four concurrent deep-context jobs, the attention path stops falling
back to eager *and* the drafter stops paying for depth 3. Output is byte-identical to the baseline
(`sha256` `750e1459e177c47e…`, 1,989 bytes).

That is the configuration to serve: **+35 % to +78 % at the concurrency a fan-out of agents reaches, with
byte-identical output.**

## Code and prose from one boot of the served configuration — results `2026-09-18-r477-daily-prose-code`

`fn_bench`, 2,048 forced tokens, greedy, two runs per cell, code first, prose, then code again to bracket drift, on the served image of 2026-09-17 12:45 CEST. Every request finished on `length`. Aggregate over the streams; per stream in parentheses.

| kind | c1 | c4 aggregate | c8 aggregate |
| --- | --- | --- | --- |
| code | 209.8–215.0 (drift bracket 215.7) | 422.7–442.1 (111.6–116.7) | 540.6–573.4 (68.7–72.9) |
| prose | 163.9–171.6 | 417.6–433.2 (104.9–108.8) | 539.0–558.4 (67.7–70.1) |

Prose is 0.78× code at c1 and within 3 % of it at c4 and c8. The draft policy is depth 3 up to four streams and depth 1 above, so at c8 both kinds run one draft per step and the batch, not acceptance, sets the rate; at c4 the acceptance difference is absorbed by the batched step. The 2026-09-16 prose figure on the pre-promotion image was 160.6–165.3 at c1 (`2026-09-16-r339-gates`).

## ExLlamaV3 against vLLM on the same checkpoint — results `2026-09-18-vllm-exl3-route`

The vllm-exl3 route serves the same `qwen3.8-flash-next-exl3-3.05bpw` checkpoint through vLLM main (2026-09-18),
TP2 on both cards. Same instrument as this stack's own rows: `fn_bench`, 2,048 forced tokens, greedy, code.

| engine | profile | kind | c1 decode (t/s) | c4 aggregate (t/s) | KV pool (tokens) | GSM8K |
| --- | --- | --- | --- | --- | --- | --- |
| ExLlamaV3 + TabbyAPI (this stack, 2026-09-17) | served, 8-bit KV, MTP depth 3 | code | 207–214 | 425–450 | 262,144 | 0.9158 (n=1319) |
| vLLM | `d-mtp3` — BF16 KV, MTP depth 3 | code | 131.6 | 475.4 | 95,183 | 0.945 (n=200) |
| vLLM | `d-mtp2` — BF16 KV, MTP depth 2 | code | 121.7 | 421.4 | 108,651 | 0.92 (n=200) |
| vLLM | `e-fp8` — fp8 KV, no MTP | code | 75.6 | 222.2 | 309,657 | — |
| vLLM | `e-fp8-mtp1` — fp8 KV, MTP depth 1 | code | 117.0 | 339.2 | 159,744 | — |
| vLLM | `e-fp8-mtp2` — fp8 KV, MTP depth 2 | code | 118.1 | 406.0 | 131,072 | 0.94 (n=200) |

At c4 the vLLM route's best profile (`d-mtp3`) reads 475.4 t/s aggregate on code against 425–450 here, 1.06–1.12× this stack; at c1 it reads 131.6 t/s against 207–214 here, 62–64 % of this stack's rate. The depth-2 profile reads 421.4 and 121.7 (parity at c4, 57–59 % at c1). The KV pool tracks the KV dtype and the draft depth across the rows: 309,657 tokens with fp8 KV and no MTP on the vLLM route, 262,144 with 8-bit KV here, 159,744 with fp8 KV at MTP depth 1, 131,072 with fp8 KV at depth 2, 108,651 with BF16 KV at depth 2 and 95,183 at depth 3. fp8 KV against BF16 KV at depth 2 costs 3 % at c1 and 4 % at c4 for 1.21× the pool. Pool figures are the engine's `GPU KV cache size` lines, collected in `kv-pools.txt` in the results directory; the e-fp8-mtp1 aggregates are the `fn_bench` summary lines in `r475-audit.txt`; the d-mtp3 row is one run from `r476-d-mtp3-records.jsonl` and its GSM8K line is in `r476-audit.txt`; the e-fp8-mtp2 row is `r475b-e-fp8-mtp2-records.jsonl` with its GSM8K line in `r475b-audit.txt`.

c8, prose, prefill and long-context are **not measured on the vLLM route**. This stack's figures for those shapes,
for reference: c8 aggregate 550–604 t/s on code; prose c1 160.6–165.3 t/s; a 27,501-token prompt prefills in 3.5 s
(7,700–7,800 t/s) and a 110,081-token prompt in 13.1 s (8,380–8,390 t/s), both on 2026-09-17; long-context
retrieval 5/5 at 26.5k, 105.7k and 158.5k prompt tokens.

## Four slots, K8V4 and one MoE layer on the CPU: the page pool at 8-bit KV — results `2026-09-18-r480-exl3-pool`, gates `2026-09-18-r481-s4-promote`

VRAM on the served configuration with 8 slots: main weights ~45.7 GiB, GDN recurrent state ~3.4 GiB (8 slots × 36 linear-attention layers × 4 fp32 copies, one per draft position plus one), page pool 262,144 tokens at 8-bit KV ~4 GiB, MTP head ~1.0 GiB, vision ~0.5 GiB. The 30.5 GiB PLE table and the 1.2 GiB token embeddings stay in host memory. Each arm below found its largest pool by booting down a ladder; the next size up failed with `RuntimeError: Insufficient VRAM in split for model and cache`. Vision on, MTP policy `[[4, 3], [8, 1]]`, window 262,144, `fn_bench` 2,048 forced tokens × 2 runs, needle at five positions, GSM8K n=200 at four concurrent requests.

| arm | pool | free MiB, card 0 / 1 | c1 / 30k greedy | code c1 / c4 | prose c1 / c4 | cold prefill, 22,625 / 90,135 tokens | needle 131k / 240k | GSM8K c4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 slots, 8,8 (previous served) | 262,144 | 947 / 2,511 | reference / reference | 214–217 / 435–442 | 168–172 / 428–436 | 3.06 s (7,390 t/s) / 8.13 s (11,090 t/s) | 5/5 / 5/5 | 0.925 |
| **4 slots, 8,8 (served since 2026-09-18 12:24)** | **360,448** | 895 / 2,581 | identical / identical | 214–217 / 429–444 | 167–172 / 422–436 | 3.12 s (7,250 t/s) / 8.19 s (11,000 t/s) | 5/5 / 5/5 | 0.925 |
| 4 slots, K8V4 (`8,4`) | 425,984 | 893 / 2,681 | differs / identical | 188–194 / 447–462 | 159–165 / 409–422 | 3.11 s (7,290 t/s) / 8.25 s (10,930 t/s) | 5/5 / 5/5 | 0.920 |
| 4 slots, 8,8, first MoE layer's routed experts on the CPU | 425,984 | 975 / 2,061 | differs / identical | 157–166 / 341–351 | 148–149 / 323–325 | 4.88 s (4,630 t/s) / 14.69 s (6,140 t/s) | 5/5 / 5/5 | 0.935 |

Four slots change no arithmetic: the same kernels run, and only the number of recurrent-state slots changes, so greedy output is byte-identical and decode and prefill are unchanged within the two-run spread. K8V4 stores V at 4 bits: the short greedy output diverges, retrieval holds at 240k, GSM8K stays within one standard error, and code decode at c1 drops about 11 %. The CPU expert path buys the same pool as K8V4 at a 20–25 % decode and 35–45 % prefill cost.

The prefill cells use only the first request of each context. The step sent the same prompt three times, so the second and third requests were served from the prefix cache in 0.19 s and 0.25 s; they are in the records and excluded here.

Promotion gates on the 4-slot configuration (R481): c1 greedy fingerprint `1474eee2f5945248` (the reference), the agent request that originally failed ends in a parsed `todo_write` tool call, eight concurrent requests on four slots all complete (439.7 t/s aggregate at 1,024 tokens, four queued behind the first four), and tool-eval 69×4 reads 84.8 ± 1.0 (CI 84.0–85.5) against 83.5–86.5 for three runs of the 8-slot configuration.

## Stamina — results `2026-09-16-r347-soak`

Forty rounds of c4, 1,024 forced code tokens each, alone on the box: per-round median decode 63.5–65.5 t/s,
**drift 101.0 % of the start** (first three rounds 64.5, last five 65.1). No decay, no error, no VRAM drift.

That is the figure the first attempt could not produce: the gate suite's soak ran while a native extension was
compiling on the same host and read 40.5 t/s from round six onward, which has the shape of stamina decay and was
not. See `docs/GOTCHAS.md` #8.

## Capabilities — `bench/capabilities.py`, results `2026-09-16-r348-capabilities`

| check | result |
| --- | --- |
| JSON-schema structured output | **PASS** — content parses *and* satisfies the schema (`city`, `population`, `coastal`, `climate`), and the server log shows the grammar engaged (`constrained generation … json_schema (req)`), so it is not the model being agreeable |
| tool call parsing | **PASS** — `write_note` with `{"file_path": "/tmp/gate-note.txt", "content": "hello from the gate"}`, arguments JSON-valid and limited to declared parameters |
| vision | **PASS** — a red 32×32 PNG built in-process, answer "Red" |
| reasoning channel | **PASS** — 309 chars in `reasoning_content`, 61 in `content` |

## Decode is content-dependent — same box, same day

| shape | kind | decode (t/s) | draft acceptance |
| --- | --- | --- | --- |
| `/completions`, 4,096 forced, c1 | code | 176.3 | 62 % |
| `/completions`, 7,107 then loop-stopped, c1 | code | 187.1 | 72 % |
| chat, 512 forced, c1 | prose analysis | 94.1 | 46 % |

Drafts are sampled greedily and the target is not, so acceptance tracks how predictable the continuation is. A
decode rate without its kind is not comparable to another one.

## Boot and footprint

| quantity | value | conditions |
| --- | --- | --- |
| model load | 11.2–11.5 s | warm Triton + coop-autotune caches on disk |
| warmup (first inference after load) | 0.27–0.36 s | same |
| VRAM at idle, model resident | 31.9 GB / 30.1 GB of 32.6 GB per card | both cards held by one process |
| GPU power at idle-resident | ~227 W / ~218 W of 600/575 W | measured during a request, layer-split duty cycle |

## A real agent turn — DSH session `session-652732d8`, 2026-09-16

The exact task that failed before the R338 fix (build a voxel pagoda scene in one HTML file and open it in
Chrome), driven end to end by DSH Desktop through port 8022:

| quantity | value |
| --- | --- |
| steps / tool calls | 20 / 23 |
| tools used | `skill`, `todo_write`, `write`, `edit`, `bash`, `read`, `read_image` |
| output tokens | 28,931 (largest single generation 11,249) |
| prompt tokens | 22,599 fresh, **813,056 served from the paged prefix cache** |
| outcome | file written, Chrome opened, screenshots read back through the vision tower, two visual iterations, then closed |

Reasoning arrived in `reasoning_content`, short purposeful text in `content`, tool calls parsed as `tool_calls` on
every step. Against the pre-fix run of the same task: no tool call at all, 19,477 characters of degenerated
deliberation delivered as the visible answer.

## Not yet measured

On the vllm-exl3 route: c8 aggregate, prose decode, prefill at depth, and long-context retrieval. On this stack:
deep-context fan-out past the eight concurrent 38,283-token requests already recorded. See `bench/r339-gates.sh` for
the gate suite and `bench/results/` for what has landed.

## 2026-09-17 — the promoted layers, in order (each admitted by its own gate; see `docker/README.md`)

| promoted (CEST) | layer | gate evidence | decode c1 / c4 / c8 (fn_bench code 2048) | prefill 30k / 120k | results |
| --- | --- | --- | --- | --- | --- |
| 02:15 | bszn16 + policy `[[4, 3], [8, 1]]` | c1 fingerprint identical; GSM8K, tool-eval under concurrency | 203 / 375 / 443 | — | `2026-09-17-r414-*` |
| 03:15 | coopwide | c1 byte-identical; c8 +7.5 % | 206–209 / 369–376 / 469–478 | — | `2026-09-17-r421-coopwide-ab` |
| 04:32 | hcmix2 (`EXL3_HC_MIX_V2=1`, `MIN_R=1`) + hostgap (`EXL3_HOST_GAP_REWIND=1`) | identical at MIN_R 1; c4 +12 %, c8 +8 % | 213–218 / 422–434 / 537–548 | — | `2026-09-17-r428-hcmix2-stack-ab` |
| 07:35 | prefill pipeline + nosync + mtpfix2 (`EXL3_LS_PREFILL_PIPELINE=1`) | c1 and 30k fingerprints identical | 213 / 430 / 540 | **3.5 s / 13.1 s** (was 5.5 / 22.3) | `2026-09-17-r442-ppipe-memfix-ab`, gates `2026-09-17-r446-gates-ppipe` |
| 12:45 | MoE coop V2 (`EXL3_MOE_COOP_V2=1`) | bit-exact at R = 1..16 in the kernel test, c1 + 30k fingerprints identical, five gates (`2026-09-17-r461-gates-moecoopv2`: GSM8K 0.935, tool-eval 85.5 ± 1.7, needle 5/5) | 207–214 / 425–450 / **550–604** (per stream 75–77) | unchanged | `2026-09-17-r460-moecoop-v2-ab`, `2026-09-17-r460b-moecoop-v2-gputest` |

Pool: 262,144 tokens at 8-bit KV is the ceiling on this box under any split (393,216 and 327,680 fail to boot: `2026-09-17-r452-exl3-cache-bits`, R337). Structured output (llguidance `json_schema` / `response_format` / `regex_pattern`) works, thinking on and off, at c4: `2026-09-17-r453-exl3-structured`.

Not promoted, 2026-09-17 14:20 CEST: MoE coop mode 3 (`EXL3_MOE_COOP_V2=3`, V1 path for singleton expert runs inside the V2 kernel) is bit-identical to mode 1 (same c1 and 30k fingerprints, GSM8K c8 0.935) but 2–3 % slower at c1, c4 and c8 (214 / 115 / 70 per stream against 220 / 118 / 73), because the mode-3 kernel is 8–19 % slower than V1 on rows that route to distinct or partly overlapping experts, which is what decode rows look like: `2026-09-17-r462-moecoop-v3-ab`. The served configuration keeps mode 1.
