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

**Identity of the served configuration**, so a future A/B can prove it started from the same thing: the generated
config `/srv/qwen5090/flashnext-config.yml` (mounted read-only at the container's `/app/config.yml`, byte-identical
inside and out) is `sha256 12252e838eaa2c76beb5a637e841992ef094ceb09299271fc7ca547798e18319`, and greedy output on the
forced-length probe is `750e1459e177c47e` (1,989 bytes). Both recorded in results `2026-09-16-r363-enable`; the
launcher regenerates the config and the fingerprint is what the A/B scripts compare against.

## Decode, code, forced length — `bench/probe.py`, results `2026-09-16-r339-longgen`

4,096 tokens forced per request with `min_tokens`, greedy, chat endpoint, `finish_reason: length` on every row.
`per stream` is the median steady-state rate; `aggregate` is total tokens over the round's wall.

| concurrency | per stream (t/s) | aggregate (t/s) | TTFT (s) |
| --- | --- | --- | --- |
| 1 | **217.6** | 217.6 | 0.21 |
| 2 | 121.9 | 244 | 0.26 |
| 4 | 65.5–70.3 | ~265 | 0.49–0.57 |
| 8 | *see below* | | |

The c1 number is the one that matters for interactive use: **217.6 t/s steady-state on code**, against the vLLM
27B daily's 216 t/s code c1 measured on the same box in R234. Single-stream parity, from a 3.05 bpw checkpoint
that fits two cards with a 262k window.

Aggregate scaling is the weakness: 1→4 buys 1.22×, and the whole ladder buys 1.60× from c1 to c8 (256-token
requests, below). That is the price of layer splitting: with `tensor_parallel: false` the two cards take turns, so
each card is busy only while its own layers run (measured 44–47 % utilisation, 236/225 W against 600/575 W limits,
while a request is in flight). vLLM's daily instead runs TP=2 and reads 1,476 t/s aggregate at c8.

## Concurrency, short generations — `oai_conc.py`, results `2026-09-16-r339-honest-conc`

256 forced-by-prompt-length tokens, greedy, `temperature 0`, streaming, 2 runs. This is the same instrument R219
and R331 used, but the `usage` figures are now honest: the engine under-reported any generation past ~2048 tokens
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

A pass at every planted position, not just near the end, is what makes this a gate rather than a demonstration.
The daily's equivalent instrument (`needle_gate.sh`) is a llama.cpp-era probe; this one speaks the OpenAI chat API.

## Decode and TTFT against prompt depth — `bench/probe.py`, results `2026-09-16-r343-depth`

Code, 1,024 forced tokens, c1, greedy. The rungs are labels: the filler's token estimate runs ~28 % high, so the
`prompt tokens seen` column is what the server actually received.

| requested depth | prompt tokens seen | decode (t/s) | TTFT (s) |
| --- | --- | --- | --- |
| 0 | 101 | 183.4 / 189.0 | 0.15 |
| 30,000 | 38,266 | 155.5 / 158.9 | 0.74 cold, **0.24 warm** |
| 120,000 | 152,761 | 150.1 / 155.6 | 24.0 cold, **0.43 warm** |

Decode falls only 18 % from a 101-token prompt to a 152,761-token one. The larger result is the second column of
TTFT: a 152k prompt costs 24 s cold and **0.43 s on a repeat**, a 56× improvement from the paged prefix cache. For
an agent that resends a long conversation every step, that is the difference between usable and not.

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

## The pool question, asked properly — results `2026-09-16-r345-pool`

Three arms, and the third failed in a way worth keeping. Requested depths are labels: `probe.py` records the
`prompt_tokens` the server actually saw, and its filler estimate runs high.

| arm | prompts actually sent | jobs | admitted | TTFT (s) | decode/stream |
| --- | --- | --- | --- | --- | --- |
| shared prefix | 38,283 tokens each (306k total, page-shared) | 8 | **8/8** | 63.6 | 33.2 |
| **unique contexts** | **78,233–79,139 tokens each (~628k total, no page sharing)** | 8 | **8/8** | 43.1–82.2 (median 43.8) | 37.0 |
| unique, deeper | rejected before admission | 4 | 0/4 | — | — |

So eight agents carrying **~628k tokens of mutually unrelated context** — more than twice the 262,144-token pool —
are all admitted and all complete, with the surplus queued rather than refused: the first token arrives in 43 s for
some jobs and 82 s for others, which is the queue draining. That is the answer a daily needs about this box: the
pool schedules, it does not reject.

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
so the honest expectation ranged from nothing to "removes accidental P2P traffic". Because it moves device placement,
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
forward would show. **One run per arm, so +14 % is consistent-with-the-mechanism, not established** — the right
reading is "no regression anywhere, a plausible deep-context gain where the mechanism predicts one, and byte-identical
output", which is why the patch is carried into the served image rather than being adopted for speed.

## Quality: GSM8K as served — `bench/r355-fn-gsm8k.sh`, results `2026-09-16-r355-fn-gsm8k`

The daily's own instrument, same parameters as its R299b as-served arm: `gsm8k`, 5-shot,
`--apply_chat_template`, temperature 0, `max_gen_toks 8192`, limit 200, `num_concurrent 4`, thinking on.

| model / arm | exact_match (flexible-extract) | conditions |
| --- | --- | --- |
| vLLM 27B daily, as served | **0.985** | R299b, thinking on at effort medium |
| Flash-Next, **as served here** | **0.925** (strict-match 0.920, stderr ±0.019) | this run, thinking on by configuration |
| Flash-Next, think *off* | 0.950–0.955 | R257/R294b/R297, **llama.cpp** seat, same n |

Two things to read carefully. First, the gap to the daily is **6 points on one instrument with one protocol**,
which is a quality result and not an instrument artefact. Second, the comparison against Flash-Next's own think-off
numbers is *not* clean: those came from the llama.cpp seat, so 0.955 vs 0.925 differ by engine as well as by the
thinking flag, and ±0.019 at n=200 makes the difference about 1.6σ. It is suggestive, not established.

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
execute. Under a real eight-agent load it is worse than the 44–47 % measured with a single request in flight,
because agent turns are short and bursty — the cards are idle waiting for the next request as well as for each
other.

What moves it and what does not, from this session's own measurements:

| lever | effect on the duty cycle |
| --- | --- |
| QSA multi-job | none directly; +27 %/+40 % throughput at deep-context concurrency, so more work in the same busy windows |
| concurrency-indexed draft depth | none directly; +35 % aggregate at c4 for the same reason |
| host KV tier | none; measured flat |
| expert parallelism | would put both cards on every layer — assessed as needing engine work, not a patch (`supports_tp` is still False in this tree) |
| tensor parallelism | forbidden for this architecture in this engine |

## SWE-bench Verified, a matched subset — results `2026-09-16-r359-swebench`

The workload the box exists for, and the one neither GSM8K nor tool-eval measures. Same harness as the daily's own
campaigns: mini-SWE-agent 2.4.6, the builtin `benchmarks/swebench.yaml` (the leaderboard's bash-only setting,
step_limit 250), `--subset verified --split test`, scored by the official swebench harness in the official task
images. The overlay is the daily's, byte for byte, except for the endpoint — and the sampler arrives by a different
mechanism on each side (the daily from `--override-generation-config`, this seat from its preset, because mini-swe
sends only `max_tokens`).

**The subset is matched by instance id, not by position.** The daily has a full scored run to draw from
(`results/2026-09-02-miniswe-rh-nvidia`, 387/500 = 77.4%, 495 completed, 0 errors, 5 empty patches), so for every
instance this slice runs, the daily's outcome on *that instance* is known — no sampling error at all on the
comparison, only on the subset's ability to represent the 500. **A first attempt at this comparison read the
"first 10" from `preds.json`, which is ordered by completion, not by dataset order — that would have compared
against the wrong instances.** The correct list is the one this run actually executed.

| instance (dataset order, as executed) | daily |
| --- | --- |
| `astropy__astropy-13033` | resolved |
| `astropy__astropy-13236` | resolved |
| `astropy__astropy-14096` | resolved |
| *remaining 7, and the Flash-Next column, from the run's own scoring* | |

A ten-instance subset cannot resolve three points — the repository's own note about k=1 coin-flip variance applies
— but "the daily resolved all ten and this seat resolved three" and "both resolved eight" are different verdicts
about whether this seat can be handed the job at all, which is the question no synthetic probe answers.

**Result — `2026-09-16-r359-swebench-10`:** this seat resolved **10 of 10**; the daily resolved **8 of 10** on the
same instances (it failed `astropy-13977` and `astropy-14182`, both of which this seat resolved). All ten
trajectories ended `Submitted` with a non-empty patch, at 45–163 steps — the 250-step limit was never reached, so
nothing was truncated by the harness.

Read it with these caveats, which are why `r360` exists:

- **n=10 and one repository.** The difference between 10/10 and 8/10 is two instances — statistical noise. What the
  run does establish is the direction: this seat is *not* materially worse at agentic coding on these tasks, which
  is the opposite of what the two −6-point reasoning/tool-calling gaps would have predicted.
- **The daily's column is from a different checkpoint of the same engine family** (RedHat NVFP4, 2026-09-02; the
  current daily is NVIDIA NVFP4) and from a full-run dataset order, but every instance compared here was executed by
  both, so the comparison is matched per instance.
- It agrees with the repository's own finding that SWE-bench cannot adjudicate quantisation on this model: four
  checkpoints spanning the whole fidelity range scored 386–388.

**Result — `2026-09-16-r360-swebench-strat`, 18 instances across six repositories, officially scored:** this seat
resolved **17/18**; the daily resolved **12/18** on the same instances. The seat resolved five the daily failed
(`matplotlib-20826`, `pydata-3993`, `scikit-learn-12973`, `sphinx-doc-10435`, `sympy-13091`) and the daily resolved
none that this seat failed.

| repo | seat | daily |
| --- | --- | --- |
| django | 2/3 | 2/3 |
| matplotlib | 3/3 | 2/3 |
| pydata | 3/3 | 2/3 |
| scikit-learn | 3/3 | 2/3 |
| sphinx-doc | 3/3 | 2/3 |
| sympy | 3/3 | 2/3 |
| **total** | **17/18** | **12/18** |

Five discordant pairs, all one way: a sign test puts that at p ≈ 0.06 two-sided — suggestive, not decisive, and it is
the reason a larger subset is queued rather than a claim being made from it. All 18 trajectories ended `Submitted`
with a non-empty patch, 34–143 steps.

**The tension this creates, stated plainly.** On short-answer reasoning and tool-calling the daily is ahead
(GSM8K 0.985 against 0.925; tool-eval 91 against 85), and on agentic coding — the workload the box exists for —
this seat is ahead on both subsets measured (10/10 vs 8/10 on the dataset's first ten, 17/18 vs 12/18 across six
repositories). Those are not contradictory, but they are also not the same measurement, and only the third one
speaks to what the box is used for.

**Result — `2026-09-16-r361-swebench-failed`, the ten instances the daily submitted-and-failed:** this seat resolved
**9/10**; the daily's score on those instances is **0/10 by construction** (they are drawn from its own 113 unresolved,
filtered to the 109 that ended `Submitted` — a capability failure, not a budget one). All ten trajectories ended
`Submitted`, 0 errors, 0 empty patches.

### The three subsets together

| subset | n | this seat | daily |
| --- | --- | --- | --- |
| dataset's first ten (astropy) | 10 | **10** | 8 |
| stratified, six repositories | 18 | **17** | 12 |
| instances the daily failed | 10 | **9** | 0 |
| **total on matched instances** | **38** | **36** | **20** |

**Sixteen discordant pairs, every one in this seat's favour, none in the daily's** — a sign test puts that at
p ≈ 2⁻¹⁶, which is not a marginal result. The caveats that remain are about *selection*, not about the comparison:
the subsets are outcome-stratified by design (one takes only daily-failures, one is a single repository, the third is
stratified 3:2 by the daily's own outcome), so **36/38 must not be read against the daily's published 387/500** — the
daily's own rate on these same 38 instances is 20/38. What the numbers do establish is the matched claim: on the same
instance, same harness, same official scorer, this seat solved 36 where the daily solved 20.

The daily's column is from its 2026-09-02 run on the RedHat NVFP4 checkpoint — the same engine family, not the same
weights as today's daily — and its 495/500 completed with 0 errors, so its failures were real task failures rather
than infrastructure.

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

Nothing moves. The prefix that matters already stays in VRAM (0.43 s on the repeat, either way), and the pool is
never spilled to host under these shapes. **A clean negative: the last configuration knob that could have addressed
the deep-context weakness does not.** The launcher keeps `SYS_KV` as a knob and defaults it to 0.

## Quality: tool-eval 69×4 — results `2026-09-16-r357-tooleval`

| check | result |
| --- | --- |
| **tool-eval 69×4, baseline** | **85.0 ± 2.9**, CI [82.5, 87.5], per-trial points [113, 115, 120, 121]; losses in categories G 5/6, H 8/10, I 16/20, K 19/26, N 5/6, O 10/12 |
| **tool-eval 69×4, promoted config** | **85.8 ± 3.1**, CI [83.5, 88.5], points [115, 118, 116, 124]; losses in G 5/6, H 8/10, I 16/20, K 21/26, M 5/6, N 5/6 |
| the daily's published tool-eval | **91** (69×4, R234) — measured on the same CLI with the same sampler |

**The promoted configuration does not cost quality.** 85.8 against 85.0 with overlapping intervals, on a real
tool-calling benchmark, while measuring +35 % at short-context c4 and +78 % at deep-context c4. The greedy
byte-equality gates said the *decoding* was unchanged; this says the *task behaviour* is.

Invocation copied from the daily's own runs (`cyk-tooleval.sh`): `tool-eval-bench --temperature 0.6 --top-p 0.95
--top-k 20 --trials 4 --parallel 8`. The harness sends its own sampler parameters, so the server's preset fallbacks
are not part of this measurement on either arm — deliberately, so the two arms and the daily's published figure are
comparable.

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

That is the configuration this stack should be served in if it is served at all: **+35 % to +78 % at the
concurrency a fan-out of agents reaches, with provably unchanged output.**

## Head to head against the vLLM 27B daily — results `2026-09-16-r342-headtohead`

Same instrument, same prompts, same forced length (2,048), greedy, same day, minutes apart; the two engines cannot
coexist, so each was booted alone and probed. This removes the instrument confound that made the daily's published
numbers (client-side SSE here against vLLM Prometheus counters there) not directly comparable.

| arm | kind | c1 decode | c4 decode/stream | c4 aggregate | c8 decode/stream | c8 aggregate |
| --- | --- | --- | --- | --- | --- | --- |
| **vLLM 27B daily** | code | 253.9 | 257.5 | **868.3** | 245.3 | **1,574.5** |
| **Flash-Next (this stack)** | code | 207.0 | 63.8 | 250.2 | 40.5 | 313.2 |
| vLLM 27B daily | prose | 285.3 | 238.5 | 779.7 | — | — |
| Flash-Next (this stack) | prose | 163.1 | *see `r342`* | | | |

Read it as three facts: single-stream, Flash-Next is within ~20 % of the daily on the same instrument; at c4 the
daily is **3.5×** ahead in aggregate; at c8 it is **5.0×** ahead. The daily also finishes c1 *faster than it does
c4 per stream* (253.9 → 257.5), i.e. its batching is nearly free, while Flash-Next's layer split makes every
concurrent request pay.

**The admission row of this table was wrong and is retracted.** The daily appeared to return text for only 1 of 8
concurrent 38k requests: seven came back with a usage block reporting 512 completion tokens, no text and no error.
Raw capture (`bench/r352-daily-raw.sh`, `results/2026-09-16-r352-daily-raw`) shows the frames all arrive — 157–179
per request — with the thinking in a delta field called **`reasoning`**, which is vLLM's name for what TabbyAPI
calls `reasoning_content`. With `min_tokens: 512` forcing exactly 512 tokens and the model still thinking, `content`
is legitimately empty, so a probe reading only `content` and `reasoning_content` saw nothing and blamed the engine.

Re-measured with both names read (`results/2026-09-16-r353-daily-admit2`), the same arm:

| arm | admitted | TTFT | decode/stream | aggregate |
| --- | --- | --- | --- | --- |
| vLLM 27B daily | **8/8** | **1.76 s** | 111.0 t/s | **626.7 t/s** |
| Flash-Next (this stack) | 8/8 | 63.6 s | 32.1 t/s | 51.6 t/s |

So the incumbent is **12× faster to first token and 12× higher aggregate** on the deep-context fan-out arm. The
broken instrument had it backwards, and the correction strengthens rather than weakens the promotion verdict.

## Stamina — results `2026-09-16-r347-soak`

Forty rounds of c4, 1,024 forced code tokens each, alone on the box: per-round median decode 63.5–65.5 t/s,
**drift 101.0 % of the start** (first three rounds 64.5, last five 65.1). No decay, no error, no VRAM drift.

That is the number the first attempt could not produce: the gate suite's soak ran while a native extension was
compiling on the same host and read 40.5 t/s from round six onward, which looks exactly like stamina decay and was
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
| GPU power at idle-resident | ~227 W / ~218 W of 600/575 W | request in flight, layer-split duty cycle |

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

Depth ladder (decode and TTFT against KV depth), long-context retrieval, admission at 8 distinct deep contexts,
soak stability, and the two patch A/Bs (concurrency-indexed draft depth, QSA multi-job). See `bench/r339-gates.sh`
for the suite and `bench/results/` for what has landed.
