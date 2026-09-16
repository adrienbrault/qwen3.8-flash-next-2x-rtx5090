# MEASUREMENTS

Every number here was measured on the `flan` box, on the date given, with the served configuration as it stood
that day, and the raw records are named next to it. Decode rates are steady-state (first token to last) unless
the column says wall.

## The served configuration these numbers describe (since 2026-09-16)

`qwen3.8-flash-next-exl3-3.05bpw` on TabbyAPI + ExLlamaV3 v1.5.0, port 8022, image `tabbyapi:53da7919-rqcount`,
262,144-token window and cache, 8-bit KV, 8 slots, layer-split across both cards (TP is not implemented for this
architecture), MTP draft depth 3, vision on, `qwen3_coder` tools, sampler preset `qwen38_thinking` (T=0.6,
top_k 20, top_p 0.95 as **fallbacks**). See `docs/CONFIG.md` for why each value.

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

The daily's admission arm returned 1 of 8 concurrent 38k-context requests with text; the other seven came back
with a usage block reporting 512 completion tokens, no text deltas and no error. That is an anomaly I could not
diagnose because `r342` stopped the container before capturing its log, so it is re-run with the log kept
(`bench/r349-daily-admit.sh`) and no claim is made about it until that run lands.

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
