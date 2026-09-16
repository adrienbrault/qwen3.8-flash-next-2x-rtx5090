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

## Admission: eight distinct deep contexts at once — results `2026-09-16-r339-gates`

Eight concurrent requests, each with its **own** ~38,283-token prompt (unique suffixes, so no page sharing) and 512
forced output tokens. `cache_size` is 262,144 tokens, so the eight prompts total 306k — more than the pool.

| | value |
| --- | --- |
| admitted / completed | **8 / 8** |
| TTFT | 59.4–62.4 s (all eight within 3 s of each other) |
| decode | 28.5–33.6 t/s per stream, ~256 t/s aggregate |
| wall | 77.3–77.8 s |

So eight deep-context agents coexist: they interleave through the pool rather than being rejected, at a ~1 minute
first-token cost when all eight arrive together. The daily's equivalent reads a 3.6 s TTFT at 30k for a *single*
request (R177) — this is the cost of an 8-slot, 262k-token pool against a 16-slot, 1.39M-token one.

## Decode at the requeue boundary — 2,048 forced tokens, results `2026-09-16-r339-gates`

2,048 is TabbyAPI's requeue budget on this config (`chunk_size 2048`, `output_chunking: true`), so these rows sit
exactly at the boundary where the engine's own token accounting used to break.

| kind | concurrency | decode per stream (t/s) | aggregate (t/s) | TTFT (s) |
| --- | --- | --- | --- | --- |
| code | 1 | 210.7–210.8 | 207.7–207.8 | 0.145 |
| code | 4 | 64.4 | 252.4 | 0.511 |
| prose | 1 | 162.4–167.2 | 160.6–165.3 | 0.14–0.15 |

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
