# Qwen3.8-Flash-Next on 2× RTX 5090 (ExLlamaV3 + TabbyAPI)

Serving configuration, launcher, image recipe, instruments and measurements for
`qwen3.8-flash-next-exl3-3.05bpw` — a 3.05 bpw EXL3 quantisation of Qwen3.8-Flash-Next — served by TabbyAPI on top
of ExLlamaV3 v1.5.0, across two RTX 5090 cards, with a 262,144-token window, vision, reasoning and tool calls all
enabled.

Every number below was measured on the `flan` box, on the date given, next to the results directory it came from.
Nothing here is a projection. The instruments that produced them are in `bench/`, because three of the numbers
this track published before 2026-09-16 turned out to be measured wrong — see `docs/GOTCHAS.md`.

**This repository is private.** The Flash-Next / ExLlamaV3 track is deliberately not published; do not push it or
mirror it into the public 27B repository. See `CLAUDE.md`.

## The numbers

Served configuration since 2026-09-16 (`scripts/launch-flashnext.sh`, image `tabbyapi:qsa-cid-pr337`, draft policy
`[[2, 3], [8, 1]]`). Verified as served, not asserted: greedy fingerprint `750e1459e177c47e` byte-identical to the
configuration every number below was measured on, the original failing agent request returning a **parsed** tool call
(`finish_reason=tool_calls`, 59,531 chars of reasoning separated into `reasoning_content`), and the 131k-context
retrieval gate **5/5** — results `2026-09-16-r363-enable`.

| | value | measured |
| --- | --- | --- |
| decode, code, 1 stream | 217.6 t/s steady-state, 4,096 forced tokens | 2026-09-16, `bench/results/2026-09-16-r339-longgen` |
| decode, code, 1 stream, same instrument as the daily | **207.0 t/s** vs the vLLM 27B daily's 253.9 | 2026-09-16, `results/2026-09-16-r342-headtohead` |
| decode, prose, 1 stream | 157.5 t/s steady-state, 4,096 forced tokens | same |
| decode, code, 8 streams | 40.6 t/s per stream, **304.6 t/s aggregate** | same |
| decode, short requests, 8 streams | 42.9 per stream, 283.5 aggregate, TTFT 0.51 s | 2026-09-16, `results/2026-09-16-r339-honest-conc` |
| TTFT, warm, short prompt | 0.10 s at c1, 0.85 s at c8 | both |
| boot to serving | 11.2–11.5 s load + 0.3 s warmup (warm kernel caches) | 2026-09-16, launcher log |
| VRAM resident | 31.9 GB / 30.1 GB of 32.6 GB per card | 2026-09-16 |
| window / cache | 262,144 tokens, 8-bit KV, 8 slots | 2026-09-16 |
| a real agent turn | 20 steps, 23 tool calls, 28,931 output tokens, 813k prompt tokens served from the prefix cache, file written and visually verified | 2026-09-16, DSH session `session-652732d8` |

The c1 code figure was the reason this track looked promising: 217.6 t/s steady-state from a 3.05 bpw checkpoint.
Measuring the incumbent on the *same instrument, same prompts, same day* puts it at 207.0 against the daily's
253.9 — an 18 % gap, not parity, and the difference between the two comparisons is instrument and prompt shape,
not the engine. What is not in doubt is the shape of the curve: this stack holds a 262k window on two cards with no
KV tier, and it collapses under concurrency where the daily does not.

Aggregate from c1 to c8 is 1.60× on short requests and about 1.4× on 4k-token generations; the daily reads 868 t/s
aggregate at c4 and 1,574 at c8 against 250 and 313 here, on one instrument, one day. `qwen4_exp` forbids tensor
parallelism in this engine (`NotImplementedError: Tensor-parallel is not currently implemented for
Qwen4ExpForConditionalGeneration`), so the two cards take turns over their own layers: each is busy only while its
own layers run, measured 44–47 % utilisation at 227/218 W against 600/575 W limits. **The idle half is the price of
avoiding per-layer all-reduce over PCIe 3.0; it is recovered by concurrency, and the concurrency available here is
eight slots in a 262k-token pool, not sixteen in 1.39M.**

**Two measured levers move that**, both validated with byte-identical output and both off by default:
`IMG=tabbyapi:qsa-cid DRAFT_POLICY='[[2, 3], [8, 1]]'` measures **+35 % aggregate at short-context c4 and +78 % at
deep-context c4** (91.4 t/s per stream against 51.4), and the exact agent request that failed before this session's
fix runs on that configuration returning parsed tool calls. See `docs/MEASUREMENTS.md` and `docs/PROMOTION.md`.

## What is in the box

| path | what it is |
| --- | --- |
| `scripts/launch-flashnext.sh` | the launcher: writes the served config and the sampler preset, mounts both, starts the container, warms the kernels, sweeps `/dev/shm` orphans first |
| `bench/probe.py` | decode/concurrency/depth instrument. Forces length with `min_tokens` (the only field that works here), records one JSONL line per request with the server's token count beside the client's |
| `bench/needle.py` | long-context retrieval gate at five planted positions per depth |
| `bench/r339-gates.sh` | the gate suite, run under the GPU lock |
| `docs/CONFIG.md` | every setting and why it has that value, including which ones are fit constraints |
| `docs/GOTCHAS.md` | the traps, as "what it looks like" vs "what it is" |
| `docs/MEASUREMENTS.md` | the numbers with their conditions |
| `docs/PROMOTION.md` | the daily-worthiness decision, gate by gate |
| `docs/RUNBOOK.md` | how to serve, measure, rebuild and hand the box back from this repo alone |

## Two defects this stack had, and what they cost

**Every client that sent no sampler was served at temperature 1.0, untruncated** (fixed 2026-09-16). TabbyAPI has
no sampling fallbacks unless a preset is named and it warns about this at boot; DSH sends only `max_tokens`, so its
reasoning spiralled into multilingual word salad, emitted an end-of-thinking tag inside the debris, and delivered
the rest of the spiral as the visible answer. The same task after the fix runs clean — see the agent-turn row
above. Full account: `kubernetes-home/flan/r338-tabby-sampling.md`.

**The engine under-reported its own generation by ~5×** for anything past the ~2048-token requeue boundary (fixed
2026-09-16, one line, asserted at image build time). A 17,544-token generation was reported as 3,500 tokens at
32.7 T/s. It made this server look five times slower than it is, and it is why the instruments now record the
server's count and the client's frames side by side.

## Reproducing a boot

```sh
ssh flan 'bash -s' < scripts/launch-flashnext.sh          # serve
PORT=8023 bash scripts/launch-flashnext.sh                # a second instance, different port
STOP=1 bash scripts/launch-flashnext.sh                   # stop
```

The launcher refuses to start if the checkpoint or the image is missing, stops the vLLM 27B daily first (the two
cannot coexist), waits for the cards to drain, and logs the warmup cost so it cannot become folklore.
