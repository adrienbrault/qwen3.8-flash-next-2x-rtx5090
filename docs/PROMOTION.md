# PROMOTION — is this stack worthy of being a daily?

A daily in this house means: one model served all day on the `flan` box, at a published configuration, with the
launcher, the measurements and the failure modes written down, and a rebuild reproducible from the repositories
alone. The incumbent is `nvidia/Qwen3.8-27B-NVFP4` on vLLM 0.29, TP=2, 16 sequences, a 1,391,795-token pool, held
since 2026-09-09 with 216 t/s code c1 and 1,476 t/s aggregate at c8.

This document answers whether `qwen3.8-flash-next-exl3-3.05bpw` on TabbyAPI + ExLlamaV3 should take that role.

## The gates, and where this stack stands

| # | gate | status | evidence |
| --- | --- | --- | --- |
| 1 | one-command boot, warm caches | **PASS** | `scripts/launch-flashnext.sh`; 11.2–11.5 s load, 0.3 s warmup, refuses to start on a missing checkpoint or image |
| 2 | single-stream decode at or near the incumbent | **PASS** | code 217.6 t/s steady-state at 4,096 forced tokens vs the incumbent's 216 on the same cards |
| 3 | concurrent aggregate | **FAIL against the incumbent** | 304.6 t/s at c8 vs 1,476; scaling 1.60× from c1 because `qwen4_exp` forbids TP=2 and the cards take turns (44–47 % utilisation at 227/218 W) |
| 4 | long-context retrieval | **PASS** | 5/5 at every planted position at 26.5k, 105.7k and 158.5k prompt tokens |
| 5 | several deep-context agents at once | **PASS with a caveat** | 8/8 admitted at ~38k tokens each (306k total against a 262k pool), 28.5–33.6 t/s per stream, TTFT 59–62 s when all arrive together |
| 6 | a real agent turn | **PASS** | DSH session: 20 steps, 23 tool calls, file written, Chrome driven, screenshots read back through vision, 813k prompt tokens served from the prefix cache |
| 7 | reasoning channel, tool parsing, vision | **PASS** | reasoning in `reasoning_content`, `qwen3_coder` calls parsed as `tool_calls`, vision used in anger |
| 8 | honest usage accounting | **PASS** | only after the R338 image patch: the engine under-reported long generations by ~5× |
| 9 | a sampler appropriate for the workload | **PASS** | only after R338: the server was serving untruncated T=1.0 to every client that sent no sampler |
| 10 | sustained load | **PASS** | 40 rounds at c4, drift 101.0 % of the start (63.5–65.5 t/s per stream, no error, no VRAM drift) |
| 11 | structured output (JSON schema) | **PASS** | content parses *and* satisfies the schema; the server log shows the grammar engaged. Tool-call args, vision on a red PNG and the reasoning channel also pass (`2026-09-16-r348-capabilities`) |

## What the numbers say

**Single-stream, this stack is close to the daily but no longer the equal of it on the same instrument.** Measured
head to head on one day (2,048 forced code tokens, greedy, same prompts, each engine alone on the box):
Flash-Next 207.0 t/s c1 against the daily's 253.9 — an 18 % gap, where the pair of separately-measured numbers
(217.6 vs 216) had suggested parity. The earlier comparison mixed client-side SSE rates with vLLM's Prometheus
counters and a `/completions` code prompt against a chat prompt; the head-to-head removes both confounds, and the
honest statement is "within a fifth, not equal".

**Under concurrency it is not in the same class.** Eight slots against sixteen, a 262k pool against 1.39M, and at
the same instrument the daily reads 868 t/s aggregate at c4 and 1,574 at c8 against 250 and 313. On the
deep-context fan-out arm — eight concurrent 38k-token requests — the daily holds a **1.76 s** first token and
**626.7 t/s** aggregate where this stack takes 63.6 s and 51.6 t/s. Layer splitting serialises the cards; there is
no tensor parallelism in this engine for this architecture, so the ceiling is structural, not a tuning miss.

## What would change the concurrency verdict

| lever | state | expected effect |
| --- | --- | --- |
| concurrency-indexed draft depth | **measured +35 % at c4** with byte-identical output (`2026-09-16-r340-ci-depth`); off by default, one config line to enable | recovers c1's depth-3 rate while dropping to depth 1 past two decoding jobs. No gain at c1 or c8; ceiling remains the layer split |
| QSA sparse multi-job | **measured +27 % at c2 and +40 % per stream at c4** on 152,761-token contexts, output byte-identical (`2026-09-16-r341-qsa`) | above the sparse threshold the captured QSA path is single-job and falls back to eager for bsz>1, which is exactly the deep-context concurrency case. Build needs a CUDA devel base; the recipe is in `kubernetes-home/flan/docker/Dockerfile.tabbyapi-qsa` |
| expert parallel for `qwen4_exp` | patch written, never executed; the authoring analysis lists upstream blockers | the only lever that attacks the layer-split ceiling directly — both cards computing every layer |
| more slots | `max_batch_size: 8` already raised from TabbyAPI's recurrent default of 4 | more slots cost recurrent VRAM; the page pool, not the slot count, binds at deep context |
| host KV tier | `sysmem_kv_cache: 0` | helps only after VRAM eviction; the deep-context admission test shows the pool is the constraint |

## Recommendation

Serve Flash-Next when the workload is one agent or a few, deep context, interactive latency: that is what it is
better at than the incumbent, and it is measurably at parity on the number that matters there.

Keep the vLLM 27B daily for multi-agent fan-out. Nothing measured here closes the 5× aggregate gap, and the four
levers above are engine work, not configuration: two are unexecuted patches and one is blocked on a build
toolchain. Promoting Flash-Next to the *only* daily today would trade a 1,476 t/s aggregate ceiling for a 305 t/s
one to gain nothing measurable at c1.

The stack itself — launcher, image, sampler policy, instruments, this document — is at daily standard. The
candidate engine is not, and that is a statement about `qwen4_exp` and ExLlamaV3's concurrency path, not about the
work done here.
