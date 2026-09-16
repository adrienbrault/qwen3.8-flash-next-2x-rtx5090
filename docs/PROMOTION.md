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
| 2 | single-stream decode at or near the incumbent | **PASS, but not at parity** | on one instrument, one day: **207.0 t/s** against the daily's **253.9** at code c1 — within a fifth. (An earlier pair, 217.6 against 216, came from two different instruments and is superseded.) |
| 3 | concurrent aggregate | **FAIL against the incumbent** | same instrument: 250 t/s aggregate at c4 and 313 at c8 against the daily's **868** and **1,574**; scaling 1.60× from c1 because `qwen4_exp` forbids TP=2 and the cards take turns (44–47 % utilisation at 227/218 W) |
| 4 | long-context retrieval | **PASS** | 5/5 at every planted position at 26.5k, 105.7k and 158.5k prompt tokens; a 152,761-token prompt costs 24 s cold and 0.43 s on a repeat |
| 5 | several deep-context agents at once | **PASS with a caveat** | 8/8 admitted and completed at ~628k of *unrelated* context against a 262k pool, TTFT 43–82 s as the queue drains — against the daily's 8/8 at **1.76 s** TTFT and 626.7 t/s aggregate |
| 6 | a real agent turn | **PASS** | DSH session: 20 steps, 23 tool calls, file written, Chrome driven, screenshots read back through vision, 813k prompt tokens served from the prefix cache |
| 7 | reasoning channel, tool parsing, vision | **PASS** | reasoning in `reasoning_content`, `qwen3_coder` calls parsed as `tool_calls`, vision used in anger |
| 8 | honest usage accounting | **PASS** | only after the R338 image patch: the engine under-reported long generations by ~5× |
| 9 | a sampler appropriate for the workload | **PASS** | only after R338: the server was serving untruncated T=1.0 to every client that sent no sampler |
| 10 | sustained load | **PASS** | 40 rounds at c4, drift 101.0 % of the start (63.5–65.5 t/s per stream, no error, no VRAM drift) |
| 11 | structured output (JSON schema) | **PASS** | content parses *and* satisfies the schema; the server log shows the grammar engaged. Tool-call args, vision on a red PNG and the reasoning channel also pass (`2026-09-16-r348-capabilities`) |
| 12 | quality on the daily's own GSM8K instrument | **0.9158** flexible-extract as served against the daily's **0.985**; **6.9 points**, several standard errors | same harness parameters as R299b's as-served arm, thinking on; n=1319 ±0.0077 (`2026-09-16-r368-gsm8k-1319`), the earlier n=200 ±0.019 reading being within its own interval of this one |
| 13 | quality on the daily's tool-eval 69×4 | **85.0 ± 2.9** against the daily's **91** | same CLI, same sampler, `--trials 4 --parallel 8` (`2026-09-16-r357-tooleval`) |
| 14 | the promoted levers preserve quality | **PASS** | tool-eval 85.8 ± 3.1 (CI [83.5, 88.5]) against the baseline's 85.0 ± 2.9, overlapping intervals, while measuring +35 %/+78 % (`2026-09-16-r357-tooleval`) |
| 15 | agentic coding, the job the box exists for | **PASS on the first subset** | SWE-bench Verified, daily's harness, official scorer: **10/10 resolved** against the daily's **8/10** on the same instances — the two the daily failed, this seat resolved. n=10, one repository, so the margin is noise; a stratified 18-instance run across six repositories is in flight (`2026-09-16-r359-swebench-10`, `r360`) |

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
| expert parallel for `qwen4_exp` | **assessed, not reachable from here**: all four blockers are code gaps, not GPU-only questions (`kubernetes-home/flan/patches/exllamav3/expert-parallel-qwen4exp-status.md`) | would be the only lever that touches the layer-split ceiling. Someone has to build QSA indexer transport, PLE module transport, a replica/output-selection policy and MTP adapters, then validate on a GPU. Upstream acceptance is optional; the engine work is not |
| more slots | `max_batch_size: 8` already raised from TabbyAPI's recurrent default of 4 | more slots cost recurrent VRAM; the page pool, not the slot count, binds at deep context |
| host KV tier | `sysmem_kv_cache: 0` | helps only after VRAM eviction; the deep-context admission test shows the pool is the constraint |

## Recommendation

The decision has moved during this session, and the honest state is this. **On the workload the box exists for —
agentic coding — this seat is ahead of the current daily on matched instances: 36 of 38 against 20, with sixteen
discordant pairs all in this seat's favour (p ≈ 2⁻¹⁶).** On short-answer reasoning and tool-calling the daily is ahead
by **6.9 points on GSM8K** (0.985 against 0.9158, the latter now at n=1319 with a ±0.0077 interval, so the gap is
several standard errors and not an artefact of the sample) and by about six points on tool-eval, which remains at its
original n.

So the verdict is no longer "serve it for one agent, keep the daily for fan-out". It is: **serve this seat with both
measured levers on, and decide by workload.** The configuration to serve is validated and verified
(`IMG=tabbyapi:qsa-cid DRAFT_POLICY='[[2, 3], [8, 1]]'`: +35 % at short-context c4, +78 % at deep-context c4,
byte-identical output, tool-eval unchanged, the original agent request returning parsed tool calls, needle 5/5), and
it is now the launcher's default.

What still argues for keeping a vLLM instance available: aggregate throughput at c4/c8 (250/313 t/s against the
daily's 868/1,574) and the two six-point quality gaps on GSM8K and tool-eval. If the box is used for fan-out of short
tool-calling turns, the daily is the better engine; if it is used the way it has been used today — one or a few
coding agents at long context — this seat wins the measurement that matters and there is no reason to keep it idle.

The structural ceiling stands and is not addressable by configuration: measured duty cycle 45 %/38 % because
`qwen4_exp` forbids tensor parallelism, and expert parallelism — the only lever that would put both cards on every
layer — is a multi-prerequisite project whose first two prerequisites (QSA indexer transport, PLE module transport)
now have implemented, unvalidated patches and whose remaining two are at least as large.

The stack itself — launcher, image, sampler policy, instruments, this document — is at daily standard, and the
three instrument defects found today are recorded in `docs/GOTCHAS.md` because two of them had already produced
confident wrong numbers before being caught.
