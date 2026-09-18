# PROMOTION — the gates for serving this stack

Serving a model all day on the `flan` box means: one published configuration, with the launcher, the measurements
and the failure modes written down, and a rebuild reproducible from the repositories alone.

This document records the gates `qwen3.8-flash-next-exl3-3.05bpw` on TabbyAPI + ExLlamaV3 has to pass to be served
that way, and where it stands on each.

## The gates, and where this stack stands

| # | gate | status | evidence |
| --- | --- | --- | --- |
| 1 | one-command boot, warm caches | **PASS** | `scripts/launch-flashnext.sh`; 11.2–11.5 s load, 0.3 s warmup, refuses to start on a missing checkpoint or image |
| 2 | single-stream decode | **207–214 t/s**, code c1 | 2,048 forced code tokens, greedy, c1, on the served configuration of 2026-09-17 (`2026-09-17-r460-moecoop-v2-ab`). The 2026-09-16 configuration read 207.0 on the same instrument |
| 3 | concurrent aggregate | **425–450 t/s at c4, 550–604 t/s at c8**, code | same instrument and date as gate 2. The 2026-09-16 configuration read 250 t/s at c4 and 313 at c8. Scaling is bounded by the layer split: `qwen4_exp` forbids TP=2, so the cards take turns (44–47 % utilisation at 227/218 W) |
| 4 | long-context retrieval | **PASS** | 5/5 at every planted position at 26.5k, 105.7k and 158.5k prompt tokens; a 152,761-token prompt costs 24 s cold and 0.43 s on a repeat |
| 5 | several deep-context agents at once | **PASS with a caveat** | 8/8 admitted and completed at ~628k of *unrelated* context against a 262k pool, TTFT 43–82 s as the queue drains (`2026-09-16-r345-pool`). Not measured on the vLLM route |
| 6 | a real agent turn | **PASS** | DSH session: 20 steps, 23 tool calls, file written, Chrome driven, screenshots read back through vision, 813k prompt tokens served from the prefix cache |
| 7 | reasoning channel, tool parsing, vision | **PASS** | reasoning in `reasoning_content`, `qwen3_coder` calls parsed as `tool_calls`, vision used in a live agent session |
| 8 | usage accounting matches the tokenizer | **PASS** | only after the R338 image patch: the engine under-reported long generations by ~5× |
| 9 | a sampler appropriate for the workload | **PASS** | only after R338: the server was serving untruncated T=1.0 to every client that sent no sampler |
| 10 | sustained load | **PASS** | 40 rounds at c4, drift 101.0 % of the start (63.5–65.5 t/s per stream, no error, no VRAM drift) |
| 11 | structured output (JSON schema) | **PASS** | content parses *and* satisfies the schema; the server log shows the grammar engaged. Tool-call args, vision on a red PNG and the reasoning channel also pass (`2026-09-16-r348-capabilities`) |
| 12 | GSM8K, lm-eval, as served | **0.9158** flexible-extract | 5-shot, `--apply_chat_template`, temperature 0, `max_gen_toks 8192`, `num_concurrent 4`, thinking on; n=1319 ±0.0077 (`2026-09-16-r368-gsm8k-1319`). The earlier n=200 reading, 0.925 ±0.019, contains this one in its interval |
| 13 | tool-eval 69×4 | **85.0 ± 2.9** | same CLI, same sampler, `--trials 4 --parallel 8` (`2026-09-16-r357-tooleval`). Not measured on the vLLM route |
| 14 | the promoted levers preserve quality | **PASS** | tool-eval 85.8 ± 3.1 (CI [83.5, 88.5]) against the baseline's 85.0 ± 2.9, overlapping intervals, while measuring +35 %/+78 % (`2026-09-16-r357-tooleval`) |
| 15 | agentic coding | **10/10 resolved on the first subset** | SWE-bench Verified, mini-SWE-agent 2.4.6 with the builtin `benchmarks/swebench.yaml`, scored by the official harness: 10 of the dataset's first ten instances resolved. n=10 and one repository (astropy), so the subset cannot resolve a few points; the stratified runs are in `docs/MEASUREMENTS.md` (`2026-09-16-r359-swebench-10`, `r360`) |

## What the numbers say

**Single-stream.** 207–214 t/s on 2,048 forced code tokens at c1, greedy, on the served configuration of
2026-09-17. Prose at c1 reads 160.6–165.3 t/s on the 2026-09-16 configuration: decode rate on this checkpoint is
content-dependent by about 2× (`docs/GOTCHAS.md` #9), so a decode figure without its kind is not comparable to
another one.

**Concurrency.** 425–450 t/s aggregate at c4 and 550–604 t/s at c8 on code, 2026-09-17. Eight slots and a
262,144-token pool are the ceiling on this box: `max_batch_size` 12 and 16 both fail to boot, and so does a
393,216-token cache. Layer splitting serialises the cards — there is no tensor parallelism in this engine for this
architecture — so the measured duty cycle is 45 %/38 % under a live eight-agent load and the ceiling is a property
of the engine rather than of the configuration.

**Against vLLM on the same checkpoint.** The vllm-exl3 route serves the same Flash-Next checkpoint through vLLM
main, TP2 on both cards. On 2026-09-18 its best profile (BF16 KV, MTP depth 3) read 131.6 t/s at c1 and 475.4 t/s aggregate at c4 on code,
against this stack's 207–214 and 425–450: 1.06–1.12× this stack at c4, 62–64 % of this stack's rate at c1. GSM8K on that route is
0.92 at n=200 against this stack's 0.9158 at n=1319. c8, prose, prefill and long-context are not measured on the
vLLM route. The full table is in `docs/MEASUREMENTS.md`, records in
`bench/results/2026-09-18-vllm-exl3-route/`.

## Levers on concurrent aggregate

| lever | state | expected effect |
| --- | --- | --- |
| concurrency-indexed draft depth | **measured +35 % at c4** with byte-identical output (`2026-09-16-r340-ci-depth`); off by default, one config line to enable | recovers c1's depth-3 rate while dropping to depth 1 past two decoding jobs. No gain at c1 or c8; ceiling remains the layer split |
| QSA sparse multi-job | **measured +27 % at c2 and +40 % per stream at c4** on 152,761-token contexts, output byte-identical (`2026-09-16-r341-qsa`) | above the sparse threshold the captured QSA path is single-job and falls back to eager for bsz>1, which is the deep-context concurrency case. Build needs a CUDA devel base; the recipe is in `kubernetes-home/flan/docker/Dockerfile.tabbyapi-qsa` |
| expert parallel for `qwen4_exp` | **assessed, not reachable from here**: all four blockers are code gaps, not GPU-only questions (`kubernetes-home/flan/patches/exllamav3/expert-parallel-qwen4exp-status.md`) | the only lever that touches the layer-split ceiling. Someone has to build QSA indexer transport, PLE module transport, a replica/output-selection policy and MTP adapters, then validate on a GPU. Upstream acceptance is optional; the engine work is not |
| more slots | `max_batch_size: 8` already raised from TabbyAPI's recurrent default of 4 | more slots cost recurrent VRAM; the page pool, not the slot count, binds at deep context. 12 and 16 fail to boot |
| host KV tier | `sysmem_kv_cache: 0` | helps only after VRAM eviction; the deep-context admission test shows the pool is the constraint. Measured flat (`2026-09-16-r358-hostkv`) |

## Recommendation

Serve this stack with both measured levers enabled. The configuration is validated and verified
(`IMG=tabbyapi:qsa-cid DRAFT_POLICY='[[2, 3], [8, 1]]'`: +35 % at short-context c4, +78 % at deep-context c4,
byte-identical output, tool-eval unchanged, the original agent request returning parsed tool calls, needle 5/5), and
it is the launcher's default as of 2026-09-16. The 2026-09-17 layers on top of it are listed with their gates at the
end of `docs/MEASUREMENTS.md`.

What the measurements do not cover: prose, c8, prefill and long-context on the vLLM route, and deep-context
fan-out, which is measured on this stack only (eight concurrent 38,283-token requests, `2026-09-16-r345-pool`) and
not on the vLLM route. Those are the open comparisons.

The layer-split ceiling is not addressable by configuration: measured duty cycle 45 %/38 % because `qwen4_exp`
forbids tensor parallelism, and expert parallelism — the only lever that would put both cards on every layer — is a
multi-prerequisite project whose first two prerequisites (QSA indexer transport, PLE module transport) now have
implemented, unvalidated patches and whose remaining two are at least as large.

The launcher, image, sampler policy, instruments and this document are in the state a served configuration
requires. The three instrument defects found on 2026-09-16 are recorded in `docs/GOTCHAS.md` because two of them
produced wrong numbers before being caught.
