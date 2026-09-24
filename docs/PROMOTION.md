# PROMOTION — the gates a candidate passes before it is served

A candidate is a launcher: image, environment flags, pool, slots and checkpoint. The promotion unit boots it on the serving port with no environment overrides, runs the gates below, and only then replaces `launch-flashnext.sh`, keeping the previous launcher for rollback. Examples: [`scripts/r514-promote-r4i.sh`](../scripts/r514-promote-r4i.sh), [`scripts/r517-promote-stack.sh`](../scripts/r517-promote-stack.sh), [`scripts/r518-slots6.sh`](../scripts/r518-slots6.sh).

## The stack track since 2026-09-24

A change that cannot alter output is judged on identity and on regression, not on effect size, and changes of this kind are promoted in batches. The track applies to a change that is flag-gated and off by default, safe under CUDA-graph capture and within the free-VRAM headroom, and bitwise-identical. A change that alters numerics keeps the gates in the next section and is promoted on its own.

**Instrument.** Speed on this track is measured in process, not by booting the server. A harness loads the served model and environment in one container, decodes at 4,096 tokens of context for a fixed batch and draft depth, times 32 captured iterates with CUDA events and no tracer, and records ms per iterate and a hash of the generated sequences. The shapes are the served ones: 1 stream at depth 3 (4 verify rows), 4 streams at depth 3 and 8 streams at depth 1 (16 verify rows each). An A/A control on the harness (the same kernels in both arms, 5 rounds) put its per-round spread at ±1.2 ms at 4 streams (±5 %), ±0.9 ms at 8 streams (±4.5 %) and ±1.1 ms at 1 stream (2026-09-24, [R702](../bench/results/r702-hcfast-r2.md)); a 1 to 2 % effect at 4 or 8 streams therefore cannot show as 5 of 5 same-sign rounds and is read from the batched served gate. An earlier version of this paragraph said the harness separates about 0.5 %, from one pair of R698 cells. Each run also records the same batch at draft depth 0, whose OFF standard deviation is 0.03 to 0.05 ms against 0.16 to 0.62 ms with drafting (2026-09-24, results `2026-09-24-r705-pdl-gate`); a change that touches kernels outside the draft is read there too. A served 1-stream rate on the canonical gate moves more than that between pairs of one session, whose prompts differ by salt: the unchanged configuration read 297.4, 301.2 and 195.1 tokens/s on leg A at 1 stream across R701's three pairs (2026-09-24, results `2026-09-24-r701-stack-gate`).

**Acceptance of one change.**

- Identity: the generated-sequence hashes are identical in every OFF/ON pair, and greedy output on the served launcher (6 prompts, one of about 100,000 tokens) shows 0 divergences from the reference with the flag on and with it off.
- Speed: at least 5 interleaved OFF/ON pairs per shape. All pairs of one sign is a gain or a regression; mixed signs is flat. A same-sign regression at 4 or 8 streams rejects the change. At 1 stream a same-sign regression of at most 2 % of the in-process mean is accepted when the ms saved at 4 or 8 streams exceed the ms added at 1 stream; that 1-stream cost is recorded as an open item for the next kernel round.
- A kernel-level microbenchmark explains the mechanism and chooses tile settings. It does not gate.
- Harness stall: about a third of the in-process runs contain one iterate 10 to 53 ms slower than the rest, at no fixed arm; it is not Python garbage collection (2026-09-24, results `2026-09-24-r715-harness-gc-aa`). Pairs are also read stall-excluded: every iterate slower than its run's median plus 8 ms is dropped from both runs of the pair, and the dropped count is reported. A gate reads the signs on the stall-excluded means or on per-iterate medians, not on raw means alone ([R712](../bench/results/r712-latchain-r1.md), [R710b](../bench/results/r710b-densegemm-gv.md)).

**Promotion of the batch.** The accepted changes are built into one image and promoted together, every two or three changes or at the next image bump:

1. A headroom boot: the greedy set, a 1-to-8-stream decode ramp, and GPU 0 free VRAM no more than 32 MiB below the served configuration's.
2. The canonical gate ([`bench/fn_gate.sh`](../bench/fn_gate.sh), 3 recorded rounds) on alternating boots of the served launcher and the candidate, which checks that the per-change deltas compose.
3. Promotion when no leg-A or leg-B cell has a mean ON/OFF below 0.99 (0.98 for leg A at 1 stream) and the mean over the cells is above 1.

This replaces the two-boots-per-arm speed rule below for changes on the track. The first promotion on it is [R701](../bench/results/r701-stack-r2.md), the second [R716b](../bench/results/r716b-stack-r3.md).

### Identity at every served shape since 2026-09-24

The greedy set runs 1 stream, so it covers verify rows 1 and 4 only. The served draft policy `[[4, 3], [5, 2], [8, 1]]` verifies at 4, 8, 12, 16, 15, 12, 14 and 16 rows with 1 to 8 active jobs, and drafts at 1 to 8 rows. A batch gate proves identity at every one of them:

- In process, logits-level: `lc_model_parity` ([`tests/latchain/lc_model_parity.py`](../docker/overlays/stack-r3/tests/latchain/lc_model_parity.py)) records every forward's digest and the generated tokens at batch 1 depth 3, 2 depth 3, 3 depth 3, 4 depth 3, 5 depth 2, 6 depth 1, 7 depth 1 and 8 depth 1, 4,096 tokens of context, for OFF, the candidate, OFF2 (A/A) and the served image with the served environment. All four identical at all 8 shapes is the pass. Batch 2 depth 3 needs `--gpu-split 28,30` on this pack; at 30,30 it runs out of memory on GPU 0 in every arm ([R716b, R716c](../bench/results/r716b-stack-r3.md)).
- Served, multi-stream greedy (`greedy_streams.py --compare` in [`docker/overlays/stack-r3/tools/`](../docker/overlays/stack-r3/tools/greedy_streams.py)), over the boots of the served launcher (A) and of the candidate (B), per concurrency level: when every A/A pair is identical, every candidate stream must be identical; when the mean A/A divergence exceeds 25 % of the streams, the level is INCONCLUSIVE and identity comes from the in-process cells above, and a gate without those cells is incomplete; in between, a pooled label-permutation test on mean between-group minus mean within-group divergence decides, together with a count of candidate outputs that no A boot produced. Flags-off boots of the candidate image are passed as controls and reported, not pooled with the candidate. An error on an A boot voids the comparison.
- On the served configuration of 2026-09-24, 2 streams are deterministic and 4 and 8 streams are not: 60 % and 76 % of streams differ between boots of the same launcher, because which requests share a verify batch depends on arrival timing and near-tied tokens flip. The rule this replaces judged each candidate boot against the largest of three A/A divergences from one reference boot, and on R716b's boots it failed 64 % of role assignments and failed the served configuration against itself in 5 of 12.

## The gates since 2026-09-18

| gate | what it checks | pass |
| --- | --- | --- |
| G0 sampler fallbacks | presence penalty, repetition penalty and stop strings on greedy requests, hashed | byte-identical to the served image |
| G1 boot and fingerprints | image, environment and config as intended; greedy 256-token reply at c1 and 128-token reply to a 30k prompt, hashed | canonical hashes, or a stated change for patches that change numerics |
| G1b cold prefill (prefill changes only) | salted 60k and 120k prompts, one invocation per length | a floor set from the evidence run |
| G2 agentic-edit | six real files rewritten with a small edit, greedy and sampled, c1 and c4 ([`bench/agentic-edit.py`](../bench/agentic-edit.py)) | 6/6 in all four modes, server alive |
| G3 needles | five planted passphrases at 131k and 240k prompt tokens ([`bench/needle.py`](../bench/needle.py)) | 5/5 at both |
| G4 tool-eval | [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) 69 scenarios × 4 | mean ≥ 82 |

A patch that changes numerics also passes GSM8K n=500 through [`bench/nostop_proxy.py`](../bench/nostop_proxy.py), paired per question against the served configuration ([R509](../bench/results/r509-gsm8k-nostop.md) explains why the proxy), before it reaches the promotion unit. Speed is judged on two boots per arm (OFF / ON / OFF2 / ON2) and on the multi-prompt probe when output changes ([R487](../bench/results/r487-pool-393k.md)).

## 2026-09-16: the first promotion of this stack

The rest of this document records the gates the 3.05 bpw pack on TabbyAPI + ExLlamaV3 passed when it was first served, and where it stood on each.

### The gates, and where this stack stood

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
| 15 | agentic coding | **10/10 resolved on the first subset** | SWE-bench Verified, mini-SWE-agent 2.4.6 with the builtin `benchmarks/swebench.yaml`, scored by the official harness: 10 of the dataset's first ten instances resolved. n=10 and one repository (astropy), so the subset cannot resolve a few points; the stratified runs are in [R359](../bench/results/r359-swebench.md) (`2026-09-16-r359-swebench-10`, `r360`) |

### What the numbers say

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
0.945 at n=200 on the depth-3 profile (0.92 at depth 2) against this stack's 0.9158 at n=1319. c8, prose, prefill and long-context are not measured on the
vLLM route. The results are indexed in [`bench/RESULTS.md`](../bench/RESULTS.md), records in
`bench/results/2026-09-18-vllm-exl3-route/`.

### Levers on concurrent aggregate

| lever | state | expected effect |
| --- | --- | --- |
| concurrency-indexed draft depth | **measured +35 % at c4** with byte-identical output (`2026-09-16-r340-ci-depth`); off by default, one config line to enable | recovers c1's depth-3 rate while dropping to depth 1 past two decoding jobs. No gain at c1 or c8; ceiling remains the layer split |
| QSA sparse multi-job | **measured +27 % at c2 and +40 % per stream at c4** on 152,761-token contexts, output byte-identical (`2026-09-16-r341-qsa`) | above the sparse threshold the captured QSA path is single-job and falls back to eager for bsz>1, which is the deep-context concurrency case. Build needs a CUDA devel base; the recipe is in `kubernetes-home/flan/docker/Dockerfile.tabbyapi-qsa` |
| expert parallel for `qwen4_exp` | **assessed, not reachable from here**: all four blockers are code gaps, not GPU-only questions (`kubernetes-home/flan/patches/exllamav3/expert-parallel-qwen4exp-status.md`) | the only lever that touches the layer-split ceiling. Someone has to build QSA indexer transport, PLE module transport, a replica/output-selection policy and MTP adapters, then validate on a GPU. Upstream acceptance is optional; the engine work is not |
| more slots | `max_batch_size: 8` already raised from TabbyAPI's recurrent default of 4 | more slots cost recurrent VRAM; the page pool, not the slot count, binds at deep context. 12 and 16 fail to boot |
| host KV tier | `sysmem_kv_cache: 0` | helps only after VRAM eviction; the deep-context admission test shows the pool is the constraint. Measured flat (`2026-09-16-r358-hostkv`) |

### Recommendation

Serve this stack with both measured levers enabled. The configuration is validated and verified
(`IMG=tabbyapi:qsa-cid DRAFT_POLICY='[[2, 3], [8, 1]]'`: +35 % at short-context c4, +78 % at deep-context c4,
byte-identical output, tool-eval unchanged, the original agent request returning parsed tool calls, needle 5/5), and
it is the launcher's default as of 2026-09-16. The 2026-09-17 layers on top of it are listed with their gates at the
[`docs/HISTORY.md`](HISTORY.md).

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
