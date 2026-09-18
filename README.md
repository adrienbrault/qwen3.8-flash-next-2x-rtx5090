# Qwen3.8-Flash-Next on 2× RTX 5090 (ExLlamaV3 + TabbyAPI)

The serving configuration, launcher, image recipe, instruments and measurements for [`qwen3.8-flash-next-exl3-3.05bpw`][ckpt], turboderp's 3.05 bpw [EXL3][exl3-convert] quantisation of [Qwen3.8-Flash-Next][qwen-hf], served by [TabbyAPI][tabby] on [ExLlamaV3][exl3] v1.5.0 across two RTX 5090 cards with a 262,144-token window, 8-bit KV, vision, reasoning, tool calls, structured output and the checkpoint's own MTP draft head all on.

Every number here was measured on the `flan` box on the date given, next to the results directory it came from; the instruments are in [`bench/`][bench]. Nothing is a projection. Three numbers this track published before 2026-09-16 were measured wrong, which is why the instruments record the server's token count beside the client's and why [`docs/GOTCHAS.md`][gotchas] exists.

**This repository is not published.** The Flash-Next / ExLlamaV3 track stays private; do not push it or mirror it into any public repository. See [`CLAUDE.md`][claude-md].

## What is served (since 2026-09-17 12:45 CEST)

| | value | where it is set |
| --- | --- | --- |
| checkpoint | `qwen3.8-flash-next-exl3-3.05bpw` (turboderp, exllamav3 1.4.4 converter, `mul1` codebook) | [`scripts/launch-flashnext.sh`][launcher] |
| image | `tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap-ppipe-nosync-mtpfix2-moecoopv2` | [`docker/README.md`][docker-readme], layer by layer |
| engine env (opt-in patches) | `EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1` | launcher `EXTRA_ENV` |
| cards | layer split `gpu_split: [30, 30]`, no tensor parallelism (forbidden for `qwen4_exp`) | [`docs/CONFIG.md`][config] |
| window / pool / slots | 262,144 tokens / 262,144-token page pool at 8-bit KV / 8 slots | [`docs/CONFIG.md`][config] |
| drafting | MTP, depth 3 at c1–c4 and depth 1 above: `draft_num_tokens_by_batch: [[4, 3], [8, 1]]` | launcher `DRAFT_POLICY` |
| reasoning / tools / vision | `reasoning: true`, budget 32,768; `tool_format: qwen3_coder`; vision on | [`docs/CONFIG.md`][config] |
| sampler fallbacks | T 0.6, top_k 20, top_p 0.95, `force: false` (a client that sends its own sampler keeps it) | [`docs/CONFIG.md`][config] |
| port | 8022 | launcher `PORT` |

Verified as served, not asserted: the c1 and 30k-prompt greedy fingerprints are byte-identical to the layer each patch was admitted on, the five gates (GSM8K, tool-eval, needle, agent replay, greedy identity) passed on the final image ([`2026-09-17-r461-gates-moecoopv2`][r461]), and the agent request that originally failed returns a parsed tool call with its reasoning in `reasoning_content` ([`2026-09-16-r356-promoted`][r356]).

## The numbers

Every decode rate in this repository names its kind, code or prose, because the two are not interchangeable on this checkpoint: draft acceptance tracks how predictable the text is, and on 2026-09-16 code decoded at 208 t/s against 161–165 t/s for prose at c1 ([r339 gates][r339]). Unless a row says prose, a decode rate here is **code**: `fn_bench` ([`bench/probe.py`][probe]) with `--kind code`, 2,048 forced tokens per request, greedy, steady-state, aggregate over the streams. Prefill is stated in prompt tokens per second with the wall time beside it. The vLLM column is the same checkpoint served by vLLM through the [vllm-exl3][vllm-exl3] route (TP2, CUDA graphs, vLLM main, 2026-09-18, [`2026-09-18-vllm-exl3-route`][vllm-route]); its cells name the KV dtype and MTP depth because the route has no single served configuration yet.

| | ExLlamaV3, served now (2026-09-17) | ExLlamaV3, 2026-09-16 | vLLM, same checkpoint (2026-09-18) |
| --- | --- | --- | --- |
| decode c1, code | **207–214 t/s** | 207.0 | 131.6 (BF16 KV, MTP depth 3); 121.7 (depth 2); 117.0 (fp8 KV, MTP depth 1) |
| decode c4 aggregate, code | **425–450 t/s** (106–113 per stream) | 250.2 (63.8) | 475.4 (BF16 KV, MTP depth 3); 421.4 (depth 2); 339.2 (fp8 KV, MTP depth 1) |
| decode c8 aggregate, code | **550–604 t/s** (69–76 per stream) | 313.2 (40.5) | not measured (the route serves 4 sequences until its next image) |
| decode c1, prose | not yet measured on this configuration (queued 2026-09-18, `r477-daily-prose-code`) | 160.6–165.3 aggregate ([r339][r339]) | not measured |
| decode c4 aggregate, prose | not yet measured on this configuration (same run) | — | not measured |
| prefill, 27,501-token prompt | **7,700–7,820 t/s** (3.52–3.57 s) | 4,880–4,990 t/s (5.51–5.64 s) | not measured |
| prefill, 110,081-token prompt | **8,380–8,390 t/s** (13.12–13.14 s) | 4,920–4,940 t/s (22.29–22.36 s) | not measured |
| TTFT, 152,761-token prompt, repeat | 0.43 s (prefix cache; cold 24 s = 6,365 t/s) | same | not measured |
| long-context retrieval | 5/5 at every planted position, 26.5k / 105.7k / 158.5k prompt tokens | same | not measured |
| pool | 262,144 tokens, 8-bit KV (327,680 and 393,216 do not boot) | same | 95,183 (BF16 KV, MTP depth 3); 108,651 (depth 2); 159,744 (fp8 KV, MTP depth 1); 309,657 (fp8 KV, no MTP) |
| boot to serving | 11.2–11.5 s load + 0.3 s warmup, warm caches | same | 390 s, warm caches (in-image preflight 100 s, weights 55 s, warmup 90 s) |
| VRAM resident | 31.9 GB / 30.1 GB of 32.6 GB per card | same | both cards, TP2 |

Sources: the served-now column is the promotion ladder in [`docs/MEASUREMENTS.md`][measurements] ([`2026-09-17-r428-hcmix2-stack-ab`][r428], [`2026-09-17-r442-ppipe-memfix-ab`][r442], [`2026-09-17-r460-moecoop-v2-ab`][r460]); the pool ceiling is [`2026-09-17-r452-exl3-cache-bits`][r452]; retrieval and prefix-cache TTFT are [`2026-09-16-r339-gates`][r339-gates] and [`2026-09-16-r343-depth`][r343]; the prefill token counts are the probe's fixed prompts tokenized with the checkpoint's tokenizer (23,000 and 92,000 words); boot and footprint are from the launcher log; the vLLM column is [`2026-09-18-vllm-exl3-route`][vllm-route].

Between 2026-09-16 and 2026-09-17 the c1 rate did not move (the c1 step is bounded by the layer split and the host launch gap, see below), c4 aggregate rose 1.7×, c8 rose 1.8×, and prefill rose 1.6–1.7×. Against vLLM on the same checkpoint, the vLLM route's best profile reads 1.06–1.12× this stack's c4 aggregate and 0.62–0.64× its c1; the route's pool with fp8 KV is 1.2× this stack's. Decode rate on this checkpoint depends on what is being generated (code decoded 1.3× faster than prose at c1 on 2026-09-16, [r339][r339]; draft acceptance tracks predictability), so a rate without its kind is not comparable to another one.

## How it got here: the promotion ladder

Every layer is one patch on the previous image, each admitted by its own gate, most of them with byte-identical greedy output at c1 and on a 30k prompt. The patches are in [`docker/`][docker-readme]; provenance and licences in [`THIRD_PARTY.md`][third-party].

| promoted (CEST) | layer | what it does | gate | c1 / c4 / c8, code | results |
| --- | --- | --- | --- | --- | --- |
| 2026-09-16 | QSA multi-job + concurrency-indexed draft depth | multi-job sparse attention above the QSA threshold; draft depth by batch size | byte-identical; +27 %/+40 % at deep-context c2/c4, +35 % at c4 | 207 / 250 / 313 | [`r341-qsa`][r341], [`r340-ci-depth`][r340], [`r354-combined`][r354] |
| 2026-09-16 | [exllamav3#337][pr337] (creslinux) | keeps the current CUDA device on the module's device during a layer-split forward | byte-identical, flat except at 152k prompts (207.5 vs 181.7) | — | [`r362-pr337`][r362] |
| 2026-09-17 02:15 | [`bszn16.patch`][bszn16] + policy `[[4, 3], [8, 1]]` | fused MoE decode path admits 16 rows instead of 8, so depth-1 drafts at c8 stay on the fast path | c1 fingerprint identical; GSM8K and tool-eval under concurrency | 203 / 375 / 443 | `2026-09-17-r414-*` |
| 03:15 | [`coopwide.patch`][coopwide] | wide stage-B MoE tile at ≥ 128 slots | byte-identical; c8 +7.5 % | 206–209 / 369–376 / 469–478 | [`r421-coopwide-ab`][r421] |
| 04:32 | [`hc-mix-v2-r2.patch`][hcmix] + [`hostgap-gated_delta_net.py`][hostgap] | bit-exact rewrite of the hyper-connection mixer kernels; host-side gaps removed from the GDN decode path | identical at `MIN_R 1`; c4 +12 %, c8 +8 % | 213–218 / 422–434 / 537–548 | [`r428-hcmix2-stack-ab`][r428] |
| 07:35 | [`prefill-pipeline.patch`][ppipe] + [`prefill-nosync`][nosync] + [`prefill-pipeline-mtp`][mtpfix] overlays | two-card prefill pipeline for the layer split, without blocking host syncs, with the MTP eligibility and free-VRAM guard fixes | c1 and 30k fingerprints identical | 213 / 430 / 540; prefill of the 27,501-token prompt 4,900 → 7,800 t/s | [`r442-ppipe-memfix-ab`][r442], gates [`r446-gates-ppipe`][r446] |
| 12:45 | [`moe-coop-v2`][moecoop] overlay | bit-exact V2 of the fused MoE decode kernel: bounded work loops, batched completions | bit-exact at R = 1..16 in the kernel test, fingerprints identical, five gates | 207–214 / 425–450 / 550–604 | [`r460-moecoop-v2-ab`][r460], [`r461-gates-moecoopv2`][r461] |

Measured and not promoted: MoE coop mode 3 (bit-identical, 2–3 % slower, [`r462-moecoop-v3-ab`][r462]); [exllamav3#303][pr303] MTP hot vocabulary (inapplicable on a two-card layer split by construction, [`r377-hotvocab-on`][r377]); [exllamav3#246][pr246] (changes numerics for a prefill gain within noise) and [exllamav3#290][pr290] (output-neutral, no gain), both in [`r365-kernels`][r365]; the host KV tier (flat, [`r358-hostkv`][r358]); our own 32-row MoE decode envelope (correct, no effect, [`r366-ourkernel`][r366]); dynamic draft (184 vs 191 t/s at c1, code). Each verdict is in [`docs/MEASUREMENTS.md`][measurements] with its arms.

## Quality

Measured on this stack as served with [lm-eval][lm-eval], [tool-eval-bench][tool-eval] and [mini-SWE-agent][mini-swe]. The vLLM route on the same checkpoint has one quality figure so far, GSM8K at n=200.

| gate | ExLlamaV3 (this stack), as served | vLLM, same checkpoint | results |
| --- | --- | --- | --- |
| GSM8K 5-shot ([lm-eval][lm-eval]), thinking on, n=1319 | **0.9158** (±0.0077) | — | [`2026-09-16-r368-gsm8k-1319`][r368] |
| GSM8K, n=200, on the final image | 0.935 | 0.945 (BF16 KV, MTP depth 3); 0.92 (depth 2); 0.93 (full CUDA graphs, depth 2) | [`r461-gates-moecoopv2`][r461], [`2026-09-18-vllm-exl3-route`][vllm-route] |
| [tool-eval-bench][tool-eval] 69×4 | **85.8 ± 3.1** (baseline 85.0 ± 2.9) | — | [`2026-09-16-r357-tooleval`][r357] |
| SWE-bench Verified, 49 matched instances, [mini-SWE-agent][mini-swe] bash-only, official scorer | **46 / 49** resolved | — | [`2026-09-16-r359-swebench`][r359] |
| structured output ([llguidance][llguidance] `json_schema`, `response_format`, `regex_pattern`), thinking on and off, c4 | PASS | — | [`2026-09-17-r453-exl3-structured`][r453] |
| tool-call parsing, vision, reasoning channel | PASS | — | [`2026-09-16-r348-capabilities`][r348] |
| stamina, 40 rounds at c4 | drift 101.0 % of the start, no error, no VRAM drift | — | [`2026-09-16-r347-soak`][r347] |

The SWE-bench subsets (68 instance-runs over 49 unique instances) were selected on earlier outcomes, not sampled, so 46/49 is a tally on those instances and not an estimate of a full-set score. The 19 instances that ran both before and after the enablement scored identically, which is the control for the promotion. Full account in [`docs/MEASUREMENTS.md`][measurements]; the serving decision, gate by gate, in [`docs/PROMOTION.md`][promotion].

The [2.05 bpw checkpoint][ckpt] from the same converter decodes at about the 3.05's speed (code, c1: 187.4 t/s) and scores 17 GSM8K points lower on this box (0.765 against 0.935, n=200; `2026-09-16-r395-baseline-205`, `2026-09-16-r396-gsm8k-305-control`). It is not served.

## The ceiling, and what has been tried against it

`qwen4_exp` raises `NotImplementedError: Tensor-parallel is not currently implemented for Qwen4ExpForConditionalGeneration` in this engine, so the two cards take turns over their own layers. A single request keeps each card 44–47 % busy at 227/218 W against 600/575 W limits; under eight bursty agents it reads 45 % / 38 %. That idle half is the price of avoiding a per-layer all-reduce over PCIe, and it is recovered only by concurrency: eight slots in a 262k pool.

- **Expert parallelism** was built and served on the 2.05 bpw checkpoint, all four prerequisites patched (QSA indexer transport, PLE module transport, MTP adapters, replica/output policy). It recovers nothing: −9.5 % at c1, −4.5 % at c4, +2.5 % at c8 against the layer split at the same settings (`2026-09-16-r408-ep-served`); deterministic and at GSM8K parity, but not bit-identical to the layer split (`2026-09-16-r410-ep-correctness`). Parked.
- **Upstream topology PRs**: [exllamav3#299][pr299] (per-layer split or fused-uniform QKV topology) does not distribute decode across the cards and needs a fresh conversion from BF16 weights; [exllamav3#284][pr284] (fused additive kernels) has no production caller. Both closed with reasons in [`docs/PROMOTION.md`][promotion].
- **The remaining lever on the served engine is host/device overlap.** A `torch.profiler` capture of the served stack puts the launch gap between decode steps at 2.5 ms of a 10 ms c1 step (the harness's own fixed prompt, not the code or prose probe), of which 1.6 ms is untraced host work and 0.7 ms an explicit 8-byte device-to-host sync. The next kernel decision is made on that trace, not on dispatch-logic arithmetic.
- **vLLM on the same checkpoint** ([vcruz305/vllm-exl3][vllm-exl3], TP2, CUDA graphs, MTP) is the comparison column above: on vLLM main (2026-09-18) its best profile (BF16 KV, MTP depth 3) reads 131.6 t/s at c1 and 475.4 aggregate at c4 on code against this stack's 207–214 and 425–450 (c4 above this stack, c1 at 62 %), its pool with fp8 KV is 309,657 tokens against 262,144 here, and its c8, prose, prefill and long-context figures are not measured yet. It is not served and its patch series is not in this repository.

## Two defects this stack had, and what they cost

**Every client that sent no sampler was served at temperature 1.0, untruncated** (fixed 2026-09-16, [GOTCHAS 5][gotchas]). TabbyAPI has no sampling fallbacks unless a preset is named and it says so at boot; a harness that sends only `max_tokens` had its reasoning degrade into mixed-language text, emit an end-of-thinking tag inside that text, and deliver the rest as the visible answer. The launcher now writes a preset of fallbacks.

**The engine under-reported its own generation by about 5×** for anything past the 2,048-token requeue boundary (fixed 2026-09-16, one line, asserted at image build time, [GOTCHAS 4][gotchas]). A 17,544-token generation was reported as 3,500 tokens at 32.7 t/s. It made this server look five times slower than it is, and it is why [`bench/probe.py`][probe] records the server's count and the client's frames side by side.

## Reproducing a boot

```sh
ssh flan 'bash -s' < scripts/launch-flashnext.sh          # serve on :8022
PORT=8023 bash scripts/launch-flashnext.sh                # a second instance
IMG=tabbyapi:qsa-cid-pr337 EXTRA_ENV= bash scripts/launch-flashnext.sh   # the 2026-09-16 image, patches off
STOP=1 bash scripts/launch-flashnext.sh                   # stop
```

The launcher refuses to start if the checkpoint or the image is missing, stops any other engine holding the cards first, waits for the cards to drain, mounts the Triton and coop-autotune caches (a cold first inference costs 34–50 s, a warm one 0.3 s) and logs the warmup cost. Rebuilding the image chain, running every measurement and handing the box back are in [`docs/RUNBOOK.md`][runbook].

## Repository map

| path | what it is |
| --- | --- |
| [`scripts/launch-flashnext.sh`][launcher] | the launcher: writes the served config and the sampler preset, mounts both, starts the container, warms the kernels |
| [`scripts/check-public-hygiene.sh`][hygiene] | fails a commit that carries private addresses, local paths or credential-shaped strings, and runs the prose check on staged Markdown |
| [`scripts/check-prose.sh`][prose] | fails on evaluative words, rhetorical devices, exclamation marks and questions in running text (the rule is in [`CLAUDE.md`][claude-md]) |
| [`docker/`][docker-readme] | the served image, layer by layer: Dockerfiles, the patches, and the SHA-pinned overlays with their installers and kernel tests |
| [`bench/probe.py`][probe] | decode / concurrency / depth instrument; forces length with `min_tokens` (the only field that works here), one JSONL line per request |
| [`bench/needle.py`][needle] | long-context retrieval gate at five planted positions per depth |
| [`bench/capabilities.py`][capabilities] | JSON schema, tool parsing, vision and reasoning-channel checks |
| [`bench/summarize.py`][summarize] | reads the records, never a printed summary |
| [`bench/r*.sh`][bench] | every runner, one per results directory, under the GPU lock |
| [`bench/results/`][bench-results] | the raw records for the runs quoted here, 2026-09-16 to 2026-09-18 |
| [`docs/CONFIG.md`][config] | every setting and why it has that value, including which are fit constraints |
| [`docs/MEASUREMENTS.md`][measurements] | the numbers with their conditions, and the promotion ladder |
| [`docs/PROMOTION.md`][promotion] | the serving decision, gate by gate |
| [`docs/GOTCHAS.md`][gotchas] | the traps, as "what it looks like" against "what it is" |
| [`docs/RUNBOOK.md`][runbook] | serve, measure, rebuild and hand the box back from this repository alone |
| [`docs/REVIEW_HANDOFF.md`][review] | an outside review of the harnesses and the fixes it produced |
| [`THIRD_PARTY.md`][third-party] | where every input came from and under which licence |
| [`CLAUDE.md`][claude-md] | the working agreement: visibility, prose rules, box rules, measurement rules |

## Attribution and licence

The original work here (documentation, instruments, launcher, overlay installers, measurements) is [MIT][license]. The model is the Qwen team's under the Qwen Community License; the checkpoints are [turboderp's][ckpt]; the engine is [ExLlamaV3][exl3] (MIT) and the server [TabbyAPI][tabby] (AGPL-3.0), and every kernel patch in [`docker/`][docker-readme] is a derivative of one of them. The patches were written with OpenAI Codex from static source dumps and admitted or rejected by box measurement. The full table, including the upstream pull requests carried and the ideas borrowed from [DominikBucko][bucko], [HaberstrohSystems][haberstroh] and [vcruz305][vllm-exl3], is in [`THIRD_PARTY.md`][third-party].

[qwen-hf]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
[ckpt]: https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3
[exl3]: https://github.com/turboderp-org/exllamav3
[exl3-convert]: https://github.com/turboderp-org/exllamav3/blob/master/doc/convert.md
[tabby]: https://github.com/theroyallab/tabbyAPI
[llguidance]: https://github.com/guidance-ai/llguidance
[vllm-exl3]: https://github.com/vcruz305/vllm-exl3
[bucko]: https://github.com/DominikBucko/qwen38-flash-next-2x3090
[haberstroh]: https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang
[tool-eval]: https://github.com/SeraphimSerapis/tool-eval-bench
[mini-swe]: https://github.com/SWE-agent/mini-swe-agent
[lm-eval]: https://github.com/EleutherAI/lm-evaluation-harness
[pr246]: https://github.com/turboderp-org/exllamav3/pull/246
[pr284]: https://github.com/turboderp-org/exllamav3/pull/284
[pr290]: https://github.com/turboderp-org/exllamav3/pull/290
[pr299]: https://github.com/turboderp-org/exllamav3/pull/299
[pr303]: https://github.com/turboderp-org/exllamav3/pull/303
[pr337]: https://github.com/turboderp-org/exllamav3/pull/337

[launcher]: scripts/launch-flashnext.sh
[hygiene]: scripts/check-public-hygiene.sh
[prose]: scripts/check-prose.sh
[docker-readme]: docker/README.md
[bszn16]: docker/bszn16.patch
[coopwide]: docker/coopwide.patch
[hcmix]: docker/hc-mix-v2-r2.patch
[hostgap]: docker/hostgap-gated_delta_net.py
[ppipe]: docker/prefill-pipeline.patch
[nosync]: docker/overlays/prefill-nosync-overlay
[mtpfix]: docker/overlays/prefill-pipeline-mtp-overlay
[moecoop]: docker/overlays/moe-coop-v2-overlay
[bench]: bench
[bench-results]: bench/results
[probe]: bench/probe.py
[needle]: bench/needle.py
[capabilities]: bench/capabilities.py
[summarize]: bench/summarize.py
[config]: docs/CONFIG.md
[measurements]: docs/MEASUREMENTS.md
[promotion]: docs/PROMOTION.md
[gotchas]: docs/GOTCHAS.md
[runbook]: docs/RUNBOOK.md
[review]: docs/REVIEW_HANDOFF.md
[third-party]: THIRD_PARTY.md
[claude-md]: CLAUDE.md
[license]: LICENSE

[r339-gates]: bench/results/2026-09-16-r339-longgen.jsonl
[r340]: bench/results/2026-09-16-r340-ci-depth
[r341]: bench/results/2026-09-16-r341-qsa
[r339]: docs/MEASUREMENTS.md#decode-at-the-requeue-boundary--2048-forced-tokens-results-2026-09-16-r339-gates
[vllm-route]: bench/results/2026-09-18-vllm-exl3-route
[r343]: docs/MEASUREMENTS.md
[r347]: bench/results/2026-09-16-r347-soak
[r348]: bench/results/2026-09-16-r348-capabilities
[r354]: bench/results/2026-09-16-r354-combined
[r356]: bench/results/2026-09-16-r356-promoted
[r357]: bench/results/2026-09-16-r357-tooleval
[r358]: docs/MEASUREMENTS.md
[r359]: docs/MEASUREMENTS.md
[r362]: docs/MEASUREMENTS.md
[r365]: docs/MEASUREMENTS.md
[r366]: docs/MEASUREMENTS.md
[r368]: docs/MEASUREMENTS.md
[r377]: docs/MEASUREMENTS.md
[r421]: docs/MEASUREMENTS.md
[r428]: docs/MEASUREMENTS.md
[r442]: docs/MEASUREMENTS.md
[r446]: docs/MEASUREMENTS.md
[r452]: docs/MEASUREMENTS.md
[r453]: docs/MEASUREMENTS.md
[r460]: docs/MEASUREMENTS.md
[r461]: docs/MEASUREMENTS.md
[r462]: docs/MEASUREMENTS.md
