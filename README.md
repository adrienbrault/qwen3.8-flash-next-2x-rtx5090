# Qwen3.8-Flash-Next on 2× RTX 5090 (ExLlamaV3 + TabbyAPI)

Serving configuration, launcher, image recipe, kernel overlays, instruments and measurements for [Qwen3.8-Flash-Next][qwen-hf], served as [r0b0tlab's 2.50 bpw EXL3 pack][ckpt-250] by [TabbyAPI][tabby] on [ExLlamaV3][exl3] v1.5.0 across two RTX 5090 cards. The window is 262,144 tokens, the KV cache is 8-bit, and vision, reasoning, tool calls, structured output and the checkpoint's own MTP draft head are all on.

Every number here was measured on one machine, on the date given, and each links the write-up that names its raw results directory. Nothing is a projection. The index of experiments is [`bench/RESULTS.md`][results], newest first.

## Numbers

Served configuration since 2026-09-19 01:26 CEST ([R517][r517]): image `tabbyapi:stack-r4-e3r2`, 4 slots, a 786,432-token page pool at 8-bit KV, layer split `[30, 30]`, MTP draft depth 3, launcher [`scripts/launch-flashnext.sh`][launcher]. Decode rates are `fn_bench` ([`bench/probe.py`][probe]): 2,048 forced tokens per request, greedy. "Aggregate" is all streams' tokens over the round's wall time; "per stream" is one request's tokens over its own wall time (first token included), averaged over the requests. Each rate names its kind, because on this checkpoint code decodes faster than prose at c1 (draft acceptance tracks how predictable the text is).

| | value | measured |
| --- | --- | --- |
| context window | 262,144 tokens | the checkpoint's native length |
| page pool | 786,432 tokens at 8-bit KV, shared by 4 slots | 2026-09-18, [R495b][r495b], [R511][r511]; 819,200 and above do not boot |
| decode, 1 stream | • code 217.7 t/s<br>• prose 190.7 t/s | 2026-09-18, [R499][r499], mean of two boots; the prefill change served after it leaves decode unchanged, [R513][r513] |
| decode, 4 streams | • code 510.5 t/s aggregate, 132.3 per stream<br>• prose 473.6 t/s aggregate, 122.9 per stream<br>• 12 distinct sampled prompts per kind: code 440.2 aggregate / 116.7 per stream, prose 367.6 / 111.6 | 2026-09-18, [R499][r499]; the sampled rows use [`bench/multiprompt.py`][multiprompt] |
| decode, agent-shaped edit | 211.9 t/s at 1 stream; at 4 streams 437.1 aggregate, 153.6 per stream (six real files rewritten with a small edit, greedy) | 2026-09-19, [R517][r517], [`bench/agentic-edit.py`][agentic-edit] |
| decode at depth | prose 172.7 / 152.8 / 169.2 t/s at 0 / 99,919 / 199,457 prompt tokens | 2026-09-18 on the 3.05 bpw pack, [R492][r492] |
| cold prefill, 1 request | • 30k tokens: 8,740 t/s<br>• 60k: 9,543–9,814 t/s<br>• 120k: 10,015–10,163 t/s | 2026-09-18 and 2026-09-19, [R513][r513], [R517][r517]; salted prompts, one invocation per length |
| long-context retrieval | 5/5 planted needles at 131k and at 240k prompt tokens | 2026-09-19, [R517][r517] |
| GSM8K 5-shot, n=500, thinking on, no stop strings | 0.978 and 0.980 on two runs | 2026-09-18, [R509][r509]; 2026-09-19, [R516][r516] |
| [tool-eval-bench][tool-eval], 69 scenarios × 4 | 85.8 ± 2.1 | 2026-09-19, [R517][r517]; 84.5 ± 1.9 and 86.0 ± 2.6 on the two configurations before it |
| [SWE-bench Verified][swebench], [mini-SWE-agent][mini-swe] 2.4.6, official scorer | 46 of 49 instances resolved | 2026-09-16 on the 3.05 bpw pack, [R359][r359]; the instances were selected on earlier outcomes, so this is a tally, not a full-set score. Cost per instance: [agent runs][agent-cost] |
| structured output ([llguidance][llguidance]) | `json_schema`, `response_format`, `regex_pattern` pass, thinking on and off, c4 | 2026-09-17, [R453][r453] |
| boot to serving | 21 s with warm kernel caches | 2026-09-19, [R517][r517] promotion boot |
| free VRAM after boot | 2,015 MiB on cuda:0, 1,099 MiB on cuda:1 | 2026-09-19, [R517][r517] |

GSM8K figures published by this project before 2026-09-18 evening (0.9158 at n=1319, 0.925, 0.935) were measured with lm-eval's stop strings, which cut the model's reasoning when it restates the problem as "Question: …"; they undercount by 7 to 18 % of questions ([R509][r509]).

## What the stack is

- **Checkpoint**: [r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw][ckpt-250], routed experts at K = 2, 3 and 4 bits. It boots a 2.18× larger pool than [turboderp's 3.05 bpw pack][ckpt-turbo] and decodes 3–8 % faster except code at c1 ([R495b][r495b]); both score the same on GSM8K ([R509][r509]).
- **Engine**: [ExLlamaV3][exl3] v1.5.0 under [TabbyAPI][tabby] `53da7919`, plus the chain in [`docker/`][docker-readme]. Every patch is opt-in by environment flag and was admitted with byte-identical greedy output, or with GSM8K, needles and tool-eval where it changes numerics:
  - [exllamav3#337][pr337]: keeps the CUDA device on the module's device during a layer-split forward ([R362][r362]).
  - multi-job QSA sparse attention and concurrency-indexed draft depth ([R341][r341], [R340][r340]).
  - the fused MoE decode path at 16 rows ([R414][r414]) and a wide stage-B tile ([R421][r421]).
  - bit-exact V2 of the hyper-connection mixer and of the fused MoE decode kernel ([R428][r428], [R460][r460]).
  - a two-card prefill pipeline without blocking host syncs ([R442][r442]).
  - the shared expert on a side CUDA stream, +3 to +7 % decode ([R490][r490]).
  - pinned draft staging, one readback per verify step and a 65,536-token draft head, +2 to +3 % decode ([R499][r499]).
  - grouped MoE prefill for every K, +16 to +21 % cold prefill ([R513][r513]).
- **Cards**: layer split, 30 GB of weights and cache per card. `qwen4_exp` raises `NotImplementedError` for tensor parallelism in this engine, so the cards take turns over their own layers and one stream keeps each card 44–47 % busy (2026-09-16, 3.05 bpw pack, [GPU duty cycle][duty]). Expert parallelism was built and measured at −9.5 % at c1 (results `2026-09-16-r408-ep-served`).
- **Speculative decoding**: the checkpoint's MTP head, depth 3 up to 4 concurrent jobs and depth 1 above (`[[4, 3], [8, 1]]`). Confidence-gated dynamic depth crashes at c4 ([R497][r497]).
- **Sampler fallbacks**: temperature 0.6, top_k 20, top_p 0.95 with `force: false`, so a client that sends its own sampler keeps it. Without a preset TabbyAPI serves sampler-less requests at temperature 1.0 untruncated ([`docs/GOTCHAS.md`][gotchas]).
- **Guard rails**: the launcher refuses to start without the checkpoint or the image, stops any other engine holding the cards, waits for them to drain and mounts the kernel caches. Every promotion re-runs the gates in [`docs/PROMOTION.md`][promotion] on the exact launcher.

## In progress (queued on the box 2026-09-19)

- 6 slots instead of 4, with an agent-replay probe that replays recorded SWE-bench conversations at 8 concurrent agents and logs cached against new prompt tokens per call: [`scripts/r518-slots6.sh`][r518-driver], [`bench/agent_replay.py`][agent-replay]. At 6 slots, 786,432 and 753,664 do not boot.
- The first decode profile of the 2.50 bpw pack: [`scripts/r519-profile-2p50.sh`][r519-driver].
- `EXL3_INT8_GEMV=0`, the fp16 kernel instead of the int8-activation one for single-row linears: [`scripts/r520-int8gemv.sh`][r520-driver].
- int8 mixer weights as a pool lever: 819,200 tokens (+4 %) at equal GSM8K, needles and tool-eval, decode 0 to −2 % ([R516][r516]).

## Measured and not served

Chunk 1024 for pool ([R483][r483], [R485][r485]); split [30, 31] at 393,216 ([R487][r487]); the n-gram table in host RAM, +1–2 % for 30.5 GiB ([R484][r484]); the host KV tier ([R358][r358], [R493][r493]); GDN state replay ([R496][r496]); a 4-bit MTP graft ([R498][r498]); prompt lookup, +3–4 % on code at c1 and flat at c4 ([R501][r501]); K8V4, +18 % pool for −11 % code at c1 ([R480][r480]); CPU-offloaded experts ([R482][r482]); MoE coop mode 3 ([R462][r462]); [exllamav3#303][pr303] MTP hot vocabulary ([R377][r377]); [exllamav3#246][pr246] and [#290][pr290] ([R365][r365]); our own 32-row MoE decode envelope ([R366][r366]). The same checkpoint on vLLM through [vllm-exl3][vllm-exl3] read 0.62× the c1 and 1.06–1.12× the c4 of this stack's 3.05 bpw configuration of 2026-09-18; work on that route stopped the same day ([vLLM route][vllm-route]).

## Reproducing a boot

```sh
ssh flan 'bash -s' < scripts/launch-flashnext.sh                  # serve on :8022
PORT=8023 bash scripts/launch-flashnext.sh                        # a second instance
IMG=tabbyapi:decode-kernels-r4 EXTRA_ENV= bash scripts/launch-flashnext.sh   # an older image, patches off
STOP=1 bash scripts/launch-flashnext.sh                           # stop
```

Rebuilding the image chain, running the measurements and handing the box back are in [`docs/RUNBOOK.md`][runbook].

## Repository map

| path | what it is |
| --- | --- |
| [`scripts/launch-flashnext.sh`][launcher] | the served launcher: writes the config and the sampler preset, starts the container, warms the kernels |
| [`scripts/launchers/`][launchers] | the launcher of each promotion and experiment, for rollback and reproduction |
| [`scripts/r*.sh`][scripts] | one driver per experiment, each under the box's GPU lock |
| [`docker/`][docker-readme] | the served image layer by layer: Dockerfiles, patches, overlays with SHA-pinned installers and kernel tests |
| [`bench/RESULTS.md`][results] | index of every experiment, newest first; one file per experiment in [`bench/results/`][bench-results] with its raw records |
| [`bench/probe.py`][probe] | decode, concurrency and depth instrument (`fn_bench`): forced length via `min_tokens`, one JSONL line per request |
| [`bench/multiprompt.py`][multiprompt], [`bench/agentic-edit.py`][agentic-edit] | sampled multi-prompt decode, and agent-shaped file edits |
| [`bench/needle.py`][needle], [`bench/capabilities.py`][capabilities] | long-context retrieval at five planted positions; JSON schema, tool parsing, vision and reasoning checks |
| [`bench/nostop_proxy.py`][nostop] | drops the request's `stop` field so lm-eval's stop strings cannot cut reasoning |
| [`bench/agent_replay.py`][agent-replay], [`bench/revisit.py`][revisit] | recorded agent conversations replayed at concurrency; revisits of evicted long sessions |
| [`docs/CONFIG.md`][config] | every setting and flag, and why it has that value |
| [`docs/HISTORY.md`][history] | how the served configuration changed, with the result behind each step |
| [`docs/PROMOTION.md`][promotion] | the gates a candidate passes before it is served |
| [`docs/GOTCHAS.md`][gotchas] | the traps, as "what it looks like" against "what it is" |
| [`docs/RUNBOOK.md`][runbook] | serve, measure, rebuild and hand the box back |
| [`THIRD_PARTY.md`][third-party] | where every input came from and under which licence |
| [`CLAUDE.md`][claude-md] | the working agreement: prose rules, box rules, measurement rules |

## Attribution and licence

The original work here (documentation, instruments, launcher, overlay installers, measurements) is [MIT][license]. The model is the Qwen team's under the Qwen Community License; the checkpoints are [r0b0tlab's][ckpt-250] and [turboderp's][ckpt-turbo]; the engine is [ExLlamaV3][exl3] (MIT) and the server [TabbyAPI][tabby] (AGPL-3.0), and every kernel patch in [`docker/`][docker-readme] is a derivative of one of them. The patches were written with coding agents from static source dumps and admitted or rejected by measurement on the box. Ideas were taken from [vcruz305's DGX Spark recipe][vcruz-recipe], [DominikBucko][bucko], [HaberstrohSystems][haberstroh] and [halogen][halogen]; the full table is in [`THIRD_PARTY.md`][third-party].

## Links

Model and checkpoints: [Qwen3.8-Flash-Next][qwen-hf] · [r0b0tlab 2.50 bpw][ckpt-250] · [turboderp EXL3 packs][ckpt-turbo]

Engine and server: [ExLlamaV3][exl3] · [EXL3 conversion][exl3-convert] · [TabbyAPI][tabby] · [llguidance][llguidance]

Upstream pull requests measured here: [exllamav3#246][pr246] · [#284][pr284] · [#290][pr290] · [#299][pr299] · [#303][pr303] · [#337][pr337]

Other setups of this model: [vcruz305 DGX Spark recipe][vcruz-recipe] · [vcruz305/vllm-exl3][vllm-exl3] · [DominikBucko, 2× RTX 3090][bucko] · [HaberstrohSystems, 24 GB SGLang][haberstroh]

Benchmarks and harnesses: [tool-eval-bench][tool-eval] · [mini-SWE-agent][mini-swe] · [SWE-bench][swebench] · [lm-evaluation-harness][lm-eval] · [halogen][halogen]

[qwen-hf]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
[ckpt-250]: https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw
[ckpt-turbo]: https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3
[exl3]: https://github.com/turboderp-org/exllamav3
[exl3-convert]: https://github.com/turboderp-org/exllamav3/blob/master/doc/convert.md
[tabby]: https://github.com/theroyallab/tabbyAPI
[llguidance]: https://github.com/guidance-ai/llguidance
[vllm-exl3]: https://github.com/vcruz305/vllm-exl3
[vcruz-recipe]: https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe
[bucko]: https://github.com/DominikBucko/qwen38-flash-next-2x3090
[haberstroh]: https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang
[halogen]: https://github.com/peonist-ai/halogen
[tool-eval]: https://github.com/SeraphimSerapis/tool-eval-bench
[mini-swe]: https://github.com/SWE-agent/mini-swe-agent
[swebench]: https://github.com/SWE-bench/SWE-bench
[lm-eval]: https://github.com/EleutherAI/lm-evaluation-harness
[pr246]: https://github.com/turboderp-org/exllamav3/pull/246
[pr284]: https://github.com/turboderp-org/exllamav3/pull/284
[pr290]: https://github.com/turboderp-org/exllamav3/pull/290
[pr299]: https://github.com/turboderp-org/exllamav3/pull/299
[pr303]: https://github.com/turboderp-org/exllamav3/pull/303
[pr337]: https://github.com/turboderp-org/exllamav3/pull/337

[launcher]: scripts/launch-flashnext.sh
[launchers]: scripts/launchers/
[scripts]: scripts/
[r518-driver]: scripts/r518-slots6.sh
[r519-driver]: scripts/r519-profile-2p50.sh
[r520-driver]: scripts/r520-int8gemv.sh
[docker-readme]: docker/README.md
[results]: bench/RESULTS.md
[bench-results]: bench/results/
[probe]: bench/probe.py
[multiprompt]: bench/multiprompt.py
[agentic-edit]: bench/agentic-edit.py
[needle]: bench/needle.py
[capabilities]: bench/capabilities.py
[nostop]: bench/nostop_proxy.py
[agent-replay]: bench/agent_replay.py
[revisit]: bench/revisit.py
[config]: docs/CONFIG.md
[history]: docs/HISTORY.md
[promotion]: docs/PROMOTION.md
[gotchas]: docs/GOTCHAS.md
[runbook]: docs/RUNBOOK.md
[third-party]: THIRD_PARTY.md
[claude-md]: CLAUDE.md
[license]: LICENSE

[agent-cost]: bench/results/swebench-agent-cost.md
[duty]: bench/results/gpu-duty-cycle.md
[vllm-route]: bench/results/vllm-exl3-route.md
[r340]: bench/results/r340-ci-depth.md
[r341]: bench/results/r341-qsa.md
[r358]: bench/results/r358-hostkv.md
[r359]: bench/results/r359-swebench.md
[r362]: bench/results/r362-pr337.md
[r365]: bench/results/r365-kernels.md
[r366]: bench/results/r366-ourkernel.md
[r377]: bench/results/r377-hotvocab-on.md
[r414]: bench/results/r414-bszn16.md
[r421]: bench/results/r421-coopwide-ab.md
[r428]: bench/results/r428-hcmix2-stack-ab.md
[r442]: bench/results/r442-ppipe.md
[r453]: bench/results/r453-exl3-structured.md
[r460]: bench/results/r460-moecoop-v2-ab.md
[r462]: bench/results/r462-moecoop-v3-ab.md
[r480]: bench/results/r480-exl3-pool.md
[r482]: bench/results/r482-cold-experts.md
[r483]: bench/results/r483-exl3-pool-chunk.md
[r484]: bench/results/r484-ngram-ram.md
[r485]: bench/results/r485-pool-frontier.md
[r487]: bench/results/r487-pool-393k.md
[r490]: bench/results/r490-shared-overlap.md
[r492]: bench/results/r492-depth.md
[r493]: bench/results/r493-host-kv-tier.md
[r495b]: bench/results/r495b-2p50-audition.md
[r496]: bench/results/r496-gdn-state-r3.md
[r497]: bench/results/r497-draft-confidence.md
[r498]: bench/results/r498-mtp4-graft.md
[r499]: bench/results/r499-decode-r4.md
[r501]: bench/results/r501-prompt-lookup.md
[r509]: bench/results/r509-gsm8k-nostop.md
[r511]: bench/results/r511-promote-2p50.md
[r513]: bench/results/r513-prefill-e3-r2.md
[r516]: bench/results/r516-int8-mixer-pool.md
[r517]: bench/results/r517-promote-stack.md
