# Qwen3.8-Flash-Next on 2× RTX 5090 (ExLlamaV3 + TabbyAPI)

Serving configuration, launcher, image recipe, kernel overlays, instruments and measurements for [Qwen3.8-Flash-Next][qwen-hf], served as [r0b0tlab's 2.50 bpw EXL3 pack][ckpt-250] by [TabbyAPI][tabby] on [ExLlamaV3][exl3] v1.5.0 across two RTX 5090 cards. The window is 262,144 tokens, the KV cache is 8-bit, and vision, reasoning, tool calls, structured output and the checkpoint's own MTP draft head are all on.

Every number here was measured on one machine, on the date given, and each links the write-up that names its raw results directory. Nothing is a projection. The index of experiments is [`bench/RESULTS.md`][results], newest first.

## Numbers

Served configuration since 2026-09-19 06:18 CEST ([R530][r530]): image `tabbyapi:mtp-pruned-r1-tc1-plefix` with a fix for ExLlamaV3's PLE checkpoint aliasing ([R530][r530]), `tool_choice` enforcement in TabbyAPI ([R529][r529]), int8 hyper-connection mixer weights (`EXL3_HC_MIX_V2_INT8=1`, [R525][r525]) and the MTP draft chain on the GPU with a 320 MiB copy of the 65,536 embedding rows the draft head can emit ([R528][r528]), 4 slots, a 819,200-token page pool at 8-bit KV, layer split `[30, 30]`, MTP draft depth 3, launcher [`scripts/launch-flashnext.sh`][launcher]. Decode rates are `fn_bench` ([`bench/probe.py`][probe]): 2,048 forced tokens per request, greedy. "Aggregate" is all streams' tokens over the round's wall time; "per stream" is one request's tokens over its own wall time (first token included), averaged over the requests. Each rate names its kind, because on this checkpoint code decodes faster than prose at c1 (draft acceptance tracks how predictable the text is).

| | value | measured |
| --- | --- | --- |
| context window | 262,144 tokens | the checkpoint's native length |
| page pool | 819,200 tokens at 8-bit KV, shared by 4 slots: **1.41 GB of VRAM per 100k tokens**, 11.6 GB for the whole pool ([derivation][config-memory]) | 2026-09-19, [R516][r516], [R525][r525]; the int8 mixer weights free the VRAM for it, and 835,584 does not boot |
| decode, 1 stream | • code 221.8 t/s<br>• prose 190.7 t/s | code: 2026-09-19, [R522][r522] (R522b), 4 boots × 3 runs after a warm-up round on the served configuration (boot means 221.3–222.1); prose: 2026-09-18, [R499][r499], two boots, before R525 and R528 |
| decode, 4 streams | • code 508.4 t/s aggregate, 139.9 per stream (4 boots × 2 runs after a warm-up round on the served configuration, [R522][r522])<br>• prose 473.6 t/s aggregate, 122.9 per stream<br>• 12 distinct sampled prompts per kind: code 440.2 aggregate / 116.7 per stream, prose 367.6 / 111.6 | 2026-09-18, [R499][r499]; the sampled rows use [`bench/multiprompt.py`][multiprompt] |
| decode, 6 streams (4 slots: 2 requests queue) | • code 448.8 t/s aggregate, 112.9 per stream<br>• prose 423.1 t/s aggregate, 104.8 per stream<br>• 6 slots would give 526.9 / 498.2 aggregate but run an 8-agent replay 9 % slower, so they are not served | 2026-09-19, [R518][r518] |
| decode, agent-shaped edit | 223.7 t/s at 1 stream; at 4 streams 502.2 aggregate, 143.1 per stream (first wave of 4 requests; the tool's whole-run figure, 421.3, includes a second wave of only 2); six real files rewritten with a small edit, greedy | 2026-09-19, [R525][r525], [`bench/agentic-edit.py`][agentic-edit] |
| decode at depth | prose 172.7 / 152.8 / 169.2 t/s at 0 / 99,919 / 199,457 prompt tokens | 2026-09-18 on the 3.05 bpw pack, [R492][r492] |
| cold prefill, 1 request | • requested 30k: 8,740 t/s<br>• requested 60k (45,073–45,163 prompt tokens): 9,354–9,841 t/s<br>• requested 120k (90,040–90,135 prompt tokens): 10,015–10,278 t/s | 2026-09-18 and 2026-09-19, [R513][r513], [R517][r517], [R528][r528], [R530][r530]; sizes are `fn_bench --ctx` targets, and the prompts carry about three quarters of that in tokens; one invocation per length |
| long-context retrieval | 5/5 planted needles at 131k and at 240k prompt tokens | 2026-09-19, [R517][r517], [R525][r525], [R528][r528], [R530][r530] |
| GSM8K 5-shot, n=500, thinking on, no stop strings | 0.974, 0.978, 0.970 and 0.974 on the last four configurations (byte-identical decode); 0.978 and 0.980 before them | 2026-09-19, [R530][r530], [R529][r529], [R528][r528], [R525][r525]; 2026-09-18, [R509][r509]; 2026-09-19, [R516][r516] |
| [tool-eval-bench][tool-eval], 69 scenarios × 4 | 84.8 ± 1.3 | 2026-09-19, [R530][r530]; 88.0 ± 1.6 on the configuration before it ([R529][r529]), where `tool_choice` enforcement turned TC-45 from 0 to 2 points on every trial (worth 2.9 points of the score). The ± is the spread of 4 trials on one boot; the last six configurations read 84.5 to 88.0, and the R529–R530 gap sits in scenarios that already change between trials of one boot ([R530][r530]) |
| [SWE-bench Verified][swebench], [mini-SWE-agent][mini-swe] 2.4.6, official scorer | 46 of 49 instances resolved | 2026-09-16 on the 3.05 bpw pack, [R359][r359]; the instances were selected on earlier outcomes, so this is a tally, not a full-set score. Cost per instance: [agent runs][agent-cost] |
| structured output ([llguidance][llguidance]) | `json_schema`, `response_format`, `regex_pattern` pass, thinking on and off, c4 | 2026-09-17, [R453][r453] |
| `tool_choice` | `required` 48/48, named 4/4, 8/8 concurrent; a forced turn that answers in content first continues into the call; `auto` / `none` unchanged | 2026-09-19, [R529][r529] |
| boot to serving | about 20 s with warm kernel caches | 2026-09-19, [R525][r525], [R528][r528] promotion boots |
| free VRAM after boot | 1,973 MiB on cuda:0, 737 MiB on cuda:1; 1,173 / 225 MiB under whole-pool load | 2026-09-19, [R528][r528] |

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

- Recurrent-state checkpoints at the end of each answer, so an agent's next call resumes after its previous answer instead of re-reading it, with an eviction order that keeps each conversation's newest checkpoint: [`scripts/r524-recurrent-tip.sh`][r524-driver], measured with the agent replay in echo mode ([`bench/agent_replay.py`][agent-replay] `--echo`).
- A persistent prefix tier on NVMe: KV pages and recurrent checkpoints written to disk in the background, restored after a restart, under a byte cap: [`scripts/r526-nvme-tier.sh`][r526-driver].

## Measured and not served

A fused shared-expert kernel: after the side-stream overlap the shared expert's residual is 1.9 µs per layer at 4 rows, at most 0.6 % of a 1-stream step and 1.2 % at 4 streams ([R521][r521]); `EXL3_INT8_GEMV=0`, −0.3 % at c1 with a 95 % interval of ±1.2 % over 8 boots ([R520b][r520b]); 6 decode slots, +17 % at c6 and 9 % slower on an 8-agent replay ([R518][r518]); chunk 1024 for pool ([R483][r483], [R485][r485]); split [30, 31] at 393,216 ([R487][r487]); the n-gram table in host RAM, +1–2 % for 30.5 GiB ([R484][r484]); the host KV tier ([R358][r358], [R493][r493]); GDN state replay ([R496][r496]); a 4-bit MTP graft ([R498][r498]); prompt lookup, +3–4 % on code at c1 and flat at c4 ([R501][r501]); K8V4, +18 % pool for −11 % code at c1 ([R480][r480]); CPU-offloaded experts ([R482][r482]); MoE coop mode 3 ([R462][r462]); [exllamav3#303][pr303] MTP hot vocabulary ([R377][r377]); [exllamav3#246][pr246] and [#290][pr290] ([R365][r365]); our own 32-row MoE decode envelope ([R366][r366]). The same checkpoint on vLLM through [vllm-exl3][vllm-exl3] read 0.62× the c1 and 1.06–1.12× the c4 of this stack's 3.05 bpw configuration of 2026-09-18; work on that route stopped the same day ([vLLM route][vllm-route]).

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
[r521-driver]: scripts/r521-shared-bound.sh
[r522-driver]: scripts/r522-mtp-pruned.sh
[r523-driver]: scripts/r523-tool-choice.sh
[r524-driver]: scripts/r524-recurrent-tip.sh
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
[config-memory]: docs/CONFIG.md#memory
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
[r525]: bench/results/r525-promote-int8mix.md
[r528]: bench/results/r528-promote-mtp-pruned.md
[r529]: bench/results/r529-promote-tool-choice.md
[r530]: bench/results/r530-promote-plefix.md
[r521]: bench/results/r521-shared-bound.md
[r522]: bench/results/r522-mtp-pruned.md
[r528-driver]: scripts/r528-promote-mtp-pruned.sh
[r522b-driver]: scripts/r522b-mtp-pruned-precise.sh
[r526-driver]: scripts/r526-nvme-tier.sh
[r518]: bench/results/r518-slots6.md
[r519]: bench/results/r519-profile-2p50.md
[r520]: bench/results/r520-int8gemv.md
[r520b]: bench/results/r520b-int8gemv-precise.md
