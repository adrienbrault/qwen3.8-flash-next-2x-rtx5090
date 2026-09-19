# Qwen3.8-Flash-Next on 2× RTX 5090 (ExLlamaV3 + TabbyAPI)

Serving configuration, launcher, image recipe, kernel overlays, instruments and measurements for [Qwen3.8-Flash-Next][qwen-hf], served as [r0b0tlab's 2.50 bpw EXL3 pack][ckpt-250] by [TabbyAPI][tabby] on [ExLlamaV3][exl3] v1.5.0 across two RTX 5090 cards. The window is 262,144 tokens, the KV cache is 8-bit, and vision, reasoning, tool calls, structured output and the checkpoint's own MTP draft head are all on.

Every number here was measured on one machine, on the date given, and each links the write-up that names its raw results directory. Nothing is a projection. The index of experiments is [`bench/RESULTS.md`][results], newest first.

## Numbers

Served configuration since 2026-09-19 17:18 CEST ([R548][r548]): image `tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16`, which stores the GDN recurrent state in bf16 (fp32 math, 864 MiB freed for the pool, [R548][r548]) and keeps a 20-row ring of the QSA indexer's raw keys per page instead of every token's key, bit-exact and +20 % page pool ([R546][r546]), one-warp launches of the GDN B/A GEMV and `hc_apply` and a re-gridded mixer state kernel, byte-identical and +1.0 % code / +1.1 % prose at 1 stream ([R538][r538], [R540][r540]), deterministic E3 prefill, so a cold prefill of a long prompt gives the same output every run ([R535][r535]), a persistent prefix tier on NVMe that restores prompts after a restart ([R534][r534]), a fix for ExLlamaV3's PLE checkpoint aliasing ([R530][r530]), `tool_choice` enforcement in TabbyAPI ([R529][r529]), int8 hyper-connection mixer weights (`EXL3_HC_MIX_V2_INT8=1`, [R525][r525]) and the MTP draft chain on the GPU with a 320 MiB copy of the 65,536 embedding rows the draft head can emit ([R528][r528]), 4 slots, a 1,032,192-token page pool at 8-bit KV, layer split `[30, 30]`, MTP draft depth 3, launcher [`scripts/launch-flashnext.sh`][launcher]. Decode rates are `fn_bench` ([`bench/probe.py`][probe]): 2,048 forced tokens per request, greedy. "Aggregate" is all streams' tokens over the round's wall time; "per stream" is one request's tokens over its own wall time (first token included), averaged over the requests. Each rate names its kind, because on this checkpoint code decodes faster than prose at c1 (draft acceptance tracks how predictable the text is).

| | value | measured |
| --- | --- | --- |
| context window | 262,144 tokens | the checkpoint's native length |
| page pool | 1,032,192 tokens at 8-bit KV, shared by 4 slots: **1.52 GB of VRAM per 100k tokens**, 15.7 GB for the whole pool ([derivation][config-memory]) | 2026-09-19: [R546][r546], the raw-key ring, +163,840 tokens with unchanged output, decode and prefill; [R548][r548], bf16 GDN state, +49,152 more, decode +0.3 to +2.1 % on 48 paired prompts; 1,048,576 would leave cuda:1 below the served boot's free VRAM. 819,200 at the start of the day ([R525][r525]) |
| decode, 1 stream | • code 225.4 t/s<br>• prose 200.2 t/s | 2026-09-19, [R538][r538], the served configuration with the NVMe tier off, 4 boots after a warm-up round: code 3 runs per boot (boot means 225.0–225.8), prose 2 runs per boot (199.8–200.4) |
| decode, 4 streams | • code 510.0 t/s aggregate, 140.3 per stream (2026-09-19, [R538][r538]: 4 boots × 2 runs after a warm-up round on the served configuration with the NVMe tier off; in every run 2 of the 4 requests stop at 1,861 tokens and the aggregate includes them)<br>• prose 473.6 t/s aggregate, 122.9 per stream<br>• 12 distinct sampled prompts per kind: code 440.2 aggregate / 116.7 per stream, prose 367.6 / 111.6 | code: 2026-09-19, [R538][r538]; prose and sampled rows: 2026-09-18, [R499][r499], before R525 and R528; the sampled rows use [`bench/multiprompt.py`][multiprompt] |
| over capacity: 6 requests on 4 slots | • code 448.8 t/s aggregate, 112.9 per stream<br>• prose 423.1 t/s aggregate, 104.8 per stream<br>• below the 4-stream row because 2 requests wait for a slot and then decode as a second wave of 2; this row measures queueing, not a 6-stream engine<br>• 6 slots give 526.9 / 498.2 aggregate on this synthetic load, but each stream decodes at ~86 t/s instead of ~135 and an 8-agent replay runs 9 % slower, so 4 slots are served | 2026-09-19, [R518][r518] |
| TTFT and decode with slots in use | • short prompts sent together: TTFT 0.13 / 0.22 / 0.31–0.34 / 0.39–0.45 s at 1 / 2 / 3 / 4 requests<br>• a new short request while 3 slots decode ~112k-token contexts: TTFT 0.26 s (0.16 s with none)<br>• 4 prompts of ~142.6k tokens sent together (570k tokens resident): 104–117 t/s per stream once all four decode, longest gap between streamed chunks 0.051 s, no failures<br>• long prompts sent together share the prefill: 4 × ~22.5k tokens reach their first token at 12.0–12.1 s, 4 × ~142.6k at 24 s and 70 s; running streams get about 2 decode steps per second while a new long prompt prefills | 2026-09-19, [R549][r549], the served configuration with the NVMe tier off; [`bench/hot_slots.py`][hot-slots] |
| decode, agent-shaped edit | 223.7 t/s at 1 stream; at 4 streams 502.2 aggregate, 143.1 per stream (first wave of 4 requests; the tool's whole-run figure, 421.3, includes a second wave of only 2); six real files rewritten with a small edit, greedy | 2026-09-19, [R525][r525], [`bench/agentic-edit.py`][agentic-edit] |
| decode at depth | prose 172.7 / 152.8 / 169.2 t/s at 0 / 99,919 / 199,457 prompt tokens | 2026-09-18 on the 3.05 bpw pack, [R492][r492] |
| cold prefill, 1 request | • requested 30k: 8,740 t/s<br>• requested 60k (45,073–45,163 prompt tokens): 9,354–9,841 t/s<br>• requested 120k (90,040–90,135 prompt tokens): 10,015–10,278 t/s | 2026-09-18 and 2026-09-19, [R513][r513], [R517][r517], [R528][r528], [R530][r530]; sizes are `fn_bench --ctx` targets, and the prompts carry about three quarters of that in tokens; one invocation per length |
| long-context retrieval | 5/5 planted needles at 131k and at 240k prompt tokens | 2026-09-19, [R548][r548], [R546][r546], [R517][r517], [R525][r525], [R528][r528], [R530][r530], [R534][r534], [R535][r535], [R540][r540] |
| cold prefill determinism | a long prompt prefilled twice gives identical output (E3 with `atomicAdd` diverged by generated token 47 to 172); costs 1.9 % prefill at the 60k target (95 % interval −4.6 to +0.9 %) and 0.2 % at 120k, decode unchanged | 2026-09-19, [R533][r533] (8 boots), [R535][r535] |
| prompt restored from the NVMe tier after a restart | • 29,952-token prompt: 0.69 s (cold prefill 3.96 s)<br>• 119,808-token prompt: 0.99 s (cold 12.33 s)<br>• output identical to the warm and cold runs; 64 GiB cap, about 2.7 million tokens | 2026-09-19, [R534][r534], [R532][r532]; with the tier idle, decode differs from the untiered daily by −0.21 % at 1 stream and +0.11 % at 4 (8 boots); while a 120k prefix drains to disk, −3.8 % / −3.0 % for about 26 s |
| GSM8K 5-shot, n=500, thinking on, no stop strings | 0.974 with bf16 GDN state; 0.976, 0.974, 0.972, 0.976, 0.974, 0.978, 0.970 and 0.974 on the eight configurations before it (byte-identical decode); 0.978 and 0.980 before them | 2026-09-19, [R548][r548], [R546][r546], [R540][r540], [R535][r535], [R534][r534], [R530][r530], [R529][r529], [R528][r528], [R525][r525]; 2026-09-18, [R509][r509]; 2026-09-19, [R516][r516] |
| [tool-eval-bench][tool-eval], 69 scenarios × 4 | 86.8 ± 2.6 | 2026-09-19, [R548][r548]; 86.5 ± 2.4 on [R546][r546], 87.2 ± 1.5 on [R540][r540], 85.0 ± 0.8 on [R535][r535], 87.0 ± 1.2 on [R534][r534], 84.8 ± 1.3 on [R530][r530] and 88.0 ± 1.6 on [R529][r529], where `tool_choice` enforcement turned TC-45 from 0 to 2 points on every trial (worth 2.9 points of the score). The ± is the spread of 4 trials on one boot; the last eleven configurations read 84.5 to 88.0, and the R529–R530 gap sits in scenarios that already change between trials of one boot ([R530][r530]) |
| [SWE-bench Verified][swebench], [mini-SWE-agent][mini-swe] 2.4.6, official scorer | 46 of 49 instances resolved | 2026-09-16 on the 3.05 bpw pack, [R359][r359]; the instances were selected on earlier outcomes, so this is a tally, not a full-set score. Cost per instance: [agent runs][agent-cost] |
| structured output ([llguidance][llguidance]) | `json_schema`, `response_format`, `regex_pattern` pass, thinking on and off, c4 | 2026-09-17, [R453][r453] |
| `tool_choice` | `required` 48/48, named 4/4, 8/8 concurrent; a forced turn that answers in content first continues into the call; `auto` / `none` unchanged | 2026-09-19, [R529][r529] |
| boot to serving | about 20 s with warm kernel caches | 2026-09-19, [R525][r525], [R528][r528] promotion boots |
| free VRAM after boot | 2,013 MiB on cuda:0, 737 MiB on cuda:1 at 1,032,192 tokens; 1,973 / 737 at 983,040 with the ring and at 819,200 without it | 2026-09-19, [R548][r548], [R546][r546]; 1,173 / 225 MiB under whole-pool load at 819,200 ([R528][r528]) |

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
  - one-warp launches of two small decode kernels at 1 stream and a re-gridded mixer state kernel, +1.0 to +1.1 % decode at 1 stream ([R538][r538]).
  - a 20-row ring for the QSA indexer's raw keys, bit-exact, +20 % page pool ([R546][r546]).
  - GDN recurrent state stored in bf16 with fp32 math, +5 % page pool, GSM8K 0.974 and tool-eval 86.8 ([R548][r548]).
- **Cards**: layer split, 30 GB of weights and cache per card. `qwen4_exp` raises `NotImplementedError` for tensor parallelism in this engine, so the cards take turns over their own layers and one stream keeps each card 44–47 % busy (2026-09-16, 3.05 bpw pack, [GPU duty cycle][duty]). Expert parallelism was built and measured at −9.5 % at c1 (results `2026-09-16-r408-ep-served`). Tensor parallelism was bounded before building it: from measured half-work kernel times and all-reduce costs, a TP step would be at most 1.07–1.08× faster at 1 and 4 streams (2026-09-19, [R527][r527]).
- **Speculative decoding**: the checkpoint's MTP head, depth 3 up to 4 concurrent jobs and depth 1 above (`[[4, 3], [8, 1]]`). Confidence-gated dynamic depth crashes at c4 ([R497][r497]).
- **Sampler fallbacks**: temperature 0.6, top_k 20, top_p 0.95 with `force: false`, so a client that sends its own sampler keeps it. Without a preset TabbyAPI serves sampler-less requests at temperature 1.0 untruncated ([`docs/GOTCHAS.md`][gotchas]).
- **Guard rails**: the launcher refuses to start without the checkpoint or the image, stops any other engine holding the cards, waits for them to drain and mounts the kernel caches. Every promotion re-runs the gates in [`docs/PROMOTION.md`][promotion] on the exact launcher.

## Hardware

Read from the box on 2026-09-19; every number in this README was measured in this state.

- Host: ASRock X870 Taichi Creator, AMD Ryzen 7 9800X3D, 64 GB DDR5-6000 (2 × 32 GB), Ubuntu 24.04.4 LTS, kernel 7.0.0-30-generic.
- GPUs: two RTX 5090 32 GB (`sm_120`) on PCIe Gen5 x8/x8. Power limits are the cards' defaults, 600 W (ASUS, `cuda:0`) and 575 W (HP OEM, `cuda:1`). Memory clock offset +4500 MHz on both cards, core clock stock; a boot-time service applies both.
- Driver: NVIDIA 610.57.04 open kernel modules, CUDA 13.3 user-mode driver.
- Storage: one KIOXIA KBG80ZNV2T04 2 TB NVMe (ext4) holds the checkpoint, including the 18.5 GiB n-gram embedding table that decode reads rows from, and the NVMe prefix tier. A sequential 16 MiB `O_DIRECT` read of a tier segment ran at 6.7 GB/s (2026-09-19).

## In progress (queued on the box 2026-09-19)

- R553: prefill chunk size 2,048 / 1,024 / 512 against the decode stalls of running streams while a new long prompt prefills, and its cost in prefill rate.
- R554: decode at ~100k and ~200k prompt tokens on the served configuration (the depth row above is from the 3.05 bpw pack).
- R555: the MTP draft's 320 MiB embedding copy moved from cuda:1 to cuda:0, which keeps about 1.2 GiB unused; expected byte-identical and 2 more pool steps.
- R552b: why 2 of the 4 code requests at 4 streams report 1,861 tokens although `min_tokens` is 2,048 (seen in every code c4 run of R538, R540 and R546). Without streaming, all 14 requests of R552 reached the forced length; R552b checks whether the streamed probe counts frames when the stream carries no usage.

## Measured and not served

A deeper MTP draft for a single decoding job (depth 4 or 5 instead of 3), all arms at a 753,664-token pool: +4.8 / +5.2 % code and −5.4 / −8.2 % prose at 1 stream against depth 3, 16,384 / 49,152 fewer pool tokens than the served 819,200, and a different greedy output (2026-09-19, [R537][r537]). The K=3 MoE decode kernel without register spills: bit-exact, slower per call in 28 of 30 kernel cells, −0.37 % code at c1 over 8 boots with a 95 % interval of −0.83 to +0.09 % (2026-09-19, [R536][r536]). Recurrent checkpoints stored at the end of each reply are correct but save about 9k prefill tokens over a 120-call agent replay, below its run-to-run spread ([R524][r524]). A fused shared-expert kernel: after the side-stream overlap the shared expert's residual is 1.9 µs per layer at 4 rows, at most 0.6 % of a 1-stream step and 1.2 % at 4 streams ([R521][r521]); `EXL3_INT8_GEMV=0`, −0.3 % at c1 with a 95 % interval of ±1.2 % over 8 boots ([R520b][r520b]); 6 decode slots, +17 % at c6 and 9 % slower on an 8-agent replay ([R518][r518]); chunk 1024 for pool ([R483][r483], [R485][r485]); split [30, 31] at 393,216 ([R487][r487]); the n-gram table in host RAM, +1–2 % for 30.5 GiB ([R484][r484]); the host KV tier ([R358][r358], [R493][r493]); GDN state replay ([R496][r496]); a 4-bit MTP graft ([R498][r498]); prompt lookup, +3–4 % on code at c1 and flat at c4 ([R501][r501]); K8V4, +18 % pool for −11 % code at c1 ([R480][r480]); CPU-offloaded experts ([R482][r482]); MoE coop mode 3 ([R462][r462]); [exllamav3#303][pr303] MTP hot vocabulary ([R377][r377]); [exllamav3#246][pr246] and [#290][pr290] ([R365][r365]); our own 32-row MoE decode envelope ([R366][r366]). The same checkpoint on vLLM through [vllm-exl3][vllm-exl3] read 0.62× the c1 and 1.06–1.12× the c4 of this stack's 3.05 bpw configuration of 2026-09-18; work on that route stopped the same day ([vLLM route][vllm-route]).

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
[r527]: bench/results/r527-tp-bound.md
[r530]: bench/results/r530-promote-plefix.md
[r524]: bench/results/r524-recurrent-tip.md
[r526]: bench/results/r526-nvme-tier.md
[r532]: bench/results/r532-nvme-tier-r4.md
[r533]: bench/results/r533-e3-det-precise.md
[r534]: bench/results/r534-promote-nvme-tier.md
[r535]: bench/results/r535-promote-e3det.md
[r536]: bench/results/r536-nospill.md
[r537]: bench/results/r537-draft-depth.md
[r538]: bench/results/r538-decode-r6.md
[r540]: bench/results/r540-promote-r6.md
[r546]: bench/results/r546-promote-rawk.md
[r548]: bench/results/r548-promote-gdnbf16-ring.md
[r549]: bench/results/r549-hot-slots.md
[hot-slots]: bench/hot_slots.py
[r521]: bench/results/r521-shared-bound.md
[r522]: bench/results/r522-mtp-pruned.md
[r528-driver]: scripts/r528-promote-mtp-pruned.sh
[r522b-driver]: scripts/r522b-mtp-pruned-precise.sh
[r526-driver]: scripts/r526-nvme-tier.sh
[r518]: bench/results/r518-slots6.md
[r519]: bench/results/r519-profile-2p50.md
[r520]: bench/results/r520-int8gemv.md
[r520b]: bench/results/r520b-int8gemv-precise.md
