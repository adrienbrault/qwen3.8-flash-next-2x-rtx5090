# Qwen3.8-Flash-Next on 2× RTX 5090 (ExLlamaV3 + TabbyAPI)

Serving configuration, launcher, image recipe, kernel overlays, instruments and measurements for [Qwen3.8-Flash-Next][qwen-hf], served as [r0b0tlab's 2.50 bpw EXL3 pack][ckpt-250] by [TabbyAPI][tabby] on [ExLlamaV3][exl3] `dev` `5783a93` (v1.5.2) across two RTX 5090 cards. The window is 262,144 tokens, the KV cache is 8-bit, and vision, reasoning, tool calls, structured output and the checkpoint's own MTP draft head are all on.

Every number here was measured on one machine on the date given, and each links the write-up that names its raw results directory. None is an estimate. The index of experiments is [`bench/RESULTS.md`][results], newest first.

## Numbers

The served image changed on 2026-09-27 to the same stack on upstream ExLlamaV3 `dev` `5783a93` ([R785][r785]). The figures and tables in this section were measured on it on 2026-09-27 ([R787][r787]); comparisons with the previous image's figures of 2026-09-20 and 2026-09-25 cross two sessions and two images. Against the previous image in one session ([R784][r784], results `2026-09-27-r784-rebase-dev-r3-1104`, short prompts, 512 forced greedy tokens), the new one reads −1.3 to +1.6 % per-stream decode at 1, 4 and 8 streams for code and prose, with 95 % intervals between −4.1 and +3.7 %, which is within what one ABBA block resolves; it prefills a cold 90k-token prompt 1.134× faster and a 22.6k-token prompt 1.159× faster, and fits a page pool 8.3 % smaller. At long cached context, an agent-shaped replay in one session ([R786][r786], results `2026-09-27-r786-replay-abba-1324`, three ABBA pairs of fresh boots, prompts of median about 29,000 tokens, 91 % from the prefix cache) reads the new image at 1.015× the previous one per stream on the requests common to all arms (pairs 1.016, 0.981 and 1.048; 95 % interval 0.980 to 1.053). That excludes a loss larger than about 2 % and does not show the two equal; one pair resolves about ±5 to 7 %. R785's earlier 123.0 and 123.7 t/s against 128.6 for the previous image in [R728][r728] came from MTP acceptance under sampling (−2.3 %), higher concurrency (−1.3 %) and an older baseline image on another day, not from the NVMe tier or the pool. The replay does not exercise the smaller page pool, and its per-stream figures read about 10 % above its steady state because of the drain after the deadline.

Decode on `tabbyapi:rebase-dev-r3`, the served configuration (901,120-token pool, memory clock offset +4500, NVMe tier off for the measurement), measured 2026-09-27 15:06 to 15:23 UTC ([R787a][r787], results `2026-09-27-r787a-decode-curve`): `fn_bench --distinct`, so each stream has its own prompt; greedy, 1,024 forced tokens per request, short prompts, all streams starting together, two boots. Rates are tokens per second after each request's first token; the aggregate is the sum over the streams running together. Method: [How the numbers are measured](#how-the-numbers-are-measured).

![Decode alone against the standard benchmark, sum over streams and per stream](docs/img/std-bench.svg)

Solid lines: decode alone, the curve described above. Dashed lines: `vllm bench serve` on ShareGPT V3 and Spec-Bench ([R787d][r787], 2026-09-27, results `2026-09-27-r787d-std-bench`), where requests arrive as others finish, so their prefill interleaves with the running streams' decode; its aggregate is wall-clock output tok/s and its per-stream rate is 1000 / TPOT p50 ([Standard benchmark](#standard-benchmark-vllm-bench-serve)).

- The aggregate rises at every step from 1 to 8 streams, from 5 to 6 streams as well (code 684 to 715 t/s, prose 687 to 737); the draft policy keeps two draft tokens from 5 to 8 streams ([Conditions](#conditions)).
- The rates depend on how often the MTP draft is accepted, which depends on the text being generated: on these ~110-token prompts a decode step yields 2.28 to 2.37 tokens at 5 to 8 streams (median per request, [R787a][r787]). Each kind is one prompt: at 1 stream code reads 254.8 t/s and prose 276.8. The previous image read code 298.4 on the same prompt; the new one takes 395 decode steps for its 1,024 tokens where the previous one took 343 (2.59 against 2.99 tokens per step), at 1.7 % more time per step. Whether the greedy text itself changed is not recorded.
- Time to the first token is 0.13 s at 1 stream and 0.69 to 0.73 s at 8 streams. The numbers behind the figure are in [How the numbers are measured](#how-the-numbers-are-measured).
- These are batches on an otherwise idle server. A three-agent session measured from the server's per-request log on 2026-09-20 read 65.6 t/s per stream ([R583][r583]); that log under-counted every generation longer than about 4,096 tokens until 2026-09-26 ([R747][r747]), so the figure is too low and is withdrawn until re-measured.
- vLLM's serving benchmark on ShareGPT V3 and Spec-Bench, a wall-clock measure that includes prefill and request turnover, is in [Standard benchmark](#standard-benchmark-vllm-bench-serve).

![Cold prefill rate against prompt length, and decode rate at depth labelled with tokens per decode step](docs/img/prefill.svg)

Cold prefill keeps its rate up to the top of the window ([R787b][r787], 2026-09-27): 11,159 t/s at 29,932 tokens, 12,037 at 59,850, 12,108 at 119,530, 12,097 at 199,425 (16.5 s to the first token) and 11,910 at 239,110 (20.1 s). That is 1.13 to 1.16× [R580][r580]'s rates of 2026-09-20, across days and images, consistent with the same-session 1.134× at 90k tokens of [R784][r784]. The dashed lines are decode at 1 stream on an already-prefilled context of about 100k and 200k tokens ([R787c][r787]). The time per decode step stays between 10.3 and 10.7 ms from a short prompt to about 200k tokens. The rate rises from 260 / 267 t/s (code / prose) on a short prompt to 333 / 373 at about 100k and 296 / 348 at about 200k because the MTP draft is accepted more often on what the model writes after the filler, which is near-deterministic text: 3.49 / 3.85 and 3.16 / 3.73 tokens per step against 2.70 / 2.76. Each point is labelled with its tokens per step, and the time per step per point is in [How the numbers are measured](#how-the-numbers-are-measured). The rise is draft acceptance, not faster decoding over a long context.

| | value | source |
| --- | --- | --- |
| context window | 262,144 tokens | checkpoint |
| page pool | 901,120 tokens, 15,236 B per token: 1.52 GB per 100k, 13.7 GB total | [R784][r784], [R579][r579], [R717c][r717] |
| free VRAM after boot | 1,181 / 1,759 MiB | [R785][r785] |
| decode, agent-shaped edit, greedy (2026-09-19) | 1 stream: 233.0 t/s decode rate per request (median), 223.7 t/s end-to-end over the run; 4 streams, first wave: 502.2 t/s end-to-end burst aggregate, 143.1 t/s per stream end-to-end (time to first token included) | [R525][r525] |
| MTP drafts accepted per verify | code 1.57, prose 1.55 of 3 | [R572][r572] |
| 8-agent SWE-bench replay, 366 calls | wall 408.6 s; latency p50 3.76 s; queue wait p50 0.12 s | [R558][r558], [R557][r557] |
| prompt restored from the NVMe tier after a restart | 29,952 tokens in 0.69 s (cold 3.96 s); 119,808 in 0.99 s (cold 12.33 s) | [R534][r534] |
| long-context retrieval | 5/5 needles at 131k and at 240k prompt tokens | [R785][r785], [R548][r548], [R546][r546] |
| GSM8K 5-shot, n=500, no stop strings | 0.980 and 0.982 on two runs, 8 concurrent (0.978 on 2026-09-20) | [R785][r785], [R565][r565] |
| [tool-eval-bench][tool-eval], 69 × 4 | 86.5 ± 3.1 and 85.5 on two runs (84.0 ± 2.4 on 2026-09-20) | [R785][r785], [R565][r565] |
| [SWE-bench Verified][swebench], all 500, [mini-SWE-agent][mini-swe] 2.4.6, task containers without network | 397 resolved (79.4 %); 7 ended without a patch, 2 of them on server errors | [R586, R586d][r586] |
| boot to serving | ~20 s, warm kernel caches | [R525][r525] |

Also passing: structured output (`json_schema`, `response_format`, `regex_pattern`, thinking on and off, [R453][r453]); `tool_choice` `required` 48/48, named 4/4, 8/8 concurrent ([R529][r529]); a long prompt prefilled twice gives identical output ([R535][r535]).

### Conditions

- **Agent traffic.** On the same server, one 3,000-token generation at ~10k context decodes at 236 t/s after its first token alone, 154 while fresh ~45k-token prompts arrive every 8 seconds, and 141 with two other long generations running (2026-09-20). Sampling at temperature 0.6 costs a further 0 to 24 % ([R584, R585][r585]). The three-agent figure is generated tokens over generation time in the server's own request log ([R583][r583]). A 45k-token prompt is 22 chunks of 2,048 tokens, and each chunk is a forward pass in which the running streams do not decode; context depth and generation length do not account for the loss ([R583][r583]).
- **Draft depth.** The served policy drafts three tokens up to 4 streams and two at 5 to 8 streams; at 6 to 8 streams that is 18 to 24 verify rows, which the decode paths take since 2026-09-25 ([R717, R717c][r717]). Before then the cooperative MoE decode kernels took 16 verify rows, and the policy dropped to one draft token at 6 streams ([R560][r560], [R562][r562], [R576][r576]).
- **Code and prose.** Each kind is one prompt. Prose decodes 8.6 % faster than code per stream at 1 stream and 1.7 % faster at 8 streams ([R787a][r787]); on the previous image code was 7.1 % faster at 1 stream ([R719b][r719b]). At 1 stream the code prompt now yields 2.59 tokens per decode step against prose's 2.82, at the same time per step. On the previous engine the draft was accepted about as often on this benchmark's code prompt as on prose (1.57 against 1.55 drafts per verify, [R572][r572]).
- **Slots.** 8 slots raise throughput over 4 on synthetic concurrency but not on the agent replay, which spends two thirds of its wall time at 5–7 concurrent calls ([R558][r558], [R557][r557]).
- **KV precision.** 8-bit KV costs 0.2–0.3 accepted drafts per verify against full precision ([R572][r572]).
- **Page pool.** The pool is bounded by whichever card holds more of the 12 full-attention layers ([R579][r579]). The `gpu_split` budget does not move the boundary, and the decode graphs take 790 MiB on the bounding card ([R581][r581]).

## Served configuration

- Since 2026-09-27 15:01 CEST ([R785][r785]): image `tabbyapi:rebase-dev-r3`, the served engine stack ported onto upstream ExLlamaV3 `dev` `5783a93` (v1.5.2) with upstream's tiled hyper-connection prefill mix on (`EXL3_GR_MIX_TILED=1`), and TabbyAPI's loop patch in its fourth round. Upstream's code changes numerics, so greedy output differs from the previous image; it was promoted on GSM8K, tool-eval, needles, agentic edit and an agent replay instead of greedy identity ([R784][r784], [R785][r785]). The rollback is the image served from 2026-09-27 11:49 CEST ([R783][r783]), `tabbyapi:stack-r3-rows32-tokcount-loopthink3` = `stack-r3-rows32` ([R717c][r717]) plus the requeue token-count fix ([R747][r747]) and a TabbyAPI patch that ends a loop in the thinking with a forced `</think>` instead of a `stop` with no content. Draft-KV window off since 2026-09-25 05:52 CEST ([R728][r728]). Launcher [`scripts/launch-flashnext.sh`][launcher]. Its patches are listed under [What the stack is](#what-the-stack-is) and in [`docker/`][docker-readme]; each promotion is a row in [`docs/HISTORY.md`](docs/HISTORY.md), and every setting is explained in [`docs/CONFIG.md`](docs/CONFIG.md).
- 8 slots, 901,120-token page pool (983,040 until 2026-09-27), 8-bit KV.
- MTP draft depth 3 up to 4 jobs and 2 at 5 to 8 jobs (`[[4, 3], [8, 2]]`); the draft cache is page-indexed over the whole pool on the second GPU, without the 16,384-token window served from [R579][r579] to [R728][r728].
- Layer split `[30, 30]`, with the MTP draft component on the second GPU ([R694][r694]).

## What the stack is

- **Checkpoint**: [r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw][ckpt-250], routed experts at K = 2, 3 and 4 bits. It boots a 2.18× larger pool than [turboderp's 3.05 bpw pack][ckpt-turbo] and decodes 3–8 % faster except code at c1 ([R495b][r495b]); both score the same on GSM8K ([R509][r509]).
- **Engine**: [ExLlamaV3][exl3] `dev` `5783a93` (v1.5.2) under [TabbyAPI][tabby] `53da7919`, with the chain in [`docker/`][docker-readme] ported onto it since 2026-09-27; until then the chain was applied to v1.5.0, and the gains below were measured there. Every patch is opt-in by environment flag and was admitted with byte-identical greedy output, or with GSM8K, needles and tool-eval where it changes numerics:
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
  - a windowed MTP draft cache, a sink page plus the last 16,384 tokens per slot, +3.4 % page pool ([R579][r579]). In the image and off since 2026-09-25: under the window a prompt revived from the prompt cache in a later request group drafts 0.655 accepted per proposed token against 0.868 without it, and an agent-shaped replay decodes 2.0 % faster per stream without it ([R728][r728]).
  - a batched draft verifier, one verify call for all jobs, +1 to +4 % decode ([R646][r646]).
  - the MTP input-norm fusion, a fused int8 state-in-up mixer kernel and grouped MTP accept-prefill batching, +4.4 % decode at 8 streams, byte-identical output ([R653][r653]).
  - a slot returned to the recurrent-state pool when state construction fails; before, each failure lost one of the 8 slots until restart ([R676][r676]).
  - the MTP draft component loaded on the second GPU (`draft_gpu_split: [0, 32]`), +2.0 to +2.9 % prose decode at 1 to 8 streams and +2.2 % at 26k context, mean of three alternating pairs, greedy output identical ([R694][r694]).
  - the hyper-connection mixer's int8 kernels with each weight converted once per iteration, the loads batched and the reduction as a reduce-scatter ([R698, R699][r698]), and the routed-expert MoE decode kernels with a cp.async weight ring, an activation prefetch and one counter arrival per item ([R700b][r700b]); both bitwise-identical, admitted on the stack-track rule in [`docs/PROMOTION.md`][promotion] and promoted together: 1.051 to 1.093 times the per-stream prose decode rate at 1 to 8 streams and 1.064 times at 26k context, mean of three alternating pairs, greedy output identical ([R701][r701]).
  - a second batch of bitwise-identical decode changes: the mixer kernels' round 2 with per-row-count tiles ([R702][r702]); the Gated-DeltaNet recurrence held in registers, the QSA indexer as a parallel CUDA-graph branch and compile options for the QSA split and combine kernels ([R712][r712]); V2 twins of the dense K=4 decode GEMM, mgemm and gemv kernels ([R714][r714]); the round-2 MoE decode kernels with the shared expert forked before the router ([R713][r713]). Promoted together on 2026-09-24 as `stack-r3`: 1.149, 1.088 and 1.087 times the per-stream prose decode rate of `stack-r2` at 1, 4 and 8 streams at about 4k tokens of context, and 1.104 times at 26k tokens and 4 streams (canonical gate, 1,024 forced tokens, greedy, mean over three alternating pairs of the per-request median, results `2026-09-24-r716b-stack-r3`); logits identical to `stack-r2` at every served decode shape ([R716b, R716c][r716b]).
  - decode paths for 17 to 32 rows (the routed-expert MoE decode in one launch, the shared expert and the dense GEMMs at 32 rows), so that 6 to 8 jobs verify at draft depth 2 instead of 1; host code only, each MoE output at 17 to 32 rows equal to two served 16-row calls. Promoted on 2026-09-25 as `stack-r3-rows32` with the policy `[[4, 3], [8, 2]]` and a page pool 1.6 % smaller: about +3 % per-stream decode (0 to +6 % across cells) at 6 and 8 streams at the 16k and 32k context settings (prompts of 12,000 to 44,000 tokens), and up to +16 to +21 % at 6 streams with 4k context, where the output is highly predictable; 1 to 5 streams unchanged (1,024 forced tokens, greedy, code and prose, mean over three alternating pairs of the per-boot median, results `2026-09-24-r717c-rows32-context`, [R717, R717b, R717c][r717]).
  - the whole chain ported onto upstream `dev` `5783a93`, which brings upstream's tiled hyper-connection prefill mix ([`825db5b`][exl3-825db5b], measured alone in [R568][r568]), fp16 GDN prefill projections and a deterministic router GEMM, and changes greedy output: cold prefill 1.134× at 90k tokens, page pool −8.3 %, short-prompt decode within one ABBA block's resolution ([R784][r784]); agent-replay decode 1.015× the previous image in one session, 95 % interval 0.980 to 1.053 ([R786][r786]). Promoted on 2026-09-27 as `rebase-dev-r3`.
- **Cards**: layer split, 30 GB of weights and cache per card. `qwen4_exp` raises `NotImplementedError` for tensor parallelism in this engine, so the cards take turns over their own layers, and one stream keeps each card 44–47 % busy (2026-09-16, 3.05 bpw pack, [GPU duty cycle][duty]). Expert parallelism was built and measured at −9.5 % at 1 stream (results `2026-09-16-r408-ep-served`). Tensor parallelism was bounded before it was built: from measured half-work kernel times and all-reduce costs, a TP step would be at most 1.07–1.08× faster at 1 and 4 streams (2026-09-19, [R527][r527]).
- **Speculative decoding**: the checkpoint's MTP head, depth 3 up to 4 concurrent jobs and depth 2 at 5 to 8 (`[[4, 3], [8, 2]]`, since [R717c][r717]; from [R576][r576] to then depth 1 at 6 to 8, `[[4, 3], [5, 2], [8, 1]]`). Confidence-gated dynamic depth crashed at 4 streams ([R497][r497]).
- **Sampler fallbacks**: temperature 0.6, top_k 20, top_p 0.95 with `force: false`, so a client that sends its own sampler keeps it. Without a preset TabbyAPI serves sampler-less requests at temperature 1.0 untruncated ([`docs/GOTCHAS.md`][gotchas]).
- **Loop detection**: TabbyAPI's default window of 800 tokens, which ends a repeated period of up to 400 tokens. Since [R783][r783] a loop in the thinking of a chat request forces `</think>` and the model answers or calls a tool; before, the request ended as `stop` with reasoning and no content, which agent clients that promote reasoning on an empty `stop` show as the reply ([`docs/GOTCHAS.md`][gotchas] 25). Since [R785][r785] a second detector on the thinking also ends periods of up to 1,000 tokens once three copies are generated; on the GPU no request has reached that case yet ([`loop-think-r4`](docker/overlays/loop-think-r4/README.md)).
- **Guard rails**: the launcher refuses to start without the checkpoint or the image, stops any other engine holding the cards, waits for them to drain and mounts the kernel caches. Every promotion re-runs the gates in [`docs/PROMOTION.md`][promotion] on the exact launcher.

## Hardware

Read from the box on 2026-09-19. Every number in this README was measured on this hardware and driver; the memory clock offset is stated per period below.

- Host: ASRock X870 Taichi Creator, AMD Ryzen 7 9800X3D, 64 GB DDR5-6000 (2 × 32 GB), Ubuntu 24.04.4 LTS, kernel 7.0.0-30-generic.
- GPUs: two RTX 5090 32 GB (`sm_120`) on PCIe Gen5 x8/x8. Power limits are the cards' defaults, 600 W (ASUS, `cuda:0`) and 575 W (HP OEM, `cuda:1`). Memory clock offset +4500 MHz on both cards, core clock stock. A boot-time service applies them once per host boot, and since 2026-09-25 the launcher sets the memory offset before every engine boot and logs the readback. The offset had reset to 0 between 2026-09-03 and 2026-09-19 without a reboot, so the numbers measured from 2026-09-19 to 2026-09-25 02:22 UTC ran at the stock memory clock; +4500 is +14.3 % DRAM bandwidth and +1.7 to +1.8 % decode per stream at 1 stream ([R726][r726]).
- Driver: NVIDIA 610.57.04 open kernel modules, CUDA 13.3 user-mode driver.
- Storage: one KIOXIA KBG80ZNV2T04 2 TB NVMe (ext4) holds the checkpoint, including the 18.5 GiB n-gram embedding table that decode reads rows from, and the NVMe prefix tier. A sequential 16 MiB `O_DIRECT` read of a tier segment ran at 6.7 GB/s (2026-09-19).

## In progress (2026-09-20)

- One fused kernel per layer for the GDN linear-attention decode block, claimed bit-exact; GDN is 0.80 ms of a 13.9 ms 1-stream step ([R519][r519]).

## Measured and not served

Each entry names the change and the number that kept it out of the served configuration. c1, c4 and c6 mean 1, 4 and 6 concurrent streams.

- Mixed draft depth per job inside one verify batch, so 5 to 7 streams fill the 16 verify rows: 0.79× at 5 streams and 0.84× at 7, code, 256 forced tokens, greedy. The verify window grows to the deepest job in the batch, which adds a sequential draft level to every step ([R678b][r678b]).
- 17 to 32 verify rows on the cooperative MoE kernels, as two calls of at most 16 rows: bit-identical, −14.5 % at 8 streams against drafting one token ([R566][r566]). One launch of up to 32 rows is served since 2026-09-25 ([R717][r717]).
- Draft depth 2 above 4 jobs on the 16-row decode paths: −32 to −39 % at 6 and 8 streams, because 18 and 24 verify rows fell off the cooperative MoE decode kernels (2026-09-19, [R560][r560], [R562][r562]). With the rows32 paths depth 2 at 6 to 8 jobs is served since 2026-09-25 ([R717][r717]).
- Draft depth 4 at 1 stream, with or without a controller: costs 32,768 page-pool tokens and returns at most about +2 % ([R567][r567]).
- A deeper MTP draft for a single decoding job (depth 4 or 5 instead of 3), all arms at a 753,664-token pool: +4.8 / +5.2 % code and −5.4 / −8.2 % prose at 1 stream against depth 3, 16,384 / 49,152 fewer pool tokens than the served 819,200 of that date, and a different greedy output (2026-09-19, [R537][r537]).
- Adaptive MTP draft depth, round 2: −2.4 to −3.5 % prose at 1 stream, no gain on code ([R556][r556]).
- Two draft chains verified together: +5 to +6.5 % accepted tokens for twice the verify rows, modelled at −10 to −13 % ([R564][r564]).
- A 4,096-token prefill chunk: does not boot beside the page pool ([R574][r574]).
- Prefill chunk 1,024 or 512 instead of 2,048: running streams get 1.5× the decode frames while another request prefills, but cold prefill runs at about half the rate and the new request waits 1.4–1.7× longer for its first token ([R553][r553]). Chunk 1,024 for pool size: [R483][r483], [R485][r485].
- The served stack on upstream `dev` `1d64111`: −49,152 pool tokens and 1–3 % decode ([R563][r563]). A later port onto `dev` `5783a93` is served since 2026-09-27, at −81,920 pool tokens ([R784][r784]).
- The MTP draft's embedding copy on cuda:0: +16,384 pool tokens for −1.8 % code and −2.0 % prose at 1 stream ([R555][r555]).
- The K=3 MoE decode kernel without register spills: bit-exact, slower per call in 28 of 30 kernel cells, −0.37 % code at c1 over 8 boots with a 95 % interval of −0.83 to +0.09 % (2026-09-19, [R536][r536]).
- Recurrent checkpoints stored at the end of each reply: correct, but they save about 9k prefill tokens over a 120-call agent replay, below its run-to-run spread ([R524][r524]).
- A fused shared-expert kernel: after the side-stream overlap the shared expert's residual is 1.9 µs per layer at 4 rows, at most 0.6 % of a 1-stream step and 1.2 % at 4 streams ([R521][r521]).
- `EXL3_INT8_GEMV=0`: −0.3 % at c1 with a 95 % interval of ±1.2 % over 8 boots ([R520b][r520b]).
- 6 decode slots: +17 % at c6 and 9 % slower on an 8-agent replay, measured before the bf16 GDN state halved the per-slot cost ([R518][r518]).
- Split [30, 31] at 393,216 ([R487][r487]).
- The n-gram table in host RAM: +1–2 % for 30.5 GiB ([R484][r484]).
- The host KV tier ([R358][r358], [R493][r493]); GDN state replay ([R496][r496]); a 4-bit MTP graft ([R498][r498]); CPU-offloaded experts ([R482][r482]); MoE coop mode 3 ([R462][r462]).
- Prompt lookup: +3–4 % on code at c1, flat at c4 ([R501][r501]).
- K8V4: +18 % pool for −11 % code at c1 ([R480][r480]).
- [exllamav3#303][pr303] MTP hot vocabulary ([R377][r377]); [exllamav3#246][pr246] and [#290][pr290] ([R365][r365]); a 32-row MoE decode envelope written for this stack ([R366][r366]).
- The same checkpoint on vLLM through [vllm-exl3][vllm-exl3]: 0.62× the c1 and 1.06–1.12× the c4 of this stack's 3.05 bpw configuration of 2026-09-18. Work on that route stopped the same day ([vLLM route][vllm-route]).

## How the numbers are measured

**Decode** ([R787a][r787], 2026-09-27 15:06 to 15:23 UTC, results `2026-09-27-r787a-decode-curve`, driver [`scripts/r787a-decode-curve.sh`](scripts/r787a-decode-curve.sh), R719b's driver with provenance checks): `fn_bench` ([`bench/probe.py`][probe]) against the served launcher of `tabbyapi:rebase-dev-r3` (42 environment keys, memory clock offset +4500) on two boots, greedy, 1,024 forced tokens (`min_tokens`), a warm-up round plus three recorded rounds per shape, NVMe tier off. `--distinct` appends a per-stream suffix to each request's prompt, so no two streams of a round share a prompt; prompts are 118 tokens (code) and 106 tokens (prose), and no request revives a cached prefix. The two boots agree within 1.3 % on the per-stream rate and on the decode aggregate in 14 of 16 cells; at 6 streams they differ by 2.5 % (code) and 3.3 % (prose) per stream.

- **Decode rate per stream**: the median over requests of (tokens − 1) / (time of the last token − time of the first token).
- **Decode aggregate**: the sum of the decode rates of the requests running together. At 2 to 8 streams every stream decodes during 91.1 to 98.9 % of the round's mean decode window, so the sum overstates the rate the streams sustain together by at most about 9 %.
- **End-to-end burst aggregate**: all streams' tokens over the round's wall time, including time to the first token and the tail after the first stream finishes.

| streams | decode per stream, t/s, code / prose | decode aggregate, t/s, code / prose | time to first token, s, code / prose | end-to-end burst aggregate, t/s, code / prose |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 254.8 / 276.8 | 255 / 277 | 0.13 / 0.13 | 247 / 268 |
| 2 | 208.1 / 201.3 | 418 / 405 | 0.24 / 0.22 | 395 / 379 |
| 3 | 178.5 / 175.4 | 529 / 525 | 0.34 / 0.32 | 487 / 491 |
| 4 | 151.5 / 154.0 | 618 / 619 | 0.45 / 0.43 | 556 / 571 |
| 5 | 136.1 / 137.3 | 684 / 687 | 0.55 / 0.53 | 627 / 632 |
| 6 | 119.1 / 122.7 | 715 / 737 | 0.64 / 0.58 | 655 / 679 |
| 7 | 113.1 / 113.0 | 800 / 790 | 0.67 / 0.64 | 720 / 728 |
| 8 | 104.3 / 106.1 | 834 / 852 | 0.73 / 0.69 | 764 / 768 |

**Prefill** ([R787b][r787], 2026-09-27 15:23 to 15:27 UTC, results `2026-09-27-r787b-prefill-curve`, R580's protocol): three salted cold prompts per target, counted by the server, NVMe tier off; 11,159 to 12,108 t/s from 29,932 to 239,110 tokens, 1.13 to 1.16× [R580][r580]'s rates of 2026-09-20, which ran on an older image at the stock memory clock ([R726][r726]). The decode-at-depth points are [R787c][r787] (results `2026-09-27-r787c-depth-decode`, R554's probe with the code targets moved to about 100k and 200k tokens, where [R554][r554]'s landed at 180k): 1 stream, 2,048 forced greedy tokens, the mean of the cold and the cached run. Time per decode step is decode seconds over streamed frames, one frame per verify step:

| prompt tokens, code / prose | decode, t/s, code / prose | tokens per decode step, code / prose | ms per decode step, code / prose |
| ---: | ---: | ---: | ---: |
| 101 / 89 | 260 / 267 | 2.70 / 2.76 | 10.4 / 10.4 |
| 99,812 / 99,724 | 333 / 373 | 3.49 / 3.85 | 10.5 / 10.3 |
| 199,083 / 199,735 | 296 / 348 | 3.16 / 3.73 | 10.7 / 10.7 |

Figures are drawn from the raw records in `bench/results/` by [`bench/plot.py`](bench/plot.py) (`uv run bench/plot.py`).

## Standard benchmark (`vllm bench serve`)

[vLLM][vllm]'s serving benchmark v0.30.0 on [ShareGPT V3][sharegpt] and [Spec-Bench][spec-bench], a harness and datasets that third-party serving results also use, run against the served configuration through [`bench/vllm_bench_tabby.py`][vllm-bench-tabby] ([R787d][r787], 2026-09-27 15:30 to 17:00 UTC, results `2026-09-27-r787d-std-bench`, driver [`scripts/r787d-std-bench.sh`](scripts/r787d-std-bench.sh), [R731b][r731b]'s protocol and sample). Output tok/s is a wall-clock figure: all completion tokens over the time from the first request's start to the last completion, closed loop at `c` concurrent requests, including prefill, time to the first token and the turnover between requests. It is therefore lower than the decode aggregate in [Numbers](#numbers), which sums steady-state decode rates with no prefill in the window. The headline metrics of this repository are the per-stream decode rate and the decode aggregate; output tok/s is a secondary metric, and a per-stream rate multiplied by the stream count is not called an aggregate.

Conditions: `tabbyapi:rebase-dev-r3` on the served launcher, power limits at the stock 600 / 575 W, core clock offset 0, memory offset +4500, NVMe tier off. Every cell ran on a fresh boot (0 cached prompt tokens over 7,040 requests), and the matrix ran twice, passes A and B; cells are the mean of the two, and the p99 columns give both passes as a range where they differ. Every concurrency sends the same sample: ShareGPT 400 conversations drawn with seed 7310, Spec-Bench all 480 questions of its 13 categories. Inputs are short: mean 272 / 322 tokens, maximum 1,070 / 1,540 (ShareGPT / Spec-Bench). Requests are greedy with thinking on, and `min_tokens` forces each output to the ShareGPT reference reply's length (mean 210 tokens) or to 256 tokens, so the outputs are truncated reasoning, not answers.

In the figure at the top, per stream, the standard benchmark is level with decode alone or above it at 1 and 2 streams (ShareGPT / Spec-Bench 291 / 295 and 207 / 208 t/s against code / prose 255 / 277 and 208 / 201) and falls below it from 4 streams (128 / 131 against 152 / 154; 71 / 74 against 104 / 106 at 8), where requests arrive while others decode and their prefill chunks take steps from the running streams. The aggregate gap also includes each request's time to the first token and the turnover between requests, which weigh more here than in the decode curve because outputs are ~210-256 tokens instead of 1,024. At 1 stream the solid lines are one prompt per kind, at 2.59 (code) and 2.82 (prose) tokens per decode step; R787d's τ over 400 ShareGPT and 480 Spec-Bench prompts is 2.65 and 2.87, as in R731b. Both panels come from `bench/plot.py`, which reads the published raw records of R787a and R787d. Against R731b on the previous image (2026-09-25), output tok/s is 0.996 to 1.007× in every cell; per-stream is 0.990 to 1.001× except ShareGPT at 4 streams, 0.964× (132.7 to 128.0 t/s); TTFT p50 is 0.7 to 6.8 % lower in 7 cells and 3.7 % higher at Spec-Bench 8 streams ([R787][r787]).

**ShareGPT V3**

| streams | output tok/s (wall clock) | A/B spread | req/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | per-stream tok/s (1000 / TPOT p50) | E2E p50 (s) | τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 221.6 | 0.08 % | 1.05 | 137 / 322–359 | 3.44 / 4.56–4.57 | 291 | 0.64 | 2.65 |
| 2 | 307.0 | 0.09 % | 1.46 | 194 / 541–542 | 4.83 / 9.06–9.19 | 207 | 0.90 | 2.65 |
| 4 | 397.0 | 0.36 % | 1.88 | 282 / 639–693 | 7.81 / 17.01–18.99 | 128 | 1.35 | 2.66 |
| 8 | 471.6 | 0.03 % | 2.24 | 382 / 1,343–1,500 | 14.02 / 23.16–24.34 | 71 | 2.23 | 2.30 |

**Spec-Bench**

| streams | output tok/s (wall clock) | A/B spread | req/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | per-stream tok/s (1000 / TPOT p50) | E2E p50 (s) | τ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 244.5 | 0.39 % | 0.95 | 130 / 384–392 | 3.40 / 4.34 | 295 | 1.04 | 2.87 |
| 2 | 343.4 | 0.07 % | 1.34 | 183 / 496–503 | 4.79 / 6.58–6.67 | 208 | 1.47 | 2.87 |
| 4 | 445.0 | 0.04 % | 1.74 | 248 / 650–657 | 7.66 / 10.75–10.78 | 131 | 2.27 | 2.87 |
| 8 | 524.7 | 0.10 % | 2.05 | 394 / 978–1,028 | 13.46 / 17.40–17.42 | 74 | 3.83 | 2.43 |

On the previous image, at 8 streams the wall-clock figure and the `fn_bench` decode aggregate differ mainly by prefill interleaved with decode (R731, 2026-09-25, results `2026-09-25-r731-std-bench`: the same samples with both cards limited to 400 W, and R731b at stock differs from it by −0.72 to +0.87 % per cell). ShareGPT / Spec-Bench: 471 / 529 tok/s wall clock; 541 / 596 counting only each request's time after its first token; 786 / 803 after also removing the frames longer than twice the median inter-token latency, which are the steps that run other requests' prefill chunks and hold 35 / 31 % of decode time. That is 98.2 / 100.4 tok/s per stream, against `fn_bench`'s 105.8 / 106.9 per stream and 845 / 853 decode aggregate (code / prose, [R719b][r719b]); the remainder is a longer decode step, not τ ([R731b][r731b]). This decomposition has not been repeated on R787d's records.

- TTFT is the client-side time to the first streamed token, which with thinking on is the first reasoning token.
- TPOT is (E2E − TTFT) / (output tokens − 1) per request and includes the time a request waits while other requests' prefills run. Per-stream tok/s is 1000 / TPOT p50, derived from it, and is not comparable to the `fn_bench` per-stream decode rate.
- τ is tokens per verify step from the server log, Σ generated / (Σ generated − Σ accepted drafts); it counts the prefill step, which puts it about 1 % low. It falls at 8 streams because the draft policy drafts two tokens above 4 streams. Spec-Bench τ is not comparable to acceptance figures published with Spec-Bench.
- The A/B spread is replication within one session. On the previous image the same samples in another session (R731) differed by up to 0.9 % per cell, so each cell reads as ±1 %. ShareGPT's TPOT p99 at 4 streams moves between sessions (13.6 to 19.4 ms in R731).
- No request ended on exllamav3's loop detector in R787d; in R731b it ended about one request per ShareGPT cell before its forced length, at most 0.36 % of the tokens.
- vLLM's `max_concurrent_requests` and `max_output_tokens_per_s` result fields are not used: the first counts requests that touch a one-second bucket, the second counts streamed frames per second.

## Reproducing a boot

```sh
# the served image, published 2026-09-27 (the served layers plus one label-only layer); tag it as the launcher expects
docker pull ghcr.io/adrienbrault/qwen3.8-flash-next-2x-rtx5090@sha256:5fecdc9d8a30eb29c197f0a1e2c5af54bff65d603c4bc1ea7541b57c3ff19a32
docker tag  ghcr.io/adrienbrault/qwen3.8-flash-next-2x-rtx5090@sha256:5fecdc9d8a30eb29c197f0a1e2c5af54bff65d603c4bc1ea7541b57c3ff19a32 tabbyapi:rebase-dev-r3
# the rollback image, tagged tabbyapi:stack-r3-rows32-tokcount-loopthink3 and served with DAILY_IMG set to that tag, CACHE=983040,
# TUNEDIR=/srv/qwen5090/.exl3cache and EXTRA_ENV without EXL3_GR_MIX_TILED=1:
#   ghcr.io/adrienbrault/qwen3.8-flash-next-2x-rtx5090@sha256:b5fc90c6c987c72356dcfa2f329982ecec2efca08ee56e85c9e5d662d1d52074

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
| [`bench/vllm_bench_tabby.py`][vllm-bench-tabby], [`bench/std_bench_summary.py`][std-bench-summary] | `vllm bench serve` v0.30.0 adapted to TabbyAPI (usage frame, CRLF events, `min_tokens`, Spec-Bench templated once); the tables and checks of a standard-benchmark run |
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

Benchmarks and harnesses: [tool-eval-bench][tool-eval] · [mini-SWE-agent][mini-swe] · [SWE-bench][swebench] · [lm-evaluation-harness][lm-eval] · [halogen][halogen] · [vLLM][vllm] (`vllm bench serve`) · [ShareGPT V3][sharegpt] · [Spec-Bench][spec-bench]

[qwen-hf]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
[ckpt-250]: https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw
[ckpt-turbo]: https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3
[exl3]: https://github.com/turboderp-org/exllamav3
[exl3-825db5b]: https://github.com/turboderp-org/exllamav3/commit/825db5b
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
[vllm]: https://github.com/vllm-project/vllm
[sharegpt]: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
[spec-bench]: https://github.com/hemingkx/Spec-Bench
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
[vllm-bench-tabby]: bench/vllm_bench_tabby.py
[std-bench-summary]: bench/std_bench_summary.py
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
[r552b]: bench/results/r552b-c4-stream-count.md
[r553]: bench/results/r553-chunk-hot-stall.md
[r554]: bench/results/r554-depth-decode.md
[r555]: bench/results/r555-headdev-mirror.md
[r556]: bench/results/r556-adaptive-draft-r2.md
[r557]: bench/results/r557-agent-replay-daily.md
[r558]: bench/results/r558-slots8.md
[r560]: bench/results/r560-c8-policy.md
[r561]: bench/results/r561-promote-slots8.md
[r562]: bench/results/r562-profile-c8.md
[r559]: bench/results/r559-ngram-prefetch.md
[r565]: bench/results/r565-promote-ngram-prefetch.md
[r563]: bench/results/r563-rebase-dev.md
[r564]: bench/results/r564-draft-topk.md
[r566]: bench/results/r566-moe-rows32.md
[r567]: bench/results/r567-adaptive-draft-r3.md
[r568]: bench/results/r568-rebase-prefill.md
[r569]: bench/results/r569-mtp-kv-window.md
[r570]: bench/results/r570-c5-draft-policy.md
[r571]: bench/results/r570-c5-draft-policy.md
[r572]: bench/results/r572-mtp-acceptance.md
[r573]: bench/results/r573-mtp-kv-window-screen.md
[r574]: bench/results/r574-chunk4096.md
[r575]: bench/results/r575-promote-mtp-kv-window.md
[r576]: bench/results/r576-promote-c5-policy.md
[r579]: bench/results/r579-promote-mtp-kv-window.md
[r580]: bench/results/r580-decode-curve.md
[r704]: bench/results/r704-decode-curve.md
[r719]: bench/results/r719-decode-curve.md
[r719b]: bench/results/r719b-decode-curve.md
[r731b]: bench/results/r731b-std-bench.md
[r726]: bench/results/r726-memoc.md
[r728]: bench/results/r728-promote-window-off.md
[r747]: bench/results/r747-tokcount.md
[r783]: bench/results/r783-loopthink.md
[r587]: bench/results/r587-tabby-metrics.md
[r583]: bench/results/r583-long-generation.md
[r585]: bench/results/r585-prefill-interference.md
[r784]: bench/results/r784-rebase-dev-r3.md
[r785]: bench/results/r785-promote-rebase-r3.md
[r786]: bench/results/r786-replay-abba.md
[r787]: bench/results/r787-bench-refresh.md
[r581]: bench/results/r581-split-rebalance.md
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
[r646]: bench/results/r646-verifybatch.md
[r653]: bench/results/r653-stack.md
[r676]: bench/results/r676-slotfix.md
[r694]: bench/results/r694-mtp-card1.md
[r678b]: bench/results/r678b-fill16.md
[r698]: bench/results/r698-hcfast.md
[r700b]: bench/results/r700b-moefast.md
[r701]: bench/results/r701-stack-r2.md
[r702]: bench/results/r702-hcfast-r2.md
[r712]: bench/results/r712-latchain-r1.md
[r713]: bench/results/r713-moefast-r3.md
[r714]: bench/results/r714-densegemm-r2.md
[r716b]: bench/results/r716b-stack-r3.md
[r717]: bench/results/r717-rows32.md
[r586]: bench/results/r586-swebench-500.md
