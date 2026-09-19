# Results

One file per experiment, newest first. Each file names its results directory on the serving host, links its driver in [`scripts/`](../scripts/), and links its raw records where they are stored in this repository. The served configuration is in [`docs/CONFIG.md`](../docs/CONFIG.md); how it got there is in [`docs/HISTORY.md`](../docs/HISTORY.md).

## 2026-09-19

- [R517: decode round 4 and grouped MoE prefill are served together; cold prefill 9.5k t/s at 60k and 10.0k at 120k](results/r517-promote-stack.md)
- [R516: int8 mixer weights buy a 819,200-token pool (+4 %) at unchanged quality and 0 to −2 % decode](results/r516-int8-mixer-pool.md)
- [R514: decode round 4 group I is served](results/r514-promote-r4i.md)
- [SWE-bench agent runs: 39 calls and 36k tokens of final context per instance, and where the wall time went](results/swebench-agent-cost.md)

## 2026-09-18

- [R513: grouped MoE prefill round 2 (all K) is +16 % at 30k, +21 % at 60k and +19 % at 120k cold prefill, decode unchanged](results/r513-prefill-e3-r2.md)
- [R511: the 2.50bpw pack at a 786,432-token pool is the served configuration](results/r511-promote-2p50.md)
- [R507: grouped MoE prefill (E3) round 1 is ×1.48–1.55 per layer at 2,048 rows and +6–8 % end to end with half the layers eligible](results/r507-prefill-e3.md)
- [R504: the reference-base image is identical in output and speed to the served image](results/r504-refbase.md)
- [R503 and R509: GSM8K on this checkpoint was measuring lm-eval's stop strings; without them 3.05bpw and 2.50bpw score 0.980 and 0.978](results/r509-gsm8k-nostop.md)
- [R501: prompt lookup beside the MTP draft is lossless and +3 to +4 % on code at c1, flat at c4](results/r501-prompt-lookup.md)
- [R499: decode round 4 — pinned draft staging, batched verify and a 64K-token draft head give +2 to +3 % with identical output; int8 mixer weights give no speed](results/r499-decode-r4.md)
- [R498: a 4-bit MTP layer grafted onto the 3.05 pack costs pool, gains nothing at c1 and fails at c4](results/r498-mtp4-graft.md)
- [R497: confidence-gated dynamic draft crashes at c4 with a CUDA-graph out-of-memory](results/r497-draft-confidence.md)
- [R496: GDN state replay serves a 425,984-token pool on the 3.05 pack with canonical output, prose decode −7 to −8 %](results/r496-gdn-state-r3.md)
- [R495b: the 2.50bpw pack boots a 786,432-token pool (2.18×) and decodes 3–8 % faster except code c1](results/r495b-2p50-audition.md)
- [R493: the host-RAM KV tier turns a 12.7 s re-prefill into 0.5–0.7 s, for 16 GiB of host RAM](results/r493-host-kv-tier.md)
- [R492: decode is nearly flat with prompt depth; cold prefill 8.2–8.4k t/s at 100–180k tokens](results/r492-depth.md)
- [R490, R491, R491b: the shared expert on a side CUDA stream is byte-identical and +3 to +7 % decode; promoted](results/r490-shared-overlap.md)
- [R487 and R488: split [30, 31] at 393,216 tokens reads as fast as the served split once content is averaged](results/r487-pool-393k.md)
- [R485 and R486: the pool frontier at chunk 2048 is 393,216 under every split, and chunk 1024 switches the prefill pipeline off](results/r485-pool-frontier.md)
- [R484: the n-gram table in host RAM is byte-identical and worth 1–2 % at c1, for 30.5 GiB](results/r484-ngram-ram.md)
- [R483: chunk 1024 with split [30, 31] boots 409,600 tokens, but prefill halves](results/r483-exl3-pool-chunk.md)
- [R482: Cold experts on the CPU](results/r482-cold-experts.md)
- [R480: Four slots, K8V4 and one MoE layer on the CPU: the page pool at 8-bit KV](results/r480-exl3-pool.md)
- [R477: Code and prose from one boot of the served configuration](results/r477-daily-prose-code.md)
- [ExLlamaV3 against vLLM on the same checkpoint](results/vllm-exl3-route.md)

## 2026-09-17

- [R462: MoE coop mode 3 is bit-identical to mode 1 and 2–3 % slower](results/r462-moecoop-v3-ab.md)
- [R460 and R461: MoE coop V2 is byte-identical and +4 % at c4, +8 % at c8; five gates pass](results/r460-moecoop-v2-ab.md)
- [R453: structured output works on the served stack, thinking on and off, at c4](results/r453-exl3-structured.md)
- [R452: at 8 slots the 8-bit pool ceiling is 262,144 tokens under any split](results/r452-exl3-cache-bits.md)
- [R442 and R446: the two-card prefill pipeline cuts cold prefill time by 36 % at 30k and 41 % at 120k with identical output](results/r442-ppipe.md)
- [R428: mixer V2 and the GDN host-gap rewind are identical at MIN_R 1 and +12 % at c4, +8 % at c8](results/r428-hcmix2-stack-ab.md)
- [R421: the wide stage-B MoE tile at 128 slots or more is byte-identical and +7.5 % at c8](results/r421-coopwide-ab.md)
- [R414: the fused MoE decode path admits 16 rows, and the draft policy becomes [[4, 3], [8, 1]]](results/r414-bszn16.md)

## 2026-09-16

- [R377: #303 MTP hot vocabulary is inapplicable on this box, by construction](results/r377-hotvocab-on.md)
- [R368: GSM8K at n=1319](results/r368-gsm8k-1319.md)
- [R367: The slot ladder and 12-agent admission](results/r367-slots.md)
- [R366: Our own 32-row MoE decode envelope: correct, and no effect](results/r366-ourkernel.md)
- [R365: upstream #246 changes numerics for a prefill gain within noise; #290 is output-neutral](results/r365-kernels.md)
- [R362: PR #337, the layer-split device context](results/r362-pr337.md)
- [R359: SWE-bench Verified, four subsets](results/r359-swebench.md)
- [R358: The host KV tier is not a lever](results/r358-hostkv.md)
- [R357: Quality: tool-eval 69×4](results/r357-tooleval.md)
- [R356: The promoted configuration, validated on the real request](results/r356-promoted.md)
- [R355: Quality: GSM8K as served](results/r355-fn-gsm8k.md)
- [R354: Both levers together](results/r354-combined.md)
- [R348: Capabilities](results/r348-capabilities.md)
- [R347: Stamina](results/r347-soak.md)
- [R345: The pool, measured with independent contexts](results/r345-pool.md)
- [R343: Decode and TTFT against prompt depth](results/r343-depth.md)
- [R341: QSA sparse multi-job: the A/B](results/r341-qsa.md)
- [R340: Concurrency-indexed draft depth: the A/B](results/r340-ci-depth.md)
- [R339: gates on the first served image — long-context retrieval, admission of deep contexts, decode at the requeue boundary](results/r339-gates.md)
- [R339: Concurrency, short generations](results/r339-honest-conc.md)
- [R339: Decode, code, forced length](results/r339-longgen.md)
- [A real agent turn — DSH session `session-652732d8`, 2026-09-16](results/agent-turn-2026-09-16.md)
- [Boot and footprint](results/boot-footprint.md)
- [Decode is content-dependent — same box, same day](results/decode-content-dependence.md)
- [GPU duty cycle — measured under a live agentic load](results/gpu-duty-cycle.md)
