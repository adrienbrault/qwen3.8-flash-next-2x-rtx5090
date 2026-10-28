# Flash-Next pf2lock screen (2026-10-28-154823-flashnext-pf2lock-screen-screen-3242976)

Launchers: deterministic renders of the sealed daily launcher (A 3 hunks, B 4 hunks; `launch-A.diff`, `launch-B.diff`), container `flashnext-pf2lock-exp-2026-10-28-154823-flashnext-pf2lock-screen-screen-3242976`. A/B config check: PASS (B-only keys ).

A = served `tabbyapi:tune268-r2` (sha256:2239954c3723…), B = candidate sha256:da386167cc76…; pool 819200, split [30, 30], 8 slots, MTP d3, :18022, FASTWARM=1, no NVMe tier. B delta vs served flags: `{"added_extra_env": []}`.

**Verdict (predeclared): parity or better.** code c1 -0.3 %, code c4 aggregate -0.5 %, replay span -1.4 % (rule: c1 >= -2.0 %, c4 >= -2.0 %, replay <= +3.0 %; mean B1,B2 vs mean A1,A2). Greedy identity is reported below, not gated. No promotion.

## FORMAT CHECK MISMATCH (reported, not a harness failure; the user decides)

- **B0: BOS probe returned null for completions_default (no usage.prompt_tokens)**
- **A1: BOS probe returned null for completions_default (no usage.prompt_tokens)**
- **B1: BOS probe returned null for completions_default (no usage.prompt_tokens)**
- **B2: BOS probe returned null for completions_default (no usage.prompt_tokens)**
- **A2: BOS probe returned null for completions_default (no usage.prompt_tokens)**
- **B0: BOS probe returned null for completions_add_bos_true (no usage.prompt_tokens)**
- **A1: BOS probe returned null for completions_add_bos_true (no usage.prompt_tokens)**
- **B1: BOS probe returned null for completions_add_bos_true (no usage.prompt_tokens)**
- **B2: BOS probe returned null for completions_add_bos_true (no usage.prompt_tokens)**
- **A2: BOS probe returned null for completions_add_bos_true (no usage.prompt_tokens)**
- **B0: BOS probe returned null for completions_add_bos_false (no usage.prompt_tokens)**
- **A1: BOS probe returned null for completions_add_bos_false (no usage.prompt_tokens)**
- **B1: BOS probe returned null for completions_add_bos_false (no usage.prompt_tokens)**
- **B2: BOS probe returned null for completions_add_bos_false (no usage.prompt_tokens)**
- **A2: BOS probe returned null for completions_add_bos_false (no usage.prompt_tokens)**
- **B0: BOS probe returned null for chat_no_think (no usage.prompt_tokens)**
- **A1: BOS probe returned null for chat_no_think (no usage.prompt_tokens)**
- **B1: BOS probe returned null for chat_no_think (no usage.prompt_tokens)**
- **B2: BOS probe returned null for chat_no_think (no usage.prompt_tokens)**
- **A2: BOS probe returned null for chat_no_think (no usage.prompt_tokens)**

## A vs B

| Metric | A1 | B1 | B2 | A2 | A mean | B mean | B vs A | A1→A2 | B1→B2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| code c1 per-stream tok/s | 249.34 | 247.31 | 248.27 | 247.78 | 248.56 | 247.79 | -0.3 % | -0.6 % | +0.4 % |
| code c4 per-stream tok/s | 147.46 | 148.11 | 146.06 | 148.14 | 147.80 | 147.08 | -0.5 % | +0.5 % | -1.4 % |
| code c4 aggregate tok/s | 589.83 | 592.44 | 584.23 | 592.56 | 591.19 | 588.34 | -0.5 % | +0.5 % | -1.4 % |
| prose c1 per-stream tok/s (thinking on) | 249.31 | 249.86 | 248.23 | 249.84 | 249.57 | 249.05 | -0.2 % | +0.2 % | -0.7 % |
| code c1 TTFT s | 0.16 | 0.16 | 0.16 | 0.16 | 0.16 | 0.16 | +0.5 % | -0.3 % | +1.2 % |
| six-agent replay span s | 46.47 | 44.98 | 45.86 | 45.64 | 46.05 | 45.42 | -1.4 % | -1.8 % | +1.9 % |
| six-agent replay span per generated token ms | 4.95 | 4.79 | 4.89 | 4.86 | 4.91 | 4.84 | -1.4 % | -1.8 % | +1.9 % |
| replay mean TTFT s | 4.71 | 4.69 | 4.76 | 4.69 | 4.70 | 4.73 | +0.6 % | -0.4 % | +1.5 % |
| replay SSE stall seconds (gaps >= 0.25 s) | 43.18 | 43.68 | 43.33 | 45.41 | 44.29 | 43.51 | -1.8 % | +5.2 % | -0.8 % |
| replay mean decode tok/s | 126.66 | 127.94 | 134.97 | 128.94 | 127.80 | 131.46 | +2.9 % | +1.8 % | +5.5 % |
| boot seconds | 20.33 | 20.56 | 20.45 | 20.26 | 20.30 | 20.51 | +1.0 % | -0.4 % | -0.6 % |

## Format checks

| Request | B0 | A1 | B1 | B2 | A2 |
|---|---:|---:|---:|---:|---:|
| prompt tokens: completions_default | None | None | None | None | None |
| prompt tokens: completions_add_bos_true | None | None | None | None | None |
| prompt tokens: completions_add_bos_false | None | None | None | None | None |
| prompt tokens: chat_no_think | None | None | None | None | None |
| prompt tokens: encode_add_bos_true | 18 | 18 | 18 | 18 | 18 |
| prompt tokens: encode_add_bos_false | 17 | 17 | 17 | 17 | 17 |
| reasoning chars / tags in content / finish | 182 / [] / stop | 182 / [] / stop | 182 / [] / stop | 182 / [] / stop | 182 / [] / stop |

## Cells

| Cell | image | boot s | warmup s | FASTWARM | VRAM used after boot MiB | after replay MiB | code c4 rounds | R828 code c1 ms/step | tok/step |
|---|---|---:|---:|---|---|---|---|---:|---:|
| B0 | sha256:da386167cc76 | 20.4 | 0.389210 | FASTWARM ok (N_p 20 #4, N_p 29 #5, N_p 1 | 30938/31336 | - | - | - | - |
| A1 | sha256:2239954c3723 | 20.3 | 0.387391 | FASTWARM ok (N_p 20 #4, N_p 29 #5, N_p 1 | 30938/31336 | 31530/31970 | [593.33, 586.33] | 10.72 | 2.61 |
| B1 | sha256:da386167cc76 | 20.6 | 0.421589 | FASTWARM ok (N_p 20 #4, N_p 29 #5, N_p 1 | 30938/31336 | 31530/31970 | [589.06, 595.83] | 10.69 | 2.61 |
| B2 | sha256:da386167cc76 | 20.4 | 0.355523 | FASTWARM ok (N_p 20 #4, N_p 29 #5, N_p 1 | 30938/31336 | 31530/31970 | [579.14, 589.32] | 10.66 | 2.61 |
| A2 | sha256:2239954c3723 | 20.3 | 0.356282 | FASTWARM ok (N_p 20 #4, N_p 29 #5, N_p 1 | 30938/31336 | 31530/31972 | [601.6, 583.52] | 10.69 | 2.61 |

R828_COUNTER rows are diagnostic only (the candidate may not log them); A1: ok; B1: ok; B2: ok; A2: ok.

## Greedy identity (24 prompts x 512 tokens, temperature 0, /v1/completions)

- B1 vs A1: 24/24 identical, 0 forks; margins measured 0, **margin errors 0**, **no logprobs 0**, length/tokenization forks 0
- B2 vs A1: 24/24 identical, 0 forks; margins measured 0, **margin errors 0**, **no logprobs 0**, length/tokenization forks 0
- A2 vs A1 (A/A control): 24/24 identical, 0 forks; margins measured 0, **margin errors 0**, **no logprobs 0**, length/tokenization forks 0
- B2 vs B1 (B/B control): 24/24 identical, 0 forks; margins measured 0, **margin errors 0**, **no logprobs 0**, length/tokenization forks 0
- B1 vs A1, margin in served engine: 24/24 identical, 0 forks; margins measured 0, **margin errors 0**, **no logprobs 0**, length/tokenization forks 0

Per prompt: `=` identical; otherwise first divergent token k, the live engine's top-1 vs top-2 logprob margin at k and |logprob(own) - logprob(ref)| at k (prefill-path probe, nats). The served-engine column scores B1's fork in A2's engine (same configuration as A1).

| pid | kind | B1 vs A1 | B2 vs A1 | A2 vs A1 (A/A control) | B2 vs B1 (B/B control) | B1 vs A1, margin in served engine |
|---|---|---|---|---|---|---|
| 0 | code | = | = | = | = | = |
| 1 | code | = | = | = | = | = |
| 2 | code | = | = | = | = | = |
| 3 | code | = | = | = | = | = |
| 4 | code | = | = | = | = | = |
| 5 | code | = | = | = | = | = |
| 6 | prose | = | = | = | = | = |
| 7 | prose | = | = | = | = | = |
| 8 | prose | = | = | = | = | = |
| 9 | prose | = | = | = | = | = |
| 10 | prose | = | = | = | = | = |
| 11 | prose | = | = | = | = | = |
| 12 | prose | = | = | = | = | = |
| 13 | agentic | = | = | = | = | = |
| 14 | agentic | = | = | = | = | = |
| 15 | agentic | = | = | = | = | = |
| 16 | agentic | = | = | = | = | = |
| 17 | agentic | = | = | = | = | = |
| 18 | french | = | = | = | = | = |
| 19 | french | = | = | = | = | = |
| 20 | french | = | = | = | = | = |
| 21 | chinese | = | = | = | = | = |
| 22 | chinese | = | = | = | = | = |
| 23 | chinese | = | = | = | = | = |

Raw: per cell `boot.log`, `engine.log`, `flashnext-18022.log*`, `decode-*.jsonl`, `agents.jsonl`, `greedy.jsonl` (text, server usage, re-encoded ids), `greedy-*.json`, `vram-*.csv`, `container-*`; root `plan.json`, `launch-{A,B}.sh` + `.diff`, `B.env`, `inputs.sha256`, `restore.log`, `status.json`.

