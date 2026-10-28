# Flash-Next pf2lock screen (2026-10-28-153418-flashnext-pf2lock-screen-screen-4071598)

Launchers: deterministic renders of the sealed daily launcher (A 3 hunks, B 4 hunks; `launch-A.diff`, `launch-B.diff`), container `flashnext-pf2lock-exp-2026-10-28-153418-flashnext-pf2lock-screen-screen-4071598`. A/B config check: PASS (B-only keys ).

A = served `tabbyapi:tune268-r2` (sha256:2239954c3723…), B = candidate sha256:da386167cc76…; pool 819200, split [30, 30], 8 slots, MTP d3, :18022, FASTWARM=1, no NVMe tier. B delta vs served flags: `{"added_extra_env": []}`.

**Verdict (predeclared): parity or better.** code c1 -0.1 %, code c4 aggregate +0.2 %, replay span -0.2 % (rule: c1 >= -2.0 %, c4 >= -2.0 %, replay <= +3.0 %; mean B1,B2 vs mean A1,A2). Greedy identity is reported below, not gated. No promotion.

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
| code c1 per-stream tok/s | 249.15 | 248.33 | 248.16 | 247.71 | 248.43 | 248.25 | -0.1 % | -0.6 % | -0.1 % |
| code c4 per-stream tok/s | 145.83 | 147.82 | 146.98 | 148.28 | 147.05 | 147.40 | +0.2 % | +1.7 % | -0.6 % |
| code c4 aggregate tok/s | 583.31 | 591.28 | 587.94 | 593.12 | 588.21 | 589.61 | +0.2 % | +1.7 % | -0.6 % |
| prose c1 per-stream tok/s (thinking on) | 250.53 | 249.13 | 249.56 | 250.41 | 250.47 | 249.35 | -0.4 % | -0.0 % | +0.2 % |
| code c1 TTFT s | 0.16 | 0.16 | 0.16 | 0.16 | 0.16 | 0.16 | +0.0 % | +0.9 % | -0.3 % |
| six-agent replay span s | 46.79 | 46.62 | 46.97 | 47.00 | 46.89 | 46.80 | -0.2 % | +0.4 % | +0.8 % |
| six-agent replay span per generated token ms | 4.99 | 4.97 | 5.00 | 5.01 | 5.00 | 4.99 | -0.2 % | +0.4 % | +0.8 % |
| replay mean TTFT s | 4.80 | 4.77 | 4.75 | 4.79 | 4.80 | 4.76 | -0.7 % | -0.2 % | -0.5 % |
| replay SSE stall seconds (gaps >= 0.25 s) | 43.69 | 43.58 | 43.12 | 44.09 | 43.89 | 43.35 | -1.2 % | +0.9 % | -1.1 % |
| replay mean decode tok/s | 131.95 | 127.93 | 127.91 | 125.70 | 128.82 | 127.92 | -0.7 % | -4.7 % | -0.0 % |
| boot seconds | 22.57 | 20.43 | 20.25 | 20.33 | 21.45 | 20.34 | -5.2 % | -9.9 % | -0.9 % |

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
| B0 | sha256:da386167cc76 | 20.4 | 0.365009 | FASTWARM ok (N_p 20 #4, N_p 29 #5, N_p 1 | 30938/31336 | - | - | - | - |
| A1 | sha256:2239954c3723 | 22.6 | 0.454825 | FASTWARM ok (N_p 20 #4, N_p 29 #5, N_p 1 | 30938/31336 | 31530/31968 | [592.54, 574.08] | 10.70 | 2.61 |
| B1 | sha256:da386167cc76 | 20.4 | 0.363051 | FASTWARM ok (N_p 20 #4, N_p 29 #5, N_p 1 | 30938/31336 | 31532/31970 | [606.79, 575.78] | 10.72 | 2.61 |
| B2 | sha256:da386167cc76 | 20.3 | 0.393170 | FASTWARM ok (N_p 20 #4, N_p 29 #5, N_p 1 | 30938/31336 | 31530/31970 | [586.27, 589.61] | 10.75 | 2.61 |
| A2 | sha256:2239954c3723 | 20.3 | 0.368639 | FASTWARM ok (N_p 20 #4, N_p 29 #5, N_p 1 | 30938/31336 | 31530/31970 | [594.39, 591.84] | 10.70 | 2.61 |

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

