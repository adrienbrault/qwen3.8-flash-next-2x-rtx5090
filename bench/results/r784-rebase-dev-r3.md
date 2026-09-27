# R784: the served stack on upstream ExLlamaV3 `dev` 5783a93 (v1.5.2) fits a 901,120-token pool (−8.3 %), prefills 1.134× faster at 90k tokens and decodes short prompts within the rule's resolution; not a candidate on the pre-registered decode rule, promoted on the quality gates in R785

Results `2026-09-27-r784-rebase-dev-r3-1104` on the serving host (2026-09-27 11:04 to 11:32 UTC). Driver: [`scripts/r784-rebase-dev-r3.sh`](../../scripts/r784-rebase-dev-r3.sh). Decision rule, pre-registered: [`docker/overlays/rebase-dev-r3/r784_decide.py`](../../docker/overlays/rebase-dev-r3/r784_decide.py). Image: [`docker/overlays/rebase-dev-r3`](../../docker/overlays/rebase-dev-r3/). Probes: [`bench/mp_decode.py`](../mp_decode.py), [`bench/probe.py`](../probe.py) (`fn_bench`), [`bench/chat_greedy.py`](../chat_greedy.py); `fn_greedy.py` and the two report-only fidelity tools (`lp_margin.py`, `exl3_fidelity_run.py` with `fidelity.py`) are not in this repository. Raw records: [`2026-09-27-r784-rebase-dev-r3-1104/`](2026-09-27-r784-rebase-dev-r3-1104/) (audit log, decision summary, boot table, greedy and chat greedy records, prefill and decode records, the paired comparisons, the fidelity summaries). The container logs, the lp_margin records (1.8 MB) and the corpus fidelity records (136 MB per arm) stay on the host.

## What was compared

- **S**, the served configuration of that morning: `tabbyapi:stack-r3-rows32-tokcount-loopthink3` ([R783](r783-loopthink.md)), ExLlamaV3 v1.5.0 plus this repository's chain, 41 environment keys, page pool 983,040.
- **P**, `tabbyapi:rebase-dev-r3` (image `sha256:30ed33eb…`): the served engine stack ported onto upstream `dev` `5783a93` (v1.5.2) with the requeue token-count fix, built on `tabbyapi:stack-r3-rows32-tokcount-loopthink4` (the served TabbyAPI plus [`loop-think-r4`](../../docker/overlays/loop-think-r4/)). The port replaces the whole `exllamav3` package and rebuilds its extension; tree SHA-256 `150497b4…` ([`impl-status.md`](../../docker/overlays/rebase-dev-r3/impl-status.md)). Environment: S's 41 keys plus `EXL3_GR_MIX_TILED=1`, upstream's tiled hyper-connection prefill mix, set explicitly so the launcher's key count records it.
- Upstream changes that alter numerics by design, all in upstream's code: GDN fp16 prefill projections, the deterministic router GEMM, the tiled HC prefill mix and the PLE in-place add ([`port-map-r2.md`](../../docker/overlays/rebase-dev-r3/port-map-r2.md)). The four commits from `73a6229` to `5783a93` (the sm_75 support of #325, the per-device shared-memory budget `f4db698`, v1.5.2, loader file streams) change nothing on sm_120 by code reading; the tree was not compiled before this unit.
- Order S1, P (pool search, then measured as P1), P2, S2. Every boot is a fresh container on port 8029 with the NVMe tier off, power limits at the stock 600 / 575 W, core clock offset 0 and memory offset +4500, read back per boot ([`boots.tsv`](2026-09-27-r784-rebase-dev-r3-1104/boots.tsv)). P used its own kernel-cache directory.
- Per measured boot: a warm-up pass; `fn_greedy` (6 prompts, one of about 100,000 tokens) and `chat_greedy` (5 chat prompts, thinking on), temperature 0; cold prefill, 3 salted unique prompts at each of two settings; `mp_decode`, 24 code and 24 prose prompts at 1, 4 and 8 streams, 512 forced tokens (`min_tokens`) per request, greedy, paired by prompt across arms (144 records per boot, 0 failed, 0 short).

## Page pool: 901,120 tokens

The rule: the largest pool on the 16,384-token grid whose boot leaves each card's free VRAM at least S1's minus 32 MiB, no out-of-memory line after a 12,000-token probe, floor 884,736. S1 booted with 1,125 / 1,573 MiB free.

| boot | pool | split | free at boot, MiB (cuda:0 / cuda:1) | result |
| --- | ---: | --- | --- | --- |
| P, first | 917,504 | 30, 30 | 679 / 1,039 | kernel cache cold (warm-up 47.4 s), not judged, re-booted |
| P, re-boot | 917,504 | 30, 30 | 1,061 / 1,619 | cuda:0 64 MiB below S1: no fit |
| P | 917,504 | 29.5, 30 | 1,785 / 873 | cuda:1 700 MiB below S1: no fit |
| P1 | 901,120 | 30, 30 | 1,181 / 1,759 | fits; layers 0 to 25 on cuda:0, 26 to 47 on cuda:1, as S |
| P2 | 901,120 | 30, 30 | 1,181 / 1,759 | same |

901,120 is 5 steps and 8.3 % below 983,040. Free VRAM after the 90k-token prefill: S1 311 / 847, S2 311 / 845, P1 273 / 629, P2 473 / 1,029 MiB. The port's memory cost was attributed on the previous port round of the same code (upstream `73a6229`, not published separately): the tiled HC prefill workspaces take 342 / 300 MiB per card, upstream's resident-state preallocation at load (`EXL3_AUTOSPLIT_PREPARE`) 86 / 62, and 118 / 110 MiB are unattributed ([`impl-status.md`](../../docker/overlays/rebase-dev-r3/impl-status.md)). With `EXL3_GR_MIX_TILED=0` the pool would be about one step below 983,040 and prefill unchanged; that arm ran only if the tiled search found nothing and did not run.

## Cold prefill: 1.134× at 90k tokens

Time to the first token of a 64-token request, counted by the server, 3 salted prompts per setting per boot; tokens per second is prompt tokens over time to the first token. The "30k" setting produced 22,550 to 22,615-token prompts, the "120k" setting 89,930 to 90,268.

| boot | ~22.6k tokens, median t/s | ~90k tokens, median t/s |
| --- | ---: | ---: |
| S1 | 10,162 | 10,793 |
| P1 | 10,957 | 12,186 |
| P2 | 11,733 | 12,210 |
| S2 | 9,870 | 10,667 |
| S (6 samples) / P (6 samples) | 9,966 / 11,548 = 1.159× | 10,756 / 12,198 = 1.134× |

At 90k tokens the six P samples (12,170 to 12,327 t/s) all lie above the six S samples (10,667 to 10,915). At 22.6k tokens the ratio is 1.078× for the P1 / S1 pair and 1.189× for P2 / S2; the first sample of each boot reads slowest. The gain is the tiled HC prefill mix, as measured upstream-side in [R568](r568-rebase-prefill.md) (+14.0 % at 60k, +11.4 % at 120k).

## Decode on short prompts: paired differences of −1.3 to +1.6 % at 1, 4 and 8 streams

The `mp_decode` prompts are one-line requests (a few dozen tokens), so this measures decode at short context. Per-request decode rate, tokens per second after the first token, averaged over the two boots of each arm per prompt; the ratio is the geometric mean over the 24 prompts of a kind of P / S, with a 95 % bootstrap interval over prompts. The last rows are the A/A control, S2 against S1.

| streams | kind | S, t/s | P, t/s | P / S | 95 % interval |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | code | 269.1 | 266.2 | −1.28 % | −4.11 to +1.67 % |
| 1 | prose | 251.8 | 254.1 | +0.76 % | −1.20 to +2.59 % |
| 4 | code | 152.3 | 154.6 | +1.58 % | −0.53 to +3.72 % |
| 4 | prose | 148.6 | 147.5 | −0.77 % | −1.99 to +0.50 % |
| 8 | code | 101.4 | 101.2 | −0.07 % | −1.80 to +1.54 % |
| 8 | prose | 98.4 | 97.8 | −0.44 % | −2.30 to +2.40 % |
| 4 | both, S2 / S1 (A/A) | 151.2 | 149.8 | −0.98 % | −2.05 to +0.24 % |
| 8 | both, S2 / S1 (A/A) | 100.4 | 99.3 | −1.02 % | −1.97 to −0.14 % |

- Prose at 8 streams contains one prompt (pid 23) at +31.2 % in both P boots, where MTP acceptance is 0.18 on S and 0.74 on P; without it the cell reads −1.63 % (−2.51 to −0.84 %). The other 23 prompts lie between −6.7 and +1.6 %.
- At long context, [R786](r786-replay-abba.md) ran the agent replay (prompts of median about 29,000 tokens, 91 % served from the prefix cache) on three ABBA pairs of fresh boots of S and P: P / S 1.015 per stream on the requests common to all arms (pairs 1.016, 0.981 and 1.048; 95 % interval 0.980 to 1.053). R785's single replay runs, 123.0 and 123.7 tokens/s against 128.6 for an older image in [R728](r728-promote-window-off.md), differ through MTP acceptance and concurrency, not the engine.
- P2 read 1.1 to 3.0 % above P1 in every cell, at identical acceptance at 1 stream (24 of 24 prompts). P1 was the search boot and the first process on the new kernel cache: it wrote the cooperative-kernel autotune file and compiled Triton kernels during its measurement, and its `fn_greedy` took 87 s against 30 to 32 s on the other boots.

## Greedy output

- `fn_greedy`: P1 and P2 each 4 of 6 identical to S1; the divergent prompts are `long100k` (S1 answers the ~100k-token prompt without thinking and stops; P opens `<think>` and runs to the length limit) and `short4` (same opening, later divergence). S2 against S1: 6 of 6.
- `chat_greedy`: P1 and P2 each 1 of 5 identical to S1 (the code prompt); arithmetic and prose differ in reasoning and content, the tool call and the two-turn prompt in reasoning. S2 against S1: 5 of 5.
- The fingerprints rolled over to R785's records ([R785](r785-promote-rebase-r3.md)); no identity gate applies to this change.

## Fidelity (report-only)

- Corpus, 200 chunks of 2,048 tokens (409,400 positions), P against S in the same container settings: mean NLL −0.04 % (agent −0.30 %, code −0.01 %, prose −0.00 %), top-1 agreement 0.968, KL(S ‖ P) 1.56e-2 ([`fid-compare.txt`](2026-09-27-r784-rebase-dev-r3-1104/fid-compare.txt)). No bf16 reference exists for this checkpoint, and this comparison has no null run.
- Top-5 margins on 40 corpus prompts at 0 and 60,000 tokens of context, 64 greedy tokens each: 29 of 80 continuations diverge; the tool flags 12 as not a near-tie. On reading the flips, 6 are at margins of 0.53 to 1.03 nats and 2 are early end-of-turn tokens at 60k (1.98 and 3.64 nats); the tool miscounts 2 near-flat third-token picks and 4 end-of-sequence divergences it does not see. On agreeing prefixes |Δ log p| is 0.0001 at the median and 0.197 at p99 (n = 3,729) ([`margins-compare.txt`](2026-09-27-r784-rebase-dev-r3-1104/margins-compare.txt)).

## Decision

The pre-registered rule required, at 4 and 8 streams for code and for prose, a geometric mean of at least −1 % and an interval lower bound of at least −2 %; cold prefill at least 0.95× at both settings; the pool at or above 884,736; 0 out-of-memory errors, tracebacks and restarts; and the A/A control within ±2 % at 4 streams. Every clause passed except prose at 8 streams, whose lower bound is −2.30 %: `NOT-A-CANDIDATE`.

A review of the run found the rule applied correctly and underpowered for this comparison. Replaying each cell's per-prompt spread with a candidate of equal speed, all four gated cells pass together 20 to 27 % of the time; a boot-to-boot drift of about 1 % fails every cell, and the A/A pair S2 against S1 read as candidate against reference fails all four. The prompt bootstrap does not carry boot-to-boot variance, and more boots on the same 24 prompts do not narrow the prose interval at 8 streams (projected ±2.61 to ±2.42 % from 1 to 8 ABBA blocks), because per-prompt acceptance differences between the two engines set its width. The operator accepted decode as flat and asked for promotion on the quality gates; [R785](r785-promote-rebase-r3.md) ran with that override, which accepts only a result whose every failing clause is decode.

## Limits of the evidence

- One ABBA block. P1's lower free VRAM after the 90k prefill (273 / 629 MiB against P2's 473 / 1,029) is attributed to the autotuner's buffers staying reserved on the tuning boot; not measured.
- The decode cells are paired by prompt, not by boot; see the review's reading above. They cover short prompts only; long cached context is R786's replay, which resolves about ±5 to 7 % per pair.
- The fidelity tools ran on the server for the first time and have no null distribution; their numbers are reported, not judged.
