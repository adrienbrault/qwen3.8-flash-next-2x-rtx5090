# R786: on the agent replay in one session, `rebase-dev-r3` reads 1.015× the previous image per stream on the common request set (95 % interval 0.980 to 1.053); R785's 123.0 / 123.7 against R728's 128.6 was acceptance luck, concurrency and an older baseline

Results `2026-09-27-r786-replay-abba-1324` on the serving host (2026-09-27 13:24 to 15:05 UTC). Driver: [`scripts/r786-replay-abba.sh`](../../scripts/r786-replay-abba.sh). Probes: [`bench/agent_replay.py`](../agent_replay.py), [`bench/tabby_log_agg.py`](../tabby_log_agg.py), [`bench/replay_agg_nonloop.py`](../replay_agg_nonloop.py) (the host's copies are identical to these). Raw records: [`2026-09-27-r786-replay-abba-1324/`](2026-09-27-r786-replay-abba-1324/) (audit log, decision summary, each arm's server-side scores `agg-*.txt` and `agg2-*.txt`, `EXL3_*` environment `env-*.txt`, and client records `replay-*.jsonl`). The container logs (230 to 250 kB per arm) and boot logs stay on the host. The session-level analysis below comes from an independent review that joined every server record to its client record (2,511 of 2,511); its scripts are not in this repository.

## Why

R785's agent replay read 123.0 and 123.7 tokens/s per stream on the new image against 128.6 for the previous image in [R728](r728-promote-window-off.md), two days earlier, with the NVMe tier on in R785 and off in R728. R784's paired decode covered short prompts only. R786 compares the two images on that replay in one session ([R785](r785-promote-rebase-r3.md)).

## What ran

- **O**, the previous served launcher: `tabbyapi:stack-r3-rows32-tokcount-loopthink3`, 41 environment keys, page pool 983,040, kernel cache `/srv/qwen5090/.exl3cache`. **N**, the served launcher: `tabbyapi:rebase-dev-r3`, 42 keys (O's plus `EXL3_GR_MIX_TILED=1`), pool 901,120, its own kernel cache. Both launchers unmodified (checked by md5), NVMe tier off in both, one fresh boot per arm, order O1 N1 N2 O2 O3 N3.
- Every arm: power limits 600 / 575 W, core clock offset 0, memory offset +4500; free VRAM at boot 1,125 / 1,573 MiB on O and 1,181 / 1,759 on N; 8 slots, split `[30, 30]`, draft policy `[[4, 3], [8, 2]]`. No out-of-memory line, traceback or failed request; every replay exited 0. Besides the readiness probe, every request in every arm's server log has the replay's shape.
- Traffic: R722's replay unchanged (`agent_replay.py --prod-ladder --respawn --drain`, generation about 700 tokens, tool results 2,020 tokens, 11 s think time, temperature 0.6, seed 722), 600 s per arm plus the drain of the sessions still running at the deadline. Prompts of median 28,572 to 30,594 tokens and p90 about 66,000, median 90 to 91 % served from the prefix cache.
- Per-stream decode is generated tokens over summed per-request decode seconds, from the server log. Pre-registered rule on the three pairs (O1, N1), (N2, O2), (O3, N3): SLOWER when the mean N/O is at most 0.97 and all three pairs are below 1; SLIGHTLY when the mean is below 0.99 and all three pairs are below 1; PARITY otherwise.

## Result

| arm | O1 | N1 | N2 | O2 | O3 | N3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| requests (replay) | 411 | 427 | 411 | 411 | 411 | 441 |
| per stream, as scored (t/s) | 133.7 | 137.6 | 127.9 | 130.4 | 130.7 | 140.6 |
| per stream, the 411 requests common to all arms (t/s) | 133.7 | 135.8 | 127.9 | 130.4 | 130.7 | 136.9 |
| non-loop-stopped requests, as scored (t/s) | 129.0 | 132.5 | 124.6 | 126.2 | 131.1 | 136.4 |
| MTP acceptance | 0.885 | 0.888 | 0.821 | 0.878 | 0.897 | 0.887 |
| tokens per verify step | 3.287 | 3.285 | 3.109 | 3.257 | 3.266 | 3.332 |
| ms per verify step | 24.57 | 23.87 | 24.30 | 24.97 | 25.00 | 23.70 |
| decode aggregate (t/s) | 424.6 | 435.6 | 412.4 | 433.1 | 430.9 | 430.7 |
| tokens in loop-stopped requests | 18.6 % | 17.9 % | 20.4 % | 13.8 % | 7.9 % | 13.2 % |

- On the 411 (session, instance, step) requests that every arm ran, N/O per pair is 1.016, 0.981 and 1.048, mean 1.015 (non-loop-stopped requests 1.007). A paired bootstrap over the 26 session instances common to all arms puts the mean at 0.980 to 1.053 (95 %). The run excludes a per-stream loss larger than about 2 % on this workload; it does not show the two images equal.
- As scored, the pairs are 1.029, 0.981 and 1.076, mean 1.029: `PARITY` on the pre-registered rule, and `PARITY` on the non-loop scores (1.018). The scored ratio favours the faster arm: a faster arm starts more session instances before the 600 s deadline, and those finish during the drain at low concurrency (N1 16 extra requests at 206 t/s per stream, N3 30 at 250). The common-set ratio is the reading this repository quotes.
- One pair resolves about ±5 to 7 % (single-pair bootstrap intervals 0.965 to 1.066, 0.913 to 1.047 and 1.002 to 1.090), the mean of three about ±3.5 %. Under the same bootstrap the pre-registered rule returns SLOWER for a true −4 % in 52 % of runs, for −5 % in 74 % and for −6 % in 90 %.
- Per verify step, which removes acceptance, N is 1.9 to 3.4 % faster in every pair on the common set. With per-request concurrency, prompt length and draft depth as covariates over 12 runs (these six, R785, R728 and R722's four), the N term on step time is +0.3 % ± 1.3 %.

## N2: two sessions at low acceptance

N2's step time is within N1's and N3's; its deficit is tokens per step (3.109 against 3.26 to 3.33). Two session instances fell into a low-acceptance trajectory: (8, 0) at 0.40 over 19,591 tokens and (9, 1) at 0.56 over 15,650. The same two instances ran at 0.87 to 0.96 in every other arm. Each reply becomes part of the next prompt, so an instance that enters such a mode stays in it for its remaining steps, and the independent unit of this replay is the session instance (26 to 27 per arm), not the request. Instances of this kind, 8,000 tokens or more at acceptance below 0.7, also occur on the previous image (R728 (4, 1) at 0.41, R722 N1 (4, 1) at 0.54) and in R785 ((7, 1) at 0.37): temperature-0.6 sampling, not the engine. Without N2's two instances in any arm, the pairs are 1.019, 1.020 and 1.035.

## R785 against R728, decomposed

| run | image | tokens per step | ms per step | per-request concurrency | per stream (t/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| R728 G7 | `stack-r3-rows32`, tier off | 3.192 | 24.86 | 4.54 | 128.4 |
| R785 G7 | `rebase-dev-r3`, tier on, after the other gates in the same container | 3.117 | 25.20 | 4.70 | 123.7 |
| R786 O, mean of 3 | `…-loopthink3`, tier off | 3.270 | 24.85 | 4.56 | 131.6 |
| R786 N, mean of 3 | `rebase-dev-r3`, tier off | 3.242 | 23.96 | 4.37 | 135.4 |

R728's 128.4 excludes two chat requests left over from its previous gate (128.6 with them). R785 / R728 = 0.963 = 0.977 in tokens per step × 0.987 in ms per step:

- −2.3 % is acceptance: 0.831 against 0.860, with R785's instance (7, 1) at 0.37 over 15,757 tokens.
- −1.3 % is step time at higher concurrency, 4.70 against 4.54 streams; with concurrency as a covariate the R785 run's term is +0.2 % ± 1.8 % against O.
- The baseline was a different image (`stack-r3-rows32`, before the token-count fix and loop-think) on another day; R786's O arms, the image served until the promotion, read 130.4 to 133.7.
- Not the NVMe tier: R785's tier wrote 0 pages and made 0 lookups during the replay. Not the pool: every run, R785 included, re-prefilled exactly one request beyond its expected increment (+2,049 tokens, the same request each time).

## What the replay cannot measure

- **The page pool.** Peak live KV (prompt plus generation over the live session instances) is 430,000 to 450,000 tokens in every arm, inside both pools. Each arm creates 1.28 to 1.38 million tokens of KV, so both pools evict, but only pages of finished sessions, which never return. Prefix-cache hits are the same in both families (median 90 to 91 %, mean 83.1 to 83.7 %). The 81,920 tokens the new pool gives up would cost in a different regime: idle agent sessions returning after other traffic has cycled the pool. R786 is no evidence that the smaller pool is free there.
- **Steady state.** The drain after the deadline holds 21 to 25 % of the scored tokens at 195 to 257 t/s per stream, against 116 to 126 before the deadline. Every replay per-stream figure, here and in R722, R728 and R785, therefore reads about 10 % above the replay's steady state ([GOTCHAS 29](../../docs/GOTCHAS.md)).
- **Production mix.** 8 to 20 % of the tokens come from requests the engine's loop detector stopped (forced lengths on synthetic filler); they run at higher acceptance and rate, and their share moves between boots (O3 7.9 %, O1 18.6 %).

Time to the first token: median 0.55 to 0.56 s on N against 0.60 s on O, and prefill time per new token 8 to 10 % lower on N, in the direction of R784's 1.13× cold prefill.

## Limits

- Three pairs. The bootstrap resamples session trajectories within each arm, not boot-to-boot variance; the three O arms on the common set (130.4 to 133.7) lie inside it.
- The unit logs image tags, not image IDs; both tags were created before the unit started.
- A single-run replay gate, such as R785's G7 at 0.98 × R722, fails when one low-acceptance session instance lands: 4 of the 12 runs looked at contain one ([GOTCHAS 30](../../docs/GOTCHAS.md)).
