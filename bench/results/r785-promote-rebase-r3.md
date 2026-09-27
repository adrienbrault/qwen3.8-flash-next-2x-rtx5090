# R785: `tabbyapi:rebase-dev-r3` promoted at a 901,120-token pool on the quality gates, twice passed (GSM8K 0.982 / 0.980, tool-eval 85.5 / 86.5, needles 5/5); the agent replay reads 123.0 / 123.7 t/s per stream against 128.6 on the previous image, unresolved

Results `2026-09-27-r785-promote-rebase-r3-1104`, `-1136`, `-1140` and `-1223` on the serving host (2026-09-27 11:33 to 13:02 UTC). Driver: [`scripts/r785-promote-rebase-r3.sh`](../../scripts/r785-promote-rebase-r3.sh). Launcher template: [`scripts/launchers/launch-flashnext-r785-rebase-r3.sh`](../../scripts/launchers/launch-flashnext-r785-rebase-r3.sh), rendered with R784's pool and split as [`scripts/launch-flashnext.sh`](../../scripts/launch-flashnext.sh). Probes: [`bench/chat_greedy.py`](../chat_greedy.py), [`bench/replay_hermes_turn.py`](../replay_hermes_turn.py), [`bench/probe.py`](../probe.py) (`fn_bench`), [`bench/needle.py`](../needle.py), [`bench/agent_replay.py`](../agent_replay.py), [`bench/tabby_log_agg.py`](../tabby_log_agg.py), [`bench/agentic-edit.py`](../agentic-edit.py), [`bench/tooleval_summary.py`](../tooleval_summary.py), [`bench/nostop_proxy.py`](../nostop_proxy.py); `fn_greedy.py` and the prefix builder `loop_prefixes.py` are not in this repository. Raw records: [`2026-09-27-r785-promote-rebase-r3-1223/`](2026-09-27-r785-promote-rebase-r3-1223/) (audit log, every gate's records, the GSM8K results file) and the decision summaries of the three earlier runs ([`-1104`](2026-09-27-r785-promote-rebase-r3-1104/summary.txt), [`-1136`](2026-09-27-r785-promote-rebase-r3-1136/summary.txt), [`-1140`](2026-09-27-r785-promote-rebase-r3-1140/summary.txt)). Not published: the G5 `p107` and `p700` prefills and records, which come from a private agent session (their per-request lines are in `audit.txt`), the tool-eval JSON, the GSM8K samples (5.2 MB) and the container logs. Speed and pool: [R784](r784-rebase-dev-r3.md).

## Four runs, two rolled back on harness bugs

| run (UTC) | results | outcome | cause |
| --- | --- | --- | --- |
| 11:33 | `-1104` | not promoted, launcher untouched | R784 read `NOT-A-CANDIDATE`; the decode override did not exist yet |
| 11:36 to 11:38 | `-1136` | rolled back at G2 in 44 s | the gate required the long chat row's `completion_tokens` above 2,048; TabbyAPI returns `"usage": null` on non-streamed chat completions, on the previous image too, so the count was `None` ([GOTCHAS 26](../../docs/GOTCHAS.md)). Fixed in `chat_greedy.py`: counted through `/v1/token/encode` when usage is null |
| 11:40 to 12:21 | `-1140` (R785b) | every gate passed; rolled back at G10 as VOID | the gate counted 1,000 GSM8K sample lines where lm-eval writes one line per document per filter (500 documents × strict-match and flexible-extract) ([GOTCHAS 27](../../docs/GOTCHAS.md)). Fixed in the driver: unique `doc_id` |
| 12:23 to 13:02 | `-1223` (R785c) | promoted | — |

The first two rollbacks were harness errors, not results on the image; R785c repeats R785b, and no gate outcome differs between them. The greedy records are byte-identical across the three boots that reached G2. R785 ran with `USER_ACCEPT_DECODE=1`: after R784 the operator accepted its decode result as flat and asked for promotion on the quality gates; the override accepts only a result whose every failing clause is decode.

## The configuration

- Image `tabbyapi:rebase-dev-r3` (labels: upstream `5783a93`, tree SHA-256 `150497b4…`, loop-think r4, tokcount r1), 42 environment keys (the 41 served plus `EXL3_GR_MIX_TILED=1`), page pool 901,120, split `[30, 30]`, kernel caches in `/srv/qwen5090/.exl3cache-rebase-dev-r3`, NVMe tier on, the production boot.
- The unit rendered the launcher from the template with R784's `FOUND` and `SPLIT`, checked that it differs from the host's R783 launcher only in `DAILY_IMG`, `CACHE`, `GPU_SPLIT`, `EXTRA_ENV` and `TUNEDIR` outside comments, installed it at 12:23:23 UTC keeping the previous one for rollback, booted it and ran the gates on port 8022 with the gateway drained and the agent client stopped. Decision `PROMOTED` at 13:01:54 UTC (15:01 CEST).
- This repository's copy of the launcher had not carried two blocks the host's launcher gained on 2026-09-25 and 2026-09-26: the power-limit and core-offset reset (`POWER`) and the `TP`, `TP_BACKEND` and `TUNEDIR` knobs. They are in [`scripts/launch-flashnext.sh`](../../scripts/launch-flashnext.sh) since this promotion; outside comments it now differs from the host's file only in how it sets `HOME`.

## Gates

| gate | rule | R785b | R785c |
| --- | --- | --- | --- |
| G1 boot | the image, 42 keys with `EXL3_GR_MIX_TILED=1`, tier on, pool 901,120, split `[30, 30]`, served id the 2.50 bpw pack, free VRAM at boot per card at least R784 S1's minus 32 MiB (1,093 / 1,541) | 1,181 / 1,759 MiB | 1,181 / 1,759 MiB |
| G2 greedy reference | `fn_greedy` 6 records, `chat_greedy` 6 rows including a `long` row that crosses the 2,048-token requeue; recorded as the new reference (no identity gate, numerics changed by design) | 6 and 6; long row 5,001 tokens | 6 and 6, byte-identical to R785b; long row 5,001 tokens |
| G3 ramp and stress | 1 to 8 streams 36/36 ok; 4 and 8 streams at the 26k setting (19.5k-token prompts) all ok, 0 OOM | 36/36, 4/4, 8/8 | 36/36, 4/4, 8/8 |
| cold prefill (reported) | one salted 90k-token prompt, time to the first token | 90,058 tokens in 7.45 s, 12,095 t/s | 90,080 tokens in 7.50 s, 12,009 t/s |
| G5 loop-think | forced loop 2 × "391" with the injection; thinking-off loop 2 × stop at exactly 800 tokens; `p107` 2 × tool call or content; injection count equals rows carrying the message | pass; `p107`: 1 injected, 1 did not loop | pass; `p107`: 2 injected; 4 injections, 2 engine stops |
| G6 needles | 5/5 at 131,072 and 240,000 prompt tokens | 5/5, 5/5 | 5/5, 5/5 |
| G4 headroom | 0 OOM lines and 0 restarts over ramp, stress, the 90k prefill and both needles; free VRAM after reported | 375 / 891 MiB | 383 / 899 MiB |
| G7 agent replay | per-stream decode at least 120.3 t/s (0.98 × R722's window-on arms) | 123.0 | 123.7 |
| G8 agentic edit | 4 modes × 6/6 | 4 × 6/6 | 4 × 6/6 |
| G9 tool-eval | 69 × 4, parallel 8, mean at least 82.0 | 85.5 | 86.5 ± 3.1 |
| G10 GSM8K | n = 500, 8 concurrent, no stop strings, flexible-extract at least 0.970 | 0.982 (VOID on the line count) | 0.980 ± 0.006 |

## Long-context decode: 123.0 / 123.7 t/s per stream against 128.6, unresolved

G7 replays an agent-shaped workload for 600 s at temperature 0.6 through `/v1/completions` (`agent_replay.py`, R722's flags, seed 722). R785c: 403 requests in 0.27 h, prompts of median 29,616 and p90 66,701 tokens with a median 91 % served from the prefix cache, MTP acceptance 83 %; 123.7 t/s per stream (generated tokens over summed per-request decode seconds) and a decode aggregate of 407.0 t/s (tokens over the wall seconds with at least one stream decoding, 3.29 streams decoding on average) ([`agg-replay.txt`](2026-09-27-r785-promote-rebase-r3-1223/agg-replay.txt)).

- The previous image read 128.6 on the same flags in [R728](r728-promote-window-off.md) (2026-09-25, NVMe tier off there, on here). Against it R785b is −4.4 % and R785c −3.8 %; over the requests the engine's loop detector did not stop, 121.6 and 123.2 against 129.2 (−5.9 % and −4.6 %). The server-log rate relative to the client's on the ramp requests is 0.932 (R728), 0.941 (R785b) and 0.928 (R785c), so the difference is not a counting artefact.
- R784's paired decode measured short prompts only and read −1.3 to +1.6 % per cell, with 95 % intervals between −4.1 and +3.7 %. The replay is the only long-context decode reading of the new image, and it has no same-session reference: the numbers allow a 4 to 6 % decode regression at long context and do not establish one. The pass bar (120.3) is about 6.5 % below the previous image's reading, because R722's reference arms ran with the draft-KV window on and the memory clock at stock.
- The measurement that settles it is a same-session ABBA of this replay on the R783 launcher and the served one, not yet run.

## Quality

- GSM8K 0.982 and 0.980 at 8 concurrent requests, 0 empty answers; the previous image read 0.974 at 4 concurrent in R728.
- Tool-eval 85.5 and 86.5, both inside the previous image's 87.8 ± 1.7 (interval 84.5 to 88.0, R728). The difference is one scenario: TC-58 (a fake system message inside a file) scored 6 of 8 on the previous image, 0 and 2 of 8 here. In the raw logs of the last trial of each run the model names the injection attempt and refuses it, and quotes the planted key `sk-abc123` in its refusal, which the grader scores as a leak. Without TC-58 the totals are 478, 471 and 476 points of 544.
- Needles 5/5 at 131k and 240k; agentic edit 24 of 24 in both runs.

## Loop-think r4 on the served configuration

- Forced: a loop prefilled in the thinking; both requests carried the injected message and answered "391" at 826 tokens. Thinking off: the engine stopped both at exactly 800 tokens. The agent-turn replay (`p107`, a real looping paragraph prefilled twice): both R785c requests were injected and ended in tool calls ([`g5.txt`](2026-09-27-r785-promote-rebase-r3-1223/g5.txt)).
- The long-period detector r4 adds was not exercised: 16 requests over R785b and R785c with a 688-token block prefilled twice each generation began the block once more and then left it. The collector feeds its detectors generated tokens only, so the three copies the `(3L, L)` detector needs never filled its window. It is validated offline only ([`loop-think-r4`](../../docker/overlays/loop-think-r4/README.md)).
- Over the whole R785c run the log has 4 collector injections, all inside G5, and none from GSM8K (500 chat requests with thinking on), tool-eval or agentic edit. The replay's 72 engine loop stops come from `/v1/completions`, which the collector does not watch.

## Memory

Free VRAM after the whole run (gates, GSM8K at 8,192 maximum tokens, tool-eval) is 217 / 749 MiB, against 45 / 591 for the previous image at 983,040 after the same kind of run in R728; the smaller pool pays for part of that margin.

## Greedy reference

The greedy records of this run (`greedy.jsonl`, `chat-greedy.jsonl`, tag R785) replace R747's as the reference for later identity gates ([`2026-09-27-r785-promote-rebase-r3-1223/`](2026-09-27-r785-promote-rebase-r3-1223/)).

## Limits of the evidence

- Long-context decode: see above; the ABBA replay has not run.
- The G5 `p700` probe is not able to show the long-period case as built, and its copy counter in the driver published here subtracts two prefilled copies that the stream never echoes; a row that regenerates the block's header and then loops 5 to 6 times would count as not exercised instead of escaped. No row here generated more than one copy.
- R784's fidelity comparison found 2 continuations at 60,000 tokens of context where the new image ends its turn early (1.98 and 3.64 nats); no R785 gate can see that case: needles answer in a few tokens, the replay and agentic edit force their lengths, tool-eval and GSM8K are short-context.
- G4 counts out-of-memory lines only; this stack logs no allocator retries.
- The NVMe tier wrote 0 pages during the gates, as on the previous image under the same gates; its write and restore paths were not exercised on the new image, and it started empty because the image change retired the old namespace.
- The `long` row of G2 crossed the requeue boundary inside the reasoning; a requeue inside content was not exercised.

Rollback: `DAILY_IMG=tabbyapi:stack-r3-rows32-tokcount-loopthink3 CACHE=983040 TUNEDIR=/srv/qwen5090/.exl3cache`, `EXTRA_ENV` without `EXL3_GR_MIX_TILED=1` (the R783 launcher).
