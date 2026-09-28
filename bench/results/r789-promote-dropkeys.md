# R789: three unused engine keys dropped (42 → 39), greedy output byte-identical, promoted on the second run; the first run failed the loop-think gate because its probe ran at the new sampler fallback of 1.0, not because of the key drop

Results `2026-09-28-r789-promote-dropkeys-0004` (first run, 00:04 to 00:10 UTC), `2026-09-28-r789-promote-dropkeys-0013` (second run, 00:13 to 00:53 UTC) and `2026-09-28-loopthink-temp-0011` (the temperature follow-up, 00:11 UTC) on the serving host. Driver: [`scripts/r789-promote-dropkeys.sh`](../../scripts/r789-promote-dropkeys.sh). Launcher: [`scripts/launchers/launch-flashnext-r789-dropkeys.sh`](../../scripts/launchers/launch-flashnext-r789-dropkeys.sh). Probes: [`bench/chat_greedy.py`](../chat_greedy.py), [`bench/replay_hermes_turn.py`](../replay_hermes_turn.py) (the run used a later revision with `--temperature`), [`bench/probe.py`](../probe.py), [`bench/needle.py`](../needle.py), [`bench/agent_replay.py`](../agent_replay.py), [`bench/agentic-edit.py`](../agentic-edit.py), [`bench/tooleval_summary.py`](../tooleval_summary.py); `fn_greedy.py`, `loop_prefixes.py` and `replay_steps.py` are not in this repository. Raw records: on the serving host only; this write-up quotes the audit log and the per-gate summaries. The G5 `p107` and `p700` records come from a private agent session and are not published.

## The change

- `EXL3_HOST_GAP_REWIND=1`, `EXL3_GR_STATE_REGRID=1` and `EXL3_HC_MIX_V3_PDL=0` removed from `EXTRA_ENV`: 42 → 39 keys. Image `tabbyapi:rebase-dev-r3` (`30ed33eb`), pool 901,120, split `[30, 30]` unchanged.
- On this image the flag-off GDN path builds the same rewind descriptors as `HOST_GAP_REWIND=1` (R788a, greedy identical); `GR_STATE_REGRID` is unreachable with the mixer flags the launcher sets; `HC_MIX_V3_PDL=0` is the default. [`docs/CONFIG.md`](../../docs/CONFIG.md) has the detail.
- The gates are R785's G1 to G10, with greedy output required byte-identical to R785c's reference.

## Two runs

| run (UTC) | results | outcome | cause |
| --- | --- | --- | --- |
| 00:04 to 00:10 | `-0004` | rolled back at G5 | 1 of 2 forced-loop rows carried the looped line into the answer (`finish_reason` stop, 2,422 tokens), and the engine logged 3 loop stops against 2 expected. The G5 probe sent no temperature, so it ran at the sampler fallback, 1.0 since 2026-09-27 20:18 UTC, while R785c's reference ran at the 0.6 fallback |
| 00:13 to 00:53 | `-0013` | promoted | G5's probes pinned at temperature 0.6, R785c's condition |

- The fallback change of 2026-09-27 touched `temperature` only; `top_k` 20 and `top_p` 0.95 are unchanged and the preset holds no other key, so an explicit 0.6 reproduces R785c's sampler.
- The pin should have been in the driver before the first run; it went in after the failure, 7 s before the second run started, without a review in between. The review of the round found that it restores the pre-registered condition and does not relax the gate. G5 was the only gate that sent no temperature: G7 sends 0.6, G8 0 or 0.6, G9 0.6 with top_p 0.95 and top_k 20, G10 0.
- The forced-loop probe is a mechanism check: 40 copies of one line prefilled in the thinking and a question ("17*23"). Every row of it has the same reasoning (80 copies, then the injected message), so each row is one sampling decision after the forced `</think>`.

## Attribution of the first run's failure to the sampler

The 42-key launcher that was serving, same probe, n 16 per arm, alternating batches of 8 (results `2026-09-28-loopthink-temp-0011`):

| arm | clean "391" | looped line in the answer |
| --- | --- | --- |
| no temperature sent (fallback 1.0) | 9 / 16 | 7 / 16 (batches 3 / 8 and 4 / 8) |
| explicit 0.6 | 16 / 16 | 0 / 16 |

- Fisher two-sided p ≈ 0.007. At 7 of 16, a 2-row gate fails the unchanged configuration 68 % of the time (1 − (9/16)²).
- The first run's failing row has the shape of every looped row in the table: 7,042 characters of content, 2,422 tokens, `finish_reason` stop. Identical token counts across rows are detector thresholds (the reasoning detector at 800 tokens plus the injection; the content backstop about 1,600 tokens later), not copied rows.
- This arm measured the 42-key launcher and could not separate "no temperature sent" from "temperature 1.0". [R790](r790-loopthink-temp.md) closes that on the 39-key configuration.

## Gates, second run

| gate | rule | result |
| --- | --- | --- |
| G1 boot | image, 39 keys verbatim, the three keys absent, pool 901,120, split `[30, 30]`, free VRAM at boot per card at least R785c's minus 32 MiB (1,149 / 1,727) | 1,181 / 1,759 MiB (= R785c) |
| G2 greedy | `fn_greedy` 6 records and `chat_greedy` 6 rows byte-identical to R785c, including the 5,001-token row that crosses the 2,048-token requeue | 6 / 6 and 6 / 6 identical |
| G3 ramp and stress | 1 to 8 streams 36/36 ok; 4 and 8 streams at the 26k setting all ok, 0 OOM | 36/36, 4/4, 8/8 |
| cold prefill (reported) | one salted 90k-token prompt | 90,117 tokens in 7.23 s, 12,464 t/s |
| G5 loop-think, at 0.6 | forced loop 2 × "391" with the injection; thinking-off loop 2 × stop at 800 tokens; `p107` agent turn 2 × tool call or content | pass; `p107` 2 injected; the `p700` long-period case not exercised (8 of 8 rows left the prefilled block), as in R785b and R785c |
| G6 needles | 5/5 at 131,072 and 240,000 prompt tokens | 5/5, 5/5 |
| G4 headroom | 0 OOM lines, 0 restarts; free VRAM after reported | 367 / 903 MiB |
| G7 agent replay | per-stream decode at least 120.3 t/s | 132.3 t/s, 3.269 tokens per verify step, 24.70 ms per verify step, 411 requests, 0 session instances excluded |
| G8 agentic edit | 4 modes × 6/6 | 4 × 6/6 |
| G9 tool-eval | 69 scenarios × 4, mean at least 82.0 | 85.2 (95 % interval 83.8 to 86.5) |
| G10 GSM8K | n = 500, 8 concurrent, flexible-extract at least 0.970 | 0.976 |

- G7's 132.3 against R785c's 123.7 is not a speedup: tokens per step 3.269 against 3.117 is MTP acceptance under sampling (R785c had one session instance at acceptance 0.37, this run none below 0.5), ms per step 24.70 against 25.20 sits inside the same image's spread, 132.3 sits inside [R786](r786-replay-abba.md)'s arms (127.9 to 140.6), and the request sets differ (411 against 402).
- G10 has one empty answer (doc 119) that R785c did not have: the request reached the 8,192-token limit at 77 % draft acceptance, not a loop signature. Greedy decoding at 8 concurrent requests is not batch-invariant (R785b 0.982, R785c 0.980 on one configuration).
- The NVMe prefix tier restarted cold on each launcher change, as the driver reports.

## What this does not show

- Any speed change: not from G7, the prefill sample or tokens per step.
- That the loop-think gate passed at the served fallback: it passed at 0.6. At 1.0 the 42-key configuration failed 7 of 16 forced rows; the 39-key configuration's behaviour at 1.0 rests on greedy identity and 2 rows here, and is measured in [R790](r790-loopthink-temp.md).
- That 0.6 is free of the failure: 0 of 22 forced rows at 0.6 so far bounds the rate below about 15 %.
- Loop incidence in real traffic: the forced probe forces entry into a loop and measures one decision after the injection. [R791](r791-temp-incidence.md) looks at ordinary turns.

The promotion row is in [`docs/HISTORY.md`](../../docs/HISTORY.md); the probe-without-temperature trap is [GOTCHAS 32](../../docs/GOTCHAS.md).
