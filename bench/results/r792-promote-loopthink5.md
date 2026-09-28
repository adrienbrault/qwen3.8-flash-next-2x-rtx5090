# R792: `tabbyapi:rebase-dev-r3-loopthink5` (TabbyAPI loop-think r5) promoted, every gate passed and greedy output byte-identical; the (6000, 2000) rung fired live on a prefilled 1,199-token loop at 6,003 generated tokens, and a tool call cut by `max_tokens` now finishes as `length`

Results `2026-09-28-r792-promote-loopthink5-0709` (2026-09-28 07:09 to 07:50 UTC) and the follow-up probe `2026-09-28-r792c-ladder-fallback-0801` on the serving host. Driver: [`scripts/r792-promote-loopthink5.sh`](../../scripts/r792-promote-loopthink5.sh), derived from [R789](r789-promote-dropkeys.md)'s with an in-image test gate (G5a) and four report-only probes. Launcher: [`scripts/launchers/launch-flashnext-r792-loopthink5.sh`](../../scripts/launchers/launch-flashnext-r792-loopthink5.sh) = [`scripts/launch-flashnext.sh`](../../scripts/launch-flashnext.sh), R789's launcher with only `DAILY_IMG` changed. Overlay: [`docker/overlays/loop-think-r5/`](../../docker/overlays/loop-think-r5/README.md). Raw records: on the serving host only; this write-up quotes the audit log, the server logs and the per-gate summaries. The G5 `p107` and `p700` records and probe (c)'s records hold the text of a private agent session and stay private.

## The change

- Image `tabbyapi:rebase-dev-r3-loopthink5` (`9f7b66f0`) = `tabbyapi:rebase-dev-r3` plus loop-think r5's `r4-to-r5.patch` on TabbyAPI's `endpoints/OAI/utils/chat_completion.py`. ExLlamaV3, the 39 engine keys, the 901,120-token pool, split `[30, 30]` and the kernel-cache directory are unchanged.
- r5 adds reasoning-loop rungs (6000, 2000) and (12000, 4000) beside r4's (3000, 1000), for loop periods of 1,000 to 4,000 tokens ([R791](r791-temp-incidence.md)), and reports `finish_reason: "length"` when `max_tokens` cuts tool-call text that does not parse, where the base reported `tool_calls` with no call.
- Rollback: `DAILY_IMG=tabbyapi:rebase-dev-r3`, R789's launcher.

## Gates

| gate | rule | result |
| --- | --- | --- |
| G1 boot | image id and labels (base = `rebase-dev-r3` `30ed33eb`, round r5, `patch_sha256` = SHA-256 of `cat fix.patch r4-to-r5.patch`), 39 keys verbatim, landing check in the container, free VRAM at boot per card at least 1,149 / 1,727 MiB | labels as required; the image's `/opt/loopthink-r5` files have this repository's md5s; landing prints the ladder `(1, 2, 4)`; 1,181 / 1,759 MiB |
| G2 greedy | `fn_greedy` 6 and `chat_greedy` 6 byte-identical to R785c, including the 5,001-token row across the requeue | 6 / 6 and 6 / 6 identical |
| G5a in-image tests | `test_loop_think.py` (md5 `5ce047a6`; the public copy differs in one comment) inside the served container: exit 0, 89 PASS, 0 FAIL | 89 / 0; detector CPU r5 1.50 against r4 0.87 µs per reasoning token (seeded Zipf), worst 4-token chunk 2.43 ms |
| G3 ramp and stress | 36/36; 4/4 and 8/8 at the 26k setting; 0 OOM | 36/36, 4/4, 8/8 |
| cold prefill (reported) | one salted 90k-token prompt | 89,832 tokens in 7.42 s, 12,112 t/s |
| G5 loop-think, at 0.6 | as R789 | forced 2 × "391" with the injection; thinking-off 2 × stop at 800 tokens; `p107` 2 injected, at 803 reasoning tokens by r4's short detector; `p700` not exercised (8 of 8 rows left the prefilled block), as in every earlier run |
| G6 needles | 5/5 at 131,072 and 240,000 tokens | 5/5, 5/5 |
| G4 headroom | 0 OOM lines, 0 restarts; free VRAM after reported | 361 / 877 MiB |
| G7 agent replay | per-stream decode at least 120.3 t/s | 133.0 t/s, 3.306 tokens per verify step, 24.86 ms per verify step, 427 requests, 0 session instances excluded, 0 collector injections |
| G8 agentic edit | 4 × 6/6 | 4 × 6/6 |
| G9 tool-eval | mean at least 82.0 | 84.2 (95 % interval 83.0 to 85.5) |
| G10 GSM8K | n = 500, 8 concurrent, at least 0.970 | 0.976 |

- No r5 branch ran during G7 to G10: the server log, with wrapped lines joined, has 16 reasoning-loop injections, all at or before 07:15:13 UTC (G5 and the report probes), and exactly 2 "did not parse" lines, both from probe (d). Those gates therefore cannot carry an r5 effect.
- G7: 133.0 against R789's 132.3 is tokens per step (3.306 against 3.269, sampling); 24.86 against 24.70 ms per step (+0.6 %) cannot resolve r5's estimated detector cost of 0.05 to 0.08 %; the request sets differ (427 against 411, longest prompt 193k against 90k tokens).
- G9 84.2 against R789's 85.2: 466 against 470 of 552 points over 4 trials, a per-trial spread of about ±3. The scenarios that lost points also flip between trials of R785b, R785c and R789; none newly fails in all 4 trials.
- G10: 12 wrong documents (R789 12, R785c 10). Two are empty answers at the 8,192-token limit at 79 % and 69 % draft acceptance, with no injection, not loops.

## Report-only probes

- **(b) the forced loop at the served fallback 1.0**, n 8: 5 clean "391", 3 carry the looped line into the answer, the same 37.5 % as [R790](r790-loopthink-temp.md)'s 12 of 32. All 8 rows share the reasoning cut by r4's 800-token detector, which r5 does not change. The answer after an injection is a sampler property.
- **(c) the ladder, live**: R791's turn 16 with its 1,199-token loop block prefilled twice after the pre-loop reasoning, temperature 0.6, n 4, 16,000 maximum tokens. The server log reads "reasoning loop detected after 6003 reasoning tokens" for all 4 requests. The 4 reasonings are byte-identical, 17,999 characters (the loop is token-locked at 0.6), so this is one firing on one trajectory, not four. 6,003 < 12,000 identifies the (6000, 2000) rung; the collector counts generated tokens only, so it fired at the rung's window floor, 5.0 copies of the block after the prefix, as the offline test predicts. The prefix was not echoed. After the injection the 4 samples diverge into 8, 8, 3 and 9 `patch` tool calls that carry out the plan the loop had been repeating, the pattern R790 saw for injected 0.6 rows.
- **(c) at the served fallback 1.0**, results `2026-09-28-r792c-ladder-fallback-0801`, n 8, same prefix: 0 of 8 continued the loop, so no injection was needed, and 8 of 8 answered with `patch` calls after 2,900 to 7,200 characters of reasoning. On this turn at the served sampler the ladder was not exercised. This follow-up ran after the round's review and has not been reviewed.
- **(d) a `write_file` call cut at `max_tokens` 300**, thinking off: `finish_reason` length and 0 tool calls, streamed and not streamed; the server logged the new WARNING "tool call text did not parse into a tool call; finish_reason length" for both requests. The base's `tool_calls` for this case rests on its code and on R791's live row; it was not re-run on the base image in this unit.

The driver's audit line reads "server fired after []" for (c) and "x0" for (d): TabbyAPI wraps its WARNING lines over two log lines, and the driver matched single lines. The server logs above are the record; the parsers must join continuation lines before the next unit is derived from this driver.

## What this does not show

- Any speed change from G7, and any tool-eval or GSM8K effect of r5.
- That the ladder catches loops in the agent client's traffic, which runs at the 1.0 fallback: the only live firing is at 0.6 on a prefilled loop, and at 1.0 the same turn did not loop.
- Four independent catches in (c).
- That the answer after an injection improved: r5 does not touch it, and at 1.0 about 1 in 3 still repeats the looped line after a forced toy loop.
- That the agent client handles the new `length` correctly end to end: its truncation recovery for `length` was read in its code, not exercised.

The promotion row is in [`docs/HISTORY.md`](../../docs/HISTORY.md); the `tool_calls` mislabel is [GOTCHAS 31](../../docs/GOTCHAS.md).
