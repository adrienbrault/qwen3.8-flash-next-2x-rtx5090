# R791: on 8 ordinary agent turns, no answer in 2 of 32 rows at the sampler fallback 0.6 and 1 of 32 at 1.0 (Fisher p = 1.0); the two 0.6 failures are exact loops with periods of about 1,200 and 3,700 tokens, above loop-think r4's 1,000; the 1.0 failure is a tool call cut by `max_tokens` that the server reported as `tool_calls`

Results `2026-09-28-r791-temp-incidence-0119` on the serving host (2026-09-28 01:19 to 01:30 UTC). Driver: [`scripts/r791-temp-incidence.sh`](../../scripts/r791-temp-incidence.sh). Probe: `replay_hermes_turn.py` with `--temperature` (a later revision of [`bench/replay_hermes_turn.py`](../replay_hermes_turn.py)). Raw records: on the serving host only; they hold the reasoning, content and tool calls of a private agent session and stay private. The [`loop-think-r5`](../../docker/overlays/loop-think-r5/README.md) test fixture carries synthetic ids that keep only the lengths and the long repeated-run structure of the four rows discussed below.

## Setup

- Client-only, on the served configuration of 2026-09-28 (image `tabbyapi:rebase-dev-r3`, 39 engine keys, loop-think r4). Every assistant turn of one private agent session (R781's snapshot: 17 messages, one task, a three.js scene), replayed from its history with no prefix: turns 1, 3, 5, 8, 10, 12, 14 and 16.
- Explicit temperature 1.0 or 0.6, the preset's top_k 20 and top_p 0.95, `max_tokens` 32,768, `reasoning_effort` medium (the value the agent client sends), n 4 per arm per turn, 4 concurrent; the arm order alternates per turn.
- A row counts as looped when an injection fired or when at least half of its reasoning lines of 20 characters or more repeat (rep ≥ 0.5).

## Result

| arm | rows | no answer | looped | loop-think injections | engine loop stops |
| --- | --- | --- | --- | --- | --- |
| 0.6 | 32 | 2 (turns 10, 16) | 2 (turns 10, 16) | 0 | 0 |
| 1.0 | 32 | 1 (turn 10) | 1 (turn 10) | 0 | 0 |

- No answer 2 of 32 against 1 of 32, Fisher p = 1.0; Wilson 95 % intervals 1.7 to 20.1 % and 0.6 to 15.7 %. All three are `max_tokens` hits on the two heaviest turns. Rows within a turn are independent draws, but the turn dominates: the effective sample is 8 turns, not 64 rows.
- **Turn 16 at 0.6**: a 3,575-character block repeated 25 times, verbatim from the second copy on; exact token tail period 1,199 (offline re-tokenization). Ran to `length` with reasoning and no answer.
- **Turn 10 at 0.6**: a 9,999-character block repeated 6 times, verbatim from the second copy on; exact token tail period 3,713. Ran to `length` with reasoning and no answer.
- **Turn 10 at 1.0**: about 2 near-exact copies (2 characters differ) of a block of about 3,300 tokens; the model left the repetition by itself 1,540 characters into a third copy, planned for another 21,000 characters, and reached 32,768 tokens inside a `write_file` call. The server logged "max_tokens reached" and no parsed tool call; the client received `finish_reason: "tool_calls"` with no call attached ([GOTCHAS 31](../../docs/GOTCHAS.md)).
- Per-turn reasoning length is similar between arms; R790's "1.0 thinks longer" was specific to that turn. Wall-time medians differ by arm order (the first arm on a turn pays the cold prefill) and are not a temperature effect.

## Why loop-think r4 did not fire

- r4's detectors on the reasoning see periods of up to L = 1.25 W = 1,000 tokens at the default window W = 800, and ExLlamaV3's `LoopDetector` fires only when its whole window is periodic. Both 0.6 loops have longer periods, so no detector could fire; the r4 README lists this gap. Raising `loop_detect_window` does not help: above W = 1,024 the 2W window no longer fits the 2,048-token output chunk and the collector does not arm.
- The 1.0 repetition is not token-exact and escapes any token-exact detector; it needed none.
- A ladder of collector-only detectors at (6000, 2000) and (12000, 4000) next to r4's (3000, 1000) catches both 0.6 loops offline, at reasoning tokens 7,687 and 19,901. That became loop-think r5, together with `length` for the cut tool call, and was promoted in [R792](r792-promote-loopthink5.md).

## External validity

- One session and one task family; every event falls in 2 of 8 turns, and turn 10 needs about 20,000 tokens even when clean.
- `max_tokens` 32,768 equals the launcher's reasoning budget (`reasoning_budget_tokens`), so the budget cannot end a loop before `length`.
- A reasoning-only `length` is not the failure loop-think was built for (a reasoning-only `stop` that the agent client shows as the reply); the agent client continues after `length`.

## What this does not show

- "1.0 answered 32 of 32": it answered 31.
- Any temperature effect, or any incidence rate.
- That the r4 detector malfunctioned: both loops are outside its designed range.

With R790, the persistent loops seen so far all occurred at 0.6 (the continued loop on turn 16, 8 of 8; the two exact loops here); none at 1.0. The fallback stayed at 1.0.
