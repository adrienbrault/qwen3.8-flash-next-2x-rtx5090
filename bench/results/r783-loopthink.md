# R781 to R783: a loop in the thinking ends the thinking instead of the request; promoted as `stack-r3-rows32-tokcount-loopthink3`

Results `2026-09-27-r782-loopthink2-0924` (R782, 2026-09-27 09:24 to 09:30 UTC, on a Qwen3.8-Flash-Next finetune) and `2026-09-27-r783-promote-loopthink-0943` (R783, 09:43 to 09:49 UTC, the served configuration) on the serving host. Layer: [`docker/overlays/loop-think-r3`](../../docker/overlays/loop-think-r3/). Driver: [`scripts/r783-promote-loopthink.sh`](../../scripts/r783-promote-loopthink.sh). Probes: [`bench/chat_greedy.py`](../chat_greedy.py), [`bench/replay_hermes_turn.py`](../replay_hermes_turn.py). Raw records of R783: [`2026-09-27-r783-promote-loopthink-0943/`](2026-09-27-r783-promote-loopthink-0943/). The agent-turn replays (`p107`) and their prefill come from a private agent session and are not published; their per-request summary lines are in `audit.txt`. R782's records stay on the host.

## What was wrong

ExLlamaV3 ends a job when its loop detector finds a repeated period in the last W generated tokens. TabbyAPI passes its `loop_detect_window` as W, 800 tokens by default, with 2 repetitions, so the detector ends periods of up to 400 tokens. TabbyAPI maps the engine's `loop_detected` end to `finish_reason: "stop"`. When the loop is in the thinking, the job ends before `</think>` and the response has reasoning and no content. Hermes Agent treats a `stop` with empty content as a "Reasoning-only clean stop" and uses the reasoning as the final response, so its user sees the looping thoughts as the reply ([GOTCHAS 25](../../docs/GOTCHAS.md)).

- The served configuration did this on 2026-09-26 in two agent turns of 18,125 and 9,218 output tokens, both ending in a loop and both shown as replies.
- A Qwen3.8-Flash-Next finetune did it on 2026-09-27: a request ended by the detector at 2,506 tokens, whose last ~1,000 characters of reasoning repeat 4 times in two alternating variants (effective period about 395 tokens).
- R781 control on the finetune, results `2026-09-27-r781-control`: a loop prefilled in the thinking (40 copies of one line through TabbyAPI's `response_prefix`; asked in the prompt alone, the model did not loop, 0 of 2) ended 2 of 2 as reasoning-only `stop`.

## The change

TabbyAPI's chat collector runs a second `LoopDetector` on the reasoning tokens at the request's window W. On a detection it forces `"\n\nI am repeating myself, so I will stop thinking here and act on what I have.\n"` and `</think>` into the stream through `constrain_generation_output`, the path of TabbyAPI's reasoning budget, and the model answers or calls a tool. On those requests the engine's detector becomes `(2W, 4)`: the collector fires first, and the engine still ends the same periods of up to W/2 tokens, in the content phase, after 2W tokens instead of W. The watch is armed only on chat requests that start in the thinking, with no forced tool call, no structured-output constraint, `loop_detect_window` above 0, and 2W within the 2,048-token output chunk. The overlay's [`README.md`](../../docker/overlays/loop-think-r3/README.md) lists the conditions and the three rounds.

## R781 and R782: the mechanism on the finetune

R781 tested r1; a review found that r1's `(2W, 2)` backstop ended reasoning loops with periods of 401 to 800 tokens, which the unpatched detector never caught, as reasoning-only stops. R782 tested r2 (`(2W, 4)`, armed only when the request starts in the thinking) on the finetune at `:8029`, served by a TabbyAPI image with the same chat endpoint; its offline test passed 18 of 18. These are checks of the patch's behaviour on another checkpoint, not measurements of the served configuration.

| probe | result |
| --- | --- |
| forced: one line prefilled 40 times in the thinking, 2 requests | 2 of 2 answered "391" at 826 tokens; the collector injected at 800 reasoning tokens (the two requests are byte-identical, one trajectory) |
| p107: an agent turn replayed with its thinking prefilled up to 2 copies of the real looping paragraph, 4 requests | the model continued the loop; the collector injected at 800 to 803 reasoning tokens; 4 of 4 then emitted a `patch` tool call with valid JSON |
| p500: a 532-token paragraph prefilled twice, 2 requests | no injection and no engine stop, as without the patch (a period above 400); both ran to `length` at 5,000 tokens; one kept looping (9 copies), the other left the loop |
| nothink: thinking off, the line prefilled 40 times as content, 2 requests | the engine stopped both at exactly 800 completion tokens; no injection |
| replay: the original agent turn, unmodified, 8 requests at 4 streams | 8 of 8 tool calls, no injection; the loop did not recur |

The container log attributes the six injections to the forced and p107 requests and the two engine stops to the nothink requests, one to one. A review of R782 found that at 1,024 < W ≤ 2,048 the 2W window never fills, because the engine rebuilds its detector at every 2,048-token requeue, so r2 left content loops unbounded there; r3 does not arm in that case and logs a DEBUG line when it arms. Its offline test passes 22 of 22. At W = 800 r3 behaves as r2.

## R783: promotion

The unit waited behind R782 for the GPU lock, booted the served configuration (`tabbyapi:stack-r3-rows32-tokcount`, R747) and recorded chat greedy output on it, installed the launcher with `DAILY_IMG=tabbyapi:stack-r3-rows32-tokcount-loopthink3` (the two launchers differ in `DAILY_IMG` and comments only, `launcher-code-diff` empty) and booted it the way the daily boots. The gateway that routes other clients to `:8022` was drained for the unit's lifetime, and the agent client that calls the port directly was pointed at another port, so the greedy requests ran alone. Gates:

- G1: image `tabbyapi:stack-r3-rows32-tokcount-loopthink3`, 41 environment keys, page pool 983,040, free VRAM at boot 1,125 / 1,573 MiB, equal to the configuration it replaces.
- G2: greedy output on the six `fn_greedy` prompts (up to ~100k tokens, `/v1/completions`) identical to R747's served-image rows, 6 of 6. `/v1/completions` does not enter the patched collector.
- G3: greedy output on five `/v1/chat/completions` prompts ([`chat_greedy.py`](../chat_greedy.py); temperature 0, thinking on; code, arithmetic, prose, a tool call, a two-turn conversation) identical between the old and the new image in the same unit, 5 of 5 in finish reason, reasoning, content and tool calls.
- G4, the loop mechanism on the served configuration:

| probe (2 requests each) | result |
| --- | --- |
| forced | 2 of 2 answered "391" at 826 completion tokens, the injected message in the reasoning |
| nothink | 2 of 2 stopped by the engine at exactly 800 completion tokens (thinking off, the engine's `(800, 2)`) |
| p107 | 2 of 2 ended in two `patch` tool calls: #0 after the injection (1,346 tokens); #1 without it (6,506 tokens), see below |

  The container log counts 3 collector injections and 2 engine loop stops.
- G5: 0 out-of-memory errors, 0 tracebacks, 0 container restarts.

Rollback: `DAILY_IMG=tabbyapi:stack-r3-rows32-tokcount`, the R747 launcher.

## Limits of the evidence

- The patch covers loop periods of up to 400 tokens (W/2 at the default W). In R783's p107 #1 the reasoning repeats one block of about 2,300 characters (about 700 tokens) 6 times verbatim and then changes topic; a period that long is outside the collector's detector, the backstop and the unpatched engine detector, and the request reached its tool calls without the patch acting. G4's "tool call or content" rule passed it. Such loops run until the model leaves them or reaches `max_tokens`, as before the patch. Raising `loop_detect_window` does not extend the coverage under r3: above W = 1,024 the chunk guard disarms the collector and the engine's `(W, 2)` stop returns.
- "Non-looping output unchanged" is measured on short outputs: the five chat prompts end in under ~250 tokens, so the detector's window never filled and no requeue was crossed. By construction the collector only reads tokens until it fires.
- On a thinking-on request a content-phase loop is now ended after a 1,600-token window instead of 800; this case was not run on the GPU. Only the thinking-off case (800 tokens) was.
- Each forced and nothink pair was byte-identical, so each is one trajectory.
- No decode-rate pair was run.
- The review of R783 found two defects in the driver, not fixed in the copy here and to be fixed before it is reused: an abort after the GPU lock (for example on the live launcher's checksum) exits without booting the daily, and G4 does not tie injections to requests (it should require the message in the forced rows, none in the nothink rows, and classify p107 rows as injected, left the loop, or no loop).
