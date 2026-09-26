# R737 and R747: the requeue token-count fix, restored and promoted

Results `2026-09-26-r737-tokcount` (R737, 2026-09-26 07:57 to 08:02 UTC) and `2026-09-26-r747-promote-tokcount-1136` (R747, 11:37 to 11:38 UTC) on the serving host. Layer: [`docker/overlays/tokcount-r1`](../../docker/overlays/tokcount-r1/). Probe: [`tokcount_check.py`](../../docker/overlays/tokcount-r1/tokcount_check.py).

## What was wrong

[GOTCHAS 4](../../docs/GOTCHAS.md) describes the bug: `Job.prepare_for_requeue` in `exllamav3/generator/job.py` carries only the current segment's token count into the requeued job, so a generation longer than the 2,048-token requeue budget is reported as its last one or two segments. `usage.completion_tokens` and the tokens-per-second figure in the server log are wrong; token limits are not, because `max_new_tokens` is carried separately.

The fix of 2026-09-16 was a `sed` in `docker/Dockerfile.tabbyapi`, which built `tabbyapi:53da7919-rqcount`. Every image from `tabbyapi:qsa-cid` on starts again from a fresh TabbyAPI and ExLlamaV3 install in `Dockerfile.tabbyapi-qsa-cid`, so the fix was not in any served image from `qsa-cid-pr337` (2026-09-16) to `stack-r3-rows32` (2026-09-26). The documentation said it was.

## R737: the fix on the served image

`tokcount-r1` is `stack-r3-rows32` plus one changed line, applied at fuzz 0 after a dry run: `"rq_new_tokens": self.rq_new_tokens + self.new_tokens`, so the count accumulates across requeues. The landing check parses `job.py`, asserts the new expression and the absence of the old one, and asserts that `rq_new_tokens` is read in one place besides the requeue path: the finish report's `new_tokens`. The requeue limit reads `last_completed_tokens`, not this field.

Three streamed generations with `min_tokens` forcing 9,000, 13,000 and 20,000 tokens, 3 streams, `loop_detect_window: 0`:

| forced tokens | served image: `usage` | served image: logged T/s | tokcount-r1: `usage` = log | tokcount-r1: logged T/s | client-timed T/s |
|---|---|---|---|---|---|
| 9,000 | 2,969 | 57.3 | 9,000 | 188.4 | 183.5 |
| 13,000 | 2,888 | 47.6 | 13,000 | 217.8 | 210.6 |
| 20,000 | 3,714 | 34.8 | 20,000 | 241.2 | 232.0 |

The served image's timer was right and its count was wrong: 2,969 tokens over 57.3 s of logged time is the true count's 173.7 T/s over the same interval. The patched log reads 2.7 to 4.0 % above the client's clock because the server excludes the time a job waits between requeued segments. Greedy output on the six `fn_greedy` prompts (up to ~100k tokens) is identical to the served image's. The long generations' text differs between arms at 3 streams, as it does between two runs of the served image (batch nondeterminism).

## R747: promotion

The unit installed the launcher with `DAILY_IMG=tabbyapi:stack-r3-rows32-tokcount` and booted it the way the daily boots. Gates:

- image `tabbyapi:stack-r3-rows32-tokcount`, 41 environment keys, page pool 983,040, free VRAM at boot 1,125 / 1,573 MiB (equal to the configuration it replaces);
- greedy output on the six `fn_greedy` prompts identical to R737's served-image rows;
- one streamed generation forced to 9,000 tokens: `usage.completion_tokens` 9,000, and the server log reads "9,000 tokens generated";
- 0 out-of-memory errors, 0 tracebacks, 0 container restarts.

## What this changes in earlier numbers

Every other decode figure in the README is timed on the client over 512 or 1,024 forced tokens, below the ~2,048-token threshold, and does not use the server's count. [R583](r583-long-generation.md)'s 65.6 tokens/s per stream for a real agent session was computed from the server's per-request log, which under-counted every generation longer than about 4,096 tokens by up to 5×, so that figure is too low by an amount that depends on how many of that session's 326 requests generated more than 4,096 tokens. It is withdrawn until re-measured on this image.

R583 read that session's nineteen requests under 50 tokens/s as 2,700 to 4,000-token generations at 13 to 20 tokens/s. That is the range every under-counted generation falls into (2,048 to 4,096 reported tokens; R737's served image reported 2,888 to 3,714 for 9,000 to 20,000), so those requests were most likely longer generations reported short, not slow ones.
