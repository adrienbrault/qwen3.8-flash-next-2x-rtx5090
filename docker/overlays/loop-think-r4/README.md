# loop-think-r4: a loop in the reasoning phase ends the reasoning, including periods of up to 1,000 tokens

`tabbyapi:stack-r3-rows32-tokcount-loopthink4` is `tabbyapi:stack-r3-rows32-tokcount` plus [`fix.patch`](fix.patch), a TabbyAPI change in two files: `endpoints/OAI/utils/chat_completion.py` and `common/sampling.py`. ExLlamaV3 is unchanged. It was not served on its own: it is the base of `tabbyapi:rebase-dev-r3`, which keeps its `/app` and replaces its ExLlamaV3 ([`../rebase-dev-r3/`](../rebase-dev-r3/)), served since 2026-09-27 15:01 CEST ([R785](../../../bench/results/r785-promote-rebase-r3.md)). r4 is [`loop-think-r3`](../loop-think-r3/README.md) plus one detector; r3's README describes the bug and the rest of the change.

```sh
docker build -f Dockerfile.box --build-arg BASE=tabbyapi:stack-r3-rows32-tokcount \
  --label local.loopthink.patch_sha256=$(sha256sum fix.patch | cut -c1-64) -t tabbyapi:stack-r3-rows32-tokcount-loopthink4 .
```

## What r4 adds

- In R783's promotion unit, one of the two agent-turn replays (`p107` #1) repeated a block of about 700 tokens 6 times in its reasoning. No detector of r3 caught it: the collector's `(W, W/2)`, the engine backstop `(2W, 4)` and the unpatched engine `(W, 2)` all stop at a period of W/2 = 400 tokens ([R783](../../../bench/results/r783-loopthink.md)).
- r4 adds a second, collector-only detector, `LoopDetector(3L, L)` with L = 1.25 W (1,000 tokens at the default W = 800). It fires on a period of up to L once 3 copies fill its 3L window, 3,000 tokens at the default. It feeds on the same reasoning tokens as r3's detector, and a detection by either forces the same one-line message and `</think>`, once.
- 3 copies, not 2, so a block restated once verbatim (a draft rewritten as the final version) does not fire.
- The collector's detector is never reset at a requeue, so this needs no engine change. The engine detector on a watched request stays at r3's `(2W, 4)`.
- Periods above L are caught by no detector, as before the patch: such a loop runs until the model leaves it or reaches `max_tokens`.
- Everything else is r3: armed only on a chat request that starts in the reasoning phase, forces no tool call, carries no structured-output constraint, has `loop_detect_window` above 0 and 2W within the output chunk (`max_rq_tokens`, 2,048 on the served configuration).

## Evidence

- Offline: [`test_loop_think.py`](test_loop_think.py), 27 checks with a fake backend and the real `LoopDetector`, run inside the image.
- On the GPU, R785 G5 on the promoted configuration ([R785](../../../bench/results/r785-promote-rebase-r3.md)): a loop prefilled in the thinking answered "391" 2 of 2 after the injection; a thinking-off content loop stopped at 800 tokens 2 of 2 (engine); the agent-turn replay ended in tool calls 2 of 2, both after an injection. The case r4 exists for was not exercised: in 8 requests with a 688-token block prefilled twice in the thinking, each generation began the block once more and then left it, so no request generated the three copies the long-period detector needs in its 3,000-token window, and no detector fired; all 8 ran to the 8,000-token limit with reasoning and no content.
- The build of `rebase-dev-r3` re-runs this overlay's landing and the TabbyAPI call-site audit (`tests/test_tabby_callsites.py`) against the ported engine, including the three engine interfaces loop-think depends on (`LoopDetector`, `max_rq_tokens` requeue, `constrain_output_now`).

## Files

- [`fix.patch`](fix.patch): `-p1` in `/app`, two files, applied at fuzz 0 after a dry run, `--forward` so a second apply fails. SHA-256 `5b8244f2158179ad64f7e9ea74fc2bf2e3af40dadb42842a803825179d0bbca6`, the value of the served image's label `local.loopthink.patch_sha256`.
- [`Dockerfile.box`](Dockerfile.box): the layer; build context this directory, CPU only.
- [`landing_loopthink.py`](landing_loopthink.py): run by the build, fails it unless the patched collector (both detectors) and `get_stop_on_loop` are in the image.
- [`test_loop_think.py`](test_loop_think.py): `docker run --rm -v $PWD:/t -w /app --entrypoint python3 tabbyapi:rebase-dev-r3 /t/test_loop_think.py`.
