# loop-think-r3: a loop in the reasoning phase ends the reasoning, not the request

`tabbyapi:stack-r3-rows32-tokcount-loopthink3` is `tabbyapi:stack-r3-rows32-tokcount` plus [`fix.patch`](fix.patch), a TabbyAPI change in two files: `endpoints/OAI/utils/chat_completion.py` and `common/sampling.py`. ExLlamaV3 is unchanged. Served since 2026-09-27 11:49 CEST ([R783](../../../bench/results/r783-loopthink.md)).

```sh
docker build -f Dockerfile.box --build-arg BASE=tabbyapi:stack-r3-rows32-tokcount -t tabbyapi:stack-r3-rows32-tokcount-loopthink3 .
```

## The bug

ExLlamaV3's loop detector ends a job when the last W generated tokens hold a repeated period; TabbyAPI passes `loop_detect_window` as W (default 800) with 2 repetitions, so it catches periods up to 400 tokens. When the loop is inside the thinking, the job ends before `</think>`, TabbyAPI reports `finish_reason: "stop"`, and the response carries reasoning and no content. An agent client that promotes the reasoning to the answer on an empty `stop` (Hermes Agent logs this as "Reasoning-only clean stop") then shows the looping thoughts as the reply. It was seen on the served configuration on 2026-09-26 (two turns of 18,125 and 9,218 output tokens) and on a Qwen3.8-Flash-Next finetune on 2026-09-27 (a request cut at 2,506 tokens, one paragraph repeated 4 times). [GOTCHAS 25](../../../docs/GOTCHAS.md) has the form it takes in a client.

## The change

- `_chat_stream_collector` runs its own `LoopDetector` (ExLlamaV3's `generator/loop_detect.py`) on the tokens of the reasoning phase, at the request's window W.
- On a detection it forces `"\n\nI am repeating myself, so I will stop thinking here and act on what I have.\n"` followed by `</think>` into the stream through `constrain_generation_output`, the path TabbyAPI's reasoning budget uses, once. The model then answers or calls a tool.
- On those requests the engine's detector becomes `(2W, 4)`: a 2W window, so the collector's detector fires first, and 4 repetitions, so it matches the same periods of up to W/2 tokens as before. It still ends content-phase loops, after 2W tokens instead of W. The override is a private attribute (`BaseSamplerRequest._loop_backstop`), so `loop_detect_window` and the per-request settings log are unchanged.
- The watch is armed only when the request starts in the reasoning phase, 2W fits in the output chunk (`max_rq_tokens`, 2,048 on the served configuration; the engine rebuilds its detector at every requeue, so a larger window never fills and the request keeps `(W, 2)`), the request forces no tool call, carries no `json_schema`, `regex_pattern` or grammar, and `loop_detect_window` is above 0. A detection on the chunk that ends the job injects nothing.
- Arming logs a DEBUG line with the engine's window and repetitions; a detection logs the WARNING `reasoning loop detected after N reasoning tokens observed`.
- Not covered, as in TabbyAPI without the patch: models that open `<think>` themselves, `continue_final_message` without think tags, `start_in_reasoning: never`, and requests the `auto` setting does not recognise as starting in reasoning.

## Rounds

- r1 gave the engine `(2W, 2)`. That doubled the longest period it detects to 800 tokens, so reasoning loops with a period of 401 to 800 tokens, which the served detector never caught, were now ended as reasoning-only stops, the bug itself. r1 also armed on thinking-off requests, logged the detector's period as the largest multiple of the true period under the cap, and wrote the override into `loop_detect_window`, which the settings log then showed as sent by the client.
- r2 (R782) fixed those four: `(2W, 4)`, armed only when the request starts in reasoning, the true period not logged, a private attribute.
- r3 adds the chunk guard above: at 1,024 < W ≤ 2,048 a 2W window never fills, so r2's backstop could not fire and content loops were not ended. r3 also adds the DEBUG line. At the default W = 800, r3 behaves as r2.

## Files

- [`fix.patch`](fix.patch): `-p1` in `/app`, two files, applied at fuzz 0 after a dry run, `--forward` so a second apply fails. SHA-256 `7769e0208bbf3252f06da4c0a8cec9de99c98034a4638a521d0de6b7a580e0a8`.
- [`Dockerfile.box`](Dockerfile.box): the layer; build context this directory, CPU only.
- [`landing_loopthink.py`](landing_loopthink.py): run by the build, fails it unless the patched collector and `get_stop_on_loop` are in the image.
- [`test_loop_think.py`](test_loop_think.py): 22 offline checks with a fake backend and the real `LoopDetector`, run inside the image: `docker run --rm -v $PWD:/t -w /app --entrypoint python3 tabbyapi:stack-r3-rows32-tokcount-loopthink3 /t/test_loop_think.py`.
