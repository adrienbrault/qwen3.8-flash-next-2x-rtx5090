# loop-think-r5: reasoning loops with periods of up to 4,000 tokens end the reasoning, and a tool call cut by `max_tokens` finishes as `length`

`tabbyapi:rebase-dev-r3-loopthink5` is `tabbyapi:rebase-dev-r3` plus [`r4-to-r5.patch`](r4-to-r5.patch), a TabbyAPI change in `endpoints/OAI/utils/chat_completion.py`. ExLlamaV3 and `common/sampling.py` are unchanged from r4. Served since 2026-09-28 09:49 CEST ([R792](../../../bench/results/r792-promote-loopthink5.md)). r5 is [`loop-think-r4`](../loop-think-r4/README.md) plus two independent changes; r3's README describes the original bug.

```sh
docker build -f Dockerfile.box --build-arg BASE=tabbyapi:rebase-dev-r3 \
  --label local.loopthink.base_id=$(docker image inspect tabbyapi:rebase-dev-r3 --format '{{.Id}}') \
  --label local.loopthink.patch_sha256=$(cat fix.patch r4-to-r5.patch | sha256sum | cut -c1-64) \
  -t tabbyapi:rebase-dev-r3-loopthink5 .
```

## 1. Long-period rungs on the reasoning loop watch

- r4's detectors on the reasoning see periods of up to L = 1.25 W = 1,000 tokens at the default window W = 800. ExLlamaV3's `LoopDetector` fires only when its whole window is periodic, so a longer period escapes all of them.
- [R791](../../../bench/results/r791-temp-incidence.md) found two such loops arising on the served configuration at temperature 0.6, with no prefix, on ordinary agent turns. One repeats a block of 1,199 tokens 25 times, the other a block of 3,713 tokens 6 times (token tail periods measured offline). Both ran to `max_tokens` 32,768 with reasoning and no answer.
- r5 runs collector-only rungs `LoopDetector(3kL, kL)` for k in `LOOP_THINK_LADDER = (1, 2, 4)`: at W = 800 that is r4's (3000, 1000) plus (6000, 2000) and (12000, 4000). They sit next to r4's short (W, W/2) detector and share its path: the same one-line message and `</think>` through `constrain_generation_output`, at most once per request, with the same WARNING line. The engine detector on a watched request stays at r3's (2W, 4).
- A single larger L does not work, because the whole window must be periodic: (12000, 4000) alone needs 12,000 tokens of loop even for a period of 1,200. A rung fires once its window holds 3kL/p copies: 3 copies at its top period, about 6 just above the rung below. Worst-case latency is about 12,000 tokens plus one copy.
- Offline, on the two R791 loops (`test_loop_think.py` §14): the 1,199-token loop fires at rung (6000, 2000) at reasoning token 7,687, 5.0 copies into the loop; the 3,713-token loop fires at rung (12000, 4000) at token 19,901, 3.2 copies in. r4's detectors stay silent on both over all 32,768 tokens. The same points hold with streaming on and with 64-token chunks (§22).
- For a loop that r4 already catches (period up to 1,000), the rungs' windows are longer, so r5 fires at r4's token: 120 random single-block loops (period 1 to 1,000, 1 to 12 copies), same token in 120 of 120 (§16). r4's 27 checks pass unchanged (§1 to §13).
- A rung fires only after at least 3 back-to-back, token-exact copies of a block fill its 6,000- or 12,000-token window. None of these fire (§15, §17): a 21,879-token reasoning with about 2 copies of a 3,300-token block that differ by 2 characters (the model left that repetition by itself); a plan restated 3 times; a 1,600-item numbered list whose items differ only in the number; a varied list; a 1,400-row markdown table; 800 lines of near-identical code; a 1,500- or 4,000-token block restated once.
- Only reasoning tokens are watched (§20), as in r4: a period-1,199 block repeated 10 times after `</think>` or inside a `<tool_call>` in the reasoning is not injected; the same block in plain reasoning is.

Detector CPU cost (§18, report only, detectors alone, per reasoning token in 4-token chunks, best of 3):

| stream | r4 µs/token | r5 µs/token | r5 worst 4-token chunk |
| --- | --- | --- | --- |
| seeded Zipf, 30k tokens, no loop, build host (macOS) | 0.77 to 0.93 | 1.31 to 1.70 | 2.3 to 3.3 ms |
| agent reasoning, 30k tokens, no loop, build host | 0.62 to 0.79 | 1.08 to 1.23 | 3.0 to 3.7 ms |
| the same tests inside the served image on the serving host (R792 G5a) | 0.72 to 0.87 | 1.27 to 1.50 | 2.43 to 3.37 ms |

The collector runs on the same event loop as `generator.iterate()`, so this cost is serial with decode. At about 130 tokens/s per stream and 8 streams it is 0.5 to 0.8 ms per second of decode, 0.05 to 0.08 %. The worst chunk comes from the (12000, 4000) rung, whose sleeping periods wake together at multiples of its window: one decode step for all streams is delayed by 3 to 5 ms about once per 12,000 reasoning tokens per stream. R792's agent replay cannot resolve a cost of this size (24.86 against 24.70 ms per verify step on a different request set) and ran no r5 branch.

Not covered:
- periods above 4L = 4,000 tokens (§17: period 4,100, 5 copies);
- loops that are not token-exact (one token differs per copy);
- the 3,713-token loop is caught at about 19,900 tokens; under a 32,768-token cap the turn that produced it needs more than the remaining 12,900 tokens for its tool call, so it probably still ends at `length`, with the reasoning ended;
- the rungs arm only where r4 arms: reasoning phase, 2W within the output chunk (`max_rq_tokens`, 2,048 on the served configuration), no forced tool call, no structured-output constraint, `loop_detect_window` above 0;
- the answer after an injection is unchanged by r5. At temperature 1.0, 12 of 32 answers after a forced line loop repeat the looped line ([R790](../../../bench/results/r790-loopthink-temp.md)).

## 2. `finish_reason` of a tool call cut by `max_tokens`

- TabbyAPI `53da7919` sets `finish_reason: "tool_calls"` whenever tool-call text was generated, in the streaming path (`if finish_reason and full_tool`) and in the non-streaming path (`if full_tool`), also when `_parse_tool_calls` returns nothing. In R791 one request reached 32,768 tokens inside a `write_file` call; the server logged no parsed tool call, and the client received `finish_reason: "tool_calls"` with no call attached ([GOTCHAS 31](../../../docs/GOTCHAS.md)).
- r5's `_finish_with_tool_calls` changes that one case: tool-call text generated, no call parsed, and the backend's `eos_reason` is `max_new_tokens` → `length`. Every other end keeps the base's `tool_calls`, with or without a parsed call, including `stop_token`, `stop_string` and `loop_detected` with unparsed tool text. A first revision kept the backend's reason in those cases too; that turned an engine loop stop inside a tool call with no preamble into `stop` with no content and no call, the shape agent clients show as the reply, and the review sent it back.
- Whenever tool text does not parse, the helper logs a WARNING `tool call text did not parse into a tool call; finish_reason length|tool_calls`.
- Covered in streaming and non-streaming (§19, §21): cut inside a call → `length`, 0 calls; complete call → `tool_calls`, 1 call; no tool text → `length`; unparsed tool text under `stop_token`, `loop_detected` or `stop_string` → `tool_calls`, 0 calls, as in the base. With `stream_options.include_usage` the usage chunk carries the finish; with n = 2 non-streaming, a cut call and a complete call get `length` and `tool_calls` independently.
- Accepted edge: one complete call followed by a second call cut by `max_tokens` reports `tool_calls` with the one parsed call; the cut call's text is dropped, as in the base.
- `check_forced_tool_calls` is untouched.

## Evidence

- Offline: [`test_loop_think.py`](test_loop_think.py), 89 checks (r4's 27 unchanged, 62 new) with a fake backend and the real `LoopDetector`. On the r4 tree the file fails exactly the 20 checks of the new behaviour. On 2026-09-28 the file ran with `run_offline.py` on a CPU-only host against TabbyAPI `53da7919` with `fix.patch` applied: 89 PASS, 0 FAIL, with the private fixture and with the synthetic public one below, and every check line identical between the two. Applying r4's `fix.patch` and then `r4-to-r5.patch` to the same tree gives files identical to the `fix.patch` route.
- In the served image, R792 G5a: the same file inside `tabbyapi:rebase-dev-r3-loopthink5` on the serving host, 89 PASS, 0 FAIL; the image's `/opt/loopthink-r5` files have the md5s of this directory's `fix.patch`, `r4-to-r5.patch` and `landing_loopthink.py`, and its label `local.loopthink.patch_sha256` is `2794fd6b744f0dbd91f59314855367133670a00a084551ebe94693ca36d2f40d`, the SHA-256 of `cat fix.patch r4-to-r5.patch`.
- On the GPU, R792 ([write-up](../../../bench/results/r792-promote-loopthink5.md)): the (6000, 2000) rung fired on the served configuration at 6,003 generated tokens, 5.0 copies into a prefilled copy of R791's 1,199-token loop, at temperature 0.6 (one looping trajectory, answered in 4 of 4 samples after the injection); a `write_file` call cut at `max_tokens` 300 with thinking off returned `length` and 0 calls, streamed and not streamed, with the new WARNING in the server log both times. At the served fallback temperature 1.0 the same prefix did not continue the loop in 8 of 8 requests, so the rung did not fire there.
- The agent client in use on this configuration handles `finish_reason: "length"` through its truncation recovery, and re-prompts on `tool_calls` with an empty call list; this was read in its code, not exercised end to end.

## Files

- [`fix.patch`](fix.patch): pristine TabbyAPI → r5, `-p1` in `/app`, two files. [`r4-to-r5.patch`](r4-to-r5.patch): r4 → r5, `chat_completion.py` only. Both applied at fuzz 0 after a dry run, `--forward`.
- [`Dockerfile.box`](Dockerfile.box): applies `r4-to-r5.patch` when the base carries `/opt/loopthink-r4` and `fix.patch` otherwise; build context this directory, CPU only.
- [`landing_loopthink.py`](landing_loopthink.py): run by the build; fails it unless the ladder, `LOOP_THINK_LADDER == (1, 2, 4)`, a (6000, 2000) detector that catches period 1,199, both finish sites, the helper's six finish cases and r4's checks are in the image.
- [`test_loop_think.py`](test_loop_think.py): the served image's file (md5 `5ce047a6d4200c8e703e0187fcd33e1a`) with one change, the 4-line comment above `FIX`, which describes this synthetic fixture instead of the re-tokenized original; no code differs. In the image, `docker run --rm -v $PWD:/t -w /app --entrypoint python3 tabbyapi:rebase-dev-r3-loopthink5 /t/test_loop_think.py`; on a CPU-only host, `python3 run_offline.py APP_DIR`.
- [`run_offline.py`](run_offline.py): CPU-only runner; serves [`base/loop_detect.py`](base/loop_detect.py), ExLlamaV3's `generator/loop_detect.py` unmodified (MIT, md5 `8ec7c1f558256ede7e42876e4d8013bc`, the same file in upstream `dev` `5783a93` and in the served image), from a stub package, so neither torch nor the ExLlamaV3 wheel is needed. APP_DIR is a TabbyAPI tree with the patch applied and TabbyAPI's CPU dependencies in the interpreter.
- [`mkpatch.sh`](mkpatch.sh): regenerates both patches as `diff -ruN a b` from `base/` (the two pristine TabbyAPI `53da7919` files), `src/` (the patched files) and `../loop-think-r4/src/` (r4's patched files). Those trees are not in this repository; rebuild them from a TabbyAPI checkout at `53da7919` and the patches here before running it.
- [`fixtures/r791-rows.json.gz`](fixtures/r791-rows.json.gz): synthetic token ids, generated by `random.Random(791)`; none is derived from a real token id. Four rows (`u16_t06_len`, `u10_t06_len`, `u10_t10_max`, `u1_t10_max`) stand for R791's rows from a private agent session: the two loops, the repetition the model left by itself, and the plan restated 3 times. Each keeps only the row's length and every maximal run of x[i] = x[i−p] with p ≤ 4,000 and length ≥ 400 in the original ids, listed in the row as `structure` (period, first and last position): for the loops that is the tail period from the loop's second copy to the end (1,199 from token 2,887; 3,713 from token 11,615, with a 3,727-token run from 8,507 to 11,620 before it), for the left repetition the five shifted near-copy runs (periods 2,081 to 3,602) whose ends mark where the copies differ, and for the restatement one run of 628 tokens at period 692 from token 1,665. Every position gets the id of its class under those runs; class ids are drawn without replacement from the 248,320-id vocabulary, so no other two positions are equal. No detector in the tests can act on a shorter run, because `LoopDetector(W, P)` needs a run of at least W − p ≥ 400. With this fixture every check line of the test file is identical to the run on the private original (re-tokenized reasoning); only the CPU-cost lines (§18) differ, with the host. The four `syn_*` rows are the tokenized template texts of the false-positive checks (a numbered list, a varied list, a markdown table, repeated code lines), kept as they are without the text.
