# R803, R809, R810: the sub-page leftover of a prefill merged into the last forward, with the recurrent stash copied asynchronously, cuts agent-turn prefill by 13 to 17 % and per-stream decode rises about 3 %; the numerics change sits inside a partition null, and `tabbyapi:merge-tok-r1` is served

Results on the serving host: `2026-09-29-r803-prefill-merge-ab-1218` (R803, 2026-09-29 12:18 to 15:26 UTC), `2026-09-29-r809-merge-quality-gate-1703` (R809, from 17:03 UTC), `2026-09-29-r809t-tier-recheck-1759` (R809t, 17:59 to 18:02 UTC), `2026-09-29-r809p-promote-merge-2000` (R809p, the promotion, served from 20:02 UTC), `2026-09-29-r810-post-checks-2004` (R810) and `2026-09-29-r810b-doc243-2006` (R810b). Background: `2026-09-29-r802-s23-prefill-ladder-0929` (R802) and `2026-09-29-r804-prefill-profile-1035` (R804). Drivers: `r803-prefill-merge-ab.sh`, `r809-merge-quality-gate.sh`, `r809t-tier-recheck.sh`, `r809p-promote-merge.sh`, `r810-post-promotion-checks.sh` and `r810b-doc243-replay.sh`, with the probes `prefill_ladder.py`, `r803_ab.py`, `partition_fidelity.py`, `merge_counter.py`, `r809_gate.py`, `gpu_nvme_ab.py` and `gsm_doc_replay.py`, not in this repository. Launcher: [`scripts/launchers/launch-flashnext-r809-merge.sh`](../../scripts/launchers/launch-flashnext-r809-merge.sh) = [`scripts/launch-flashnext.sh`](../../scripts/launch-flashnext.sh), R808's launcher with `DAILY_IMG` changed and two keys added. Overlay: [`docker/overlays/prefill-merge-r1/`](../../docker/overlays/prefill-merge-r1/README.md). Raw records: on the serving host only; this write-up quotes the analyzers' summaries and independent reviews that recomputed every gate from the raw per-request records.

## Background: R802 and R804

- R802 (09:29 to 09:45 UTC, the served configuration, no peers) ran a solo prefill ladder: N new tokens behind a warm 27,136-token stash, cold, and re-hits. The served forward rule cuts the forward that reaches the last full 256-token page at that page, and the rows after it are one more forward. At N = 2,807 about 40 % of the prefill was fixed cost: that leftover forward (about 7 to 20 ms at 1 row, about 67 ms at 51 rows, 100 to 120 ms at 246 rows and more), a bulk forward's fixed part (65 to 70 ms warm) and about 20 ms of recurrent-state copy to the host.
- R804 (a profile in a harness, contexts 257 to 4,095, cold and behind a 27k prefix) put one stash at 8 to 10 ms (73 pageable device-to-host copies, 57 MiB) and found two per 2,048-to-4,095-token agent turn: the checkpoint at the 2,048 boundary and the last page.
- Two levers followed: (a) run the leftover in the last forward, with the last-page stash captured inside that forward, and (d) copy stashes to the host asynchronously. The overlay implements both behind `EXL3_PREFILL_MERGE` and `EXL3_STASH_ASYNC`.

## R803: A/B on the served configuration

Image `tabbyapi:prefill-merge-r1` (`9fef9276f0d6`) = `tabbyapi:rebase-dev-r3-loopthink5` + `fix.patch` (`a5102079…`). Four arms, OFF (both keys unset), MERGE, MERGE+ASYNC (MA) and OFF2, in two blocks, one fresh boot and NVMe tier directory each. Per arm: a prefill ladder, the greedy reference prompts (`fn_greedy` 6, `chat_greedy` 6) and a 600 s agent-shaped replay at 8 slots (seed 38107, temperature 0.6, 457 to 466 requests, 94 to 95 % of them 2,048 to 4,095 new tokens behind a median 91 % cached ~30k-token prefix).

**Step 0, a GPU stash-equivalence gate in one process** (17 cases of cached prefix and new tokens; OFF, ASYNC, MERGE, MERGE+ASYNC and a CTRL arm on one loaded model, [`tests/gpu_stash_equiv.py`](../../docker/overlays/prefill-merge-r1/tests/gpu_stash_equiv.py)), three runs:

1. Stopped: the first asynchronous stash failed in the worker thread with "Inplace update to inference tensor outside InferenceMode is not allowed". `torch.inference_mode` is thread-local; the fix and its test are in the overlay ([GOTCHAS 36](../../docs/GOTCHAS.md)).
2. Stopped on the pre-registered whole-stash bound: MERGE against OFF 0.126 relative L2 against a bound of 0.05. That bound could not separate a capture error from row-count rounding. The stop gate (not the verdict bars) was amended before the next run, and marked as such in the driver: the first GDN layer within 1e-2, the whole stash reported, and a CTRL arm (OFF prefilled at a 256-row chunk size, another valid partition with no merge code).
3. Passed: first GDN layer 0.0004 to 0.0034, growing smoothly with depth to 0.02 to 0.05 at layers 12 to 14; whole stash MERGE 0.058 to 0.126 against CTRL 0.06 to 0.14; where CTRL does not change the partition it is bitwise. ASYNC == OFF and MERGE+ASYNC == MERGE bitwise on every stash of all 17 cases, with the turn's and a follow-up turn's greedy tokens equal; merges exactly where the rule predicts; follow-up reuse equal to OFF.

**Prefill** (the ladder; the replay's 2,048-to-4,095-token class under 8-slot load; client time to the first token over the replay):

| measure | OFF / OFF2 | MERGE | MERGE+ASYNC |
| --- | --- | --- | --- |
| prefill, N = 2,807 (ladder, both blocks) | 0.575 s | 0.48 s (−16.5 %) | 0.465 to 0.49 s |
| agent-class prefill under load, mean per block | 584.6 to 594.4 ms | 509.9 / 515.3 ms (−13.3 %) | 491.5 / 493.7 ms (−16.6 %) |
| server time to the first token, median | 590 to 600 ms | 510 to 520 ms | 490 to 500 ms |
| client time to the first token, median | 0.84 to 0.86 s | 0.77 / 0.78 s | 0.75 / 0.76 s |
| client time to the first token, N = 512 minus N = 513 | +108 to +112 ms (a 255-row leftover forward at 512) | +31 to +36 ms | +25 to +41 ms |
| prefill share of the replay's wall time | 29.1 to 29.6 % | 24.2 to 26.0 % | |

Client time to the first token over the replay falls about 10 % in both merge arms.

**Decode, per arm** (replay; `step` = verify steps per decode second = (generated − accepted drafts) / decode seconds, which removes MTP acceptance; `conc` = decode-second-weighted mean number of concurrently decoding requests; common keys = the 455 requests present in all 8 arms):

| arm | per stream, all (t/s) | per stream, common | step, common | acceptance | conc | client TTFT median |
| --- | --- | --- | --- | --- | --- | --- |
| block 1 OFF | 127.4 | 126.5 | 39.00 | 0.870 | 4.61 | 0.845 s |
| block 1 MERGE | 133.2 | 132.0 | 40.28 | 0.880 | 4.42 | 0.767 s |
| block 1 MA | 129.5 | 128.9 | 40.03 | 0.864 | 4.59 | 0.753 s |
| block 1 OFF2 | 126.0 | 125.3 | 38.48 | 0.885 | 4.68 | 0.862 s |
| block 2 OFF2 | 124.4 | 124.1 | 39.33 | 0.837 | 4.55 | 0.854 s |
| block 2 MA | 129.8 | 129.4 | 40.16 | 0.858 | 4.56 | 0.764 s |
| block 2 MERGE | 128.2 | 127.9 | 39.48 | 0.886 | 4.88 | 0.776 s |
| block 2 OFF | 123.8 | 123.5 | 38.38 | 0.876 | 4.76 | 0.841 s |

- The driver's per-stream figures (+5.1 % in block 1, +3.3 % in block 2) mix in concurrency (block 1's MERGE boot decoded at 4.42 concurrent streams against 4.61 and 4.68) and MTP acceptance (block 2's MERGE boot drew 0.886 against an OFF mean of 0.856). They are not the estimate.
- Step rate, the 4 merge-class boots (MERGE and MA) against the 4 OFF-class boots: 39.99 against 38.80, **+3.1 %, 95 % interval +1.3 to +4.9 %** (Welch t on boot means). Per stream on the same boots +3.75 % [+1.6, +5.9], which includes acceptance. MERGE alone (2 boots) +2.8 %, roughly 0 to +6 %.
- The A/A floor, OFF against OFF2 on the same statistic: +1.35 % (block 1) and −2.4 % (block 2). Two blocks resolve +3 % from zero, not +2 % from +4 %.
- The decode-stream seconds with a prefill resident fell from 29.5 to 30.1 % to 26.1 to 27.1 %. R585's dose-response (about 0.7 % decode per point of exposure) predicts about +2.4 %.
- ASYNC against MERGE: step rate +0.55 % [−1.9, +3.3], not resolved; the expected size is under 1 %.

**ASYNC.** On the ladder, the 256 → 257 step (one added last-page stash, no merge at N = 257) costs 7.4 / 8.6 ms of client time to the first token with MERGE+ASYNC, against 16.5 to 21.5 ms in OFF and 15.3 / 24.6 ms in MERGE: about 10 ms per stash, the whole of R804's stash cost. Under load it saves 18 / 22 ms per agent turn (the prefill table). The driver's bar asked for at least 15 ms per stash, more than the stash costs; that bar was mis-set, not failed by the mechanism. Its tier bar was not evaluable: the NVMe prefix tier refused every write in every arm because the fast file system was below its free-space floor.

**Greedy output.** OFF and OFF2 equal R792's greedy reference in all four boots (6 of 6 + 6 of 6). Under MERGE exactly two rows change, identically in all four merge boots: chat `tool` (313-token prompt, 2 forwards → 1) at reasoning character 95 of 261, still ending in the same two parallel `web_search` calls, and chat `long` at about generated token 4,500, after its second requeue at 4,096 re-prefills as a job that merges. `long100k` (120,312-token prompt, 3 forwards → 2) stayed identical over its 128 tokens.

**Health and memory.** 0 OOM, 0 restarts, 0 failed requests in all 8 arms. Each arm's log carries one traceback, after "Shutdown signal called", from TabbyAPI's `unload` calling `torch.cuda.empty_cache()` on a destroyed context, in the OFF arms too; the driver counted it as a failure, and the analyzer now counts tracebacks only before the shutdown line. Boot free VRAM 1,181 / 1,759 MiB in every arm; peak VRAM on the ladder 22 to 26 MiB lower per card with MERGE, and 70 to 110 MiB lower on `cuda:0` over the whole arm.

## R809: quality gate for MERGE + ASYNC on the served stack

Image `tabbyapi:merge-tok-r1` (`ac16920f72cf`) = the R808 daily's contents plus `fix.patch`. Five boots: REF, NULL and CAND for fidelity, CANDB for the behaviour gates and the tier write, CANDR for the restore. Every bar was pre-registered in the driver's header, and an independent review recomputed every number from the raw records and matched the printed values.

**Fidelity against a partition null.** 80 agent-shaped prompts (60 behind a warmed prefix with 300 to 4,000 new tokens and N_p mod 256 between 1 and 255, over the one-, two- and three-forward classes and 4 prompts with a pipelined window before a merged tail; 20 cold prompts of 300 to 3,000 tokens), greedy, 64 tokens with the top 5 logprobs at every position, a fresh boot per arm:

- REF = both keys off, each prompt behind its warmed prefix;
- NULL = both keys off, the prefix warmed 512 tokens earlier (P − 512), or cold: the same served code over a different row partition, which the served configuration already produces when a prompt arrives with another cached prefix;
- CAND = MERGE + ASYNC, behind the same prefix as REF.

| gate | rule | NULL against REF | CAND against REF | result |
| --- | --- | --- | --- | --- |
| prompts diverging within 64 tokens | CAND ≤ NULL + 10 points | 29 of 80 | 34 of 80 (+6.25 points) | pass |
| McNemar, one-sided | p ≥ 0.05 | NULL-only 5 | CAND-only 10 | p 0.151, pass |
| confident flips (REF margin ≥ 1 nat, or the token outside REF's top 5) | CAND ≤ NULL + 2 | 0 | 0 | pass |
| first-token KL (REF ‖ arm), top 5 renormalised, median | ≤ 1.5 × NULL | 0.0037 | 0.0031 | pass |
| first-token KL, p95 | ≤ 1.5 × NULL | 0.046 | 0.023 | pass |
| NULL differs numerically from REF | | 80 of 80 | | the null is not degenerate |

- Every divergence in both arms is a near-tie: REF's margin between its own token and the arm's is at most 0.469 nat in both. In 12 of the 24 prompts that diverge in both arms, NULL and CAND flip at the same position to the same token.
- CAND diverges earlier (median position 3.5 against 10), on first-word near-ties such as "The" / "A" after `<think>`.
- KL at every position where both arms still share REF's context: NULL median 0.00075 (3,764 positions), CAND 0.00066 (3,345). By this measure the merge moves the output by as much as a 512-token change of the served partition, or less.
- Engagement: the merge counter moved by 133 = 80 measured prompts + 53 warm-ups predicted to merge; `merge_no_slab` 0. CAND's requests were faster than REF's on 77 of 80 prompts (median −62 ms), consistent with one forward fewer per merged request.

**Behaviour.**

| gate | rule | result |
| --- | --- | --- |
| tool-eval-bench, 69 scenarios × 4, temperature 0.6, 8 concurrent | ≥ 82.0 | 85.0 (sd 2.4 over the 4 trials), pass |
| GSM8K 5-shot, n = 500, greedy, 8 concurrent, flexible extract | ≥ 0.970 | 0.978 (strict 0.976), pass |
| boot free VRAM | ≥ reference − 32 MiB | 1,181 / 1,759 MiB on every boot, pass |
| minimum free VRAM during the fidelity run | CAND ≥ REF − 32 MiB | REF 563 / 1,099, CAND 577 / 1,133 MiB, pass |
| health | 0 OOM, 0 tracebacks before shutdown | pass |

Seven of GSM8K's 11 wrong answers are wrong in every or most earlier runs; doc 243 (R810b below) answered `#### 25` after computing 250.

**NVMe tier with asynchronous stashes.** CANDB wrote to a scratch tier (8 GB cap, free-space floor 0): 149 pages and 4 checkpoints, 0 errors, 0 write-verify failures, 0 stash mutations. After a crash (`docker rm -f`), CANDR's open scan found 4 of 4 checkpoints intact, and a 30,030-token prompt restored 29,952 cached tokens from disk (117 of 117 pages) in 0.297 s against 2.93 s cold. The output check compared three outputs of one token each (`<|im_end|>`, empty text, the hash of the empty string): the fill prompt's first token is a known near-tie, so it measured nothing ([GOTCHAS 35](../../docs/GOTCHAS.md)). The review rated this gate borderline.

**R809t** re-ran the restore on a scratch tier with a fill prompt that generates at least 32 tokens and a first-token logprob comparison: 117 pages and 3 checkpoints written, 0 errors, open scan 3 of 3 intact; the restored output equals the warm output over 64 tokens (cached 29,952; 0.57 s against 0.36 s warm and 3.73 s cold); first-token top-5 KL 0.0 and largest logprob difference 0.0 (bar ≤ 1e-5). Cold differs from warm (KL 3.1e-4), as expected under MERGE, where the cold prefill merges and the warm one reads a stash. Pass.

## R809p: the promotion (2026-09-29, served from 20:02 UTC)

The launcher differs from R808's in `DAILY_IMG` and the two keys only (41 keys). After the boot: image `ac16920f72cf`, 41 keys, pool 901,120, slots 8, draft policy `[[4, 3], [8, 2]]`, the knob readbacks `tokenize-offloop-r2 1 1 1 12000 0 1` and `prefill-merge-r1 1 1` inside the container, boot free VRAM 1,181 / 1,759 MiB, memory clock offset +4500 on both cards. `fn_greedy` 6 of 6 identical to R792's reference (these prompts do not merge). A new chat greedy reference was recorded: 4 of 6 rows identical to R792's, `tool` and `long` diverging at characters 95 and 12,404, the two rows R803 found changed. After the greedy step the log carried the first-merged-prefill line. Served since 2026-09-29 22:02 CEST. Rollback: `DAILY_IMG=tabbyapi:tokenize-offloop-r2` and `EXTRA_ENV` without the two keys, R808's launcher.

## R810 and R810b: checks after the promotion

- Needles: 5 of 5 at the 131,072 and at the 240,000 settings, 105,680 and 193,464 prompt tokens, the prompts of R792's needle gate, on the promotion boot. These long prompts go through the two-card prefill pipeline and then a merged tail.
- Greedy reference: on a fresh second boot, `fn_greedy` 6 of 6 and `chat_greedy` 6 of 6 identical to R809p's reference, including `long` (5,001 tokens, reasoning only). The merged path is stable across boots.
- GSM8K doc 243: R810's replay probe crashed on `usage: null` and captured nothing. R810b replayed it at 1 stream on the promoted configuration: 3 of 3 replays (on one boot, sharing a 1,280-token cached prefix) return the same text ending `#### 25`, identical to R809's sample. A `/v1/completions` prefill of that text puts `0` at 0.527 against `<|im_end|>` at 0.473 (0.109 nat); that is the prefill path's distribution, the decode path's margin was not measured, and on the same text a different cache split alone moved `<|im_end|>` by 0.18 nat. Of 23 earlier GSM8K runs on this 2.50 bpw checkpoint, 9 (R528 to R579, 2026-09-19) gave the identical `#### 25` and 14 gave `#### 250`. The answer is a near-tie that flips with the stack's numerics and predates the merge.

## Not covered

- Half of the fidelity prompts are uninformative: 41 of 80 REF outputs are `the the the` loops continued from the prompt builder's pad word. On the 39 prompts where REF does not loop, NULL diverges on 25 and CAND on 27 (McNemar p 0.36); REF-coherent prompts that loop in the arm: NULL 3, CAND 7 (sign test p about 0.17). The divergence test saturates on coherent prompts (the null already diverges on 64 % of them); the discriminating evidence is the confident-flip count and the KL, both at or below the null. The pad needs replacing before the probe's next use.
- The prompts with a pipelined window before a merged tail (n = 4) are the one class where CAND's first-token KL sits above the null (median 0.0143 against 0.0034; every flip a near-tie). n = 4 is not a test; R810's needles exercise that path at depth and passed.
- R809 ran the minimum gate: fidelity, engagement, tool-eval and GSM8K, plus the tier check. Greedy roll-over across CAND boots before promotion, the ramp and 26k stress with the VRAM floor after it, the loop gate, the agent replay and agentic edit were not run. CANDB's lowest free VRAM, at the 30k tier fill after the 8-stream phases, was 343 / 873 MiB, ungated and without a comparator arm.
- The decode and prefill figures are R803's, on `tabbyapi:prefill-merge-r1` without `tokenize-offloop-r2`; the promoted image has no speed measurement of its own yet. The decode kernels are unchanged, so a steady-state decode curve on short prompts, which do not merge, is not expected to move.
- The served NVMe prefix tier refuses writes while its file system is below the free-space floor, so asynchronous stashes are not written to or restored from disk in service; the RAM path of asynchronous stashes and merged snapshots is what the served configuration exercises, and CAND's 60 warm fidelity prompts each restored one.
- A prompt whose last page ends exactly on a 256-token boundary is not reused from its deepest stash by the next turn (served behaviour, found by step 0), and MERGE leaves about 30 ms at N = 512 unexplained. Neither is addressed by r1.
