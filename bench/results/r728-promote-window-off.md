# R721, R722, R723, R728: the windowed MTP draft cache is turned off, promoted at an unchanged 983,040-token pool; a prompt revived from the prompt cache drafts 0.655 accepted per proposed token with the window and 0.868 without it

Results directories on the serving host: `results/2026-09-25-r721-draft-revive`, `results/2026-09-25-r722-window-agent-replay`, `results/2026-09-25-r723-window-off-fit` and `results/2026-09-25-r728-promote-window-off`. Raw records: [`2026-09-25-r721-draft-revive/`](2026-09-25-r721-draft-revive/), [`2026-09-25-r722-window-agent-replay/`](2026-09-25-r722-window-agent-replay/), [`2026-09-25-r723-window-off-fit/`](2026-09-25-r723-window-off-fit/), [`2026-09-25-r728-promote-window-off/`](2026-09-25-r728-promote-window-off/) (the boot, container and environment logs, the tool-eval JSON and the GSM8K per-sample file stay on the host; queue lines naming unpublished units are replaced by `(unit not published)` in the audit copies). Drivers [`scripts/r721-draft-revive.sh`](../../scripts/r721-draft-revive.sh), [`scripts/r722-window-agent-replay.sh`](../../scripts/r722-window-agent-replay.sh), [`scripts/r723-window-off-fit.sh`](../../scripts/r723-window-off-fit.sh) and [`scripts/r728-promote-window-off.sh`](../../scripts/r728-promote-window-off.sh). Image `tabbyapi:stack-r3-rows32`, model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, 8 slots, 8-bit KV, layer split `[30, 30]`, MTP draft component on the second GPU, draft policy `[[4, 3], [8, 2]]`.

## The change

One launcher key: `EXL3_MTP_KV_WINDOW=16384` leaves `EXTRA_ENV`, so the served set has 41 keys instead of 42. Image, page pool (983,040 tokens) and draft policy are unchanged. With the key set ([R579](r579-promote-mtp-kv-window.md), served from 2026-09-20), the MTP draft layer kept its K/V in a per-slot ring of a sink page plus the last 16,384 tokens. Without it the draft K/V lives in the page-indexed draft cache, which covers the whole page pool and is revived with the target's pages when a prompt prefix is found in the prompt cache. The draft component has been on cuda:1 since [R694](r694-mtp-card1.md), so the larger draft cache costs cuda:1 only.

## R721: the window and revived prompts

2026-09-25 00:54 to 01:03 UTC, four boots in the order W1 N1 N2 W2. W is the served launcher at 983,040; N is the same 41 keys without the window at 950,272, the first pool rung that booted with the window off in that unit (R723 later booted it at 983,040). Each boot ran `fn_bench` code at 4, 6 and 8 streams on 4k-token filler prompts (5,365 to 5,661 prompt tokens) with one salt, then 8 streams on a second salt never sent before; greedy, 1,024 forced tokens, one warm-up and two recorded rounds per cell. Because the stream prompts depend on the stream index and not on the concurrency, streams 0 to 5 of the 8-stream cell revive prompts that the 4- and 6-stream cells prefilled.

Draft acceptance, mean accepted/proposed per request in the recorded rounds of the 8-stream cells, from the server's request log (`requests-*.jsonl`):

| boot | streams 0-5, revived | 8-stream cell, all | second salt, never seen | per-stream decode, 8-stream cell | per-stream decode, second salt |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1 | 0.655 | 0.719 | 0.911 | 111.4 | 128.9 |
| N1 | 0.867 | 0.878 | 0.909 | 124.9 | 130.3 |
| N2 | 0.868 | 0.877 | 0.910 | 125.4 | 131.5 |
| W2 | 0.653 | 0.717 | 0.911 | 111.0 | 129.2 |

Per-stream rates are the server's per-request rate in tokens per second ([`summary.txt`](2026-09-25-r721-draft-revive/summary.txt)). On prompts never seen before the two arms accept alike, and the arm means of the per-stream rate are 1.4 % apart. On revived prompts the window arm accepts 0.21 fewer drafted tokens per proposed token, and the arm without the window decodes 12.5 % faster per stream in the 8-stream cell. The step rate per stream is 44.8 to 46.5 steps per second in both arms; the difference is in tokens per step. A prompt revived inside the same request group keeps its draft K/V under the window; a prompt revived in a later request group does not.

## R722: agent-shaped replay, window on and off

2026-09-25 01:06 to 02:15 UTC, boots W1 N1 W2 N2 (W at 983,040 with the window, N at 950,272 without it), each running [`bench/agent_replay.py`](../agent_replay.py) for 600 s (`--prod-ladder --respawn --drain --gen 700 --tool 2020 --think 11 --stagger 3 --temp 0.6 --seed 722`), scored from the server log by [`bench/tabby_log_agg.py`](../tabby_log_agg.py) over the replay window. All four arms ran the same session steps at the same forced lengths (the seed fixes them); prompts had a median of 28.8k tokens and a 90th percentile of 66k, with 91 % of each prompt cached at the median.

| boot | per-stream decode, t/s | decode aggregate, t/s | MTP acceptance |
| --- | ---: | ---: | ---: |
| W1 | 123.3 | 414.5 | 87 % |
| N1 | 125.8 | 413.5 | 87 % |
| W2 | 122.2 | 415.3 | 86 % |
| N2 | 124.7 | 428.7 | 89 % |

Window off is +2.0 % per stream in both pairs. The aggregate here is tokens over the wall seconds with at least one stream decoding ([`agg-*.txt`](2026-09-25-r722-window-agent-replay/)). 27 to 74 replies per arm ended before their forced length on the server's loop detector (`gen_tokens` below `asked_tokens` in `replay-*.jsonl`), which alone moves an arm by several percent. This round does not resolve how much of the +2.0 % comes from draft acceptance and how much from the cost of draft attention over the full cache.

## R723: fit at the served pool

2026-09-25 02:15 to 02:21 UTC, boots D (the served launcher), N0 (window off at 983,040) and N1 (window off at 999,424), each running the six-prompt greedy capture (including a ~100k-token prompt), a 1-to-8-stream ramp and the canonical `fn_gate` once.

| boot | pool | free MiB at boot, cuda:0 / cuda:1 | free MiB after the sequence | greedy against the reference |
| --- | ---: | --- | --- | --- |
| D | 983,040 | 1,125 / 2,513 | 229 / 1,701 | 6 of 6 identical |
| N0 | 983,040 | 1,125 / 1,573 | 229 / 763 | 6 of 6 identical |
| N1 | 999,424 | 985 / 1,433 | 107 / 623 | 6 of 6 identical |

At 983,040 cuda:0 is unchanged and the layer placement is the same (cuda:0 holds layers 0 to 25); cuda:1 has 940 MiB less, the draft cache growing from 133,120 to 983,040 tokens. N1 leaves cuda:0 140 MiB below the served configuration and was not proposed. Gate cells, prose, one run each, D against N0, per-stream decode: 4 streams at ~3,100 tokens 153.2 against 154.6, 8 streams 103.2 against 105.2, 4 streams at ~19,600 tokens 144.8 against 151.9 ([`audit.txt`](2026-09-25-r723-window-off-fit/audit.txt)). The unit printed `DECISION: TIGHT` because its cuda:1 clause (free after the sequence ≥ the served configuration's minus 32 MiB) fails for any change that moves memory onto cuda:1 by design; the sequence drew 812 and 810 MiB from cuda:1 in D and N0. R728 replaced the clause for this promotion: cuda:1 draw during the sequence within 32 MiB of the served configuration's, and at least 500 MiB free on cuda:1 afterwards.

## R728: the promotion gates

2026-09-25 03:28 to 04:11 UTC, one unit. REF is the served launcher with the window, CAND the candidate launcher, CANDT the candidate with the NVMe prefix tier on. Pre-registered gates, each against REF on this unit's boots ([`summary.txt`](2026-09-25-r728-promote-window-off/summary.txt)):

| gate | result |
| --- | --- |
| G1 greedy | CAND 6 of 6 identical to REF, including the ~100k-token prompt; CANDT 6 of 6 identical as well |
| G2 ramp | 36 of 36 requests ok at 1 to 8 streams, 0 out-of-memory, `TORCH_CHECK` or traceback lines |
| G3 stress | 4 of 4 at 4 streams and 8 of 8 at 8 streams on ~19,600-token prompts, 0 out-of-memory lines |
| G4 decode | canonical `fn_gate`, prose, 2 runs, per-stream decode CAND / REF: 1.003 at 4 streams and ~3,100 tokens, 1.217 at 8 streams, 1.209 at 4 streams and ~19,600 tokens; 0.976 at 1 stream, reported and not gated |
| G5 headroom | free MiB at boot / after the sequence: REF 1,125 / 2,513 → 245 / 1,677, CAND 1,125 / 1,573 → 249 / 779; cuda:1 draw 836 (REF) and 794 (CAND) |
| G7 replay | R722's replay on the CAND boot: 128.6 t/s per stream against a bar of 120.3 (0.98 × R722's window-arm mean), 0 errors, 62 loop-detected stops |
| G6 tier | CANDT boots, greedy 6 records, ramp 36 of 36, 4 of 4 at ~19,600 tokens, 0 errors |

The live launcher was swapped for the candidate at 03:52 UTC, and the post-promotion gates ran on the promoted daily:

| gate | result |
| --- | --- |
| P1 agentic-edit | 6 of 6 ok in each of greedy and sampled at 1 and 4 streams |
| P2 needles | 5 of 5 at 105,680 prompt tokens and 5 of 5 at 193,464 |
| P3 tool-eval, 69 × 4 at parallel 8 | 87.8 ± 1.7 (bar 82.0) |
| P4 GSM8K 5-shot, n = 500, no stop strings | 0.974 flexible-extract, 0.972 strict (bar 0.970) |

The whole run logged 0 out-of-memory, `TORCH_CHECK` or traceback lines.

Two G4 cells need their conditions stated. `fn_gate` runs its cells in the order 1, 4, 8 streams with one salt, so streams 0 to 3 of the 8-stream cell revive the 4-stream cell's prompts, and the ~19,600-token cell's measured runs revive their own warm-up. Under the window those revived streams drafted 2.27 and 2.79 tokens per step, without it 2.91 and 3.32. The 1.217 and 1.209 ratios measure revived prompts and are an upper bound for fresh ones; on prompts never seen before, R721's pairs read 1.011 and 1.018. The effect on agent traffic is R722's +2.0 % per stream.

G6 shows that the tier-on boot is clean and no more: the NVMe tier refused every write during the run (21,409 refusals) because the volume that holds it had less free space than the tier's 15 % floor, so no page went to disk.

Rollback: add `EXL3_MTP_KV_WINDOW=16384` back to `EXTRA_ENV` in [`scripts/launch-flashnext.sh`](../../scripts/launch-flashnext.sh).
