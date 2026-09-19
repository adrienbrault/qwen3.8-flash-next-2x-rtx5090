# R564: two draft chains do not pay, and the draft head's confidence is well calibrated

Results directory on the serving host: `results/2026-09-19-r564-draft-topk`. Raw records: [`2026-09-19-r564-draft-topk/`](2026-09-19-r564-draft-topk/). Driver: [`scripts/r564-draft-topk.sh`](../../scripts/r564-draft-topk.sh).

An instrument, not a change: with `EXL3_DRAFT_TOPK_STATS=1` the engine records, for every MTP draft position, the draft head's top four tokens and where the target's token ranked when the draft was rejected. Output is byte-identical with the flag on, and both boots carried the canonical fingerprints.

## Acceptance by draft position

24 code and 24 prose prompts at 1 stream, 512 forced tokens, depth 3.

| kind | rounds | accepted / rank 2 / rank 3 / rank 4 / miss, position 0 | position 1 | position 2 |
| --- | --- | --- | --- | --- |
| code | 4,728 | 3415 / 564 / 219 / 107 / 423 | 2433 / 358 / 158 / 76 / 382 | 1726 / 243 / 107 / 54 / 300 |
| prose | 4,914 | 3578 / 509 / 186 / 117 / 524 | 2339 / 393 / 166 / 98 / 574 | 1473 / 263 / 101 / 62 / 437 |

Calibration is good: in the top confidence bin the first draft position was accepted 1,550 times against 30 rejections on code, and 2,202 against 109 on prose, while the 0.1–0.2 bins are mostly rejected (13 accepted against 79 rejected at position 2 on code).

## What the estimator says

`estimate_two_chain.py` prices each alternative against the served step cost (4 verify rows 13.91 ms, 8 rows 16.45 ms):

| alternative | expected tokens per step | verify rows | modelled throughput, code | prose |
| --- | --- | --- | --- | --- |
| served, one chain depth 3 | 2.604 | 4 | — | — |
| two chains, branch at position 1 | 2.735 (+5.0 %) | 8 | −11.2 % | −11.0 % |
| two chains, branch at position 0 | 2.773 (+6.5 %) | 8 | −10.0 % | −12.7 % |
| one chain depth 4 (extrapolated) | 2.865 (+10.0 %) | 5 | +5.2 % | +2.9 % |

Doubling the verify rows costs more than the extra accepted tokens return, so branching drafts are closed. The depth-4 row of the model is optimistic: measured at 1 stream, fixed depth 4 came out at −0.8 % on code and −2.4 % on prose ([R567](r567-adaptive-draft-r3.md)).
