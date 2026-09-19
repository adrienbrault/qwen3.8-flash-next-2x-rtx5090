# R340: Concurrency-indexed draft depth: the A/B

Results directory on the serving host: `results/2026-09-16-r340-ci-depth`. Raw records: [`2026-09-16-r340-ci-depth/`](2026-09-16-r340-ci-depth/). Driver: [`scripts/r340-ci-depth.sh`](../../scripts/r340-ci-depth.sh). Date: 2026-09-16.


Three arms, one variable each, 2,048 forced code tokens, greedy, chat endpoint. Control is the served baseline;
parity is the patched engine with the policy unset; treatment is the patched engine with
`draft_num_tokens_by_batch: [[2, 3], [8, 1]]` — depth 3 up to two decoding jobs, depth 1 above.

| arm | c1 decode | c4 decode/stream | c4 aggregate | c8 decode/stream | c8 aggregate |
| --- | --- | --- | --- | --- | --- |
| control (unpatched) | 209.2–213.2 | 64.4–64.8 | 252–258 † | 40.8–42.6 | 316–320 |
| parity (patched, no policy) | 209.2–211.6 | 64.3–65.7 | 252–258 | 40.8–42.6 | 316–320 |
| **treatment (policy on)** | 209.2–214.1 | **87.1–89.4** | **338–347** | 39.6 | 311 |

† the control rows carry two attempts' records in the same file (an aborted first run and this one), so
`bench/summarize.py` prints AMBIG rather than a sum of both; the decode figures are per request and unaffected.

**+35 % aggregate at c4** from one load-time policy, with no change at c1 (where it deliberately keeps depth 3) and
no gain at c8. The correctness gate passed on all three arms: the greedy capture — reasoning and content together,
2,003 bytes — is byte-identical (`sha256` `95726ace17d5…`) across control, parity and treatment, so the patch alone
changes nothing and enabling the policy does not change what the model says.

Two requests across the arms stopped before the forced length (`finish_reason: stop` — the engine's loop detector);
they are flagged and excluded from the rates above.
