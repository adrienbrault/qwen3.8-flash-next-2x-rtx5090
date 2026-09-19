# R354: Both levers together

Results directory on the serving host: `results/2026-09-16-r354-combined`. Raw records: [`2026-09-16-r354-combined/`](2026-09-16-r354-combined/). Driver: [`scripts/r354-combined.sh`](../../scripts/r354-combined.sh). Date: 2026-09-16.


Image `tabbyapi:qsa-cid` (QSA multi-job + concurrency-indexed draft depth, built from the same devel base, same
pip resolution, same native rebuild) against the served baseline, with `DRAFT_POLICY='[[2, 3], [8, 1]]'`.

| shape | baseline | combined | change |
| --- | --- | --- | --- |
| short context, c1 | 207.9 | 207.3 | unchanged, by design |
| short context, c4 | 64.0 / 251.1 | **87.3 / 338.0** | **+35 %** |
| short context, c8 | 39.7 / 307.5 | 39.6 / 310.7 | unchanged |
| deep context, c2 | 99.2 | 125.5 | **+27 %** |
| **deep context, c4** | 51.4 / 190.4 | **91.4 / 323.6** | **+78 %** |

The two levers compose and the deep-context c4 cell exceeds either alone (QSA alone read 72.5 there, the draft
policy alone is measured at short context): at four concurrent deep-context jobs, the attention path stops falling
back to eager *and* the drafter stops paying for depth 3. Output is byte-identical to the baseline
(`sha256` `750e1459e177c47e…`, 1,989 bytes).

That is the configuration to serve: **+35 % to +78 % at the concurrency a fan-out of agents reaches, with
byte-identical output.**
