# R496: GDN state replay serves a 425,984-token pool on the 3.05 pack with canonical output, prose decode −7 to −8 %

Results directory on the serving host: `results/2026-09-18-r496-gdn-state-r3`. Raw records: [`2026-09-18-r496-gdn-state-r3/`](2026-09-18-r496-gdn-state-r3/). Driver: [`scripts/r496-gdn-state-r3.sh`](../../scripts/r496-gdn-state-r3.sh). Date: 2026-09-18.

The patch (image `tabbyapi:gdn-state-r3`) places the GDN recurrent state so that its reservation can be released to the page pool. It was built on the image before the shared-expert overlap. Identity gates: kernel test pass, served placement identical (53 modules, 26 / 26 plus CPU 1), 669,874,176 bytes per card released by the placement lock, both arms canonical. Ladder: 458,752 does not boot; **425,984 boots** with the same placement and canonical fingerprints.

| | ON at 425,984 | OFF2 at 360,448 |
| --- | --- | --- |
| code c1 / c4 aggregate (`fn_bench`, mean of 2) | 219.9 / 434.9 t/s | 216.6 / 433.7 |
| prose c1 / c4 aggregate | **156.8 / 396.2** | 170.3 / 427.3 |
| multiprompt code c4 / prose c4 aggregate | 388.5 / 357.4 | 365.4 / 342.1 |
| prefill 22.6k / 90.1k | 7,580 / 11,070 t/s | 7,460 / 11,008 |
| needles, GSM8K n=200 | 5/5, 0.925 | — |

Greedy tokens are identical, so the prose gap is per-step cost of the replay or noise. Not promoted; the 2.50bpw pack's 786,432-token pool superseded the pool lever the same evening.
