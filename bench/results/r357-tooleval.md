# R357: Quality: tool-eval 69×4

Results directory on the serving host: `results/2026-09-16-r357-tooleval`. Raw records: [`2026-09-16-r357-tooleval/`](2026-09-16-r357-tooleval/). Driver: [`scripts/r357-tooleval.sh`](../../scripts/r357-tooleval.sh). Date: 2026-09-16.


| check | result |
| --- | --- |
| **tool-eval 69×4, baseline** | **85.0 ± 2.9**, CI [82.5, 87.5], per-trial points [113, 115, 120, 121]; losses in categories G 5/6, H 8/10, I 16/20, K 19/26, N 5/6, O 10/12 |
| **tool-eval 69×4, promoted config** | **85.8 ± 3.1**, CI [83.5, 88.5], points [115, 118, 116, 124]; losses in G 5/6, H 8/10, I 16/20, K 21/26, M 5/6, N 5/6 |

**The promoted configuration does not cost quality.** 85.8 against 85.0 with overlapping intervals, on a
tool-calling benchmark, while measuring +35 % at short-context c4 and +78 % at deep-context c4. The greedy
byte-equality gates establish that the *decoding* is unchanged; this establishes that the *task behaviour* is.

Invocation: `tool-eval-bench --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8`. The harness sends
its own sampler parameters, so the server's preset fallbacks are not part of this measurement on either arm, which
is what makes the two arms comparable.
