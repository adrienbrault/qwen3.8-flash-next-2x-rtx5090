# tool-eval-bench on ExLlamaV3 v1.6.7 r2

Measured 2026-11-26 on one served boot of `tabbyapi:rebase167-r2` (`96ab0d09`), source `20261126T011043Z-spec-r1-flashnext`. Each arm uses the default name, 69 scenarios × 4 repeats, c8, temperature 0.6, top-p 0.95 and top-k 20. Scores are mean ± SD across repeats. The [echo shim](../../scripts/tooleval_echo.py) changes reasoning replay in arms B and D.

| arm | harness and history | mean ± SD | median | pass@k | pass^k | observed echo |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A | 2.1.0, stock | 89.0 ± 2.3 | 89.0 | 88.4 | 78.3 | no reasoning sent back |
| B | 2.1.0, tool-call turns | 87.8 ± 1.5 | 87.0 | — | 72.5 | 434/434 tool-call turns, 0/328 other turns |
| D | 2.1.0, every assistant turn | 88.8 ± 1.0 | 88.5 | — | 73.9 | 432/432 tool-call turns, 328/328 other turns |
| C | 3.0.0, stock | 93.5 ± 2.1 | 93.5 | 95.7 | 84.1 | tool-call reasoning replay built into the harness |

The same-boot live reasoning-history probe passed on the serving frontend: with tools, the prompt grew from 326 to 657 tokens when `reasoning_content` was replayed. A/B/D differ in mean by at most 1.2 points, within run-to-run variation in this test. The 3.0.0 score uses the revised scorer and is a separate series. The echo controls show no change beyond noise from reasoning replay in these scenarios.
