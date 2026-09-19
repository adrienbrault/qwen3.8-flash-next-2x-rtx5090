# R501: prompt lookup beside the MTP draft is lossless and +3 to +4 % on code at c1, flat at c4

Results directory on the serving host: `results/2026-09-18-r501-prompt-lookup`. Raw records: [`2026-09-18-r501-prompt-lookup/`](2026-09-18-r501-prompt-lookup/). Driver: [`scripts/r501-prompt-lookup.sh`](../../scripts/r501-prompt-lookup.sh). Date: 2026-09-18.

Prompt lookup proposes continuations copied from the prompt when the last tokens match a prompt n-gram, next to the MTP draft (after [halogen][halogen] 0.6.0). Image `tabbyapi:prompt-lookup-r1`, `EXL3_PROMPT_LOOKUP=1`, match 3, continuation 3, 3.05bpw pack at 360,448. Served fingerprints canonical on all four arms.

| load (`fn_bench` 2,048 × 2, aggregate) | OFF / OFF2 (t/s) | ON / ON2 (t/s) |
| --- | --- | --- |
| code c1 | 216–221 / 218–221 | **226–231 / 226–228** |
| code c4 | 446–459 / 455–461 | 434–454 / 436–470 |
| prose c1 | 173–180 / 179–180 | 172–179 / 173–177 |
| prose c4 | 435–450 / 444–453 | 428–446 / 431–447 |
| agentic-edit greedy c1 ([`bench/agentic-edit.py`](../agentic-edit.py), 6 files rewritten with a small edit) | 210.3 / 213.2 | 218.7 / 218.4 |

Lookup hit 459 of 1,011 rounds on ON and 642 of 863 on ON2. Not served yet: it is a candidate for a stacked image with the other decode changes.

[halogen]: https://github.com/peonist-ai/halogen
