# R503 and R509: GSM8K on this checkpoint was measuring lm-eval's stop strings; without them 3.05bpw and 2.50bpw score 0.980 and 0.978

Results directory on the serving host: `results/2026-09-18-r509-gsm8k-nostop`. Raw records: [`2026-09-18-r509-gsm8k-nostop/`](2026-09-18-r509-gsm8k-nostop/). Driver: [`scripts/r503-gsm8k-2p50.sh`](../../scripts/r503-gsm8k-2p50.sh), [`scripts/r509-gsm8k-nostop.sh`](../../scripts/r509-gsm8k-nostop.sh). Date: 2026-09-18.

R503 (results `2026-09-18-r503-gsm8k-2p50`): 5-shot GSM8K, n=500, chat template, temperature 0, `max_gen_toks` 8192, c4. 3.05bpw: 0.914; 2.50bpw: 0.804. Most wrong answers were empty: 36 of 44 on 3.05bpw, 89 of 98 on 2.50bpw. The model's reasoning restates the problem as "Question: …", and TabbyAPI applies lm-eval's request-level stop list (`Question:`, `</s>`, `<|im_end|>`) to the reasoning text, so the reply ends mid-thought with empty content. Replayed without the stop list, the same reasoning reaches the right answer. On the 2.50bpw pack's live log, 65 of 354 requests ended on "Question:" at a median of 23 tokens.

R509: the same run through [`bench/nostop_proxy.py`](../nostop_proxy.py), which drops the request's `stop` field.

| pack | flexible | strict | wrong (empty) |
| --- | --- | --- | --- |
| 3.05bpw at 360,448 | **0.980** (SE 0.0063) | 0.978 | 10 (2) |
| 2.50bpw at 786,432 | **0.978** (SE 0.0066) | 0.978 | 11 (1) |

Paired per question: wrong on both 7, only 3.05bpw 3, only 2.50bpw 4. Every GSM8K figure this track published before 2026-09-18 evening (0.925 / 0.935 as served, 0.815 / 0.804 on 2.50bpw, 0.765 on 2.05bpw) undercounts by the share of replies whose reasoning restates "Question:", about 7 % of questions on 3.05bpw and 18 % on 2.50bpw. GSM8K gates since then run through the proxy.
