# R572: what the MTP draft's acceptance rate actually depends on

Results directory on the serving host: `results/2026-09-19-r572-mtp-accept-ablation`. Raw records: [`2026-09-19-r572-mtp-accept-ablation/`](2026-09-19-r572-mtp-accept-ablation/). Driver: [`scripts/r572-mtp-accept-ablation.sh`](../../scripts/r572-mtp-accept-ablation.sh).

Drafts accepted per verification round decide how much the MTP head is worth. On the same prompt, greedy, 2,048 forced tokens at 1 stream, the served daily accepts 1.566 of 3 drafts on code and 1.547 on prose — against 2.26 measured on [the vLLM route](vllm-exl3-route.md), which served a 3.05 bpw checkpoint with BF16 KV. Six boots isolate the reason, using the [R564](r564-draft-topk.md) instrument.

| arm | accepted drafts per round, code | prose | by position, code |
| --- | --- | --- | --- |
| served | 1.566 | 1.547 | 0.738 / 0.504 / 0.326 |
| depth 1 only | 0.772 | 0.746 | 0.773 |
| draft cache FP16 (served is Q8) | 1.550 | 1.544 | 0.727 / 0.499 / 0.326 |
| **target KV FP16 + draft FP16** | **1.771** | **1.833** | 0.770 / 0.566 / 0.440 |

- The draft path is sound: at depth 1 the first position is accepted as often as at depth 3, so nothing degrades across draft positions.
- The draft's own cache precision changes nothing.
- 8-bit KV on the target costs 0.2 to 0.3 accepted drafts per round. Quantised KV moves the target's hidden states and its chosen tokens away from the model the MTP head was trained against, and the deepest draft position loses most: 0.33 against 0.44.

Full-precision KV halves the page pool, so it is a diagnostic here, not an option. Two arms produced no data: with the draft vocabulary unpruned, drafting falls back to a host path that the instrument cannot follow, and with every overlay flag off the image does not boot at the served pool.
