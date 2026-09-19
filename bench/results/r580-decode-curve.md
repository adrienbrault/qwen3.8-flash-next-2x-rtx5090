# R580: the whole decode curve and cold prefill to the top of the window, on one boot

Results directory on the serving host: `results/2026-09-20-r580-decode-curve-try2`. Raw records: [`2026-09-20-r580-decode-curve-try2/`](2026-09-20-r580-decode-curve-try2/). Driver: [`scripts/r580-decode-curve.sh`](../../scripts/r580-decode-curve.sh).

Every decode row in the README used to be stitched from several rounds — one stream from one round's `mp_decode`, 4 to 8 streams from two others' `fn_bench` arms — and 2 and 3 streams had never been measured at all. This round reads all of it with one instrument on one boot of the served launcher: greedy, 1,024 forced tokens, a warm-up round plus three recorded rounds per shape, NVMe tier off.

## Decode, 1 to 8 streams

| streams | code, aggregate | code, per stream | prose, aggregate | prose, per stream |
| --- | --- | --- | --- | --- |
| 1 | 213.2 | 213.2 | 191.5 | 191.5 |
| 2 | 361.1 | 180.6 | 337.8 | 168.9 |
| 3 | 399.6 | 133.7 | 403.4 | 137.2 |
| 4 | 515.5 | 133.0 | 495.3 | 124.5 |
| 5 | 550.9 | 110.7 | 537.1 | 109.1 |
| 6 | 506.6 | 84.6 | 508.2 | 85.5 |
| 7 | 595.1 | 85.6 | 605.0 | 87.5 |
| 8 | 626.5 | 79.9 | 630.3 | 79.0 |

The aggregate curve is not monotone: it dips at 6 streams, where the draft policy `[[4, 3], [5, 2], [8, 1]]` drops to one draft token, because 6 jobs at depth 2 would need 18 verify rows and the fast cooperative MoE decode kernels take 16 ([R560](r560-c8-policy.md), [R562](r562-profile-c8.md)). Before [R576](r576-promote-c5-policy.md) the same dip sat at 5 streams; drafting 2 tokens there moved it one place right. Per-stream rate is flat from 6 to 8 streams, so past the dip the cards are not the constraint.

Code decodes faster than prose at one stream (213 against 192) and the gap closes as concurrency rises, which is what draft acceptance does: it tracks how predictable the text is, and at high concurrency the draft depth is the same one token for both.

## Cold prefill at true prompt lengths

`fn_bench`'s `--ctx` is a filler budget, not a token count — the salted passage is `ctx / 1.6` words, which lands near 0.75 tokens per budget unit. Earlier prefill rows were therefore labelled with the budget: the published "120k" point was really 90,008 tokens, and nothing deeper had been measured cold. This round calibrated the ratio on the boot from the server's own usage counter (60,000 budget → 45,015 tokens, 0.7502 per unit) and then asked for the budget that lands on each target. Three salted runs each, tier off.

| prompt tokens counted | prefill rate | time to first token |
| --- | --- | --- |
| 30,133 | 9,706 t/s | 3.10 s |
| 60,014 | 10,381 t/s | 5.78 s |
| 120,075 | 10,538 t/s | 11.39 s |
| 199,844 | 10,636 t/s | 18.79 s |
| 240,047 | 10,543 t/s | 22.77 s |

Every target landed within 0.2 % of its mark. Prefill rate rises from 30k to 60k and is then flat to the top of the 262,144-token window: a 199,844-token prompt prefills in 18.8 s, and 240,047 tokens in 22.8 s. Depth costs latency, not rate.

Free VRAM at the end of the round was 245 / 2,003 MiB, matching the post-ramp figures in [R581](r581-split-rebalance.md).
