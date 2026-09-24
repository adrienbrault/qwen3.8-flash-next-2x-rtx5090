# R701: stack-r2 (hcfast r1 + moefast r1), promoted

Results directory on the serving host: `results/2026-09-24-r701-stack-gate`. Raw records: [`2026-09-24-r701-stack-gate/`](2026-09-24-r701-stack-gate/). Driver [`scripts/r701-stack-gate.sh`](../../scripts/r701-stack-gate.sh), run with `MOE_MODE=2`. Image `tabbyapi:stack-r2` = `tabbyapi:slotfix-r1` plus [`hcfast-r1`](../../docker/overlays/hcfast-r1/) plus [`moefast-r1`](../../docker/overlays/moefast-r1/), built in the order given in [`docker/overlays/stack-r2/README.md`](../../docker/overlays/stack-r2/README.md). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, 8 slots, 999,424-token page pool at 8-bit KV, layer split `[30, 30]`, MTP draft component on the second GPU, draft policy `[[4, 3], [5, 2], [8, 1]]`.

This is the first promotion under the stack-track rule in [`docs/PROMOTION.md`](../../docs/PROMOTION.md#the-stack-track-since-2026-09-24): each change is bitwise-identical and flag-gated, each was accepted on identity and on no same-sign regression, and the accepted changes were promoted together through one canonical gate.

## The change

Four environment keys, each default-off in the image:

- `EXL3_HC_MIX_V3=1 EXL3_HC_MIX_V3_DOTS_B=2 EXL3_HC_MIX_V3_UP_B=8`: the hyper-connection boundary mixer's two int8 kernels (dots, up) through `hc_mix_v3.cu`. dots converts each int8 weight once per iteration instead of once per row and batches its loads; up reduces with a 35-shuffle reduce-scatter instead of a 160-shuffle butterfly. Same launches, same grids, same arithmetic order per output ([R698, R699](r698-hcfast.md)).
- `EXL3_MOE_COOP_V3=2`: the routed-expert MoE decode kernels with a cp.async weight ring, a one-slice activation prefetch, one acquire/release counter arrival per item instead of the block-wide fence pairs, and the narrow down stage merged per 128-column chunk ([R700b](r700b-moefast.md)).

## What was measured

2026-09-24 07:56 to 08:45 UTC, one GPU session. The image was built inside the GPU lock before any measurement (07:56 to 08:04); both landing markers were present (`landed.txt`).

**Part 1, in-process harness.** 4,096 tokens of context, untraced, CUDA-event timing of 32 captured iterates, ms per iterate. Five rounds per shape, each round running OFF (the served flags), HC (plus the hcfast keys) and ST (plus all four keys) in a rotated order. Values are the arm minus the same round's OFF, in ms per iterate.

| shape | arm | round 1 | round 2 | round 3 | round 4 | round 5 | mean | sign |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 streams, depth 3 | HC | −1.38 | −2.24 | −1.41 | −0.21 | −0.40 | −1.13 (−5.2 %) | 5 of 5 negative |
| 4 streams, depth 3 | ST | −1.89 | −0.84 | −1.30 | −1.77 | −1.34 | −1.43 (−6.5 %) | 5 of 5 negative |
| 8 streams, depth 1 | HC | −0.12 | −0.42 | −0.58 | +0.09 | −0.65 | −0.34 (−1.7 %) | mixed |
| 8 streams, depth 1 | ST | −0.70 | +0.17 | −1.18 | −0.82 | −1.35 | −0.78 (−4.0 %) | mixed, 4 of 5 negative |
| 1 stream, depth 3 | HC | −0.70 | −0.64 | −1.48 | +0.27 | −0.08 | −0.53 (−3.9 %) | mixed |
| 1 stream, depth 3 | ST | −1.66 | −1.48 | −1.43 | −2.01 | −1.22 | −1.56 (−11.6 %) | 5 of 5 negative |

The generated-sequence hashes of HC and ST equalled the round's OFF in all 30 comparisons. No shape shows a same-sign regression. R700b's reading of +1.5 % at 1 stream for moefast alone, from 2 pairs, is not reproduced: the stack is faster than OFF at 1 stream in all 5 rounds.

**Part 2, served canonical gate.** Three alternating pairs of boots of the served launcher: OFF is the launcher of that date (`tabbyapi:slotfix-r1`, 23 keys), ON adds `IMG=tabbyapi:stack-r2` and the four keys. Each boot ran the greedy identity set (6 prompts, one of them about 100,000 tokens), then [`bench/fn_gate.sh`](../fn_gate.sh) with 3 recorded rounds: leg A is prose at about 4,000 tokens of context at 1, 4 and 8 streams, leg B is prose at about 26,000 tokens of context at 4 streams, both with 1,024 forced tokens, greedy. Values are the median per-request decode rate in tokens/s per stream, and the ON/OFF ratio.

| cell | pair 1 | pair 2 | pair 3 | mean ON/OFF |
| --- | --- | --- | --- | ---: |
| leg A, prose, 1 stream | 297.4 → 324.9 | 301.2 → 328.3 | 195.1 → 213.9 | 1.093 |
| leg A, prose, 4 streams | 159.9 → 169.1 | 157.4 → 168.8 | 162.7 → 169.6 | 1.057 |
| leg A, prose, 8 streams | 92.4 → 98.4 | 88.2 → 93.4 | 101.8 → 104.7 | 1.051 |
| leg B, prose, 26k context, 4 streams | 126.3 → 134.3 | 128.5 → 136.7 | 125.8 → 133.9 | 1.064 |

The mean over the four cells is 1.066. Every one of the 12 per-pair ratios is above 1.02.

| check | OFF | ON |
| --- | --- | --- |
| greedy output against the first OFF boot | identical, 6 of 6 | identical, 6 of 6, in all three boots |
| leg A rows, leg B rows | 39, 12 of 12 in every boot | 39, 12 of 12 in every boot |
| CUDA out-of-memory lines, tracebacks | 0, 0 | 0, 0 |
| free VRAM after boot, GPU 0 / GPU 1 | 1,041 / 2,429 MiB | 1,041 / 2,431 MiB |
| free VRAM after the gate, GPU 0 / GPU 1 | 121-131 / 1,603-1,645 MiB | 193-199 / 1,671-1,713 MiB |

The per-round numbers are in [`audit.txt`](2026-09-24-r701-stack-gate/audit.txt), the per-request records in `gate-<arm>/bench-A.jsonl` and `bench-B.jsonl`, and the greedy texts in [`greedy.jsonl`](2026-09-24-r701-stack-gate/greedy.jsonl).

The decision rule was written into the driver before the run: identity in every P1 round and every ON boot; no P1 shape with all five deltas positive; no served cell with a three-pair mean below 0.99; a positive mean over the cells; 0 out-of-memory lines, leg B complete in every boot, and GPU 0 free after the gate no lower than the OFF boots' minimum less 32 MiB. Every condition held.

## Verdict

Promoted 2026-09-24 08:46 UTC (10:46 CEST): the launcher's `DAILY_IMG` is `tabbyapi:stack-r2` and its default environment carries 27 keys ([`promotion-boot.txt`](2026-09-24-r701-stack-gate/promotion-boot.txt): the four keys present, free VRAM 1,041 / 2,431 MiB). Rollback is `IMG=tabbyapi:slotfix-r1` with the four keys removed from `EXTRA_ENV`.
