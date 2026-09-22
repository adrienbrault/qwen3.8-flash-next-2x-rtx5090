# R652: batching the per-job MTP accept-prefills gains +4.4 % decode at c8

Results directory on the serving host: `results/2026-09-22-r652-prefbatch-gate`. Raw records: [`2026-09-22-r652-prefbatch-gate/`](2026-09-22-r652-prefbatch-gate/). Image `tabbyapi:prefbatch-r1` = `tabbyapi:mtpnorm-r1` plus the `prefbatch` patch: the accept path's per-job `draft_model.prefill` calls (~6–8 per step at c8/d1, ~24 launches each) are grouped by accepted length and run as one call per group. Under the served `EXL3_MTP_KV_WINDOW` ring a padded single call could evict a still-readable row, so groups run at exact width — worst case equals today's call count. Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

## What was measured

Output equivalence — `bench/fn_greedy.py` reference captured on the running daily (`bverify-r1`), then on `prefbatch-r1`: all 6 prompts byte-identical.

Canonical gate (`bench/fn_gate.sh`, salt 424242, `min_tokens` 1024, 2 recorded runs, 1 warmup per cell) against the `bverify-r1` gate from `results/2026-09-22-r646-verifybatch`. Every row has `finish_reason: length`.

| cell | bverify-r1 decode t/s | prefbatch-r1 | Δ |
|---|---:|---:|---:|
| c1, ~4k ctx | 299.9 | 300.1 | +0.1 % |
| c4, ~4k ctx | 179.5 | 185.0 | +3.1 % |
| c8, ~4k ctx | 83.6 | 87.3 | +4.4 % |
| c4, ~26k ctx | 125.3 | 126.4 | +0.9 % |

Acceptance parity holds (c8: 1.79 vs 1.80 tokens/iterate pooled median).

## Verdict

The first measurable win of the launch-sweep set: ~140 fewer launches per step at c8 shows up as +4.4 % decode there. Folded into `stack-r1` (R653) for promotion.
