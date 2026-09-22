# R653: the launch-sweep stack is promoted — +4.4 % decode at c8, byte-identical output

Results directory on the serving host: `results/2026-09-22-r653-stack-gate`. Raw records: [`2026-09-22-r653-stack-gate/`](2026-09-22-r653-stack-gate/). Image `tabbyapi:stack-r1` = `tabbyapi:mixstate-r1` plus the `prefbatch` patch — three changes on top of `bverify-r1`: the MTP input-norm fusion (R649), the fused int8 state-in-up mixer kernel under `EXL3_GR_STATE_IN_UP=1` (R651), and grouped MTP accept-prefill batching (R652). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

## What was measured

Output equivalence — `bench/fn_greedy.py` reference captured on the running daily (`bverify-r1`), then on `stack-r1`: all 6 prompts byte-identical. Every component preserves greedy output, so the served numerics are unchanged.

Canonical gate (`bench/fn_gate.sh`, salt 424242, `min_tokens` 1024, 2 recorded runs, 1 warmup per cell) against the `bverify-r1` gate from `results/2026-09-22-r646-verifybatch`. Every row has `finish_reason: length`.

| cell | bverify-r1 decode t/s | stack-r1 | Δ |
|---|---:|---:|---:|
| c1, ~4k ctx | 299.9 | 303.6 | +1.2 % |
| c4, ~4k ctx | 179.5 | 186.4 | +3.8 % |
| c8, ~4k ctx | 83.6 | 87.3 | +4.4 % |
| c4, ~26k ctx | 125.3 | 125.2 | −0.1 % |

The +4.4 % at c8 reproduces the `prefbatch-r1` gate's +4.4 % — two independent boots, same number; acceptance parity holds in both.

## Verdict

Promoted 2026-09-22: the daily now runs `tabbyapi:stack-r1` with `EXL3_GR_STATE_IN_UP=1` added to the launcher's default environment (23 keys). Rollback is `IMG=tabbyapi:bverify-r1` minus that flag.
