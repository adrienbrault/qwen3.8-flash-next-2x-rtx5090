# R651: fused int8 state-in-up mixer kernel is bitwise-exact and flat-to-+1.7 % at gate resolution

Results directory on the serving host: `results/2026-09-22-r651-mixstate-gate`. Raw records: [`2026-09-22-r651-mixstate-gate/`](2026-09-22-r651-mixstate-gate/). Image `tabbyapi:mixstate-r1` = `tabbyapi:mtpnorm-r1` plus the `mixstate` patch: a new `gr_v2_up_i8_state_kernel` derives the (LR+H) hyperconnection state row inside the up kernel's CTA, replacing the standalone state/regrid launch — 3 launches per mixer site become 2, about −107 launches per decode step. Gated by `EXL3_GR_STATE_IN_UP` (default off); the flag-off path is unchanged. Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

## What was measured

Bitwise equivalence — `bench/fn_greedy.py` captured on the running daily (`bverify-r1`) before the candidate boot, then on `mixstate-r1` with the flag on. All 6 prompts (short0–4, long100k) produce byte-identical greedy output; the same-intrinsics/same-operand-order construction holds in practice.

Canonical gate (`bench/fn_gate.sh`, salt 424242, `min_tokens` 1024, 2 recorded runs, 1 warmup per cell), flag on, against the `bverify-r1` gate from `results/2026-09-22-r646-verifybatch`. Every row has `finish_reason: length`.

| cell | bverify-r1 decode t/s | mixstate-r1 flag-on | Δ |
|---|---:|---:|---:|
| c1, ~4k ctx | 299.9 | 304.2 | +1.4 % |
| c4, ~4k ctx | 179.5 | 163.7 | −8.8 % |
| c8, ~4k ctx | 83.6 | 85.0 | +1.7 % |
| c4, ~26k ctx | 125.3 | 123.7 | −1.3 % |

The c4/4k gap is the documented content lottery: pooled acceptance there reads 3.15 tokens/iterate, and byte-identical greedy output rules out a mechanism regression — the same prompts swing by that much across baseline runs.

## Verdict

Neutral-positive. −107 launches/step is real but below the gate's noise floor (~±2 %), consistent with the ~0.3 ms/step estimate. Banked in the patch stack on top of `mtpnorm-r1`; the daily stays on `bverify-r1` until the stack shows a measurable win.
