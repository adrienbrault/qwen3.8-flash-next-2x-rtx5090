# R649: MTP input-norm fusion is decode-neutral at gate resolution; acceptance parity proven via tokens-per-frame

Results directory on the serving host: `results/2026-09-22-r649-mtpnorm-gate`. Raw records: [`2026-09-22-r649-mtpnorm-gate/`](2026-09-22-r649-mtpnorm-gate/). Image `tabbyapi:mtpnorm-r1` = `tabbyapi:bverify-r1` plus the MTP input-layer patch: the ~11-launch grouped RMS-normalization chain in `qwen4_exp_mtp.py` is replaced by one `ext.rms_norm` call plus a widening `add_`, removing ~55–65 launches per decode step on the draft side. Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

## What was measured

The canonical gate (`bench/fn_gate.sh`, salt 424242, `min_tokens` 1024, 2 recorded runs, 1 warmup per cell) on `mtpnorm-r1`, compared against the `bverify-r1` gate from `results/2026-09-22-r646-verifybatch`. Every row in both runs has `finish_reason: length`.

| cell | bverify-r1 decode t/s (pooled median) | mtpnorm-r1 | Δ |
|---|---:|---:|---:|
| c1, ~4k ctx | 299.9 | 297.7 | −0.7 % |
| c4, ~4k ctx | 179.5 | 164.1 | −8.6 % |
| c8, ~4k ctx | 83.6 | 84.4 | +1.0 % |
| c4, ~26k ctx | 125.3 | 126.3 | +0.8 % |

## Acceptance parity

`completion_tokens / client_frames` is the number of tokens produced per engine iterate — a free acceptance-per-iterate proxy since each SSE delta carries all tokens accepted in one step. On the matched run0 of the c4/4k cell the two images agree to ±0.05 on all four requests (3.765/3.765/3.631/2.667 vs 3.779/3.779/3.631/2.716): the fused norm did not move draft quality.

The c4/4k pooled gap is the documented content lottery, not a regression: on `bverify-r1` itself the same prompt swung 3.63 → 2.43 tokens/iterate across its two runs, and `mtpnorm-r1` run1 drew one additional low-acceptance request (i0: 3.75 → 2.58). A systematic draft regression would also appear in run0; it does not.

## Verdict

Neutral-positive. ~60 launches × ~1.5 µs of dispatch each ≈ 0.1 ms/step is a real but sub-noise reduction. The image is banked as the base for the fusion sweep (int8 fused mixer kernel, draft-prefill batching, MoE prefetch) rather than promoted alone — the daily stays on `bverify-r1`.
