# R646: the batched draft-verify path had never run; enabling it gains ~1–4 % decode

Results directory on the serving host: `results/2026-09-22-r646-verifybatch`. Raw records: [`2026-09-22-r646-verifybatch/`](2026-09-22-r646-verifybatch/). Driver: [`scripts/r646-verifybatch.sh`](../scripts/r646-verifybatch.sh). Gate: [`bench/fn_gate.sh`](../bench/fn_gate.sh). Candidate image `tabbyapi:bverify-r1` = `tabbyapi:mtpwin-r2-metrics1` + [`docker/verifybatch-r1.patch`](../../docker/verifybatch-r1.patch) (three hunks, Python-only). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, 22 resolved `EXL3_`/`PYTORCH` env keys.

## The defect

`EXL3_BATCH_VERIFY=1` has been in the served environment since the round-4 batched MTP verifier landed. A py-spy profile of the serving generator (R645, 2026-09-22) attributed 43 % of c4 wall time to one line — `Job.receive_sample`'s `next_token.cpu()` — and showed zero samples under the verifier's `ready.synchronize()`: the batched path never executed. Two causes, confirmed in-image:

1. `CustomSampler.__init__` OR-ed `reqs_past_ids`/`reqs_torch_seed` over the *input* step stack, before `alt()` simplification. TabbyAPI appends `SS_RepP` + `SS_PresFreqP` to every request's stack; a neutral penalty step's `alt()` returns `SS_NoOp`, but its `reqs_past_ids()` had already set the flag, and `verification_batch_mode()` vetoes any sampler that needs past ids. Every request the frontend ever built was ineligible.
2. `verification_batch_mode()` also vetoed `job.device_logit_mask is not None`. The mask is per-iterate constant for filter-free jobs (the `min_new_tokens` EOS block, checkpoint explored tokens), so every forced-length benchmark carried a live mask on every token. The patch forwards the mask into the batched `sampler.forward`, which applies it identically to the serial path; `job.filters` stays vetoed because filter masks are stateful per accepted position.

## Gate (canonical, salt 424242, 1,024 forced tokens, medians over per-request rows)

| shape | `metrics1` decode t/s | `bverify-r1` decode t/s | Δ |
|---|---:|---:|---:|
| c1 / ~4k prose | 289.9 | 299.9 | +3.4 % |
| c4 / ~4k prose | 178.0 | 179.5 | +0.8 % |
| c8 / ~4k prose | 80.5 | 83.6 | +3.9 % |
| c4 / ~26k prose | 121.1 | 125.3 | +3.5 % |

Every row `finish_reason: length`. Acceptance per verify unchanged: 3.78 / 3.69 / 1.83 / 2.60 vs 3.78 / 3.69 / 1.78 / 2.57 on the baseline gate earlier the same day ([`2026-09-22-canonical-gate.md`](2026-09-22-canonical-gate.md)).

## Mechanism check

py-spy on the patched boot under the same c4 load: the `job.py:622` leaf collapses from 43.3 % to below 1 %, and one `synchronize` (streams.py:231) absorbs 49.4 % — a single pinned readback per iterate that waits on the verify forward, replacing ~10 per-token device-to-host round trips. The residual gain is +1–4 % rather than larger because most of the old leaf was that same wait; what the batching removes is the redundant per-token round trips, not the forward.

Smoke tests on the candidate: `temperature=0` + `min_tokens` and `temperature=0.6` + `min_tokens` both complete with `finish_reason: length`, natural EOS still terminates, a sampled c4 burst returned 8/8 rows at ~110 decode t/s, and the engine log showed no errors. The `sampled` round-4 mode is newly reachable by the same fix; its per-row RNG order differs from the serial path by design (the code comment documents the Philox row-subsequence choice).

Promoted 2026-09-22: `DAILY_IMG=tabbyapi:bverify-r1` in `scripts/launch-flashnext.sh`. Rollback: `IMG=tabbyapi:mtpwin-r2-metrics1`.
