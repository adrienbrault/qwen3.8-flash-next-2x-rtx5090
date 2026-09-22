# The canonical gate on the current daily: 1,024 forced tokens, every row `length`

Results directory on the serving host: `results/2026-09-22-gate-hardened`. Raw records: [`2026-09-22-gate-hardened/`](2026-09-22-gate-hardened/). Driver: [`bench/fn_gate.sh`](../bench/fn_gate.sh). Instrument: [`bench/probe.py`](../bench/probe.py). Served container: `flashnext`, image `tabbyapi:mtpwin-r2-metrics1`, model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, 22 resolved `EXL3_`/`PYTORCH` environment keys per the launcher log.

`fn_gate.sh` is the gate this repository's measurement rules add up to: one script that refuses to publish below 1,024 forced tokens (the start-of-generation transient, [GOTCHAS 19](../docs/GOTCHAS.md)), records the served model id, image and resolved env-key count in the audit log, runs a warm-up round per shape, and summarises per-request medians with the finish-reason histogram. This file is its first run against the daily.

Conditions: greedy, `min_tokens` = `max_tokens` = 1,024, salt 424242, one warm-up round plus two recorded rounds per shape, unique salted prompts. Leg A prompts carry ~3.1k tokens of scrambled-word context; leg B ~19.7k. Both legs ran against the same boot of the daily launcher.

| leg | streams | n | prompt tokens | decode t/s, per-stream median | wall t/s | TTFT s | accepted tokens per verify |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A, ~4k | 1 | 2 | 3,106 | 289.9 | 281.1 | 0.11 | 3.78 |
| A, ~4k | 4 | 8 | 3,106 | 178.0 | 167.7 | 0.36 | 3.69 |
| A, ~4k | 8 | 16 | 3,097 | 80.5 | 76.3 | 0.57 | 1.78 |
| B, ~26k | 4 | 8 | 19,661 | 121.1 | 112.5 | 0.65 | 2.57 |

Every request finished on `length`; server and client token counts agree on every row.

## The acceptance column is the number to read with these

The leg A decode rates sit ~40–50 % above the README's decode curve ([R580](r580-decode-curve.md)), and the acceptance column says why: the ~3.1k-token scrambled-word context produces list-formatted, enumeration-heavy generations that the MTP draft predicts almost perfectly — 3.78 tokens per verify step at 1 stream, against ~2.5 implied by R580's bare-prompt prose rows. Per-step rate is unchanged: 76.8 verify steps per second here against 78.1 in R580's c1 code row. The cards are not faster; the shape moved the draft's acceptance, and acceptance is most of a drafted decode rate ([R572](r572-mtp-acceptance.md)).

Two consequences. The gate is a same-shape instrument: it is for saying boot A and boot B differ, or do not, at a fixed prompt shape — not for refreshing the README curve, which is measured on its own documented shape (bare prompts, no context). And the draft policy's contribution shrinks where it matters least: acceptance falls to 1.78 tokens per verify step at 8 streams, where the policy already runs depth 1.

## What changed in the launcher the same day

The daily launcher grew the checks that used to be the failure modes of this repo's experiment days: a `DRAFT_MODE` knob that emits a schema-legal draft block (the `disabled` literal plus no policy line — the combination the schema rejects), in-image config validation before the serving container is stopped (~0.4 s, verified against a deliberately bad config with the daily untouched), fail-fast on `restarting`/`exited`/`dead` container states during the boot wait, a `/v1/model` identity assertion after `/health`, and a refusal to boot over occupied GPUs unless `ALLOW_BUSY=1`. The experiment-side helpers moved into [`scripts/lib/serve-ctl.sh`](../scripts/lib/serve-ctl.sh) and the static checks into [`scripts/check-unit.sh`](../scripts/check-unit.sh).
