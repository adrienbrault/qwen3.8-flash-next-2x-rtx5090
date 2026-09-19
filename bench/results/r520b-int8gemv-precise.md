# R520b: with 8 counterbalanced boots, switching off the int8-activation GEMV is −0.3 % at c1 (95 % CI −1.5 to +0.9); no effect

Results directory on the serving host: `results/2026-09-19-r520b-int8gemv-precise`. Raw records: [`2026-09-19-r520b-int8gemv-precise/`](2026-09-19-r520b-int8gemv-precise/). Driver: [`scripts/r520b-int8gemv-precise.sh`](../../scripts/r520b-int8gemv-precise.sh). Date: 2026-09-19.

[R520](r520-int8gemv.md) could not resolve a 0.5 % effect: its default arm always booted first, and first rounds after a boot read low. This run was designed to answer the question.

- 8 boots in the order A B B A B A A B. A is the served configuration; B adds `EXL3_INT8_GEMV=0`, checked in the container. Linear drift cancels within each block of four.
- Each boot: a c1 greedy fingerprint, then [`bench/probe.py`](../probe.py) code, 2,048 tokens, with one full-length unrecorded warm-up round per shape (`--warmup-runs 1`), then 3 runs at c1 and 2 at c4.
- GPU clocks, temperature and power are logged before every boot's runs; draft acceptance is read from each boot's TabbyAPI log.

| shape | A boot means (t/s) | B boot means (t/s) | B − A | 95 % CI |
| --- | --- | --- | --- | --- |
| code c1 | 219.2, 220.3, 221.4, 219.8 (mean 220.2) | 220.9, 218.9, 217.9, 220.5 (mean 219.6) | −0.26 % | −1.46 to +0.94 % |
| code c4 | 519.6, 532.4, 522.0, 515.6 (mean 522.4) | 519.1, 516.7, 516.7, 517.8 (mean 517.6) | −0.92 % | −3.13 to +1.29 % |

- Every boot gave the served fingerprint `ae890c45d1000582`. Draft acceptance: A 61.38 %, B 61.71 % over about 139k drafted tokens each.
- Clocks were 2,932 / 2,842 MHz and temperatures 57–60 / 37–41 °C at every boot: no thermal drift.
- Verdict: no effect in either direction; the default stays.

## What the design teaches

Within one boot, repeated runs differ by 0.26 % (c1) and 0.64 % (c4). Between boots of the same configuration, the means differ by 0.4–0.6 % at c1 and up to 1.4 % at c4. Precision therefore comes from the number of boots, not the number of runs. With 4 boots per arm the c1 interval is ±1.2 %. Resolving ±0.5 % takes about 8 boots per arm, about 35 minutes of GPU time.

The served configuration measured this way, arm A over 4 boots with warm-up: **code 220.2 t/s at c1 and 522.4 t/s aggregate at c4**.
