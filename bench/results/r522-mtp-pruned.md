# R522 and R522b: keeping the MTP draft chain on the GPU with a pruned embedding mirror is byte-identical and +2.0 % at 1 stream, +1.5 % at 4

Results directories on the serving host: `results/2026-09-19-r522-mtp-pruned` and `results/2026-09-19-r522b-mtp-pruned-precise`. Raw records: [`2026-09-19-r522-mtp-pruned/`](2026-09-19-r522-mtp-pruned/), [`2026-09-19-r522b-mtp-pruned-precise/`](2026-09-19-r522b-mtp-pruned-precise/). Drivers: [`scripts/r522-mtp-pruned.sh`](../../scripts/r522-mtp-pruned.sh), [`scripts/r522b-mtp-pruned-precise.sh`](../../scripts/r522b-mtp-pruned-precise.sh). Date: 2026-09-19.

The MTP draft head emits only the first 65,536 of the 248,320 vocabulary rows (`EXL3_MTP_HEAD_N=65536`), but each draft step looked its token up in the full 1.27 GB embedding table in host memory. Image `tabbyapi:mtp-pruned-r1` keeps the draft chain on the GPU (`EXL3_MTP_DEVICE_DRAFT=1 EXL3_EMBED_GPU=1 EXL3_EMBED_GPU_PRUNED=1`) with a 320 MiB copy of only those 65,536 rows on cuda:1; a token outside the copied rows falls back to the full table.

## Identity (R522)

- Standalone unit tests on the GPU: all pass. Chain identity: draft windows identical over 639 windows against the unpruned device chain and against the host path.
- Served fingerprints on both arms equal the daily's (c1 `ae890c45d1000582`, 30k `4a255910dee2d9c5`, on the configuration before [R525](r525-promote-int8mix.md)).
- Free VRAM: 2,015 / 779 MiB after boot with the copy, 1,219 / 273 MiB after a 30k c4 stress pass, against 2,015 / 1,099 MiB without it.
- One boot per arm read code c1 +1.4 %, prose c1 +1.6 %, c4 +0.3 / +0.4 %, which is inside the between-boot spread ([R520b](r520b-int8gemv-precise.md)).

## Speed (R522b)

The [R520b](r520b-int8gemv-precise.md) design on the [R525](r525-promote-int8mix.md) configuration: 8 boots in the order A B B A B A A B on the same image, A without the flags, B with them. Each boot: c1 greedy fingerprint, then `bench/probe.py` code, 2,048 tokens, one unrecorded warm-up round, 3 runs at c1 and 2 at c4.

| shape | A boot means (t/s) | B boot means (t/s) | B − A | 95 % CI |
| --- | --- | --- | --- | --- |
| code c1 | 217.9, 216.9, 217.6, 217.3 (mean 217.4) | 221.3, 222.1, 221.9, 221.8 (mean 221.8) | +1.99 % | +1.60 to +2.39 % |
| code c4 | 501.0, 500.2, 499.5, 502.8 (mean 500.9) | 506.6, 509.6, 510.2, 507.0 (mean 508.4) | +1.49 % | +0.77 to +2.22 % |

- All 8 boots gave the served c1 fingerprint `e7fb377c987d685c`. Draft acceptance: A 62.16 %, B 62.15 % over about 134k drafted tokens each, so the gain is the draft step getting cheaper, not more tokens accepted.
- Both 4-boot blocks agree: +1.96 % and +2.03 % at c1, +1.50 % and +1.49 % at c4.

The promotion run with the full gate set, including whole-pool VRAM with the copy on top of the 819,200-token pool, is queued (`scripts/r528-promote-mtp-pruned.sh`).
