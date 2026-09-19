# R560: drafting 2 tokens above 4 jobs costs 32–39 % at 6 and 8 streams; the served policy stays

Results directory on the serving host: `results/2026-09-19-r560-c8-policy`. Raw records: [`2026-09-19-r560-c8-policy/`](2026-09-19-r560-c8-policy/). Driver: [`scripts/r560-c8-policy.sh`](../../scripts/r560-c8-policy.sh). Date: 2026-09-19.

Configuration: 8 slots at 966,656 tokens ([R558](r558-slots8.md)), NVMe tier off, one boot per arm, `fn_bench` 1,024 forced tokens × 2 runs after a warm-up round, aggregate t/s. c1 fingerprints equal on every boot, no new verify shapes, free VRAM unchanged.

| arm | draft policy | 6 streams, code | 6 streams, prose | 8 streams, code | 8 streams, prose |
| --- | --- | --- | --- | --- | --- |
| P1 (served) | `[[4, 3], [8, 1]]` | 513.5 | 512.7 | 640.2 | 628.2 |
| P2 | `[[4, 3], [6, 2], [8, 1]]` | 350.2 | 327.2 | 635.0 | 626.1 |
| P3 | `[[4, 3], [8, 2]]` | 345.8 | 326.1 | 405.0 | 381.3 |

At depth 2, 6 jobs verify 18 rows and 8 jobs 24 rows. Above 16 rows the MoE layers leave the cooperative decode kernels ([R562](r562-profile-c8.md)), and the step costs about twice as much while the extra draft token adds about 40 % more tokens per step.
