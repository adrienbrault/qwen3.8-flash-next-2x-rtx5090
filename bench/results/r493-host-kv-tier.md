# R493: the host-RAM KV tier turns a 12.7 s re-prefill into 0.5–0.7 s, for 16 GiB of host RAM

Results directory on the serving host: `results/2026-09-18-r493-host-kv-tier`. Raw records: [`2026-09-18-r493-host-kv-tier/`](2026-09-18-r493-host-kv-tier/). Driver: [`scripts/r493-host-kv-tier.sh`](../../scripts/r493-host-kv-tier.sh). Date: 2026-09-18.

[`bench/revisit.py`](../revisit.py): five sessions of about 105.6k tokens each (525k in total, beyond the 360,448-token GPU pool), then sessions 0 and 1 again, greedy, 64 tokens. 3.05bpw pack.

| arm | first requests | revisits of sessions 0 and 1 |
| --- | --- | --- |
| A, `sysmem_kv_cache: 0` (served) | 12.73–12.87 s | 12.74 / 12.74 s (full re-prefill) |
| B, `sysmem_kv_cache: 16384` | 12.79–13.23 s | **0.51 / 0.67 s**, output identical to the first request |

A host page image is a byte copy of every paged cache tensor, about 16.8 KiB per token (8-bit KV plus the QSA planes, 15.75 KiB, plus the MTP draft layer's page), so 16 GiB holds about 1.0M tokens. Host MemAvailable went 46.6 → 22.9 GiB at boot and 15.9 GiB after the run. No decode or pool effect. Not served: host RAM is not spent on KV; a disk tier is in development instead.
