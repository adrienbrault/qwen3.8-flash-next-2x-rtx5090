# R576: the 5-stream draft policy is promoted, +11 to +14 % at 5 streams for −2 % at 6-stream prose

Results directory on the serving host: `results/2026-09-19-r576-promote-c5-policy`. Raw records: [`2026-09-19-r576-promote-c5-policy/`](2026-09-19-r576-promote-c5-policy/). Driver: [`scripts/r576-promote-c5-policy.sh`](../../scripts/r576-promote-c5-policy.sh).

Served since 2026-09-19 21:46:52 UTC: `draft_num_tokens_by_batch` `[[4, 3], [8, 1]]` → `[[4, 3], [5, 2], [8, 1]]`, one line of [`scripts/launch-flashnext.sh`](../../scripts/launch-flashnext.sh). Image, pool, slots, split, chunk and KV mode are unchanged. Rollback: `launch-flashnext.sh.pre-r576`.

The size of the gain was already measured twice ([R570 and R571](r570-c5-draft-policy.md), +19.8 % and +17.3 % at 5 streams on code), and the decision to accept the 6-stream prose cost was the operator's. This round re-measured it a third time on one boot per arm, then ran the promotion gates.

| shape | served `[[4, 3], [8, 1]]` | candidate `[[4, 3], [5, 2], [8, 1]]` | change |
| --- | --- | --- | --- |
| 5 streams, code | 506.9 | 561.2 | **+10.7 %** |
| 5 streams, prose | 474.0 | 540.9 | **+14.1 %** |
| 6 streams, code | 509.8 | 513.5 | +0.7 % |
| 6 streams, prose | 520.7 | 510.2 | −2.0 % |

Two runs per arm per shape, 1,024 forced tokens, tier off, 8 slots at 966,656, aggregate tokens per second over the round's wall time. Both boots read free VRAM 2,085 / 867 MiB and the canonical fingerprints (`f4add302e176d78e` at one stream, `4a255910dee2d9c5` at 30k), so the change is a scheduling decision and nothing else.

Gates on the promoted launcher, NVMe tier on:

| gate | result |
| --- | --- |
| boot, tier on | pool 966,656, 8 slots, split `[30, 30]`, chunk 2,048, vision on; fingerprints canonical |
| agentic edit, greedy and sampled, 1 and 5 streams | 6/6 each; 231.9 t/s at 1 stream, 458.4 aggregate at 5 |
| needles, 131k and 240k prompt tokens | 5/5 and 5/5 |
| tool-eval-bench, 69 × 4 | 86.0 ± 0.8, CI [85.2, 86.8] |
| GSM8K 5-shot, n=500, no stop strings, 5 concurrent | 0.976 |

The README's decode rows and the [decode figure](../../docs/img/decode-scaling.svg) are drawn from the candidate arm from here on, since that is what the daily serves. The 5-stream point still sits below 4 streams: what remains of the dip is the 16-row limit of the fast MoE decode path, not contention.
