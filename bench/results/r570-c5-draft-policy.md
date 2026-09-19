# R570 / R571: depth 2 at 5 streams, +17 % where the draft policy used to give up

Results directories on the serving host: `results/2026-09-19-r570-promote-c5-policy` and `results/2026-09-19-r571-promote-c5-policy-2`. Raw records: [`2026-09-19-r570-promote-c5-policy/`](2026-09-19-r570-promote-c5-policy/), [`2026-09-19-r571-promote-c5-policy-2/`](2026-09-19-r571-promote-c5-policy-2/). Drivers: [`scripts/r570-promote-c5-policy.sh`](../../scripts/r570-promote-c5-policy.sh), [`scripts/r571-promote-c5-policy-2.sh`](../../scripts/r571-promote-c5-policy-2.sh).

One launcher line: `draft_num_tokens_by_batch` `[[4, 3], [8, 1]]` → `[[4, 3], [5, 2], [8, 1]]`. The served policy drafts one token from the fifth stream on, so a 5-stream step verifies 10 rows; depth 2 verifies 15, still inside the 16-row limit of the fast MoE decode path.

Four boots per round, interleaved served / candidate / candidate / served, 8 slots at 966,656, NVMe tier off, aggregate t/s over 1,024 forced tokens.

| shape | R570, 4 runs per arm | R571, 8 runs per arm |
| --- | --- | --- |
| 5 streams, code | 470.1 → 563.1 (**+19.8 %**) | 479.7 → 562.8 (**+17.3 %**) |
| 5 streams, prose | 475.4 → 545.4 (**+14.7 %**) | — |
| 4 streams, code / prose | +0.0 % / −0.4 % | — |
| 6 streams, code | −0.0 % | +0.0 % |
| 6 streams, prose | −2.1 % | −3.6 % (−3.1 % pooled over both rounds, 12 runs per arm) |
| 7 streams, code / prose | — | −0.2 % / +2.6 % |
| 8 streams, code / prose | −0.8 % / +1.3 % | — |

Fingerprints stay canonical on every boot and free VRAM is unchanged, since the policy only changes how many tokens are drafted at 5 streams.

The 6-stream prose cost is real and reproduces across rounds. Both policies draft the same depth at 6 streams, so it comes from rounds that pass through 5 streams as requests start and finish, where the depth now changes. In the 8-agent replay the load sits at 5–7 calls in flight 46 % of the time, which is why the trade is worth taking.
