# R526: a persistent NVMe prefix tier restores 30k and 120k prompts after a restart in 0.7 and 1.0 s, output identical to cold

Results directory on the serving host: `results/2026-09-19-r526-nvme-tier` (try 5; tries 1 to 4 archived beside it). Raw records: [`2026-09-19-r526-nvme-tier/`](2026-09-19-r526-nvme-tier/). Driver: [`scripts/r526-nvme-tier.sh`](../../scripts/r526-nvme-tier.sh). Date: 2026-09-19.

The tier writes KV pages and recurrent checkpoints to a dedicated NVMe filesystem in the background, under a byte cap with pruning, and serves them after VRAM eviction or a restart. It keys pages by content hash and namespaces them by a hash of every installed engine source file, so a changed engine never reads another engine's pages. Rounds 1 to 3 ran on `tabbyapi:stack-r4-e3r2` with the pruned-draft flags off; [R532](r532-nvme-tier-r4.md) repeats the steps on the served chain.

| try | outcome |
| --- | --- |
| 1 | launcher patch did not apply (context moved) |
| 2 | writes, drain, cap and crash restart pass; no restore ever hit: every PLE checkpoint's digest mismatched on read, because ExLlamaV3 stores PLE checkpoints as views of the live slot ([R530](r530-promote-plefix.md)) |
| 3 | the boot check read the wrong tier log line after the restart (harness) |
| 4 | first restore after a crash-restart: 120k fully restored, 1.00 s against 11.99 s cold; 30k restored to 28,672 of 29,952 tokens, because two checkpoints changed while being serialized and were refused (no PLE fix in that image) |
| 5 | with the PLE fix layered on: every step passes |

Try 5:

| step | result |
| --- | --- |
| crash restart, then the same prompts | 30k: 29,952 of 29,952 tokens from disk, 0.69 s (warm in VRAM 0.39 s, cold 3.77 s). 120k: 119,808 of 119,808, 0.98 s (warm 0.44 s, cold 12.07 s). Restored, warm and cold outputs identical |
| reopen | 15 of 15 checkpoints intact, scanned in 2.1 s; 0 refused |
| cap 2 GiB, four 40k prompts | at most 1.95 GiB after each drain |
| decode while a 120k prefix drains to disk | −5.7 % at 1 stream, −1.9 % at 4 |
