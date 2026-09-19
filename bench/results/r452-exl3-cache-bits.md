# R452: at 8 slots the 8-bit pool ceiling is 262,144 tokens under any split

Results directory on the serving host: `results/2026-09-17-r452-exl3-cache-bits`. Driver: [`scripts/r452-exl3-split-budget.sh`](../../scripts/r452-exl3-split-budget.sh). Date: 2026-09-17.

3.05bpw pack, 8 slots, 8-bit KV. 393,216 and 327,680 do not boot under the served split or a split weighted towards cuda:1 ("Insufficient VRAM in split for model and cache"). The ceiling moved to 360,448 when the slot count dropped to 4 ([R480](r480-exl3-pool.md)) and to 786,432 with the 2.50bpw pack ([R495b](r495b-2p50-audition.md)).
