# R566: a draft policy fixes the 5-stream dip; 17–32 verify rows on the cooperative kernels do not

Results directory on the serving host: `results/2026-09-19-r566-moe-rows32-try3`. Raw records: [`2026-09-19-r566-moe-rows32-try3/`](2026-09-19-r566-moe-rows32-try3/). Driver: [`scripts/r566-moe-rows32.sh`](../../scripts/r566-moe-rows32.sh).

The fast MoE decode path handles at most 16 verify rows, where rows = streams × (draft depth + 1). Above that the generic kernel runs and costs about twice as much per step ([R562](r562-profile-c8.md)). Two ways past it were measured on one image: a policy that keeps every batch inside 16 rows, and a kernel that runs 17–32 rows as two cooperative calls of at most 16 rows each.

## Kernel harness

On both cards, for every row count 17..32 and both routing patterns, the split path is `torch.equal` to running the same rows in 16-row chunks, repeatable, and refuses to run with the flag off. A single-call "wide" mode was also built: it is **not repeatable** at most row counts on one card (relative L2 0.70–1.40 against the split reference), so it was dropped before serving.

## Serving arms

Four boots, 8 slots at 966,656, NVMe tier off, aggregate t/s over 1,024 forced tokens × 2 runs. All four boots carried the canonical fingerprints and equal free VRAM.

| arm | 4 streams code / prose | 5 streams | 6 streams | 8 streams |
| --- | --- | --- | --- | --- |
| served `[[4, 3], [8, 1]]` | 528 / 506 | 478 / 472 | 514 / 523 | 645 / 637 |
| **policy `[[4, 3], [5, 2], [8, 1]]`** | 531 / 509 | **566 / 546** | 516 / 514 | 649 / 640 |
| 17–32 rows, `[[4, 3], [8, 2]]` | 531 / 513 | — | 516 / 523 | 551 / 544 |
| 17–32 rows, `[[8, 3]]` | 488 / 513 | 465 / 452 | 493 / 478 | 589 / 564 |

The served policy drops to depth 1 as soon as a fifth stream joins, which is why 5 streams ran slower than 4. Depth 2 at 5 streams is 15 rows, still inside the cap, and lifts that point by +18.5 % on code and +15.8 % on prose without moving the others.

The kernel is correct but slower than depth 1 wherever it would be used: −14.5 % at 8 streams for depth 2, −8.7 % for depth 3. Two cooperative calls do not beat one call of half the rows plus the deeper draft's cost.
