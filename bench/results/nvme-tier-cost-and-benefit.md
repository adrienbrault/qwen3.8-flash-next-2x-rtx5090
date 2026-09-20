# The NVMe prefix tier: what it costs, what it returns, and a measurement error in between

Results directories on the serving host: `results/2026-09-20-r589-nvme-tier-cost`, `results/2026-09-20-r590-tier-tax-source`, `results/2026-09-20-r591-tier-scan-and-benefit`. Raw records: [`2026-09-20-r589-nvme-tier-cost/`](2026-09-20-r589-nvme-tier-cost/), [`2026-09-20-r590-tier-tax-source/`](2026-09-20-r590-tier-tax-source/), [`2026-09-20-r591-tier-scan-and-benefit/`](2026-09-20-r591-tier-scan-and-benefit/). Drivers: [`scripts/r589-nvme-tier-cost.sh`](../../scripts/r589-nvme-tier-cost.sh), [`scripts/r590-tier-tax-source.sh`](../../scripts/r590-tier-tax-source.sh), [`scripts/r591-tier-scan-and-benefit.sh`](../../scripts/r591-tier-scan-and-benefit.sh).

Measurement only. The tier remains enabled on the daily, and the figures in the first two rounds below are corrected by the third.

The victim throughout is greedy, 3,000 forced tokens at roughly 9,850 prompt tokens, prose content, `cache_size` 999,424. Every arm reads its tier state from the running container rather than from its own label.

## What was measured first, and why it was wrong

R589 and R590 compared the tier on against the tier off, one boot per arm, and read a decode cost of 26.4 percent and 28.6 percent at one stream. R590 added an arm with an empty namespace — the same code path and cap, nothing to evict — and it still lost 19.6 percent, which was read at the time as most of the cost being inherent to having the tier attached.

R591 booted the tier with `EXL3_NVME_TIER_SCAN=0` and the result did not fit:

| arm | c1 | c4 | configuration |
| --- | --- | --- | --- |
| base | 251.1 t/s | 150.5 t/s | tier off |
| scanoff | 277.5 t/s | 164.5 t/s | tier on, open scan disabled |

The tier is faster than the control in both columns. Lining every c1 reading of the series up in one place resolves it:

| configuration | c1 readings |
| --- | --- |
| tier off | 283, 275.4, 287.5, 284.3, 251.1 |
| tier on, open scan on | 236.4, 211.5, 203.1, 228.5 |
| tier on, open scan off | 277.5 |

The tier with its scan disabled sits inside the tier-off cluster. The tier with its scan enabled sits 50 to 80 t/s below it. The boot line names the mechanism: `open scan: 250/250 checkpoints intact in 16.0 s`. The scan validates every stored checkpoint once per boot on a background thread. These rounds begin warming about three seconds after the server answers and finish the victim inside a minute, so a sixteen-second boot cost was being measured as steady-state decode. A server that boots once and serves for hours amortises it to nothing.

The second correction is about method and applies to every figure above. R591's tier-off control read 251.1 t/s where four earlier tier-off measurements read 275.4 to 287.5 — a 13 percent swing on an identical configuration between boots. Each round in this series compared one boot against one boot, so a swing of that size sits underneath all of them. The percentages should be read as "the tier costs something at one stream while its boot scan is running", not as the specific numbers. R592 re-measures with the arms repeated (on, off, on, off) at four concurrencies and four runs per point, printing each difference beside the same-configuration repeat gap.

## What the tier returns

R589 submitted one prompt of about 45,000 tokens, restarted the server, and resubmitted the identical prompt: 4.56 s to first token cold, 4.75 s after the restart. That restart was a hard kill, and the tier is write-ahead, so R591 repeated it with a graceful stop and a two-minute grace period: 4.42 s cold, 4.52 s after the restart.

The tier counters are more informative than the timings. `tier before P` and `tier after P` in R591 are byte-identical — `wrote 353 pages + 33 ckpts (3.02 GiB)` on both sides — so submitting a 45,130-token prompt wrote nothing to the tier. The prompt was never admitted, which means no restart or eviction policy could have made it available afterwards.

Seven hours of real agent traffic say the same thing at scale. The run in `results/2026-09-20-r586-swebench-full` covers 9,926 requests with prompts of median 26,032 tokens, 97 percent of them prefix-cached, and the engine writes the tier's counters into the container log:

```
wrote 36938 pages + 18152 ckpts (1134.34 GiB)
restored 0 pages + 3 ckpts
evicted 31579 pages 18681 ckpts (interior 18457)
compactions 17183
lookups 10156: hit 3, load-failed 0, no-ckpt 2, ram 8621, position 0, aborts 0
pump 37648 pages (0.77 ms/page), idle 300
last miss: ram-as-deep (prefix 271/272 pages, disk ckpt at 270)
```

Three disk hits in 10,156 lookups, for 1,134 GiB written and 17,183 compactions. 31,579 of the 36,938 pages written were evicted again within the same run. The 8,621 `ram` hits belong to a different layer, which the tier-off configuration also has; that is why the tier-off arms are no slower at finding prefixes.

`last miss: ram-as-deep (prefix 271/272 pages, disk ckpt at 270)` describes the shape: the RAM layer already holds a deeper prefix than the disk checkpoint, so the disk copy has nothing to contribute. For traffic where a conversation's revisits are close together in time, that is expected.

## Where this leaves the configuration

The throughput case against the tier does not hold. Most of the apparent cost is a per-boot scan, `EXL3_NVME_TIER_SCAN=0` removes it, and the remainder is inside the boot-to-boot variation this series exposed.

What stands is narrower. Over seven hours of the traffic the tier was built for it wrote 1,134 GiB and returned three cache hits, and a probe prompt of 45,130 tokens was not admitted at all. That is a question about admission policy and about write endurance, not about decode throughput.

No change has been made to the daily.
