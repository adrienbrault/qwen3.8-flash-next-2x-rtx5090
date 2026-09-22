# R655: expert parallelism across the two GPUs is refuted — the P2P exchange alone eats the prize

Results directory on the serving host: `results/2026-09-22-r655-ep-p2p`. Raw records: [`2026-09-22-r655-ep-p2p/`](2026-09-22-r655-ep-p2p/). A ~60-line probe ([`ep-p2p-probe.py`](2026-09-22-r655-ep-p2p/ep-p2p.log) output) timing the exchange an expert-parallel MoE would pay per layer: ship the 80 KB activation row plus routing picks to the peer GPU, get a 160 KB fp32 partial back, with event synchronization each way.

## What was measured (200 iterations, medians, both cards otherwise idle)

| phase | wall |
|---|---:|
| copy out, 80.5 KB d0→d1 | 28.3 µs |
| copy back, 160 KB d1→d0 | 30.6 µs |
| event roundtrip alone | 6.6 µs |
| **full exchange wall** | **53.7 µs** |

Small copies are latency-bound (~3 GB/s effective), so the exchange cost does not shrink meaningfully with payload.

## Verdict

Kill. Per-layer wall = 53.7 µs exchange + the peer's expert-half kernel (65–85 µs by the existing c4 fit) = 118–138 µs, versus ~126 µs single-device — break-even at best. The pre-registered kill line was an exchange ≥ ~35 µs; measured 53.7 µs. Combined with `qwen4_exp` forbidding tensor parallelism, this closes the two-device serialization question: the layer-split makespan (dev0 ~10.5 ms, dev1 ~5.4 ms of a ~19.5 ms step) is final on this hardware pair. Remaining decode headroom is launch overhead (R654's ~762 eager launches/step) and kernel efficiency, not device overlap.
