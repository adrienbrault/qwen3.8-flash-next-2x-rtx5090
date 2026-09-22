# R650: MoE coop L2-prefetch is refuted — +7..+15 % slower per kernel, end-to-end flat

Results directory on the serving host: `results/2026-09-22-r650-moepf-gate`. Raw records: [`2026-09-22-r650-moepf-gate/`](2026-09-22-r650-moepf-gate/). Image `tabbyapi:moepf-r1` = `tabbyapi:mtpnorm-r1` plus the `moepf` patch: `prefetch.global.L2` issued ~12 k-slices ahead inside the shared `gemv_tile_v2` loop of the SM120 MoE cooperative kernels (K bands 1–8, codebook classes, A/B stages), behind `EXL3_MOE_PREFETCH_L2`. The template is dual-instantiated, so flag-off is the previous binary path; the prefetch is a cache hint, numerics-identical by construction. Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

## What was measured

Phase A — an Nsight Compute A/B on the same image, 40 cooperative-kernel launches per arm at c8/d1 over ~26k prompts. The on-arm profile shows the `PFL2=1` template instantiations, so the flag took effect:

| kernel | off median | on median | Δ |
|---|---:|---:|---:|
| `exl3_moe_coop_a_kernel` | 68.5 µs | 79.1 µs | +15.5 % |
| `exl3_moe_coop_b_kernel` | 67.7 µs | 72.7 µs | +7.3 % |

`dram__bytes_read` is unchanged (+0.0 %) and `launch__registers_per_thread` sits at the 64-per-thread ceiling in both arms. The prefetch does not reduce DRAM traffic — the 4-deep register ring plus L2 already cover the miss latency — and the extra prefetch instructions cost issue slots at zero register headroom.

Phase B — the canonical gate (`bench/fn_gate.sh`, salt 424242, `min_tokens` 1024, 2 recorded runs, 1 warmup per cell) on `moepf-r1` with `EXL3_MOE_PREFETCH_L2=1`, against the `bverify-r1` gate from `results/2026-09-22-r646-verifybatch`. Every row has `finish_reason: length`.

| cell | bverify-r1 decode t/s | moepf-r1 flag-on | Δ |
|---|---:|---:|---:|
| c1, ~4k ctx | 299.9 | 292.7 | −2.4 % |
| c4, ~4k ctx | 179.5 | 181.7 | +1.2 % |
| c8, ~4k ctx | 83.6 | 84.1 | +0.6 % |
| c4, ~26k ctx | 125.3 | 126.9 | +1.3 % |

## Verdict

Refuted. The cooperative kernels are slower with the prefetch at kernel level and the end-to-end gate is flat inside noise, meaning the MoE block is not fully on the makespan's critical path at these batch shapes. `moepf-r1` is shelved; the daily stays on `bverify-r1`. The ~500–600 GB/s the coop kernels reach at ≤160 rows is therefore not a prefetch-depth problem — the remaining candidates from the same survey are wider k-tiles and different work decomposition, which this same ncu A/B recipe can answer in about 20 minutes per arm.
