# R536: the K=3 MoE decode kernel without register spills is bit-exact and not faster; not served

Results directory on the serving host: `results/2026-09-19-r536-nospill` (try 2; try 1 in `-try1`). Raw records: [`2026-09-19-r536-nospill/`](2026-09-19-r536-nospill/). Driver: [`scripts/r536-nospill.sh`](../../scripts/r536-nospill.sh). Date: 2026-09-19.

The served fused MoE decode kernel (`EXL3_MOE_COOP_V2=1`, [R460](r460-moecoop-v2-ab.md)) compiles its K=3 instances at 64 registers with 5 to 13 local-memory loads and stores each. The idea to audit the SASS for spills comes from [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks#217](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/217). Decode-kernels round 3 (Codex, `EXL3_MOE_COOP_V2_NOSPILL=1`) moves the prefetch ring from registers to an 8-stage ring in shared memory, with the same arithmetic and order. On this pack routed experts are K=3 on layers 0–11 and 37–47 (23 of 48); K=2 layers keep the served kernels. Here it was rebased onto the served chain as round 3b (image `tabbyapi:nvme-tier-r4-e3det-nospill`). The round-3b overlay is not published.

## Gates

- SASS (`sass-counts.tsv`): all 12 no-spill symbols have `LOCAL:0`, 61 to 64 registers and a stack of 8 or 16 bytes (the served kernels also have 8). They still carry 0–1 local loads and 1–2 local stores each.
- Bit-exactness against the served V2 kernel: 10 runs (real layer 0, K=3, on both cards at split-k 1 and 2; synthetic wide 0/1 × codebook 0/1/2), 96 comparisons each over rows 1 to 16 and six routing patterns, 0 mismatches (`real-*.json`, `synth-*.json`).
- Per-call kernel time from the same runs, served over no-spill: on the `distinct` routing pattern at 1, 4 and 16 rows the no-spill kernel is slower in 28 of 30 cells (ratio 0.93 to 0.99; one 1.001 and one 1.056, both at 1 row). Over all 180 timed cells it is slower in 116, median ratio 0.982.

## Serving A/B

One image, 8 boots in the order A B B A B A A B. A: the daily's environment. B: the same plus `EXL3_MOE_COOP_V2_NOSPILL=1`. The NVMe tier is off in both arms. Per boot: c1 and 30k greedy fingerprints, then `bench/probe.py` code, 2,048 forced tokens, greedy, one unrecorded warm-up round, c1 × 3 and c4 × 2.

| shape | A boot means (t/s) | B boot means (t/s) | B − A | 95 % CI |
| --- | --- | --- | --- | --- |
| code c1 | 222.5, 223.0, 223.4, 223.2 (mean 223.1) | 222.5, 221.5, 222.6, 222.4 (mean 222.2) | −0.37 % | −0.83 to +0.09 % |
| code c4, aggregate | 534.8, 513.8, 523.9, 525.5 (mean 524.5) | 509.7, 510.0, 519.5, 509.5 (mean 512.2) | −2.35 % | −5.35 to +0.66 % |

The c4 row mixes two kinds of run. In 4 of the 8 A runs and 1 of the 8 B runs all four requests reached 2,048 tokens (529.6 to 544.1 t/s); in the others 2 of the 4 requests stopped at 1,861 tokens (506.8 to 514.4 t/s). Restricted to runs with two early stops, A reads 512.0 and B 509.7 t/s (−0.45 %). Every boot of both arms produced c1 `e7fb377c987d685c` and 30k `4a255910dee2d9c5`.

Removing the 1–2 local-memory accesses by moving the prefetch ring to shared memory does not make the kernel or the served decode faster. Not served.

Try 1 failed the bit-exactness gate in the harness, not the kernels: the per-stage timing walk returned CUDA error 1 on every real-layer run, and the synthetic runs read their shapes from the 3.05 bpw pack. Try 2 times the whole call with CUDA events and passes the model path.
