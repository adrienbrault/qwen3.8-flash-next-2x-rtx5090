# R648: the accept loop is 0.32 % of decode; the residual is ~1691 launches/step plus GPU serialization

Results directory on the serving host: `results/2026-09-22-r648-attrib-consume`. Raw records: [`2026-09-22-r648-attrib-consume/`](2026-09-22-r648-attrib-consume/). Image `tabbyapi:bverify-r1` (the running daily). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`.

## What was measured

py-spy (speedscope, 200 Hz) attached to the live serving engine while `bench/fn_bench.py` drove a forced greedy burst: c8, ~6.1k prompt tokens, `min_tokens` 2048, 8/8 rows `finish_reason: length`, 78.6 decode t/s per stream, 584 aggregate t/s ([`bench-c8.jsonl`](2026-09-22-r648-attrib-consume/bench-c8.jsonl)). The capture covers ~18 s of steady drafted decode at the production policy (c8 → draft depth 1).

Decode-thread budget on the serving thread:

| frame | share of decode time |
|---|---:|
| `streams.py:231 synchronize` — the single batched verify readback (GPU wait) | ~53 % |
| forward dispatch: `hyperconnections._mix` 4.1 %, `get_bucketed` 2.2 %, `routing_std` 1.9 %, `to_device` 1.0 %, `make_key` 0.7 %, MTP draft forward ~1.5 % | ~34 % |
| `Job.receive_sample` inclusive — the entire accept path | **0.32 %** |

The R647 working theory (per-token `.item()` dispatches in the accept loop, est. 1–1.5 ms/iterate) is refuted: the ~119 `aten::item` reads per iterate are cheap pinned-CPU accesses spread across bookkeeping, and the whole accept path costs ~0.1 ms/iterate. No concentrated host patch exists there.

## Launch census and device serialization

Per-step kernel census at c8/d1/26k (64 iterates, torch-profiler capture from R647): **1691 launches per step**, ~40 % under 2 µs. Per-family:

| family | ms/step | launches/step |
|---|---:|---:|
| dense gemm/mgemm (+lm_head) | 4.51 | 279 |
| MoE coop a+b+rot | 6.20 | 147 |
| GDN (dots/up/delta-rule/conv/regrid) | 3.73 | 360 |
| attention (qsa+dsa) | 1.37 | 168 |
| routing | 0.15 | 49 |
| glue tail (~20 families, ~1.7 µs avg) | ~1.15 | ~690 |

Per-stream GPU timeline of the same trace: dev0's main stream is busy 10.5 ms/step (48 % of the window), dev1's 5.4 ms/step (25 %). The two cards run nearly serially — layer-split pipeline-parallel [30,30] is the only supported mode for this architecture (`tensor_parallel: true` raises `NotImplementedError`), so each GPU idles while the other computes. Rebalancing the split boundary does not help: the serial-layer sum is invariant to where the boundary sits.

## Consequence for the next lever

The ~2.1–2.4 ms/iterate host residual is launch-count-bound — diffuse dispatch across ~1691 kernels/step — not an accept-loop cost. Remaining decode levers in prize order: (1) fuse/eliminate the ~690 sub-2 µs glue launches per step (est. −1 to −2 ms/step); (2) MoE coop small-rows efficiency, already tile-tuned, uncertain −0 to −3 ms; (3) GDN dots+up consolidation (−0.5 to −1 ms); (4) CUDA-graph-per-shape decode step, the largest ceiling (~15–20 %) but no graph infrastructure exists in the engine; (5) the PP serialization itself is architectural — only speculative cross-device pipelining moves it, and that is a research-grade project.
