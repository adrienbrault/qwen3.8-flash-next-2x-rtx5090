# R517: decode round 4 and grouped MoE prefill are served together; cold prefill 9.5k t/s at 60k and 10.0k at 120k

Results directory on the serving host: `results/2026-09-19-r517-promote-stack`. Raw records: [`2026-09-19-r517-promote-stack/`](2026-09-19-r517-promote-stack/). Driver: [`scripts/r517-promote-stack.sh`](../../scripts/r517-promote-stack.sh). Date: 2026-09-19.

Promoted 2026-09-18 23:26 UTC. Decode round 4 and E3 round 2 were both written against the reference base and share one file, `exllamav3_ext/bindings.cpp`; their patches apply in sequence with fuzz 0. The stacked overlay is [`docker/overlays/stack-r4-e3r2/`](../../docker/overlays/stack-r4-e3r2/), image `tabbyapi:stack-r4-e3r2`, served with `EXL3_MOE_PREFILL_E3=1` on top of [R514](r514-promote-r4i.md). Launcher: [`scripts/launch-flashnext.sh`](../../scripts/launch-flashnext.sh).

| gate | result |
| --- | --- |
| G0 sampler fallbacks | byte-identical to the served image |
| G1 fingerprints | c1 `ae890c45d1000582` (canonical); 30k `4a255910dee2d9c5`, E3's prefill-order hash, now the served 30k fingerprint |
| G1b cold prefill, salted | 60k: 9,543 t/s (floor 9,200); 120k: 10,015 t/s (floor 9,500) |
| G2 agentic-edit | 6/6 in four modes; greedy c1 211.9, greedy c4 437.1, sampled c1 231.1, sampled c4 423.1 t/s |
| G3 needles 131k / 240k | 5/5 / 5/5 |
| G4 tool-eval 69×4 | 85.8 ± 2.1 (trials 116 / 116 / 121 / 120) |

Free VRAM after boot: 2,015 / 1,099 MiB.
