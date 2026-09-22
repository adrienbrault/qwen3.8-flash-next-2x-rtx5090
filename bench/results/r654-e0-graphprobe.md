# R654+R656: per-step API inventory on the served image — ~762 eager launches and ~117 graph replays

Results directories on the serving host: `results/2026-09-22-r654-e0-graphprobe`, `results/2026-09-22-r656-bctrace`. Raw records: [`2026-09-22-r654-e0-graphprobe/`](2026-09-22-r654-e0-graphprobe/) (the 94 MB `.nsys-rep` stays on the box; its exported CSVs are here). An Nsight Systems capture of the decode harness on `tabbyapi:stack-r1` at c8/d1 over ~26k prompts, plus a boot with `EXL3_BC_ATTN_TRACE=1` driven by a c8 burst. This is the E0 prerequisite measurement for the segmented decode-graph project.

## Per-step inventory (steady decode window, ~20 steps over 0.4 s)

| API class | calls per step |
|---|---:|
| host kernel launches (`cudaLaunchKernel` + `cuLaunch*` + cooperative) | ~762 |
| `cudaGraphLaunch` — module-level BC graphs replaying | ~117 |
| `cudaMemcpyAsync` — pipeline-parallel handoffs | ~21 |
| `cudaStreamSynchronize` | ~0 |
| GPU kernels executed (eager + graph-internal) | ~1,488 |

Compared with the R648 census on `bverify-r1` (~1,691 host launches/step), the eager pool is down to ~762 — the stack's launch eliminations are visible in the census. The BC-attn trace shows graph builds at every exercised shape (bsz 1–8 × q_len 2–4 × regime 0/1 on both devices) and **zero DECLINED lines**: the existing module-level graph path is healthy and covers its declared domain.

## What this decides

Steady decode contains no stream synchronizations and ~21 small memcpys; the residual host-side cost is the ~762 eager launches at ~1.5–2.5 µs each (≈1.1–1.9 ms/step). That is the quantity a segmented whole-step graph would remove. `cudaGraphLaunch` at 117/step confirms the engine's per-shape module graphs already replay — a step-level capture would be additional capture around them, not a replacement.
