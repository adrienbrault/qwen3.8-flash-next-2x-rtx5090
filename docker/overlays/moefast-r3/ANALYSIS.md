# moefast r3: keep r2's kernels, restore the shared-expert overlap

Opus sub-agent, 2026-09-24. No GPU was used for this round. Every number carries its source. "estimate" marks arithmetic on those sources.

Sources:
- R703b results and review: `review-r703b/results/` (`timeline-c{1,4}d3.txt`, `p0.log`, `p0-ncu-raw.csv`, `p1-summary.txt`) and `review-r703b/REVIEW.md`.
- Source tree: `src/exllamav3` (slotfix-r1). `work/a` is the stack-r2 tree (src + hcfast-r1 + moefast-r1). `work/c` is r3.

## 1. Mechanism (verdict: confirmed from source and timelines; the missing link is that the shared expert's GEMMs are cooperative 90 KB-smem launches)

**Who launches what, in host order** (`libtorch/blocksparse_mlp.cpp` `run_bszN`, served `EXL3_SHARED_EXPERT_OVERLAP=1`):
1. Python `BlockSparseMLP.forward` runs the router on the main stream: `ext.routing_std`, which is `hgemm` (cuBLAS `Kernel2` + `splitKreduce`) followed by `routing_std_topk_kernel<<<bsz>>>`.
2. `run_bszN` records `shared_input_ready` on main. The side stream waits on it and replays the shared expert's `BC_GatedMLP` graph there, then records `shared_done`.
3. The coop launcher (`exl3_moe_coop.cu`) enqueues on main:
   - `[rot]` (modes 0-2), then kernel A;
   - `cudaStreamWaitEvent(main, shared_done)`, then kernel B.

**The shared expert's kernels** (`modules/mlp.py`, `BC_GatedMLP::run_bszN_gr`):
- gate/up run in one `exl3_mgemm`. `use_mgemm` is true because 640 < `EXL3_MGEMM_N_THRESHOLD` 8192 (`model/config.py:32,77`).
- then `act_mul`, then the down projection through `exl3_gemm_gr`. That is the `exl3_gemv` narrow kernel at ≤ 8 rows (grid 2560/32 = 80 CTAs × 512 threads) and the autotuned `exl3_gemm` above 8 rows.
- **Every GEMM here is `cudaLaunchCooperativeKernel`** (`exl3_gemm.cu:296,654`, `exl3_gemv.cu:157`, `coop_autotune.cu:410-667`). mgemm and gemm take `SMEM_MAX` = 90 KB of dynamic shared memory per CTA (`exl3_gemm_inner.cuh:7`).

**Residency** (`p0-ncu-raw.csv`, R703b):
- The device has 102,400 B of shared memory per SM.
- Routed A and B use 512 threads × 64 registers, so `launch__occupancy_limit_registers` = 2 in V2, V3 and r2. Two CTAs fill the SM's 64 K registers. The grid is 340 = 170 SMs × 2, one wave.
- Dynamic smem per CTA:

  | kernel | dynamic smem |
  |---|---|
  | V3 A (M2) | 24,576 / 28,672 B |
  | r2 A | 44,144 / 43,104 B (+1,328 static) |
  | rot | 0 (2,128 static), 39 registers, grid 100 at 4 rows |

- **Consequence:** a 90 KB mgemm CTA cannot share an SM with even one A CTA in any mode (90 + 24 > 100 KB). A gemv CTA (512 threads) needs a whole A slot's registers. So while A is resident the shared expert has nowhere to run.
- A is persistent. It is a static strided loop, `for (item = blockIdx.x; item < count; item += gridDim.x)` (`exl3_moe_coop_r2_kernel.cuh:757`). SMs free only as A's CTAs finish their item share, i.e. near its end. CTAs with `blockIdx >= count` exit at once.

**What the timelines show** (per-layer medians, µs, `timeline-c1d3.txt` / `timeline-c4d3.txt`):

| | c1d3 M2 | c1d3 R2 | c4d3 M2 | c4d3 R2 |
|---|---|---|---|---|
| rot | 3.58 | 0 | 8.03 | 0 |
| A | 38.24 | 34.46 | 82.66 | 90.85 |
| A→B gap | 1.90 | 14.14 | 2.21 | 8.58 |
| shared start − A start | −3.58 | +22.94 | −5.31 | +68.13 |
| shared end − A end | −9.33 | +11.07 | −7.04 | +6.18 |
| shared span | 31.15 | 22.21 | 81.02 | 25.28 |
| layer | 70.46 | 73.81 | 151.87 | 156.83 |

- **Under M2 the shared expert starts with rot.** Shared start − A start = −3.58 = −rot. So the side stream's event wait resolves at the same moment as main's next kernel. The mgemm's CTAs are placed during the rot kernel, which uses 100 of 170 SMs and 2 KB of smem. A then fills the remaining SMs.
- **Under R2 there is no head kernel.** A is placed first on every SM, and the shared expert starts only when A's CTAs begin to retire:
  - at c1d3, 23 of 34.5 µs into A;
  - at c4d3, 68 of 91 µs. At 16 rows the item count is roughly 770 over 340 CTAs, so CTAs 90-339 finish after two items (estimate: ~77 distinct experts × 2 proj × 5 chunks).
  - B waits on `shared_done`: that is the 14.1 / 8.6 µs gap.
- **The review's inference is right, with one correction.** It said r2's larger smem is not the cause, since residency is register-bound. That holds for A's own residency. The binding constraint is the shared expert's 90 KB cooperative CTAs, which need A-free SMs. Only a kernel ahead of A on main (rot) or an earlier side-stream start gives them one.
- **The overlap is not free.** M2's in-situ A is 38.24 µs, against P0-isolated V3 A of 29.07 µs (K-weighted 25 × K2 + 23 × K3 of 27.2 / 31.1, `p0.log` r4/D28). R2's in-situ A is 34.46 against P0 r2 A of 30.37 (28.5 / 32.4).
  - The 9.2 − 4.1 = **~5.1 µs per layer at c1d3 is what co-running the shared expert costs kernel A** (estimate: it assumes the same in-situ D and cache inflation in both arms).
  - The shared expert alone spans 22.2 µs (R2).
- **So r2's kernel saving is real, and the scheduling ate it.** At c1d3, r2 saves 8.4 µs per layer of kernel time (rot 3.6 + A 3.8 + B 1.0). It adds 12.2 µs of wait. That is +3.35 µs per layer, and matches the layer spans.

## 2. Levers, and why

| lever | what | cost | expected effect (estimate) | status |
|---|---|---|---|---|
| **E: early fork** (`EXL3_SHARED_EXPERT_EARLY=1`) | Python calls `bc.start_shared(y)` before the router. The side stream starts during `hgemm` + `topk`. `run_bszN` joins that launch: same kernels, buffers and `shared_done`. | 0 kernels, 0 VRAM | The shared expert gets SMs while the router's few CTAs run. The router window at c1d3 is ≈ 0.34 + 0.154 ms per step over ~51 routing calls ≈ 9.7 µs of kernel time plus gaps (`timeline-c1d3.txt` Kernel2 + splitKreduce + routing_std_topk). That is ~45 % of the shared expert's 22 µs. What is still running when A launches is resident, so it overlaps as under M2. | **primary** |
| **P: side-stream priority** (`EXL3_SHARED_EXPERT_PRIO=1`) | `cudaStreamCreateWithPriority(greatest)` for the side stream. Graph kernel nodes run at the launch stream's priority, since `Graph` instantiates with flags 0 (`graph.cu:51`). | 0 | When shared and A CTAs are pending at the same time, the shared ones dispatch first. That covers the chain's later kernels (act, gemv), which become pending while A has unplaced CTAs. It does nothing once A is fully resident: priority does not preempt resident CTAs. | helper; P0 decides E vs E+P |
| **H: head kernel** (`EXL3_MOE_COOP_V3_HEAD=<ns>`, mode 3 only) | One 32-thread CTA on main right before r2's A that spins N ns on `%globaltimer`. It is what rot did for M2. | 1 launch + N ns per layer | Restores M2's placement at ≈ 2-3 µs per layer. Net vs M2 ≈ −3.58 (rot) + 2.5 (head) + 1.3 (r2 A) − 1.0 (r2 B) ≈ −0.8 µs per layer: flat (estimate). | **diagnostic**: it tests the mechanism directly |
| shrink r2 smem / reserve slots in A | — | — | Dead. A 90 KB CTA needs an A-free SM and CTA placement is breadth-first, so freeing k CTAs of A frees no whole SM until k ≥ 170. The number of SMs the autotuned mgemm needs is unknown. | rejected |
| shared expert inside the coop kernels | — | — | mgemm / gemv fold and reduction orders ≠ the coop GEMV: not bitwise. | rejected |
| move the join (sh added by an epilogue after B) | — | +1 launch | The shared expert would still wait for A, then B, to drain. | rejected |

**Why E is safe:**
- The shared expert reads only `y`. The routing kernels do not write `y`.
- `start_shared` records `shared_input_ready` after everything that produced `y`, as `run_bszN` does.
- The join (`cudaStreamWaitEvent(main, shared_done)` before B) is unchanged.
- The guard, `BlockSparseMLP._shared_early_ok`, forks only when the call is certain to reach `run_bszN` with the same `y`. It requires:
  - the fused-decode branch: not CPU offload, not an empty slice, quantized, reconstruct allowed;
  - no routed pre-norm or latent projection, which would replace `y`;
  - no routing broadcast and no CPU split.
- The C++ side refuses a second fork before the join, and a join on a different input (`TORCH_CHECK`).

**CUDA graphs:**
- **E:** the fork/join is the same event pattern the served path uses, only earlier. It is capture-legal if a caller captures the forward.
- **P:** inside an outer capture, node priorities follow the outer launch stream (instantiate flags 0), so P is inert there, but correct.
- **H:** H is a plain kernel.

**VRAM:** none of the three allocates device memory.

## 3. Expected gains per shape (estimates; P0 with the side stream and then P1 decide)

The nsys numbers are traced; R465 puts profiler inflation at ~10 %. Layers per step = 48 target layers at the verify row count.

- **c1d3 (4 verify rows):**
  - R3 layer ≈ R2 layer − (R2 gap − 1.9) + (added contention, somewhere between router-window-only ≈ 1 µs and M2-like 5.1 µs) = 73.81 − 12.24 + [1, 5.1] = [62.6, 66.7] µs.
  - Against M2's 70.46 that is **−3.8 to −7.9 µs per layer ≈ −0.18 to −0.38 ms per step, −1.5 to −3.2 % of the M2 P1 median ≈ 11.8 ms** (`REVIEW.md` claim 2).
  - Failure mode: if the shared chain's gemv still becomes pending after A is fully resident, the gap returns. E+P is meant to cover that; P0's m3E / m3EP gap column shows which.
- **c4d3 / c8d1 (16 verify rows):**
  - P0 kernel-only delta, r2 vs r1 at r16/D77 = (A + B) − (rot + A + B), K-weighted = (72.7 + 51.3) − (7.85 + 67.5 + 51.6) = **−2.95 µs per layer** (`p0.log`).
  - At D40 it is +3.7 (K2) / +1.9 (K3).
  - The gap fix is worth ≈ 0: M2's gap is already 2.2.
  - Contention is the unknown: R2's in-situ A at c4d3 was 90.85 vs M2's rot + A 90.69.
  - **R3 − M2 ≈ −1.5 ± 3 µs per layer ≈ −0.07 ± 0.14 ms per step (−0.4 ± 0.7 %): expect flat, possibly a small gain.**
  - P0 picks mode 2 or 3 for 13-16 rows under the chosen lever (`EXL3_MOE_COOP_V3_MAP=13-16:2` if mode 2 is faster there). E then also applies to mode 2, removing part of M2's in-situ contention (M2 A in situ 82.66 vs P0 67.5 at r16/D77).
- **c2d3 (8 rows, not a §16 shape):**
  - P0 r8/D40 kernel delta m3 − m2 = −2.9 (K2) / −3.7 (K3) µs.
  - The c8d0 (8 rows) R2 − M2 = −0.12 ms 5/5 (R703b) already shows the kernel gain surviving there.
  - Estimate −0.16 to −0.3 ms per step.

## 4. What this round ships (see HOW-TO-VERIFY.md)

**Flags (all default off, bitwise identical by construction):**
- `EXL3_SHARED_EXPERT_EARLY`, `EXL3_SHARED_EXPERT_PRIO`: construction-time, and they require `EXL3_SHARED_EXPERT_OVERLAP=1`, else a loud error.
- `EXL3_MOE_COOP_V3_HEAD`: per launch, mode 3 only.
- `exl3_moe_coop_ev`: a test entry with the served join event.
- Landing marker `moe_coop_v3_revision = 3`.
- r2's kernels and modes 1-3 are unchanged: `work/b` → `work/c` touches no kernel body.

**P0:** `bench_moefast_p0.py` now runs the real side-stream shared expert (`BC_GatedMLP` from the checkpoint's `shared_expert`, rotated over all 48 layers so its weights come from DRAM) and the real router (`ext.routing_std` on the layer's router weight) on the main stream.
- It reads the A→B gap and the shared start/end per call from the profiler timeline.
- **Instrument acceptance:** the m3 − m2 gap difference ≥ 5 µs at r4/D28, i.e. it reproduces R703b's 12 µs.
- `--no-side` gives r2's P0.

**Parity:**
- adds rows 6, 7, 9, 10, 11, 14, 15 (served c7d1 = 14, c5d2 = 15);
- a side-stream section: serial vs late fork vs early fork (router in between) vs high-priority side stream vs head, all through `exl3_moe_coop_ev`, bitwise against the serial V2 reference;
- a join stress: 200 back-to-back early-fork launches, where a missing wait would show as a stale `sh`.

**Gate** (`moefast-r3-gate.sh`):
- Arms OFF (V2) / M2 (served) / R3 = mode 3 + E [+ P] [+ MAP], picked by P0's pre-registered rule.
- 6 rounds, each arm first twice, a discarded warm-up run per round.
- d and d0 cells, medians, DECISION with the §16 gain clause and the c1 bound.
- fn_greedy of R3's exact env, gated at 0 divergences.
- nsys acceptance printout: R3 A→B gap ≤ 4 µs at c1d3 and c4d3.

## 5. Not covered, and who owns what

- The router chain itself (`hgemm` + topk) belongs to out-latchain. If they fuse or shorten it, E's head start shrinks, and P or H matter more. The fork point `start_shared(y)` stays valid (before whatever routing does).
- r2's A at 16 rows pays +4.4-5.9 µs for building the block-local run table in every CTA (`p0.log` r16/D77). That is a kernel-side item for a later round: build the table once, e.g. in a 1-CTA head that also serves as H's window.
- moe_timeline per-step normalisation counts multi-row A launches across both cards and the MTP draft (review defect 8). Per-layer numbers are unaffected; the ms/step columns are labelled estimates.
