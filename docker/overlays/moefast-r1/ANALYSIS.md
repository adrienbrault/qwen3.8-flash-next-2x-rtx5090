# ANALYSIS.md (moefast r1)

Written by the MoE-fast Opus sub-agent (2026-09-24). The operator saved it from the agent's final report, because the agent could not write .md files.

## Question

The routed-expert MoE coop kernels (`exl3_moe_coop_v2_ns::exl3_moe_coop_{a,b}_kernel<K,2,WIDE>`, served with `EXL3_MOE_COOP_V2=1`) sit well above the byte floor: 2,824 µs over it at c1d3 and 4,491 µs at c8d1. This analysis locates the time and names the smallest bitwise-identical change that removes the dominant stall.

## Evidence

**R682** (d0 microbench on real weights, events and ncu).

rot + a + b per cell:

| cell | µs |
| --- | ---: |
| K2 r4 D10 | 44.7 |
| K2 r4 D20 | 57.7 |
| K2 r4 D28 | 66.7 |
| K2 r4 D40 | 97.4 |
| K3 r4 D28 | 78.9 |
| K2 r16 D77 | 130.1 |
| K3 r16 D77 | 156.7 |

ncu at K2 r4 D40:
- a: 52.8 µs, 33.4 MB DRAM, issue-active 58 %.
- b: 50.3 µs; barrier 6.15, long_sb 4.33, membar 1.90.

The A kernel's decomposition estimate stays about 27-31 µs across the D sweep. So it is latency-bound, not byte-bound, which is consistent with R650's flat e2e, R536 and R419.

**Served SASS** (quick-nvcc of the slotfix-r1 sources).
- Each k-slice of `gemv_tile_v2` issues three loads:
  1. `LD.E` for the activation (generic, because A2 may be `sh_A` at bsz 1);
  2. `LDG.E.EF`, the weight prefetch for slice i+4;
  3. another `LD.E`.
- All three carry `&wr=0x5`, and the first HMMA carries `&req={5}`. The MMA of slice i therefore waits on the fresh DRAM prefetch of slice i+4 as well as on its activation, so the 4-deep ring has an effective distance of zero.
- This settles the review's open "scoreboard sharing" question: they share SB5.

**R696 source-level ncu** (synthetic weights, `--cache-control all`), read per `review-r696-2026-09-24.md`:
- **K2 r4 a:** long_sb is 43.5 % of samples. 37.0 % sits on the activation `LD.E` (e.g. `1d990 LD.E R20` → `1dc50 HMMA`, 421 samples) and 2.6 % on weights. The weight LDG shares SB5, so the HMMA wait covers both.
- **K3 r4 a:** 27.9 % sits on the weight `LDG.E.EF`, consumed by a MOV 20-25 instructions after issue (`45ab0 LDG.E.EF R42` → `45bf0 MOV R44,R42`, 9.7 %). The conditional K3 load is compiled with a branch into a temp register and then copied, so the ring collapses. Activation is 8.8 %.
- **r16:** K2 a is 39.6 % activation; K3 a is 30.5 % weights.
- **B at r4 (narrow 32-column items):**
  - The per-item handoff costs 26.6 % (K2) and 23.3 % (K3). It is the `__syncthreads` at l.429 behind one warp's slot scan plus the returning atomic, plus two all-warp `__threadfence` (MEMBAR.SC + ERRBAR + CCTL.IVALL, 7.7-8.8 %).
  - The k-split imbalance at the reduction barrier adds 8.7-10.3 %: 40 slices over 16 k-warps splits as 13×3, 1×1, 2×0.
- **B at r16 (wide):** handoff 13-14 %, activation 26.1 % (K2).
- **Magnitude bound:** long_sb samples on no-issue cycles are only 15.7-22.5 % in A. Activation not-issued is 13.2 % at K2r4-a; K3 weight not-issued is 11.7 %.
- **DRAM throughput of the A kernel:** 35.7 % (K2r4), 45.1 % (K3r4), 65.0 % (K3r16). r4 is latency-bound; K3 r16 is nearer to bandwidth.

## Step 1 — where the time goes

- **A kernel** (40-45 % of MoE µs at r4, most at r16): long_scoreboard on per-slice operand loads.
  - K2: the unprefetched activation fragment, plus the SB5 sharing with the weight prefetch.
  - K3: the weight ring collapsed by the compiler copy.
  - The gather (routing/pointer metadata) is small in A: 1.8-5.8 %, flush-upper-bound.
- **B kernel at r4:** per-item overhead — the handoff (fences, slot-scan barrier, atomic), the reduction imbalance and the metadata chain.
  - Instructions per weight are 0.54, against A's 0.29 (R682).
  - KSPLIT2 costs +26 µs at K2 r4 D28 (66.7 → 92.7).
- **B kernel at r16:** activation latency plus a smaller handoff.

## Step 2 — design (V3, `EXL3_MOE_COOP_V3=1|2`)

Bitwise identity forbids any geometry change: WN/WK tiling, k-partition, fold points and the order of the fp32 cross-warp sums all stay. V3 changes only where operands come from, when loads are issued, and memory ordering.

1. **Weights go through a per-warp cp.async ring in shared memory** (`V3_PF=4` slots).
   - `cp.async.wait_group(V3_PF-2)`, then `__syncwarp`, then the refill is issued, so 3 slices are outstanding.
   - SASS: `LDGSTS` → `LDGDEPBAR` → `DEPBAR.LE SB0, 0x2`.
   - There is no register ring, so no K3 copy: 0 `LDG.E.EF` in the V3 functions.
   - Each lane reads from the ring exactly the words V2's shuffles gave it.
2. **Activations via `ld.global.nc`, one slice ahead, issued after the current slice's MMAs.**
   - These are `LDG.E.CONSTANT` on their own scoreboard. The HMMA waits only on a load that is one slice old (about 160 instructions of its own warp).
   - Staging the rows in shared memory was tried first: 41 KB at 8-row runs plus the ring broke 2 CTAs/SM at K3.
3. **Counter handoff:** `__syncthreads(); atom.add.acq_rel.gpu` by the counting threads; `__syncthreads()`.
   - This replaces V2's all-thread `__threadfence()` before and after the atomic.
   - SASS: one `MEMBAR.ALL.GPU` inside the counting branch, and no `MEMBAR.SC.GPU` or all-warp `CCTL.IVALL`.
4. **Mode 2 only: narrow B merged per 128-column chunk.**
   - The 4 narrow 32-column groups become one item. Warp w computes V2's partial for group `w&3` and k-warp `(w>>2)+4q`, q = 0..3.
   - The running fp32 sum is kept in rounds of 4 k-warps, in V2's k-warp order, in a 16 KB round buffer.
   - Each 128-column chunk needs 1 run-table read and 1 handoff instead of 4, and the counter adds arrivals=4.
   - The warp imbalance ratio is unchanged: max 12 vs mean 10 slices, the same 1.2 as 3 vs 2.5. No bitwise-identical assignment does better at unit granularity.

**When V3 engages:** only at bsz > 1, K_gu and K_d in {2,3,4}, on sm_120, on top of V2. The flag is read per launch. With it unset the served kernels are untouched: 193/193 served coop functions are SASS-identical after the full rebuild.

**Cost:**
- Instructions per slice:
  - A K2: 156 → 162;
  - A K3: 180 → 185;
  - B narrow K2: 152 → 165;
  - merge K2: 152 → 195 (+28 %, pointer bookkeeping).
- Registers ≤ 64 with 0 spills (the served A kernels spill 8-52 B).
- `__launch_bounds__(512,2)`.
- Dynamic smem 24/28/32 KB for K2/3/4, which keeps 2 CTAs/SM.

## Not in r1 (r2 candidates, gated on the P0 ncu of the V3 arm)

- Precompute each row's active-slot count, which takes the one-warp 10-slot scan out of the l.429 critical section.
- Hoist item i+1's metadata chain (runs → sel/rw → trellis pointer) above item i's handoff.
- Have A stage its own activation rows, so the rot kernel goes (2.9-4.0 % of MoE µs, R696 raw).
- A fused a→b handoff.

The discriminator is the V3-arm SourceCounters from gate step 2b. If B r4 is still barrier-bound at l.429, the slot-count precompute comes next.

## Predictions (estimates, labelled)

Each range is bracketed by a latency model (every slice pays one DRAM/L2 round trip) and the R696 not-issued bound.

| cell | m1/ref | m2/ref |
| --- | --- | --- |
| K2 r4 D28 | 0.85-0.92 | 0.72-0.85 |
| K3 r4 D28 | 0.80-0.90 | 0.70-0.82 |
| K2 r16 D77 | 0.75-0.90 | same as m1 (B is wide at r16, so the merge does not apply) |
| K3 r16 D77 | 0.72-0.88 | same as m1 |

Predicted step savings:
- c1d3: about 0.4-1.1 ms;
- c8d1 and c4d3 (r16): about 0.6-1.7 ms.

The e2e conversion is uncertain: R650's flatness and shared-expert overlap contention may absorb part of any kernel win.
