# hcfast r1: why the HC mix chain costs what it costs, and a two-kernel fix

2026-09-24. This is a kernel change, and it keeps the served structure: `hc_apply` → `gr_v2_dots_i8` → `gr_v2_up_i8_state`, two mixer launches, grids ≥ the served ones. It sits behind `EXL3_HC_MIX_V3=1` (default off), and every output is bitwise-identical by construction. Nothing here ran on a GPU. The operator's steps are in HOW-TO-VERIFY.md.

## 0. Verdict

- **dots is bound by a redundant int8→fp32 conversion.** The served kernel runs one `I2F` per FFMA, which means the conversion is redone for every row a CTA owns. `I2F` issues on the XU pipe at 1/8 the FFMA rate. At R=16 the XU pipe is busy **56.8 %** of dots' active cycles. That is a throughput floor of about 19.5 k cycles, ~7 µs at the served clock (estimate: 1,658,880 I2F warp-inst / 680 SMSP × 8 cycles). Served dots at R=16 is 15.1 µs (GAP-TABLE §4), so the floor is about half of it.
- **up is bound by per-row serial work on few warps.** B LDS per 4-byte weight word and a 5-level butterfly per value load the LSU/MIO path (LSU **60 %** at R=16). The reduction is 16-27 % of stall samples. The weight words are first touched only after the state prelude.
- **R-scaling.** The grid is fixed for R ≤ 8 while rows per CTA (B) grow 1→8. Per-warp instruction count therefore grows ∝ B, and at 13-23 % issue-active each added instruction lands on a warp's serial path at ~15-18 cycles. From R=8 to R=16, B stays 8 and CTAs double, which is why the slope halves (1.7 → 0.9 µs/row). The pipes that saturate are XU (dots) and LSU (up). L2 traffic does grow ∝ R (dots 16.8 → 60.4 MB of L2→L1 reads), but L2 sits at 28-45 % and is not the bound.
- **The fix** (`hc_mix_v3.cu`):
  - dots converts each weight once per thread-iteration, with an exact PRMT+FADD conversion on full-rate pipes;
  - up reads the B state values of a rank index with B/4 `LDS.128`, reduces with a same-tree reduce-scatter (31 SHFL instead of 160 at B=8), and issues the weight words and epilogue operands first;
  - grids and CTA counts are unchanged at the default caps; ptxas reports 0 spills everywhere.
- **Estimate (not measured; revised down after R697, §7):** chain −15-30 % at R=4 and −23-42 % at R=16. That would take **~0.7-1.3 ms/iterate** off c4 d3 and c8 d1 (≈ 4-6 % of it_ms) and ~0.2-0.5 ms off c1 d3. The P0 bar asks for ≥ 25 % at both R=4 and R=16, and both estimates straddle it (§5, §7). R697 shows dots is latency-bound more than XU-bound, so removing the conversion is worth less than the XU floor suggested. The pre-registered bars are in HOW-TO-VERIFY §3.

## 1. Evidence used

| source | what |
|---|---|
| R687 ncu `--set full` (`flan:/srv/qwen5090/results/2026-09-24-r687-hcfuse-warm-ncu/ncu-raw.csv`) | rows 9-11 = `reference R=4` (hc_apply, dots<4>, up<4>), rows 33-35 = `reference R=16` (hc_apply, dots<8>, up<8>) |
| SASS source counters, exported CPU-only from `p0-ncu.ncu-rep` (`ncu --import … --page source --print-source sass --csv`, in `tabbyapi:slotfix-r1` without `--gpus`) | `flan:/srv/qwen5090/scratch/hcfast/src-{dots,up}{4,16}.csv`. Each file lists the kernel twice (the `Instructions Executed` sum is exactly 2 × `smsp__inst_executed.sum`), so the counts below are halved |
| P0 graph µs/chain, served chain | R684 cold 9.93 / 15.27 / 22.06 / 29.12, R687 warm 8.20 / 14.35 / 22.53 / 30.74 at R = 1/4/8/16 (`…/2026-09-24-r687-hcfuse-warm-ncu/audit.log`) |
| GAP-TABLE §2/§4 (`flan/decode-push-2026-09-24/roofline/GAP-TABLE.md:147-155`) | traced per-kernel µs: dots<1> 3.9, up<1> 5.1; dots<8> 15.1, up<8> 12.6 (grid y = 2); chains/step 97-108 |
| served sources | `src/exllamav3/exllamav3_ext/hc_mix.cu:996-1093` (dots_i8), `:1328-1468` (up_i8_state), `:1716-1746` (launch), `:1945-1948` (B rule); `modules/hyperconnections.py:412-431` |

Caveats. ncu ran at 1.63-1.98 GHz (`gpc__cycles_elapsed.avg.per_second`) against ~2.8 GHz served. It also flushes caches before every pass, so ncu µs are inflated and every first touch is DRAM. Cycle counts and ratios carry over; absolute µs do not.

## 2. Per-kernel bound (ncu, reference variant)

| metric | dots<4> R=4 | up<4> R=4 | dots<8> R=16 | up<8> R=16 |
|---|---:|---:|---:|---:|
| CTAs × threads, regs | 328 × 128, 66 | 320 × 256, 47 | 656 × 128, 94 | 640 × 256, 64 |
| warps active % of peak | 15.4 | 29.5 | 30.0 | 55.1 |
| issue active % | 13.2 | 22.7 | 23.2 | 41.9 |
| warp-inst (`smsp__inst_executed.sum`) | 1,418,916 | 2,816,115 | 5,355,184 | 9,504,356 |
| warp-inst per warp | 1,081 | 1,100 | 2,041 | 1,856 |
| active cycles (`sm__cycles_active.avg`) | 16,110 | 19,019 | 34,333 | 33,901 |
| active cycles per warp-inst per warp | 14.9 | 17.3 | 16.8 | 18.3 |
| **XU pipe % of active** | **30.3** | 10.2 | **56.8** | 14.4 |
| **LSU pipe % of active** | 12.2 | 27.9 | 21.8 | **60.0** |
| FMA pipe % | 5.3 | 8.6 | 9.5 | 17.5 |
| L2→L1 read (`lts__t_sectors_srcunit_tex_op_read` × 32 B) | 16.8 MB | 7.0 MB | 60.4 MB | 21.5 MB |
| L2 throughput % | 28.4 | 10.7 | 45.4 | 14.0 |
| DRAM read (cache flushed) | 3.50 MB | 3.54 MB | 4.00 MB | 4.10 MB |
| stall/issue: long_sb, short_sb, barrier, mio | 6.3, 2.1, 2.0, 0.4 | 8.4, 1.5, 2.0, 0.5 | 6.7, 2.2, 1.7, 1.6 | 5.5, 1.9, 1.2, 2.5 |

Every grid is a single wave (`launch__waves_per_multiprocessor` 0.28-0.94). Warps per SMSP are 2-4 (dots) and 4-7 (up).

### dots

- **Conversion redone per row.** Source counters give `I2F.S8` 414,720 and `FFMA` 416,000 at R=4, and 1,658,880 and 1,664,000 at R=16. The ratio is 1.00: the served code writes `(float) q[k]` inside the `if (r0 + b < R)` row loop (`hc_mix.cu:1050-1068`), and nvcc did not hoist it. In `up` the compiler did hoist it (I2F:FFMA = 0.24).
- **The XU arithmetic checks out.** XU executes one warp-instruction per 8 cycles per SMSP. 414,720 / 680 SMSP × 8 = 4,879 cycles = 30.3 % of 16,110 (ncu: 30.29 %). 1,658,880 / 680 × 8 = 19,516 = 56.8 % of 34,333 (ncu: 56.84 %).
- **Stall samples (R=16).** FFMA waiting on stream loads (long_sb) accounts for 28 % of samples. FFMA waiting on I2F results (short_sb) plus I2F queue-full (mio) accounts for 19 %. `EXIT` is 20 %: warps 2-3 have 2 column iterations and warps 0-1 have 3 (320 column groups over 128 threads), so they wait on the CTA tail.
- **Streams are re-read by all 81 j-CTAs of an h.** L2→L1 traffic = 81 × R × 40 KB + weights ≈ 3.3 MB × (R + tiles) (estimate). That gives 16.6 / 59.7 MB, against 16.8 / 60.4 MB measured. L2 at 28-45 % is not the bound at R ≤ 16.

### up

Stall samples by region, split at the kernel's barriers (source counters):

| region | R=4 | R=16 |
|---|---:|---:|
| rmr + state prelude (derive B × 324 state values from `dots`) | 42.3 % | 41.4 % |
| rank loop (10 iterations: 1 LDG.32 + B LDS + 4B FFMA) | 31.3 % | 23.6 % |
| butterfly reduction + scale + gates | 16.2 % | 27.2 % |
| epilogue | 7.0 % | 6.6 % |

- **The prelude share is inflated by ncu's cache flush.** `dots` is DRAM-cold in every ncu pass. In decode it was written by the preceding kernel and is L2-resident. The served SASS unrolls the prelude k-loop by 2, so B=8 still pays about 5 dependent load rounds. Its served-time share is not measured. V3 cuts it to 1-2 load rounds anyway (§4), but the estimate below does not count on it.
- **The rank loop issues B `LDS` per 4-byte weight word.** At R=16 that is 1,000,964 LDS (halved) against 620,816 LDG. LSU runs at 60 % and `mio_throttle` is the #2 stall.
- **The reduction is a full butterfly on 4B values.** At B=8 that is 160 `SHFL.BFLY` + 160 `FADD` per warp. `SHFL` alone is 179 of 1,404 samples at R=16, mostly mio.
- **Weight words are loaded only after the prelude barrier.** At R=4, the first-use `I2F` waits long_sb on the up_q load in 155 of 789 samples.

## 3. Why time grows with R

Served graph µs/chain: 9.93 → 15.27 → 22.06 → 29.12 at R = 1/4/8/16, i.e. +1.78, +1.70 and +0.88 µs per added row (R684 cold).

1. **R ≤ 8: fixed grid, serial rows.** Rows per CTA are B = 1, 2, 4, 8 (`hc_mix.cu:1945-1948`). The grid stays 328 (dots) and 320 (up) CTAs, 1.9 CTAs/SM. Each warp loops over all B rows of its CTA, so warp-instructions per warp grow with B: 1,081 at B=4 and 2,041 at B=8 for dots, 1,100 and 1,856 for up. At 13-23 % issue-active a warp runs at ~15-18 active cycles per instruction (the table). So time grows almost linearly in per-warp instructions, even though bytes do not grow.
2. **What a row costs per warp.**
   - dots, per column iteration: 2 `LDG.128` of fp32 streams + 32 FFMA + **32 I2F**; 2.5 iterations; plus 4 × 5 × 2 reduction ops.
   - up: 10 × (1 LDS + 4 FFMA), plus 4 × 5 × 2 butterfly ops, plus 324 / 256 × ~20 prelude ops.
   - In dots the I2F share is set by throughput, not latency: the XU pipe is a 4-lane unit. At R=16 it alone accounts for 19.5 k of 34.3 k active cycles.
3. **R 8 → 16: B stays 8 and grid.y doubles.** Per-warp work is unchanged, but twice the warps share each SM's XU (dots) and LSU/MIO (up) pipes: XU 57 %, LSU 60 %, `mio_throttle` 1.6-2.5 per issue. The slope halves instead of vanishing.
4. **"Row scaling comes from on-chip traffic" (MEGAKERNEL-PLAN §5.2) is now measured.** L2→L1 bytes do grow with R (dots 16.8 → 60.4 MB). But L2 throughput is 28-45 %, while XU is 57 % and LSU 60 %. The row cost is instruction issue on few warps plus two narrow pipes, not L2 bandwidth.
5. **Weights are not the bound.** This agrees with R687: L2-warm only moves 15.27 → 14.35 µs at R=4, and at R=16 warm is slower (30.74 vs 29.12). DRAM reads per kernel are ~3.5 MB, which is 2 µs at 1.8 TB/s, against 7-15 µs kernels.

hcfuse r1 failed for the reason the review gave: it cut phase-A CTAs to 88. It also kept the per-row I2F: its fused dots path converts inside the row loop too (`hcfuse-r1.patch:302-317`; `hcb.sass` has 14,643 static `I2F` against 23,066 `FFMA`). Neither is repeated here.

## 4. The change (`hc_mix_v3.cu`, `EXL3_HC_MIX_V3=1`)

Two launches, same grid shapes, same 128/256 threads. In each kernel, each thread computes the same accumulators from the same operands in the same order, and the reductions use the same lane/warp trees. What changes is instruction selection and scheduling. `hc_mix.cu` is not touched. The flag-off path is the served code, plus one class-attribute test in `_mix`.

| # | kernel | change | removes | bitwise because |
|---|---|---|---|---|
| D1 | dots | weight bytes converted once per thread-iteration, outside the row loop, by `__byte_perm(w ^ 0x80808080, 0x4B000000, 0x7440 \| k)` → `float − 8388736.0f` (PRMT + FADD) | (B−1)/B of the conversions, and all XU use (static SASS: `I2F` 256 → 0 at B=8) | 2²³ + (q + 128) − (2²³ + 128) = q exactly for every int8, and q = 0 gives +0.0f as `(float) q` does. The operand of each `fmaf` is the same float |
| D2 | dots | `fn_s` loaded before the main loop | the post-barrier `LDG → FMUL` tail | same product `sum * fn_s[j]` |
| D3 | dots | D = 2560 as a template constant; column loop fully unrolled (B ≤ 4) with the next iteration's weight words double-buffered; per-row / per-j branches replaced by zero-filled operands for out-of-range rows | 1-2 exposed DRAM latencies at B ≤ 4; the branches around every 8-FFMA group | per-thread column set and order unchanged (`c = tid + 128 k`, k ascending). Out-of-range slots are never written |
| U1 | up | all 10 `up_q` words per lane issued at kernel entry | the post-prelude DRAM wait | same words |
| U2 | up | state prelude: 8/B warps per row, one `LDG.128` for the 4 h values of an element, rmr broadcast by `__shfl_sync` (no barrier), all of a lane's loads issued before use | 3 of 4 load instructions and 4-5 dependent load rounds at B=8 | per element the served expression verbatim: `rsqrtf(x / (float) D + eps)` with runtime D, `fmaf` h-chain from 0.0f, `v *= 1/H`, `v * sigmoidf_(v)`, `2 * sigmoidf_(v)` |
| U3 | up | state stored transposed, `headT[i][b]` (row stride 12 at B=8, 4 at B=4: conflict-free `LDS.128`) | B → B/4 LDS per weight word (static: 39 → 25 LDS at B=8) | same values, same `fmaf(t, q, g)` order |
| U4 | up | reduce-scatter over the same xor offsets 16, 8, 4, 2, 1 | `SHFL` 20B → 4B−1 + (5 − log₂4B) (160 → 31 at B=8; static SASS 160 → 35 incl. the rmr broadcast) | each value's sum is formed over the same lane pairs in the same level order. IEEE `a+b == b+a`, and a butterfly already leaves identical sums in every lane, so the lane that keeps a value holds the butterfly's result |
| U5 | up | epilogue `streams`/`w` loaded before the reduction; head's rmr tail read from `rmr_s` | the post-barrier L2 round trip | the same values (`head[b][LR+h]` *is* `rmr_s[b][h]` in the served kernel) |

**Registers and spills** (`ptxas -v`, sm_120, `--use_fast_math -O3`, the extension's flags; `flan:/srv/qwen5090/scratch/hcfast/ptxas-v3.log`):

| kernel | served regs | V3 regs | spills | CTAs/SM (reg-limited) | grid at served R (one wave?) |
|---|---:|---:|---:|---:|---|
| dots B=1/2/4/8 | 44 / 56 / 66 / 94 | 60 / 64 / 128 / 128 | 0 | 8 / 8 / 4 / 4 | 328 (R≤8) ✓; 656 (R=16) ✓ ≤ 680 |
| up B=1/2/4/8 | 40 / 40 / 47 / 64 | 60 / 62 / 64 / 64 | 0 | 4 at every B | 320 ✓; 640 (R=16) ✓ ≤ 680 |

dots B=8 keeps its column loop rolled. Fully unrolled, ptxas spills 8 bytes at the 128-register budget. The copied served kernels compile to the served register counts (94/66/64/47) and the served SASS length (1,432 / 1,240 instructions for dots<8> / up<8,half>, the same as the ncu listing). That is the evidence the copies are the served code.

**Tile caps (exploratory, bitwise by construction).** `EXL3_HC_MIX_V3_DOTS_B` and `EXL3_HC_MIX_V3_UP_B` cap B (8 = served tiling, the default). `0` = auto: the smallest B ≤ the served one whose whole grid fits one wave on the device, from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, cached per device. A smaller B only changes which rows a CTA owns, and it can only raise the CTA count. At R=4, dots B=1 means 1,312 CTAs of 60 registers (1,360 slots, one wave), each doing R=1's per-warp work. P0 measures whether that trade is faster: 4× the weight L2 reads for ~4× the warps.

**Not done, and why.**
- Split-K / J=8: J=8 halves dots CTAs, and a warp-granular split changes the `dots` layout. The streams re-read is not the bound (§2).
- `fma.rn.f32x2`: halves FFMA issue, but FMA is at 5-17 % of peak, so it is not the bound.
- PDL (programmatic dependent launch) to prefetch up_q during dots: DRAM is not the bound (R687 warm ≈ cold), and it adds a launch-attribute and graph-capture risk. It is the next lever if P0 shows the V3 chain at ≲ 2× the DRAM floor.
- Changing `hc_apply`: 2.5-3.2 µs under ncu at 2-5 % issue, 80-160 single-warp CTAs. It is a launch-latency item and is not in this chain's bound.

## 5. Estimates (labelled; P0 decides)

Per kernel at the served clock. Served numbers come from GAP-TABLE §4 (R=1, R=16) and from the ncu time split applied to P0's chain (R=4: dots ≈ 6.6, up ≈ 7.5, hc_apply ≈ 1.2 µs; estimate).

| R | served dots / up / chain µs | V3 dots (estimate) | V3 up (estimate) | V3 chain (estimate) |
|---|---|---|---|---|
| 1 | 3.9 / 5.1 / 9.9 | 3.5-3.8 (no redundancy at B=1; XU → ALU) | 4.0-4.6 (U1, U2, U4) | 8.5-9.5 (−5-15 %) |
| 4 | ~6.6 / ~7.5 / 15.3 | 4.5-5.5 (XU was 30 %) | 4.5-6.0 | 10.5-13 (−15-30 %) |
| 16 | 15.1 / 12.6 / 29.1 | 9-12.5 (revised after R697; was 7-9) | 6.5-8.5 | 17-22.5 (−23-42 %) |

Arithmetic for R=16 dots: served 15.1 µs, of which the XU issue floor is 19.5 k / 34.3 k active cycles, i.e. 57 %, i.e. up to 8.6 µs. Whatever the XU does not overlap comes out, and the latency paths remain. Assuming 50-80 % of the XU floor is recovered gives 7-9 µs. **Revised after R697 (§7):** the source-level stall split at R=16 puts only ~17 % of dots' stall samples on the conversion (I2F mio 6.7 % + FFMA short_sb 10 %), with long_sb 51.5 % and the EXIT barrier 14.1 %. Removing the conversion (−17 %) and batching the loads (part of the long_sb) gives 9-12.5 µs (estimate: 15.1 × 0.83 = 12.5 at the low end of the gain; 9 if batching halves the exposed load wait).

Per iterate: GAP-TABLE §4 gives 108 chains/step at c4 d3 and 101 at c8 d1 (R=16), and 107 at c1 d3 (R≈4). With the R=16 saving at 6.7-12 µs/chain (revised; estimate: 29.1 − 22.4 to 29.1 − 17) that is **0.7-1.3 ms/iterate at c4 d3 and c8 d1**; with 2-5 µs/chain at R=4 it is **0.2-0.5 ms at c1 d3**. The review found P0 deltas × chains/step reproduce P1 within ~25 % (review-r682-r684 §R684).

## 6. What each P0 variant discriminates

| question | P0 variants that answer it |
|---|---|
| does the binding/copy add anything? | `copy` vs `reference` (control; must be within ±3 %) |
| which kernel carries the gain? | `v3-dots` (mode 1), `v3-up` (mode 2), `v3` (mode 3) |
| is row-serialization or pipe pressure the residual bound? | `v3-d1/d2/d4`, `v3-u2/u4`, `v3-auto` vs `v3`. If smaller tiles win at R=4 and lose at R=16, the residual is latency; if they lose everywhere, it is pipe/L2 throughput |
| is the V3 chain now DRAM-bound (then PDL prefetch is next)? | `--sites 1` (L2-hot) vs the default 24 cold sites: a hot/cold gap > 30 % at R=4 would say yes |
| what bounds V3 after the change? | the optional ncu pass in HOW-TO-VERIFY §3 (same metrics as this table for `v3 R=4` / `v3 R=16`) |

## 7. R697 check: which of the measured stalls the patch hits

The R697 data (`out-hcfast/r697/`) is source-level ncu of the served chain, from `bench_hcfuse_p0 --ncu` on card 0 with `--cache-control all` (cold, so the times are upper bounds) and `--clock-control none`. Kernel times at R=4 / R=16 (`r697/ncu-raw-nvtx.csv` IDs 9-11 / 21-23):

- dots 8.2 / 16.0 µs, with 15 / 31 % warps active;
- up 9.3 / 15.8 µs, with 30 / 57 % warps active;
- hc_apply 2.1 / 2.5 µs.

The stall shares below are from `python3 r697/src_an.py r697/ncu-source-sass.csv 18,20,22,42,44,46`, which covers source blocks 20 / 44 (dots) and 22 / 46 (up), as a share of the kernel's stall samples. The V3 load order is from `sass-load-order.txt`, the static SASS of the same build as `ptxas-v3.log`.

**What R697 changes in §0.** dots is latency-bound more than it is XU-bound. One caveat: R697 ran with `--cache-control all`, so every first touch, including the stream rows that hc_apply has just written, comes from DRAM, and long_sb is inflated against the served L2-warm chain. P0 warm ≈ cold (R684 15.27 / 29.12 against R687 14.35 / 30.74 µs at R=4 / 16) says chain time barely depends on this, but in the warm chain the latency-versus-XU split is likely less lopsided than 51.5 % against 17 %. It does not follow that the XU was never a cost. At R=16 the conversion's own stalls are about 17 % of samples (I2F mio 6.7 % + FFMA short_sb 10 %). Load waits (long_sb 51.5 %) and the EXIT barrier (14.1 %) are the larger part. The XU-floor argument in §0 and §5 overstated what D1 buys at R=16, so the R=16 estimate is revised from −35-48 % to −23-42 % (§5).

| R697 stall (served) | share | what it is | patch item | attacked? |
|---|---:|---|---|---|
| dots R=4: `I2F.S8` long_sb (first use of a weight word) | 26.5 % | wait on the weight `LDG`. The served loop issues 2 loads and consumes them at once, then again for each j (`sass-load-order.txt`, served_dots<4>: `LDGx2 … I2F FFMA …` repeated) | D3 + D1: V3 dots<4> issues 21 loads back-to-back before the first FFMA, then 7, then 8 (`LDGx21 FFMAx10 LDG FFMAx198 LDGx7 FFMAx48 … LDGx8 FFMAx128`). The next iteration's weight words are double-buffered | **yes** (lever 1), as three load rounds instead of one per j |
| dots R=4: FFMA long_sb | 20.7 % | wait on the stream `LDG.128` | the same batching: streams for all B rows of an iteration are issued together | **partly**: the first round of each iteration is still exposed |
| dots R=4: short_sb | 14 % | FFMA waiting on the I2F result (XU) | D1: PRMT+FADD on ALU/FMA pipes, 1 conversion per B FFMAs | **yes** |
| dots R=4 / R=16: EXIT barrier | 12.4 / 14.1 % | 320 column groups over 128 threads: warps 0-1 run 3 iterations, warps 2-3 run 2 | D2 (fn_s prefetch) only shortens the tail after the barrier | **no** (lever 3), see below |
| dots R=16: FFMA long_sb | 37.7 % | stream-load wait at B=8 | The served dots issues the weights at the top of an iteration and then 2 stream loads per row, each consumed at once (served_dots<8>: `LDGx4 BRA LDGx2 BRA I2Fx2 FFMA …` repeated). That is ≈ 1 + B exposed load rounds per iteration: ~15 at B=4 and ~27 at B=8 over 3 iterations. V3 dots<8> keeps a rolled loop and issues 9-14 of its 20 loads before the first FMA, then 7 more between FMA groups (`LDGx5 BRA LDGx9 FFMAx23 LDG FFMAx12 LDG …`). That is ≈ 2-3 rounds per iteration, ~9 in total, with no cross-iteration prefetch, because a second buffer does not fit in 128 registers. V3 dots<4> needs 1 round per iteration, 3 in total | **partly**: ~3× fewer exposed rounds at B=8 |
| dots R=16: I2F long_sb + mio, FFMA short_sb | 12.5 + 6.7 + 10 % | weight wait, then XU queue and result latency | D1 removes the I2F. The weight wait moves to the PRMT and is batched as above | mio and short_sb **yes**; the load wait partly |
| up R=4: `I2F.S8` long_sb in the rank loop | 14.3 + 7.0 % | up_q words loaded only after the prelude barrier (served_up<4>: `BAR LDG LDSx4 I2F…`) | U1: all 10 words issued at entry (V3 up<4>: `LDGx2 MUFU LDGx10 …`, before the prelude). The conversion is also PRMT+FADD in up (`hc_mix_v3.cu:348`), so V3 up has 0 I2F | **yes** (lever 1) |
| up R=4 / R=16: prelude FFMA-RZ long_sb | ~28 / 28.9 % | the per-element `dots` loads of the state prelude, one dependent round per row (served: `LDGx4 LDS FFMAx4` ×4 serial) | U2: one `LDG.128` per element, 8/B warps per row, all of a lane's loads issued before use (V3 up<8>: `LDGx8 FFMAx4 …`) | **yes** (lever 1 + more warps on the prelude) |
| up R=4: BSSY/LDCU barrier | ~9.5 % | the prelude → rank-loop barrier | U2 shortens the prelude that the barrier waits for. The barrier stays | partly |
| up R=16: SHFL | 11.2 % (mio 7.1) | 5-level butterfly per value, 160 SHFL | U4: reduce-scatter, 35 SHFL at B=8 | **yes** |
| up R=16: LDS / mio overall | 6 / 12.7 % | B scalar LDS per weight word | U3: transposed head, LDS.128 (39 → 25 static LDS) | **yes** |
| hc_apply: HADD2 long_sb | (2.1-2.5 µs total) | y load | not touched | no |

**Levers named in the R697 message:**

1. **More independent loads outstanding per thread: attacked at every B, fully only at B ≤ 4.** At B ≤ 4, V3 dots issues the whole iteration's loads (4 weight `LDG.64` + 2B stream `LDG.128`) before the first FMA and also prefetches the next iteration's weights: 3 exposed load rounds against the served ~15. At B=8 the compiler splits the 20 loads into 9-14 before the first FMA and 7 between FMA groups: ~9 rounds against the served ~27 (`sass-load-order.txt`). In up, U1 and U2 do the same for up_q and the prelude. Two gaps remain:
   - the stream loads are never prefetched across iterations;
   - the weights at B=8 are not prefetched across iterations either (register budget).

   The literal "16-byte int8 loads" variant is not used, and that is deliberate. Each thread's weight access is 8 bytes per j row (`c = tid + 128 k`, 8 columns). A 16-byte load would give each thread 16 columns, which changes the column-to-thread assignment and therefore the fp32 summation order. It is not bitwise.
2. **More warps for the same work: attacked through the tile caps, not split-K.** `EXL3_HC_MIX_V3_DOTS_B` / `_UP_B` and auto are bitwise and never lower the CTA count.
   - At R=4, auto chooses dots B=1: 1,312 CTAs of 60 registers, each warp doing R=1's work.
   - At R=16, auto keeps B=8, because B=4 would need 1,312 dots CTAs against 680 slots (4 CTAs/SM at 128 registers) and 1,280 up CTAs against 680. The explicit caps `v3-d4` / `v3-u4` in P0 measure the two-wave trade.

   Split-K or a wider CTA would add warps without the extra weight re-reads, but it changes which thread sums which columns, so it is not bitwise.
3. **EXIT-barrier imbalance in dots: not fixed.** Warps 2-3 finish after 2 iterations and wait for warps 0-1, which run 3. Every rebalancing tried moves columns between threads, which changes each thread's partial sum and the `shfl_down` / `red` tree operands. Those options are 160 threads for 2 iterations each, spreading the tail columns, or letting warps 2-3 take part of warps 0-1's work. The result is then not bitwise. The patch only moves the `fn_s` load ahead of the loop (D2), so the post-barrier tail is one FMUL and one STG.

**Next levers, in order (not in r1, not bitwise).** Each would need a tolerance-based parity gate, which is the user's call because the brief asked for bitwise outputs:
- (a) a 160-thread dots CTA (5 warps × 2 iterations), which removes the 12-14 % EXIT tail;
- (b) a stream-row prefetch across iterations at B=8, with the rows split over two CTAs;
- (c) a column-split dots with a two-stage reduction, for R=16 warps.

**Consequence for the P0 bars.** Both of the brief's bars (≥ 25 % at R=4 and at R=16) now sit inside the estimate ranges (R=4 −15-30 %, R=16 −23-42 %). A MARGINAL result is a real possibility. If P0 lands there, the `v3-dots` / `v3-up` split and the ncu pass in HOW-TO-VERIFY §3 show which remaining row of the table above is binding. If EXIT and the stream long_sb dominate the V3 dots profile, the next step is (a) or (b) above.
