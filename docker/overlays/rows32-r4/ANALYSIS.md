# rows32 r4: analysis (2026-09-24)

Written by the rows32 Opus sub-agent before the gate ran; saved by the operator from its final report. The gate became R717 (driver [`scripts/r717-rows32-r4.sh`](../../../scripts/r717-rows32-r4.sh)); the measured outcome is in [`bench/results/r717-rows32.md`](../../../bench/results/r717-rows32.md).

rows32 r4 is rows32-r3 rebased onto `tabbyapi:stack-r3` (INCLUDE "mf3 dg2": hcfast-r2, latchain-r1b RR/QT/QF, moefast-r3, densegemm-r2 mode 1, 06-dense-lcguard). It adds 4b `EXL3_MOE_COOP_ROWS32` and 4c `EXL3_SHARED_EXPERT_ROWS32` on top of the 4a dense twins (`EXL3_DENSE_ROWS32`) already in densegemm-r2. With those it serves the policy `[[4, 3], [8, 2]]`: c6, c7 and c8 verify at depth 2, which is 18, 21 and 24 rows. The served policy `[[4, 3], [5, 2], [8, 1]]` verifies c6-c8 at depth 1, 12-16 rows. c1-c5 keep their shapes.

## 1. What changed from r3

| item | r3 | r4 |
|---|---|---|
| base | stack-r2 + densegemm-r2 | stack-r3 series INCLUDE "mf3 dg2" (fuzz 0, by exit code, `verify-patch.sh`) |
| `moe_rows32_revision` | 1 | 2 (Python refuses < 2 with "rows32 r4 extension") |
| `start_shared` | slices `out_d_sh` before any bound check | `TORCH_CHECK` on the `num_tokens` range and the scratch size, before the slice (moefast-r3's shared-expert early fork added this entry point) |
| lcguard side region | 2 MiB (stack-r3) | 4 MiB when `EXL3_DENSE_ROWS32` is on, else 2 MiB; a refused side call is logged once per device (`densegemm lcguard: device ... served kernel`) |
| MoE MAP | not needed | B arm `EXL3_MOE_COOP_V3_MAP=2-4:2,17-32:2` (§3) |
| parity rows | 17 / 24 / 32 | dense, hcfast and rows32 at 1 4 8 16 17 18 21 24 32; latchain at (6,3) (7,3) (8,3) |

The patch touches 9 files and is host-only. SASS identity requires 0 changed and 0 added functions. Every line r3 added is still in r4 except the revision and message bumps; `verify-patch.sh` prints that diff.

## 2. Expected gain at c6-c8 (the R708 economics model re-based on stack-r3)

The model, `econ_model_r4.py`, imports the R708 model unchanged and makes two changes to it. Neither file is in this repository; the tables below are their output.

1. **S1** (the served step at the served depth) is lowered by stack-r3's saving at 12-16 rows. The pre-registered δ is 1.0 ms, with a range of 0.7-1.3. It is built from three parts:
   - HC2 + RR + QT + QF: about −0.45 ms at c8d1 (stack-r3 ANALYSIS §5).
   - densegemm-r2 mode 1: about −0.35 ms of dense critical path at 16 rows. R714 P0 went from m0 2435.7 to m1 2089.3 µs/step, and the R714 P1 c8d1 median of pairs was −0.37 ms.
   - moefast-r3: about −0.22 ms (R713, −1.2 % at c8d1).

   Replace δ with R716b's measured U-vs-OFF c8d1 cell when it lands (`--delta-s1`).
2. **Dense residual above 16 rows.** The R708 model assumed 0.3 ms after 4a. On stack-r3 the A arm already runs mode 1 at ≤ 16 rows, and mode 1 gains nothing at 17-32 rows (m1 = m4 there). So the residual is R714 P0's m1 critical path at the new row count minus the A arm's at the served row count:

   | c | rows served → rows32 | crit m1 µs/step | residual |
   |---|---|---|---|
   | 6 | 12 (interp.) → 18 | 1994.5 → 2633.9 | +0.64 ms |
   | 7 | 14 (interp.) → 21 | 2041.9 → 2631.6 | +0.59 ms |
   | 8 | 16 → 24 | 2089.3 → 2642.7 | +0.55 ms |

   The two spilling TN256 M32 mgemm twins are already inside these P0 numbers.

Scenarios:
- **mid-s3:** the R708 model's mid (row 0.25, pass 0.75, MoE ×1.10) with the per-c residual.
- **mid-s3f:** the same with the flat 0.55 ms the R708 model's addendum names.
- **pes-s3:** pes (0.30 / 0.90 / ×1.25) with the per-c residual, plus the shared expert's growth from 16 to 18-24 rows (+0.14 to +0.20 ms, R714 P0 side column), counted as if the side stream stopped hiding it.

Per-stream gain of B over A at δ = 1.0 ms:

| acceptance curve | mid-s3 c6/c7/c8 | mid-s3f c6/c7/c8 | pes-s3 c6/c7/c8 |
|---|---|---|---|
| R708 files c8d3 = R707-implied code | +8.7 / +7.4 / +6.7 % | +9.1 / +7.6 / +6.7 % | +4.5 / +3.3 / +2.2 % |
| R707-implied prose | +5.2 / +3.9 / +3.2 % | +5.6 / +4.1 / +3.3 % | +1.2 / −0.1 / −1.1 % |
| R564 mp_decode code | +1.1 / −0.1 / −0.8 % | +1.5 / +0.1 / −0.7 % | −2.8 / −3.9 / −4.9 % |

Sensitivity to δ, for mid-s3:
- δ 0.7: code c6 +9.1 / c8 +7.1, prose c6 +5.5 / c8 +3.6.
- δ 1.3: code c6 +8.3 / c8 +6.3, prose c6 +4.8 / c8 +2.9.

Production-weighted (R560 wall shares, c1-c5 unchanged), mid-s3 at δ 1.0 gives code +3.81 % and prose +2.05 %.

Against the bar (B/A ≥ 1.03 at c6 and c8, code and prose):
- **Code passes comfortably at mid.** Under pes-s3 it still passes at c6 but fails at c8 (+2.2 %).
- **Prose is borderline at c8.** mid gives +3.2 %, only just above the 1.03 bar, and δ 1.3 gives +2.9 %. The R708 model's addendum expected about +6 / +3.5 % for prose and +10 / +7 % for code; the model gives a little less because the residual at 18-21 rows is +0.59 to 0.64 ms rather than 0.55.
- **R564 mp_decode** (the short-generation agent regime) is about zero at c7-c8. The served instrument (1,024 forced tokens) does not measure that regime; this is a known blind spot, not a gate cell.

The bar is pre-registered and user-approved; this unit does not relax it.

## 3. MAP decision: `EXL3_MOE_COOP_V3_MAP=2-4:2,17-32:2`

- **Why mode 3 cannot run above 16 rows:** moefast-r3 mode 3 uses the r2 kernels, which cap at `R2_MAX_ROWS 16` and `R2_MAX_SLOTS 192`. `moe_coop_r2_launch` returns false for more than 16 rows, and the launcher then sets `v3 = 2`. So with A's settings (`EXL3_MOE_COOP_V3=3`, MAP `2-4:2`), 17-32 rows already run mode 2.
- **What the pin does:** `17-32:2` makes that fallback explicit. The MAP parser takes the first match, else `EXL3_MOE_COOP_V3`.
- **How it is checked:**
  - CPU replay (`test_rows32_cpu.py` part 5): A and B are identical over 64 (cap, rows) cells, and 17-32 rows map to mode 2 in both.
  - GPU parity by kernel name: at 17-32 rows, the V3 kernels run and no `r2_` kernel runs, under both A and B. At 16 rows under A, the r2 kernels run.
  - The c8d2 A/A cell `R32noMAP` must hash-equal R32.
- **Why not extend mode 3 instead:** that needs a kernel change, which is out of scope. The R713 nsys also found mode 3 slower than mode 2 at 16 rows.

## 4. lcguard at 17-32 rows

With QF on, densegemm calls on the QSA side branch run from their own slots in a side region, `DENSE_V2_WS_SIDE_BYTES`, which stack-r3 sets to 2 MiB.

- **Sizing:** index_qk (k 2560, n 640) uses tag 2 (tk 32, tn 128): 400 tiles, grid ≤ min(400, 170). A 32-row slot is 32 × 128 × 4 B = 16 KiB per CTA.
  - 2 MiB therefore holds 128 CTAs.
  - A full 170-CTA grid at 17-32 rows needs 2.66 MiB.
  - At 9-16 rows it needs at most 1.33 MiB, which is why stack-r3 fits today.
- **What would happen without the fix:** lcguard refuses the side call and the served kernel runs instead. The output is bitwise the same, but it costs about 7.8 µs × 12 calls ≈ 0.09 ms/step on the side branch, mostly hidden by the fork. (The nsys hint that index_qk is tuned to grid 20 at 16 rows is inferred, not proven; if it holds, nothing is refused at 24 rows either.)
- **r4 fix:** the side region is 4 MiB when `EXL3_DENSE_ROWS32` is on (+2 MiB per card, B only), and refusals are logged once per device.
- **Checks:**
  - The gate counts lcguard and refusal lines in the R32 P1 logs and the B container logs.
  - The c8d2 `R32noQF` A/A cell must hash-equal R32.
  - stack-r3's `test_dense_lcguard_cpu.py --tree <r4 tree>` passes and reports the 4 MiB side region with rows32 + fork.

## 5. Risks

- **Prose c8 below the bar** (§2). This is the most likely REJECT.
- **Headroom.**
  - B adds about 14 MiB per card: +7.8 MiB MoE/shared scratch (CPU part 2), +4 MiB densegemm slots (8 vs 4 MiB), +2 MiB side region. On top of that come transient activations at 24 rows.
  - cuda:0 had 24-130 MiB free at the 999,424 pool, so the margin is thin.
  - The gate requires min B UP-line free ≥ min A − 32 MiB per card, and reports post-ramp free as well.
  - Shrinking the pool if B fails is the user's call.
- **Spills in the M32 twins.** Two TN256 M32 mgemm twins spill (STACK 64, LDL 13-30 / STL 24-25). in_proj takes 34.3 µs and qkv 29-30 µs, so the cost is at most about 0.1-0.2 ms/step. §2's residual already includes it.
  - Follow-up, not built: `FRAG_STAGES=2` for these two twins should remove the spill with bitwise-identical output. It is device code, so it needs its own SASS and parity round.
- **Greedy divergence under concurrency (condition 2).**
  - Depth 2 at c8 changes batch composition. rows32's kernels are bitwise-equal per row to the served 16-row calls, so any extra divergence would come from scheduling.
  - greedy_conc measures it against A's own c8-vs-c1 variation: B passes if its divergences are at most the worse A half × the number of B pairs, and the same for early divergences.
  - The greedy probes run first after each boot on fresh salted prompt sets, which avoids R709c's prefix-reuse irreproducibility.
- **latchain at (6,3) (7,3) (8,3)** had never been run. The extra parity covers it; a FAIL stops the unit before any served boot.
- **R716b outcome.** The gate aborts without `DECISION UNION: PASS` (override with `S3_ALLOW_UNACCEPTED=1`). It also aborts if the A env differs from R716b's RECOMMENDED EXTRA_ENV on any key=value; in that case pass `A_EXTRA_ENV`.
- **The short-generation regime is not measured** (§2, R564).

## 6. GPU minutes

| part | minutes |
|---|---|
| 0 build (CPU, in-lock, only if the image is missing or stale; the daily keeps serving) | ~20 |
| 1 kernel parity: rows32 ~20, dense ~10, latchain ~5, hcfast ~8, CPU + P0 ~3 | ~45 |
| 2 P1 at the stack-track shapes (OFF/U × 6 rounds, c4d3 c8d1 c1d3 + d0) | ~25 |
| 3 P1 rows32 (SRV/R32 × 4, ON1, c8d2 A/A matrix, OFF2) | ~22 |
| 4 served ABBA, 4 boots (per boot: boot 3, greedy_conc 3 for A / 1 for B, fn_greedy 4, R707 instrument ~14) | ~90 |
| total | ~3 h GPU; 3.5-4 h with the build and the daily restore |

The daily is restored once at the end of the chain. This unit does not promote.
