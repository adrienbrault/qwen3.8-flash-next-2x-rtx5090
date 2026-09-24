# moefast r3: how to verify

Every GPU step below is for the operator. No GPU was used to write this round. Paths on flan: `MF=/srv/qwen5090/patches/exllamav3/moefast-r3`.

## 0. Local (Mac, no GPU): already run

- `bash make-patch.sh`: rebuilds stack-r2 (src + hcfast-r1 + moefast-r1), checks it against `work/a`, diffs `work/a` against `work/c`, and round-trips `moefast-r3.patch` at fuzz 0. Expected output: `round trip OK over stack-r2 (12 files)`.
- `python3 test_moefast_r3_cpu.py`: part 1 covers the timeline, kernel_role and the pick rule. Part 2 (the early-fork guard in `BlockSparseMLP.forward`) needs an installed exllamav3, so it runs in the image (`MOEFAST_R3_REQUIRE_PART2=1`).
- `python3 test_moefast_trace_cpu.py` (r2's trace test, unchanged).

## 1. Install on flan

```
rsync -a docker/overlays/moefast-r3/ flan:/srv/qwen5090/patches/exllamav3/moefast-r3/   # operator
```

Optional CPU-only build check, allowed while the GPU queue is busy (OPERATIONS §15):

```
docker run --rm --entrypoint bash -v $MF:/opt/moefast-r3:ro -v /tmp/mf3:/out tabbyapi:stack-r2 /opt/moefast-r3/cpu-build-check.sh
```

The check must end with `CPU BUILD CHECK OK`. During the install, `sass_identity.py` requires:

- 0 changed base functions;
- exactly 36 new r2 A/B kernels and 1 head kernel, and nothing else.

The SASS goes through `cuobjdump -sass` into a FILE. The landing check imports `torch` before `exllamav3_ext`.

## 2. The gate

Install `moefast-r3-gate.sh` as `/srv/qwen5090/rNNN-moefast-r3.sh` (the operator picks the R number) and queue it. It builds `tabbyapi:stack-moefast-r3` in-lock if it is missing or stale. The image labels carry the patch sha256 and the base.

Knobs:

| var | default | meaning |
|---|---|---|
| `ROUNDS` | 6 | P1 rounds per shape. Must be a multiple of 3 and at least 6, so each arm goes first equally often. |
| `C2D3` | 0 | 1 adds c2d3 (8 rows) to P1. Reported only, not gated. |
| `R3_ENV` | (P0 pick) | Operator override for the R3 arm. It is logged. |

Steps, and what stops the unit:

1. **Parity**, on one card:
   - r2's suite at rows 1-16, including 14 = c7d1 and 15 = c5d2;
   - the side-stream layer calls. At rows 2/4/8/12/14/15/16 on the real K2/K3 tensors, the serial, late, early, early+prio and head arms of modes 2/3 must match serial V2 bit for bit (routed output and shared output);
   - a join stress of 200 early high-priority calls;
   - dispatch: the head kernel appears only in mode 3 with HEAD set.

   Stops on anything but `PARITY PASS`.
2. **P0 with the served side stream and router** (`--extra`). Arms: `ref m2 m3 m3H m3H2 m3P m3E m3EP m2E m2EP ser m3N`.
   - Instrument acceptance: `INSTRUMENT OK` needs gap(m3) − gap(m2) ≥ 5 µs at r4/D28. R703b nsys c1d3 showed 1.9 → 14.1.
   - The pick prints `[p0] R3_ENV=...`.
   - If the instrument is NOT REPRODUCED or the pick fails, R3 = `EXL3_MOE_COOP_V3=3 EXL3_SHARED_EXPERT_EARLY=1` (pre-registered default), and the audit says to read P0 first.
   - Stops on rc ≠ 0.

   2a. **L2 probe** (legacy calls, arms ref/m2/m3; the R703 TypeError is fixed). Stops on rc ≠ 0.
3. **nsys** at c1d3 and c4d3, arms OFF/M2/R3, plus M2E at c1d3 (mode 2 with R3's overlap flags). These run through `moe_timeline.py --gap-target 4`, which prints `GAP <arm> <median µs> OK|HIGH`. The r3 target is ~2 µs, M2's value. Stops on any nsys or timeline error. GAP HIGH is reported, not gated.
4. **P1**: c4d3, c8d1 and c1d3 (+ c2d3). Each round:
   - opens with one discarded warm-up run;
   - rotates the arm order OFF/M2/R3 so each arm goes first ROUNDS/3 times;
   - checks the sequence hashes of M2 and R3 against the round's OFF in both the d cell and the `ctx4096_b{1,4,8}_d0` cell.

   A crashed run is logged as `DIAGNOSTIC ERROR` and counted as "missing", separately from hashes that DIFFER. `p1-summary.txt` prints, per shape and cell, R3−M2, M2−OFF and R3−OFF. For each: every pair, the median and its % of the reference median, the mean, and the sign count.
5. **Greedy** (`fn_greedy.py`) on the served launcher. It runs three boots:
   - ref = the daily;
   - on = the image with the full `EXTRA_ENV` override: the served env minus the arm keys, plus R3. There are no duplicate keys, so nothing relies on docker's last-wins;
   - off = the image with the served env.

   The container env is asserted through `docker exec` (R3 keys present, no duplicate keys, and off = mode 2 with no early fork). Pass needs `GREEDY on vs ref: N identical, IDENTICAL` and the same for off.

## 3. DECISION (pre-registered; R3 − M2 paired by round; medians read)

- **Identity**:
  - `PARITY PASS`;
  - P1 hashes identical in every round, shape and cell (d and d0);
  - greedy on and off with 0 divergences.
- **Pairs**: at least 5 valid R3/M2 pairs at c1d3, c4d3 and c8d1.
- **No regression**: c4d3 or c8d1 all-positive = REJECT.
- **c1 bound**: an all-positive c1d3 is tolerated only if both hold:
  - it is ≤ 2 % of M2's c1d3 value by both the in-process mean (the §16 wording) and the median;
  - c4d3 and/or c8d1 is an all-negative GAIN whose summed ms, by mean and by median, exceeds the c1d3 loss.
- **Gain clause**: at least one of c1d3/c4d3/c8d1 is all-negative (6/6). Otherwise the result is `NOT AN ENTRY (flat)`.
- The d0 cells are reported and hash-checked, not gated.

Final line: `DECISION: ACCEPT into the stack ledger | ...`, `DECISION: NOT AN ENTRY (flat) | ...` or `DECISION: REJECT | ...`. The unit never promotes.

## 4. What the parity test cannot see

`EXL3_SHARED_EXPERT_EARLY=1` is a Python-side reorder: `BlockSparseMLP.forward` calls `bc.start_shared(y)` before routing. The parity test reproduces it at the kernel level (fork before `routing_std`, the same event pattern), but not through the module. The module path is covered by:

- CPU test part 2, which checks the call order start → route → run on the same tensor and 11 guard-negative cases;
- the C++ TORCH_CHECKs in `run_bszN`. A started fork must match the call's input pointer and token count, or the call throws, never computes wrong;
- P1 sequence hashes and fn_greedy on the real model.

## 5. Reading the result

- `INSTRUMENT OK` plus `GAP R3 ... OK` in the nsys timelines means the mechanism is fixed.
- The P1 R3−M2 sign is then the answer. Estimates from ANALYSIS.md §3:
  - c1d3: −0.18 to −0.38 ms per step;
  - c4d3 and c8d1: roughly flat (−0.07 ± 0.14 ms).
- If P0 reproduces the gap but R3's nsys gap stays HIGH, the early fork does not beat the co-residency limit. Read the m3H/m3H2 arms: a head kernel that restores the gap confirms the placement mechanism, and a round 4 would move to a kernel-side fix.
