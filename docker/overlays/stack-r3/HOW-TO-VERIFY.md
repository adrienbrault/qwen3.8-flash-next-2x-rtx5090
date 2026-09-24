# stack-r3: how to verify

Written 2026-09-24 by the stack-r3 Opus sub-agent before the gate ran; saved by the operator from its final report. The gate became R716b (driver [`scripts/r716b-stack-r3.sh`](../../../scripts/r716b-stack-r3.sh)); the build it served is INCLUDE `mf3 dg2` with empty `DGV2_NVCC_DEFS` ([`README.md`](README.md)).

## Local, CPU only (before copying)
- `cd series && shasum -a 256 -c <(awk '/^[0-9]/{print $3"  "$2}' SERIES)`: all six patches OK.
- `bash series/check-subsets.sh SRC [OVERLAYS]` (SRC = the installed `exllamav3` package of `tabbyapi:slotfix-r1`, OVERLAYS = `docker/overlays`): last line "SUBSET MATRIX OK". It rebuilds stack-r2 from the pristine package plus hcfast-r1 and moefast-r1, applies every subset, checks permuted orders, markers and guard placement, and runs the negative controls.
- `python3 tests/test_dense_lcguard_cpu.py --build --src SRC`: "LCGUARD CPU PASS" (dg1 and dg2 trees, with and without the guard).
- `python3 tests/latchain/test_latchain_cpu.py --src <series tree> --served <stack-r2 tree>`: "CPU PASS".
- The hcfast flow (torch) and densegemm (numpy) CPU tests run inside the image build.

## Install (repo-first)
1. This directory is `docker/overlays/stack-r3/`; the gate is [`scripts/r716b-stack-r3.sh`](../../../scripts/r716b-stack-r3.sh).
2. From the repo, copy the directory to `/srv/qwen5090/patches/exllamav3/stack-r3/` and the gate to `/srv/qwen5090/r716b-stack-r3.sh`, then queue it.
3. Choose the configuration (ANALYSIS §6) by editing the variable defaults at the top of the gate in the repo, then deploying it (write `.new`, then `mv`; never overwrite a running script): `INCLUDE`, `DG_ENV`, `MF3_ENV`, `DGV2_NVCC_DEFS`, `UNION_PRESET`. If R715 shows `--gc-mode freeze` removes the ~34 ms iterate stall, also set `P1_HARNESS_ARGS`.
4. Pre-flight aborts (no GPU used):
   - DAILY_IMG is not tabbyapi:stack-r2, or the live env lacks the stack-r2 keys or already sets an LC_/DENSE_ key;
   - dg1 together with dg2;
   - DGV2_NVCC_DEFS without dg2;
   - dg1 with a mode other than 3;
   - ROWS32 on;
   - GF in the union;
   - moefast keys without mf3;
   - a key set twice;
   - HC2 differs from R702's p0.json RECOMMEND.

## What the unit does and where to read it (results dir /srv/qwen5090/results/<date>-rNNN-stack-r3/)
- `build.log`: install lines (base markers, served SASS identity, STACK-BOUND, compile bar for dg2, landing, CPU PASS). `landed.txt`: the landing check. The unit aborts if the guard is required and missing.
- `parity-{hcfast,latchain,moefast,densegemm}.log`: kernel parity, each "PARITY PASS". A skipped densegemm parity is logged.
- `mp-<shape>-compare.txt`: model parity OFF / OFF2 / U. "aa=IDENTICAL arms=DIFFERENT" fails the unit. A non-identical A/A downgrades the evidence to token level. The lcguard line count in the U logs is also logged.
- `p1-summary.txt` / `p1-verdicts.json` (tools/p1_stack.py):
  - per arm and shape: stall-excluded, per-iterate-median and raw paired deltas, with dropped iterate counts;
  - hashes vs OFF in the d and d0 cells;
  - P1-VERDICT UNION, UNION-<X> (the union without X, vs OFF), A/A (OFF2 − OFF), and each X's KEEP/DROP marginal.
- `gstreams-compare.txt`: multi-stream identity at c2/c4/c8, full texts vs A1. The judgement is strict where the A/A boots are identical and BOUNDED where the daily itself varies.
- `greedy-compare.txt`: fn_greedy, including the 100k prompt.
- `served.txt` / `served.json` (tools/served_stack.py): the per-boot headroom, safety and env checks, fn_gate ratios, and SERVED-VERDICT.
  - HEADROOM: UP-line free per card vs min(A) − 32 MiB.
  - SAFETY: OOM / TORCH_CHECK / tracebacks, ramp c1..c8, leg B.
  - ENV: union keys present, no duplicates.
  - lcguard: required only if the guard engaged in-process.
  - RATIO / MEAN / AGGREGATE: fn_gate 3 pairs; no cell < 0.99, leg-A c1 ≥ 0.98, aggregate > 1.
- `audit.log`:
  - DECISION UNION: PASS, REJECT, or VOID (an A/A daily boot diverged; re-run);
  - DECISION per component: HC2 and RR go with the union; QT, QF, DG and MF3 need the union PASS plus a P1 KEEP;
  - RECOMMENDED daily: the exact IMG and EXTRA_ENV to promote.
  - The unit never promotes.
- Every boot passes `NVME_TIER=` and asserts that no EXL3_NVME_TIER key reached the container.
- If B1 diverges and the extra A/A boot (A1b) also diverges, B1 is re-booted for its fn_gate. boot-B1.log and env-B1.txt then come from the re-boot, while B1's identity records come from the first boot.

## After the run
- Run an independent review of the round (script plus raw results) before writing the result file.
- Promotion, done by the operator, not the unit:
  1. `cp /srv/qwen5090/launch-flashnext.sh /srv/qwen5090/launch-flashnext.sh.pre-stack-r3`.
  2. Set `DAILY_IMG` and the `EXTRA_ENV` default of [`scripts/launch-flashnext.sh`](../../../scripts/launch-flashnext.sh) in the repo to the gate's RECOMMENDED line.
  3. Deploy from the repo.
- Rollback: `/srv/qwen5090/launch-flashnext.sh.pre-stack-r3`, which is IMG=tabbyapi:stack-r2 with the 27 stack-r2 keys.
