# rows32 r4: how to verify

Written by the rows32 Opus sub-agent before the gate ran; saved by the operator from its final report. The gate became R717 (driver [`scripts/r717-rows32-r4.sh`](../../../scripts/r717-rows32-r4.sh)); its follow-ups are [`scripts/r717b-rows32-followup.sh`](../../../scripts/r717b-rows32-followup.sh) and [`scripts/r717c-rows32-context.sh`](../../../scripts/r717c-rows32-context.sh), and the results are in [`bench/results/r717-rows32.md`](../../../bench/results/r717-rows32.md). `rows32_decide.py` and `greedy_conc.py` here carry the two fixes made after the R717 review (the CPU-test success line, and the early-divergence count reported instead of judged).

## A. Locally (CPU only, no GPU, no docker)

Run everything from this directory with `SRC` set to the installed `exllamav3` package of `tabbyapi:slotfix-r1`, copied out of the image. The work trees also need `docker/overlays/{hcfast-r1,moefast-r1,stack-r3}` (the default `FLAN`).

1. `bash make-patch.sh` rebuilds `work/a` (stack-r3 = src + hcfast-r1 + moefast-r1 + `apply-series.sh` INCLUDE="mf3 dg2") and regenerates `rows32-r4.patch` from `work/b`. Expect `round trip OK (9 files, sha256 8d274c73a36d2237)`. `--init` reseeds `work/b` from rows32-r3 and needs `R3=<rows32-r3.patch>`; rows32 r1 to r3 are not in this repository.
2. `bash verify-patch.sh` should end with `VERIFY OK`:
   - "mf3 dg2" OK and "mf3 dg1" OK;
   - "dg2", "dg1" and bare stack-r2 REJECT (no moefast-r3 context for the `start_shared` hunk);
   - `test_rows32_cpu.py` and stack-r3's `test_dense_lcguard_cpu.py` pass on the patched target;
   - with `R3` set, the list of r3 lines missing from r4 shows only the revision and message bumps.
3. `python3 test_rows32_cpu.py --tree work/b/exllamav3` should print `CPU TESTS PASS`. Part 4 (flags) is skipped without torch; the gate reruns it with `--require-flags`.
4. `python3 rows32_decide.py --selftest` should print `SELFTEST PASS`.
5. `bash -n *.sh` and `python3 -m py_compile *.py`.

The economics model of [`ANALYSIS.md`](ANALYSIS.md) §2 (`econ_model_r4.py` and the R708 model it imports) is not in this repository.

## B. Install on flan (operator)

1. Copy this directory to `/srv/qwen5090/patches/exllamav3/rows32-r4/`. stack-r3's directory must already be at `/srv/qwen5090/patches/exllamav3/stack-r3`, because the gate uses its tools and tests.
2. Install the gate, [`scripts/r717-rows32-r4.sh`](../../../scripts/r717-rows32-r4.sh), as `/srv/qwen5090/r717-rows32-r4.sh` by copying to `.new` and then `mv`; never overwrite a running script. The unit name is the file name (`R32_UNIT`). `RuntimeMaxSec` must be ≥ 43200.
3. Queue only after R716b's audit shows `DECISION UNION: PASS`.
   - If R716b's `RECOMMENDED daily: ... EXTRA_ENV="..."` line differs from the default `S3_UNION` result, set `A_EXTRA_ENV="<that line>"`. The gate compares full key=value sets and aborts on any difference.
   - If stack-r3 was promoted with a different union, set `S3_UNION` instead.
4. The image `tabbyapi:stack-r3-rows32` is built in-lock by the gate when it is missing or its labels (base ID, patch sha, include, defines) are stale. The daily keeps serving during the build. To build by hand: `docker build -f Dockerfile.box --build-arg BASE_ID=$(docker image inspect -f '{{.Id}}' tabbyapi:stack-r3) --build-arg STACK_INCLUDE="mf3 dg2" --build-arg DGV2_NVCC_DEFS="<stack-r3's>" -t tabbyapi:stack-r3-rows32 .`
5. Useful knobs (all optional): `A_EXTRA_ENV`, `S3_UNION`, `S3_AUDIT` (defaults to the latest `*-r716b-stack-r3/audit.log`), `S3_ALLOW_UNACCEPTED=1` (not recommended), `MAP_ADD` (default `17-32:2`), `POLICY_B` (default `[[4, 3], [8, 2]]`), `BASE_R32`, `IMG_R32`, `ROUNDS`, `R32_ROUNDS`.

## C. Reading the results (`/srv/qwen5090/results/<date>-r717-rows32-r4/`)

- `audit.log` is the step log. It ends with `DECISION: ACCEPT|REJECT|VOID ...` from `rows32_decide.py` and, on ACCEPT, a `RECOMMENDED (...) IMG=... DRAFT_POLICY=... EXTRA_ENV="..."` line.
- Part 0: the SASS identity report (0 changed, 0 added), the landing log and the CPU test logs.
- Part 1:
  - `parity-*.log` must each show PASS.
  - The dense engagement at 17/18/21/24/32 must be non-vacuous.
  - `p0.log` is diagnostic only.
- Parts 2-3:
  - `p1-verdicts.json` and `p1_stack.py` output: U == OFF hashes, no same-sign regression at c4d3/c8d1, c1 within bound.
  - R32 identical across rounds, ON1 == SRV.
  - The c8d2 A/A cells R32noQF, R32noDR and R32noMAP must each equal R32.
  - lcguard refusal counts are reported.
- Part 4:
  - `served-*.json` hold the per-boot, per-cell medians; the B/A table is in the decision output.
  - Bar: c6 and c8 ≥ 1.03 for code and prose, both pairs > 1; c2-c5 and c7 ≥ 0.985; c1 ≥ 0.98.
  - `greedy_conc` should report `GREEDYC-VERDICT PASS`.
  - fn_greedy B1 must equal ref; the result is VOID if A2 diverges.
  - Headroom: min B UP-line free ≥ min A − 32 MiB per card.
  - Every boot's env and policy must be as intended, with `NVME_TIER=` empty, and there must be 0 OOM, TORCH_CHECK or traceback lines.

## D. After the run

1. Run an independent review of the round with an Opus sub-agent before writing the result file.
2. This unit does not promote. On ACCEPT, the operator sets `DAILY_IMG`, `DRAFT_POLICY` and the `EXTRA_ENV` default of [`scripts/launch-flashnext.sh`](../../../scripts/launch-flashnext.sh) to the RECOMMENDED line and deploys from the repo (promotion conditions agreed 2026-09-24: bitwise 4b/4c parity, greedy_conc no worse than the served configuration's own variation, served B/A ≥ 1.03 at c6/c8). R717 ended REJECT and the promotion went through R717b and R717c instead ([`bench/results/r717-rows32.md`](../../../bench/results/r717-rows32.md)).
3. Follow-up, not built: `FRAG_STAGES=2` for the two spilling TN256 M32 mgemm twins (bitwise-identical, device code; needs its own SASS and parity round).
