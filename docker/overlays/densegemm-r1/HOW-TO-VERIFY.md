# HOW-TO-VERIFY (densegemm r1)

Written by the dense-GEMM Opus sub-agent, 2026-09-24; saved by the operator from its final report. Bars: OPERATIONS §16 (identity + no same-sign regression + the gain clause, ≥ 5 in-process pairs per shape). P0 is diagnostic. All GPU steps are the operator's. `densegemm-gate.sh` runs §0-§4 in one queue unit and restores the daily at the end. Queued as `flan/r710-densegemm-gate.sh` (unit name = file name).

## Install on flan
1. Copy this directory, including `d0/`, to `/srv/qwen5090/patches/exllamav3/densegemm-r1/`.
2. Copy `densegemm-gate.sh` to `/srv/qwen5090/rNNN-densegemm-gate.sh` and queue `rNNN-densegemm-gate`.
3. Results go to `/srv/qwen5090/results/<date>-rNNN-densegemm-gate/`.

## §0 Build (in the lock, before any measurement)
```
docker build -f Dockerfile.box --build-arg BASE=tabbyapi:stack-r2 --build-arg MAX_JOBS=4 \
  --build-arg PATCH_SHA=$(sha256sum densegemm-r1.patch | cut -c1-64) -t tabbyapi:densegemm-r1 .
```
- The gate rebuilds when the labels `local.densegemm.patch_sha256` / `local.densegemm.base` don't match.
- The build must print "densegemm: served SASS identical (whitespace collapsed); 32 twins added", "CPU TESTS PASS", "densegemm r1 landed".
- Check the twin register-usage lines; LOCAL > 0 means spills (report it).
- Compile-only pre-check without Docker: `quick-nvcc.sh`.
- On a compile error: fix it in the agent workspace `work/b`, re-run `make-patch.sh` (base check, fuzz-0 round trip, composition with pdl-l2-r1), then rebuild.

## §1 Parity (one card, daily env, tune cache mounted)
```
python3 /opt/densegemm-r1/test_densegemm_parity.py --model /models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab --json /results/parity.json
```
Pass: every cell equal (OFF vs ON vs ON again) over every dense K4 projection at rows 1, 2, 3, 4, 8, 12, 16, 17, 24, 32, fp16 and fp32 outputs; V2 engaged at rows 4 and 16 for every served family (gdn.out_proj, attn.o_proj, index_qk, shared.down, sliced in_proj and qkv, shared gate/up); stress (200 interleaved launches) and graph (20 replays vs eager served) 0 mismatches; last line "PARITY PASS". Rows 1-2 are not engaged by design (int8 GEMV).

## §2 P0 (diagnostic)
```
python3 /opt/densegemm-r1/bench_densegemm_p0.py --model /models/<ckpt> --json /results/p0.json [--rounds 5] [--step-ms-c1 <OFF c1d3 ms>]
```
µs per call for m0 (served), m1 (all), m2 (gemm + mgemm), m3 (gemv), rotated, median of per-round medians; RECOMMEND block. Expect out_proj r16 m1 − m0 ≈ −3 to −5 µs; in_proj r4 ≈ −5 to −7 µs; gemv r4 −1.5 to −4.5 µs. m1 slower than m2 at rows 4 means the gemv ring is slower; prefer `EXL3_DENSE_V2=2`.

## §3 P1 (in-process harness, 4k, untraced)
ROUNDS = 6, rotated so each of OFF / ON / GM goes first twice; shapes c4d3, c8d1, c1d3; draft-0 cells `ctx4096_b{1,4,8}_d0` from the same runs. ON and GM hashes must equal the round's OFF in both the drafting and draft-0 cells. `p1-summary.txt` prints per-pair deltas, sign, %, and `P1 VERDICT ON|GM`.

## §4 Served greedy
Tags `ref` (daily), `off` (image, flags unset), `on` (`EXL3_DENSE_V2=1 EXL3_DENSE_ROWS32=1`). Pass requires `GREEDY-SUMMARY reference=ref divergences=0` with both arms present; anything else rejects both arms.

## DECISION (pre-registered, per arm)
| clause | rule |
|---|---|
| identity | parity PASS; every hash identical; greedy 0 divergences |
| pairs | ≥ 5 pairs at each shape |
| regression | no same-sign regression at c4d3 or c8d1 |
| c1 bound | a same-sign c1d3 regression passes only if ≤ 2 % and smaller in ms than a same-sign c4/c8 gain; record it in STACK.md |
| gain | at least one shape is a same-sign GAIN, else FLAT (not a stack entry) |

ACCEPT means a STACK.md row and batching into the next stacked image; this unit does not promote. Before promotion, the headroom boot must show free ≥ REF − 32 MiB (the workspace is 4-8 MiB per card).
