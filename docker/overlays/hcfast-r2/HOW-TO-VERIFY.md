# hcfast r2: how to verify

Everything GPU-side is the operator's, run through `hcfast-gate.sh` or by hand as below. Paths assume this directory has been landed repo-first as `flan/patches/exllamav3/hcfast-r2/` and synced to `/srv/qwen5090/patches/exllamav3/hcfast-r2/`.

## 0. What already ran (CPU only, flan and local, 2026-09-24)

- **Single-TU compile** (`quick-nvcc.sh`, nvcc at `nice -n 19` in a `--rm --cpus=1` slotfix-r1 container, while r701 held the lock; the coordinator allowed single-file compiles):
  - 47 of 47 kernels, **0 spills** (`ptxas-r2.log`);
  - SASS in `flan:/srv/qwen5090/scratch/hcfast/r2/hcr2.sass`.
- **r1 identity:** the 24 r1 kernels have SASS identical to the r1 build that R698/R699 measured (`r1-sass-identity.txt`).
- **Load rounds:** `sass_rounds.py` (scoreboard-level, from cuobjdump's `&wr`/`&req` control info) → `sass-rounds-r2.txt`. ANALYSIS §3 has the table.
- **CPU flow test** (`test_hcfast_flow_cpu.py`, fake extension, r2 `hyperconnections.py` mounted over the slotfix-r1 module): `FLOW PASS` (`flow-cpu.log`).
  - 40 dispatch cases (site form × rows × level 1/2);
  - level-1 arguments identical to r1's;
  - per-R table values reach the call;
  - env parsing accepts values and tables and rejects bad input.
- **Patch:**
  - `make-patch.sh` round trip OK (4 files);
  - `check-composition.sh` → `patch-composition.txt`: **COMPOSITION OK**. hcfast-r2 applies at `--fuzz=0` alone, after moefast-r1, and before it. Both orders give the same bindings.cpp. The pristine tree's own `model_ls.py.orig` is the baseline.
- **Not run:** the full extension build (`cpu-build-check.sh` / `Dockerfile.box`). Per OPERATIONS §15 it belongs to the gate unit, inside the lock, before its first measurement.

## 1. Build (inside the gate unit's lock, before any measurement)

```bash
cd /srv/qwen5090/patches/exllamav3/hcfast-r2
sudo docker build -f Dockerfile.box --build-arg BASE=tabbyapi:slotfix-r1 --build-arg MAX_JOBS=4 -t tabbyapi:hcfast-r2 .
# stacked on moefast r1 (composition checked; the r2 patch is cumulative over slotfix-r1, so the base must NOT be hcfast-r1):
sudo docker build -f Dockerfile.box --build-arg BASE=tabbyapi:moefast-r1 --build-arg MAX_JOBS=4 -t tabbyapi:stack-r3 .
```

The build runs the following; every `python3 -c` imports torch before `exllamav3_ext`, which is the moefast `libc10.so` lesson:
- the patch, with a dry run first;
- the extension rebuild;
- the symbol asserts: `hc_mix_v3_revision == 2`, plus the moefast marker when the base has it;
- the CPU flow test;
- the landing markers: default off, level 1 = r1, level 2 = r2 with DOTS_B[16] == 2, DOTS_PF[16] == 1 and UP_Q[16] == 4.

`install-hcfast.sh` refuses a base that already has `hc_mix_v3.cu`.

Composition re-check on flan: `bash check-composition.sh <pristine exllamav3 tree> /srv/qwen5090/patches/exllamav3/moefast-r1/moefast-r1.patch`. The patch argument defaults to `../moefast-r1/moefast-r1.patch`.

## 2. Parity (bitwise, one card)

```bash
sudo docker run --rm --gpus '"device=0"' -v /srv/qwen5090/models:/models:ro -v $R:/results --entrypoint python3 \
  tabbyapi:hcfast-r2 /opt/hcfast-r2/test_hcfast_parity.py --model /models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab --json /results/parity.json
```

Every output (`dots`, `state`, `post`, `mixed`) must be storage-bit equal to the served `gr_mix_v2_int8_statein`, starting from different sentinels. The returned launch mask must match the expected one: 1/2 = V3, 4/8 = r2, 16 = PDL on a launch. Up at B=8, Q=2 reports 2 (r1's kernel).

| section | what it checks |
|---|---|
| 1 | real weights (layers 12 and 40, attn + mlp, and the final mixer); rows 1-32; half and fp32 mixed; all r1 configurations plus every r2 dots (B, J, PF) and up (B, Q) instantiation, and mixed selections |
| 2 | `GatedResidual._mix` at level 1, at level 2 defaults, and at level 2 with per-R tables, vs off |
| 3 | 200 fuzz seeds |
| 4 | CUDA graph: a producer kernel writes the streams, then mixer A, then mixer B **sharing A's `dots` workspace** (the write-after-read case), 20 replays, for r1, r2 and r2 J=8/Q=4 |
| 4b | r1's auto tile captured cold in a fresh process |
| 5 | misaligned `dots` workspace |
| 6 | **every PDL configuration through sections 1-5 in a fresh process** |

Pass is `PARITY PASS` with exit 0. Section 6 can end three ways:
- a PDL output mismatch fails parity;
- a launch or capture error means the platform does not take the attribute. It is recorded as `"pdl": "unsupported: …"` in parity.json, does not fail the non-PDL verdict, and the gate then runs P0 with `--no-pdl`;
- `"pdl": "ok"` means PDL is available to P0.

## 3. P0 (explains and chooses; does not gate, §16)

```bash
sudo docker run --rm --gpus '"device=0"' -v /srv/qwen5090/models:/models:ro -v $R:/results --entrypoint python3 \
  tabbyapi:hcfast-r2 /opt/hcfast-r2/bench_hcfast_p0.py --model /models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab \
  --rows 1 4 8 16 --reps 504 [--no-pdl] --json /results/p0.json
```

Setup: 24 cold sites (about 160 MB of int8 weights, above the 96 MB L2), graph-replay timing, µs per chain (`hc_apply` → dots → up). Per R there are three stages:
- **A.** A dots sweep over every instantiated (B ≤ served tile, J, PF), with up at the r2 default.
- **B.** An up sweep (B, Q) at A's best.
- **C.** 5 interleaved rounds of [reference, r1-d2, pick, pick+pdl]. The medians are the reported numbers.

Pre-registered:
- **Control:** reference#2 within ±3 % of reference at every R. Otherwise the gate stops (`P0-CONTROL-FAIL`).
- **RECOMMEND:** `EXL3_HC_MIX_V3=2` plus per-R tables built from the picks. R ≤ r_k uses the pick measured at r_k, and the largest measured R covers up to 32.
- **PDL** is one switch. It is on if pick+pdl ≤ 0.99 × pick at R=16 and nowhere above 1.01 ×.
- **Reading it:** the `R=16: … r2 / r1-d2` median is the expected sign of HC2 − HC1 in P1.
  - At R=16, |Δ| < 1 µs/chain means FLAT is the expected P1 result at c8d1 (ANALYSIS §6: 101 chains per step against the 0.1 ms floor).
  - R=4 differences below about 1.5 µs/chain will not resolve at c1d3.
- **Optional ncu** of the pick vs r1-d2: `--ncu --ncu-variants reference r1-d2 <pick as m,bd,bu,j,pf,q>` under `ncu --nvtx`. The cells to compare are `lts__t_sectors_srcunit_tex_op_read.sum` (L2 bytes, ANALYSIS §2) and the long-scoreboard share on the dots FFMAs.

## 4. P1 (the gate's decision), in-process harness, untraced, 4k

Five rounds of (OFF, HC1, HC2), with the order rotated per round as in r701, at c4d3, c8d1 and c1d3, read per shape from `p1-<tag>/ctx4096_b<B>_d<D>/`:
- **HC1** = `EXL3_HC_MIX_V3=1 EXL3_HC_MIX_V3_DOTS_B=2 EXL3_HC_MIX_V3_UP_B=8`, the R699 candidate. It gets a clean c4d3 re-measure as a side effect.
- **HC2** = the RECOMMEND line.

Per round, both arms' `sequence-hashes.json` must equal OFF's.

**Stack baseline.** `STACK_FLAGS` (default empty) is prepended to every arm, OFF included, and to both greedy boots, so P1 measures HC on top of what STACK.md has already accepted. The operator fills it from STACK.md at queue time. If R701 accepts moefast m2, run with `STACK_FLAGS="EXL3_MOE_COOP_V3=2" HC_BASE=tabbyapi:moefast-r1`. The gate then builds `tabbyapi:hcfast-r2-moe` from `Dockerfile.box` on the moefast-r1 image. It aborts if `STACK_FLAGS` enables moefast on a base without it.

Decision, per flag set F (§16, pre-registered in `hcfast-gate.sh`):
- **Identity:** hashes identical in every round and shape. For HC2, greedy must also show 0 divergences, on and off.
- **Reject:** all 5 F−OFF deltas > 0 at c4d3, or at c8d1.
- **Accept:** no rejection, and at least one shape with all 5 F−OFF deltas < 0.
- **HC2 replaces HC1 in STACK.md** if HC2 is accepted and HC2−HC1 is not a same-sign regression at any shape.
- Mixed signs mean FLAT.

The predicted HC2−HC1 at c4d3/c8d1 is −0.15 to −0.65 ms (an estimate, ANALYSIS §6). The lower half is at or below the harness floor.

## 5. Greedy identity on the served launcher

`fn_greedy.py`: ref (the daily, OFF) vs a boot with `IMG=tabbyapi:hcfast-r2 EXTRA_ENV_ADD="<HC2 flags>"` vs a boot of the image with the flag off. Divergences must be 0 for both.

The per-R tables (`8:2,32:4`) pass through the launcher intact. `launch-flashnext.sh` appends `EXTRA_ENV_ADD` and splits it with `for kv in $EXTRA_ENV` (IFS whitespace only; the values contain no glob characters), checked 2026-09-24.

At each boot the gate logs free VRAM, the image, `EXL3_HC_MIX_V3=2`, the `EXL3_HC_MIX_V3_DOTS_B=…` value in the container env (so a dropped table shows in audit.log), and OOM lines.
