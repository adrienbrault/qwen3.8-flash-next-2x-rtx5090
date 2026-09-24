# HOW-TO-VERIFY: hcfast r1 (EXL3_HC_MIX_V3)

> **Bars superseded 2026-09-24 by OPERATIONS §16 (stack track).** The effect-size bars below (P0 %, P1 ms) are diagnostic only. A bitwise-identical flag is accepted into the stack on identity (P1 hashes + fn_greedy) plus no same-sign regression at c4d3/c8d1 over ≥ 5 in-process pairs, and is promoted in a batched stack gate (`flan/r701-stack-gate.sh` pattern). Ledger: `flan/STACK.md`.

`EXL3_HC_MIX_V3=1` makes the served GatedResidual decode mixer (`gr_mix_v2_int8_statein`: `gr_v2_dots_i8` + `gr_v2_up_i8_state`) run the V3 kernels in `exllamav3_ext/hc_mix_v3.cu`. It applies at every site with 1 ≤ R ≤ 32 on the int8 V2 branch, the final mixer and MTP sites included.

- **Structure:** two launches as before, the same grid shapes and thread counts, and CTA count equal to the served one at the default caps.
- **Numerics:** every output is bitwise-identical by construction (ANALYSIS.md §4).
- **Flag unset:** the image runs the served launches from an unchanged `hc_mix.cu`.
- **Two optional knobs** (only with the flag): `EXL3_HC_MIX_V3_DOTS_B` and `EXL3_HC_MIX_V3_UP_B` cap the rows per CTA of each launch. Values are 8 (default, served tiling), 4, 2 or 1, or 0 = one-wave auto. They change only which rows a CTA owns, and they can only raise the CTA count.

Why V3 should be faster, with ncu numbers: ANALYSIS.md.

## Deliverables (build context = this directory)

| file | what |
|---|---|
| `hcfast-r1.patch` | `diff -ruN` of `src/exllamav3` → `work/exllamav3`, paths `a/…` `b/…`, applied with `patch -p1 --fuzz=0` inside the package dir. `./make-patch.sh` regenerates it and checks the round trip |
| `Dockerfile.box` | `ARG BASE=tabbyapi:slotfix-r1`, `FROM ${BASE}`. Applies the patch, rebuilds `exllamav3_ext` for sm_120, runs the CPU flow test, and asserts the landing marker plus the default-off state |
| `install-hcfast.sh`, `rebuild-native.py` | patch (dry run first) + full native rebuild into site-packages (hcfuse/mixstate's builder, unchanged) |
| `test_hcfast_parity.py` | GPU parity test: bitwise, 5 sections |
| `test_hcfast_flow_cpu.py` | CPU test of the Python dispatch, with a recording fake ext. It runs inside `docker build` |
| `bench_hcfast_p0.py` | P0 microbench, forked from `bench_hcfuse_p0.py`: reference / copy / v3 / attribution / tile variants, µs per chain |
| `hcfast-gate.sh` | the whole sequence below as one queue-chained unit (draft: set `TAG`/`R`, copy this dir to `HCF`) |
| `test_hcfast_math_cpu.py` | host numpy check of the two bitwise claims (reduce-scatter tree, int8 conversion) |
| `cpu-build-check.sh`, `quick-nvcc.sh`, `ptxas-v3.log`, `sass-static-counts.txt`, `sass-load-order.txt`, `cpu-build.log` | the CPU-only checks that ran on flan (section 0) and their outputs |

Files the patch touches: `exllamav3_ext/hc_mix_v3.cu` and `hc_mix_v3.cuh` (new), `exllamav3_ext/bindings.cpp` (+1 include, +1 `m.def`, +1 `m.attr("hc_mix_v3_revision") = 1`), and `modules/hyperconnections.py`. The Python change is the landing marker `_HC_MIX_V3_BUILD = "r1"`, three class attributes read at import, and an `if self.MIX_V3 and self.STATE_IN_UP` in front of the served int8 selection in `_mix`. With the flag off, that test is False and the served call is unchanged.

## 0. Already verified on CPU (flan, `docker run --rm` without `--gpus`, 2026-09-24)

- `./make-patch.sh` checks the round trip: pristine + patch == `work/exllamav3` (4 files).
- `quick-nvcc.sh`: a single-TU `nvcc` of `hc_mix_v3.cu` with the extension's flags (`-O3 --use_fast_math -lineinfo`, sm_120), with `ptxas -v`. Log: `/srv/qwen5090/scratch/hcfast/ptxas-v3.log`, SASS in `hcv3.sass`.
  - **0 bytes spilled in all 24 instantiations.**
  - V3 dots B=1/2/4/8: 60/64/128/128 registers. V3 up: 60-64.
  - The copied served kernels land on the served counts (dots 44/56/66/94, up 40/40/47/64). They also have the same SASS length as the ncu listing (dots<8> 1,432, up<8,half> 1,240 instructions).
  - Static SASS (`sass-static-counts.txt`): V3 dots<8> has `I2F` 0 (served 256) and `PRMT` 32 (one rolled column iteration; dots<4> has 96 over 3 unrolled iterations). V3 up<8> has `SHFL` 35 (served 160) and `LDS` 25 (served 39).
  - Load order (`sass-load-order.txt`, used by ANALYSIS.md §7): V3 dots<4> issues 21 `LDG` back-to-back before its first FMA block, where the served dots<4> issues the weights and then 2 stream loads per row, each consumed at once (≈ 1 + B exposed rounds per iteration). V3 dots<8> is only partly batched: 9-14 loads come before the first FMA and 7 are interleaved. V3 up issues the 10 up_q words at entry, before the prelude.
- `cpu-build-check.sh` in `tabbyapi:slotfix-r1` covers:
  - patch dry run + apply (fuzz 0);
  - the full `rebuild-native.py` build (MAX_JOBS=4);
  - the symbol and revision checks, the landing marker, and default-off / flag-on selection;
  - `test_hcfast_flow_cpu.py` (**FLOW PASS**);
  - `py_compile` of the GPU scripts, and a `cuobjdump` count of the V3 and copied-served kernels in the rebuilt `.so`.

  **Result (06:52-06:59 UTC, after R695 ended):** all 166 build steps passed, with no diagnostics from `hc_mix_v3.cu`; `hcfast r1 landed (default off)`; `EXL3_HC_MIX_V3=1 selects gr_mix_v2_int8_v3`; `FLOW PASS` (20 dispatch cases, env parsing); 12 V3 and 12 copied-served kernel instantiations in the rebuilt `.so`; `CPU BUILD CHECK OK`, rc=0. Log: `cpu-build.log` here, a copy of `/srv/qwen5090/scratch/hcfast/cpu-build.log`.
- `test_hcfast_math_cpu.py` (numpy, host) checks the two arithmetic claims. Reduce-scatter vs butterfly: 12,000 random cases, B = 1/2/4/8, 0 bit mismatches. PRMT/FADD conversion vs `(float) q`: exact for all 256 int8. **MATH PASS.**

Nothing above ran on a GPU. Every step below is the operator's. `hcfast-gate.sh` runs steps 1-5 in order, with the stop rules.

## 1. Build (CPU, ~5-10 min at MAX_JOBS=4)

```sh
HCF=/srv/qwen5090/patches/exllamav3/hcfast-r1   # copy of this directory
cd $HCF && sudo docker build -f Dockerfile.box --build-arg BASE=tabbyapi:slotfix-r1 --build-arg MAX_JOBS=4 \
  -t tabbyapi:hcfast-r1 . > build.log 2>&1 || { tail -30 build.log; exit 1; }
grep -E "hcfast r1 landed|selects gr_mix_v2_int8_v3|FLOW PASS" build.log     # all three lines must be present
```

Build inside the GPU lock, or when no unit is measuring. A 4-job native build loads the host, and several units measure host launch gaps.

## 2. Parity (GPU, one card, ~2-4 min; the daily down or on the other card)

```sh
R=/srv/qwen5090/results/2026-09-2X-hcfast-r1; mkdir -p $R
sudo docker run --rm --gpus '"device=0"' -v /srv/qwen5090/models:/models:ro -v $R:/results \
  --entrypoint python3 tabbyapi:hcfast-r1 /opt/hcfast-r1/test_hcfast_parity.py \
  --model /models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab --json /results/parity.json 2>&1 | tee $R/parity.log
```

The reference is `ext.gr_mix_v2_int8_statein`, the served launches, with the served flags forced by the script. Every output (`dots`, `state`, `post`, `mixed`) must be storage-bit equal. The two arms start from different sentinel bit patterns.

1. **Real weights.** HC weights of layers 12 and 40 (attention and MLP sites) and the final mixer. Rows 1-9, 12, 13, 15, 16, 17, 24, 32. `mixed` both half and fp32. 13 configurations: mode 3/1/2/0 at caps 8/8, dots caps 4/2/1, up caps 4/2/1, (0,1,1), auto (3,0,0) and (0,0,0). The returned V3 mask must equal the requested mode.
2. **Python integration.** `GatedResidual._mix` with `MIX_V3` on vs off, all rows.
3. **Fuzz.** 200 seeds: rows 1..32, site, stream magnitude 1e-2..3e2, configuration, mixed dtype.
4. **Graph capture.** Capture V3 at R = 1, 4, 8, 16, then 20 replays on fresh inputs vs the eager reference. 4b runs in a fresh process: the auto tile (caps 0/0) is captured at R = 4 and 16 with its occupancy query first run inside the capture, as a served graph would, followed by 5 replays each. The gate may pick an auto tile from P0 and serve it, so 4b must pass.
5. **Fallback.** A 16-byte-misaligned `dots` workspace must return mask 1 (V3 up declined) and still be exact.

**Pass:** `PARITY PASS`, exit 0. **Any FAIL stops the round**; report the first failing line. The design has no approximate step (ANALYSIS.md §4, "bitwise because"), so any mismatch is a bug, not a tolerance question.

## 3. P0 microbench (GPU, one card, ~6 min)

```sh
sudo docker run --rm --gpus '"device=0"' -v /srv/qwen5090/models:/models:ro -v $R:/results \
  --entrypoint python3 tabbyapi:hcfast-r1 /opt/hcfast-r1/bench_hcfast_p0.py \
  --model /models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab --rows 1 4 8 16 --reps 504 --json /results/p0.json \
  2>&1 | tee $R/p0.log
# diagnostic, L2-hot weights (not gated):
...  bench_hcfast_p0.py --sites 1 --variants reference v3 v3-auto reference#2 ... --json /results/p0-hot.json
```

The chain per variant is `ext.hc_apply` + the mixer call. It is timed as a CUDA graph over 24 real sites (~160 MB int8, above the 96 MB L2, so weights are DRAM-cold as in decode). "graph us/chain" is the bar. Variants:

- `reference`, then `reference#2` at the end (drift check);
- `copy` (mode 0: the served kernels copied into the new TU);
- `v3` (the candidate);
- `v3-dots` and `v3-up` (attribution);
- `v3-d4/d2/d1`, `v3-u4/u2`, `v3-auto/dauto/uauto` (tiles).

The bench prints its verdict lines itself.

**Pre-registered bars (graph µs, 24 cold sites):**

- **Control:** `copy` within ±3 % of `reference` at every R, and `reference#2` within ±3 % of `reference`. If not, the bench is not isolating the kernels: stop and report.
- **PASS:** `v3 ≤ 0.75 × reference` at R=4 **and** at R=16 (≥ 25 % faster at both).
- **KILL:** `v3 > 0.92 × reference` at R=4 **and** at R=16 (< 8 % at both). Stop.
- **MARGINAL:** anything between the two. Run P1; it decides.
- **Tile:** a cap variant replaces the default caps for P1 and serving only if it is ≥ 5 % below `v3` at R=4 and R=16 and not > 3 % above `v3` at R=1 and R=8. If several qualify, take the lowest R=4 + R=16 sum (`hcfast-gate.sh` does this from `p0.json`).
- **Predicted** (estimates, ANALYSIS.md §5):
  - `v3/reference` 0.70-0.85 at R=4 and 0.52-0.65 at R=16;
  - `v3-dots` carries most of the R=16 gain (XU floor ~7 µs);
  - at R=1, 0.85-0.95.
  - A result far off these numbers says the model in ANALYSIS.md §3 is wrong somewhere. Read the ncu pass before tuning further.

**Optional ncu pass** (~2 min; confirms the mechanism: XU %, LSU %, SHFL/LDS counts):

```sh
sudo docker run --rm --gpus '"device=0"' --cap-add SYS_ADMIN -v /srv/qwen5090/models:/models:ro -v $R:/results \
  --entrypoint /usr/local/cuda/bin/ncu tabbyapi:hcfast-r1 --set full --nvtx \
  -k 'regex:hc_apply_kernel|gr_v2_dots_i8|gr_v2_up_i8_state|dots_i8_v3|up_i8_state_v3' \
  -o /results/p0-ncu python3 /opt/hcfast-r1/bench_hcfast_p0.py --ncu --rows 4 16 --sites 3 --variants reference v3
```

Read `sm__inst_executed_pipe_xu` (V3 dots should be ≈ 0), `sm__inst_executed_pipe_lsu`, `smsp__issue_active`, and `sm__cycles_active.avg` for the `v3 R=4` / `v3 R=16` ranges against `reference`. Expected: V3 dots XU ≈ 0 % (was 30 / 57 %), V3 up LSU well under 60 % at R=16.

## 4. P1: untraced harness A/B, ABAB (GPU, both cards, ~20 min)

R680's harness without nsys, `--capture-mode events`. The env comes from the live launcher's `EXTRA_ENV` (23 keys). The flag is read at import, so each arm is its own process. Order per shape is OFF, ON, OFF, ON, all on `tabbyapi:hcfast-r1`.

```sh
LIVE=/srv/qwen5090/launch-flashnext.sh; MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
HARNESS=/srv/qwen5090/probes/r465/profile_decode_events.py; METER=/srv/qwen5090/probes/r465/events_meter.py
EXTRA=$(sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-([^}]*)\}.*/\1/p' "$LIVE"); ENVS=""; for kv in $EXTRA; do ENVS="$ENVS -e $kv"; done
CAPENV=""          # or "-e EXL3_HC_MIX_V3_DOTS_B=.. -e EXL3_HC_MIX_V3_UP_B=.." if P0 named a tile winner
. /srv/qwen5090/lib/gpu-queue.sh; . /srv/qwen5090/lib/serve-ctl.sh; gpu_lock; served_stop
arm(){ local tag=$1 flag=$2 b=$3 d=$4 caps=""; [ $flag = 1 ] && caps="$CAPENV"
  sudo docker run --rm --name hcfast-p1 --gpus all --ipc=host --shm-size=16g \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v $HARNESS:/probe/profile_decode_events.py:ro \
    -v $METER:/probe/events_meter.py:ro -v $R:/results $ENVS -e EXL3_HC_MIX_V3=$flag $caps \
    --entrypoint python3 tabbyapi:hcfast-r1 /probe/profile_decode_events.py --model /models/$MODEL \
    --out /results/p1-$tag --cache-quant 8,8 --gpu-split 30,30 --max-chunk-size 2048 --tokens 64 \
    --warmup-steps 16 --settle-steps 8 --capture-steps 32 --capture-mode events \
    --contexts 4096 --batch $b --draft $d > $R/p1-$tag.log 2>&1; }
for shape in "c4d3 4 3" "c8d1 8 1" "c1d3 1 3"; do set -- $shape
  arm $1-A1 0 $2 $3; arm $1-B1 1 $2 $3; arm $1-A2 0 $2 $3; arm $1-B2 1 $2 $3; done
grep -H "ms/iterate" $R/p1-*/ctx4096_b*_d*/kernels.txt
```

Read each shape at its own path, `p1-<tag>/ctx4096_b<B>_d<D>/` (`c4d3` → `ctx4096_b4_d3`, `c8d1` → `ctx4096_b8_d1`, `c1d3` → `ctx4096_b1_d3`). The `ctx4096_bN_d0` dirs of the same runs are informational controls. Per-device spans are in `events.json` → `iterations[].device_stream_span_ms`.

- **Identity (must hold):** `ctx4096_b<B>_d<D>/sequence-hashes.json` of each ON arm is byte-equal to the OFF arm of the same pair (`cmp`). The harness uses `GreedySampler`, so any difference is a numerics change: FAIL.
- **Pass (decision shapes c4 d3 and c8 d1):** OFF − ON ≥ **0.6 ms/iterate** in **both** ABAB pairs (A1−B1 and A2−B2) at both shapes.
- **Kill:** mean OFF − ON < **0.3 ms/iterate** at both c4 d3 and c8 d1. The chain is then off the critical path (or P0's gain does not survive the served step).
- **Predicted** (estimates, ANALYSIS.md §5, revised after R697 in §7): 0.7-1.3 ms at c4 d3 and c8 d1. c1 d3 is informational, predicted 0.2-0.5 ms, inside the ~20 % single-stream boot-to-boot band (R592).
- **Noise reference:** R684's A-arm spread was 0.09 ms (c4 d3) and 0.38 ms (c8 d1), `review-r682-r684-2026-09-24.md` §R684.

## 5. Served greedy identity (fn_greedy, 6/6)

The R651 / R676 / R684 procedure. The reference comes from the live daily; the candidate then boots through the same launcher with `IMG` and `EXTRA_ENV_ADD`.

```sh
GREEDY=/srv/qwen5090/probes/fn_greedy.py; G=$R/greedy.jsonl
boot(){ served_stop; wait_unserved 45; env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  "$@" bash $LIVE > $R/boot-$(date +%s).log 2>&1 && wait_served_id "$MODEL" 200 8; }
boot                                                                  # the unchanged daily (slotfix-r1)
python3 $GREEDY --url http://127.0.0.1:8022 --tag ref --out $G
boot IMG=tabbyapi:hcfast-r1 EXTRA_ENV_ADD="EXL3_HC_MIX_V3=1"            # + the P0 tile caps if any
sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep EXL3_HC_MIX_V3   # must show =1
python3 $GREEDY --url http://127.0.0.1:8022 --tag on --out $G
python3 $GREEDY --compare --ref ref --out $G | tee -a $R/greedy.log   # divergences=0 (6/6)
boot IMG=tabbyapi:hcfast-r1                                           # flag off on the rebuilt image
python3 $GREEDY --url http://127.0.0.1:8022 --tag off --out $G
python3 $GREEDY --compare --ref ref --out $G | tee -a $R/greedy.log   # divergences=0: the rebuild is inert
boot; finish_restore $LIVE                                            # back to the daily
```

- **Pass:** `divergences=0` for both `on` and `off` against `ref`, over short0-4 and long100k.
- A 5/6 set with a missing or errored prompt is not a pass. The review noted that `--compare` printed IDENTICAL on 5/6 in R683; read the per-prompt lines.
- A non-zero `on` with a zero `off` contradicts step 2: report the first divergent prompt.
- A non-zero `off` means the rebuilt extension differs from the base image's, independent of the flag.

Promotion is the usual autonomous Flash-Next rule once P1 passes and the canonical gate (c1/c4/c8 at 4k, c4 at 26k vs R653) passes: `DAILY_IMG=tabbyapi:hcfast-r1`, plus `EXL3_HC_MIX_V3=1` (and any P0 tile caps) in `EXTRA_ENV`.

## Notes

- The mixer workspaces are the served ones (`g_tensor_cache` buckets). V3 needs `dots` 16-byte aligned. Bucket slices start at offset 0 of a caching-allocator block, so they are aligned; if one is not, V3 up declines to the copied served kernel for that call (parity section 5 tests this).
- Rows 33+ (prefill) never reach this code. Shapes other than hidden 2560 / rank 320 take the copied served kernels, so the same image stays correct on other checkpoints.
- The auto tile (`0`) queries `cudaOccupancyMaxActiveBlocksPerMultiprocessor` once per device and kernel, host-side, and caches it. The choice is fixed per device, so captured graphs stay consistent.
