#!/usr/bin/env bash
# R710b (2026-09-24, operator): R710 re-armed after its P0. P0 (R710 audit.log) measured the gemm/mgemm V2 twins 3-6x SLOWER
# than served at every row count (in_proj r4 26 -> 104 us, out_proj r16 20 -> 77 us) and the gemv twin 26-35 % faster at rows
# 4-8 (RECOMMEND EXL3_DENSE_V2=3, ~ -296 us/step at c1d3). R710 was stopped in P1 round 1 (ON and GM both carry the slow
# twins). Arms here: OFF, GV = EXL3_DENSE_V2=3 (gemv twin only, ROWS32 off: served never exceeds 16 rows), OFF2 = an
# in-run A/A (its verdict must read flat). Greedy on = EXL3_DENSE_V2=3. Image, parity and P0 are unchanged from R710.
# densegemm r1 gate (queue as one unit; modelled on r705-pdl-gate.sh / r703-moefast-r2.sh). The operator assigns the
# R number: install this directory as /srv/qwen5090/patches/exllamav3/densegemm-r1, this file as
# /srv/qwen5090/rNNN-densegemm-gate.sh, and queue rNNN-densegemm-gate (DG_UNIT = the file name).
# Flags under test, both default off, bitwise identical by construction (ANALYSIS.md, test_densegemm_cpu.py):
#   EXL3_DENSE_V2=1      V2 twins of the dense K4 decode GEMMs: gemm + sliced/per-matrix mgemm (gathered stream-K
#                        fixup, weight prologue before the input stage, trimmed barriers, fused output transform) and
#                        the 3-8-row gemv (cp.async weight ring, last-arriver output transform)
#   EXL3_DENSE_V2=2      gemm + mgemm twins only (the gemv stays served);  =3 gemv twin only
#   EXL3_DENSE_ROWS32=1  one-pass 17-32-row gemm / mgemm (inert at the served decode shapes, which are <= 16 rows)
# Image: tabbyapi:densegemm-r1 = the daily image (DAILY_IMG, tabbyapi:stack-r2 = slotfix-r1 + hcfast-r1 + moefast-r1
# since 2026-09-24 08:46 UTC) + densegemm-r1.patch. Every arm carries the daily's 27-key EXTRA_ENV.
# Parts:
#   0 build in-lock (BEFORE any measurement; OPERATIONS §15) when the image is missing or was built from another
#     patch or base (labels local.densegemm.patch_sha256 / local.densegemm.base). The install fails unless every
#     function of the base .so keeps its SASS and exactly 32 twins were added. Landing check imports torch first.
#   1 parity, one card (test_densegemm_parity.py): every dense K4 projection of the checkpoint, rows 1-32, fp16 and
#     fp32 outputs, OFF vs V2 bitwise, V2 engaged at rows 4 and 16, stress, CUDA graph capture/replay -> FAIL stops
#   2 P0 (bench_densegemm_p0.py): us per call per projection per rows, arms m0/m1/m2/m3, RECOMMEND. Diagnostic only.
#   3 P1 in-process harness, 4k, untraced: ROUNDS x (OFF, ON, GM) per shape, order rotated so each arm runs first
#     ROUNDS/3 times, at c4d3, c8d1, c1d3. The harness also runs every batch at draft 0 (ctx4096_b{1,4,8}_d0): those
#     cells are read and reported next to the drafting cells. Sequence hashes of ON and GM == the round's OFF, in
#     the drafting and the draft-0 cells.
#       OFF = daily env   ON = + EXL3_DENSE_V2=1 EXL3_DENSE_ROWS32=1   GM = + EXL3_DENSE_V2=2 EXL3_DENSE_ROWS32=1
#   4 served greedy (fn_greedy): ref = the daily; $IMG flags off; $IMG with EXL3_DENSE_V2=1 EXL3_DENSE_ROWS32=1 (every
#     twin; GM runs a subset of the same kernels) -> GREEDY-SUMMARY divergences must be 0.
# DECISION (OPERATIONS §16 gate template, pre-registered), per arm ON / GM:
#   identity    parity PASS; P1 hashes identical in every round, shape and draft-0 cell; greedy divergences = 0
#   pairs       >= 5 OFF/arm pairs at each of c4d3, c8d1, c1d3 (ROUNDS >= 6, a multiple of 3)
#   regression  no same-sign regression at c4d3 or c8d1 (all pairs > 0 = reject)
#   c1 bound    a same-sign c1d3 regression passes only if its mean is <= 2 % AND a same-sign c4d3 or c8d1 gain is
#               larger in ms than the c1d3 loss; recorded as the arm's c1 cost
#   gain        at least one of c1d3 / c4d3 / c8d1 is a same-sign GAIN (all pairs < 0). Otherwise FLAT: not a stack
#               entry, whatever the other clauses say.
# The draft-0 cells are reported (sign, mean, %) but do not decide. This unit does not promote.
# GPU ~85 min (estimate: build ~15 if needed, parity ~8, P0 ~10, P1 54 harness runs ~60, three boots + greedy ~12).
# Queue-chained; the daily is restored at the end.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
DG_UNIT=${DG_UNIT:-$(basename "$0" .sh)}   # r515-queue-chain runs /srv/qwen5090/<name>.sh with no env: the file name is the unit
R=${R:-/srv/qwen5090/results/$(date +%F)-$DG_UNIT}; mkdir -p "$R"
PD=${PD:-/srv/qwen5090/patches/exllamav3/densegemm-r1}
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
HARNESS=/srv/qwen5090/probes/r465/profile_decode_events.py
METER=/srv/qwen5090/probes/r465/events_meter.py
GREEDY=/srv/qwen5090/probes/fn_greedy.py
IMG=${IMG_DG:-tabbyapi:densegemm-r1}
ROUNDS=${ROUNDS:-6}
ARMS="OFF GV OFF2"
log(){ echo "$(date -Is) [$DG_UNIT] $*" | tee -a "$R/audit.log"; }
[ $(( ROUNDS % 3 )) = 0 ] && [ "$ROUNDS" -ge 6 ] || { log "ABORT: ROUNDS=$ROUNDS must be a multiple of 3 and >= 6 (fair rotation, >= 5 pairs)"; exit 3; }
export GPU_QUEUE_NAME=$DG_UNIT
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ sudo docker rm -f dg-p dg-p1 >/dev/null 2>&1 || true
  sudo docker logs flashnext > "$R/container-final.log" 2>&1 || true
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== densegemm r1 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$HARNESS" "$METER" "$GREEDY" "$PD/Dockerfile.box" "$PD/densegemm-r1.patch" "$PD/install-densegemm.sh" \
         "$PD/test_densegemm_cpu.py" "$PD/test_densegemm_parity.py" "$PD/bench_densegemm_p0.py" "$PD/d0/d0_microbench.py" \
         "$PD/rebuild-native.py" "$PD/sass_hashes.py"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
BASE=$(grep -oE '^DAILY_IMG=\S+' "$LIVE" | cut -d= -f2)
[ -n "$BASE" ] && sudo docker image inspect "$BASE" >/dev/null 2>&1 || { log "ABORT: daily image '$BASE' not found"; exit 3; }
EXTRA=$(sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-([^}]*)\}.*/\1/p' "$LIVE"); ENVS=""; for kv in $EXTRA; do ENVS="$ENVS -e $kv"; done
echo "$EXTRA" | tr ' ' '\n' | grep -qx 'EXL3_MOE_COOP_V2=1' || { log "ABORT: live env lacks EXL3_MOE_COOP_V2=1"; exit 3; }
echo "$EXTRA" | tr ' ' '\n' | grep -qE '^EXL3_DENSE_(V2|ROWS32)=' && { log "ABORT: live env already sets EXL3_DENSE_V2 / EXL3_DENSE_ROWS32"; exit 3; }
log "live env: $(echo "$EXTRA" | wc -w) keys"
PATCH_SHA=$(sha256sum "$PD/densegemm-r1.patch" | cut -c1-64)

gpu_lock
log "lock held; served at entry: $(served_id || echo none); base $BASE; image $IMG; patch sha256 $PATCH_SHA"

# ---- 0. build + landing ----
have=$(sudo docker image inspect "$IMG" --format '{{index .Config.Labels "local.densegemm.patch_sha256"}} {{index .Config.Labels "local.densegemm.base"}}' 2>/dev/null || true)
if [ "$have" != "$PATCH_SHA $BASE" ]; then
  log "building $IMG on $BASE (image labels: '${have:-none}')"
  ( cd "$PD" && sudo docker build -f Dockerfile.box --build-arg BASE="$BASE" --build-arg MAX_JOBS=4 \
      --build-arg PATCH_SHA="$PATCH_SHA" -t "$IMG" . ) > "$R/build.log" 2>&1 \
    || { log "BUILD FAILED"; tail -40 "$R/build.log" | sed 's/^/  /' | tee -a "$R/audit.log"; finish BUILD-FAILED; exit 3; }
  grep -aE "densegemm: |densegemm r1 landed|CPU TESTS" "$R/build.log" | sed 's/^/  [build] /' | tee -a "$R/audit.log"
else
  log "image $IMG already built from this patch on $BASE"
fi
sudo docker run --rm -v "$R":/results --entrypoint bash "$IMG" -c \
  'cp /opt/densegemm-r1/install/base-sass.txt /results/sass-base.txt && cp /opt/densegemm-r1/install/rebuilt-sass.txt /results/sass-rebuilt.txt && cp /opt/densegemm-r1/install/rebuilt-res-usage.txt /results/res-usage.txt' \
  || log "WARNING: install SASS lists not found in the image"
sudo docker run --rm --entrypoint python3 "$IMG" -c '
import torch, exllamav3_ext as e
assert e.dense_v2_revision == 1
assert tuple(e.dense_v2_modes()) == (0, 0), "flags must default off"
print("landed: dense_v2_revision", e.dense_v2_revision, "modes", tuple(e.dense_v2_modes()),
      "| base: moefast", getattr(e, "moe_coop_v3_revision", None))' > "$R/landed.txt" 2>&1 \
  && sudo docker run --rm -e EXL3_DENSE_V2=1 -e EXL3_DENSE_ROWS32=1 --entrypoint python3 "$IMG" -c \
     'import torch, exllamav3_ext as e; assert tuple(e.dense_v2_modes()) == (1, 1); print("env on -> (1, 1)")' >> "$R/landed.txt" 2>&1 \
  || { log "ABORT: landing check failed: $(tail -2 "$R/landed.txt")"; finish ABORTED; exit 3; }
log "image $IMG: $(tr '\n' ' ' < "$R/landed.txt")"

served_stop; wait_unserved 45
# The persisted tune cache is mounted like P1's, so parity and P0 run the served daily's tuned configs (V2 reuses them)
one(){ sudo docker run --rm --name dg-p --gpus '"device=0"' --ipc=host -v /srv/qwen5090/models:/models:ro -v "$R":/results \
  -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
  $ENVS -e CUDA_VISIBLE_DEVICES=0 --entrypoint python3 "$IMG" "$@"; }

# ---- 1. parity ----
one /opt/densegemm-r1/test_densegemm_parity.py --model /models/$MODEL --json /results/parity.json > "$R/parity.log" 2>&1
rc=$?
log "parity rc=$rc: $(grep -aE 'PARITY (PASS|FAIL)' "$R/parity.log" | tail -1)"
grep -aE "\[parity\] (gemm|mgemm)/.* rows (4|16):|\[parity\] (stress|graph|VRAM|cells):|NOT ENGAGED|MISMATCH|EXCEPTION" "$R/parity.log" \
  | head -60 | sed 's/^/  /' | tee -a "$R/audit.log"
grep -aq "PARITY PASS" "$R/parity.log" && [ $rc = 0 ] || { tail -25 "$R/parity.log" | sed 's/^/  /' | tee -a "$R/audit.log"; finish PARITY-FAIL; exit 5; }

# ---- 2. P0 (diagnostic) ----
one /opt/densegemm-r1/bench_densegemm_p0.py --model /models/$MODEL --json /results/p0.json > "$R/p0.log" 2>&1
log "P0 rc=$?"
grep -aE "^\[p0\] |^ *[0-9]+ +[0-9.]+ |^m[0-9]:|^RECOMMEND|^rows32" "$R/p0.log" | sed 's/^/  /' | tee -a "$R/audit.log"

# ---- 3. P1 in-process, ROUNDS x (OFF, GV, OFF2), rotated ----
flags_of(){ case $1 in OFF|OFF2) echo "";; GV) echo "EXL3_DENSE_V2=3";; esac; }
p1(){ local tag=$1 flags=$2 b=$3 d=$4 fe=""
  for kv in $flags; do fe="$fe -e $kv"; done
  sudo docker run --rm --name dg-p1 --gpus all --ipc=host --shm-size=16g \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v "$HARNESS":/probe/profile_decode_events.py:ro \
    -v "$METER":/probe/events_meter.py:ro -v "$R":/results $ENVS $fe \
    --entrypoint python3 "$IMG" /probe/profile_decode_events.py --model /models/$MODEL \
    --out /results/p1-$tag --cache-quant 8,8 --gpu-split 30,30 --max-chunk-size 2048 --tokens 64 \
    --warmup-steps 16 --settle-steps 8 --capture-steps 32 --capture-mode events \
    --contexts 4096 --batch $b --draft $d > "$R/p1-$tag.log" 2>&1
  log "  P1 $tag rc=$?: d$d $(grep -ahoE '[0-9.]+ ms/iterate' "$R"/p1-$tag/ctx4096_b${b}_d${d}/kernels.txt 2>/dev/null | head -1); d0 $(grep -ahoE '[0-9.]+ ms/iterate' "$R"/p1-$tag/ctx4096_b${b}_d0/kernels.txt 2>/dev/null | head -1)"; }
for shape in "c4d3 4 3" "c8d1 8 1" "c1d3 1 3"; do set -- $shape; sh=$1 b=$2 d=$3
  for r in $(seq 1 "$ROUNDS"); do
    k=$(( (r - 1) % 3 )); order=$(echo $ARMS $ARMS | cut -d' ' -f$(( k + 1 ))-$(( k + 3 )))
    for a in $order; do p1 $sh-$a$r "$(flags_of $a)" $b $d; done
    for a in GV OFF2; do for dd in $d 0; do
      cmp -s "$R/p1-$sh-OFF$r/ctx4096_b${b}_d$dd/sequence-hashes.json" "$R/p1-$sh-$a$r/ctx4096_b${b}_d$dd/sequence-hashes.json" \
        && log "  $sh round $r $a d$dd: hashes IDENTICAL" || log "  $sh round $r $a d$dd: hashes DIFFER -> FAIL"
    done; done
  done
done
python3 - "$R" "$ROUNDS" <<'EOF' 2>&1 | tee "$R/p1-summary.txt" | tee -a "$R/audit.log"
import re, sys, statistics as st, filecmp, os
R, N = sys.argv[1], int(sys.argv[2])
SH = {"c4d3": (4, 3), "c8d1": (8, 1), "c1d3": (1, 3)}
def ms(tag, b, d):
    try:
        t = open(f"{R}/p1-{tag}/ctx4096_b{b}_d{d}/kernels.txt").read()
        return float(re.search(r"([0-9.]+) ms/iterate", t).group(1))
    except Exception:
        return None
def cell(sh, arm, d):
    b = SH[sh][0]
    ds, offs, same = [], [], 0
    for r in range(1, N + 1):
        o, x = ms(f"{sh}-OFF{r}", b, d), ms(f"{sh}-{arm}{r}", b, d)
        if o is not None: offs.append(o)
        if o is not None and x is not None: ds.append(x - o)
        h0 = f"{R}/p1-{sh}-OFF{r}/ctx4096_b{b}_d{d}/sequence-hashes.json"
        h1 = f"{R}/p1-{sh}-{arm}{r}/ctx4096_b{b}_d{d}/sequence-hashes.json"
        same += os.path.exists(h0) and os.path.exists(h1) and filecmp.cmp(h0, h1, shallow=False)
    if not ds:
        return dict(n=0, same=same, sign="no-data")
    sign = "GAIN" if all(v < 0 for v in ds) else "REGRESSION" if all(v > 0 for v in ds) else "flat"
    m = st.mean(ds)
    return dict(n=len(ds), same=same, sign=sign, mean=m, pct=100 * m / st.mean(offs), ds=ds, off=st.mean(offs))
for arm in ("GV", "OFF2"):
    fails, gains, c1cost = [], [], None
    res = {}
    for sh, (b, d) in SH.items():
        for dd, label in ((d, "draft"), (0, "d0")):
            c = cell(sh, arm, dd)
            res[(sh, label)] = c
            if c["n"]:
                print(f"P1 {arm} {sh} {label:5s} ({arm}-OFF ms): " + " ".join(f"{v:+.3f}" for v in c["ds"])
                      + f"  mean {c['mean']:+.3f} ({c['pct']:+.2f} %) {c['sign']}; hashes {c['same']}/{N}  [n={c['n']}]")
            else:
                print(f"P1 {arm} {sh} {label}: no data")
            if c["same"] != N:
                fails.append(f"{sh} {label} hashes {c['same']}/{N}")
    for sh in SH:
        c = res[(sh, "draft")]
        if c["n"] < 5:
            fails.append(f"{sh} only {c['n']} pairs")
            continue
        if c["sign"] == "GAIN":
            gains.append(sh)
    for sh in ("c4d3", "c8d1"):
        if res[(sh, "draft")].get("sign") == "REGRESSION":
            fails.append(f"{sh} same-sign REGRESSION {res[(sh, 'draft')]['pct']:+.2f} %")
    c1 = res[("c1d3", "draft")]
    if c1.get("sign") == "REGRESSION":
        gain_ms = max([-res[(s, "draft")]["mean"] for s in ("c4d3", "c8d1") if res[(s, "draft")].get("sign") == "GAIN"] or [0.0])
        c1cost = f"c1d3 cost {c1['pct']:+.2f} % ({c1['mean']:+.3f} ms) vs best c4/c8 gain {gain_ms:.3f} ms"
        if c1["pct"] > 2.0 or gain_ms <= c1["mean"]:
            fails.append(f"c1d3 same-sign regression outside the bound ({c1cost})")
    if fails:
        verdict = "FAIL: " + "; ".join(fails)
    elif not gains:
        verdict = "FLAT (identity + no regression, but no same-sign gain at any shape): not a stack entry"
    else:
        verdict = "PASS pending greedy (identity + no same-sign regression; GAIN at " + ", ".join(gains) + ")" \
                  + (f"; {c1cost}, record in STACK.md" if c1cost else "")
    d0 = ", ".join(f"{sh} {res[(sh, 'd0')]['sign']} {res[(sh, 'd0')].get('pct', 0):+.2f} %" for sh in SH if res[(sh, "d0")]["n"])
    print(f"P1 draft-0 cells {arm}: {d0}")
    print(f"P1 VERDICT {arm}: {verdict}")
EOF

# ---- 4. served greedy identity (gated) ----
G=$R/greedy.jsonl
boot(){ served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "$@" bash "$LIVE" > "$R/boot-$(date +%s).log" 2>&1 \
    && wait_served_id "$MODEL" 200 8; }
GREEDY_OK=1
boot || { log "daily NO BOOT"; finish NO-BOOT; exit 3; }
python3 "$GREEDY" --url http://127.0.0.1:8022 --tag ref --out "$G" > "$R/greedy-ref.log" 2>&1
for a in off on; do
  if [ $a = on ]; then boot IMG=$IMG EXTRA_ENV_ADD="EXL3_DENSE_V2=3"; else boot IMG=$IMG; fi \
    || { log "$a NO BOOT"; GREEDY_OK=0; continue; }
  log "  greedy $a: $(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'); flags in env: $(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^EXL3_DENSE_' | tr '\n' ' '); free $(vram_free)"
  python3 "$GREEDY" --url http://127.0.0.1:8022 --tag $a --out "$G" > "$R/greedy-$a.log" 2>&1
  sudo docker logs flashnext > "$R/container-greedy-$a.log" 2>&1
  log "  greedy $a: OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-greedy-$a.log"); tracebacks $(grep -ac Traceback "$R/container-greedy-$a.log")"
done
python3 "$GREEDY" --compare --ref ref --out "$G" > "$R/greedy-compare.log" 2>&1
grep -aE "vs ref|GREEDY-SUMMARY|MISSING" "$R/greedy-compare.log" | sed "s/^/  [greedy] /" | tee -a "$R/audit.log"
n_tags=$(grep -acE "^GREEDY (on|off) vs ref" "$R/greedy-compare.log")
grep -aqE "^GREEDY-SUMMARY reference=ref divergences=0$" "$R/greedy-compare.log" && [ "$n_tags" = 2 ] && [ $GREEDY_OK = 1 ] \
  || GREEDY_OK=0

# ---- DECISION ----
for arm in GV OFF2; do
  v=$(grep -aE "^P1 VERDICT $arm:" "$R/p1-summary.txt" | sed -E "s/^P1 VERDICT $arm: //")
  if [ $GREEDY_OK = 0 ]; then d="REJECT (greedy: divergences != 0, a missing arm or a failed boot; P1: $v)"
  else case "$v" in PASS*) d="ACCEPT into the stack (${v#PASS pending greedy })";; FLAT*) d="FLAT, not a stack entry";; *) d="REJECT ($v)";; esac; fi
  log "DECISION $arm ($(flags_of $arm)): $d"
done
finish DONE
