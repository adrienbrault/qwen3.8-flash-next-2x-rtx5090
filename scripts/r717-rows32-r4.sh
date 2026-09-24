#!/usr/bin/env bash
# rows32 r4 gate: MTP verify at 17-32 rows on stack-r3, with the pre-registered served policy [[4, 3], [8, 2]].
# Queue as one unit. The operator assigns the R number: install out-rows32-r4/ (without work/) as
# /srv/qwen5090/patches/exllamav3/rows32-r4 and this file as /srv/qwen5090/rNNN-rows32-r4.sh, then queue
# rNNN-rows32-r4 (R32_UNIT = the file name). Queue it only after R716b (stack-r3) has a DECISION UNION: PASS.
# 2026-09-24: R716b ended REJECT on the greedy_streams max-of-3 false alarm; R716c proved stack-r3 bitwise at every
# served shape and stack-r3 was promoted 20:10 UTC. R717 runs with S3_ALLOW_UNACCEPTED=1 and S3_ACCEPT_NOTE (logged).
#
# Arms (A = the daily-to-be, B = A + rows32):
#   A  IMG=$BASE (tabbyapi:stack-r3, INCLUDE "mf3 dg2") with EXTRA_ENV = the served env minus the families S3_UNION
#      sets, plus S3_UNION (stack-r3's arm_env). S3_UNION's pre-registered default is stack-r3's RECOMMENDED union for
#      INCLUDE "mf3 dg2" (r716b-stack-r3.sh: HC2 RR QT QF + EXL3_DENSE_V2=1 + R713's MF3 line). A_EXTRA_ENV="..." (the
#      whole line, e.g. R716b's RECOMMENDED EXTRA_ENV) overrides it. DRAFT_POLICY = [[4, 3], [5, 2], [8, 1]] (served).
#   B  IMG=$IMG (tabbyapi:stack-r3-rows32 = $BASE + rows32-r4.patch) with EXTRA_ENV = A's, where
#      EXL3_MOE_COOP_V3_MAP gets ",17-32:2" appended (moefast r3's mode 3 never ran above 16 rows; the launcher already
#      falls back to mode 2 there because its run table holds 16 rows: the MAP makes that explicit), plus
#      EXL3_DENSE_ROWS32=1 EXL3_MOE_COOP_ROWS32=1 EXL3_SHARED_EXPERT_ROWS32=1, and DRAFT_POLICY = [[4, 3], [8, 2]]
#      (c6 / c7 / c8 at depth 2 = 18 / 21 / 24 verify rows; c1-c5 unchanged; nothing above 24 rows is served).
#   Every boot passes the whole EXTRA_ENV (never EXTRA_ENV_ADD: the MAP key must be replaced, not duplicated), IMG,
#   DRAFT_POLICY and NVME_TIER= explicitly (§16 R709c), so A is the same whether or not stack-r3 is promoted yet.
#
# Parts:
#   0 build $IMG in-lock before any measurement (§15) when missing or built from another patch / base image ID;
#     SASS identity (0 changed, 0 added), landing (re-exec without the image's EXL3_* ENV, stack-r3's landing, torch
#     first, rows32 markers and flags), CPU tests; all read back into $R.
#   1 kernel parity, one card, FAIL stops:
#       rows32   test_rows32_parity.py under the A env: 4b at 17-32 rows (every row count, so 18 / 21 / 24), arms V3
#                unset/1/2/3, the A env (MAP 2-4:2) and the B env (MAP 2-4:2,17-32:2), bitwise vs two served 16-row
#                calls (mode 3) which must equal the mode-2 16-row call; <= 16 rows unchanged; dispatch by kernel
#                names (17-32 rows: V3 kernels, no r2 kernel, in both envs; 16 rows: r2 kernels); stress; 4c eager ==
#                graph at 17-32 rows with EXL3_DENSE_ROWS32 on and off, on == off; join (exl3_moe_coop_ev) at 18/24/32
#       dense    densegemm-r2's parity (from the image) with ROWS 1 4 8 16 17 18 21 24 32, --mode <A's EXL3_DENSE_V2>;
#                every served family must be ENGAGED at 17 / 18 / 21 / 24 / 32 rows (else the parity is vacuous)
#       latchain latchain's parity with the (bsz, q_len) shapes the policy serves: (6,3) (7,3) (8,3) + controls
#       hcfast   hcfast r2's parity with ROWS 1 4 8 16 17 18 21 24 32
#       then test_rows32_cpu.py --require-flags under the A env, and the P0 timing table (--bench, diagnostic)
#   2 P1 §16 shapes (c4d3, c8d1, c1d3) + d0 cells, in-process harness, 4k, untraced: ROUNDS x (OFF = A env, U = B env
#     without the policy), first arm alternating; hashes U == OFF in every pair and cell; stack-r3's p1_stack.py
#     reads raw / per-iterate median / stall-excluded paired means (its UNION verdict asks for a gain; rows32 is inert
#     at <= 16 rows, so rows32_decide.py applies identity + no regression + the c1 bound instead)
#   3 P1 rows32 shapes c6 / c7 / c8: SRV (A env, depth 1) vs R32 (B env, depth 2), R32_ROUNDS pairs alternating;
#     ON1 (B env, depth 1) == SRV in rounds 1 and 3; R32 identical across rounds; at c8 depth 2 in round 1 the A/A
#     matrix R32noQF (QF off: the lcguard side path at 24 rows), R32noDR (EXL3_DENSE_ROWS32=0: the dense twins),
#     R32noMAP (A's MAP: the automatic mode-3 -> 2 fallback) must all equal R32; OFF2 (A env, depth 2 = the generic
#     > 16-row path) once for the mechanism; lcguard side-slot refusals counted in the R32 logs
#   4 served ABBA, boots A1 B1 B2 A2 (NVME_TIER= on every boot). Per boot, first after the boot: greedy_conc.py
#     (A1: c1 P + c8 Q; B1: c8 P; B2: c8 Q; A2: c8 P + c1 Q), fn_greedy (A1 = ref, B1, A2 = A/A); then the R707
#     instrument with --distinct (fn_bench, greedy, 1,024 forced tokens, 1 warm-up + 3 runs, c1..c8, code + prose);
#     UP-line free (boot log), free after the ramp, container env / log.
# DECISION (pre-registered, user-approved promotion conditions; rows32_decide.py; this unit does not promote):
#   1 kernel parity PASS (every suite above)
#   2 greedy_conc: B's c8-vs-c1 divergences <= the worse A half's (the daily's own c8-vs-c1 variation measured the
#     same way; the two A halves are its A/A), early divergences too, no errors
#   3 served per-stream decode B/A (median per boot and cell, mean of the two pairs): >= 1.03 at c6 and c8, code AND
#     prose, both pairs > 1.00 there; c2-c5 (and c7) >= 0.985; c1 >= 0.98
#   usual: §16 P1 identity + no same-sign regression at c4d3/c8d1 + c1 bound; rows32 P1 identities; fn_greedy B1 = ref
#     (VOID if the A/A A2 diverges); headroom per card: min B UP-line free >= min A - 32 MiB; 0 OOM / TORCH_CHECK /
#     tracebacks; env and policy as intended in every boot.
# GPU ~4 h: build ~20 min CPU in-lock if needed (the daily keeps serving); parity ~45 (rows32 ~20, dense ~10,
# latchain ~5, hcfast ~8, cpu + P0 ~3); P1 §16 36 runs ~25; P1 rows32 ~31 runs ~22; served 4 boots ~90 (boot 3,
# greedy_conc 3 (A) / 1 (B), fn_greedy 4, R707 instrument ~14). Queue-chained; the daily is restored at the end.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
R32_UNIT=${R32_UNIT:-$(basename "$0" .sh)}
R=${R:-/srv/qwen5090/results/$(date +%F)-$R32_UNIT}; mkdir -p "$R"
PD=${PD:-/srv/qwen5090/patches/exllamav3/rows32-r4}
S3D=${S3D:-/srv/qwen5090/patches/exllamav3/stack-r3}
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
API=http://127.0.0.1:8022/v1
HARNESS=/srv/qwen5090/probes/r465/profile_decode_events.py
METER=/srv/qwen5090/probes/r465/events_meter.py
GREEDY=/srv/qwen5090/probes/fn_greedy.py
BENCH=/srv/qwen5090/probes/fn_bench.py
P1S=$S3D/tools/p1_stack.py
BASE=${BASE_R32:-tabbyapi:stack-r3}
IMG=${IMG_R32:-tabbyapi:stack-r3-rows32}
S3_UNION_DEFAULT="EXL3_HC_MIX_V3=2 EXL3_HC_MIX_V3_DOTS_B=1:1,4:2,32:4 EXL3_HC_MIX_V3_UP_B=1:1,8:4,32:8 EXL3_HC_MIX_V3_DOTS_J=1:4,32:8 EXL3_HC_MIX_V3_DOTS_PF=1:1,32:0 EXL3_HC_MIX_V3_UP_Q=1:4,8:2,32:4 EXL3_HC_MIX_V3_PDL=0 EXL3_LC_GDN_RR=1 EXL3_LC_QSA_SPLIT_STAGES=2 EXL3_LC_QSA_COMBINE_STAGES=1 EXL3_LC_QSA_DIV16=1 EXL3_LC_QSA_FORK=1 EXL3_DENSE_V2=1 EXL3_MOE_COOP_V3=3 EXL3_MOE_COOP_V3_MAP=2-4:2 EXL3_SHARED_EXPERT_EARLY=1"
S3_UNION=${S3_UNION:-$S3_UNION_DEFAULT}
A_EXTRA_ENV=${A_EXTRA_ENV:-}
S3_AUDIT=${S3_AUDIT:-$(ls -t /srv/qwen5090/results/*-r716b-stack-r3/audit.log 2>/dev/null | head -1)}
FLAGS="EXL3_DENSE_ROWS32=1 EXL3_MOE_COOP_ROWS32=1 EXL3_SHARED_EXPERT_ROWS32=1"
MAP_ADD=${MAP_ADD:-17-32:2}
POLICY_A='[[4, 3], [5, 2], [8, 1]]'
POLICY_B=${POLICY_B:-[[4, 3], [8, 2]]}
ROUNDS=${ROUNDS:-6}
R32_ROUNDS=${R32_ROUNDS:-4}
log(){ echo "$(date -Is) [$R32_UNIT] $*" | tee -a "$R/audit.log"; }
[ $(( ROUNDS % 2 )) = 0 ] && [ "$ROUNDS" -ge 6 ] || { log "ABORT: ROUNDS=$ROUNDS must be even and >= 6 (balanced first arm, >= 5 pairs)"; exit 3; }
[ $(( R32_ROUNDS % 2 )) = 0 ] && [ "$R32_ROUNDS" -ge 4 ] || { log "ABORT: R32_ROUNDS=$R32_ROUNDS must be even and >= 4"; exit 3; }
export GPU_QUEUE_NAME=$R32_UNIT
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ sudo docker rm -f r32-p r32-p1 >/dev/null 2>&1 || true
  sudo docker logs flashnext > "$R/container-final.log" 2>&1 || true
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== rows32 r4 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$HARNESS" "$METER" "$GREEDY" "$BENCH" "$P1S" "$PD/Dockerfile.box" "$PD/rows32-r4.patch" \
         "$PD/install-rows32.sh" "$PD/rebuild-native.py" "$PD/sass_hashes.py" "$PD/sass_identity.py" "$PD/landing_rows32.py" \
         "$PD/test_rows32_cpu.py" "$PD/test_rows32_parity.py" "$PD/rows32_extra_parity.py" "$PD/greedy_conc.py" \
         "$PD/rows32_decide.py"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
[ -e "$PD/work" ] && log "WARNING: $PD/work exists (make-patch.sh's trees); .dockerignore keeps it out of the build context"

# ---- configuration: families, A env, B env ----
fam(){ case "${1%%=*}" in
  EXL3_HC_MIX_V3|EXL3_HC_MIX_V3_*) echo HC2 ;;
  EXL3_LC_GDN_RR) echo RR ;;
  EXL3_LC_QSA_SPLIT_STAGES|EXL3_LC_QSA_COMBINE_STAGES|EXL3_LC_QSA_DIV16) echo QT ;;
  EXL3_LC_QSA_FORK) echo QF ;;
  EXL3_LC_GDN_FORK) echo GF ;;
  EXL3_DENSE_V2) echo DG ;;
  EXL3_MOE_COOP_V3|EXL3_MOE_COOP_V3_MAP|EXL3_MOE_COOP_V3_HEAD|EXL3_MOE_COOP_V3_L2EF|EXL3_SHARED_EXPERT_EARLY|EXL3_SHARED_EXPERT_PRIO) echo MF3 ;;
  EXL3_DENSE_ROWS32|EXL3_MOE_COOP_ROWS32|EXL3_SHARED_EXPERT_ROWS32) echo R32 ;;
  *) echo - ;; esac; }
# the served env with every family the flags set removed, then the flags (stack-r3's arm_env; no reliance on docker's
# last-wins)
arm_env(){ local flags=$1 kv f fams=" " out=""
  for kv in $flags; do fams="$fams$(fam "$kv") "; done
  for kv in $EXTRA; do f=$(fam "$kv"); if [ "$f" != - ]; then case "$fams" in *" $f "*) continue ;; esac; fi; out="$out $kv"; done
  echo $out $flags; }
getv(){ local kv; for kv in $2; do [ "${kv%%=*}" = "$1" ] && { echo "${kv#*=}"; return; }; done; echo ""; }
# replace (or add) key $1 with value $2 in env $3; an empty value drops the key
setv(){ local k=$1 v=$2 kv out="" seen=0
  for kv in $3; do if [ "${kv%%=*}" = "$k" ]; then seen=1; [ -n "$v" ] && out="$out $k=$v"; else out="$out $kv"; fi; done
  [ $seen = 0 ] && [ -n "$v" ] && out="$out $k=$v"; echo $out; }
dflags(){ local kv fe=""; for kv in $1; do fe="$fe -e $kv"; done; echo "$fe"; }
# kvset: the sorted key=value entries (values compared, not only keys)
kvset(){ for kv in $1; do echo "$kv"; done | sort | xargs; }

DAILY=$(grep -m1 -oE '^DAILY_IMG=[^ ]+' "$LIVE" | cut -d= -f2 | tr -d "'\"")
EXTRA=$(sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-([^}]*)\}.*/\1/p' "$LIVE")
[ -n "$DAILY" ] && [ -n "$EXTRA" ] || { log "ABORT: could not read DAILY_IMG / EXTRA_ENV from $LIVE"; exit 3; }
grep -qF 'DRAFT_POLICY=${DRAFT_POLICY-[[4, 3], [5, 2], [8, 1]]}' "$LIVE" \
  || log "WARNING: the launcher's default policy is not $POLICY_A; A passes DRAFT_POLICY='$POLICY_A' explicitly anyway"
case "$DAILY" in
  tabbyapi:stack-r2) log "daily is stack-r2: A = the stack-r3 candidate (R716b), not the served daily" ;;
  tabbyapi:stack-r3) for kv in $S3_UNION; do echo "$EXTRA" | tr ' ' '\n' | grep -qxF "$kv" \
                       || { log "ABORT: the promoted stack-r3 daily lacks $kv: set S3_UNION (or A_EXTRA_ENV) to what was promoted"; exit 3; }; done
                     log "daily is stack-r3 with S3_UNION" ;;
  *) log "ABORT: DAILY_IMG is $DAILY (expected tabbyapi:stack-r2 or tabbyapi:stack-r3)"; exit 3 ;;
esac
if [ -n "$A_EXTRA_ENV" ]; then AENV=$(echo $A_EXTRA_ENV); ASRC="A_EXTRA_ENV override"
else AENV=$(arm_env "$S3_UNION"); ASRC="served env minus S3_UNION's families + S3_UNION"; fi
[ "$S3_UNION" = "$S3_UNION_DEFAULT" ] || log "S3_UNION differs from the pre-registered default: $S3_UNION"
for kv in $S3_UNION; do echo "$AENV" | tr ' ' '\n' | grep -qxF "$kv" || { log "ABORT: A env lacks S3_UNION's $kv"; exit 3; }; done
dupk=$(for kv in $AENV; do echo "${kv%%=*}"; done | sort | uniq -d | xargs)
[ -z "$dupk" ] || { log "ABORT: A env sets $dupk twice"; exit 3; }
echo "$AENV" | tr ' ' '\n' | grep -qx 'EXL3_MOE_COOP_V2=1' || { log "ABORT: A env lacks EXL3_MOE_COOP_V2=1"; exit 3; }
echo "$AENV" | tr ' ' '\n' | grep -qE '^EXL3_(DENSE_ROWS32|MOE_COOP_ROWS32|SHARED_EXPERT_ROWS32)=' && { log "ABORT: A env sets a rows32 flag"; exit 3; }
echo "$AENV" | tr ' ' '\n' | grep -qE '^EXL3_NVME_TIER' && { log "ABORT: A env sets EXL3_NVME_TIER"; exit 3; }
if [ -n "$S3_AUDIT" ] && [ -s "$S3_AUDIT" ]; then
  grep -q "DECISION UNION: PASS" "$S3_AUDIT" || [ "${S3_ALLOW_UNACCEPTED:-0}" = 1 ] \
    || { log "ABORT: $S3_AUDIT has no 'DECISION UNION: PASS' (queue after stack-r3 passes, or S3_ALLOW_UNACCEPTED=1)"; exit 3; }
  grep -q "DECISION UNION: PASS" "$S3_AUDIT" || log "S3_ALLOW_UNACCEPTED=1: $S3_AUDIT has no PASS line; reason: ${S3_ACCEPT_NOTE:-<none given>}"
  REC=$(sed -nE 's/.*RECOMMENDED daily: .*EXTRA_ENV="([^"]*)".*/\1/p' "$S3_AUDIT" | tail -1)
  if [ -n "$REC" ]; then
    [ "$(kvset "$REC")" = "$(kvset "$AENV")" ] || { log "ABORT: A env != R716b's RECOMMENDED EXTRA_ENV ($S3_AUDIT): set A_EXTRA_ENV to it"; exit 3; }
    log "A env == R716b's RECOMMENDED EXTRA_ENV ($S3_AUDIT)"
  fi
else log "WARNING: no stack-r3 audit found (S3_AUDIT); A env not cross-checked against R716b's RECOMMENDED line"; fi
AV3=$(getv EXL3_MOE_COOP_V3 "$AENV"); AMAP=$(getv EXL3_MOE_COOP_V3_MAP "$AENV"); DGV=$(getv EXL3_DENSE_V2 "$AENV")
QFV=$(getv EXL3_LC_QSA_FORK "$AENV")
# no A MAP entry may already cover 17-32 rows (first match wins: it would shadow the appended pin)
python3 -c '
import re, sys
for e in [x for x in sys.argv[1].split(",") if x]:
    m = re.fullmatch(r"(\d+)(?:-(\d+))?:([0-3])", e)
    assert m, f"MAP entry {e!r}"
    lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
    assert hi < 17, f"A MAP entry {e!r} covers rows above 16: the 17-32 pin would be shadowed"' "$AMAP" \
  || { log "ABORT: A's EXL3_MOE_COOP_V3_MAP='$AMAP' covers rows above 16"; exit 3; }
BMAP=${AMAP:+$AMAP,}$MAP_ADD
BENV=$(echo $(setv EXL3_MOE_COOP_V3_MAP "$BMAP" "$AENV") $FLAGS)
BNOQF=$(setv EXL3_LC_QSA_FORK "" "$BENV")
BNODR=$(setv EXL3_DENSE_ROWS32 0 "$BENV")
BNOMAP=$(setv EXL3_MOE_COOP_V3_MAP "$AMAP" "$BENV")
dupk=$(for kv in $BENV; do echo "${kv%%=*}"; done | sort | uniq -d | xargs)
[ -z "$dupk" ] || { log "ABORT: B env sets $dupk twice"; exit 3; }
[ "${DGV:-0}" != 0 ] || log "note: A env has EXL3_DENSE_V2 off; the rows32 dense twins engage on EXL3_DENSE_ROWS32 alone"

sudo docker image inspect "$BASE" >/dev/null 2>&1 || { log "ABORT: base image $BASE missing (R716b builds it)"; exit 3; }
blbl(){ sudo docker image inspect "$BASE" --format "{{ index .Config.Labels \"$1\" }}" 2>/dev/null; }
[ "$(blbl local.stack.round)" = r3 ] || { log "ABORT: $BASE is not a stack-r3 image (label local.stack.round '$(blbl local.stack.round)')"; exit 3; }
SINC=$(blbl local.stack.include); SDEFS=$(blbl local.stack.dgv2_defs)
case " $SINC " in *" dg2 "*) ;; *) log "ABORT: $BASE INCLUDE '$SINC' lacks dg2 (EXL3_DENSE_ROWS32 needs densegemm-r2's rows32 twins)"; exit 3 ;; esac
BASE_ID=$(sudo docker image inspect "$BASE" --format '{{.Id}}')
PATCH_SHA=$(sha256sum "$PD/rows32-r4.patch" | cut -c1-64)
log "A ($ASRC): IMG=$BASE ($BASE_ID, INCLUDE '$SINC', defines '$SDEFS') policy '$POLICY_A'"
log "  A env ($(echo $AENV | wc -w) keys): $AENV"
log "B: IMG=$IMG policy '$POLICY_B'; MAP '$AMAP' -> '$BMAP'; flags $FLAGS"
log "  B env ($(echo $BENV | wc -w) keys): $BENV"

gpu_lock
log "lock held; served at entry: $(served_id || echo none); patch sha256 $PATCH_SHA"

# ---- 0. build + landing + SASS identity ----
ilbl(){ sudo docker image inspect "$IMG" --format "{{ index .Config.Labels \"$1\" }}" 2>/dev/null; }
if ! sudo docker image inspect "$IMG" >/dev/null 2>&1 || [ "$(ilbl local.rows32.patch_sha256)" != "$PATCH_SHA" ] \
   || [ "$(ilbl local.rows32.base_id)" != "$BASE_ID" ]; then
  log "building $IMG on $BASE"
  ( cd "$PD" && sudo docker build -f Dockerfile.box --build-arg BASE="$BASE" --build-arg BASE_ID="$BASE_ID" \
      --build-arg STACK_INCLUDE="$SINC" --build-arg DGV2_NVCC_DEFS="$SDEFS" --build-arg PATCH_SHA="$PATCH_SHA" \
      --build-arg MAX_JOBS=4 -t "$IMG" . ) > "$R/build.log" 2>&1 \
    || { log "BUILD FAILED"; tail -40 "$R/build.log" | sed 's/^/  /' | tee -a "$R/audit.log"; finish BUILD-FAILED; exit 3; }
  grep -aE "rows32 r4: |rows32 r4 landed|stack-r3 landed|CPU TESTS|LCGUARD CPU" "$R/build.log" | tail -20 | sed 's/^/  [build] /' | tee -a "$R/audit.log"
else
  log "image $IMG already built from this patch on $BASE_ID"
fi
sudo docker run --rm -v "$R":/results --entrypoint bash "$IMG" -c \
  'for f in base-sass.txt rebuilt-sass.txt sass-identity.txt landing.txt cpu-rows32.txt cpu-lcguard.txt; do cp /opt/rows32-r4/install/$f /results/install-$f; done' \
  || log "WARNING: install artifacts not found in the image"
SID=$(head -1 "$R/install-sass-identity.txt" 2>/dev/null)
log "SASS: ${SID:-no identity report}"
echo "$SID" | grep -q "changed-or-missing 0 added 0" || { log "ABORT: SASS identity not clean (host-only patch)"; finish ABORTED; exit 3; }
sudo docker run --rm --entrypoint python3 "$IMG" /opt/rows32-r4/landing_rows32.py --include "$SINC" > "$R/landed.txt" 2>&1 \
  || { log "ABORT: landing check failed: $(tail -3 "$R/landed.txt" | tr '\n' ' ')"; finish ABORTED; exit 3; }
log "image $IMG: $(tail -1 "$R/landed.txt")"

served_stop; wait_unserved 45
# ---- 1. kernel parity, one card ----
one(){ local name=$1 fe=$2 want=$3; shift 3
  sudo docker run --rm --name r32-p --gpus '"device=0"' --ipc=host -v /srv/qwen5090/models:/models:ro -v "$R":/results \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    $fe -e CUDA_VISIBLE_DEVICES=0 --entrypoint python3 "$IMG" "$@" > "$R/parity-$name.log" 2>&1
  local rc=$?
  log "parity $name rc=$rc: $(grep -aE "$want|PARITY FAIL|CPU TESTS FAIL" "$R/parity-$name.log" | tail -1)"
  grep -aq "$want" "$R/parity-$name.log" && [ $rc = 0 ] \
    || { grep -aE "MISMATCH|FAIL|Error|error" "$R/parity-$name.log" | head -20 | sed 's/^/  /' | tee -a "$R/audit.log"; finish "PARITY-FAIL ($name)"; exit 5; }; }
AF=$(dflags "$AENV")
one rows32 "$AF" "PARITY PASS" /opt/rows32-r4/test_rows32_parity.py --model /models/$MODEL --b-map "$BMAP" --json /results/parity-rows32.json
grep -aE "\[parity\] (4b|4c|<=16|dispatch|stress|join|shared)|PRECONDITION" "$R/parity-rows32.log" | head -60 | sed 's/^/  /' | tee -a "$R/audit.log"
one dense "$AF" "PARITY PASS" /opt/rows32-r4/rows32_extra_parity.py dense --model /models/$MODEL --mode "${DGV:-1}" --json /results/parity-dense.json
python3 - "$R/parity-dense.json" <<'EOF' 2>&1 | tee -a "$R/audit.log"
import json, sys
j = json.load(open(sys.argv[1]))
served = {"gemm/gdn.out_proj", "gemm/attn.o_proj", "gemm/attn.index_qk_proj", "gemm/shared.down",
          "mgemm/gdn.in_proj_qkvz", "mgemm/attn.qkv", "mgemm/shared.gate_up"}
bad = []
for rows in (17, 18, 21, 24, 32):
    for fam in sorted(served):
        cells = [c for c in j["cells"] if c["family"] == fam and c["rows"] == rows]
        if cells and not any(c["engaged"] for c in cells):
            bad.append(f"{fam}@{rows}")
        if not cells:
            bad.append(f"{fam}@{rows} (no cell)")
print("  dense rows32 engagement: " + ("every served family engaged at 17/18/21/24/32 rows" if not bad else "NOT ENGAGED: " + ", ".join(bad)))
sys.exit(1 if bad else 0)
EOF
[ "${PIPESTATUS[0]}" = 0 ] || { log "PARITY-FAIL: dense rows32 cells not engaged (the parity would be vacuous)"; finish "PARITY-FAIL (dense engagement)"; exit 5; }
one latchain "$AF" "PARITY PASS" /opt/rows32-r4/rows32_extra_parity.py latchain --json /results/parity-latchain.json
one hcfast "" "PARITY PASS" /opt/rows32-r4/rows32_extra_parity.py hcfast --model /models/$MODEL --json /results/parity-hcfast.json
one cpuflags "$AF" "CPU TESTS PASS" /opt/rows32-r4/test_rows32_cpu.py --require-flags
sudo docker run --rm --name r32-p --gpus '"device=0"' --ipc=host -v /srv/qwen5090/models:/models:ro \
  -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
  $AF -e CUDA_VISIBLE_DEVICES=0 --entrypoint python3 "$IMG" /opt/rows32-r4/test_rows32_parity.py --model /models/$MODEL \
  --b-map "$BMAP" --bench --skip-synth > "$R/p0.log" 2>&1
log "P0 rc=$? (diagnostic)"; grep -aE "^\[bench\]" "$R/p0.log" | sed 's/^/  /' | tee -a "$R/audit.log"

# ---- P1 runner ----
p1(){ local tag=$1 envs=$2 b=$3 d=$4
  sudo docker run --rm --name r32-p1 --gpus all --ipc=host --shm-size=16g \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v "$HARNESS":/probe/profile_decode_events.py:ro \
    -v "$METER":/probe/events_meter.py:ro -v "$R":/results $(dflags "$envs") \
    --entrypoint python3 "$IMG" /probe/profile_decode_events.py --model /models/$MODEL \
    --out /results/p1-$tag --cache-quant 8,8 --gpu-split 30,30 --max-chunk-size 2048 --tokens 64 \
    --warmup-steps 16 --settle-steps 8 --capture-steps 32 --capture-mode events \
    --contexts 4096 --batch $b --draft $d > "$R/p1-$tag.log" 2>&1
  log "  P1 $tag rc=$?: d$d $(grep -ahoE '[0-9.]+ ms/iterate' "$R"/p1-$tag/ctx4096_b${b}_d${d}/kernels.txt 2>/dev/null | head -1); d0 $(grep -ahoE '[0-9.]+ ms/iterate' "$R"/p1-$tag/ctx4096_b${b}_d0/kernels.txt 2>/dev/null | head -1)"; }
same(){ [ -s "$R/p1-$1/ctx4096_b$3_d$4/sequence-hashes.json" ] && cmp -s "$R/p1-$1/ctx4096_b$3_d$4/sequence-hashes.json" "$R/p1-$2/ctx4096_b$3_d$4/sequence-hashes.json"; }
# R717 review: a run with no hashes (e.g. a harness OOM) logs "DIFFER -> FAIL" below; read it as MISSING (see p1-<tag>.log)

# ---- 2. P1 §16 shapes, OFF (A env) vs U (B env, no policy) ----
for shape in "c4d3 4 3" "c8d1 8 1" "c1d3 1 3"; do set -- $shape; sh=$1 b=$2 d=$3
  for r in $(seq 1 "$ROUNDS"); do
    if [ $(( r % 2 )) = 1 ]; then order="OFF U"; else order="U OFF"; fi
    for a in $order; do if [ $a = U ]; then p1 $sh-U$r "$BENV" $b $d; else p1 $sh-OFF$r "$AENV" $b $d; fi; done
    for dd in $d 0; do same $sh-OFF$r $sh-U$r $b $dd && log "  $sh round $r d$dd: hashes IDENTICAL" || log "  $sh round $r d$dd: hashes DIFFER -> FAIL"; done
  done
done
python3 "$P1S" --results "$R" --arms "OFF U" --rounds "$ROUNDS" --json "$R/p1-verdicts.json" > "$R/p1-summary.txt" 2>&1
tee -a "$R/audit.log" < "$R/p1-summary.txt"
log "note: p1_stack.py's 'P1-VERDICT UNION' asks for a gain; the rows32 flags are inert at <= 16 rows, so rows32_decide.py reads identity + no same-sign regression + the c1 bound from p1-verdicts.json instead"

# ---- 3. P1 rows32 shapes ----
for b in 6 7 8; do
  for r in $(seq 1 "$R32_ROUNDS"); do
    if [ $(( r % 2 )) = 1 ]; then order="SRV R32"; else order="R32 SRV"; fi
    for a in $order; do if [ $a = R32 ]; then p1 c${b}-R32$r "$BENV" $b 2; else p1 c${b}-SRV$r "$AENV" $b 1; fi; done
    if [ $r = 1 ] || [ $r = 3 ]; then
      p1 c${b}-ON1$r "$BENV" $b 1
      same c${b}-SRV$r c${b}-ON1$r $b 1 && log "  c${b}d1 round $r ON1 vs SRV: hashes IDENTICAL" || log "  c${b}d1 round $r ON1 vs SRV: hashes DIFFER -> FAIL"
    fi
    [ $r -gt 1 ] && { same c${b}-R321 c${b}-R32$r $b 2 && log "  c${b}d2 R32 round $r vs round 1: hashes IDENTICAL" || log "  c${b}d2 R32 round $r vs round 1: hashes DIFFER -> FAIL"; }
    if [ $b = 8 ] && [ $r = 1 ]; then
      if [ "$QFV" = 1 ]; then p1 c8-R32noQF1 "$BNOQF" 8 2; else log "  c8d2 R32noQF: QF is off in A, cell skipped"; cp -R "$R/p1-c8-R321" "$R/p1-c8-R32noQF1"; fi
      p1 c8-R32noDR1 "$BNODR" 8 2
      p1 c8-R32noMAP1 "$BNOMAP" 8 2
      for x in R32noQF R32noDR R32noMAP; do same c8-R321 c8-${x}1 8 2 && log "  c8d2 $x vs R32: hashes IDENTICAL" || log "  c8d2 $x vs R32: hashes DIFFER -> FAIL"; done
    fi
  done
done
p1 c8-OFF21 "$AENV" 8 2
LCG=$(cat "$R"/p1-c[678]-R32*.log 2>/dev/null | grep -ac 'densegemm lcguard: device')
LCS=$(cat "$R"/p1-c[678]-R32*.log 2>/dev/null | grep -a 'side call' | head -2 | tr '\n' ' ')
log "lcguard in the R32 runs: $LCG lines; side-slot refusals: ${LCS:-none (every side-branch twin fit the rows32 side region)}"

# ---- 4. served A/B: A1 B1 B2 A2 ----
REC=$R/records.jsonl; GC=$R/greedy-conc.jsonl; G=$R/greedy.jsonl
boot(){ local tag=$1; shift
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { log "[$tag] NO BOOT"; sudo docker logs flashnext > "$R/container-$tag.log" 2>&1; return 1; }
  sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$R/env-$tag.txt"
  log "[$tag] booted $(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'); $(wc -l < "$R/env-$tag.txt") EXL3 keys; $(grep -aoE "policy '[^']*'" "$R/boot-$tag.log" | tail -1); $(grep -aoE 'UP on .*' "$R/boot-$tag.log" | tail -1)"
  grep -q '^EXL3_NVME_TIER=' "$R/env-$tag.txt" && { log "[$tag] ABORT: the NVMe tier is on (NVME_TIER= did not reach the launcher, R709c)"; return 1; }
  return 0; }
gconc(){ python3 "$PD/greedy_conc.py" --url http://127.0.0.1:8022 --tag "$1-c$2-$3" --set "$3" --conc "$2" --out "$GC" \
           > "$R/gc-$1-c$2-$3.log" 2>&1; log "[$1] greedy_conc c$2 set $3 rc=$?"; }
arm(){ local tag=$1; shift
  boot $tag "$@" || { echo "$tag NOBOOT" >> "$R/free.txt"; return 1; }
  case $tag in                                            # greedy probes first after the boot (fresh cache, R709c)
    A1) gconc A1 1 P; gconc A1 8 Q; python3 "$GREEDY" --url http://127.0.0.1:8022 --tag ref --out "$G" > "$R/greedy-ref.log" 2>&1 ;;
    B1) gconc B1 8 P; python3 "$GREEDY" --url http://127.0.0.1:8022 --tag B1 --out "$G" > "$R/greedy-B1.log" 2>&1 ;;
    B2) gconc B2 8 Q ;;
    A2) gconc A2 8 P; gconc A2 1 Q; python3 "$GREEDY" --url http://127.0.0.1:8022 --tag A2 --out "$G" > "$R/greedy-A2.log" 2>&1 ;;
  esac
  for conc in 1 2 3 4 5 6 7 8; do for kind in code prose; do
    python3 "$BENCH" --url "$API" --model "$MODEL" --tag "$tag-c$conc-$kind" --kind $kind --distinct \
      --tokens 1024 --warmup-runs 1 --conc $conc --runs 3 --out "$REC" > "$R/bench-$tag-c$conc-$kind.log" 2>&1
  done; done
  echo "$tag $(vram_free)" >> "$R/free.txt"
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  log "[$tag] done; OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$tag.log"); TORCH_CHECK $(grep -acE 'TORCH_CHECK|c10::Error' "$R/container-$tag.log"); tracebacks $(grep -ac Traceback "$R/container-$tag.log"); lcguard $(grep -ac 'densegemm lcguard: device' "$R/container-$tag.log") (side refusals $(grep -ac 'side call' "$R/container-$tag.log")); free after the ramp $(tail -1 "$R/free.txt")"; }
AARM=(IMG="$BASE" "EXTRA_ENV=$AENV" "DRAFT_POLICY=$POLICY_A")
BARM=(IMG="$IMG" "EXTRA_ENV=$BENV" "DRAFT_POLICY=$POLICY_B")
arm A1 "${AARM[@]}" || { finish NO-BOOT; exit 3; }
arm B1 "${BARM[@]}" || { finish "NO-BOOT (B1)"; exit 3; }
arm B2 "${BARM[@]}" || { finish "NO-BOOT (B2)"; exit 3; }
arm A2 "${AARM[@]}" || { finish "NO-BOOT (A2)"; exit 3; }

# ---- 5. DECISION ----
python3 "$GREEDY" --compare --ref ref --out "$G" > "$R/greedy-compare.txt" 2>&1
grep -aE "^GREEDY |MISSING|DIVERGE" "$R/greedy-compare.txt" | sed 's/^/  [greedy] /' | tee -a "$R/audit.log"
python3 "$PD/greedy_conc.py" --compare --out "$GC" --pair A_P=A2-c8-P:A1-c1-P --pair A_Q=A1-c8-Q:A2-c1-Q \
  --pair B_P=B1-c8-P:A1-c1-P --pair B_Q=B2-c8-Q:A2-c1-Q --verdict A_P,A_Q B_P,B_Q > "$R/gc-compare.txt" 2>&1
sed 's/^/  [greedyc] /' "$R/gc-compare.txt" | tee -a "$R/audit.log"
python3 "$PD/rows32_decide.py" --results "$R" --r32-rounds "$R32_ROUNDS" --policy-a "$POLICY_A" --policy-b "$POLICY_B" \
  > "$R/decision.txt" 2>&1
tee -a "$R/audit.log" < "$R/decision.txt"
if grep -q "^DECISION rows32 r4: ACCEPT" "$R/decision.txt"; then
  log "RECOMMENDED (user-approved conditions met; promotion is the operator's step): IMG=$IMG DRAFT_POLICY='$POLICY_B' EXTRA_ENV=\"$BENV\""
fi
finish DONE
