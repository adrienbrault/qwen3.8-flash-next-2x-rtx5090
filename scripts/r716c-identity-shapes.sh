#!/usr/bin/env bash
# R716c (2026-09-24): stack-r3 identity at every served decode shape (R716b review follow-up, in-process only).
# R716b ended "DECISION UNION: REJECT" on greedy_streams alone. The independent review of R716b found
# that rule unsound (max of 3 A/A divergences vs A1 at a 60-85 % A/A base rate: 64 % false alarms over role
# assignments; it fails the daily against itself 5/12 times) but also a coverage gap: model parity ran only at
# b1d3 / b4d3 / b8d1 (verify rows 4 / 16 / 16), while the served policy [[4,3],[5,2],[8,1]] reaches c2d3 (8 rows),
# c3d3 (12), c5d2 (15), c6d1 (12), c7d1 (14). And served C (stack-r3 image, flags off) was the most divergent boot,
# which asks whether the stack-r3 image with the served env is bitwise the stack-r2 image.
# This unit answers both in-process (no served boots):
#   arms  OFF  = $IMG + served env            (as R716b)
#         U    = $IMG + served env with the union families replaced (R716b's union, exactly)
#         OFF2 = OFF again (A/A)
#         R2   = $BASE (tabbyapi:stack-r2, the daily image) + served env, same parity script mounted from $S3/tests
#   shapes b2d3 b3d3 b5d2 b6d1 b7d1 (new) + b1d3 b4d3 b8d1 (R716b's, R2 arm new there)
# DECISION (pre-registered):
#   per shape: aa = OFF2 vs OFF; U and R2 vs OFF (forward digests + tokens, lc_model_parity --compare)
#   PASS   every shape aa IDENTICAL, U IDENTICAL, R2 IDENTICAL -> stack-r3 is bitwise the daily at every served
#          decode shape; R716b's greedy_streams REJECT is the rule's artefact and the union is promotable on R716b's
#          other bars (P1 PASS, SERVED-VERDICT PASS)
#   FAIL   aa IDENTICAL and U or R2 DIFFERENT at any shape -> hold; the shape and arm name the suspect
#   VOID   aa DIFFERENT at a shape (logit-level test unusable there; token-level only)
# GPU ~20-25 min (32 runs x ~30-40 s incl. load). The daily is restored at the end. Does not promote.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
UNIT=${UNIT:-$(basename "$0" .sh)}
R=${R:-/srv/qwen5090/results/$(date +%F)-$UNIT}; mkdir -p "$R"
S3=${S3:-/srv/qwen5090/patches/exllamav3/stack-r3}
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
IMG=${IMG_S3:-tabbyapi:stack-r3}
R716B=${R716B:-/srv/qwen5090/results/2026-09-24-r716b-stack-r3}
SHAPES=${SHAPES:-"b2d3 b3d3 b5d2 b6d1 b7d1 b1d3 b4d3 b8d1"}
MP_ARGS=${MP_ARGS:-}   # R716c-b2: b2d3 OOMs on cuda:0 at the default --gpu-split 30,30 in every arm (incl. stack-r2); rerun with 28,30
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$UNIT
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ sudo docker rm -f s3-m >/dev/null 2>&1 || true
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== r716c $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$S3/tests/latchain/lc_model_parity.py" "$R716B/env-A1.txt" "$R716B/env-B1.txt"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
BASE=$(grep -m1 -oE '^DAILY_IMG=[^ ]+' "$LIVE" | cut -d= -f2 | tr -d "'\"")
[ "$BASE" = tabbyapi:stack-r2 ] || { log "ABORT: DAILY_IMG is $BASE (expected tabbyapi:stack-r2)"; exit 3; }
for i in "$IMG" "$BASE"; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: image $i missing"; exit 3; }; done
# arm envs = exactly the container envs R716b served (A1 = daily, B1 = union), EXL3_* keys, minus the NVMe tier
envof(){ grep -oE '^EXL3_[A-Z0-9_]+=[^ ]*' "$R716B/env-$1.txt" | grep -v '^EXL3_NVME_TIER' | sort -u; }
A_ENV=$(envof A1 | xargs); B_ENV=$(envof B1 | xargs)
[ -n "$A_ENV" ] && [ -n "$B_ENV" ] && [ "$A_ENV" != "$B_ENV" ] || { log "ABORT: could not read distinct A1/B1 envs from $R716B"; exit 3; }
log "OFF/OFF2/R2 env ($(echo $A_ENV | wc -w) keys): $A_ENV"
log "U env ($(echo $B_ENV | wc -w) keys): $B_ENV"
log "U - OFF: $(comm -13 <(echo $A_ENV | tr ' ' '\n') <(echo $B_ENV | tr ' ' '\n') | xargs)"
fe(){ local kv o=""; for kv in $1; do o="$o -e $kv"; done; echo "$o"; }

gpu_lock
log "lock held; served at entry: $(served_id || echo none); IMG $IMG; BASE $BASE; shapes $SHAPES; extra args ${MP_ARGS:-none}"
served_stop; wait_unserved 45
# wait_unserved only sees the API go away; the first run here OOMed at 3 s (R716b had kernel parity in between)
for i in $(seq 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${used:-99999}" -lt 1024 ] && break; sleep 2; done
log "GPU max used before the first run: ${used} MiB"
[ "${used:-99999}" -lt 1024 ] || { log "ABORT: GPU memory not released"; finish ABORTED; exit 3; }
mp(){ local tag=$1 img=$2 envs=$3 b=$4 d=$5
  sudo docker run --rm --name s3-m --gpus all --ipc=host --shm-size=16g \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v "$R":/results -v "$S3/tests":/r716c-tests:ro $(fe "$envs") \
    --entrypoint python3 "$img" /r716c-tests/latchain/lc_model_parity.py --model /models/$MODEL \
    --batch $b --draft $d $MP_ARGS --out /results/mp-$tag.json > "$R/mp-$tag.log" 2>&1
  local rc=$?
  log "  $tag rc=$rc: $(grep -aE '^lc_model_parity' "$R/mp-$tag.log" | tail -1 | cut -c1-140)"
  [ $rc = 0 ] && [ -s "$R/mp-$tag.json" ]; }
NPASS=0; NFAIL=0; NVOID=0; NERR=0; SUM=""
for s in $SHAPES; do
  b=${s#b}; b=${b%d*}; d=${s#*d}
  log "shape $s (batch $b, draft $d)"
  ok=1
  mp $s-OFF "$IMG" "$A_ENV" $b $d || ok=0
  mp $s-U "$IMG" "$B_ENV" $b $d || ok=0
  mp $s-R2 "$BASE" "$A_ENV" $b $d || ok=0
  mp $s-OFF2 "$IMG" "$A_ENV" $b $d || ok=0
  if [ $ok = 0 ]; then NERR=$((NERR + 1)); SUM="$SUM $s:ERROR"; log "  $s: a run failed (see mp-$s-*.log)"; continue; fi
  python3 "$S3/tests/latchain/lc_model_parity.py" --compare "$R/mp-$s-OFF.json" "$R/mp-$s-OFF2.json" \
    "$R/mp-$s-U.json" "$R/mp-$s-R2.json" > "$R/mp-$s-compare.txt" 2>&1
  sed "s/^/  [$s] /" "$R/mp-$s-compare.txt" | tee -a "$R/audit.log"
  aa=$(grep -c "mp-$s-OFF2.json: .*tokens identical yes" "$R/mp-$s-compare.txt")
  aaf=$(grep -E "mp-$s-OFF2.json: forwards identical" "$R/mp-$s-compare.txt" | grep -oE 'identical [0-9]+/[0-9]+' | head -1)
  v=PASS
  for arm in OFF2 U R2; do
    line=$(grep -E "mp-$s-$arm.json: forwards identical" "$R/mp-$s-compare.txt")
    n=$(echo "$line" | grep -oE 'identical [0-9]+/[0-9]+' | head -1 | cut -d' ' -f2)
    tok=$(echo "$line" | grep -c "tokens identical yes")
    if [ -z "$n" ] || [ "${n%/*}" != "${n#*/}" ] || [ "$tok" != 1 ]; then
      if [ $arm = OFF2 ]; then v=VOID; break; else v=FAIL; log "  $s: $arm differs from OFF ($line)"; fi
    fi
  done
  case $v in PASS) NPASS=$((NPASS + 1)) ;; FAIL) NFAIL=$((NFAIL + 1)) ;; VOID) NVOID=$((NVOID + 1)) ;; esac
  SUM="$SUM $s:$v"
  log "  $s: $v (aa $aaf, tokens $aa)"
done
log "SUMMARY:$SUM"
if [ $NFAIL -gt 0 ]; then log "DECISION: FAIL ($NFAIL shape(s) where the A/A is identical and U or R2 differs)"
elif [ $NERR -gt 0 ] || [ $NVOID -gt 0 ]; then log "DECISION: VOID (errors $NERR, A/A-nondeterministic shapes $NVOID; the rest PASS $NPASS)"
else log "DECISION: PASS (stack-r3 union and the stack-r3 image with the served env are bitwise the stack-r2 daily at all $NPASS shapes)"; fi
finish DONE
