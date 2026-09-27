#!/usr/bin/env bash
# R784 (2026-09-27): rebase-dev r3 fit + speed, the first half of the promotion chain (the plan's prose:
#   docs/PROMOTION.md, "A rebase onto a new upstream engine").
#   P = tabbyapi:rebase-dev-r3: the served ExLlamaV3 stack (41 launcher keys) ported onto upstream exllamav3 dev 5783a93
#       (v1.5.2; r2 + upstream's four commits, docker/overlays/rebase-dev-r3/impl-status.md) + tokcount-r1, on the base
#       tabbyapi:stack-r3-rows32-tokcount-loopthink4 (the served TabbyAPI + loop-think r4, docker/overlays/loop-think-r4:
#       adds the collector-only long-period detector LoopDetector(3L, L), L = 1.25W = 1,000 at W = 800). Served env +
#       EXL3_GR_MIX_TILED=1 (42 keys; upstream's tiled HC prefill, parsed != "0", explicit so the launcher records it).
#   S = the served image (live DAILY_IMG) with the live env (41 keys) at the live pool 983,040.
# Derived from r741-rebase-dev-r2.sh (run 2; R741 measured the r2 port and is not in this repository), with the plan's changes:
#   - pool search on P: 917,504 -> 901,120 -> 884,736 (the floor set for this promotion: 983,040 - ~100k), splits
#     "30, 30" -> "29.5, 30" -> "29, 30" at each pool while only cuda:0 is short; one step up (933,888) when 917,504 fits;
#     <= 5 judged boots + the cold-autotune re-boot. 983,040..933,888 are skipped on R741 run 2's evidence (R786 re-opens).
#   - every search boot: after the 12k layout probe, an OOM / Traceback line in the container log (or a container that died
#     under the probe) = no-fit (R741's miss: a boot that fit at UP and OOMed at its first long prefill).
#   - cold prefill: 3 salted unique prompts per ctx (30k, 120k) after the layout probe, median (R754: the first sample reads
#     17-19 % slow). Salts are paired across boots (every boot is a fresh container, tier off) and distinct within a boot.
#   - ABBA S1 P1 P2 S2 (P2 always; R741's single P boot was its weak point). S2 is skipped when P never fit.
#   - PNT (tiled OFF, EXL3_GR_MIX_TILED=0) only when the tiled-ON search finds no pool >= the floor: descent from 966,656,
#     <= 3 boots, prefill-only measure -> the CANDIDATE-NOTILED rule (route 2).
#   - fidelity record: F5 lp_margin.py (corpus, 40 chunks x 64 tokens, ctx 0 and 60,000) on S1 and P1, compared --ref S1;
#     F6 exl3_fidelity_run.py --chunks 2 smoke in both images, then OFF (served) / ON (r3) --chunks 200 --tokens 2048 and
#     probes/fidelity.py compare. Both report-only (first server run of lp_margin; F6 bars UNCALIBRATED, R756 floor).
#   - writes $R/promote.env for R785 (DECISION FOUND SPLIT S1_UP S1_POST120K P_IMG_ID BASE_ID P_TREE_SHA ...).
#   - early prints in audit.log (prefix "EARLY"): the pool verdict (~40 min) and the P1/S1 prefill ratio (~52 min).
# DECISION: docker/overlays/rebase-dev-r3/r784_decide.py (pre-registered; the plan's text is the prose):
#   CANDIDATE | CANDIDATE-NOTILED | NOT-A-CANDIDATE (<clause>) | INCOMPLETE (missing ... | VOID: ...).
# IMAGE BUILD: pre-lock CPU work (plan). The unit waits until no unit holds the GPU-exclusive lock, holds it on fd 8 for the
#   build only (never a build beside a measurement; the daily keeps serving), releases it, THEN registers in the GPU
#   queue and queues on the lock (fd 9). Registering after the build keeps a finishing unit from skipping the daily restore
#   for the whole build. Label-gated (base id + tree sha): a re-run reuses the image.
# UNIT HYGIENE (R783 review): preflight before the lock and again after it; the rollback trap is set right after the lock;
#   any abort after the lock ensures the daily serves from $LIVE on :8022 (nothing else restores :8022;
#   daily-restore-retry.sh restores the 27B) and repoints Hermes to :8022. A normal finish restores the daily only when no
#   unit is queued (R785 queued -> it skips: lib/serve-ctl.sh finish_restore). Port :8029, NVME_TIER= on every boot.
# GPU ~95-120 min (TIMEBOX_MIN, default 150), build ~40-60 min CPU before the lock when the image is missing or stale.
# Install (operator, .new + mv for every file that a queued unit might read):
#   docker/overlays/rebase-dev-r3/ (after prepare-tree.sh) -> /srv/qwen5090/overlay-src/rebase-dev-r3/   (Dockerfile.box, out/,
#     landing, r784_decide.py)
#   lp_margin.py, exl3_fidelity_run.py (F5 / F6, report-only) and fidelity.py, fn_greedy.py are not in this repository
#   bench/chat_greedy.py, bench/mp_decode.py, bench/probe.py (fn_bench) -> /srv/qwen5090/probes/ ; this file -> /srv/qwen5090/r784-rebase-dev-r3.sh
# Launch (TREE_SHA_P = the r3 tree's sha, 150497b44e6f49b06b8a13f2568e68fc41841bc28964928509e181b633285501, impl-status.md):
#   sudo systemd-run --unit=r784-rebase-dev-r3 --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#     -p Environment=HOME=$HOME -E TREE_SHA_P=<sha> /usr/bin/bash /srv/qwen5090/r784-rebase-dev-r3.sh
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
UNIT=${UNIT:-r784-rebase-dev-r3}
D=/srv/qwen5090
R=${R:-$D/results/$(date +%F)-$UNIT-$(date +%H%M)}
[ -e "$R/audit.log" ] && R=$R-$(date +%S)              # never append to another invocation's files
mkdir -p "$R"
LIVE=$D/launch-flashnext.sh
CONF=$D/flashnext-config.yml
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
PORT=8029
BASEURL=http://127.0.0.1:$PORT
API=$BASEURL/v1
SRC=${SRC:-$D/overlay-src/rebase-dev-r3}
PR=$D/probes
MP=$PR/mp_decode.py
FB=$PR/fn_bench.py
GREEDY=$PR/fn_greedy.py
CHAT=$PR/chat_greedy.py
FIDPY=$PR/fidelity.py
HT=${HT:-$D/patches/exllamav3/hctiled-r2/tools}
LPM=$HT/lp_margin.py
FIDRUN=$HT/exl3_fidelity_run.py
CORPUS=$D/results/2026-08-23-fidelity/corpus.jsonl
DECIDE=$SRC/r784_decide.py
HSET=$D/hermes-set-model.sh
EXPECT_DAILY=${EXPECT_DAILY:-tabbyapi:stack-r3-rows32-tokcount-loopthink3}   # the live daily (R783)
BUILD_BASE=${BUILD_BASE:-tabbyapi:stack-r3-rows32-tokcount-loopthink4}       # served TabbyAPI + loop-think r4
PIMG=${PIMG:-tabbyapi:rebase-dev-r3}
TREE_SHA_P=${TREE_SHA_P:?set TREE_SHA_P to the sha of out/rebase-dev-r3/exllamav3 (impl-status.md)}
# The r3 port's image contract mirrors r2's (landing_r2.py --tree r2 -> "rebase-dev-r2 landed:"); overridable if the r3
# port names them differently.
LANDING_CMD=${LANDING_CMD:-/opt/rebase-dev-r3/landing_r3.py --tree r3}
LANDING_OK=${LANDING_OK:-rebase-dev-r3 landed:}
TUNE_P=$D/.exl3cache-rebase-dev-r3
TILED_ON=EXL3_GR_MIX_TILED=1
TILED_OFF=EXL3_GR_MIX_TILED=0
STEP=16384
FLOOR=884736                                    # the floor: 983,040 - ~100k on the 16,384 grid
REF_UP=${REF_UP:-1125/1573}                     # S1's expected UP free (R783 G1); VOID outside +- 32
POOLS_P=(917504 901120 884736)
POOLS_PNT=(966656 950272 933888 917504 901120 884736)
SPLITS_P=("30, 30" "29.5, 30" "29, 30")
SPLIT0="30, 30"
PCAP=${PCAP:-7}; NCAP=${NCAP:-3}   # 7: 3 pools x the split ladder must reach the 884,736 floor (pre-launch review F3)
TIMEBOX_MIN=${TIMEBOX_MIN:-150}
ARMS=${ARMS:-"S P P S PNT"}
FID=${FID:-1}                                   # F6 corpus fidelity at the end (report-only)
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
early(){ log "EARLY $*"; echo "$(date -Is) $*" >> "$R/early.txt"; }
ilbl(){ sudo docker image inspect "$1" --format "{{ index .Config.Labels \"$2\" }}" 2>/dev/null; }
iid(){ sudo docker image inspect "$1" --format '{{.Id}}' 2>/dev/null; }
treesha(){ ( cd "$1" && find exllamav3 -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64 ); }
envline(){ sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$1" | head -1; }
dflags(){ local kv o=""; for kv in $1; do o="$o -e $kv"; done; echo "$o"; }
gpcoff(){ timeout 30 sudo python3 -c 'import pynvml as N;N.nvmlInit();print(*[N.nvmlDeviceGetGpcClkVfOffset(N.nvmlDeviceGetHandleByIndex(i)) for i in range(N.nvmlDeviceGetCount())])' 2>/dev/null || echo "?"; }
memoff(){ timeout 30 sudo python3 -c 'import pynvml as N;N.nvmlInit();print(*[N.nvmlDeviceGetMemClkVfOffset(N.nvmlDeviceGetHandleByIndex(i)) for i in range(N.nvmlDeviceGetCount())])' 2>/dev/null || echo "?"; }
pwrlim(){ nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | awk '{printf "%s%.0f", (NR>1?" ":""), $1}'; }
WANT_PWR=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | awk '{printf "%s%.0f", (NR>1?" ":""), $1}')
cp "$0" "$R/" 2>/dev/null
trap 'log "signal before the lock: nothing mutated"; exit 4' TERM INT HUP

# ---------------- preflight (run before the lock and again after it) ----------------
LIVE_MD5=
preflight(){ local f why=
  for f in "$LIVE" "$MP" "$FB" "$GREEDY" "$CHAT" "$FIDPY" "$LPM" "$FIDRUN" "$DECIDE" "$HSET" "$SRC/Dockerfile.box" \
           "$SRC/out/rebase-dev-r3/exllamav3/__init__.py" "$D/lib/gpu-queue.sh" "$D/lib/serve-ctl.sh" "$D/lib/gateway-drain.sh"; do
    [ -e "$f" ] || why="$why missing $f;"; done
  [ -e "$CORPUS" ] || log "WARNING: $CORPUS missing: F5 / F6 will be skipped (report-only)"
  [ -n "$why" ] && { log "preflight:$why"; return 1; }
  [ "$(treesha "$SRC/out/rebase-dev-r3")" = "$TREE_SHA_P" ] || { log "preflight: r3 tree sha $(treesha "$SRC/out/rebase-dev-r3") != TREE_SHA_P $TREE_SHA_P"; return 1; }
  grep -q -- '--concs' "$MP" || { log "preflight: $MP lacks --concs"; return 1; }
  python3 "$DECIDE" --selftest > "$R/decide-selftest.txt" 2>&1 || { log "preflight: r784_decide.py selftest failed"; return 1; }
  for f in "$LPM" "$FIDRUN" "$CHAT"; do PYTHONPYCACHEPREFIX=/tmp/r784-pyc python3 -m py_compile "$f" || { log "preflight: $f does not compile"; return 1; }; done
  grep -qE '^IMG=\$\{IMG:-\$DAILY_IMG\}' "$LIVE" || { log "preflight: live launcher takes no IMG override"; return 1; }
  grep -qE '^PORT=\$\{PORT:-8022\}' "$LIVE" || { log "preflight: live launcher takes no PORT override"; return 1; }
  grep -qE '^GPU_SPLIT=\$\{GPU_SPLIT:-30, 30\}' "$LIVE" || { log "preflight: live launcher has no GPU_SPLIT knob"; return 1; }
  grep -qE '^TUNEDIR=\$\{TUNEDIR:-/srv/qwen5090/.exl3cache\}' "$LIVE" || { log "preflight: live launcher has no TUNEDIR knob (P needs its own kernel cache)"; return 1; }
  grep -q '^CKPT_NAME=${CKPT_NAME:-qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab}' "$LIVE" || { log "preflight: live launcher CKPT_NAME is not the r0b0tlab 2.50 daily (a finetune checkpoint is not promoted)"; return 1; }
  DAILY_IMG=$(grep -oE '^DAILY_IMG=\S+' "$LIVE" | cut -d= -f2)
  [ "$DAILY_IMG" = "$EXPECT_DAILY" ] || { log "preflight: live DAILY_IMG=$DAILY_IMG, expected $EXPECT_DAILY"; return 1; }
  LENV=$(envline "$LIVE"); LN=$(echo $LENV | wc -w | tr -dc 0-9)
  [ "$LN" = 41 ] || { log "preflight: live EXTRA_ENV has $LN keys, the port was audited against 41"; return 1; }
  echo "$LENV" | tr ' ' '\n' | grep -q '^EXL3_GR_MIX_TILED=' && { log "preflight: live EXTRA_ENV already sets EXL3_GR_MIX_TILED"; return 1; }
  LPOOL=$(sed -n 's/^CACHE=\${CACHE:-\([0-9]*\)}.*/\1/p' "$LIVE")
  [ "$LPOOL" = 983040 ] || { log "preflight: live pool '$LPOOL' != 983040 (the plan's S reference)"; return 1; }
  if [ -z "$LIVE_MD5" ]; then LIVE_MD5=$(md5sum < "$LIVE" | cut -c1-32)
  else [ "$(md5sum < "$LIVE" | cut -c1-32)" = "$LIVE_MD5" ] || { log "preflight: live launcher changed since the unit started ($LIVE_MD5)"; return 1; }; fi
  iid "$DAILY_IMG" >/dev/null || { log "preflight: $DAILY_IMG missing"; return 1; }
  BASE_ID=$(iid "$BUILD_BASE") || { log "preflight: build base $BUILD_BASE missing (loop-think r4 not built)"; return 1; }
  [ "$(ilbl "$BUILD_BASE" local.loopthink.round)" = r4 ] || { log "preflight: $BUILD_BASE is not loop-think r4 (label local.loopthink.round '$(ilbl "$BUILD_BASE" local.loopthink.round)')"; return 1; }
  command -v flock >/dev/null || { log "preflight: flock missing"; return 1; }
  return 0; }
preflight || { log "ABORT (pre-lock preflight): nothing touched"; exit 3; }
log "start: daily $DAILY_IMG (live md5 $LIVE_MD5), pool $LPOOL, $LN keys; P $PIMG on $BUILD_BASE ($BASE_ID); tree $TREE_SHA_P; arms '$ARMS'; timebox ${TIMEBOX_MIN} min; stock power '$WANT_PWR'; results $R"
df -h / | tail -1 | tee -a "$R/audit.log"
LIVE_BAK=$R/launch-flashnext.sh.at-start; cp -p "$LIVE" "$LIVE_BAK"

# ---------------- image build: pre-lock CPU work, never beside a measurement ----------------
build_needed(){ ! iid "$PIMG" >/dev/null || [ "$(ilbl "$PIMG" local.rebase.base_id)" != "$BASE_ID" ] \
  || [ "$(ilbl "$PIMG" local.rebase.tree_sha256)" != "$TREE_SHA_P" ]; }
if build_needed; then
  exec 8>$D/gpu-exclusive.lock
  flock -n 8 || { log "build: waiting until no unit holds the GPU-exclusive lock (the build runs only while nothing measures)"; flock 8; }
  log "build: lock free, held on fd 8 for the build only; building $PIMG on $BUILD_BASE (full extension rebuild, MAX_JOBS 4, nice 19); the daily keeps serving"
  DEFS=$(ilbl "$BUILD_BASE" local.stack.dgv2_defs)
  ( cd "$SRC" && sudo nice -n 19 docker build -f Dockerfile.box --build-arg BASE="$BUILD_BASE" --build-arg BASE_ID="$BASE_ID" \
      --build-arg DGV2_NVCC_DEFS="$DEFS" --build-arg TREE_SHA="$TREE_SHA_P" --build-arg MAX_JOBS=4 -t "$PIMG" . ) > "$R/build-P.log" 2>&1
  brc=$?
  flock -u 8; exec 8>&-
  [ "$brc" = 0 ] || { log "ABORT: $PIMG build failed (rc $brc; nothing else touched)"; grep -aE 'FAILED|Error|error:|Assertion' "$R/build-P.log" | tail -8 | cut -c1-240 | sed 's/^/  /' | tee -a "$R/audit.log"; exit 3; }
  build_needed && { log "ABORT: $PIMG built but its labels do not carry base id $BASE_ID / tree $TREE_SHA_P (Dockerfile.box label contract)"; exit 3; }
  log "build: done in $(grep -c . "$R/build-P.log") log lines; lock released"
else log "image $PIMG current (base id + tree sha labels match)"; fi
P_IMG_ID=$(iid "$PIMG")
[ "$(ilbl "$PIMG" local.loopthink.round)" = r4 ] || { log "ABORT: $PIMG does not carry loop-think r4 (label inherited from $BUILD_BASE)"; exit 3; }
[ "$(ilbl "$PIMG" local.rebase.round)" = rebase-dev-r3 ] || { log "ABORT: $PIMG label local.rebase.round is '$(ilbl "$PIMG" local.rebase.round)', not rebase-dev-r3"; exit 3; }

# ---------------- queue + lock ----------------
export GPU_QUEUE_NAME=$UNIT
. $D/lib/gpu-queue.sh
. $D/lib/serve-ctl.sh
. $D/lib/gateway-drain.sh
SCTL_LOG="$R/audit.log"
SCTL_API=$API
FINISHED=0
BOOTED=0
DECIDED=
hermes_daily(){ [ "$(SCTL_API=http://127.0.0.1:8022/v1 served_id)" = "$MODEL" ] || { log "Hermes: :8022 does not serve $MODEL (a queued unit holds the GPUs): not repointed"; return 0; }
  PORT=8022 MODEL_ID=$MODEL bash "$HSET" >> "$R/audit.log" 2>&1 && log "Hermes on :8022" || log "WARN: Hermes not repointed"; }
# promote.env: R785's input. Written on every path after the lock, so R785 aborts on data, never on a missing file.
write_promote(){ local tmp=$R/promote.env.tmp
  { echo "DECISION=$1"
    echo "FOUND=${PPOOL:-}"
    echo "SPLIT=${PSPLIT:+${PSPLIT// /}}"
    echo "S1_UP=${S1_UP:-}"
    echo "S1_POST120K=${S1_POST:-}"
    echo "P1_UP=${P1_UP:-}"
    echo "P1_POST120K=${P1_POST:-}"
    echo "PNT_POOL=${NPOOL:-}"
    echo "P_IMG=$PIMG"
    echo "P_IMG_ID=${P_IMG_ID:-}"
    echo "BASE_IMG=$BUILD_BASE"
    echo "BASE_ID=${BASE_ID:-}"
    echo "P_TREE_SHA=$TREE_SHA_P"
    echo "DAILY_IMG=${DAILY_IMG:-}"
    echo "LIVE_MD5=${LIVE_MD5:-}"
    echo "R784_DIR=$R"
    echo "R784_FINISHED=1"
    echo "WRITTEN=$(date -Is)"; } > "$tmp" && mv "$tmp" "$R/promote.env"
  log "promote.env: DECISION=$1 FOUND=${PPOOL:-} SPLIT=${PSPLIT:-}"; }
# finish HOW [force]: force (every abort after the lock) = the daily must serve from $LIVE on :8022 whatever is queued;
# default = finish_restore (skips the restore while another unit, e.g. R785, is queued).
finish(){ [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  sudo docker rm -f r784-fid >/dev/null 2>&1 || true
  [ -n "$(served_id)" ] && sudo docker logs flashnext > "$R/container-final.log" 2>&1
  [ "$BOOTED" = 1 ] && served_stop       # only once this unit owns the GPUs (never the daily while queued)
  SCTL_API=http://127.0.0.1:8022/v1
  if [ "${2:-}" = force ]; then
    if [ "$BOOTED" = 0 ] && [ "$(served_id)" = "$MODEL" ]; then log "daily already serving on :8022 (never stopped)"
    else served_stop; wait_unserved 45 || true
      env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" > "$R/boot-daily-restore.log" 2>&1
      wait_served_id "$MODEL" 240 10 && log "daily restored from $LIVE: $(served_id)" || log "DAILY RESTORE FAILED: $(tail -2 "$R/boot-daily-restore.log" | tr '\n' ' ' | cut -c1-200)"
    fi
  else finish_restore "$LIVE"; fi
  hermes_daily
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  log "offsets core $(gpcoff) / memory $(memoff); daily $(served_id || echo none)"
  sudo chown -R "$(stat -c %U $D/results)" "$R" 2>/dev/null || true
  log "=== $UNIT $1 ==="; }
abort_post(){ log "ABORT (after the lock): $1"; write_promote "ABORTED"; finish "ABORTED ($1)" force; exit "${2:-3}"; }

gpu_lock
trap 'log "signal"; abort_post "signal" 4' TERM INT HUP
gateway_drain   # no-op in effect (:8029 is not routed and the daily goes down); held so nothing routes to :8022 mid-unit
log "lock held; daily at entry: $(SCTL_API=http://127.0.0.1:8022/v1 served_id || echo none); offsets core $(gpcoff) / memory $(memoff); power $(pwrlim)"
preflight || abort_post "post-lock preflight"
[ "$(iid "$PIMG")" = "$P_IMG_ID" ] || abort_post "$PIMG changed while queued"
build_needed && abort_post "$PIMG labels no longer match base id / tree sha"
# landing (CPU, seconds): under the lock, so it never runs beside another unit's measurement
LP=$(sudo docker run --rm --network none --entrypoint python3 "$PIMG" $LANDING_CMD 2>&1 | tee "$R/landing-P.txt" | tail -1)
case "$LP" in "$LANDING_OK"*) log "P landing: ${LP:0:200}";; *) abort_post "landing failed on $PIMG: ${LP:0:200}";; esac
sudo mkdir -p "$TUNE_P"
T0=$(date +%s); END=$(( T0 + TIMEBOX_MIN * 60 ))

# ---------------- helpers (r741 run 2 + the plan's fixes) ----------------
printf 'tag\tarm\timage\tpool\ttp\tenv_n\tup_free\tpost_free\theadroom\tpower\tgpc\tmem\tgen\toom\tvoid\tmeasured\tsplit\tlayout\ttb\trestarts\tmp_n\tmp_fail\n' > "$R/boots.tsv"
S_F0=; S_F1=; CUR=
B_TAG=; B_ARM=; B_IMG=; B_POOL=; B_ENVN=; B_UP=; B_PWR=; B_GPC=; B_MEM=; B_GEN=; B_VOID=; B_HEAD=; B_SPLIT=; B_LAYOUT=; B_SHORT=; B_FAIL=
PENDING=0
# row [post] [oom] [measured] [tb] [restarts] [mp_n] [mp_fail]
row(){ PENDING=0; printf '%s\t%s\t%s\t%s\tfalse\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$B_TAG" "$B_ARM" "$B_IMG" \
  "$B_POOL" "$B_ENVN" "$B_UP" "${1:--}" "$B_HEAD" "$B_PWR" "$B_GPC" "$B_MEM" "$B_GEN" "${2:--}" "${B_VOID:--}" "${3:-0}" \
  "${B_SPLIT:--}" "${B_LAYOUT:--}" "${4:--}" "${5:--}" "${6:--}" "${7:--}" >> "$R/boots.tsv"; }
# boot TAG ARM IMG POOL [VAR=VAL ...]: one launcher boot on :8029. rc 0 = serving, sane, right image and env (B_* filled).
boot(){ local t=$1 arm=$2 img=$3 pool=$4 got up lp lrc rc kv want=; shift 4
  [ "$PENDING" = 1 ] && row
  B_TAG=$t; B_ARM=$arm; B_IMG=$img; B_POOL=$pool; B_ENVN=; B_UP=; B_GEN=NO_BOOT; B_VOID=; B_HEAD=-; CUR=
  B_SPLIT=${SPLIT0// /}; B_LAYOUT=-; B_SHORT=1; B_FAIL=func
  for kv in "$@"; do case "$kv" in GPU_SPLIT=*) B_SPLIT=${kv#GPU_SPLIT=}; B_SPLIT=${B_SPLIT// /};; EXTRA_ENV=*) want=${kv#EXTRA_ENV=};; esac; done
  served_stop; wait_unserved 45 || { log "[$t] :$PORT still answering after stop"; row; return 1; }
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin PORT=$PORT NVME_TIER= IMG="$img" \
      CACHE="$pool" "$@" bash "$LIVE" > "$R/boot-$t.log" 2>&1 &
  lp=$!
  # restart-loop breaker (r741 run 2): one container restart = the load failed
  while kill -0 "$lp" 2>/dev/null; do
    rc=$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null)
    if [ "${rc:-0}" -gt 0 ] 2>/dev/null; then
      sudo docker update --restart=no flashnext >/dev/null 2>&1; sudo docker stop -t 10 flashnext >/dev/null 2>&1
      log "[$t] the container restarted (failed load): restart loop stopped"; break; fi
    sleep 3
  done
  wait "$lp"; lrc=$?
  [ "$lrc" = 0 ] && wait_served_id "$MODEL" 240 10
  if [ -z "$(served_id)" ]; then
    sudo docker logs flashnext > "$R/container-$t-noboot.log" 2>&1
    grep -aqE 'Insufficient VRAM|out of memory|OutOfMemory' "$R/boot-$t.log" "$R/container-$t-noboot.log" && B_FAIL=vram
    log "[$t] NO BOOT @ $pool [$B_SPLIT] ($B_FAIL): $(grep -ahE 'Insufficient VRAM|out of memory|OutOfMemory|Error|ABORT' "$R/boot-$t.log" "$R/container-$t-noboot.log" | tail -1 | cut -c1-220)"
    served_stop; row; return 1; fi
  B_GEN=$(generation_state "$MODEL")
  got=$(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}')
  B_ENVN=$(grep -aoE 'env keys \([0-9]+\)' "$R/boot-$t.log" | tail -1 | tr -dc '0-9')
  up=$(grep -aoE 'VRAM free MiB [0-9]+/[0-9]+' "$R/boot-$t.log" | tail -1 | grep -oE '[0-9]+/[0-9]+')
  B_UP=${up:-?}; B_PWR=$(pwrlim); B_GPC=$(gpcoff); B_MEM=$(memoff)
  [ "$B_PWR" = "$WANT_PWR" ] || B_VOID="power '$B_PWR' != stock '$WANT_PWR'"
  [ "$B_GPC" = "0 0" ] || B_VOID="${B_VOID:+$B_VOID; }core offsets '$B_GPC' != '0 0'"
  [ "$B_MEM" = "4500 4500" ] || B_VOID="${B_VOID:+$B_VOID; }memory offsets '$B_MEM' != '4500 4500'"
  sudo cp "$CONF" "$R/config-$t.yml" 2>/dev/null
  sudo docker exec flashnext env 2>/dev/null | grep '^EXL3_' | sort > "$R/env-$t.txt"
  grep -q '^EXL3_NVME_TIER=' "$R/env-$t.txt" && { log "[$t] NVMe tier is on: not this unit's config"; row; return 1; }
  [ "$got" = "$img" ] || { log "[$t] serves $got, not $img"; row; return 1; }
  [ "$B_ENVN" = "$(echo $want | wc -w | tr -dc 0-9)" ] || { log "[$t] env keys $B_ENVN != requested $(echo $want | wc -w | tr -dc 0-9)"; row; return 1; }
  for kv in $want; do grep -qxF "$kv" "$R/env-$t.txt" || { log "[$t] container env lacks $kv"; row; return 1; }; done
  [ "$(cards_loaded)" = OK ] || { log "[$t] a card is empty: $(cards_loaded)"; row; return 1; }
  sudo docker logs flashnext > "$R/container-$t-up.log" 2>&1
  tr -s ' \n' ' ' < "$R/container-$t-up.log" | grep -aoE 'Loading model [^ ]+ \((tensor parallel|manual GPU split|autosplit)\)' | tail -1 > "$R/split-$t.txt"
  log "[$t] UP @ $pool [$B_SPLIT]: $got, env keys $B_ENVN, free $B_UP MiB, $B_GEN, power $B_PWR, offsets $B_GPC / $B_MEM${B_VOID:+, VOID: $B_VOID}"
  [ "$B_GEN" = GEN_SANE ] || { row; return 1; }
  PENDING=1; return 0; }
# layout_probe TAG: one salted 12k prefill (r741): prints the LS prefill pipeline's stage layout; B_LAYOUT = card:first-last.
layout_probe(){ local t=$1
  grep -q '^EXL3_LS_PREFILL_PIPELINE=1' "$R/env-$t.txt" 2>/dev/null || { B_LAYOUT=n/a; return 0; }
  python3 "$FB" --url "$API" --model "$MODEL" --tag "layout-$t" --kind prose --tokens 4 --conc 1 --runs 1 --ctx 12000 --unique \
    --salt $(( (SALT0 + RANDOM) % 100000 )) --out "$R/layout-probe.jsonl" > "$R/layout-$t.log" 2>&1
  sudo docker logs flashnext > "$R/container-$t-up.log" 2>&1
  B_LAYOUT=$(grep -a 'LS prefill pipeline' "$R/container-$t-up.log" | tail -1 | python3 -c '
import ast, sys
s = sys.stdin.read(); i = s.find("{")
d = ast.literal_eval(s[i:].strip()) if i >= 0 else None
out = []
for dev, stage in zip(d["devices"], d["stages"]):
    ls = [int(k.split(".")[-1]) for k, _, _ in stage if k.split(".")[-2:-1] == ["layers"]]
    out.append("%s:%s" % (dev.replace("cuda:", ""), "%d-%d" % (min(ls), max(ls)) if ls else "-"))
print(",".join(out))' 2>/dev/null)
  B_LAYOUT=${B_LAYOUT:-?}
  log "[$t] layer placement (card:first-last layer): $B_LAYOUT"; }
# oom_after_probe TAG: rc 0 = the container survived the layout probe with no OOM / Traceback line (R741's miss).
# B_SHORT = 0 when every OOM line names GPU 0 only (a lower cuda:0 budget may fix it).
oom_after_probe(){ local t=$1 n
  n=$(grep -acE 'OutOfMemoryError|out of memory|Traceback' "$R/container-$t-up.log")
  if [ -z "$(served_id)" ]; then B_HEAD="DIED(probe)"; B_GEN=DIED; B_FAIL=vram; B_SHORT=1; return 1; fi
  [ "$n" = 0 ] && return 0
  B_HEAD="OOM($n)"; B_FAIL=vram; B_SHORT=1
  grep -aE 'out of memory' "$R/container-$t-up.log" | grep -q 'GPU 0' && ! grep -aE 'out of memory' "$R/container-$t-up.log" | grep -q 'GPU 1' && B_SHORT=0
  return 1; }
# headroom: per-card free at UP >= S1's - 32 MiB; B_SHORT = 0 when only cuda:0 is short.
headroom(){ local f0=${B_UP%/*} f1=${B_UP#*/} ok0=0 ok1=0
  B_SHORT=1
  [ -n "$S_F0" ] || { B_HEAD=REF; return 0; }
  [ "$f0" -ge $(( S_F0 - 32 )) ] 2>/dev/null && ok0=1
  [ "$f1" -ge $(( S_F1 - 32 )) ] 2>/dev/null && ok1=1
  if [ "$ok0" = 1 ] && [ "$ok1" = 1 ]; then B_HEAD=OK; B_SHORT=; return 0; fi
  [ "$ok1" = 1 ] && B_SHORT=0
  B_HEAD="BELOW($f0/$f1 vs $S_F0/$S_F1)"; return 1; }
# fits TAG ARM IMG POOL [VAR=VAL ...]: boot (+ one warm re-boot when the kernel cache was cold) + layout probe + OOM grep
# + headroom. A fitting boot stays up (CUR = TAG, PENDING = 1).
fits(){ local t=$1 w; shift; boot "$t" "$@" || return 1
  w=$(grep -aoE 'warmup [0-9.]+s' "$R/boot-$t.log" | tail -1 | tr -dc '0-9.')
  if awk -v w="${w:-0}" 'BEGIN { exit !(w > 5) }'; then
    B_HEAD="COLD(warmup ${w}s)"; log "[$t] cold kernel cache (warmup ${w}s, free $B_UP): not judged, re-booting once"; row
    t="$t+w"; boot "$t" "$@" || return 1
  fi
  layout_probe "$t"
  oom_after_probe "$t" || { log "[$t] NO FIT: $B_HEAD after the 12k layout probe ($(grep -aE 'out of memory|Traceback' "$R/container-$t-up.log" | tail -1 | cut -c1-160))"; row; return 1; }
  if headroom; then CUR=$t; return 0; fi
  B_FAIL=vram; log "[$t] headroom $B_HEAD (layout $B_LAYOUT)"; row; return 1; }
# psearch BASE ARM IMG CAP REFINE POOLS_NAME SPLITS_NAME [VAR=VAL ...]: the first pool of the list that fits, splits tried
# in order while only cuda:0 is short; REFINE=1: one step above the first pool when it fits. FOUND / FSPLIT; that boot is
# left up, unmeasured. rc 1 = nothing fits (VRAM), rc 2 = a functional failure (no smaller pool can fix it).
psearch(){ local b=$1 arm=$2 img=$3 cap=$4 refine=$5 p s n=0 best= bests= up; local -n pl=$6 sl=$7; shift 7; FOUND=; FSPLIT=
  for p in "${pl[@]}"; do
    for s in "${sl[@]}"; do
      [ "$n" -lt "$cap" ] || { log "[$b] search cap ($cap judged boots) reached at $p"; break 2; }
      n=$(( n + 1 ))
      if fits "$b@$p-s${s//, /-}" "$arm" "$img" "$p" GPU_SPLIT="$s" "$@"; then best=$p; bests=$s; break 2; fi
      [ "$B_FAIL" = vram ] || { log "[$b] functional failure at $p [$s], not VRAM: search stopped"; return 2; }
      [ "$B_SHORT" = 0 ] || break
    done
  done
  [ -n "$best" ] || return 1
  if [ "$refine" = 1 ] && [ "$best" = "${pl[0]}" ]; then
    up=$(( best + STEP ))
    if fits "$b@$up-s${bests//, /-}" "$arm" "$img" "$up" GPU_SPLIT="$bests" "$@"; then best=$up
    else
      [ "$B_FAIL" = vram ] || { log "[$b] functional failure at $up, not VRAM: search stopped"; return 2; }
      fits "$b@$best-s${bests//, /-}-again" "$arm" "$img" "$best" GPU_SPLIT="$bests" "$@" \
        || { log "[$b] $best [$bests] did not fit on the re-boot"; return 1; }
    fi
  fi
  FOUND=$best; FSPLIT=$bests; log "[$b] search: $best [$bests] after $n judged boot(s)"; return 0; }
SALT0=$(( $(date +%s) % 100000 ))   # per-invocation nonce (a killed run's prompts never serve the re-run)
pf_median(){ python3 -c 'import json,sys,statistics as st
r=[x for x in map(json.loads,open(sys.argv[1])) if x.get("tag")==sys.argv[2] and x.get("ttft_s")]
print("%.0f %d" % (st.median([x["prompt_tokens"]/x["ttft_s"] for x in r]), len(r)) if r else "NA 0")' "$R/prefill.jsonl" "pf-$1-$2" 2>/dev/null || echo "NA 0"; }
# measure TAG [full|pfonly]: the boot that is up is measured under TAG (B_TAG renamed)
measure(){ local t=$1 mode=${2:-full} post oom tb rs c k mpn=- mpf=- line=
  log "[$t] measuring boot $B_TAG (boot-/env-/config-/split- files carry that tag; data files carry $t)"
  B_TAG=$t
  if [ "$mode" = full ]; then
    python3 "$MP" run --url "$API" --model "$MODEL" --tag "warm$t" --n 8 --concs 1 4 8 --tokens 128 --out "$R/warm.jsonl" > "$R/warm-$t.log" 2>&1
    python3 "$GREEDY" --url "$BASEURL" --tag "$t" --out "$R/greedy.jsonl" > "$R/greedy-$t.log" 2>&1
    log "[$t] fn_greedy: $(grep -c "^GREEDY $t " "$R/greedy-$t.log") prompts, $(grep -c ' ERROR ' "$R/greedy-$t.log") errors"
    python3 "$CHAT" --url "$API" --model "$MODEL" --tag "$t" --out "$R/chat-greedy.jsonl" > "$R/chat-greedy-$t.log" 2>&1
    c=$?; log "[$t] chat_greedy (recorded): $(grep -c "^$t " "$R/chat-greedy-$t.log") prompts, rc $c"
  fi
  for c in 30000 120000; do
    for k in 1 2 3; do
      python3 "$FB" --url "$API" --model "$MODEL" --tag "pf-$t-$c" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c --unique \
        --salt $(( (SALT0 + k * 991 + c / 1000) % 100000 )) --out "$R/prefill.jsonl" >> "$R/prefill-$t-$c.log" 2>&1
    done
    line="$line ${c%000}k $(pf_median "$t" $c | awk '{print $1 " t/s (n " $2 ")"}');"
  done
  post=$(vram_free | tr ' ' '/' | sed 's#/$##')
  log "[$t] cold prefill medians:$line free after the 120k $post MiB"
  if [ "$mode" = full ]; then
    python3 "$MP" run --url "$API" --model "$MODEL" --tag "$t" --n 24 --concs 1 4 8 --tokens 512 --out "$R/mp.jsonl" > "$R/mp-$t.log" 2>&1
    read -r mpn mpf <<< "$(python3 -c 'import json,sys
r=[x for x in map(json.loads,open(sys.argv[1])) if x.get("tag")==sys.argv[2]]
print(len(r), sum(1 for x in r if not x.get("ok")))' "$R/mp.jsonl" "$t" 2>/dev/null || echo "0 0")"
    log "[$t] mp_decode: $mpn records ($mpf failed; want 144 ok)"
    # F5 (report-only): top-5 margins on the corpus, after decode so its 60k prefills move no measured number
    if { [ "$t" = S1 ] || [ "$t" = P1 ]; } && [ -e "$CORPUS" ]; then
      for c in 0 60000; do
        timeout 1800 python3 "$LPM" run --url "$BASEURL" --model "$MODEL" --tag "$t" --set corpus --chunks 40 --tokens 64 --ctx $c \
          --probes "$PR" --corpus "$CORPUS" --out "$R/margins.jsonl" >> "$R/margins-$t.log" 2>&1
      done
      log "[$t] F5 lp_margin: $(grep -c "\"tag\": \"$t\"" "$R/margins.jsonl" 2>/dev/null) records, $(grep -c ' ERROR ' "$R/margins-$t.log") errors"
    fi
  fi
  sudo docker logs flashnext > "$R/container-$t.log" 2>&1
  oom=$(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$t.log"); tb=$(grep -ac Traceback "$R/container-$t.log")
  rs=$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo "?")
  log "[$t] OOM $oom; tracebacks $tb; restarts $rs; served $(served_id || echo none)"
  [ -n "$(served_id)" ] || B_GEN=DIED
  row "$post" "$oom" 1 "$tb" "$rs" "$mpn" "$mpf"; }
left(){ echo $(( (END - $(date +%s)) / 60 )); }
need(){ [ "$(left)" -ge "$1" ] && return 0; log "timebox: $(left) min left, $2 needs ~$1: skipped"; return 1; }

# ---------------- arms: S1 P(search)+P1 P2 S2 [PNT] ----------------
PPOOL=; PSPLIT=; PFAIL=; NPOOL=; S1_UP=; S1_POST=; P1_UP=; P1_POST=
declare -A CNT=()
S_ENV="$LENV"
P_ENV="$LENV $TILED_ON"
N_ENV="$LENV $TILED_OFF"
for arm in $ARMS; do
  CNT[$arm]=$(( ${CNT[$arm]:-0} + 1 )); n=${CNT[$arm]}
  case $arm in
    S) if [ "$n" -gt 1 ] && [ -z "$PPOOL" ]; then log "S$n skipped: no P boot to bracket"; continue; fi
       need 14 "S$n" || break
       if ! boot "S$n" S "$DAILY_IMG" "$LPOOL" EXTRA_ENV="$S_ENV"; then
         [ -n "$S_F0" ] || abort_post "S1 did not boot"
         log "S$n did not boot"; continue; fi
       if [ -z "$S_F0" ]; then S_F0=${B_UP%/*}; S_F1=${B_UP#*/}; S1_UP=$B_UP; B_HEAD=REF
         r0=${REF_UP%/*}; r1=${REF_UP#*/}
         if [ "${S_F0:-0}" -ge $(( r0 - 32 )) ] 2>/dev/null && [ "$S_F0" -le $(( r0 + 32 )) ] && [ "$S_F1" -ge $(( r1 - 32 )) ] && [ "$S_F1" -le $(( r1 + 32 )) ]; then
           log "headroom reference (S1): $S_F0/$S_F1 MiB (within the plan reference $REF_UP +- 32)"
         else early "S1 UP free $S_F0/$S_F1 MiB OUTSIDE the plan reference $REF_UP +- 32: the decision will read VOID (run continues for the record)"; fi
       else headroom || true; fi
       layout_probe "S$n"
       measure "S$n"
       [ "$n" = 1 ] && S1_POST=$(awk -F'\t' '$1 == "S1" {print $8}' "$R/boots.tsv" | tail -1) ;;
    P) [ -z "$PFAIL" ] || continue
       need 14 "P$n" || break
       if [ -z "$PPOOL" ]; then
         need 40 "P pool search" || break
         psearch P P "$PIMG" "$PCAP" 1 POOLS_P SPLITS_P EXTRA_ENV="$P_ENV" TUNEDIR="$TUNE_P"; src=$?
         if [ "$src" != 0 ]; then
           PFAIL=$([ "$src" = 1 ] && echo vram || echo func)
           early "pool: NO FIT for tiled ON at >= $FLOOR (search rc $src: $PFAIL) -> NOT-A-CANDIDATE on pool unless PNT (tiled OFF) fits"
           continue; fi
         PPOOL=$FOUND; PSPLIT=$FSPLIT
         early "pool: FOUND $PPOOL [$PSPLIT] tiled ON ($(( (PPOOL - LPOOL) / STEP )) steps vs $LPOOL; floor $FLOOR: PASS) UP $B_UP (REF $S_F0/$S_F1)"
       else
         boot "P$n" P "$PIMG" "$PPOOL" GPU_SPLIT="$PSPLIT" EXTRA_ENV="$P_ENV" TUNEDIR="$TUNE_P" || { log "P$n did not boot at $PPOOL"; continue; }
         layout_probe "P$n"
         oom_after_probe "P$n" || log "[P$n] $B_HEAD after the layout probe (counted in health)"
         headroom || log "[P$n] headroom $B_HEAD"
       fi
       [ "$n" = 1 ] && P1_UP=$B_UP
       measure "P$n"
       if [ "$n" = 1 ]; then
         P1_POST=$(awk -F'\t' '$1 == "P1" {print $8}' "$R/boots.tsv" | tail -1)
         pr=$(for c in 30000 120000; do s=$(pf_median S1 $c | cut -d' ' -f1); p=$(pf_median P1 $c | cut -d' ' -f1)
           python3 -c "import sys; print('%dk x%.3f' % ($c // 1000, $p / $s))" 2>/dev/null || echo "${c%000}k n/a"; done | tr '\n' ' ')
         early "prefill P1/S1 medians: $pr(bar 0.95x, expectation >= 1.10x at 120k)"
       fi ;;
    PNT) [ -z "$PPOOL" ] || continue
       [ "$PFAIL" = vram ] || { log "PNT skipped: the tiled-ON search did not end on VRAM (${PFAIL:-not run})"; continue; }
       need 20 PNT || continue
       psearch PNT PNT "$PIMG" "$NCAP" 0 POOLS_PNT SPLITS_P EXTRA_ENV="$N_ENV" TUNEDIR="$TUNE_P"; src=$?
       [ "$src" = 0 ] || { early "PNT (tiled OFF): no fit in $NCAP boots from ${POOLS_PNT[0]} (rc $src)"; continue; }
       NPOOL=$FOUND; early "PNT (tiled OFF): fits $NPOOL [$FSPLIT]"
       measure "PNT$n" pfonly ;;
  esac
done
[ "$PENDING" = 1 ] && row

# ---------------- F6 corpus fidelity (report-only): OFF = served image, ON = r3 + tiled ----------------
fid_arm(){ local arm=$1 img=$2 env=$3 tune=$4 chunks=$5 out=$6
  sudo timeout 1500 docker run --rm --name r784-fid --gpus all --ipc=host -v $D/models:/models:ro -v "$CORPUS":/corpus.jsonl:ro \
    -v "$R":/results -v "$FIDRUN":/opt/fid/exl3_fidelity_run.py:ro -v "$tune":/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache \
    -e EXLLAMAV3_TUNE_CACHE=/exl3-cache $(dflags "$env") --entrypoint python3 "$img" /opt/fid/exl3_fidelity_run.py \
    --model /models/$MODEL --corpus /corpus.jsonl --out "/results/$out.jsonl" --chunks "$chunks" --tokens 2048 > "$R/$out.log" 2>&1; }
fid_sane(){ grep -aq '^\[fid\] DONE' "$R/$1.log" && python3 -c 'import re,sys
m=re.search(r"mean NLL ([0-9.]+)", open(sys.argv[1]).read()); sys.exit(0 if m and 0.05 < float(m.group(1)) < 8 else 1)' "$R/$1.log"; }
if [ "$FID" = 1 ] && [ -n "$PPOOL" ] && [ -e "$CORPUS" ] && need 25 "F6 fidelity"; then
  served_stop; wait_unserved 45 || true
  ok=1
  for a in "OFF|$DAILY_IMG|$S_ENV|$D/.exl3cache" "ON|$PIMG|$P_ENV|$TUNE_P"; do IFS='|' read -r arm img env tune <<< "$a"
    fid_arm "$arm" "$img" "$env" "$tune" 2 "fid-$arm-smoke"; rc=$?
    fid_sane "fid-$arm-smoke" && log "F6 smoke $arm ($img): rc $rc, $(grep -a '^\[fid\] DONE' "$R/fid-$arm-smoke.log" | cut -c1-120)" \
      || { ok=0; log "F6 smoke $arm FAILED (rc $rc): $(tail -2 "$R/fid-$arm-smoke.log" | tr '\n' ' ' | cut -c1-240); F6 not run (report-only)"; }
  done
  if [ "$ok" = 1 ]; then
    for a in "OFF|$DAILY_IMG|$S_ENV|$D/.exl3cache" "ON|$PIMG|$P_ENV|$TUNE_P"; do IFS='|' read -r arm img env tune <<< "$a"
      fid_arm "$arm" "$img" "$env" "$tune" 200 "fid-$arm"; log "F6 $arm: rc $?, $(grep -a '^\[fid\] DONE' "$R/fid-$arm.log" | cut -c1-120)"; done
    python3 "$FIDPY" compare --ref "$R/fid-OFF.jsonl" "$R/fid-ON.jsonl" > "$R/fid-compare.txt" 2>&1
    sed 's/^/  [F6, report-only, UNCALIBRATED: GATE.md bars +0.05 dNLL% \/ top1 0.995 \/ KL 5e-4; R756 floor ~6e-3 KL] /' "$R/fid-compare.txt" | tee -a "$R/audit.log"
  fi
else log "F6 fidelity skipped (FID=$FID, P pool ${PPOOL:-none}, corpus $( [ -e "$CORPUS" ] && echo present || echo missing))"; fi

# ---------------- read ----------------
python3 "$GREEDY" --compare --ref S1 --out "$R/greedy.jsonl" > "$R/greedy-compare.txt" 2>&1
grep -aE "^GREEDY|MISSING" "$R/greedy-compare.txt" | sed 's/^/  [greedy vs S1] /' | tee -a "$R/audit.log"
for c in P1 P2 S2; do grep -q "\"tag\": \"$c\"" "$R/chat-greedy.jsonl" 2>/dev/null || continue
  python3 "$CHAT" --compare --ref S1 --cand $c --out "$R/chat-greedy.jsonl" > "$R/chat-greedy-compare-$c.txt" 2>&1
  tail -1 "$R/chat-greedy-compare-$c.txt" | sed 's/^/  [chat greedy, record only] /' | tee -a "$R/audit.log"; done
if grep -q '"tag": "P[0-9]' "$R/mp.jsonl" 2>/dev/null; then
  python3 "$MP" compare --a S --b P --summary-json "$R/cmp-S-P.json" "$R/mp.jsonl" 2>&1 | sed "s/^/[P vs S] /" | tee -a "$R/analysis.txt"; fi
# A/A: mp_decode pairs arms by tag prefix, so S1 / S2 are relabelled SA1 / SB1 for the band check
python3 -c 'import json,sys
m={"S1":"SA1","S2":"SB1"}
with open(sys.argv[2],"w") as o:
    for l in open(sys.argv[1]):
        r=json.loads(l)
        if r.get("tag") in m: r["tag"]=m[r["tag"]]; o.write(json.dumps(r)+"\n")' "$R/mp.jsonl" "$R/mp-aa.jsonl" 2>/dev/null
grep -q '"tag": "SB1"' "$R/mp-aa.jsonl" 2>/dev/null && python3 "$MP" compare --a SA --b SB --summary-json "$R/cmp-SA-SB.json" "$R/mp-aa.jsonl" 2>&1 \
  | sed "s/^/[S2 vs S1 A\/A] /" | tee -a "$R/analysis.txt"
[ -s "$R/margins.jsonl" ] && { python3 "$LPM" compare "$R/margins.jsonl" --ref S1 > "$R/margins-compare.txt" 2>&1
  grep -aE "DIVERGE|MARGIN-SUMMARY" "$R/margins-compare.txt" | cut -c1-220 | sed 's/^/  [F5 margin, report-only] /' | tee -a "$R/audit.log"; }
python3 "$DECIDE" --dir "$R" --floor "$FLOOR" --lpool "$LPOOL" --ref-up "$REF_UP" --band 0.02 > "$R/summary.txt" 2>&1   # 2 %: operator decision (R750 same-day drift <= 1.7 %)
tee -a "$R/audit.log" < "$R/summary.txt"
DEC=$(sed -n 's/^DECISION: \([A-Z-]*\).*/\1/p' "$R/summary.txt" | tail -1)
write_promote "${DEC:-INCOMPLETE}"
trap 'log "signal during the final restore: ignored, finishing"' TERM INT HUP
finish "DONE ($(grep -a '^DECISION' "$R/summary.txt" | cut -c1-200))"
