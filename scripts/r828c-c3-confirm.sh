#!/usr/bin/env bash
# R828c (2026-10-02): prospective c3-code confirmation, REVIEW-R828.md findings 6/2.
# PRE-REGISTERED ORDER (before data), 8 NEW independent pairs / 16 fresh boots:
# P01_OFF -> P01_ON; P02_ON -> P02_OFF; P03_OFF -> P03_ON; P04_ON -> P04_OFF;
# P05_OFF -> P05_ON; P06_ON -> P06_OFF; P07_OFF -> P07_ON; P08_ON -> P08_OFF.
# Each boot: unchanged R828 fn_shape c3-code (warmup + nine measured streams).
# g_i=log(median_ON_i/median_OFF_i); PASS iff
# mean(g)-1.8946*sample_sd(g)/sqrt(8) > log(0.99), strictly, only these new pairs.
# No interim stopping, no extra pairs, no stream pooling or valid-boot exclusions.
# Harness-invalid pair aborts INCONCLUSIVE; requires a newly registered run.
# Small-sample paired-t assumption: independent approximately normal log gains.
# Singleton tails retain real lookup checks/opener reads; c3 is not lookup-free.
# R828's original REJECT remains in evidence/original-decision.json; its other
# speed/identity/activation gates must reproduce before this confirmation starts.
# Candidate tabbyapi:r828-prompt-lookup-r3 pinned by prompt-lookup-r3/IMAGE_ID.env;
# prompts, production sources, cache/clocks/geometry frozen against R828 archive.
# PASS promotion PRE-APPROVED: same R828 promotion section and promote.py;
# generated repo launcher, atomic LIVE, .pre-r828 rollback, FASTWARM, c1,
# fn/chat/fnstyle greedy identity vs archived OFF2, VRAM >= reference -32 MiB.
# GPU ESTIMATE: target <=45 min + <=10 min promotion, excluding queue/Olla drain.
# 16 boots ~32 min, 16 c3 cells ~7 min, restore/margin ~6 min. No daily between pairs.
# DRY-RUN: ~2 min, no restart: actual R828 raw c3 parser replay + both generated
# launchers' PREBOOT guards, daily runtime/clocks/VRAM. No synthetic speed data.
# DEPLOY: this unit; entire prompt-lookup-r3/ incl confirm_c3.py and tests;
# launch-flashnext-r828-lookup.sh; lib/{gpu-queue,serve-ctl,gateway-drain}.sh;
# complete original R828 result archive (default below, or dry-run argument).
# No image build: use the already pinned R828 image. Run on flan:
# bash /srv/qwen5090/r828c-c3-confirm.sh dry-run /path/to/original/R828/results
# bash /srv/qwen5090/r828c-c3-confirm.sh gpu /path/to/THIS/DRY_RUN_RESULT
set -euo pipefail
export HOME=${HOME:-/srv/qwen5090/operator-home}
D=/srv/qwen5090
P=$D/prompt-lookup-r3
LIVE=$D/launch-flashnext.sh
LIVE_MD5=262e9c31f714724409635fbac9df8ac2
PARENT=tabbyapi:r825c-hostprepare
PARENT_ID=sha256:aa04a1cbe94b60bbd91e3d69d15f5e1b6eb679f14885e599404de218d8a897c9
IMG=tabbyapi:r828-prompt-lookup-r3
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
API=http://127.0.0.1:8022/v1
CLEAN_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
MODE=${1:-}; DRY_RESULT=${2:-}
RAW_R828=${2:-$D/results/2026-10-02-r828-prompt-lookup-r3-em6iO4}
case "$MODE" in dry-run|gpu) ;; *) echo 'usage: r828c-c3-confirm.sh dry-run [RAW_R828] | gpu DRY_RUN_RESULT' >&2; exit 3;; esac
UNIT=r828c-c3-confirm; [ "$MODE" != dry-run ] || UNIT=r828c-dry
R=$(mktemp -d "$D/results/$(date +%F)-$UNIT-XXXXXX")
mkdir -p "$R/packet" "$R/launchers"
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
md5f(){ md5sum < "$1" | cut -d' ' -f1; }
fail(){ echo "$*" > "$R/VOID"; log "DECISION INCONCLUSIVE (INVALID harness): $*"; exit 3; }
for f in "$LIVE" "$P/gate.py" "$P/confirm_c3.py" "$P/promote.py" "$P/selftest.py" "$P/replay_gate.py" "$P/fn_bench.py" "$P/fn_driver.py" "$D/lib/gpu-queue.sh" "$D/lib/serve-ctl.sh" "$D/lib/gateway-drain.sh"; do
  [ -f "$f" ] || fail "missing $f"
done
[ "$(md5f "$LIVE")" = "$LIVE_MD5" ] || fail 'LIVE launcher md5'
cp -R "$P/." "$R/packet/"
cp "$LIVE" "$R/launch-live.sh"
cp "$0" "$R/unit.sh"
UNIT_SHA=$(sha256sum < "$R/unit.sh" | cut -d' ' -f1)
# All invoked Python and fixed prompts are archived before queue wait.
Q=$R/packet
packet_sha(){ python3 - "$Q" <<'PY'
import hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]);h=hashlib.sha256()
for f in sorted(p.rglob('*')):
    if not f.is_file() or '__pycache__' in f.parts or f.suffix=='.pyc' or f.name=='IMAGE_ID.env': continue
    h.update(str(f.relative_to(p)).encode()+b'\0'+f.read_bytes())
print(h.hexdigest())
PY
}
PACKET_SHA=$(packet_sha)
python3 -B "$Q/confirm_c3.py" declare --root "$R" | tee -a "$R/audit.log"
python3 -B "$Q/selftest.py" > "$R/selftest.txt" 2>&1 || fail 'CPU install/parser replay'
python3 -B "$Q/replay_gate.py" > "$R/gate-replay.txt" 2>&1 || fail 'CPU gate replay'
python3 -B "$Q/gate.py" prepare --live "$R/launch-live.sh" --out "$R/launchers" > "$R/launchers.txt" || fail 'candidate construction'
CANDIDATE_ID=$(sed -n 's/^R828_IMAGE_ID=//p' "$Q/IMAGE_ID.env")
[[ "$CANDIDATE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || fail 'fill IMAGE_ID.env with operator-built full R828 image ID'
if [ "$MODE" = dry-run ]; then
  python3 -B "$Q/confirm_c3.py" replay --source "$RAW_R828" --out "$R/evidence" > "$R/real-replay.txt" 2>&1 || fail 'real R828 replay / frozen candidate / inherited gates'
else
  [ -d "$DRY_RESULT/evidence" ] || fail 'missing bound R828 evidence'
  cp -R "$DRY_RESULT/evidence" "$R/evidence"
fi
EVIDENCE_SHA=$(python3 -B "$Q/confirm_c3.py" digest --root "$R/evidence")
if [ "$MODE" = gpu ]; then
  [ -f "$DRY_RESULT/DRY_PASS.json" ] || fail 'run dry-run first; supply its results directory'
  python3 - "$DRY_RESULT/DRY_PASS.json" "$PACKET_SHA" "$LIVE_MD5" "$UNIT_SHA" "$CANDIDATE_ID" "$EVIDENCE_SHA" <<'PY' || fail 'dry PASS is stale/foreign'
import json,sys,time
v=json.load(open(sys.argv[1])); assert v['packet_sha']==sys.argv[2] and v['live_md5']==sys.argv[3]
assert v['unit_sha']==sys.argv[4] and v['candidate_id']==sys.argv[5]
assert v['evidence_sha']==sys.argv[6]
assert 0 <= time.time()-v['time'] <= 86400
PY
  CANDIDATE_ID=$(sed -n 's/^R828_IMAGE_ID=//p' "$Q/IMAGE_ID.env")
  [[ "$CANDIDATE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || fail 'fill IMAGE_ID.env with operator-built full image ID'
  [ "$(sudo docker image inspect -f '{{.Id}}' "$IMG")" = "$CANDIDATE_ID" ] || fail 'candidate image tag drift'
  sudo docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --entrypoint python3 "$CANDIDATE_ID" \
    -B /opt/r828/install.py --check > "$R/landing-candidate.txt" 2>&1 || fail 'candidate landing'
  sudo docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --entrypoint python3 "$CANDIDATE_ID" \
    -B /opt/r828/selftest.py > "$R/install-selftest.txt" 2>&1 || fail 'installed candidate selftest'
fi
[ "$(sudo docker image inspect -f '{{.Id}}' "$PARENT")" = "$PARENT_ID" ] || fail 'parent image drift'
export GPU_QUEUE_NAME=$UNIT
. "$D/lib/gpu-queue.sh"
. "$D/lib/serve-ctl.sh"
. "$D/lib/gateway-drain.sh"
SCTL_LOG=$R/audit.log
BOOTED=0 FINISHED=0 CHILD_PID= WAS_RUNNING= PHASE=QUEUED MUTATED=0 PROMOTED=0 REFERENCE_LIVE=0
# No child inherits GPU/drain descriptors. timeout owns the process group and
# forwards TERM, so a cancelled client cannot keep submitting during restoration.
run(){
  local rc=0
  if [ -n "${GATEWAY_DRAIN_FD:-}" ]; then "$@" 9>&- {GATEWAY_DRAIN_FD}<&- &
  else "$@" 9>&- & fi
  CHILD_PID=$!
  wait "$CHILD_PID" || rc=$?
  CHILD_PID=
  return "$rc"
}
snapshot(){
  local tag=$1 flag=$2 launcher=$3 image=$4 boot=${5:-}
  sudo docker inspect flashnext > "$R/inspect-$tag.json" || return 1
  sudo docker exec flashnext cat /app/config.yml > "$R/config-$tag.yml" || return 1
  local args=()
  [ -z "$boot" ] || args=(--boot "$boot")
  python3 -B "$Q/gate.py" runtime --inspect "$R/inspect-$tag.json" --config "$R/config-$tag.yml" \
    --launcher "$launcher" --flag "$flag" --image-id "$image" "${args[@]}" > "$R/identity-$tag.txt" || return 1
  [ "$(served_id)" = "$MODEL" ] || return 1
}
clocks(){ sudo python3 - <<'PY'
import json,pynvml as n
n.nvmlInit(); hs=[n.nvmlDeviceGetHandleByIndex(i) for i in range(n.nvmlDeviceGetCount())]
assert len(hs)==2
v=dict(core=[n.nvmlDeviceGetGpcClkVfOffset(h) for h in hs],memory=[n.nvmlDeviceGetMemClkVfOffset(h) for h in hs],
       power=[n.nvmlDeviceGetPowerManagementLimit(h) for h in hs],default_power=[n.nvmlDeviceGetPowerManagementDefaultLimit(h) for h in hs])
assert v['core']==[0,0] and v['memory']==[4500,4500] and v['power']==v['default_power'],v
print(json.dumps(v,sort_keys=True))
PY
}
finish(){
  [ "$FINISHED" = 0 ] || return 0
  FINISHED=1
  trap '' TERM INT HUP
  local rc=0
  if [ -n "$CHILD_PID" ]; then kill -TERM "$CHILD_PID" 2>/dev/null || true; wait "$CHILD_PID" 2>/dev/null || true; CHILD_PID=; fi
  if [ "$MUTATED" = 1 ] && [ "$PROMOTED" = 0 ]; then
    cp -p "$LIVE.pre-r828" "$LIVE.new" && mv "$LIVE.new" "$LIVE" || rc=3
    log 'promotion failed: atomic rollback to .pre-r828'
  fi
  if [ "$BOOTED" = 1 ] && [ "$PROMOTED" = 0 ]; then
    if [ -z "$(gpu_queue_others)" ]; then
      # Restore from LIVE only, with no selectors/IMG/config overrides.
      served_stop || true
      run timeout -k 30 600 env -i HOME="$HOME" PATH="$CLEAN_PATH" bash "$LIVE" > "$R/restore.log" 2>&1 || rc=3
      snapshot restored live "$LIVE" "$PARENT_ID" || rc=3
      [ "$(md5f "$LIVE")" = "$LIVE_MD5" ] || rc=3
    else log 'queue continues; daily restore deferred'; fi
  fi
  if [ -n "$WAS_RUNNING" ]; then sudo docker start $WAS_RUNNING > "$R/restarted.txt" 2>&1 || rc=3; fi
  if [ "$rc" != 0 ]; then
    echo restore-or-client-restart-failure > "$R/VOID"
    log 'DECISION INCONCLUSIVE (INVALID harness): restore or quiesced-client restart failed'
  fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  log "finish rc=$rc; results $R"
  return "$rc"
}
on_signal(){
  trap '' TERM INT HUP
  if [ "$PHASE" = QUEUED ] && [ "$MODE" = gpu ]; then
    exec 9>"$D/gpu-exclusive.lock"
    if flock -n 9 && [ -z "$(served_id)" ] && [ -z "$(gpu_queue_others)" ]; then BOOTED=1; fi
  fi
  echo signal > "$R/VOID"
  log 'DECISION INCONCLUSIVE (INVALID harness): signal'
  exit 4
}
trap on_signal TERM INT HUP
trap 'rc=$?; trap - EXIT; if [ "$rc" != 0 ] && [ ! -f "$R/VOID" ]; then echo "unhandled harness exit $rc" > "$R/VOID"; log "DECISION INCONCLUSIVE (INVALID harness): exit $rc"; fi; finish || rc=3; exit "$rc"' EXIT
gpu_lock
PHASE=LOCKED
[ "$(md5f "$LIVE")" = "$LIVE_MD5" ] && [ "$(md5f "$R/launch-live.sh")" = "$LIVE_MD5" ] || fail 'launcher drift in queue'
if [ "$MODE" = gpu ]; then
  [ "$(sudo docker image inspect -f '{{.Id}}' "$IMG")" = "$CANDIDATE_ID" ] || fail 'candidate drift in queue'
  [ -n "$(served_id)" ] || BOOTED=1
fi
if [ -n "$(served_id)" ]; then
  LOCKED_IMAGE=$(sudo docker inspect -f '{{.Image}}' flashnext) || fail 'locked container inspect'
  if [ "$LOCKED_IMAGE" = "$PARENT_ID" ]; then
    snapshot locked live "$LIVE" "$PARENT_ID" || fail 'locked LIVE identity'
    REFERENCE_LIVE=1
  elif [ "$MODE" = gpu ]; then
    # A queued predecessor deliberately left its experimental arm running.
    # Archive it; do not reboot the daily just to stop it for this experiment.
    sudo docker inspect flashnext > "$R/inspect-predecessor.json" || fail 'predecessor inspect'
    BOOTED=1
    log "chain predecessor image=$LOCKED_IMAGE; daily VRAM reference comes from bound dry run"
  else fail 'dry-run requires the served parent daily'; fi
elif [ "$MODE" = dry-run ]; then fail 'dry-run requires running served daily'; fi
# Both stages measure on :8022, so Olla must be drained before any request.
# Fail closed on missing gateway/schema; helper historically returned success on curl failure.
curl -fsS -m 5 "$GATEWAY_STATUS_URL" > "$R/gateway.json" || fail 'gateway status unavailable'
python3 - "$R/gateway.json" <<'PY' || fail 'gateway schema'
import json,sys
v=json.load(open(sys.argv[1])); assert isinstance(v['endpoints'],list)
for e in v['endpoints']: assert isinstance(e['name'],str) and isinstance(e['status'],str) and type(e['active_connections']) is int
PY
gateway_drain
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || fail 'Olla drain timeout'
clocks > "$R/clocks-start.json" || fail 'clock/power identity'
serial(){
  sudo docker logs flashnext > "$R/serial.log" 2>&1
  python3 -B "$Q/gate.py" serial --log "$R/serial.log"
}
cell_capture(){
  local tag=$1 name=$2 lo=$3 expected=$4 flag=$5 parent=${6:-} measured=${7:-$4}
  local args=(); [ "$parent" != parent ] || args=(--parent)
  # Finish metrics can lag the final SSE chunk. Retry missing rows, but a parser error
  # still aborts the stage if not resolved within this bounded log-flush window.
  local attempt
  for attempt in $(seq 20); do
    sudo docker logs flashnext > "$R/container-$tag-$name.log" 2>&1
    if python3 -B "$Q/gate.py" cell --log "$R/container-$tag-$name.log" --lo "$lo" --expected "$expected" \
      --flag "$flag" --measured "$measured" --out "$R/counters-$tag-$name.json" "${args[@]}" > "$R/parser-$tag-$name.txt" 2>&1; then return 0; fi
    sleep 1
  done
  fail "counter/completion parser $tag/$name; see parser file"
}
identity_suite(){
  local tag=$1 suite=$2 flag=$3 parent=${4:-} lo n args=()
  [ "$parent" != parent ] || args=(--short)
  lo=$(serial); n=6; [[ "$suite" != agent && "$suite" != fnstyle ]] || n=12
  run timeout -k 30 1200 python3 -B "$Q/gate.py" client --url "$API" --model "$MODEL" --tag "$tag" \
    --suite "$suite" --out "$R/identity-$tag-$suite.jsonl" "${args[@]}" > "$R/client-$tag-$suite.txt" 2>&1 \
    || fail "identity client $tag/$suite"
  python3 -B "$Q/gate.py" validate --rows "$R/identity-$tag-$suite.jsonl" --identity || fail "identity schema $tag/$suite"
  cell_capture "$tag" "identity-$suite" "$lo" "$n" "$flag" "$parent"
  if [ "$parent" != parent ]; then
    python3 -B "$Q/gate.py" annotate --rows "$R/identity-$tag-$suite.jsonl" \
      --counters "$R/counters-$tag-identity-$suite.json" || fail 'identity adaptive-state join'
  fi
}
fn_shape(){
  local tag=$1 c=$2 kind=$3 flag=$4 parent=${5:-} lo tokens=1024
  [ "$parent" != parent ] || tokens=8
  lo=$(serial)
  run timeout -k 30 600 python3 -B "$Q/fn_driver.py" --url "$API" --model "$MODEL" --tag "$tag-c$c-$kind" \
    --kind "$kind" --distinct --tokens "$tokens" --warmup-runs 1 --conc "$c" --runs 3 --timeout 300 \
    --out "$R/fn-$tag-c$c-$kind.jsonl" > "$R/bench-$tag-c$c-$kind.log" 2>&1 || fail "fn client $tag/c$c/$kind"
  python3 -B "$Q/gate.py" validate --rows "$R/fn-$tag-c$c-$kind.jsonl" || fail 'fn row schema'
  cell_capture "$tag" "c$c-$kind" "$lo" "$((5*c))" "$flag" "$parent" "$((3*c))"
}
# Execute candidate launchers themselves through their image/pin/config guards.
# PREBOOT exits before any docker run/create/start or stop. Configs stay in results.
if [ "$MODE" = dry-run ]; then
  for flag in 0 1; do
    mkdir -p "$R/preflight-$flag"
    run timeout -k 10 60 env -i HOME="$HOME" PATH="$CLEAN_PATH" R828_PREBOOT_ONLY=1 \
      R828_PREFLIGHT_DIR="$R/preflight-$flag" bash "$R/launchers/launch-lookup-$flag.sh" \
      > "$R/preboot-$flag.log" 2>&1 || fail "candidate preboot guards $flag"
    grep -q 'R828 PREBOOT PASS' "$R/preboot-$flag.log" || fail "missing preboot PASS $flag"
  done
fi
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | python3 -c \
  'import json,sys; v=[int(x.strip()) for x in sys.stdin if x.strip()]; assert len(v)==2; print(json.dumps(v))'; }
if [ "$REFERENCE_LIVE" = 1 ]; then
  vram > "$R/vram-reference.json" || fail 'current daily VRAM reference'
else
  cp "$DRY_RESULT/vram-reference.json" "$R/vram-reference.json" || fail 'dry daily VRAM reference'
fi
if [ "$MODE" = dry-run ]; then
  log 'DRY-RUN started: no restart; real raw c3 replay and both preboot guards PASS; 120s target after drain'
  start=$SECONDS
  while [ $((SECONDS-start)) -lt 120 ]; do
    curl -fsS -m 5 "$API/model" | python3 -c 'import json,sys; assert json.load(sys.stdin)["id"]==sys.argv[1]' "$MODEL" || fail 'dry health schema'
    sleep 5
  done
  snapshot dry-end live "$LIVE" "$PARENT_ID" || fail 'dry end identity'
  python3 - "$R/inspect-locked.json" "$R/inspect-dry-end.json" <<'PY' || fail 'dry container changed'
import json,sys
assert json.load(open(sys.argv[1]))[0]['Id']==json.load(open(sys.argv[2]))[0]['Id']
PY
  clocks > "$R/clocks-end.json" && cmp "$R/clocks-start.json" "$R/clocks-end.json" || fail 'dry hardware drift'
  python3 - "$R/DRY_PASS.json" "$PACKET_SHA" "$LIVE_MD5" "$UNIT_SHA" "$CANDIDATE_ID" "$EVIDENCE_SHA" <<'PY'
import json,sys,time
with open(sys.argv[1],'w') as f: json.dump(dict(time=time.time(),packet_sha=sys.argv[2],live_md5=sys.argv[3],unit_sha=sys.argv[4],candidate_id=sys.argv[5],evidence_sha=sys.argv[6]),f)
PY
  log "DRY PASS; DRY_RUN_RESULT=$R"
  exit 0
fi
for c in ${QUIESCE-'hermes hermes-webui owui-proxy'}; do
  if [ "$(sudo docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" = true ]; then
    WAS_RUNNING="$WAS_RUNNING $c"; sudo docker stop "$c" >/dev/null || fail "quiesce $c"
  fi
done
# The literal boot order matches the header and confirm_c3.ORDER. Never stop on
# an interim rate: a valid slow boot is retained; only harness failure aborts.
for tag in P01_OFF P01_ON P02_ON P02_OFF P03_OFF P03_ON P04_ON P04_OFF P05_OFF P05_ON P06_ON P06_OFF P07_OFF P07_ON P08_ON P08_OFF; do
  flag=0; [[ "$tag" != *_ON ]] || flag=1
  L=$R/launchers/launch-lookup-$flag.sh
  [ "$(md5f "$LIVE")" = "$LIVE_MD5" ] || fail 'LIVE drift'
  log "boot $tag image=$CANDIDATE_ID launcher=$L md5=$(md5f "$L")"
  BOOTED=1
  served_stop
  wait_unserved 30 || fail 'endpoint failed to stop'
  run timeout -k 30 600 env -i HOME="$HOME" PATH="$CLEAN_PATH" bash "$L" > "$R/boot-$tag.log" 2>&1 || fail "boot $tag"
  snapshot "$tag" "$flag" "$L" "$CANDIDATE_ID" "$R/boot-$tag.log" || fail "boot identity $tag"
  cards_loaded > "$R/cards-$tag.txt" || fail 'missing GPU residency'
  sudo docker exec flashnext python3 -B /opt/r828/install.py --check > "$R/landing-$tag.txt" 2>&1 || fail "served source landing $tag"
  fn_shape "$tag" 3 code "$flag"
  snapshot "$tag-end" "$flag" "$L" "$CANDIDATE_ID" || fail "end identity $tag"
  python3 -B "$Q/confirm_c3.py" boot --root "$R" --tag "$tag" || fail "boot independence / continuity $tag"
  clocks > "$R/clocks-$tag.json" && cmp "$R/clocks-start.json" "$R/clocks-$tag.json" || fail "hardware drift $tag"
done
python3 -B "$Q/confirm_c3.py" report --root "$R" > "$R/reading.txt" 2>&1 || fail 'confirmatory parser / bound'
cat "$R/reading.txt" | tee -a "$R/audit.log"
DECISION=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["DECISION"])' "$R/decision.json")
if [ "$DECISION" != PASS ]; then
  log "DECISION $DECISION; daily restore through LIVE; no promotion"
  exit 0
fi
# The unchanged R828 promotion helper uses original OFF2 c1 and greedy references.
# These remain evidence only: never part of the eight new c3 log-gain estimate.
cp "$R/evidence/fn-OFF2-c1-code.jsonl" "$R/"
for suite in fn chat fnstyle; do cp "$R/evidence/identity-OFF2-$suite.jsonl" "$R/"; done
# Pre-approved promotion; record the fully resolved repo launcher before LIVE.
[ "$(md5f "$LIVE")" = "$LIVE_MD5" ] || fail 'LIVE drift before promotion'
[ ! -e "$LIVE.pre-r828" ] || fail 'rollback .pre-r828 already exists; preserve existing backup'
cp "$R/launchers/launch-lookup-1.sh" "$D/launch-flashnext-r828-lookup.sh" || fail 'repo promotion launcher'
cp -p "$LIVE" "$LIVE.pre-r828" || fail 'promotion backup'
MUTATED=1
cp "$D/launch-flashnext-r828-lookup.sh" "$LIVE.new" && mv "$LIVE.new" "$LIVE" || fail 'atomic promotion install'
served_stop
wait_unserved 30 || fail 'promotion endpoint did not stop'
run timeout -k 30 600 env -i HOME="$HOME" PATH="$CLEAN_PATH" bash "$LIVE" > "$R/boot-promote.log" 2>&1 || fail 'promotion boot'
snapshot promote 1 "$LIVE" "$CANDIDATE_ID" "$R/boot-promote.log" || fail 'promotion runtime'
sudo docker exec flashnext python3 -B /opt/r828/install.py --check > "$R/landing-promote.txt" 2>&1 || fail 'promotion landing'
cards_loaded > "$R/cards-promote.txt" || fail 'promotion residency'
vram > "$R/vram-promote.json" || fail 'promotion VRAM readback'
fn_shape PROMOTE 1 code 1
for suite in fn chat fnstyle; do identity_suite PROMOTE "$suite" 1; done
clocks > "$R/clocks-promote.json" && cmp "$R/clocks-start.json" "$R/clocks-promote.json" || fail 'promotion clocks'
python3 -B "$Q/promote.py" --root "$R" > "$R/promotion-check.txt" 2>&1 || fail 'post-promotion gates; automatic rollback'
# No trap restoration after this commit point: promoted daily remains live.
# Restart quiesced clients before committing, so restart failure also rolls back.
if [ -n "$WAS_RUNNING" ]; then
  sudo docker start $WAS_RUNNING > "$R/restarted.txt" 2>&1 || fail 'post-promotion client restart'
  WAS_RUNNING=
fi
PROMOTED=1
log "PROMOTED $LIVE; repo launcher $D/launch-flashnext-r828-lookup.sh; rollback $LIVE.pre-r828"
