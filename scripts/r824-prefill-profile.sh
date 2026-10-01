#!/usr/bin/env bash
# Operator only. Deploy this whole directory to /srv/qwen5090/prefill-throughput,
# plus existing probes/profile_decode.py and lib/{gpu-queue,serve-ctl,gateway-drain}.sh.
# sudo systemd-run --unit=r824-prefill-profile --collect -p RuntimeMaxSec=43200 \
#   -p TimeoutStopSec=900 -p Environment=HOME=${HOME} \
#   /usr/bin/bash /srv/qwen5090/prefill-throughput/r824-prefill-profile.sh
# No build/pull, serving-config edits or promotion. Probe timebox 900s.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
D=/srv/qwen5090
UNIT=r824-prefill-profile
P=$D/prefill-throughput
LIVE=$D/launch-flashnext.sh
LIVE_MD5=8b644c17e60049a8069fc901b6f091fa
IMG=tabbyapi:r823c-cachetail-inforward
IMAGE_ID=sha256:f5a3c35e2e47647f2200ff33ca822405036835f0729544fad11bad7c56c226b7
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
QUIESCE=${QUIESCE-'hermes hermes-webui owui-proxy'}
R=$(mktemp -d "$D/results/$(date +%F)-r824-prefill-profile-XXXXXX") || exit 3
mkdir -p "$R/packet" "$R/preflight"
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
md5f(){ md5sum < "$1" | cut -d' ' -f1; }
abort_pre(){ log "ABORT before lock: $*"; exit 3; }
for f in "$LIVE" "$P/r824_probe.py" "$P/test_r824.py" "$P/prompt.txt" "$P/fixtures/daily-layout.json" "$D/probes/profile_decode.py" "$D/lib/gpu-queue.sh" "$D/lib/serve-ctl.sh" "$D/lib/gateway-drain.sh"; do
  [ -f "$f" ] || abort_pre "missing $f"
done
[ "$(md5f "$LIVE")" = "$LIVE_MD5" ] || abort_pre 'foreign LIVE launcher'
[ "$(sudo docker image inspect -f '{{.Id}}' "$IMG")" = "$IMAGE_ID" ] || abort_pre 'foreign image tag'
cp -R "$P/." "$R/packet/" || abort_pre 'packet archive failed'
cp "$D/probes/profile_decode.py" "$R/packet/profile_decode.py" || abort_pre 'harness archive failed'
cp "$LIVE" "$R/launch-live.sh" || abort_pre 'LIVE archive failed'
L=$R/launch-live.sh
python3 "$R/packet/test_r824.py" > "$R/preflight/selftest.txt" 2>&1 || abort_pre 'offline self-test failed'
sudo docker run --rm --network none --entrypoint python3 "$IMAGE_ID" /opt/r823c-cachetail-inforward/landing_r823c.py \
  > "$R/preflight/landing.txt" 2>&1 || abort_pre 'image source landing failed'
snapshot(){ local tag=$1
  sudo docker inspect flashnext > "$R/inspect-$tag.json" || return 1
  sudo docker exec flashnext cat /app/config.yml > "$R/config-$tag.yml" || return 1
  python3 "$R/packet/r824_probe.py" identity --launcher "$LIVE" --inspect "$R/inspect-$tag.json" \
    --config "$R/config-$tag.yml" --out "$R/env-$tag.json" > "$R/identity-$tag.txt" 2>&1
}
clocks(){ sudo python3 - <<'PY'
import json,pynvml as n
n.nvmlInit();h=[n.nvmlDeviceGetHandleByIndex(i) for i in range(n.nvmlDeviceGetCount())]
assert len(h)==2
v=dict(core=[n.nvmlDeviceGetGpcClkVfOffset(x) for x in h],
       memory=[n.nvmlDeviceGetMemClkVfOffset(x) for x in h],
       power=[n.nvmlDeviceGetPowerManagementLimit(x)//1000 for x in h],
       default_power=[n.nvmlDeviceGetPowerManagementDefaultLimit(x)//1000 for x in h])
assert v['core']==[0,0] and v['memory']==[4500,4500] and v['power']==v['default_power'],v
print(json.dumps(v,sort_keys=True))
PY
}
wait_endpoint_stop(){
  local deadline=$((SECONDS+90))
  while [ "$SECONDS" -lt "$deadline" ]; do
    [ -z "$(served_id)" ] && return 0
    sleep 2
  done
  return 1
}
# A queued successor can see the predecessor's daily already down. Check runtime
# identity when present; never boot the daily between queued units.
if [ "$(sudo docker inspect -f '{{.State.Running}}' flashnext 2>/dev/null)" = true ]; then
  snapshot pre || abort_pre 'LIVE runtime identity failed'
fi
export GPU_QUEUE_NAME=$UNIT
. "$D/lib/gpu-queue.sh"
. "$D/lib/serve-ctl.sh"
. "$D/lib/gateway-drain.sh"
SCTL_LOG=$R/audit.log
BOOTED=0 FINISHED=0 CLIENT_PID= WAS_RUNNING=
# R824 lifecycle begin: self-test executes these functions under shell mocks.
stop_probe(){
  sudo docker rm -f r824-probe > "$R/probe-cleanup.txt" 2>&1 || true
  if [ -n "$CLIENT_PID" ]; then kill "$CLIENT_PID" 2>/dev/null || true; wait "$CLIENT_PID" 2>/dev/null || true; CLIENT_PID=; fi
}
restore_identity(){
  [ "$(served_id)" = "$MODEL" ] && snapshot restored && \
    [ "$(md5f "$L")" = "$LIVE_MD5" ] && [ "$(md5f "$LIVE")" = "$LIVE_MD5" ]
}
restore_daily(){
  [ -z "$(gpu_queue_others)" ] || return 0
  # Bound the restore too: stop <=60s, probe <=930s, restore <=630s,
  # endpoint-stop polls <=90s. No unbounded post-launch health wait.
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    timeout -k 30 600 bash "$L"
}
finish(){
  [ "$FINISHED" = 1 ] && return 0
  FINISHED=1
  trap 'log "signal during restore ignored"' TERM INT HUP
  local rc=0
  stop_probe
  if [ "$BOOTED" = 1 ]; then
    # Restore archived LIVE R823p daily, never .pre-r823 / f3975d68.
    restore_daily > "$R/boot-restore.log" 2>&1 || rc=3
    if [ -z "$(gpu_queue_others)" ]; then
      restore_identity || { log 'FAILED: LIVE restore identity'; rc=3; }
    else log 'queue continues; daily restore deferred'; fi
  fi
  if [ -n "$WAS_RUNNING" ]; then sudo docker start $WAS_RUNNING > "$R/restarted.txt" 2>&1 || rc=3; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  log "finish rc=$rc; results $R"
  return "$rc"
}
queued_signal(){
  trap '' TERM INT HUP
  log 'signal while queued; no serving mutation'
  exec 9>"$D/gpu-exclusive.lock"
  if flock -n 9 && [ -z "$(served_id)" ] && [ -z "$(gpu_queue_others)" ]; then
    BOOTED=1; finish || true
  fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  exit 4
}
# R824 lifecycle end
trap queued_signal TERM INT HUP
trap 'rm -f "${GPU_QUEUE_MARK:-/nonexistent}"' EXIT
gpu_lock
[ "$(md5f "$LIVE")" = "$LIVE_MD5" ] && [ "$(md5f "$L")" = "$LIVE_MD5" ] && \
  [ "$(sudo docker image inspect -f '{{.Id}}' "$IMG")" = "$IMAGE_ID" ] || { log 'identity changed in queue'; exit 3; }
trap 'log "signal: capture invalidated"; echo signal > "$R/VOID"; exit 4' TERM INT HUP
trap 'rc=$?; trap - EXIT; finish || rc=3; exit "$rc"' EXIT
if [ -n "$(served_id)" ]; then
  snapshot locked || { log 'LIVE runtime mismatch after queue'; exit 3; }
  [ "$(served_id)" = "$MODEL" ] || { log 'foreign daily'; exit 3; }
fi
for c in $QUIESCE; do
  if [ "$(sudo docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" = true ]; then
    WAS_RUNNING="$WAS_RUNNING $c"
    sudo docker stop "$c" >/dev/null 2>&1 || exit 3
  fi
done
gateway_drain
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || { log 'drain failed'; exit 3; }
clocks > "$R/clocks-start.json" || { log 'foreign clock/power controls'; exit 3; }
# Resolve selectors from the immutable launcher, including image's trace default.
python3 - "$L" "$R/selectors.txt" <<'PY' || exit 3
import re,sys
s=open(sys.argv[1]).read();line=re.search(r'^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$',s,re.M)[1]
with open(sys.argv[2],'w') as f:
    for item in line.split(): f.write(item+'\n')
    f.write('EXL3_CACHE_TRACE=1\n')
PY
ARGS=()
while IFS= read -r kv; do ARGS+=(-e "$kv"); done < "$R/selectors.txt"
sudo docker logs --timestamps flashnext > "$R/container-live.log" 2>&1 || true
BOOTED=1
served_stop
wait_endpoint_stop || { log 'daily did not stop in 90s'; exit 3; }
# No inherited GPU/drain descriptors in probe children; parent owns both.
spawn(){ if [ -n "${GATEWAY_DRAIN_FD:-}" ]; then "$@" 9>&- {GATEWAY_DRAIN_FD}<&- & else "$@" 9>&- & fi; }
spawn timeout -k 30 900 sudo docker run --rm --name r824-probe --network none --gpus all --ipc=host --shm-size=16g \
  "${ARGS[@]}" -v "$D/models:/models:ro" -v "$D/.exl3cache-rebase-dev-r3:/exl3-cache" \
  -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
  -v "$R:/results" --entrypoint python3 "$IMAGE_ID" /results/packet/r824_probe.py gpu \
  --model "/models/$MODEL" --harness /results/packet/profile_decode.py --out /results/gpu \
  --daily-layout /results/packet/fixtures/daily-layout.json --prompt-file /results/packet/prompt.txt \
  > "$R/probe.log" 2>&1
CLIENT_PID=$!
if ! wait "$CLIENT_PID"; then CLIENT_PID=; echo probe-failed > "$R/VOID"; log 'profile failed/timebox; restoring'; exit 3; fi
CLIENT_PID=
[ "$(md5f "$LIVE")" = "$LIVE_MD5" ] || { echo launcher-drift > "$R/VOID"; exit 3; }
clocks > "$R/clocks-end.json" && cmp "$R/clocks-start.json" "$R/clocks-end.json" \
  || { echo hardware-drift > "$R/VOID"; exit 3; }
finish || exit 3
python3 "$R/packet/r824_probe.py" analyze --root "$R/gpu" --out "$R/reading.json" > "$R/reading.txt" 2>&1 \
  || { echo analysis-incomplete > "$R/VOID"; log 'analysis failed closed'; exit 3; }
cat "$R/reading.txt" >> "$R/audit.log"
