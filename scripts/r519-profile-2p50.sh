#!/usr/bin/env bash
# R519 — survey M1 (flan/patches/exllamav3/decode-survey/r1 §10): the first decode profile of the 2.50bpw daily (every existing
# kernel budget is the 3.05 pack, R464/R465). Daily image stack-r4-e3r2 + the daily's full EXTRA_ENV, split 30/30, cache 8,8,
# chunk 2048, the R465 harness (probes/r465/profile_decode_events.py): wall then profiler mode for c1 d3, c4 d3, c6 d1 at 1k
# context, 64 warmup + 8 settle + 64 capture steps. GPU TIMEBOX 15 min after the lock (user 2026-09-19): shapes that do not fit
# are skipped and logged. Output feeds the survey's §4 tables at 2.50 (kernel groups, launches, host gap).
# RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=${HOME:?}
R=/srv/qwen5090/results/2026-09-19-r519-profile-2p50; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
HARNESS=/srv/qwen5090/probes/r465/profile_decode_events.py
METER=/srv/qwen5090/probes/r465/events_meter.py
IMG=tabbyapi:stack-r4-e3r2
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
TUNEDIR=/srv/qwen5090/.exl3cache
API=http://127.0.0.1:8022/v1
BOOTED=0
log(){ echo "$(date -Is) [r519] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
finish(){ sudo docker rm -f r519-probe >/dev/null 2>&1 || true
  if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"
    env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R519 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$HARNESS" "$METER" /srv/qwen5090/models/$MODEL/config.json; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
ENVS=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE")
[ -n "$ENVS" ] || { log "ABORT: cannot read the daily EXTRA_ENV from $LIVE"; exit 3; }
DENV=(); for kv in $ENVS; do DENV+=(-e "$kv"); done
export GPU_QUEUE_NAME=r519-profile-2p50
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 900 ))
log "lock held; timebox ends $(date -Is -d @$END); env: $ENVS"
BOOTED=1
sudo docker stop -t 30 flashnext >/dev/null 2>&1; sudo docker rm -f flashnext >/dev/null 2>&1; sleep 3
RUN="$R/run-$(date +%H%M%S)"; mkdir -p "$RUN"
profile(){ local sub=$1; shift; local left=$(( END - $(date +%s) ))
  [ $left -lt 90 ] && { log "SKIP $sub: timebox ($left s left)"; return 1; }
  log "shape set $sub: $* (timeout ${left}s)"
  timeout $left sudo docker run --rm --name r519-probe --gpus all --ipc=host --shm-size=16g \
    -v "$TUNEDIR":/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v "$HARNESS":/probe/profile_decode_events.py:ro -v "$METER":/probe/events_meter.py:ro -v "$RUN":/results "${DENV[@]}" \
    --entrypoint python3 "$IMG" /probe/profile_decode_events.py --model /models/$MODEL --out /results/$sub \
    --cache-quant 8,8 --gpu-split 30,30 --max-chunk-size 2048 --tokens 128 --warmup-steps 64 --settle-steps 8 --capture-steps 64 "$@" > "$RUN/$sub.log" 2>&1
  local rc=$?; sudo docker rm -f r519-probe >/dev/null 2>&1; log "shape set $sub exit $rc"; return $rc; }
for arm in wall profiler; do
  profile c1d3-$arm --contexts 1024 --batch 1 --draft 3 --capture-mode $arm
  profile c4d3-$arm --contexts 1024 --batch 4 --draft 3 --capture-mode $arm
  profile c6d1-$arm --contexts 1024 --batch 6 --draft 1 --capture-mode $arm
done
python3 - "$RUN" <<'PY' | tee "$RUN/summary.txt" | tee -a "$R/audit.log"
import json,sys,pathlib
run=pathlib.Path(sys.argv[1])
for d in sorted(p for p in run.iterdir() if p.is_dir()):
    for f in sorted(d.rglob("*.json")):
        if f.name not in ("summary.json","events.json","kernels.json"): continue
        try: j=json.loads(f.read_text())
        except Exception as e: print(f"{d.name}/{f.name}: unreadable {e}"); continue
        for e in (j if isinstance(j,list) else [j]):
            if not isinstance(e,dict): continue
            cap=e.get("capture",{}) if isinstance(e.get("capture"),dict) else {}
            wall=cap.get("elapsed_ms_per_step") or e.get("wall_ms_per_step") or e.get("elapsed_ms_per_step")
            print(f"{d.name}/{f.relative_to(d)}: status={e.get('status')} wall_ms_per_step={wall} kernel_ms={e.get('total_kernel_ms')}")
PY
finish DONE
