#!/usr/bin/env bash
# R557 — the R518 8-agent SWE-bench replay on today's daily (R548: ring + bf16 GDN state, pool 1,032,192, decode kernels r6,
# E3-DET, int8 mixer, pruned device draft; tier off like R518). R518's S4 arms (786,432 pool, 2026-09-19 00:13 UTC stack)
# read wall 422.2 / 444.8 s, 366 calls, p50/p90 latency 4.5 / 12.4 s. Same trajectories, 8 agents, 16 conversations x 24
# calls, 2 s tool gap, greedy, recorded history re-sent. Two runs on one boot (the second shows run-to-run spread).
# GPU TIMEBOX 25 min. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r557-agent-replay-daily; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
TRAJS="/srv/qwen5090/results/2026-09-16-r359-swebench-10/out/*/*.traj.json /srv/qwen5090/results/2026-09-16-r360-swebench-strat/out/*/*.traj.json /srv/qwen5090/results/2026-09-16-r361-swebench-failed/out/*/*.traj.json /srv/qwen5090/results/2026-09-16-r369-swebench-30/out/*/*.traj.json"
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r557] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){ if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R557 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" /srv/qwen5090/probes/agent_replay.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
[ "$(ls $TRAJS 2>/dev/null | wc -l)" -ge 16 ] || { log "ABORT: fewer than 16 trajectories"; exit 3; }
export GPU_QUEUE_NAME=r557-agent-replay-daily
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; live image $(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
"${CLEAN_ENV[@]}" NVME_TIER= bash "$LIVE" > "$R/boot.log" 2>&1 || { log "NO BOOT"; finish FAILED; exit 3; }
for i in $(seq 150); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
[ "$(served_id)" = "$MODEL" ] || { log "NO BOOT"; finish FAILED; exit 3; }
log "UP: $(sudo grep -E '^  (cache_size|max_batch_size|cache_mode):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '); VRAM free $(vram)"
for t in D1 D2; do
  since=$(date -Is)
  timeout 2400 python3 /srv/qwen5090/probes/agent_replay.py --url "$API" --model "$MODEL" --trajs $TRAJS --agents 8 --convs 16 --calls 24 \
    --tool-gap 2 --tag "$t" --out "$R/replay.jsonl" 2>&1 | tail -1 | sed "s/^/[replay $t] /" | tee -a "$R/audit.log"
  sudo docker logs --since "$since" flashnext > "$R/replay-$t.tabby.log" 2>&1
  [ -e /srv/qwen5090/probes/tabby_log_stats.py ] && python3 /srv/qwen5090/probes/tabby_log_stats.py "$R/replay-$t.tabby.log" 2>&1 | tail -6 | sed "s/^/[tabby $t] /" | tee -a "$R/audit.log"
done
[ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext)" = 0 ] && log "server alive after the replay" || log "SERVER NOT ALIVE after the replay"
finish DONE
