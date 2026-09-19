#!/usr/bin/env bash
# R521 — shared-expert residual probe (Opus round shared-expert-fused r1, patches/exllamav3/shared-expert-fused/r1). The
# round built nothing: after R490's side stream a fused/regridded shared kernel cannot be bit-identical and has ≤0.8 ms/step
# to win at c1 d3, ~0 at c4 d3. This unit measures the one open number: per-layer residual = overlap-on − routed-only, rows
# 1/2/4/8/16, layers 0/29 (cuda:0) and 30/47 (cuda:1) of the 2.50 pack, served image + daily EXTRA_ENV, a COPY of the tune
# cache. Decision printed by the probe: worst residual at rows 4/8/16 < 2 µs/layer close P1; 2–5 park; ≥ 5 open a
# scheduling round. No build, nothing served changes. GPU TIMEBOX 15 min (user 2026-09-19).
# RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=${HOME:?}
R=/srv/qwen5090/results/2026-09-19-r521-shared-bound; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
T=/srv/qwen5090/probes/r521-shared
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
API=http://127.0.0.1:8022/v1
BOOTED=0
log(){ echo "$(date -Is) [r521] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
finish(){ if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"
    env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R521 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$T/run_box_probe.sh" "$T/shared_expert_bound_gpu.py" "$T/autotune_geometry.py"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
(cd "$T" && python3 -m unittest test_cpu 2>&1 | tail -3) | tee -a "$R/audit.log"
export GPU_QUEUE_NAME=r521-shared-bound
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; timebox 900 s"
BOOTED=1
sudo docker stop -t 30 flashnext >/dev/null 2>&1; sudo docker rm -f flashnext >/dev/null 2>&1; sleep 3
RESULTS="$R" timeout 900 bash "$T/run_box_probe.sh" > "$R/wrapper.log" 2>&1; rc=$?
log "probe exit $rc"
grep -aiE "decision|verdict|residual|FAIL|Traceback|Error" "$R/probe.log" 2>/dev/null | tail -20 | cut -c1-240 | tee -a "$R/audit.log"
[ $rc = 0 ] && finish DONE || finish FAILED
