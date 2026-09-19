#!/usr/bin/env bash
# R527 — TP=2 bounding probe (user 2026-09-19 "Do it": 15-min probe, GO only if projected c1 d3 >= 1.25x, else TP is closed).
# Opus round patches/exllamav3/tp-bound/r1: half-shape GDN / attention / MoE / shared / head modules built with exllamav3's
# own tp_export/tp_import (each full layer must equal rank0 + rank1), a P2P gate, NCCL (eager + CUDA graph, must say
# "via P2P") and peer-copy all-reduce latencies, then the budget against the R519 profile. One docker run of the served image
# with the live EXTRA_ENV; nothing built. GPU TIMEBOX 15 min. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r527-tp-bound; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/tp-bound-r1
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r527] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
finish(){ if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R527 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
[ -x "$SRC/tests/run_box_probe.sh" ] || { log "ABORT: missing $SRC/tests/run_box_probe.sh"; exit 3; }
ENVS=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE"); [ -n "$ENVS" ] || { log "ABORT: no daily EXTRA_ENV"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE"); [ -n "$LIMG" ] || { log "ABORT: no daily IMG"; exit 3; }
export GPU_QUEUE_NAME=r527-tp-bound
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; image $LIMG, env: $ENVS"
BOOTED=1
sudo docker stop -t 30 flashnext >/dev/null 2>&1; sudo docker rm -f flashnext >/dev/null 2>&1; sleep 3
timeout 900 env RESULTS="$R" IMG="$LIMG" EXTRA_ENV="$ENVS" bash "$SRC/tests/run_box_probe.sh" > "$R/wrapper.log" 2>&1; rc=$?
log "probe rc=$rc"
grep -aE "FAIL|P2P|via P2P|DECISION|speedup|NO-GO|GO " "$R/wrapper.log" | tail -14 | cut -c1-240 | tee -a "$R/audit.log"
[ -f "$R/tp_bound_budget.txt" ] && tail -25 "$R/tp_bound_budget.txt" | cut -c1-240 | tee -a "$R/audit.log"
[ $rc = 0 ] && finish DONE || finish FAILED
