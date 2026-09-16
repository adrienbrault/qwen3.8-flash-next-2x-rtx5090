#!/usr/bin/env bash
# R349 — the daily's deep-context admission, re-run with the server log captured before teardown.
#
# WHY THIS EXISTS. In the head-to-head (r342), the daily answered 1 of 8 concurrent 38k-context requests: seven
# records came back with a usage block reporting 512 completion tokens, zero text deltas and no error, each after
# ~16 s. That is not a queueing story and not an engine refusal I can name, and it cannot be diagnosed from the
# records alone. r342 stopped the container without capturing its log, so the evidence is gone. This re-run boots
# the daily, runs only the admission arm, and captures `docker logs` BEFORE stopping anything.
#
# It also re-runs the same arm against Flash-Next in the same unit, so the comparison is made under identical
# conditions minutes apart rather than from a different run.
#
# RUN: sudo systemd-run --unit=r349-daily-admit --collect -p User=adrienbrault -p RuntimeMaxSec=10800 \
#        bash /srv/qwen5090/r349-daily-admit.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r349-daily-admit; mkdir -p "$R"
FN_L=/srv/qwen5090/launch-flashnext-r340.sh
FN_API=http://127.0.0.1:8022/v1
FN_MODEL=qwen3.8-flash-next-exl3-3.05bpw
DAILY_API=http://127.0.0.1:8020/v1
log(){ echo "$(date -Is) [r349] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r349-admit
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R349 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

admit(){  # admit <tag> <api> <model>
  python3 /srv/qwen5090/probes/fn_bench.py --url "$2" --model "$3" --tag "$1-admit" --kind code \
     --tokens 512 --ctx 30000 --conc 8 --runs 1 --distinct --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"
}

# --- arm 1: the daily -------------------------------------------------------------------------------
STOP=1 bash "$FN_L" >/dev/null 2>&1 || true
log "booting the daily"
FORCE_RESTORE=1 bash /srv/qwen5090/daily-restore-retry.sh >> "$R/daily-boot.log" 2>&1 || { log "DAILY BOOT FAILED"; finish "ABORTED"; exit 1; }
DAILY_MODEL=$(curl -s -m 10 "$DAILY_API/models" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
log "daily serves: $DAILY_MODEL"
admit daily "$DAILY_API" "$DAILY_MODEL"
# Capture the evidence BEFORE the container goes away -- the mistake r342 made.
sudo docker logs vllm-27b > "$R/daily-server-log.txt" 2>&1
log "daily log captured: $(wc -l < "$R/daily-server-log.txt") lines"
grep -aiE "abort|preempt|cancel|error|exceed|finish_reason|EngineCore|too long" "$R/daily-server-log.txt" | tail -15 | tee -a "$R/audit.log"

# --- arm 2: Flash-Next, same arm, minutes later ------------------------------------------------------
log "stopping the daily"
sudo docker rm -f vllm-27b >/dev/null 2>&1 || true
for i in $(seq 24); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done
bash "$FN_L" >> "$R/audit.log" 2>&1 || { log "FLASHNEXT BOOT FAILED"; finish "ABORTED"; exit 1; }
admit flashnext "$FN_API" "$FN_MODEL"
sudo docker logs flashnext > "$R/flashnext-server-log.txt" 2>&1
log "flashnext log captured: $(wc -l < "$R/flashnext-server-log.txt") lines"

finish DONE
