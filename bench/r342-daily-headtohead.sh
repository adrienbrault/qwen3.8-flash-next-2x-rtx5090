#!/usr/bin/env bash
# R342 — head to head: the Flash-Next TabbyAPI instance against the vLLM 27B daily, one instrument, one day.
#
# WHY. The daily's published numbers (code c1 216 t/s, c8 1,476 aggregate, R234) came from a different instrument
# than the Flash-Next numbers in this repo (vLLM Prometheus counters vs client-side SSE), so quoting them side by
# side is an instrument comparison as much as an engine comparison. This runs bench/probe.py — the same code, the
# same prompts, the same forced length — against both.
#
# THE TWO CANNOT COEXIST: both need both cards resident. The script stops whichever is running, drains the cards,
# boots the other, and restores the Flash-Next baseline at the end because that is what the user's harness points
# at. FORCE_RESTORE=1 is required inside a queued unit: daily-restore-retry.sh deliberately skips the restore when
# another GPU-queue marker is live, and this unit's marker is live by construction.
#
# RUN: sudo systemd-run --unit=r342-h2h --collect -p User=adrienbrault -p RuntimeMaxSec=14400 \
#        bash /srv/qwen5090/r342-headtohead.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r342-headtohead; mkdir -p "$R"
FN_API=http://127.0.0.1:8022/v1
FN_MODEL=qwen3.8-flash-next-exl3-3.05bpw
FN_L=/srv/qwen5090/launch-flashnext-r340.sh
DAILY_API=http://127.0.0.1:8020/v1
log(){ echo "$(date -Is) [r342] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r342-h2h
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R342 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

probe_arm(){  # probe_arm <tag> <api> <model>
  local tag=$1 api=$2 model=$3
  log "--- $tag: code 2048 forced, c1/c4/c8 ---"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$api" --model "$model" --tag "$tag-code" --kind code \
     --tokens 2048 --conc 1 4 8 --runs 1 --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"
  log "--- $tag: prose 2048 forced, c1/c4 ---"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$api" --model "$model" --tag "$tag-prose" --kind prose \
     --tokens 2048 --conc 1 4 --runs 1 --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"
  log "--- $tag: 8 distinct 30k-context jobs at once (admission) ---"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$api" --model "$model" --tag "$tag-admit" --kind code \
     --tokens 512 --ctx 30000 --conc 8 --runs 1 --distinct --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"
}

# --- arm 1: the daily ------------------------------------------------------------------------------
log "stopping Flash-Next (the two cannot coexist)"
STOP=1 bash "$FN_L" >/dev/null 2>&1 || true
log "booting the daily via daily-restore-retry.sh (FORCE_RESTORE=1: this unit holds a queue marker)"
FORCE_RESTORE=1 bash /srv/qwen5090/daily-restore-retry.sh >> "$R/audit.log" 2>&1 || { log "DAILY BOOT FAILED"; finish "ABORTED (daily)"; exit 1; }
DAILY_MODEL=$(curl -s -m 10 "$DAILY_API/models" | python3 -c 'import sys,json; d=json.load(sys.stdin)["data"]; print(d[0]["id"])' 2>/dev/null)
log "daily serves: $DAILY_MODEL"
[ -n "$DAILY_MODEL" ] || { log "no model id from the daily"; finish "ABORTED (daily model id)"; exit 1; }
probe_arm daily "$DAILY_API" "$DAILY_MODEL"

# --- arm 2: Flash-Next -----------------------------------------------------------------------------
log "stopping the daily"
sudo docker rm -f vllm-27b >/dev/null 2>&1 || true
for i in $(seq 24); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done
bash "$FN_L" >> "$R/audit.log" 2>&1 || { log "FLASHNEXT BOOT FAILED"; finish "ABORTED (flashnext)"; exit 1; }
curl -sf -m 8 "$FN_API/model" >/dev/null || { log "flashnext did not come up"; finish "ABORTED (flashnext)"; exit 1; }
probe_arm flashnext "$FN_API" "$FN_MODEL"

log "left serving: Flash-Next on 8022 (the configuration the user's harness points at). The daily is DOWN."
finish DONE
