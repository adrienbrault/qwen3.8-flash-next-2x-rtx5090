#!/usr/bin/env bash
# R353 — the daily's deep-context admission, re-measured with an instrument that reads vLLM's field names.
#
# WHY. r349 recorded 2 of 8 concurrent 38k-context requests as failures on the daily: a usage block, no text and no
# error. r352 captured the raw SSE and settled it — the frames arrive (157-179 per request) and the thinking is in
# a delta field called `reasoning`, not `reasoning_content`. With 512 forced tokens spent entirely on thinking,
# `content` is legitimately empty, so an instrument reading only `content` and `reasoning_content` sees nothing and
# blames the engine. The probes now read both names.
#
# This run re-measures the arm so the comparison against Flash-Next (8/8) is made with the same instrument on both
# sides, and keeps the daily's log because the previous attempt without it is what left this open.
#
# RUN: sudo systemd-run --unit=r353-daily-admit2 --collect -p User=adrienbrault -p RuntimeMaxSec=7200 \
#        bash /srv/qwen5090/r353-daily-admit2.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r353-daily-admit2; mkdir -p "$R"
FN_L=/srv/qwen5090/launch-flashnext-r340.sh
log(){ echo "$(date -Is) [r353] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r353-admit
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R353 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

STOP=1 bash "$FN_L" >/dev/null 2>&1 || true
log "booting the daily"
FORCE_RESTORE=1 bash /srv/qwen5090/daily-restore-retry.sh >> "$R/daily-boot.log" 2>&1 || { log "DAILY BOOT FAILED"; finish ABORTED; exit 1; }
DAILY_MODEL=$(curl -s -m 10 http://127.0.0.1:8020/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
log "daily serves: $DAILY_MODEL"
python3 /srv/qwen5090/probes/fn_bench.py --url http://127.0.0.1:8020/v1 --model "$DAILY_MODEL" --tag daily-admit2 \
   --kind code --tokens 512 --ctx 30000 --conc 8 --runs 1 --distinct --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"
sudo docker logs vllm-27b > "$R/daily-server-log.txt" 2>&1
log "daily log captured: $(wc -l < "$R/daily-server-log.txt") lines"

log "restoring Flash-Next"
sudo docker rm -f vllm-27b >/dev/null 2>&1 || true
for i in $(seq 24); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done
bash "$FN_L" >> "$R/audit.log" 2>&1 || log "FLASHNEXT BOOT FAILED"
finish DONE
