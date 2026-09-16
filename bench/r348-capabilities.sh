#!/usr/bin/env bash
# R348 — the capability gate: structured output, tools, vision, reasoning.
#
# Closes the one promotion row that was still "not measured". Run it on the served baseline, and keep the server's
# own log next to the results: a grammar filter that silently does nothing looks exactly like a model that
# happened to obey, and the log is what tells the two apart.
#
# RUN: sudo systemd-run --unit=r348-caps --collect -p User=adrienbrault -p RuntimeMaxSec=3600 \
#        bash /srv/qwen5090/r348-capabilities.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r348-capabilities; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
log(){ echo "$(date -Is) [r348] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r348-caps
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R348 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

curl -sf -m 8 "$API/model" >/dev/null || { log "ABORT: no server"; exit 3; }
log "served model: $(curl -s -m 5 $API/model | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')"

python3 /srv/qwen5090/probes/fn_capabilities.py --url "$API" --model "$MODEL" --out "$R/checks.jsonl" 2>&1 | tee -a "$R/audit.log"
rc=${PIPESTATUS[0]}
log "capability gate exit code = $rc (number of failed checks)"

# The grammar filter's own trace, so a pass cannot be confused with luck.
sudo docker logs --since 30m flashnext > "$R/server-log.txt" 2>&1
grep -aiE "grammar|json_schema|filter" "$R/server-log.txt" | tail -20 | tee -a "$R/audit.log"
finish "DONE (failures=$rc)"
[ "$rc" = 0 ] || exit "$rc"
