#!/usr/bin/env bash
# R343 — the depth ladder that step 1/2 of the gate suite could not run.
#
# The first gate run passed `--ctx 0 30000 120000 240000` to a probe whose `--ctx` accepted one value, and the
# failure was invisible because the suite filtered its probe output through grep. Both are fixed: the probe takes a
# ladder, and the suite logs verbatim. This runs the ladder on its own so the depth table is not entangled with a
# suite re-run.
#
# Note on the requested depth: the filler's token estimate runs ~28% high (a "30000" rung really sends 38,283
# prompt tokens, recorded per request). The rungs are labels; `prompt_tokens` is the truth.
#
# RUN: sudo systemd-run --unit=r343-depth --collect -p User=adrienbrault -p RuntimeMaxSec=10800 \
#        bash /srv/qwen5090/r343-depth.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r343-depth; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
P="python3 /srv/qwen5090/probes/fn_bench.py --url $API --model $MODEL"
log(){ echo "$(date -Is) [r343] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r343-depth
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R343 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

curl -sf -m 8 "$API/model" >/dev/null || { log "ABORT: no server"; exit 3; }
log "GET /v1/model: $(curl -s -m 5 $API/model | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')"

# Decode and TTFT against depth, 1024 forced tokens so the ladder does not spend an hour on output.
log "=== depth ladder, code, 1024 forced, c1 (rungs are labels; prompt_tokens is what was sent) ==="
$P --tag depth-c1 --kind code --tokens 1024 --ctx 0 30000 120000 240000 --conc 1 --runs 2 \
   --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"

log "=== depth ladder, code, 1024 forced, c4 with distinct prefixes (does depth cost concurrency?) ==="
$P --tag depth-c4 --kind code --tokens 1024 --ctx 30000 120000 --conc 4 --runs 2 --distinct \
   --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"

finish DONE
