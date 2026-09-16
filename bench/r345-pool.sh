#!/usr/bin/env bash
# R345 — the pool question, asked properly: shared prefix vs genuinely independent deep contexts.
#
# WHY THIS EXISTS. The first admission run varied only a suffix between concurrent requests, so all eight jobs
# carried the same 38k-token filler and no page could be attributed to "independent context footprint". It said
# nothing about how much unique context the 262k-token pool can hold, and its MEASUREMENTS.md entry claimed more
# than the run proved. `--unique` rebuilds the filler per request from a different RNG stream, so two requests
# share no token sequence at all.
#
# The server log is captured into the results directory as the run happens: `cached_tokens` per request is the
# direct evidence of whether pages were shared, and it is not inferred from timings.
#
# RUN: sudo systemd-run --unit=r345-pool --collect -p User=adrienbrault -p RuntimeMaxSec=10800 \
#        bash /srv/qwen5090/r345-pool.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r345-pool; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
P="python3 /srv/qwen5090/probes/fn_bench.py --url $API --model $MODEL --kind code"
log(){ echo "$(date -Is) [r345] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r345-pool
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R345 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

curl -sf -m 8 "$API/model" >/dev/null || { log "ABORT: no server on $API"; exit 3; }
log "served model: $(curl -s -m 5 $API/model | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')"

# Arm 1: the realistic agent case — eight jobs sharing one long system/tool prefix, each with a short unique tail.
# This is what a fan-out of agents on the same harness actually sends.
log "=== 1. SHARED prefix: 8 x 30k-context jobs, 512 forced tokens ==="
$P --tag shared-8x30k --tokens 512 --ctx 30000 --conc 8 --runs 1 --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"

# Arm 2: eight genuinely independent contexts of the same size. If the pool cannot hold them, this is where the
# engine queues or rejects rather than sharing pages.
log "=== 2. UNIQUE contexts: 8 x 30k-context jobs, 512 forced tokens ==="
$P --tag unique-8x30k --unique --tokens 512 --ctx 30000 --conc 8 --runs 1 --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"

# Arm 3: fewer, deeper independent contexts — 4 x 120k — which is the shape that should approach the pool limit.
log "=== 3. UNIQUE contexts: 4 x 120k-context jobs, 512 forced tokens ==="
$P --tag unique-4x120k --unique --tokens 512 --ctx 120000 --conc 4 --runs 1 --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"

# The server's own cache accounting for the whole run, kept as evidence rather than inferred.
sudo docker logs --since 60m flashnext > "$R/server-log.txt" 2>&1
log "server log captured: $(wc -l < "$R/server-log.txt") lines"
grep -cE "% cached" "$R/server-log.txt" | xargs -I{} log "requests with a cache figure: {}"
finish DONE
