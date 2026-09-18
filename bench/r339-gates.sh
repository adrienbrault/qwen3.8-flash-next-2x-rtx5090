#!/usr/bin/env bash
# R339 — the gate suite for the Flash-Next TabbyAPI instance, after the R338 fixes.
#
# WHY. R329-R336 measured this engine at 128-512 output tokens per request, which is below TabbyAPI's ~2048-token
# requeue boundary and does not exercise a real agent turn. The R338 sampler fix also changed the decode regime
# (draft acceptance 18% -> 46-72%), so the older numbers describe a configuration nobody serves any more. This
# suite measures what the served stack actually does at generation lengths, depths and concurrencies a DSH
# session reaches.
#
# RUN (probe only, no restart, but it holds the GPU lock so another engine's restore cannot start mid-measurement):
#   sudo systemd-run --unit=r339-gates --collect -p User=adrienbrault -p RuntimeMaxSec=28800 \
#     bash /srv/qwen5090/r339-gates.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r339-gates; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
P="python3 /srv/qwen5090/probes/fn_bench.py --url $API --model $MODEL"
log(){ echo "$(date -Is) [r339] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r339-gates
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R339 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

curl -sf -m 5 "$API/models" >/dev/null || { log "ABORT: no server on $API"; exit 3; }
log "server up: $(curl -s -m 5 $API/model | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')"

# 1. Decode and TTFT against prompt depth. Same forced length at every rung so the only variable is the KV depth.
log "=== 1. depth ladder, code, 1024 forced tokens, c1 ==="
$P --tag depth-code --kind code --tokens 1024 --ctx 0 30000 120000 240000 --conc 1 --runs 2 \
   --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"

log "=== 2. depth ladder, code, 1024 forced tokens, c4 (does depth cost concurrency?) ==="
$P --tag depth-code-c4 --kind code --tokens 1024 --ctx 30000 120000 --conc 4 --runs 2 --distinct \
   --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"

# 3. Long-context retrieval. A pass at one depth is not a gate: the fact is planted at five fractions per depth.
log "=== 3. needle, ctx 32768/131072/262144 ==="
python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$MODEL" --tag needle \
   --ctx-tokens 32768 131072 196608 --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" \
   2>&1 | tee -a "$R/audit.log"

# 4. Admission: eight DISTINCT 30k-context jobs at once. cache_size is 262144 tokens, so the arithmetic says they
#    do not all fit; what this step records is whether the surplus queues or is rejected.
log "=== 4. admission, 8 x distinct 30k-context jobs, 512 forced tokens ==="
$P --tag admit-8x30k --kind code --tokens 512 --ctx 30000 --conc 8 --runs 1 --distinct \
   --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"

# 5. Code vs prose at the same shape, because decode is content-dependent through MTP acceptance.
log "=== 5. code vs prose, 2048 forced tokens, c1 and c4 ==="
$P --tag ab-code-2048 --kind code --tokens 2048 --conc 1 4 --runs 2 --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"
$P --tag ab-prose-2048 --kind prose --tokens 2048 --conc 1 4 --runs 2 --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"

# 6. Stamina: a sustained c4 load. Records decode-rate decay, error rate and VRAM drift per round, so the result
#    is a per-round series rather than an impression.
log "=== 6. soak, c4, 1024 forced tokens, 30 rounds ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$R/audit.log"
$P --tag soak-c4 --kind code --tokens 1024 --conc 4 --runs 30 --out "$R/soak.jsonl" 2>&1 | tee -a "$R/audit.log"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$R/audit.log"

finish DONE
