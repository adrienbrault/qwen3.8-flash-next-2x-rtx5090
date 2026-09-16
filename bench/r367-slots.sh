#!/usr/bin/env bash
# R367 — the slot ladder: does more concurrency inside the fast decode path buy aggregate throughput?
#
# WHY. Two levers are already enabled and validated (QSA multi-job, concurrency-indexed draft depth), and a kernel
# change is in flight (the 32-row MoE envelope). But the one *config* lever never tested is the slot count. TabbyAPI
# derives 4 slots for a recurrent model, this seat has been served at `max_batch_size: 8`, and the incumbent daily
# runs 16 sequences. Aggregate throughput is the measured gap (250 t/s at c4, 313 at c8, against 868 and 1,574), so
# the question is whether more slots let more jobs share the fast path — or whether the layer split saturates and
# they only queue.
#
# WHAT IT COSTS. Slots are recurrent-state allocations: more of them consume VRAM that the page pool also wants. So
# each rung records VRAM free after boot, and a rung that cannot boot is itself the answer.
#
# ARMS: MAXBS 8 (served), 12, 16 — at c4, c8, c12, c16 with forced 512-token code decode, plus an admission arm of
# 12 concurrent jobs at ~20k context each to see whether extra slots admit real agent contexts or only queue them.
#
# RUN: sudo systemd-run --unit=r367-slots --collect -p User=adrienbrault -p RuntimeMaxSec=21600 \
#        bash /srv/qwen5090/r367-slots.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r367-slots; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
L=/srv/qwen5090/launch-flashnext-r340.sh
log(){ echo "$(date -Is) [r367] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r367-slots
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R367 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

for MAXBS in 8 12 16; do
  log "===== slots $MAXBS ====="
  if ! MAXBS=$MAXBS bash "$L" >> "$R/audit.log" 2>&1; then log "slots $MAXBS: BOOT FAILED"; continue; fi
  curl -sf -m 8 "$API/model" >/dev/null || { log "slots $MAXBS: no server"; continue; }
  log "  served config says max_batch_size: $(sudo grep 'max_batch_size' /srv/qwen5090/flashnext-config.yml | tr -d ' ')"
  log "  VRAM free MiB: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/')"
  # The ladder: concurrency up to and beyond the slot count, so a ceiling shows as flat aggregate.
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "slots$MAXBS-code" --kind code \
     --tokens 512 --conc 1 4 8 12 16 --runs 2 --out "$R/records-$MAXBS.jsonl" 2>&1 | grep -E "^  c=|FAILED" | tee -a "$R/audit.log"
  # Admission of real agent-sized contexts at the new ceiling.
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "slots$MAXBS-admit" --kind code \
     --tokens 512 --ctx 20000 --conc 12 --runs 1 --distinct --out "$R/records-$MAXBS.jsonl" 2>&1 | grep -E "^  c=|FAILED" | tee -a "$R/audit.log"
  log "  VRAM free after the arms: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/')"
done

log "=== ladder, all rungs side by side ==="
python3 /srv/qwen5090/probes/summarize.py "$R"/records-*.jsonl 2>&1 | grep -E "^==|^slots" | tee -a "$R/audit.log"

log "restoring the served configuration (slots 8) with the enabled levers"
MAXBS=8 IMG=tabbyapi:qsa-cid bash "$L" >> "$R/audit.log" 2>&1 || log "RESTORE FAILED"
finish DONE
