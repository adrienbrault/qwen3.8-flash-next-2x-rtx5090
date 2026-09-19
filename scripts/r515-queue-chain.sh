#!/usr/bin/env bash
# R515 — the Flash-Next priority chain with a MUTABLE order. R512 fixed its order at start, so every reprioritisation needed a
# swap unit that stopped the chain at a unit boundary (R512b, R512c). This chain takes the GPU-exclusive lock once and, after each
# unit, pops the next script name from the queue file; editing the file reorders or extends the queue without stopping anything.
# Queue file: /srv/qwen5090/flashnext-queue.txt, one script name per line (without .sh), '#' comments and blank lines ignored.
# Edits must be atomic (write a .new and mv). A popped name is moved to flashnext-queue.done with its rc.
# RUN: sudo systemd-run --unit=r515-queue-chain --collect -p RuntimeMaxSec=604800 -E HOME=$HOME /bin/bash /srv/qwen5090/r515-queue-chain.sh
set -uo pipefail
Q=/srv/qwen5090/flashnext-queue.txt; D=/srv/qwen5090/flashnext-queue.done
L=/srv/qwen5090/results/2026-09-18-r512-priority-chain.log
log(){ echo "$(date -Is) [r515] $*" | tee -a "$L"; }
next(){ grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$Q" 2>/dev/null | head -1 | tr -d '[:space:]'; }
pop(){ local n=$1; awk -v n="$n" 'done==0 && $0 ~ "^[[:space:]]*"n"[[:space:]]*$" {done=1; next} {print}' "$Q" > "$Q.pop" && mv "$Q.pop" "$Q"; }
export GPU_QUEUE_NAME=r515-queue-chain
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; queue: $(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$Q" | tr '\n' ' ')"
while s=$(next); [ -n "$s" ]; do
  pop "$s"
  if [ ! -f "/srv/qwen5090/$s.sh" ]; then log "skip $s: missing /srv/qwen5090/$s.sh"; echo "$(date -Is) $s missing" >> "$D"; continue; fi
  log "start $s"
  bash "/srv/qwen5090/$s.sh"; rc=$?
  log "end $s (rc $rc)"; echo "$(date -Is) $s rc=$rc" >> "$D"
done
log "=== R515 queue empty ==="
