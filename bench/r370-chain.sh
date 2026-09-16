#!/usr/bin/env bash
# R370 — the whole remaining program as ONE chain, in a fixed order.
#
# WHY A CHAIN. Seven units were queued on the same flock, and the order in which waiters acquire a flock is not
# defined. Worse, the queue stalled: a waiter sat in `locks_lock_inode_wai` while /proc/locks showed nobody holding
# the lock, so no unit ran and nothing would have advanced it. A single process that runs the steps in order, with a
# per-step timeout, cannot wedge and cannot be reordered.
#
# ORDER IS BY VALUE TO THE OPERATOR, not by convenience:
#   1  r363  enablement — PR #337's gate, build, SERVE the improved image, verify it. This is the thing that changes
#            what the box does for the user, so it goes first.
#   2  r365  the kernel ports, gated as their own assessments prescribed (#290 identity, #246 TTFT).
#   3  r366  the 32-row MoE decode envelope written FOR these shapes — the c4/c8 cliff.
#   4  r364  the hot-vocabulary experiment, disabled-path identity gate first.
#   5  r367  the slot ladder (8/12/16) plus a 12-agent admission arm.
#   6  r368  GSM8K at n=1319, tightening a +/-0.019 interval.
#   7  r369  SWE-bench, 30 instances across six repositories.
#
# Each step still takes the GPU lock itself, which is now uncontended, so the sequence is the chain's and nothing
# else can interleave. A step that fails or times out is logged and the chain moves on: one broken experiment must
# not cost the rest of the programme.
#
# RUN: sudo systemd-run --unit=r370-chain --collect -p User=adrienbrault -p RuntimeMaxSec=86400 \
#        bash /srv/qwen5090/r370-chain.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-16-r370-chain; mkdir -p "$R"
STEP_TIMEOUT=${STEP_TIMEOUT:-10800}
log(){ echo "$(date -Is) [r370] $*" | tee -a "$R/audit.log"; }

steps=(
  "r363-enable:/srv/qwen5090/r363-enable.sh"
  "r365-kernels:/srv/qwen5090/r365-kernels.sh"
  "r366-ourkernel:/srv/qwen5090/r366-ourkernel.sh"
  "r364-hotvocab:/srv/qwen5090/r364-hotvocab.sh"
  "r367-slots:/srv/qwen5090/r367-slots.sh"
  "r368-gsm8k-1319:/srv/qwen5090/r368-gsm8k-1319.sh"
  "r369-swebench-30:/srv/qwen5090/r369-swebench-30.sh"
)

for step in "${steps[@]}"; do
  name=${step%%:*}; script=${step##*:}
  [ -f "$script" ] || { log "MISSING $script — skipped"; continue; }
  t0=$(date +%s)
  log "### START $name (timeout ${STEP_TIMEOUT}s)"
  if timeout "$STEP_TIMEOUT" bash "$script" >> "$R/audit.log" 2>&1; then
    log "### DONE  $name in $(( $(date +%s) - t0 ))s"
  else
    rc=$?
    log "### FAILED/TIMEOUT $name after $(( $(date +%s) - t0 ))s (rc=$rc)"
    log "    its own results dir holds the detail; the chain continues"
  fi
done

log "### CHAIN COMPLETE — served configuration: $(sudo docker inspect flashnext --format '{{.Config.Image}}' 2>/dev/null)"
log "### policy line in config: $(sudo grep -c draft_num_tokens_by_batch /srv/qwen5090/flashnext-config.yml 2>/dev/null)"
