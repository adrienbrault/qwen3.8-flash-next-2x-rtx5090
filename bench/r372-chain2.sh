#!/usr/bin/env bash
# R372 — re-run the three steps of the programme that ran as EMPTY FILES.
#
# WHAT HAPPENED. Pushing updated scripts to the host, one loop did
#     for f in a b c; do ssh flan "sudo cat > /srv/qwen5090/$f && chmod +x ..."; done
# with no input redirection, so `cat >` truncated each target and read nothing. All three became 0 bytes, the
# verification I ran was `bash -n`, which an empty script passes, and the chain then executed them as "DONE in 0s":
# three experiments finished in the log and did no work at all. It is the same failure as GOTCHAS 12, with a
# consequence: silently skipped steps look exactly like completed ones.
#
# WHY A SEPARATE CHAIN. The main chain has moved on to the quality work (GSM8K at n=1319, then the 30-instance
# SWE-bench subset) and those are deliverables in their own right; interrupting them to re-insert three kernel and
# slot experiments would cost more than it saves. This waits for the main chain to be inactive, then runs the three
# steps in their original order. Each takes the GPU lock itself, and nothing else is queued behind it, so the
# sequence is deterministic.
#
# RUN: sudo systemd-run --unit=r372-chain2 --collect -p User=adrienbrault -p RuntimeMaxSec=86400 \
#        bash /srv/qwen5090/r372-chain2.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-16-r372-chain2; mkdir -p "$R"
WAIT_FOR=${WAIT_FOR:-r370c-chain}
log(){ echo "$(date -Is) [r372] $*" | tee -a "$R/audit.log"; }

# Integrity gate first: this whole unit exists because a script ran empty. Refuse to start if any step is not a real
# file, and say which one instead of discovering it from a "DONE in 0s" line three hours later.
steps=(r366-ourkernel.sh r364-hotvocab.sh r367-slots.sh)
bad=0
for s in "${steps[@]}"; do
  size=$(wc -c < "/srv/qwen5090/$s" 2>/dev/null || echo 0)
  if [ "$size" -lt 500 ]; then log "ABORT: /srv/qwen5090/$s is ${size} bytes — restore it from the repo first"; bad=1; fi
done
[ "$bad" = 0 ] || exit 3
for s in "${steps[@]}"; do bash -n "/srv/qwen5090/$s" || { log "ABORT: $s does not parse"; exit 3; }; done
log "integrity: all three steps are non-empty and parse"

log "waiting for $WAIT_FOR to finish (poll 60 s)"
while systemctl is-active --quiet "$WAIT_FOR"; do sleep 60; done
log "$WAIT_FOR is no longer active; starting"

for s in "${steps[@]}"; do
  t0=$(date +%s)
  log "### START $s"
  if timeout 10800 bash "/srv/qwen5090/$s" >> "$R/audit.log" 2>&1; then
    log "### DONE  $s in $(( $(date +%s) - t0 ))s"
  else
    log "### FAILED $s after $(( $(date +%s) - t0 ))s (rc=$?)"
  fi
done

log "### CHAIN2 COMPLETE — served image: $(sudo docker inspect flashnext --format '{{.Config.Image}}' 2>/dev/null)"
