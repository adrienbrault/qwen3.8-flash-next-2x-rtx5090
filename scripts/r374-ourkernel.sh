#!/usr/bin/env bash
# R374 — re-run the our-kernel envelope A/B, whose build failed on a missing build-context path.
#
# WHAT HAPPENED. r366's build stopped at
#     COPY out/ourkernel-applied.patch out/ourkernel-source-sha256.json /opt/ourkernel/
#     COPY failed: file not found in build context or excluded by .dockerignore: stat out/ourkernel-applied.patch
# The host layout was flat (/srv/qwen5090/ourkernel/ourkernel-applied.patch) while the Dockerfile -- unchanged from the
# worktree where the port was assessed -- expects the worktree's `out/` layout. The artifacts are now in both places,
# so the assessed Dockerfile stays byte-identical instead of being edited to match a directory I created.
#
# This is the same defect class the day kept producing: the build failed *for a reason unrelated to the code under
# test*, and had it failed one step later it would have been read as "the kernel does not compile".
#
# WAITS for the other two units rather than racing them, then runs the step and restores through the live launcher.
#
# RUN: sudo systemd-run --unit=r374-ourkernel --collect -p User=adrienbrault -p RuntimeMaxSec=28800 \
#        bash /srv/qwen5090/r374-ourkernel.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-16-r374-ourkernel; mkdir -p "$R"
log(){ echo "$(date -Is) [r374] $*" | tee -a "$R/audit.log"; }

for s in /srv/qwen5090/r366-ourkernel.sh; do
  size=$(wc -c < "$s" 2>/dev/null || echo 0)
  [ "$size" -gt 500 ] || { log "ABORT: $s is ${size} bytes"; exit 3; }
  bash -n "$s" || { log "ABORT: $s does not parse"; exit 3; }
done
for f in /srv/qwen5090/ourkernel/out/ourkernel-applied.patch /srv/qwen5090/ourkernel/out/ourkernel-source-sha256.json \
         /srv/qwen5090/ourkernel/ourkernel-Dockerfile; do
  [ -s "$f" ] || { log "ABORT: build input missing: $f"; exit 3; }
done
log "integrity: the step parses and every build input exists"

for u in r372-chain2 r373-restore; do
  if systemctl is-active --quiet "$u"; then
    log "waiting for $u (poll 60 s)"
    while systemctl is-active --quiet "$u"; do sleep 60; done
  fi
done

t0=$(date +%s)
log "### START r366-ourkernel.sh"
if timeout 10800 bash /srv/qwen5090/r366-ourkernel.sh >> "$R/audit.log" 2>&1; then
  log "### DONE  r366-ourkernel.sh in $(( $(date +%s) - t0 ))s"
else
  log "### FAILED r366-ourkernel.sh after $(( $(date +%s) - t0 ))s"
fi
log "### COMPLETE — served image: $(sudo docker inspect flashnext --format '{{.Config.Image}}' 2>/dev/null)"
