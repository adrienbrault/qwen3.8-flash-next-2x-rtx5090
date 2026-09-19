#!/usr/bin/env bash
# R376 — the hot-vocabulary A/B, on the rebased recipe.
#
# WHY IT EXISTS. r364's build failed and the failure read like a code problem: `exllamav3/generator/generator.py:
# FAILED` and `job.py: FAILED`. It wasn't. Reading that build log line by line, the patch applied — with offsets of 58
# lines, because CID's patch added lines above the hunks — and then the NEXT command in the same layer ran
# `sha256sum --check` against a manifest generated from a PRISTINE v1.5.0 tree, while the image carries v1.5.0 + CID.
# `FILE: FAILED` is sha256sum's output format, not patch's. Two files legitimately differed, `set -eu` aborted the
# layer, and the conclusion I first drew ("the port conflicts with CID") was wrong in mechanism though right that
# something about the overlap needed care.
#
# The rebased artifacts were verified offline before this ran: patch exit 0, zero failed hunks, seven checksums OK.
# This unit waits for the other units, runs the A/B, and restores through the live launcher.
#
# RUN: sudo systemd-run --unit=r376-hotvocab --collect -p User=adrienbrault -p RuntimeMaxSec=14400 \
#        bash /srv/qwen5090/r376-hotvocab.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-16-r376-hotvocab; mkdir -p "$R"
log(){ echo "$(date -Is) [r376] $*" | tee -a "$R/audit.log"; }

size=$(wc -c < /srv/qwen5090/r364-hotvocab.sh 2>/dev/null || echo 0)
[ "$size" -gt 500 ] || { log "ABORT: r364-hotvocab.sh is ${size} bytes"; exit 3; }
bash -n /srv/qwen5090/r364-hotvocab.sh || { log "ABORT: r364-hotvocab.sh does not parse"; exit 3; }
for f in /srv/qwen5090/hotvocab-rebase/out/Dockerfile.hotvocab-rebased \
         /srv/qwen5090/hotvocab-rebase/out/hotvocab-rebased.patch \
         /srv/qwen5090/hotvocab-rebase/out/hotvocab-rebased-sha256.txt \
         /srv/qwen5090/hotvocab-rebase/out/served-image-sha256.txt \
         /srv/qwen5090/hotvocab-rebase/util/build_mtp_hot_blocks.py; do
  [ -s "$f" ] || { log "ABORT: build input missing: $f"; exit 3; }
done
log "integrity: step parses, every build input present"

for u in r375-246-c8 r373-restore; do
  if systemctl is-active --quiet "$u"; then
    log "waiting for $u (poll 60 s) -- before taking the lock"
    while systemctl is-active --quiet "$u"; do sleep 60; done
  fi
done

t0=$(date +%s)
log "### START r364-hotvocab.sh (rebased recipe)"
if timeout 10800 bash /srv/qwen5090/r364-hotvocab.sh >> "$R/audit.log" 2>&1; then
  log "### DONE  r364-hotvocab.sh in $(( $(date +%s) - t0 ))s"
else
  log "### FAILED r364-hotvocab.sh after $(( $(date +%s) - t0 ))s"
fi
log "### COMPLETE — served image: $(sudo docker inspect flashnext --format '{{.Config.Image}}' 2>/dev/null)"
