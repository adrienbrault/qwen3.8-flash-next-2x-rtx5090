#!/usr/bin/env bash
# R351 — the two remaining measurements, in order, so they cannot interleave with each other's reboots.
#
#   1. the capability gate (structured output, tools, vision, reasoning) against the served baseline
#   2. the daily's deep-context admission, re-run with its server log captured before teardown
#
# They are one unit rather than two because r349 stops Flash-Next to boot the daily: if r348 happened to run
# inside that window it would probe an endpoint that is not there and fail for the wrong reason.
#
# Deliberately NO image build here: a native rebuild compiling beside a measurement takes c4 per-stream decode
# from 64.9 to 40.5 t/s on this box (recorded in the repo's GOTCHAS.md). Builds run when nothing is measuring.
#
# RUN: sudo systemd-run --unit=r351-last --collect -p User=adrienbrault -p RuntimeMaxSec=14400 \
#        bash /srv/qwen5090/r351-last.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-16-r351-last; mkdir -p "$R"
log(){ echo "$(date -Is) [r351] $*" | tee -a "$R/audit.log"; }

for step in "capabilities:/srv/qwen5090/r348-capabilities.sh" "daily-admit:/srv/qwen5090/r349-daily-admit.sh"; do
  label=${step%%:*}; script=${step##*:}
  log "### $label: $script"
  if bash "$script" >> "$R/audit.log" 2>&1; then log "### $label OK"; else log "### $label FAILED (its own results dir holds why)"; fi
done
log "### DONE"
