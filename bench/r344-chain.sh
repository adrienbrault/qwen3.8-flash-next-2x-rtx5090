#!/usr/bin/env bash
# R344 — the chain: everything the 2026-09-16 session still owes, in a fixed order, on one GPU.
#
# WHY A CHAIN AND NOT QUEUED UNITS. Each step takes the same exclusive GPU flock, so running them as separate
# systemd units puts them in a queue whose order is not defined — and two of them have a real dependency:
# the QSA A/B cannot run until a CONTROL image exists (built with APPLY_QSA=0), which in turn must wait for the
# treatment build. A single process that executes the steps in order, each still taking the lock so nothing else
# can interleave, is deterministic where a set of waiting units is not.
#
# WHY THIS ORDER. Cheap read-only probes first (the depth ladder), then the two engine A/Bs, then the head to head
# against the vLLM daily — which is the slowest and the only step that has to touch the daily — and last the QSA
# pair, because its control image has to be built first and it is the most likely step to fail.
#
# RUN: sudo systemd-run --unit=r344-chain --collect -p User=adrienbrault -p RuntimeMaxSec=28800 \
#        bash /srv/qwen5090/r344-chain.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-16-r344-chain; mkdir -p "$R"
log(){ echo "$(date -Is) [r344] $*" | tee -a "$R/audit.log"; }
step(){  # step <label> <script>
  log "### $1: $2"
  if bash "$2" >> "$R/audit.log" 2>&1; then log "### $1 OK"; else log "### $1 FAILED (continuing; its own results dir holds why)"; fi
}

step depth-ladder       /srv/qwen5090/r343-depth.sh

step ci-depth-ab        /srv/qwen5090/r340-ci-depth.sh

step daily-headtohead   /srv/qwen5090/r342-daily-headtohead.sh

# QSA: the treatment image must exist, then its control twin is built on the identical tree so the compiler is not
# a confound between the arms.
log "### waiting for the qsa-devel build unit to finish before building the control arm"
for i in $(seq 240); do systemctl is-active --quiet qsa-build || break; sleep 15; done
if sudo docker image inspect tabbyapi:qsa-devel >/dev/null 2>&1; then
  log "qsa-devel present; building the control arm (APPLY_QSA=0)"
  if (cd /srv/qwen5090/docker && sudo docker build -f Dockerfile.tabbyapi-qsa --build-arg APPLY_QSA=0 \
        -t tabbyapi:qsa-devel-control . >> "$R/qsa-control-build.log" 2>&1); then
    log "### qsa control image OK"
  else
    log "### qsa control image FAILED (tail below)"; tail -5 "$R/qsa-control-build.log" | cut -c1-160 | tee -a "$R/audit.log"
  fi
else
  log "### qsa-devel never appeared: the QSA step is skipped, not attempted with a substituted toolchain"
fi

if sudo docker image inspect tabbyapi:qsa-devel-control >/dev/null 2>&1 && sudo docker image inspect tabbyapi:qsa-devel >/dev/null 2>&1; then
  step qsa-ab /srv/qwen5090/r341-qsa-ab.sh
else
  log "### qsa-ab SKIPPED (one of the two images is missing)"
fi

log "### restoring the Flash-Next baseline as the served configuration"
STOP=1 bash /srv/qwen5090/launch-flashnext-r340.sh >/dev/null 2>&1 || true
bash /srv/qwen5090/launch-flashnext-r340.sh >> "$R/audit.log" 2>&1 || log "BASELINE BOOT FAILED"
log "### CHAIN DONE"
