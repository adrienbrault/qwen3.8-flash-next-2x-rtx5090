#!/usr/bin/env bash
# R346 — the second chain: the steps the first chain owed, with the two broken gates repaired.
#
# WHY A SECOND CHAIN. The first chain's greedy-equality gates captured only `delta.content`, and with
# `reasoning: true` the first hundreds of tokens of this model arrive as `reasoning_content` — so both captures
# were empty and `cmp` reported them equal. A gate that passes on empty input is worse than no gate; both gates
# now keep reasoning and content in a fixed order and refuse to compare when either capture is implausibly short.
# The first chain was stopped mid-run rather than edited in place, because bash reads a running script
# incrementally and an edit can make it misparse what is left.
#
# ORDER. Cheapest and most certain first:
#   1. r340 ci-depth A/B with the repaired gate (three boots, ~15 min)
#   2. QSA control image, then r341 QSA A/B (~30 min, most likely to fail)
#   3. r345 pool test: shared prefix vs genuinely independent deep contexts (~15 min)
#   4. a clean soak, alone on the box this time (~15 min)
#   5. baseline restored as the served configuration
#
# RUN: sudo systemd-run --unit=r346-chain --collect -p User=adrienbrault -p RuntimeMaxSec=28800 \
#        bash /srv/qwen5090/r346-chain.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-16-r346-chain; mkdir -p "$R"
log(){ echo "$(date -Is) [r346] $*" | tee -a "$R/audit.log"; }
step(){  # step <label> <script>
  log "### $1: $2"
  if bash "$2" >> "$R/audit.log" 2>&1; then log "### $1 OK"; else log "### $1 FAILED (its own results dir holds why)"; fi
}

step ci-depth-ab      /srv/qwen5090/r340-ci-depth.sh

log "### waiting for the qsa-devel build before building its control twin"
for i in $(seq 240); do systemctl is-active --quiet qsa-build || break; sleep 15; done
if sudo docker image inspect tabbyapi:qsa-devel >/dev/null 2>&1; then
  if sudo docker image inspect tabbyapi:qsa-devel-control >/dev/null 2>&1; then
    log "### qsa control image already present"
  else
    log "### building the QSA control arm (APPLY_QSA=0) on the identical tree"
    if (cd /srv/qwen5090/docker && sudo docker build -f Dockerfile.tabbyapi-qsa --build-arg APPLY_QSA=0 \
          -t tabbyapi:qsa-devel-control . >> "$R/qsa-control-build.log" 2>&1); then
      log "### qsa control image OK"
    else
      log "### qsa control image FAILED"; tail -5 "$R/qsa-control-build.log" | cut -c1-160 | tee -a "$R/audit.log"
    fi
  fi
fi
if sudo docker image inspect tabbyapi:qsa-devel-control >/dev/null 2>&1 && sudo docker image inspect tabbyapi:qsa-devel >/dev/null 2>&1; then
  step qsa-ab /srv/qwen5090/r341-qsa-ab.sh
else
  log "### qsa-ab SKIPPED (one of the two images is missing)"
fi

step pool-test /srv/qwen5090/r345-pool.sh
step clean-soak /srv/qwen5090/r347-soak.sh

log "### restoring the Flash-Next baseline as the served configuration"
STOP=1 bash /srv/qwen5090/launch-flashnext-r340.sh >/dev/null 2>&1 || true
bash /srv/qwen5090/launch-flashnext-r340.sh >> "$R/audit.log" 2>&1 || log "BASELINE BOOT FAILED"
log "### CHAIN DONE"
