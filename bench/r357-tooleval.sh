#!/usr/bin/env bash
# R357 — tool-eval 69x4 on both the baseline and the promoted configuration.
#
# WHY. The daily's published tool-eval figure is 91 (69x4, R234). This stack has never been measured on it, so
# "Flash-Next can call tools" rests on one agent session and a four-check capability gate — good evidence, but not
# the instrument the daily is quoted on. And the two levers being promoted are documented as *greedy*
# byte-identical; tool-eval runs at the daily's sampler (0.6/0.95/20), where that guarantee does not apply, so the
# promoted configuration needs a behavioural check of its own rather than an inherited one.
#
# Invocation copied from the daily's own tool-eval runs (cyk-tooleval.sh): same CLI, same sampler, --trials 4 to
# match the published 69x4.
#
# The harness sends its own sampler parameters, so the server's preset fallbacks are not part of this measurement
# on either arm — that is deliberate: it makes the two arms, and the daily's published number, comparable.
#
# RUN: sudo systemd-run --unit=r357-tooleval --collect -p User=adrienbrault -p RuntimeMaxSec=21600 \
#        bash /srv/qwen5090/r357-tooleval.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r357-tooleval; mkdir -p "$R"
API=http://127.0.0.1:8022
MODEL=qwen3.8-flash-next-exl3-3.05bpw
L=/srv/qwen5090/launch-flashnext-r340.sh
log(){ echo "$(date -Is) [r357] $*" | tee -a "$R/audit.log"; }

command -v tool-eval-bench >/dev/null 2>&1 || { log "ABORT: tool-eval-bench not on PATH"; exit 3; }
export GPU_QUEUE_NAME=r357-tooleval
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R357 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

arm(){  # arm <tag> <image> <policy>
  local tag=$1 img=$2 pol=$3
  log "booting $tag: IMG=$img policy='${pol}'"
  IMG="$img" DRAFT_POLICY="$pol" bash "$L" >> "$R/audit.log" 2>&1 || { log "$tag BOOT FAILED"; return 1; }
  curl -sf -m 8 "$API/v1/model" >/dev/null || { log "$tag: no server"; return 1; }
  log "--- $tag tool-eval 69x4, sampler 0.6/0.95/20, parallel 8 ---"
  ( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "$API" --model "$MODEL" \
      --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 \
      --json-file "$R/tooleval-$tag.json" > "$R/tooleval-$tag.log" 2>&1 )
  local rc=$?
  log "$tag harness exit code: $rc"
  if [ -s "$R/tooleval-$tag.json" ]; then
    python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval-$tag.json" "$tag" 2>&1 | tee -a "$R/audit.log"
  else
    log "$tag: no JSON produced; tail of the run log follows"
    tail -5 "$R/tooleval-$tag.log" 2>/dev/null | cut -c1-160 | tee -a "$R/audit.log"
  fi
}

arm baseline tabbyapi:53da7919-rqcount ""
arm promoted tabbyapi:qsa-cid '[[2, 3], [8, 1]]'

log "restoring the baseline as the served configuration"
STOP=1 bash "$L" >/dev/null 2>&1 || true
bash "$L" >> "$R/audit.log" 2>&1 || log "BASELINE BOOT FAILED"
finish DONE
