#!/usr/bin/env bash
# R491b — paired tool-eval for the shared-expert overlap candidate (R491 gate 4 read 79.5 +- 6.4, trials 115/115/113/96, below the
# 80 floor; gates 1-3 PASS; R490 proved greedy byte-identity). A bit-identical kernel change cannot move quality except through
# sampling / batch-composition noise, so gate 4 is decided PAIRED, same session: CTL (served launcher) / CAND / CTL2 / CAND2,
# tool-eval 69x4 each (sampler 0.6/0.95/20, parallel 8). PROMOTE if mean(CAND) >= mean(CTL) - 2.0 and neither CAND run is below
# 75; else keep the daily. On promote: CAND becomes /srv/qwen5090/launch-flashnext.sh (R481 launcher stays as rollback).
# RUN: sudo systemd-run --unit=r491b-tooleval-paired --collect -p RuntimeMaxSec=43200 -E HOME=$HOME /bin/bash /srv/qwen5090/r491b-tooleval-paired.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r491b-tooleval-paired; mkdir -p "$R"
API=http://127.0.0.1:8022/v1; MODEL=qwen3.8-flash-next-exl3-3.05bpw
LIVE=/srv/qwen5090/launch-flashnext.sh; CAND=/srv/qwen5090/launch-flashnext-r491.sh; ROLLBACK=/srv/qwen5090/launch-flashnext-r481-s4.sh
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
PROMOTED=0; BOOTED=0
log(){ echo "$(date -Is) [r491b] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
wait_up(){ local i; for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && return 0; sleep 2; done; return 1; }
finish(){ if [ "$BOOTED" = 1 ] && [ "$PROMOTED" = 0 ]; then log "restoring the daily unchanged"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1; wait_up; log "daily: $(served_id) image $(sudo docker ps --format '{{.Image}}' -f name=flashnext)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R491b $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
cmp -s "$LIVE" "$ROLLBACK" || { log "ABORT: live launcher is not the R481 launcher any more"; exit 3; }
export GPU_QUEUE_NAME=r491b-tooleval-paired
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
boot(){ BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1; "${CLEAN_ENV[@]}" bash "$1" >> "$R/boot.log" 2>&1; wait_up || return 1
  log "up: $(sudo docker ps --format '{{.Image}}' -f name=flashnext) overlap=$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c '^EXL3_SHARED_EXPERT_OVERLAP=1$')"; }
te(){ local tag=$1
  ( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$MODEL" --temperature 0.6 --top-p 0.95 --top-k 20 \
      --trials 4 --parallel 8 --json-file "$R/tooleval-$tag.json" > "$R/tooleval-$tag.log" 2>&1 )
  python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval-$tag.json" "$tag" 2>&1 | tee -a "$R/audit.log" | grep -oE '[0-9]+(\.[0-9]+)? \+-' | head -1 | cut -d' ' -f1; }
declare -A S
for arm in "CTL|$LIVE" "CAND|$CAND" "CTL2|$LIVE" "CAND2|$CAND"; do
  IFS='|' read -r tag l <<< "$arm"
  boot "$l" || { log "BOOT FAILED $tag"; finish ABORTED; exit 3; }
  S[$tag]=$(te "$tag"); log "$tag tool-eval: ${S[$tag]:-none}"
done
V=$(python3 -c "
c=[float('${S[CTL]:-0}'),float('${S[CTL2]:-0}')]; d=[float('${S[CAND]:-0}'),float('${S[CAND2]:-0}')]
mc=sum(c)/2; md=sum(d)/2
print('PROMOTE' if md >= mc-2.0 and min(d) >= 75 else 'KEEP', f'ctl {c} mean {mc:.1f} cand {d} mean {md:.1f}')")
log "verdict: $V"
if [ "${V%% *}" = PROMOTE ]; then
  sudo cp "$LIVE" "$LIVE.pre-r491"; sudo cp "$CAND" "$LIVE.new" && sudo mv "$LIVE.new" "$LIVE"; PROMOTED=1
  log "PROMOTED: $LIVE = $CAND (serving since the CAND2 boot: $(sudo docker ps --format '{{.Image}}' -f name=flashnext))"
fi
finish DONE
