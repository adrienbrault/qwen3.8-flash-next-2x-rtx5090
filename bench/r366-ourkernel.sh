#!/usr/bin/env bash
# R366 — the MoE decode envelope change written FOR this box's shapes (not a port), built and A/B'd.
#
# THE CHANGE. v1.5.0 flattens batch x query rows for the fused MoE decode kernel and caps the fast path at 8 rows.
# A depth-3 draft window is four rows per job, so c1 (4 rows) and c2 (8) take the cooperative path while **c4 (16)
# and c8 (32) fall back to the grouped path at every MoE block** — precisely this box's weakest shapes (250 t/s
# aggregate at c4, 313 at c8, against the vLLM daily's 868 and 1,574). The change raises that envelope to 32, grows
# the shared-expert graph capacity to match, and keeps the CUDA kernel's separate 256-slot structural bound by
# chunking token rows in C++. The arithmetic is untouched; dispatch and staging changed.
#
# DRAFT DEPTH IS PINNED OFF IN BOTH ARMS, deliberately: the concurrency-indexed policy would drop to depth 1 at c4/c8
# and the verification window would shrink to 2 rows, which is exactly the cliff under test. The measurement is the
# full-window case.
#
# THE FALSIFIER, FROM THE CHANGE'S OWN AUTHOR: warmed baseline/candidate c4, depth 3, forced 512-token decode,
# confirming 16-row dispatch. No repeatable decode-time improvement falsifies the expected win.
#
# BOTH ARMS CARRY THE QSA PATCH, as the author required — the control is tabbyapi:qsa-devel and the candidate is that
# image plus this change, so the only variable is the envelope.
#
# RUN: sudo systemd-run --unit=r366-ourkernel --collect -p User=adrienbrault -p RuntimeMaxSec=21600 \
#        bash /srv/qwen5090/r366-ourkernel.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r366-ourkernel; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
L=/srv/qwen5090/launch-flashnext-r340.sh
log(){ echo "$(date -Is) [r366] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r366-ourkernel
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/greedy-compare.sh   # greedy_same(): directory-safe identity checks
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R366 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

log "building the candidate on top of tabbyapi:qsa-devel (both arms therefore carry the QSA patch)"
if (cd /srv/qwen5090/ourkernel && sudo docker build --build-arg DEVEL_IMAGE=tabbyapi:qsa-devel \
      -f ourkernel-Dockerfile -t tabbyapi:ourkernel . >> "$R/build.log" 2>&1); then
  log "BUILD OK tabbyapi:ourkernel"
else
  log "BUILD FAILED — this is a codex job; tail follows"
  tail -10 "$R/build.log" | cut -c1-180 | sed 's/^/    /' | tee -a "$R/audit.log"
  finish ABORTED; exit 1
fi

arm(){  # arm <tag> <image>
  log "booting $1: IMG=$2 (DRAFT_POLICY pinned off: full depth-3 windows on purpose)"
  IMG="$2" DRAFT_POLICY='' bash "$L" >> "$R/audit.log" 2>&1 || { log "$1 BOOT FAILED"; return 1; }
  curl -sf -m 8 "$API/model" >/dev/null || { log "$1: no server"; return 1; }
  python3 /srv/qwen5090/probes/hotvocab-greedy-capture.py --url "$API" --model "$MODEL" --out-dir "$R/greedy-$1" \
    >> "$R/audit.log" 2>&1 || log "  capture FAILED"
  local f; f=$(ls "$R/greedy-$1"/* 2>/dev/null | head -1)
  [ -n "$f" ] && log "  $1 greedy: sha $(shasum -a 256 "$f" | cut -c1-16) ($(wc -c < "$f") bytes)"
  # The matrix the change's author specified: warmed, c1/c2/c4/c8, forced length, depth 3.
  for C in 1 2 4 8; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-c$C" --kind code \
       --tokens 512 --conc "$C" --runs 3 --out "$R/records-$1.jsonl" 2>&1 | grep -E "^  c=|FAILED" | tee -a "$R/audit.log"
  done
  # One deep arm, because the real workload is long-context.
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-deep" --kind code \
     --tokens 512 --ctx 30000 --conc 4 --runs 2 --distinct --out "$R/records-$1.jsonl" 2>&1 | grep -E "^  c=" | tee -a "$R/audit.log"
}

arm control   tabbyapi:qsa-devel
arm candidate tabbyapi:ourkernel

log "=== gates ==="
if greedy_same "$R/greedy-control" "$R/greedy-candidate"; then
  log "PASS control == candidate (greedy byte-identical): the change is dispatch-only"
else
  log "FAIL/NOT-RUN greedy differs — the change touches arithmetic after all; investigate before reading rates"
fi
log "=== the falsifier, side by side (decode per stream at the pinned depth 3) ==="
python3 /srv/qwen5090/probes/summarize.py "$R/records-control.jsonl" "$R/records-candidate.jsonl" 2>&1 | grep -E "^==|c=" | tee -a "$R/audit.log"

log "restoring the enabled configuration"
# No IMG override: restore to the launcher default, which is the served configuration.
bash "$L" >> "$R/audit.log" 2>&1 || log "RESTORE FAILED"
finish DONE
