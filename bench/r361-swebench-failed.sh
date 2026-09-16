#!/usr/bin/env bash
# R361 — the complement test: ten instances the DAILY FAILED. Does this seat pass them?
#
# WHY. r359 sampled the dataset's first ten (all astropy) and this seat resolved 10/10 where the daily resolved
# 8/10 — including both instances the daily failed. Two data points is not a signal, so this takes ten more from the
# daily's own failure set and asks the same question. If the seat resolves several of them, the two engines differ
# on this distribution rather than by luck; if it resolves none, r359's two were luck and the daily's failure set is
# simply hard for both.
#
# THE SELECTION IS FROM CAPABILITY FAILURES, NOT BUDGET ONES. The daily's 113 unresolved instances are 109
# `Submitted` (the agent patched and the tests failed), 3 `RepeatedFormatError` and 1 `ContextWindowExceededError`.
# Only the `Submitted` ones are a capability signal, and only those are eligible here. The first ten are taken in
# sorted order, minus the three already covered by r359 and r360, so this is a fresh sample:
#   astropy 14369, 14598, 7606 · django 10999, 11141, 11265, 11400, 11477, 11532, 11728
#
# The daily's expected score on these ten is 0/10 by construction. The number to read is this seat's, scored by the
# same official harness, plus whether each trajectory ended `Submitted` (a real attempt) or hit a limit.
#
# RUN: sudo systemd-run --unit=r361-swebench-failed --collect -p User=adrienbrault -p RuntimeMaxSec=21600 \
#        bash /srv/qwen5090/r361-swebench-failed.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r361-swebench-failed; mkdir -p "$R/out"
API=http://127.0.0.1:8022
MODEL=qwen3.8-flash-next-exl3-3.05bpw
CFG=/srv/qwen5090/miniswe/flashnext-local.yaml
VENV=/srv/qwen5090/miniswe/.venv
FILTER='^(astropy__astropy-14369|astropy__astropy-14598|astropy__astropy-7606|django__django-10999|django__django-11141|django__django-11265|django__django-11400|django__django-11477|django__django-11532|django__django-11728)$'
log(){ echo "$(date -Is) [r361] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r361-swebench
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R361 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

[ -x "$VENV/bin/mini-extra" ] || { log "ABORT: mini-extra missing"; exit 3; }
curl -sf -m 8 "$API/v1/model" >/dev/null || { log "ABORT: no seat on $API"; exit 3; }
BUILTIN=$("$VENV/bin/python" -c "from minisweagent.config import builtin_config_dir; print(builtin_config_dir/'benchmarks'/'swebench.yaml')" | tail -1)
log "seat: $(curl -s -m 5 "$API/v1/model" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])') image $(sudo docker inspect flashnext --format '{{.Config.Image}}')"
log "10 instances the daily SUBMITTED-and-FAILED: astropy 14369/14598/7606, django 10999/11141/11265/11400/11477/11532/11728"
log "the daily's score on these is 0/10 by construction; the seat's is the measurement"

( cd "$HOME" && timeout 18000 "$VENV/bin/mini-extra" swebench --subset verified --split test \
    --filter "$FILTER" -w 8 -o "$R/out" -c "$BUILTIN" -c "$CFG" > "$R/run.log" 2>&1 )
log "mini-extra exit code: $?"
python3 - "$R/out" <<'PY' | tee -a "$R/audit.log"
import json, glob, sys, collections
out = sys.argv[1]
preds = json.load(open(out + "/preds.json")) if glob.glob(out + "/preds.json") else {}
st = collections.Counter()
for f in glob.glob(out + "/*/*.traj.json"):
    try: st[json.load(open(f))["info"].get("exit_status")] += 1
    except Exception: st["<unreadable>"] += 1
print(f"  predictions={len(preds)} nonempty={sum(1 for v in preds.values() if v.get('model_patch'))} exits={dict(st)}")
PY

log "scoring with the official harness"
SCORE_WORKERS=6 bash /srv/qwen5090/miniswe-score.sh "$R" "fn-failed" 2>&1 \
  | grep -aE "predictions|OFFICIAL|resolved|error_ids|non-zero" | tee -a "$R/audit.log"

log "matched tally (daily's column is 0 for these by construction)"
python3 /srv/qwen5090/probes/match-daily-swebench.py "$R" 2>&1 | tee -a "$R/audit.log"
finish DONE
