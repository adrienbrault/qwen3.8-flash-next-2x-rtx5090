#!/usr/bin/env bash
# R360 — SWE-bench Verified, stratified: six repositories, sampled by the daily's own outcome.
#
# WHY NOT THE FIRST TEN. r359 took `--slice 0:10`, a contiguous slice in dataset order, and the dataset's first ten
# are ALL astropy. That answers "can this seat do the astropy tasks the daily could" and nothing about the other
# eleven repositories in this benchmark, which is the wrong shape for a comparison.
#
# THE SELECTION, AND WHY IT IS DEFENSIBLE. From the daily's full scored run (`2026-09-02-miniswe-rh-nvidia`: 500
# submitted, 387 resolved, 113 unresolved), the six repositories with the most resolved instances are taken, and
# from each: the first two instances the daily RESOLVED and the first one it did NOT. That gives
#   - repository diversity (6 against 1),
#   - a 2:1 resolved:unresolved split, so the subset contains instances where the daily failed and a difference can
#     show in either direction,
#   - exact matching: the daily's outcome is known per instance, so the comparison carries no sampling error —
#     only the subset's ability to represent the 500 does.
# n=18 gives ±11 % at 1σ on a pass rate, which is enough to separate "comparable to the daily" from "materially
# worse", and not enough to resolve three points. The selection is deterministic and is reproduced by the regex
# below rather than by a random draw, so the same 18 can be re-run.
#
# RUN: sudo systemd-run --unit=r360-swebench-strat --collect -p User=adrienbrault -p RuntimeMaxSec=21600 \
#        bash /srv/qwen5090/r360-swebench-strat.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r360-swebench-strat; mkdir -p "$R/out"
API=http://127.0.0.1:8022
MODEL=qwen3.8-flash-next-exl3-3.05bpw
CFG=/srv/qwen5090/miniswe/flashnext-local.yaml
VENV=/srv/qwen5090/miniswe/.venv
FILTER='^(django__django-10554|django__django-10880|django__django-10097|sympy__sympy-11618|sympy__sympy-12096|sympy__sympy-13091|sphinx-doc__sphinx-10323|sphinx-doc__sphinx-10449|sphinx-doc__sphinx-10435|scikit-learn__scikit-learn-10297|scikit-learn__scikit-learn-10844|scikit-learn__scikit-learn-12973|matplotlib__matplotlib-13989|matplotlib__matplotlib-14623|matplotlib__matplotlib-20826|pydata__xarray-2905|pydata__xarray-3095|pydata__xarray-3993)$'
log(){ echo "$(date -Is) [r360] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r360-swebench
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R360 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

[ -x "$VENV/bin/mini-extra" ] || { log "ABORT: mini-extra missing"; exit 3; }
curl -sf -m 8 "$API/v1/model" >/dev/null || { log "ABORT: no seat on $API"; exit 3; }
BUILTIN=$("$VENV/bin/python" -c "from minisweagent.config import builtin_config_dir; print(builtin_config_dir/'benchmarks'/'swebench.yaml')" | tail -1)
log "seat: $(curl -s -m 5 "$API/v1/model" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])') image $(sudo docker inspect flashnext --format '{{.Config.Image}}')"
log "18 instances across django, sympy, sphinx-doc, scikit-learn, matplotlib, pydata; 2 daily-resolved + 1 daily-unresolved each"
log "builtin: $BUILTIN"
log "cached sweb images: $(sudo docker images --format '{{.Repository}}' | grep -c 'sweb' || true)"

log "agents starting (filtered run, 8 workers)"
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
SCORE_WORKERS=6 bash /srv/qwen5090/miniswe-score.sh "$R" "fn-strat" 2>&1 \
  | grep -aE "predictions|OFFICIAL|resolved|error_ids|non-zero" | tee -a "$R/audit.log"

log "matched comparison against the daily"
python3 /srv/qwen5090/probes/match-daily-swebench.py "$R" 2>&1 | tee -a "$R/audit.log"
finish DONE
