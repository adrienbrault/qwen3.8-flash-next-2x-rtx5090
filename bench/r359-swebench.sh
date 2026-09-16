#!/usr/bin/env bash
# R359 — SWE-bench Verified, a subset, on the Flash-Next seat: the workload the box is actually used for.
#
# WHY. GSM8K and tool-eval bracket this seat six points behind the daily on two instruments. Neither is the job:
# the daily's headline capability is agentic coding — SWE-bench Verified, 387/500 (77.4%) at k=1. This runs the same
# harness, the same builtin `benchmarks/swebench.yaml` (leaderboard bash-only, step_limit 250) and the same limits
# against :8022 with an overlay identical to the daily's except for the endpoint.
#
# A SUBSUET, AND WHY IT IS STILL WORTH RUNNING. The repository's own note says k=1 single-task flips carry coin-flip
# variance, so a subset cannot resolve 3 points. It can resolve 20: if this seat lands near the daily's 77%, the
# six-point gaps elsewhere are a consistent picture; if it lands near 50%, the seat is materially worse at the job
# the box exists for, which no synthetic probe would have shown.
#
# Scoring is the official swebench harness in the official task images (miniswe-score.sh), cache_level=instance.
#
# RUN: sudo systemd-run --unit=r359-swebench --collect -p User=adrienbrault -p RuntimeMaxSec=21600 \
#        bash /srv/qwen5090/r359-swebench.sh [N]     # default 10
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
N=${1:-10}
R=/srv/qwen5090/results/2026-09-16-r359-swebench-$N; mkdir -p "$R/out"
API=http://127.0.0.1:8022
MODEL=qwen3.8-flash-next-exl3-3.05bpw
CFG=/srv/qwen5090/miniswe/flashnext-local.yaml
VENV=/srv/qwen5090/miniswe/.venv
log(){ echo "$(date -Is) [r359] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r359-swebench
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R359 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

[ -x "$VENV/bin/mini-extra" ] || { log "ABORT: mini-extra missing"; exit 3; }
[ -f "$CFG" ] || { log "ABORT: overlay missing"; exit 3; }
curl -sf -m 8 "$API/v1/model" >/dev/null || { log "ABORT: no seat on $API"; exit 3; }
log "seat: $(curl -s -m 5 "$API/v1/model" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])') image $(sudo docker inspect flashnext --format '{{.Config.Image}}')"
log "slice 0:$N, 8 workers, sampler from the seat's preset (0.6/0.95/20), max_tokens 32768, step_limit 250"
# `tail -1`: importing minisweagent prints a version banner to stdout before anything else, and capturing that as
# part of the path made every config lookup fail with a filename containing the banner (the first run's error).
BUILTIN=$("$VENV/bin/python" -c "from minisweagent.config import builtin_config_dir; print(builtin_config_dir/'benchmarks'/'swebench.yaml')" | tail -1)
log "builtin: $BUILTIN"

# The slice's task images: report how many are already cached, because a missing image turns into an agent
# timeout rather than a clear error inside mini-swe.
log "cached sweb task images: $(sudo docker images --format '{{.Repository}}' | grep -c 'sweb' || true)"

log "agents starting"
( cd "$HOME" && timeout 14400 "$VENV/bin/mini-extra" swebench --subset verified --split test --slice "0:$N" \
    -w 8 -o "$R/out" -c "$BUILTIN" -c "$CFG" > "$R/run.log" 2>&1 )
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

log "scoring with the official harness (official task images, cache_level=instance)"
SCORE_WORKERS=6 bash /srv/qwen5090/miniswe-score.sh "$R" "fn-subset" 2>&1 \
  | grep -aE "predictions|OFFICIAL|resolved|error_ids|non-zero" | tee -a "$R/audit.log"
ls -1 "$R" | grep -E "\.json$" | head -5 | sed 's/^/  result file: /' | tee -a "$R/audit.log"
finish DONE
