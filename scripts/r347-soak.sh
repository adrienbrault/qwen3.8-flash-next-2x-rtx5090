#!/usr/bin/env bash
# R347 — a clean soak, alone on the box.
#
# The gate suite's soak is unusable: its first five rounds read 62-66 t/s per stream at c4 and round six onwards
# read 40.5, because a native extension rebuild was compiling on the same host and starving the decode path's CPU.
# That is recorded as a gotcha rather than as stamina, and this run re-measures it with nothing else on the box.
#
# A soak is only meaningful against a fixed reference, so the first three rounds are the reference and the drift is
# reported as the ratio of the last five rounds to those.
#
# RUN: sudo systemd-run --unit=r347-soak --collect -p User=adrienbrault -p RuntimeMaxSec=10800 \
#        bash /srv/qwen5090/r347-soak.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r347-soak; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
log(){ echo "$(date -Is) [r347] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r347-soak
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R347 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

curl -sf -m 8 "$API/model" >/dev/null || { log "ABORT: no server"; exit 3; }
log "nothing else is running on this box during this soak (that is the point)"
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader | tee -a "$R/audit.log"

python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag soak-clean --kind code \
   --tokens 1024 --conc 4 --runs 40 --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"

nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader | tee -a "$R/audit.log"

# Drift: median of the last five rounds against the median of the first three, from the records themselves.
python3 - "$R/records.jsonl" <<'PY' | tee -a "$R/audit.log"
import json, statistics, sys
rows = [r for r in (json.loads(l) for l in open(sys.argv[1])) if r.get("ok")]
per_round = {}
for r in rows:
    per_round.setdefault(r["run"], []).append(r.get("decode_tps") or 0)
rounds = sorted(per_round)
med = {k: statistics.median(v) for k, v in per_round.items()}
head = statistics.median(med[k] for k in rounds[:3]) if len(rounds) >= 3 else None
tail = statistics.median(med[k] for k in rounds[-5:]) if len(rounds) >= 5 else None
print(f"rounds measured: {len(rounds)}  per-round decode median (t/s): "
      f"{', '.join(f'{k}:{med[k]:.1f}' for k in rounds)}")
if head and tail:
    print(f"drift: first-3 median {head:.1f} -> last-5 median {tail:.1f} t/s ({tail / head * 100:.1f}% of the start)")
PY
finish DONE
