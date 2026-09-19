#!/usr/bin/env bash
# R375 — is the c8 TTFT halving from #246 real, or queueing noise?
#
# THE OBSERVATION. In r365's #246 arms, decode TTFT at c8 read control 1.249 s, patched-with-feature-OFF 1.200 s and
# patched-with-feature-ON 0.635 s -- half. Every prefill column in the same run moved by 1-2 %, so the enabled feature
# did something at c8 that it did nowhere else, on a column where queueing rather than kernel speed usually dominates.
# One run cannot tell those apart.
#
# WHY THIS IS ~10 MINUTES AND NOT 30.
#   * TWO ARMS, not three. The patched feature-OFF path is byte-identical to control and matched its TTFT to within
#     0.5 %, so it IS the control. Two model loads, not three.
#   * SHORT GENERATIONS. 512 forced tokens instead of 2048: the reading is TTFT, which does not need long decodes.
#   * THE IMAGES ARE ALREADY BUILT (kernel246:1). Nothing is compiled here.
#   * FOUR RUNS, which is what it takes to see a spread at c8 at all; a 2x effect needs few runs, a noise floor needs
#     the spread.
#
# It also re-captures greedy output from both arms, because the other half of the #246 question is whether the enabled
# path still changes numerics under repetition (it did, on all six responses, in r365).
#
# RUN: sudo systemd-run --unit=r375-246-c8 --collect -p User=adrienbrault -p RuntimeMaxSec=7200 \
#        bash /srv/qwen5090/r375-246-c8.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r375-246-c8; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
LIVE=/srv/qwen5090/launch-flashnext.sh
IMG=kernel246:1
log(){ echo "$(date -Is) [r375] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=r375-246-c8
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/greedy-compare.sh
WAIT_FOR=${WAIT_FOR:-r374-ourkernel r373-restore}
for u in $WAIT_FOR; do
  if systemctl is-active --quiet "$u"; then
    log "waiting for $u (poll 60 s) -- before taking the lock"
    while systemctl is-active --quiet "$u"; do sleep 60; done
  fi
done
gpu_lock
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R375 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: $IMG missing"; finish ABORTED; exit 3; }

for arm in off on; do
  flag=$([ "$arm" = on ] && echo 1 || echo 0)
  log "=== arm $arm (EXL3_MOE_ROUTE_PACKED=$flag) ==="
  if ! IMG="$IMG" EXTRA_ENV="EXL3_MOE_ROUTE_PACKED=$flag" bash "$LIVE" >> "$R/audit.log" 2>&1; then
    log "  boot FAILED"; continue
  fi
  curl -sf -m 8 "$API/model" >/dev/null || { log "  no server"; continue; }
  # TTFT is the column: 512 forced tokens, four runs, three concurrencies. The probe reports its own aggregate; the
  # per-run ttft_s values are what this run is about, so they are kept in the records file.
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "c8ttft-$arm" --kind code \
    --tokens 512 --conc 4 8 16 --runs 4 --out "$R/records-$arm.jsonl" 2>&1 \
    | grep -E "^  c=|FAILED" | tee -a "$R/audit.log"
  log "  greedy capture (does the enabled path still change numerics?)"
  python3 /srv/qwen5090/probes/hotvocab-greedy-capture.py --url "$API" --model "$MODEL" --out-dir "$R/greedy-$arm" \
    >> "$R/audit.log" 2>&1 || log "  capture FAILED"
done

log "=== TTFT by arm, per concurrency (median and full spread) ==="
python3 - "$R" <<'PY' | tee -a "$R/audit.log"
import json, sys, statistics, collections
R = sys.argv[1]
g = collections.defaultdict(dict)
for arm in ("off", "on"):
    try: rows = [json.loads(l) for l in open(f"{R}/records-{arm}.jsonl")]
    except FileNotFoundError: continue
    for d in rows:
        if not d.get("ok"): continue
        g[(d["conc"], d.get("ctx_requested"))].setdefault(arm, []).append(d["ttft_s"])
for k in sorted(g):
    off, on = g[k].get("off", []), g[k].get("on", [])
    if not off or not on:
        print(f"  c={k[0]:<3d} ctx={k[1]}: off={off} on={on}  (one arm missing)"); continue
    mo, mn = statistics.median(off), statistics.median(on)
    print(f"  c={k[0]:<3d} ctx={k[1]:<6}: off median {mo:.3f} {[round(x,3) for x in off]}")
    print(f"  {'':16s} on  median {mn:.3f} {[round(x,3) for x in on]}   delta {(mn/mo-1)*100:+.1f}%")
PY

log "=== output identity across arms ==="
if greedy_same "$R/greedy-off" "$R/greedy-on"; then
  log "  same (the enabled path no longer changes numerics -- contradicts r365; re-check before believing it)"
else
  log "  differ (as in r365: enabling the feature changes what the model says)"
fi

log "=== restoring the served configuration (live launcher, no override) ==="
bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE FAILED"
finish DONE
