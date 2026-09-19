#!/usr/bin/env bash
# R341 — QSA sparse multi-job: control vs patched, at the concurrency the patch exists for.
#
# WHY. Above its sparse threshold, the QSA captured attention path is single-job: `BCAttn.step` returns None for
# bsz>1 and falls back to eager. Concurrent deep-context decode is therefore the case this patch changes — and it
# changes attention, so correctness is the first gate, not speed.
#
# ARMS. Both built from the same devel base with the same pip resolution and the same native rebuild
# (`Dockerfile.tabbyapi-qsa`, APPLY_QSA=0/1), so the compiler is not a confound. The control is the arm built with
# APPLY_QSA=0 on the identical tree.
#
# GATES, in order:
#   1. the engine marker is present only in the treatment arm (proves which code the process imported);
#   2. two CONCURRENT greedy requests with the same 120k-token context must produce identical text within an arm
#      — if batching itself perturbs greedy output, the arm's batching is not deterministic and no cross-arm
#      comparison means anything;
#   3. the same text must match across arms — the patch must not change what the model says.
# A faster wrong answer is a failure, so the diff runs before the throughput table is read.
#
# RUN: sudo systemd-run --unit=r341-qsa --collect -p User=adrienbrault -p RuntimeMaxSec=14400 \
#        bash /srv/qwen5090/r341-qsa.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r341-qsa; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
L=/srv/qwen5090/launch-flashnext-r340.sh
log(){ echo "$(date -Is) [r341] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r341-qsa
. /srv/qwen5090/lib/gpu_queue.sh 2>/dev/null || . /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R341 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

boot(){  # boot <tag> <image>
  log "booting $1: IMG=$2"
  IMG="$2" bash "$L" >> "$R/audit.log" 2>&1 || { log "$1 BOOT FAILED"; return 1; }
  curl -sf -m 8 "$API/model" >/dev/null || { log "$1: no server"; return 1; }
  sudo docker exec flashnext python3 - <<'PY' >> "$R/audit.log" 2>&1
import exllamav3, inspect, pathlib
p = pathlib.Path(inspect.getfile(exllamav3)).parent / "modules" / "attention_fn" / "bc_attn.py"
print("QSA marker present:", "BC-attn QSA slot layer" in p.read_text())
PY
}

# Two concurrent greedy requests, identical 120k-token context. --distinct is OFF on purpose: the two requests
# must be the same work so their outputs are comparable, both within the arm and across arms.
deep_pair(){  # deep_pair <prefix>
  python3 /srv/qwen5090/probes/deep_pair.py --url "$API" --model "$MODEL" --ctx 120000 --tokens 256 \
     --conc 2 --out-prefix "$R/$1" 2>&1 | tee -a "$R/audit.log" || log "deep_pair FAILED for $1"
}

probe(){ python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --kind code --tokens 1024 \
           --ctx 120000 --conc 2 4 --runs 1 --out "$R/records-$1.jsonl" --tag "$1" 2>&1 | tee -a "$R/audit.log"; }

boot control tabbyapi:qsa-devel-control || finish "ABORTED (control boot)"
deep_pair greedy-control
probe control

boot treatment tabbyapi:qsa-devel || finish "ABORTED (treatment boot)"
deep_pair greedy-treatment
probe treatment

log "=== gate 2: within-arm determinism of concurrent greedy output ==="
for arm in control treatment; do
  if cmp -s "$R/greedy-$arm.0.txt" "$R/greedy-$arm.1.txt"; then log "PASS $arm: the two concurrent outputs match"
  else log "FAIL $arm: concurrent greedy outputs differ ($(cmp -l "$R/greedy-$arm.0.txt" "$R/greedy-$arm.1.txt" | wc -l | tr -d ' ') bytes) — batching perturbs greedy output in this arm"; fi
done

log "=== gate 3: cross-arm equality ==="
if cmp -s "$R/greedy-control.0.txt" "$R/greedy-treatment.0.txt"; then log "PASS control == treatment"
else log "FAIL control != treatment ($(cmp -l "$R/greedy-control.0.txt" "$R/greedy-treatment.0.txt" | wc -l | tr -d ' ') differing bytes)"; fi

# Back to the served baseline image, whatever the outcome.
boot baseline tabbyapi:53da7919-rqcount || log "WARNING: baseline image did not boot"
finish DONE
