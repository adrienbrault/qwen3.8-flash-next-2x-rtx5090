#!/usr/bin/env bash
# R489b — determinism control for R489's gate 2: codex's model_decode_ids.py `single` scenario (512 tokens, cache 8192, 4 slots,
# max_history 3, vision, five served flags) on tabbyapi:gdn-state-r1 run OFF, OFF, ON, ON in fresh containers. OFF == OFF proves
# the harness is deterministic, so R489's OFF-vs-ON mismatch (token 87) is the patch; ON == ON says whether replay is at least
# self-consistent. The daily is restored at the end.
# RUN: sudo systemd-run --unit=r489b-determinism --collect -p RuntimeMaxSec=43200 -E HOME=$HOME /bin/bash /srv/qwen5090/r489b-determinism.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-18-r489b-determinism; mkdir -p "$R/g"; sudo chmod 777 "$R/g"
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw; IMG=tabbyapi:gdn-state-r1; LIVE=/srv/qwen5090/launch-flashnext.sh
API=http://127.0.0.1:8022/v1; MODEL=qwen3.8-flash-next-exl3-3.05bpw
BASE_ENV="EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1"
log(){ echo "$(date -Is) [r489b] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
BOOTED=0
finish(){ sudo docker rm -f r489b-test >/dev/null 2>&1
  if [ "$BOOTED" = 1 ]; then env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" >> "$R/audit.log" 2>&1
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R489b $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
export GPU_QUEUE_NAME=r489b-determinism
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1; sleep 5
EV=(); for kv in $BASE_ENV; do EV+=(-e "$kv"); done
for run in off-a off-b on-a on-b; do
  rp=$([ "${run%%-*}" = on ] && echo 1 || echo 0)
  sudo docker run --rm --name r489b-test --gpus all --ipc=host --entrypoint bash -v "$CKPT":/models:ro -v "$R/g":/results "${EV[@]}" -e EXL3_GDN_STATE_REPLAY=$rp "$IMG" \
    -lc "python3 /opt/gdn-state-r1/tests/model_decode_ids.py --model /models --cache-size 8192 --max-batch-size 4 --max-history 3 --vision --tokens 512 --scenario single --output /results/single-$run.json" \
    > "$R/g/single-$run.log" 2>&1 || log "$run exited non-zero: $(tail -1 "$R/g/single-$run.log" | cut -c1-200)"
done
python3 - "$R/g" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json, sys, os
d = sys.argv[1]
def ids(n): return json.load(open(os.path.join(d, f"single-{n}.json")))["token_ids"]
for a, b in (("off-a", "off-b"), ("on-a", "on-b"), ("off-a", "on-a")):
    try:
        x, y = ids(a), ids(b); diff = [i for i, (p, q) in enumerate(zip(x, y)) if p != q]
        print(f"{a} vs {b}: {'IDENTICAL' if not diff and len(x) == len(y) else f'{len(diff)} differ, first at {diff[0] if diff else None}'} ({len(x)} / {len(y)} tokens)")
    except Exception as e: print(f"{a} vs {b}: error {e}")
PY
finish DONE
