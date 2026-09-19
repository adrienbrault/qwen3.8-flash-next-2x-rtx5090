#!/bin/bash
# R442 promotion boot: verifies (and boots if needed) the prefill-pipeline daily (launch-flashnext-r442-ppipe.sh); queued behind R443,
# whose own restore already boots the promoted launcher. Takes the GPU lock, boots /srv/qwen5090/launch-flashnext.sh (env -i, like
# every experiment restore), and verifies image + the three EXTRA_ENV keys inside the container. No measurement.
set -u
R=/srv/qwen5090/results/2026-09-17-r442-ppipe-memfix-ab; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
WANT_IMG=tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap-ppipe-nosync-mtpfix2
log(){ echo "$(date -Is) [r442-promote] $*" | tee -a "$R/audit.log"; }
grep -q "IMG:-$WANT_IMG}" "$LIVE" || { log "ABORT: $LIVE does not default to $WANT_IMG"; exit 3; }
GPU_QUEUE_NAME=r442-promote
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
cur=$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)
log "lock held; serving $cur"
if [ "$cur" = "$WANT_IMG" ]; then log "already on $WANT_IMG"; else
  env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" >> "$R/promote-boot.log" 2>&1 || log "LAUNCHER FAILED (see promote-boot.log)"
  for i in $(seq 120); do curl -sf -m 5 http://127.0.0.1:8022/v1/model >/dev/null 2>&1 && break; sleep 2; done
fi
cur=$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)
# the "LS prefill pipeline" layout line is printed at the FIRST pipelined prefill, not at boot: send one 30k prompt first
python3 - <<'PY' | tee -a "$R/audit.log"
import json,random,time,urllib.request
rng=random.Random(time.time_ns()%100000)
words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
msg=" ".join(rng.choice(words) for _ in range(23000))+"\n\nReply with one word."
req=json.dumps({"model":"qwen3.8-flash-next-exl3-3.05bpw","temperature":0,"max_tokens":1,"messages":[{"role":"user","content":msg}]}).encode()
t=time.time(); urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8022/v1/chat/completions",data=req,headers={"Content-Type":"application/json"}),timeout=600).read()
print("[r442-promote] 30k uncached prefill wall: %.2f s (R442 base 5.5, pipeline 3.5)"%(time.time()-t))
PY
envs=$(sudo docker exec flashnext env 2>/dev/null | grep -E "^EXL3_(HOST_GAP_REWIND|HC_MIX_V2|HC_MIX_V2_MIN_R|LS_PREFILL_PIPELINE)=" | sort | tr "\n" " ")
log "serving $(curl -s -m 5 http://127.0.0.1:8022/v1/model | cut -c1-80) on $cur env: $envs"
[ "$cur" = "$WANT_IMG" ] && echo "$envs" | grep -q "EXL3_HC_MIX_V2=1" && echo "$envs" | grep -q "EXL3_HC_MIX_V2_MIN_R=1" && echo "$envs" | grep -q "EXL3_HOST_GAP_REWIND=1" && echo "$envs" | grep -q "EXL3_LS_PREFILL_PIPELINE=1" && sudo docker logs flashnext 2>&1 | grep -aq "LS prefill pipeline" && log "=== PROMOTED: prefill pipeline stack live ===" || log "=== PROMOTION UNVERIFIED ==="
