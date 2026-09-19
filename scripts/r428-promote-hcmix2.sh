#!/bin/bash
# R428 promotion boot: R428's own restore ran seconds before the promoted launcher (launch-flashnext-r428-hcmix2.sh) was moved
# into place, so the box still served the R425 stack. Takes the GPU lock, boots /srv/qwen5090/launch-flashnext.sh (env -i, like
# every experiment restore), and verifies image + the three EXTRA_ENV keys inside the container. No measurement.
set -u
R=/srv/qwen5090/results/2026-09-17-r428-hcmix2-stack-ab; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
WANT_IMG=tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap
log(){ echo "$(date -Is) [r428-promote] $*" | tee -a "$R/audit.log"; }
grep -q "IMG:-$WANT_IMG}" "$LIVE" || { log "ABORT: $LIVE does not default to $WANT_IMG"; exit 3; }
GPU_QUEUE_NAME=r428-promote
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
cur=$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)
log "lock held; serving $cur"
if [ "$cur" = "$WANT_IMG" ]; then log "already on $WANT_IMG"; else
  env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" >> "$R/promote-boot.log" 2>&1 || log "LAUNCHER FAILED (see promote-boot.log)"
  for i in $(seq 120); do curl -sf -m 5 http://127.0.0.1:8022/v1/model >/dev/null 2>&1 && break; sleep 2; done
fi
cur=$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)
envs=$(sudo docker exec flashnext env 2>/dev/null | grep -E "^EXL3_(HOST_GAP_REWIND|HC_MIX_V2|HC_MIX_V2_MIN_R)=" | sort | tr '\n' ' ')
log "serving $(curl -s -m 5 http://127.0.0.1:8022/v1/model | cut -c1-80) on $cur env: $envs"
[ "$cur" = "$WANT_IMG" ] && echo "$envs" | grep -q "EXL3_HC_MIX_V2=1" && echo "$envs" | grep -q "EXL3_HC_MIX_V2_MIN_R=1" && echo "$envs" | grep -q "EXL3_HOST_GAP_REWIND=1" && log "=== PROMOTED: mixer V2 stack live ===" || log "=== PROMOTION UNVERIFIED ==="
