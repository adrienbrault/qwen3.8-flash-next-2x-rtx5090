#!/usr/bin/env bash
# R377 — why does the hot-vocabulary "on" arm fail to boot? Ask docker.
#
# r364's patched-on arm printed "docker run FAILED" and nothing else, because the launcher's run command discarded stderr.
# That is now captured, so the same boot reports the reason. This unit also restores the served configuration afterwards.
set -uo pipefail
R=/srv/qwen5090/results/2026-09-16-r377-hotvocab-on; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
LIVE=/srv/qwen5090/launch-flashnext.sh
log(){ echo "$(date -Is) [r377] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=r377-hotvocab-on
. /srv/qwen5090/lib/gpu-queue.sh
for u in r373-restore r376-hotvocab; do
  if systemctl is-active --quiet "$u"; then log "waiting for $u"; while systemctl is-active --quiet "$u"; do sleep 30; done; fi
done
gpu_lock
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R377 $1 ==="; }
trap 'log "### SIGTERM"; finish ABORTED; exit 4' TERM

MAP=/srv/qwen5090/mtp-hot-blocks.txt
IMG=tabbyapi:qsa-cid-pr337-hotvocab
log "map: $(sudo ls -l "$MAP" 2>&1 | awk '{print $1, $5, $9}')"
log "image: $(sudo docker image inspect $IMG --format '{{.Id}}' 2>/dev/null | cut -c1-20)"
log "=== attempting the boot with docker's error captured ==="
IMG="$IMG" HOTVOCAB_MAP="$MAP" bash "$LIVE" >> "$R/audit.log" 2>&1
log "boot exit=$?"
log "--- docker's own words ---"
sudo tail -6 /srv/qwen5090/logs/flashnext-8022.log.docker 2>/dev/null | cut -c1-170 | tee -a "$R/audit.log"
log "=== restoring the served configuration ==="
bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE FAILED"
log "serving: $(sudo docker inspect flashnext --format '{{.Config.Image}}' 2>/dev/null)"
finish DONE
