#!/usr/bin/env bash
# R493 — exllamav3's host-RAM second-tier KV cache (`sysmem_kv_cache`, CPUPageCache: complete KV pages evicted from VRAM go to
# pinned system memory and are restored on prefix hits instead of re-prefilled; exllamav3 generator.py:128-131, pagetable.py
# 334-706). User 2026-09-18: "What about kv cache offloading to nvme?" — exllamav3 has no NVMe tier; the RAM tier is the
# existing mechanism and was never measured here. It does NOT enlarge the live pool (active sequences' KV stays in VRAM); it
# turns revisits of evicted prefixes (agent sessions) from a full prefill into a host-to-device copy.
# Arms: A = the served daily as found (tier 0, no reboot); B = the same launcher with SYS_KV=16384 (16 GiB pinned ≈ 1.06M
# tokens at 15.75 KiB/token). Each runs probes/revisit.py: 5 distinct ~90k-token sessions (450k > the 360,448 pool, so the
# first ones are evicted), then sessions 0 and 1 again; greedy 64 tokens; revisit TTFT and output identity vs the first answer.
# RUN: sudo systemd-run --unit=r493-host-kv-tier --collect -p RuntimeMaxSec=43200 -E HOME=$HOME /bin/bash /srv/qwen5090/r493-host-kv-tier.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-18-r493-host-kv-tier; mkdir -p "$R"
API=http://127.0.0.1:8022/v1; MODEL=qwen3.8-flash-next-exl3-3.05bpw; LIVE=/srv/qwen5090/launch-flashnext.sh; CFG=/srv/qwen5090/flashnext-config.yml
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r493] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
memavail(){ awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo; }
wait_up(){ local i; for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && return 0; sleep 2; done; return 1; }
finish(){ if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged live launcher)"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"; wait_up; log "daily: $(served_id) $(sudo grep -E '^  sysmem_kv_cache:' $CFG)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R493 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
[ -e /srv/qwen5090/probes/revisit.py ] || { log "ABORT: missing probes/revisit.py"; exit 3; }
export GPU_QUEUE_NAME=r493-host-kv-tier
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
[ "$(served_id)" = "$MODEL" ] || { log "ABORT: daily not serving"; finish ABORTED; exit 3; }
log "A: served as found: image $(sudo docker ps --format '{{.Image}}' -f name=flashnext) $(sudo grep -E '^  (cache_size|sysmem_kv_cache|sysmem_recurrent_cache):' $CFG | tr -s ' ' | tr '\n' ' ') MemAvailable $(memavail) GiB"
python3 /srv/qwen5090/probes/revisit.py --url "$API" --model "$MODEL" --tag noTier --out "$R/revisit.jsonl" --nonce 49301 2>&1 | tee -a "$R/audit.log"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
"${CLEAN_ENV[@]}" SYS_KV=16384 bash "$LIVE" >> "$R/boot-tier.log" 2>&1 || { log "BOOT FAILED (tier)"; finish ABORTED; exit 3; }
wait_up || { log "BOOT UNVERIFIED (tier)"; finish ABORTED; exit 3; }
log "B: tier 16 GiB: $(sudo grep -E '^  (cache_size|sysmem_kv_cache):' $CFG | tr -s ' ' | tr '\n' ' ') MemAvailable $(memavail) GiB"
python3 /srv/qwen5090/probes/revisit.py --url "$API" --model "$MODEL" --tag tier16G --out "$R/revisit.jsonl" --nonce 49302 2>&1 | tee -a "$R/audit.log"
log "B after: MemAvailable $(memavail) GiB; tier lines: $(sudo docker logs flashnext 2>&1 | grep -aiE 'tier|cpu page|sysmem' | tail -3 | tr '\n' ' ' | cut -c1-300)"
finish DONE
