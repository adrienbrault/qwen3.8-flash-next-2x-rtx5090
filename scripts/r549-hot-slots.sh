#!/usr/bin/env bash
# R549 — hot slots on the served daily (probes/hot_slots.py). Question from a reader of the public README: "217 t/s c1 and 510
# agg at c4 ... At those concurrency steps, does TTFT stay flat, or does the 8-bit KV pool start stalling once all four slots
# are hot?" Only short-prompt TTFT (c1 0.14 s, c4 0.45 s) and 3.05-era long-context c4 decode (R492) exist.
#   A short prompts c1..c4 x 2 runs; B ~30k prompts c1..c4 (all cold, distinct); C late arrival (short + 30k) with 0..3 slots
#   decoding ~150k contexts, hot-stream frame rate and longest gap during the arrival's prefill; D 4 x ~190k simultaneous
# Live launcher with NVME_TIER= (salted junk must not evict the daily's tier); fingerprints recorded. GPU TIMEBOX 30 min.
# RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r549-hot-slots; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
HS=/srv/qwen5090/probes/hot_slots.py
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r549] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){ if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R549 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$HS" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r549-hot-slots
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; live image $(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
"${CLEAN_ENV[@]}" NVME_TIER= bash "$LIVE" > "$R/boot.log" 2>&1 || { log "NO BOOT"; finish FAILED; exit 3; }
for i in $(seq 150); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
[ "$(served_id)" = "$MODEL" ] || { log "NO BOOT"; finish FAILED; exit 3; }
POOL=$(sudo sed -n 's/^  cache_size: \([0-9]*\).*/\1/p' $CFG)
log "UP: $(sudo grep -E '^  (cache_size|max_batch_size|cache_mode):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '); VRAM free $(vram)"
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits -lms 200 > "$R/smi.csv" 2>/dev/null & SMI=$!
timeout 1800 python3 "$HS" --url "$API" --model "$MODEL" --pool "$POOL" --out "$R/hot.jsonl" > "$R/hot.log" 2>&1; rc=$?
kill $SMI 2>/dev/null
log "probe exit $rc; $(wc -l < "$R/hot.jsonl") phase records"
python3 - "$R/hot.jsonl" <<'PY' 2>&1 | tee "$R/summary.txt" | tee -a "$R/audit.log"
import json,sys,statistics as st
f=lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v,float) else str(v))
for l in open(sys.argv[1]):
    r=json.loads(l); p=r["phase"]
    if p=="A": print(f"A c{r['conc']} run{r['run']}: TTFT {[f(x) for x in r['ttft']]} s, per-stream decode {r['decode_tps']}, aggregate {r['aggregate_tps']} t/s {r['errors'] or ''}")
    if p=="B": print(f"B c{r['conc']} x ~30k ({r['prompt_tokens']}): TTFT sorted {[f(x) for x in r['ttft']]} s {r['errors'] or ''}")
    if p=="C": print(f"C {r['hot']} hot x {r['hot_ctx']:,}: new short TTFT {f(r['short_ttft'])} s, decode {r['short_decode_tps']}; new 30k TTFT {f(r['ctx30k_ttft'])} s"
                     + (f"; hot frames/s before {r['hot_rate_before']} during-30k-prefill {r['hot_rate_during_30k_prefill']}, max gap before {r['hot_max_gap_before']} during {r['hot_max_gap_during_30k_prefill']} s" if r['hot'] else "")
                     + f" {r['errors'] or ''}")
    if p=="D": print(f"D 4 x {r['ctx_each']:,} ({r['prompt_tokens']}): TTFT sorted {[f(x) for x in r['ttft']]} s; all-hot window {r.get('all_hot_window_s')} s, per-stream decode {r.get('all_hot_decode_tps')}, max gap {r.get('all_hot_max_gap')}; finish {r['finish']} {r['errors'] or ''}")
PY
[ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext)" = 0 ] && log "server alive after the probe" || log "SERVER NOT ALIVE after the probe"
finish DONE
