#!/usr/bin/env bash
# R554 — decode at depth on the served 2.50 bpw daily (ring, 983,040 pool). The README's "decode at depth" row is from the
# 3.05 bpw pack (R492, 2026-09-18). fn_bench, forced 2,048 tokens, greedy, salted unique filler, live launcher tier off:
# c1 code and prose at ctx targets 0 / 133,000 / 266,000 (server counts ~0.75x: ~0 / ~100k / ~200k prompt tokens), 2 runs
# after fn_bench's warm-up round; c4 prose at the 133,000 target (4 distinct ~100k contexts = ~400k resident). TIMEBOX 30 min.
# RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r554-depth-decode; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
HS=/srv/qwen5090/probes/hot_slots.py
FB=/srv/qwen5090/probes/fn_bench.py
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r554] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){ if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R554 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$HS" "$FB"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r554-depth-decode
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; live image $(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")"
BOOTED=1
SALT=$(( $(date +%s) % 100000 ))
sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
"${CLEAN_ENV[@]}" NVME_TIER= bash "$LIVE" > "$R/boot.log" 2>&1 || { log "NO BOOT"; finish FAILED; exit 3; }
for i in $(seq 150); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
[ "$(served_id)" = "$MODEL" ] || { log "NO BOOT"; finish FAILED; exit 3; }
log "UP: $(sudo grep -E '^  (cache_size|chunk_size|max_batch_size):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '); VRAM free $(vram)"
n=0
for kind in code prose; do for c in 0 133000 266000; do n=$((n+1))
  timeout 900 python3 "$FB" --url "$API" --model "$MODEL" --tag "c1-$kind-$c" --kind $kind --tokens 2048 --conc 1 --runs 2 --ctx $c --unique \
    --salt $(( SALT + n*1009 )) --out "$R/depth.jsonl" >> "$R/fn_bench.log" 2>&1 || log "fn_bench c1 $kind $c rc $?"
done; done
timeout 1200 python3 "$FB" --url "$API" --model "$MODEL" --tag "c4-prose-133000" --kind prose --tokens 2048 --conc 4 --runs 2 --ctx 133000 --unique \
  --salt $(( SALT + 7777 )) --out "$R/depth.jsonl" >> "$R/fn_bench.log" 2>&1 || log "fn_bench c4 rc $?"
python3 - "$R/depth.jsonl" <<'PY' 2>&1 | tee "$R/summary.txt" | tee -a "$R/audit.log"
import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    print(f"{r.get('tag')} run{r.get('run')}: prompt {r.get('prompt_tokens')} tokens, n_ok {r.get('n_ok')}, tokens {r.get('tokens_median')}, ttft {r.get('ttft_s')} s, per-stream decode {r.get('decode_tps')}, aggregate {r.get('aggregate_tps')} t/s")
PY
[ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext)" = 0 ] && log "server alive after the probe" || log "SERVER NOT ALIVE after the probe"
finish DONE
