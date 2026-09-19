#!/usr/bin/env bash
# R553 — prefill chunk size vs running-stream stalls. R549 phase C: while a newcomer's ~22.5k prompt prefills, every running
# stream gets a frame only every ~0.54 s (73.8 -> 4.2 frames/s with 1 hot slot). The generator runs one 2,048-token prefill
# chunk per step; a smaller chunk should shorten the gap at some cost in prefill rate. Measures CHUNK 2048 / 1024 / 512 on
# the live launcher (tier off): hot_slots phase C (0..3 slots decoding ~112k contexts + a short and a ~22.5k newcomer) and
# salted cold prefill at the 60k and 120k targets. Pool unchanged (a smaller chunk only lowers the loader's dummy-chunk
# budget). Changing CHUNK can change greedy text (prefill numerics); fingerprints recorded, not gated. GPU TIMEBOX 30 min.
# RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r553-chunk-hot-stall; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
HS=/srv/qwen5090/probes/hot_slots.py
FB=/srv/qwen5090/probes/fn_bench.py
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r553] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){ if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R553 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$HS" "$FB"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r553-chunk-hot-stall
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; live image $(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")"
BOOTED=1
SALT=$(( $(date +%s) % 100000 ))
for CH in 2048 1024 512; do
  sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
  "${CLEAN_ENV[@]}" NVME_TIER= CHUNK=$CH bash "$LIVE" > "$R/boot-$CH.log" 2>&1 || { log "NO BOOT chunk $CH"; continue; }
  for i in $(seq 150); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
  [ "$(served_id)" = "$MODEL" ] || { log "NO BOOT chunk $CH"; continue; }
  POOL=$(sudo sed -n 's/^  cache_size: \([0-9]*\).*/\1/p' $CFG)
  log "UP chunk $CH: $(sudo grep -E '^  (cache_size|chunk_size):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '); VRAM free $(vram)"
  python3 "$FB" --url "$API" --model "$MODEL" --tag "fp-$CH" --kind prose --tokens 256 --conc 1 --runs 1 --ctx 0 --out "$R/fp.jsonl" > /dev/null 2>&1
  for c in 60000 120000; do
    python3 "$FB" --url "$API" --model "$MODEL" --tag "prefill-$CH-$c" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c --unique \
      --salt $(( SALT + CH + c/1000 )) --out "$R/prefill.jsonl" 2>&1 | grep -E "FAILED|Traceback" | tee -a "$R/audit.log"
    python3 -c 'import json,sys; r=[x for x in map(json.loads, open(sys.argv[1])) if x.get("tag")==sys.argv[2]][-1]; pt=r["prompt_tokens"]; print("cold prefill %s: %d tokens, %.0f tok/s" % (sys.argv[2], pt, pt/r["ttft_s"]))' \
      "$R/prefill.jsonl" "prefill-$CH-$c" 2>&1 | tee -a "$R/audit.log"
  done
  timeout 900 python3 "$HS" --url "$API" --model "$MODEL" --pool "$POOL" --phases C --salt $(( SALT + CH )) --out "$R/hot-$CH.jsonl" > "$R/hot-$CH.log" 2>&1
  log "chunk $CH phase C exit $?"
  python3 - "$R/hot-$CH.jsonl" "$CH" <<'PY' 2>&1 | tee -a "$R/summary.txt" | tee -a "$R/audit.log"
import json,sys
f=lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v,float) else str(v))
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r["phase"]!="C": continue
    print(f"chunk {sys.argv[2]} C {r['hot']} hot: new short TTFT {f(r['short_ttft'])} s, decode {r['short_decode_tps']}; new ~22.5k TTFT {f(r['ctx30k_ttft'])} s"
          + (f"; hot frames/s before {r['hot_rate_before']} during {r['hot_rate_during_30k_prefill']}, max gap before {r['hot_max_gap_before']} during {r['hot_max_gap_during_30k_prefill']} s" if r['hot'] else "")
          + f" {r['errors'] or ''}")
PY
done
finish DONE
