#!/usr/bin/env bash
# R484 — n-gram (PLE) table in host RAM vs the served per-row pread from disk, on the served config.
#
# WHY (user 2026-09-18: "Keep experimenting and pushing the setup. Decode speed and kv pool size"): R465 put the decode host
# gap with drafting at 1.0–2.2 ms/step, and the PLE host gather (`_gather_rows`) at 0.72 ms/step mean at c4 depth 3 (p90 1.66,
# max 2.37) and 0.30 at c8 depth 1 — the served engine streams the table (trellis_disk: threaded pread of each unique row
# through a worker pool, exllamav3_ext/ngram.cu). TabbyAPI's `ngram_ram` holds the 30.5 GiB table in RAM (trellis_ram: one
# index_select into the pinned staging buffer). Same rows, same GPU decode of the trellis -> greedy output must be identical.
# Host RAM: 60 GiB, MemAvailable ~48 GiB with the table in page cache; ON makes it anonymous memory. Watchdog: MemAvailable
# < 6 GiB removes the engine and aborts.
#
# ARMS (served config: 4 slots, 360,448 at 8,8, [30, 30], chunk 2048, vision on, EXTRA_ENV = daily): OFF, ON, OFF2, ON2.
# MEASURE per arm: MemAvailable after load; c1 + 30k greedy fingerprints; fn_bench code + prose c1/c4, 2,048 forced, 2 runs;
# cold prefill 30k / 120k (one run each). The daily is restored unchanged at the end.
#
# RUN: sudo systemd-run --unit=r484-ngram-ram --collect -p RuntimeMaxSec=43200 -E HOME=$HOME \
#        /bin/bash /srv/qwen5090/r484-ngram-ram.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r484-ngram-ram; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r484.sh
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0; WD=
log(){ echo "$(date -Is) [r484] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
memavail(){ awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|chunk_size|ngram_ram|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
finish(){ [ -n "$WD" ] && kill "$WD" 2>/dev/null
  if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged live launcher)"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
    log "daily: $(served_id) $(cfgline) VRAM free $(vram) MemAvailable $(memavail) GiB"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R484 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r484-ngram-ram
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline) MemAvailable $(memavail) GiB"
( while sleep 2; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$a" -lt 6144 ]; then echo "$(date -Is) [r484] WATCHDOG: MemAvailable ${a} MiB < 6144, removing the engine" | tee -a "$R/audit.log"
      sudo docker rm -f flashnext >/dev/null 2>&1; touch "$R/WATCHDOG"; fi; done ) & WD=$!

boot(){ local tag=$1 ng=$2 i st lp got
  BOOTED=1; log "boot $tag: NGRAM_RAM=$ng"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" NGRAM_RAM=$ng bash "$CAND" >> "$R/boot-$tag.log" 2>&1 ) & lp=$!
  for i in $(seq 360); do
    [ -e "$R/WATCHDOG" ] && { kill $lp 2>/dev/null; return 1; }
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp; got=$(cfgline)
      case "$got" in *"cache_size: 360448 cache_mode: 8,8 max_batch_size: 4 ngram_ram: $([ $ng = 1 ] && echo true || echo false) chunk_size: 2048 vision: true"*) ;;
        *) log "ABORT: generated config is '$got'"; return 2;; esac
      log "UP $tag: VRAM free $(vram) MemAvailable $(memavail) GiB; engine log: $(sudo docker logs flashnext 2>&1 | grep -aE 'n-gram|ngram' | tail -1 | cut -c1-160)"; return 0; fi
    st=$(cstatus); case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 60 ] && { sleep 3; continue; }
      log "NO BOOT $tag ($st): $(sudo docker logs --tail 60 flashnext 2>&1 | grep -aE 'Error|memory' | tail -1 | cut -c1-220)"
      sudo docker logs flashnext > "$R/boot-$tag.docker.log" 2>&1; kill $lp 2>/dev/null; wait $lp 2>/dev/null; return 1;; esac
    sleep 3; done
  log "NO BOOT $tag (timeout)"; kill $lp 2>/dev/null; return 1; }

measure(){ local t=$1 kind
  log "=== measure $t: $(cfgline) VRAM free $(vram) MemAvailable $(memavail) GiB ==="
  curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$t.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(sys.argv[2], "c1 greedy sha", hashlib.sha256(t.encode()).hexdigest()[:16], "(canonical 1474eee2f5945248)")' "$R/greedy-$t.json" "$t" 2>&1 | tee -a "$R/audit.log"
  python3 - "$API" "$MODEL" "$R/greedy30k-$t.json" "$t" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,hashlib,urllib.request,random
api,model,out,tag=sys.argv[1:5]
rng=random.Random(4242); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
body=" ".join(rng.choice(words) for _ in range(23000))
msg=body+"\n\nSummarize the passage above in one sentence, then write a haiku about it."
req=json.dumps({"model":model,"temperature":0,"max_tokens":128,"min_tokens":128,"messages":[{"role":"user","content":msg}]}).encode()
d=json.loads(urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=req,headers={"Content-Type":"application/json"}),timeout=900).read())
open(out,"w").write(json.dumps(d)); m=d["choices"][0]["message"]
print(tag,"30k greedy sha",hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16],"(canonical 4a255910dee2d9c5)")
PY
  for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$t-$kind" --kind $kind --tokens 2048 --conc 1 4 --runs 2 \
      --out "$R/records-$t.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$t $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "ttft-$t" --kind prose --tokens 64 --conc 1 --runs 1 --ctx 30000 120000 --unique \
    --out "$R/ttft-$t.jsonl" 2>&1 | grep -E "FAILED|Traceback|Error" | tee -a "$R/audit.log"
  python3 - "$R/ttft-$t.jsonl" "$t" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("ttft_s") and r.get("prompt_tokens"):
        print(f"[{sys.argv[2]} prefill] ctx {r['ctx_requested']}: {r['prompt_tokens']:,} prompt tokens, cold TTFT {r['ttft_s']:.2f} s = {r['prompt_tokens']/r['ttft_s']:,.0f} tok/s")
PY
  log "[$t] after measure: MemAvailable $(memavail) GiB; engine $(served_id)"; }

for arm in "OFF|0" "ON|1" "OFF2|0" "ON2|1"; do
  IFS='|' read -r tag ng <<< "$arm"
  boot "$tag" "$ng"; rc=$?
  [ $rc = 2 ] && { finish ABORTED; exit 3; }
  [ -e "$R/WATCHDOG" ] && { finish "ABORTED (RAM watchdog)"; exit 3; }
  [ $rc = 0 ] && measure "$tag"
  [ -e "$R/WATCHDOG" ] && { finish "ABORTED (RAM watchdog)"; exit 3; }
done
finish DONE
