#!/usr/bin/env bash
# R558 — 8 slots on today's daily (user 2026-09-19: "explore going back to 8 slots, and improving decode at c8"). R518 (6 slots,
# fp32 GDN state, 786,432 era): 6 slots cost 65,536 pool tokens, c6 synthetic +17 %, but per stream ~86 t/s instead of ~135
# and the 8-agent replay ran 9 % slower. bf16 GDN state (R548) halves the per-slot state, so the pool cost should be smaller.
#   S0 REF: live launcher (4 slots, 1,032,192), tier off: fingerprints, free VRAM at boot, min free under 120k + c8 load
#   S1 descending ladder MAXBS=8 from 1,015,808 in 16,384 steps (max 12): first pool that boots with fingerprints = REF
#      (normal placement), cuda:1 free at boot >= REF - 32 MiB and min under 120k + c8 >= REF_MIN1 - 32, cuda:0 likewise
#      (R548 try 2: a late CUDA graph capture OOM'd with 100 MiB less on cuda:1; 8 slots capture more graph shapes)
#   S2 at POOL8: fn_bench code + prose, c1 / c4 / c8, 2,048 forced x 2 runs after a warm-up; then the R518/R557 8-agent
#      replay (compare with R557, same day, 4 slots); tabby log stats; docker log kept
#   S3 REF (4 slots) fn_bench c8 code + prose (4 slots queue the other 4: the over-capacity row) for the same-boot-day pair
# Measurement only: nothing is promoted. GPU TIMEBOX 45 min. RUN (queued): popped by r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r558-slots8; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
FB=/srv/qwen5090/probes/fn_bench.py
TRAJS="/srv/qwen5090/results/2026-09-16-r359-swebench-10/out/*/*.traj.json /srv/qwen5090/results/2026-09-16-r360-swebench-strat/out/*/*.traj.json /srv/qwen5090/results/2026-09-16-r361-swebench-failed/out/*/*.traj.json /srv/qwen5090/results/2026-09-16-r369-swebench-30/out/*/*.traj.json"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SALT=$(( $(date +%s) % 100000 ))
BOOTED=0
log(){ echo "$(date -Is) [r558] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
free0(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0; }
free1(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
finish(){
  if [ "$BOOTED" = 1 ]; then sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done; log "daily: $(served_id)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R558 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$FB" /srv/qwen5090/probes/agent_replay.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
grep -q '^MAXBS=${MAXBS:-4}$' "$LIVE" || { log "ABORT: live launcher has no MAXBS knob at 4"; exit 3; }
LPOOL=$(sed -n 's/^CACHE=\${CACHE:-\([0-9]*\)}.*/\1/p' "$LIVE")
export GPU_QUEUE_NAME=r558-slots8
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; live pool $LPOOL; disk $(df --output=pcent / | tail -1)"
BOOTED=1
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$NEWM" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
up(){ local tag=$1 i st lp; shift
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 ) & lp=$!
  for i in $(seq 200); do
    [ "$(served_id)" = "$NEWM" ] && { wait $lp; return 0; }
    st=$(cstatus); case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 40 ] && { sleep 3; continue; }
      log "NO BOOT $tag ($st): $(sudo docker logs --tail 80 flashnext 2>&1 | grep -aE 'RuntimeError|Error|memory' | tail -1 | cut -c1-160)"
      kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1;; esac
    [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 0)" -ge 1 ] && {
      log "NO BOOT $tag (restart loop): $(sudo docker logs --tail 80 flashnext 2>&1 | grep -aE 'RuntimeError|Error|memory' | tail -1 | cut -c1-160)"
      kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1; }
    sleep 3; done
  log "NO BOOT $tag (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }
# min free VRAM per card during a cold 120k prefill + a c8 code round -> "min0 min1"
loadmin(){ local tag=$1 sp
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -lms 100 > "$R/smi-$tag.csv" 2>/dev/null & sp=$!
  python3 "$FB" --url "$API" --model "$NEWM" --tag "$tag-pf" --kind prose --tokens 64 --conc 1 --runs 1 --ctx 120000 --unique --salt $(( SALT + ${#tag} * 7 + RANDOM )) --out "$R/load.jsonl" > /dev/null 2>&1
  python3 "$FB" --url "$API" --model "$NEWM" --tag "$tag-c8" --kind code --tokens 1024 --conc 8 --runs 1 --out "$R/load.jsonl" > /dev/null 2>&1
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  python3 -c 'import sys; m={}
for l in open(sys.argv[1]):
    p=[x.strip() for x in l.split(",")]
    if len(p)==2 and p[1].isdigit(): m[p[0]]=min(m.get(p[0],10**9),int(p[1]))
print(m.get("0",0), m.get("1",0))' "$R/smi-$tag.csv"; }
bench(){ local tag=$1 conc=$2 kind
  for kind in code prose; do
    python3 "$FB" --url "$API" --model "$NEWM" --tag "$tag-$kind" --kind $kind --tokens 2048 --warmup-runs 1 --conc $conc --runs 2 --out "$R/records.jsonl" 2>&1 \
      | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done; }
log "=== S0 REF: live launcher (4 slots), tier off ==="
up REF NVME_TIER= || { finish FAILED; exit 3; }
REF_F0=$(free0); REF_F1=$(free1); REF1=$(greedy REF)
read REF_MIN0 REF_MIN1 < <(loadmin REF)
log "REF @ $LPOOL x 4: boot free $REF_F0/$REF_F1, c1 $REF1, min under 120k + c8 $REF_MIN0/$REF_MIN1"
log "=== S3 (first, same boot) REF c8: 4 slots queue the other 4 ==="
bench REF-s4 8
log "=== S1 descending ladder at MAXBS=8 ==="
POOL8=0; c=$(( LPOOL - 16384 ))
for k in $(seq 1 12); do
  up "L8-$c" NVME_TIER= MAXBS=8 CACHE=$c || { c=$(( c - 16384 )); continue; }
  sudo grep -q "max_batch_size: 8" $CFG && sudo grep -q "cache_size: $c$" $CFG || { log "ABORT: config lacks MAXBS 8 / cache $c"; finish FAILED; exit 3; }
  f0=$(free0); f1=$(free1); v=$(vram); a=$(greedy "L8-$c")
  if [ "$a" != "$REF1" ]; then log "[L8] $c: c1 $a != REF (placement moved?), boot free $v"; c=$(( c - 16384 )); continue; fi
  if [ "$f1" -lt $(( REF_F1 - 32 )) ] || [ "$f0" -lt $(( REF_F0 - 32 )) ]; then log "[L8] $c: boot free $f0/$f1 < floors $((REF_F0-32))/$((REF_F1-32))"; c=$(( c - 16384 )); continue; fi
  read m0 m1 < <(loadmin "L8-$c")
  alive || { log "[L8] $c dies under 120k + c8"; c=$(( c - 16384 )); continue; }
  if [ "$m1" -lt $(( REF_MIN1 - 32 )) ] || [ "$m0" -lt $(( REF_MIN0 - 32 )) ]; then log "[L8] $c: min under load $m0/$m1 < floors $((REF_MIN0-32))/$((REF_MIN1-32))"; c=$(( c - 16384 )); continue; fi
  POOL8=$c; log "[L8] $c: PASS (boot free $v, c1 = REF, min under load $m0/$m1)"; break
done
[ "$POOL8" -gt 0 ] || { log "no 8-slot pool passes down to $c"; finish DONE; exit 0; }
log "=== S2 8 slots @ $POOL8: fn_bench c1 / c4 / c8, then the 8-agent replay ==="
for conc in 1 4 6 8; do bench "S8-c$conc" $conc; done   # c6 on the same boot: the baseline of the user's c8 >= 1.5 x c6 target
since=$(date -Is)
timeout 2400 python3 /srv/qwen5090/probes/agent_replay.py --url "$API" --model "$NEWM" --trajs $TRAJS --agents 8 --convs 16 --calls 24 \
  --tool-gap 2 --tag S8 --out "$R/replay.jsonl" 2>&1 | tail -1 | sed "s/^/[replay S8] /" | tee -a "$R/audit.log"
sudo docker logs --since "$since" flashnext > "$R/replay-S8.tabby.log" 2>&1
python3 /srv/qwen5090/probes/tabby_log_stats.py "$R/replay-S8.tabby.log" 2>&1 | tail -6 | sed "s/^/[tabby S8] /" | tee -a "$R/audit.log"
alive && log "server alive after the replay" || log "SERVER NOT ALIVE after the replay"
finish DONE
