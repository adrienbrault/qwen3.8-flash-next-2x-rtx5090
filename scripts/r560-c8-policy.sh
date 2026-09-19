#!/usr/bin/env bash
# R560 — c8 draft-policy sweep at 8 slots (user 2026-09-19: "8 slots with 900k KV would be nice ... at least +50 % decode agg
# at c8 vs c6"). The served policy [[4, 3], [8, 1]] drops to depth 1 above 4 jobs: c8 = 8 x 2 = 16 verify rows, the bszn16
# MoE path's ceiling. Depth 2 at c5-8 = up to 24 rows on the slower >16-row path: does the extra accepted token pay for it?
# Pool: the 8-slot pool R558's ladder passed (parsed from its audit log; no pass -> skip, rc 0). One boot per arm, tier off:
#   P1 [[4, 3], [8, 1]] (served policy at 8 slots, = R558 S2)   P2 [[4, 3], [6, 2], [8, 1]]   P3 [[4, 3], [8, 2]]
# Per arm: c1 fingerprint (must equal P1: c1 runs depth 3 in all arms), fn_bench code + prose at c6 and c8, 1,024 forced x 2
# after a warm-up; new verify shapes, OOM grep, min free VRAM during the c8 rounds.
# Measurement only: nothing is promoted. GPU TIMEBOX 30 min. RUN (queued): popped by r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r560-c8-policy; mkdir -p "$R"
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
log(){ echo "$(date -Is) [r560] $*" | tee -a "$R/audit.log"; }
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
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R560 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$FB" /srv/qwen5090/probes/agent_replay.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
grep -q '^MAXBS=${MAXBS:-4}$' "$LIVE" || { log "ABORT: live launcher has no MAXBS knob at 4"; exit 3; }
POOL8=$(grep -a "\[L8\] [0-9]*: PASS" /srv/qwen5090/results/2026-09-19-r558-slots8/audit.log 2>/dev/null | tail -1 | sed -n "s/.*\[L8\] \([0-9]*\): PASS.*/\1/p")
[ -n "$POOL8" ] || { log "SKIP: R558 passed no 8-slot pool"; exit 0; }
LPOOL=$(sed -n 's/^CACHE=\${CACHE:-\([0-9]*\)}.*/\1/p' "$LIVE")
export GPU_QUEUE_NAME=r560-c8-policy
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
bench(){ local tag=$1 conc=$2 kind
  for kind in code prose; do
    python3 "$FB" --url "$API" --model "$NEWM" --tag "$tag-$kind" --kind $kind --tokens 1024 --warmup-runs 1 --conc $conc --runs 2 --out "$R/records.jsonl" 2>&1 \
      | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done; }
log "8-slot pool from R558: $POOL8"
P1F=""
for arm in P1 P2 P3; do
  case $arm in P1) pol='[[4, 3], [8, 1]]';; P2) pol='[[4, 3], [6, 2], [8, 1]]';; P3) pol='[[4, 3], [8, 2]]';; esac
  up $arm NVME_TIER= MAXBS=8 CACHE=$POOL8 DRAFT_POLICY="$pol" || continue
  sudo grep -q "max_batch_size: 8" $CFG && sudo grep -q "cache_size: $POOL8$" $CFG || { log "ABORT: config lacks MAXBS 8 / cache $POOL8"; finish FAILED; exit 3; }
  cp_=$(sudo grep -E 'draft_num_tokens_by_batch' $CFG | awk '{$1=$1; print}')
  fp=$(greedy $arm); [ $arm = P1 ] && P1F=$fp
  log "UP $arm: $cp_; VRAM free $(vram); c1 $fp$([ "$fp" = "$P1F" ] || echo ' (!= P1)')"
  for conc in 6 8; do
    [ $conc = 8 ] && { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -lms 200 > "$R/smi-$arm.csv" 2>/dev/null & sp=$!; }
    bench "$arm-c$conc" $conc
  done
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  mn=$(python3 -c 'import sys; m={}
for l in open(sys.argv[1]):
    p=[x.strip() for x in l.split(",")]
    if len(p)==2 and p[1].isdigit(): m[p[0]]=min(m.get(p[0],10**9),int(p[1]))
print(m.get("0",0), m.get("1",0))' "$R/smi-$arm.csv")
  sudo docker logs flashnext > "$R/docker-$arm.log" 2>&1
  grep -aE 'out of memory|GPU assert|Traceback' "$R/docker-$arm.log" | head -3 | tee -a "$R/audit.log"
  alive && a=alive || a="NOT ALIVE"
  log "$arm done: $a; min free during c8 $mn; new verify shapes $(grep -ac 'new verify shape' "$R/docker-$arm.log")"
done
python3 - "$R/records.jsonl" <<'PY' 2>&1 | tee -a "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
tok=collections.defaultdict(int); wall={}
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("ok"): tok[(r["tag"],r["run"])]+=r["completion_tokens"] or 0
    wall[(r["tag"],r["run"])]=r["round_wall_s"]
agg=collections.defaultdict(list)
for k,v in tok.items(): agg[k[0]].append(v/wall[k])
for t in sorted(agg): print(f"{t}: aggregate {st.mean(agg[t]):.1f} t/s (runs {len(agg[t])}: {', '.join(f'{x:.1f}' for x in agg[t])})")
for kind in ("code","prose"):
    for arm in ("P1","P2","P3"):
        c6=agg.get(f"{arm}-c6-{kind}"); c8=agg.get(f"{arm}-c8-{kind}")
        if c6 and c8: print(f"{arm} {kind}: c8/c6 = {st.mean(c8)/st.mean(c6):.3f}")
PY
finish DONE
