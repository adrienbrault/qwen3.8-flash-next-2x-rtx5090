#!/usr/bin/env bash
# R568 — where does the rebase's 120k prefill gain come from? (R563: P 11,112 vs S 10,156 t/s at 120k, one sample; 30k equal)
# Candidate: upstream 825db5b EXL3_GR_MIX_TILED (default on in the rebased tree): prefill-sized HC mixes on a deterministic int8
# tensor-core GEMM. Arms, one boot each, tier off, 8 slots: S = R561 image @ 966,656; T1 = rebase image @ 917,504 (tiled on);
# T0 = rebase image @ 917,504 with EXL3_GR_MIX_TILED=0. Per arm: free VRAM at boot (does the tiled path own part of the
# rebase's missing memory?), 3 salted cold prefills each at 60k and 120k.
# Decision: if T1/T0 >= +5 % at 120k (mean of 3), port 825db5b alone onto the served stack (glm round); if T0 ~ T1 and both
# beat S, the gain is elsewhere in the rebase and the port brief must bisect. Survey-grade: GPU TIMEBOX 15 min.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r568-rebase-prefill; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
BIMG=tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16
PIMG=tabbyapi:rebase-dev-r1
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r568] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){
  if [ "$BOOTED" = 1 ]; then sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done; log "daily: $(served_id)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R568 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for i in "$BIMG" "$PIMG"; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: $i missing"; exit 3; }; done
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE" | tr ' ' '\n' | grep -v '^EXL3_NGRAM_PREFETCH2=' | tr '\n' ' ' | sed 's/ $//')
export GPU_QUEUE_NAME=r568-rebase-prefill
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id)"
BOOTED=1
SALT=$(( $(date +%s) % 100000 ))
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
pf(){ local tag=$1 c k
  for k in 1 2 3; do for c in 60000 120000; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "pf-$tag-$c" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c --unique \
      --salt $(( SALT + c/1000 + k*997 + ${#tag}*13 + RANDOM )) --out "$R/prefill.jsonl" > /dev/null 2>&1; done; done; }
for arm in S T1 T0; do
  case $arm in
    S)  up S  NVME_TIER= IMG="$BIMG" EXTRA_ENV="$LENV" CACHE=966656 ;;
    T1) up T1 NVME_TIER= IMG="$PIMG" EXTRA_ENV="$LENV" CACHE=917504 ;;
    T0) up T0 NVME_TIER= IMG="$PIMG" EXTRA_ENV="$LENV EXL3_GR_MIX_TILED=0" CACHE=917504 ;;
  esac || { log "NO BOOT $arm"; continue; }
  log "UP $arm: $(sudo docker inspect -f '{{.Config.Image}}' flashnext); tiled env: $(sudo docker exec flashnext env | grep -c '^EXL3_GR_MIX_TILED=0') (1 = off); free at boot $(vram)"
  pf $arm
  alive || log "$arm: NOT ALIVE"
done
python3 - "$R/prefill.jsonl" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
d=collections.defaultdict(list)
for r in map(json.loads,open(sys.argv[1])):
    if r.get("ttft_s") and r.get("prompt_tokens"): d[r["tag"]].append(r["prompt_tokens"]/r["ttft_s"])
m={}
for t in sorted(d): m[t]=st.mean(d[t]); print(f"{t}: {', '.join(f'{x:.0f}' for x in d[t])} mean {m[t]:.0f}")
for c in ("60000","120000"):
    s,t1,t0=(m.get(f"pf-{a}-{c}") for a in ("S","T1","T0"))
    if s and t1 and t0: print(f"{c}: T1/S {t1/s:.3f}  T0/S {t0/s:.3f}  T1/T0 {t1/t0:.3f}")
PY
finish DONE
