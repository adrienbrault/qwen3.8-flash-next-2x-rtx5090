#!/usr/bin/env bash
# R569 — mtp-kv-window r1 screen (patches/exllamav3/mtp-kv-window/r1; glm round rebuilt by an Opus subagent): EXL3_MTP_KV_WINDOW
# shrinks the MTP draft layer's cache to a per-slot ring (sink page + W tokens). W16 = 16,384 frees ~932 MiB, all on cuda:0,
# but cuda:1 binds the pool (R539-R548), so it becomes pool only if a split change moves a layer off cuda:1 (impl-status).
# Screen (no acceptance A/B yet), image tabbyapi:mtpwin-r1 on the R561 image, tier off, 8 slots:
#   OFF [30, 30] @ 966,656   W16 [30, 30] @ 966,656   OFFS [31, 29] @ 966,656   WS W16 [31, 29] @ 966,656
#   then WS ladder +16,384 (<= 5 steps). Floor = OFF's tightest card free at boot - 32 MiB, on BOTH cards; survival per step =
#   cold 120k + one c8 code round. c1 fingerprint must equal the same-split OFF arm (W16 vs OFF, WS vs OFFS).
# Decision: k >= 2 steps (+3.4 %) -> write the full acceptance A/B + promotion unit (deep-context acceptance, needles, gates);
# k < 2 -> close the window idea for pool (the R539/R541 cuda:1 bind stands). GPU TIMEBOX 25 min.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r569-mtp-kv-window; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/mtp-kv-window-r1
BIMG=tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16
WIMG=tabbyapi:mtpwin-r1
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r569] $*" | tee -a "$R/audit.log"; }
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
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R569 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
[ -e "$SRC/Dockerfile.box" ] || { log "ABORT: missing $SRC"; exit 3; }
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE" | tr ' ' '\n' | grep -v '^EXL3_NGRAM_PREFETCH2=' | tr '\n' ' ' | sed 's/ $//')
log "building $WIMG on $BIMG"
(cd "$SRC" && sudo docker build --build-arg BASE="$BIMG" -f Dockerfile.box -t "$WIMG" . ) > "$R/build.log" 2>&1 \
  || { log "ABORT: build: $(grep -aiE 'error|mismatch|abort|refus|assert' "$R/build.log" | head -3 | tr '\n' ' ' | cut -c1-300)"; exit 3; }
log "build OK: $(grep -a 'imports through' "$R/build.log" | tail -1 | cut -c1-160)"
export GPU_QUEUE_NAME=r569-mtp-kv-window
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id)"
BOOTED=1
SALT=$(( $(date +%s) % 100000 ))
END=$(( $(date +%s) + 1500 ))
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
free0(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0; }
free1(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1; }
survive(){ python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "surv-$1" --kind prose --tokens 64 --conc 1 --runs 1 --ctx 120000 --unique \
    --salt $(( SALT + RANDOM )) --out "$R/survive.jsonl" > /dev/null 2>&1
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "c8-$1" --kind code --tokens 1024 --conc 8 --runs 1 --out "$R/survive.jsonl" 2>&1 \
    | grep -E "^  c=|FAILED|Traceback" | cut -c1-200 | sed "s/^/[$1] /" | tee -a "$R/audit.log"
  alive; }
# arm TAG SPLIT POOL WINDOW(0=off) -> sets F0 F1 FP; returns 1 on no boot
arm(){ local tag=$1 sp=$2 pool=$3 w=$4 ev="$LENV" wl
  [ "$w" != 0 ] && ev="$LENV EXL3_MTP_KV_WINDOW=$w"
  up "$tag" NVME_TIER= IMG="$WIMG" EXTRA_ENV="$ev" MAXBS=8 CACHE=$pool GPU_SPLIT="$sp" || return 1
  F0=$(free0); F1=$(free1)
  wl=$(sudo docker logs flashnext 2>&1 | grep -a 'EXL3_MTP_KV_WINDOW' | tail -1 | cut -c1-160)
  sudo docker logs flashnext 2>&1 | grep -aq 'window slots but max_batch_size' && { log "STOP $tag: draft cache has fewer window slots than max_batch_size (spec S0)"; finish FAILED; exit 1; }
  FP=$(greedy "$tag")
  log "UP $tag: split $sp pool $pool W $w; free at boot $F0/$F1; c1 $FP; ${wl:-no window line}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$sp" "$pool" "$w" "$F0" "$F1" "$FP" >> "$R/boots.tsv"; }
: > "$R/boots.tsv"
arm OFF '30, 30' 966656 0 || { log "OFF does not boot"; finish FAILED; exit 1; }
OFF0=$F0; OFF1=$F1; OFFFP=$FP; FLOOR=$(( (OFF0 < OFF1 ? OFF0 : OFF1) - 32 )); log "floor (tightest OFF card - 32) = $FLOOR MiB on both cards"
LSPLIT='31, 29'
if arm W16 '30, 30' 966656 16384; then
  log "W16 vs OFF free: cuda:0 $(( F0 - OFF0 )) / cuda:1 $(( F1 - OFF1 )) MiB"
  # moved placement (spec S0): cuda:1 gains >= 512 MiB at the same split -> layer 23 left cuda:1; ladder there, no forced split
  if [ $(( F1 - OFF1 )) -ge 512 ]; then LSPLIT='30, 30'; log "W16 moved placement at [30, 30]: ladder at [30, 30] (fingerprint compared against OFFS as the moved control)"; fi
  [ "$FP" = "$OFFFP" ] || [ "$LSPLIT" = '30, 30' ] || log "W16 c1 $FP != OFF $OFFFP (STOP rule at same placement)"
fi
arm OFFS '31, 29' 966656 0 && { OFFSFP=$FP; OS0=$F0; OS1=$F1; } || OFFSFP=none
arm WS "$LSPLIT" 966656 16384 || { log "WS does not boot"; finish DONE; exit 0; }
[ "$FP" = "$OFFSFP" ] || log "WS c1 $FP != OFFS $OFFSFP"
log "WS vs OFFS free: cuda:0 $(( F0 - ${OS0:-0} )) / cuda:1 $(( F1 - ${OS1:-0} )) MiB"
k=0; top=966656
if [ "$F0" -ge $FLOOR ] && [ "$F1" -ge $FLOOR ] && survive WS-966656; then
  for c in 983040 999424 1015808 1032192 1048576; do
    [ $(( END - $(date +%s) )) -lt 180 ] && { log "timebox: ladder stops before $c"; break; }
    arm "WS-$c" "$LSPLIT" $c 16384 || { log "ladder $c: no boot"; break; }
    [ "$F0" -ge $FLOOR ] && [ "$F1" -ge $FLOOR ] || { log "ladder $c: free $F0/$F1 below floor $FLOOR"; break; }
    survive "WS-$c" || { log "ladder $c: died under 120k + c8"; break; }
    k=$((k+1)); top=$c; log "ladder $c: PASS"
  done
else log "WS at 966,656 fails the floor or survival: k = 0"; fi
log "RESULT: k = $k steps above 966,656 (top $top) with W16 at [$LSPLIT]; decision threshold k >= 2"
finish DONE
