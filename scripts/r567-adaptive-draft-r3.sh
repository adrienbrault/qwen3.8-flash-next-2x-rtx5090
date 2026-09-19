#!/usr/bin/env bash
# R567 — adaptive MTP draft r3 box A/B (patches/exllamav3/adaptive-draft/r3; omp glm round finished by an Opus subagent after
# the opencode rate limit): deeper-only controller at one decoding job, EXL3_ADAPTIVE_DRAFT3=1. Floor = the served depth
# (never below 3); per round a binary choice 3 vs 4 by W*a > c*E(floor), c = 0.13 fixed at boot (R556 fit), cumulative per-job
# counts with priors, context split, probe every 16. r2's loss (R556) replays offline from its EMA + censoring, not the clock.
# Image tabbyapi:adaptive-draft-r3-gdnbf16 on the R561 image (hash pin refuses the R565 prefetch generator.py), all arms.
# Try 2: S at the served 966,656, F4 / R3 at P = 933,888 (try 1's ladder). Pool: depth 4 anywhere reserves GDN history 4 for 8 slots (~450 MiB): ladder F4 from 966,656 down 16,384 (<= 4 steps) until
#   free at boot >= 2,053 / 835 MiB (served 2,085 / 867 - 32); P = highest passing; all arms run at P; verdict states 966,656 - P.
# Arms (one boot each, tier off, 8 slots @ P), tags distinct after rstrip(digits): S [[4, 3], [8, 1]] flag off; F4
#   [[1, 4], [4, 3], [8, 1]] flag off; R3 same policy + flag; R3n (+ CONTEXT=0) if the timebox allows.
# Decision rule (fixed now; the ALTERNATIVE prose criterion of the spec, chosen before boot because content, not boots, sets
# the CI width): R3 passes iff code c1 R3/S >= +2.0 %, prose c1 R3/S point >= -1.0 % and CI upper >= 0, code c4 and prose c4
# points >= -1 %, no depth < 3, no OOM/traceback. Then: R3/F4 CI contains 0 on both c1 rows -> launcher line only; R3/F4 prose
# c1 CI lower > 0 -> the controller; else mixed, no call. Measurement only (promotion needs a rebase onto the R565 image).
# GPU TIMEBOX 30 min. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r567-adaptive-draft-r3-try2; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/adaptive-draft-r3
MP=/srv/qwen5090/probes/mp_decode.py
CFG=/srv/qwen5090/flashnext-config.yml
BIMG=tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16
AIMG=tabbyapi:adaptive-draft-r3-gdnbf16
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
F0MIN=2053; F1MIN=835
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r567] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){
  if [ "$BOOTED" = 1 ]; then sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done; log "daily: $(served_id)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R567 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$MP" "$SRC/Dockerfile.box" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
sudo docker image inspect "$BIMG" >/dev/null 2>&1 || { log "ABORT: base image $BIMG missing"; exit 3; }
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE")
BENV=$(echo "$LENV" | tr ' ' '\n' | grep -v '^EXL3_NGRAM_PREFETCH2=' | tr '\n' ' ' | sed 's/ $//')
log "EXTRA_ENV for every arm (live minus EXL3_NGRAM_PREFETCH2): $BENV"
log "prose criterion: ALTERNATIVE (point >= -1.0 % and CI upper >= 0), chosen before the first boot"
R3E="EXL3_ADAPTIVE_DRAFT3=1 EXL3_ADAPTIVE_DRAFT3_LOG=1"
POL4='[[1, 4], [4, 3], [8, 1]]'
log "building $AIMG on $BIMG (before the lock)"
(cd "$SRC" && sudo docker build --build-arg BASE="$BIMG" -f Dockerfile.box -t "$AIMG" . ) > "$R/build.log" 2>&1 \
  || { log "ABORT: build: $(grep -aiE 'error|mismatch|abort|refus|assert' "$R/build.log" | head -3 | tr '\n' ' ' | cut -c1-300)"; exit 3; }
log "build OK: image $(sudo docker image inspect -f '{{.Id}}' "$AIMG" | cut -c1-19)"
export GPU_QUEUE_NAME=r567-adaptive-draft-r3
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id)"
BOOTED=1
END=$(( $(date +%s) + 1800 ))
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
# try 1 (2026-09-19 19:04 UTC) laddered depth 4: 966,656 1,863/647, 950,272 1,963/787, 933,888 2,063/907 -> P = 933,888 (cost 32,768).
# Its S arm at P moved a layer (free 581/2,971, c1 18238d63): less GDN history at depth 3 changes autosplit (R544b), so S
# now runs the served pool 966,656 (normal placement, canonical c1) and only F4 / R3 run at P.
P=933888; log "P = $P from try 1's ladder (pool cost 32,768)"
for c in; do
  up "L$c" NVME_TIER= IMG="$AIMG" EXTRA_ENV="$BENV" MAXBS=8 CACHE=$c DRAFT_POLICY="$POL4" || { log "ladder $c: no boot"; continue; }
  f0=$(free0); f1=$(free1)
  if [ "$f0" -ge $F0MIN ] && [ "$f1" -ge $F1MIN ]; then P=$c; log "ladder $c: free at boot $f0/$f1 >= $F0MIN/$F1MIN: P = $c (pool cost $(( 966656 - c )))"; break; fi
  log "ladder $c: free at boot $f0/$f1 below $F0MIN/$F1MIN"
done
[ -n "$P" ] || { log "no pool down to 901,120 passes the boot check with depth 4 in the policy"; finish DONE; exit 0; }
rc=0
for arm in S F4 R3 R3n; do
  [ $arm = R3n ] && [ $(( END - $(date +%s) )) -lt 300 ] && { log "timebox: R3n skipped"; break; }
  case $arm in
    S)   up S   NVME_TIER= IMG="$AIMG" EXTRA_ENV="$BENV" MAXBS=8 CACHE=966656 || { rc=1; break; } ;;
    F4)  up F4  NVME_TIER= IMG="$AIMG" EXTRA_ENV="$BENV" MAXBS=8 CACHE=$P DRAFT_POLICY="$POL4" || { rc=1; break; } ;;
    R3)  up R3  NVME_TIER= IMG="$AIMG" EXTRA_ENV="$BENV $R3E" MAXBS=8 CACHE=$P DRAFT_POLICY="$POL4" || { rc=1; break; } ;;
    R3n) up R3n NVME_TIER= IMG="$AIMG" EXTRA_ENV="$BENV $R3E EXL3_ADAPTIVE_DRAFT3_CONTEXT=0" MAXBS=8 CACHE=$P DRAFT_POLICY="$POL4" || { rc=1; break; } ;;
  esac
  f0=$(free0); f1=$(free1)
  ep=$P; [ $arm = S ] && ep=966656
  sudo grep -q "cache_size: $ep$" $CFG && sudo grep -q "max_batch_size: 8" $CFG || { log "FAIL $arm: config"; rc=1; break; }
  pol=$(sudo grep -E 'draft_num_tokens_by_batch' $CFG | awk '{$1=$1; print}')
  ae=$(sudo docker exec flashnext env | grep -c '^EXL3_ADAPTIVE_DRAFT3')
  bl=$(sudo docker logs flashnext 2>&1 | grep -a 'EXL3_ADAPTIVE_DRAFT3=1:' | tail -1 | cut -c1-260)
  case $arm in
    S|F4) [ "$ae" = 0 ] && [ -z "$bl" ] || { log "FAIL $arm: adaptive env/boot line present"; rc=1; break; } ;;
    R3)   [ "$ae" = 2 ] && [ -n "$bl" ] || { log "FAIL R3: env ($ae keys) / boot line missing"; rc=1; break; } ;;
    R3n)  [ "$ae" = 3 ] && [ -n "$bl" ] || { log "FAIL R3n: env ($ae keys) / boot line missing"; rc=1; break; } ;;
  esac
  [ "$f0" -ge $F0MIN ] && [ "$f1" -ge $F1MIN ] || log "WARN $arm: free at boot $f0/$f1 below $F0MIN/$F1MIN"
  fp=$(greedy "$arm")
  log "UP $arm: $pol; free at boot $f0/$f1, after first request $(vram); c1 $fp; ${bl:-no adaptive line}"
  python3 "$MP" run --url "$API" --model "$NEWM" --tag "${arm}x" --tokens 512 --out "$R/mp.jsonl" 2>&1 | tail -3 | tee -a "$R/audit.log"
  sudo docker logs flashnext > "$R/docker-$arm.log" 2>&1
  grep -aE 'out of memory|GPU assert|Traceback' "$R/docker-$arm.log" | head -3 | tee -a "$R/audit.log"
  log "$arm done; VRAM free $(vram); new verify shapes: $(grep -a 'new verify shape' "$R/docker-$arm.log" | grep -oE '\([0-9, ]+\)' | sort -u | tr '\n' ' ')"
done
for pair in "Sx R3x" "Sx F4x" "F4x R3x" "Sx R3nx" "R3x R3nx"; do set -- $pair
  grep -q "\"tag\": \"$2\"" "$R/mp.jsonl" 2>/dev/null || continue
  python3 "$MP" compare --a $1 --b $2 "$R/mp.jsonl" 2>&1 | sed "s/^/[$1 vs $2] /" | tee -a "$R/analysis.txt" | tee -a "$R/audit.log"; done
python3 - "$R/analysis.txt" "$R/docker-R3.log" <<'PY' 2>&1 | tee -a "$R/audit.log"
import re,sys,statistics as st
t=open(sys.argv[1]).read()
def get(pair,shape):
    m=re.search(r"\[%s\] %s: .*?paired geo-mean \S+/\S+ ([+-][0-9.]+) %%, 95 %% CI \[([+-][0-9.]+), ([+-][0-9.]+)\]" % (re.escape(pair),re.escape(shape)), t)
    return tuple(map(float,m.groups())) if m else None
sr={s:get("Sx vs R3x",s) for s in ("code c1","prose c1","code c4","prose c4")}
fr={s:get("F4x vs R3x",s) for s in ("code c1","prose c1")}
print("R3/S", sr); print("R3/F4", fr)
try: log=open(sys.argv[2],errors="replace").read()
except Exception: log=""
bad=re.findall(r"applied \{[^}]*\b([0-2]):",log)
why=[]
if None in sr.values(): why.append("missing R3/S rows")
else:
    if sr["code c1"][0] < 2.0: why.append(f"code c1 {sr['code c1'][0]:+.2f} < +2.0")
    if sr["prose c1"][0] < -1.0 or sr["prose c1"][2] < 0: why.append(f"prose c1 {sr['prose c1']}")
    for s in ("code c4","prose c4"):
        if sr[s][0] < -1.0: why.append(f"{s} {sr[s][0]:+.2f} < -1")
if bad: why.append(f"depth below 3 in R3 ({len(bad)})")
if why: print("R3 FAILS: "+"; ".join(why)+" -> promote nothing")
elif None in fr.values(): print("R3 PASSES; R3/F4 rows missing -> no attribution")
elif fr["code c1"][1] <= 0 <= fr["code c1"][2] and fr["prose c1"][1] <= 0 <= fr["prose c1"][2]: print("R3 PASSES; F4 within noise -> launcher line [[1, 4], [4, 3], [8, 1]] only")
elif fr["prose c1"][1] > 0: print("R3 PASSES; R3/F4 prose c1 CI > 0 -> the controller")
else: print("R3 PASSES; mixed R3/F4 reading -> no promotion call")
d=[float(x) for x in re.findall(r"mean_depth[=: ]+([0-9.]+)",log)]
if d: print(f"R3 per-request mean depth: n {len(d)}, mean {st.mean(d):.2f}, min {min(d):.2f}, max {max(d):.2f}")
PY
[ $rc = 0 ] && finish DONE || finish FAILED
