#!/usr/bin/env bash
# R564 — draft-topk r1 (omp round, patches/exllamav3/draft-topk/r1): EXL3_DRAFT_TOPK_STATS=1 records, per MTP draft
# position, the draft head's top-4 and the rank of the target's token on rejection. Instrument only: flag off = served
# bytes, flag on = output byte-identical. The estimator (tests/estimate_two_chain.py) then says whether two draft chains
# (branch at position 1 on the draft's rank-2 token, 8 verify rows at c1) or a deeper single chain would pay at c1.
#   build IMG-topkstats (python only) before the lock
#   A: flag off, tier off, 8 slots @ live pool: fingerprints c1 / 30k must be canonical; fn_bench c1 code + prose 2048
#   B: flag on (stats to /exl3-cache/draft-topk-r564 = $TUNEDIR/draft-topk-r564): fingerprints canonical, same fn_bench,
#      then collect_topk.py 24 code + 24 prose c1 x 512 (time windows -> meta.jsonl), JSONL copied, estimator run
# Decision rule (fixed now): the instrument is sound iff A and B fingerprints are canonical and B's c1 decode is within
# -2 % of A on code and prose. A two-chain implementation round is briefed iff the estimator's (b) gain divided by (a) is
# above the 8-row / 4-row step ratio 1.183 on code or prose; otherwise the idea closes with the numbers.
# Measurement only. GPU TIMEBOX 25 min after the lock. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r564-draft-topk; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/draft-topk-r1
CFG=/srv/qwen5090/flashnext-config.yml
C1=f4add302e176d78e; C30=4a255910dee2d9c5
SDIR_H=/srv/qwen5090/.exl3cache/draft-topk-r564; SDIR_C=/exl3-cache/draft-topk-r564
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r564] $*" | tee -a "$R/audit.log"; }
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
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R564 $1 ==="; }
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
fp30(){ python3 - "$API" "$NEWM" <<'PY' 2>/dev/null || echo none
import json,sys,hashlib,urllib.request,random
api,model=sys.argv[1:3]
rng=random.Random(4242); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
msg=" ".join(rng.choice(words) for _ in range(23000))+"\n\nSummarize the passage above in one sentence, then write a haiku about it."
req=json.dumps({"model":model,"temperature":0,"max_tokens":128,"min_tokens":128,"messages":[{"role":"user","content":msg}]}).encode()
d=json.loads(urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=req,headers={"Content-Type":"application/json"}),timeout=900).read())
m=d["choices"][0]["message"]; print(hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16])
PY
}
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$SRC/Dockerfile.box" "$SRC/tests/collect_topk.py" "$SRC/tests/estimate_two_chain.py" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE")
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE"); NIMG="$LIMG-topkstats"
[ -n "$LENV" ] && [ -n "$LIMG" ] || { log "ABORT: cannot parse live IMG / EXTRA_ENV"; exit 3; }
log "building $NIMG on $LIMG (python only)"
(cd "$SRC" && sudo docker build --build-arg BASE="$LIMG" -f Dockerfile.box -t "$NIMG" . ) > "$R/build.log" 2>&1 \
  || { log "ABORT: build: $(grep -aiE 'error|assert|abort' "$R/build.log" | head -4 | tr '\n' ' ' | cut -c1-400)"; exit 3; }
log "build OK: $(grep -aE 'default off|wired|hook present' "$R/build.log" | tr '\n' ' ' | cut -c1-240)"
sudo rm -rf "$SDIR_H"; sudo mkdir -p "$SDIR_H"; sudo chmod 777 "$SDIR_H"
export GPU_QUEUE_NAME=r564-draft-topk
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id)"
BOOTED=1
END=$(( $(date +%s) + 1500 ))
rc=0
bench(){ local tag=$1 k
  for k in code prose; do python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$tag-$k" --kind $k --tokens 2048 --warmup-runs 1 --conc 1 --runs 2 \
    --out "$R/c1.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | cut -c1-200 | sed "s/^/[$tag $k] /" | tee -a "$R/audit.log"; done; }
arm(){ local tag=$1 a b f0 f1; shift
  up "$tag" NVME_TIER= IMG="$NIMG" "$@" || return 1
  f0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0); f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  a=$(greedy "$tag"); b=$(fp30)
  log "UP $tag: $(sudo docker inspect -f '{{.Config.Image}}' flashnext), topk env: $(sudo docker exec flashnext env | grep -c '^EXL3_DRAFT_TOPK_STATS=1'); boot free $f0/$f1; c1 $a / 30k $b"
  [ "$a" = "$C1" ] && [ "$b" = "$C30" ] || { log "$tag: FINGERPRINT MISMATCH (want $C1 / $C30)"; return 1; }
  bench "$tag"; alive || { log "$tag: SERVER NOT ALIVE"; return 1; }; }
arm A || rc=1
if [ $rc = 0 ]; then
  arm B EXTRA_ENV="$LENV EXL3_DRAFT_TOPK_STATS=1 EXL3_DRAFT_TOPK_STATS_DIR=$SDIR_C" || rc=1
fi
if [ $rc = 0 ]; then
  sudo docker logs flashnext 2>&1 | grep -a 'EXL3_DRAFT_TOPK_STATS' | tail -3 | cut -c1-240 | tee -a "$R/audit.log"
  # the engine keeps the JSONL open: split by line offset, never move it
  N0=$(sudo cat "$SDIR_H/draft_topk.jsonl" 2>/dev/null | wc -l); log "JSONL before collect: $N0 lines (B screens)"
  log "collect: 24 code + 24 prose, c1, 512 forced"
  python3 "$SRC/tests/collect_topk.py" --url "$API" --model "$NEWM" --tokens 512 --out "$R/collect.jsonl" --meta "$R/meta.jsonl" > "$R/collect.log" 2>&1 || { log "collect FAILED: $(tail -2 "$R/collect.log" | tr '\n' ' ' | cut -c1-200)"; rc=1; }
  sudo head -n "$N0" "$SDIR_H/draft_topk.jsonl" > "$R/draft_topk.pre-ab.jsonl"; sudo tail -n +"$((N0+1))" "$SDIR_H/draft_topk.jsonl" > "$R/draft_topk.jsonl"
  /bin/ls "$SDIR_H" | grep -q 'draft_topk.jsonl.1' && log "WARN: JSONL rotated during the run"
  log "JSONL: $(wc -l < "$R/draft_topk.jsonl" 2>/dev/null) lines, $(grep -c '"rec": *"request"' "$R/draft_topk.jsonl" 2>/dev/null) requests; $(sudo docker logs flashnext 2>&1 | grep -ac 'EXL3_DRAFT_TOPK_STATS.*fail') writer failures"
  alive || { log "B: SERVER NOT ALIVE after collect"; rc=1; }
  python3 "$SRC/tests/estimate_two_chain.py" "$R/draft_topk.jsonl" --meta "$R/meta.jsonl" --depth 3 > "$R/estimate.txt" 2>&1 || log "estimator FAILED: $(tail -2 "$R/estimate.txt" | tr '\n' ' ')"
  cat "$R/estimate.txt" >> "$R/audit.log"
fi
sudo docker logs flashnext > "$R/docker-B.log" 2>&1
python3 - "$R/c1.jsonl" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,collections
d=collections.defaultdict(list)
for r in map(json.loads,open(sys.argv[1])):
    if r.get("aggregate_tps"): d[r["tag"]].append(r["aggregate_tps"])
m={k:sum(v)/len(v) for k,v in d.items()}
for k in ("code","prose"):
    a,b=m.get("A-"+k),m.get("B-"+k)
    if a and b: print("c1 %s: A %.1f B %.1f B/A %.3f" % (k,a,b,b/a))
PY
[ $rc = 0 ] && finish DONE || finish FAILED
