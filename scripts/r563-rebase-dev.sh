#!/usr/bin/env bash
# R563 — rebase-dev r1 box A/B: the served exllamav3 stack ported onto upstream dev 1d64111 (omp round rebase-dev-r1,
# patches/exllamav3/rebase-dev/r1; survey flan/surveys/2026-09-19-exllamav3-dev-upstream.md). Greedy bytes may differ by design
# (deterministic router GEMM, tiled HC prefill), so decode is judged by mp_decode paired prompts, not fingerprints.
#   build P = tabbyapi:rebase-dev-r1 on the live image (whole package replaced, extension rebuilt for sm_120)
#   boots at the LIVE config (8 slots @ 966,656 after R561, served EXTRA_ENV), tier off, order S P P S:
#     free VRAM at boot per card, c1 + 30k fingerprints, cold prefill 30k / 120k, mp_decode (24 code + 24 prose, c1 + c4, 512)
#   flag-flip arm C (P without EXL3_SHARED_EXPERT_OVERLAP = upstream's fused shared-expert coop launch), mp_decode vs P
#   fn_bench c8 code on S1 and P2 (the 8-slot shape)
# Decision rule (fixed before the run, from the round's box-ab-spec): P is a promotion candidate iff code and prose c1 / c4
# paired geo-means are each >= -1 % vs S with CI lower bound >= -2 %, cold prefill >= 0.95x S at 30k and 120k, per-card boot
# free VRAM >= S - 32 MiB, and the server stays alive. C replaces our overlap if it is >= 0 on every row with CI excluding -1 %.
# Try 4: P / C boot at PPOOL, the largest pool (16,384 steps down from the live pool) the rebase boots at; a PPOOL below the
# live pool is a cost the decode result has to pay for (or a port bug to find) and blocks a promotion as-is.
# Measurement only (promotion = a separate gated unit). GPU TIMEBOX 40 min after the lock. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r563-rebase-dev-try4; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/rebase-dev-r1
MP=/srv/qwen5090/probes/mp_decode.py
CFG=/srv/qwen5090/flashnext-config.yml
BIMG=tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16
PIMG=tabbyapi:rebase-dev-r1
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r563] $*" | tee -a "$R/audit.log"; }
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
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R563 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$MP" "$SRC/Dockerfile.box" "$SRC/exllamav3/__init__.py" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
sudo docker image inspect "$BIMG" >/dev/null 2>&1 || { log "ABORT: base image $BIMG missing"; exit 3; }
# S is pinned to the R561 image and env: R565 may promote the n-gram prefetch (another image + EXL3_NGRAM_PREFETCH2) first,
# and the rebase tree carries no prefetch overlay
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE" | tr ' ' '\n' | grep -v '^EXL3_NGRAM_PREFETCH2=' | tr '\n' ' ' | sed 's/ $//')
case " $LENV " in *" EXL3_GDN_STATE_BF16=1 "*) ;; *) log "ABORT: live EXTRA_ENV lacks bf16 (expected R548 or later)"; exit 3;; esac
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE"); case "$LIMG" in "$BIMG"|tabbyapi:ngram-prefetch-r1-gdnbf16) ;; *) log "ABORT: live image $LIMG unexpected"; exit 3;; esac
LPOOL=$(sed -n 's/^CACHE=\${CACHE:-\([0-9]*\)}.*/\1/p' "$LIVE"); LBS=$(sed -n 's/^MAXBS=\${MAXBS:-\([0-9]*\)}$/\1/p' "$LIVE")
df -h / | tail -1 | tee -a "$R/audit.log"
log "S1 building $PIMG on $BIMG (before the lock; full extension rebuild)"
(cd "$SRC" && sudo docker build --build-arg BASE="$BIMG" -f Dockerfile.box -t "$PIMG" . ) > "$R/build.log" 2>&1 \
  || { log "ABORT: build: $(grep -aiE 'error|assert|abort' "$R/build.log" | head -4 | tr '\n' ' ' | cut -c1-400)"; exit 3; }
log "build OK: $(grep -a 'import assertion OK' "$R/build.log" | tail -1 | cut -c1-120)"
export GPU_QUEUE_NAME=r563-rebase-dev
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id)"
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
SALT=$(( $(date +%s) % 100000 ))
prefill(){ local tag=$1 c
  for c in 30000 120000; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "pf-$tag-$c" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c --unique \
      --salt $(( SALT + c/1000 + ${#tag}*13 + RANDOM )) --out "$R/prefill.jsonl" > /dev/null 2>&1; done
  python3 -c 'import json,sys; r=[x for x in map(json.loads,open(sys.argv[1])) if x.get("tag","").startswith("pf-"+sys.argv[2]+"-") and x.get("ttft_s")]; print(" ".join("%dk %d" % (x["ctx_requested"]//1000, x["prompt_tokens"]/x["ttft_s"]) for x in r))' "$R/prefill.jsonl" "$tag"; }
END=$(( $(date +%s) + 2400 ))
rc=0; n=0; S0=; S1=; PPOOL=
# try 3 (2026-09-19 18:00): P did not boot at the live pool ("Insufficient VRAM in split for model and cache"). Per the round's
# box-ab-spec, P is laddered down 16,384 at a time (at most 8 steps) on its first arm; every P / C arm then runs at PPOOL.
pladder(){ local c=$LPOOL k
  for k in $(seq 0 8); do
    up "P-L$c" NVME_TIER= IMG="$PIMG" EXTRA_ENV="$LENV" CACHE=$c && { PPOOL=$c; log "P ladder: $c boots ($(( LPOOL - c )) below the live $LPOOL)"; return 0; }
    log "P ladder: $c does not boot"; c=$(( c - 16384 )); done
  return 1; }
for arm in S P P S C; do
  n=$((n+1)); tag="$arm$n"
  [ $(( END - $(date +%s) )) -lt 300 ] && { log "timebox: stopping before $tag"; break; }
  case $arm in
    S) up "$tag" NVME_TIER= IMG="$BIMG" EXTRA_ENV="$LENV" || { rc=1; break; } ;;
    P) if [ -z "$PPOOL" ]; then pladder || { log "P: no boot down to $(( LPOOL - 8*16384 ))"; rc=1; break; }
       else up "$tag" NVME_TIER= IMG="$PIMG" EXTRA_ENV="$LENV" CACHE=$PPOOL || { rc=1; break; }; fi ;;
    C) up "$tag" NVME_TIER= IMG="$PIMG" CACHE=$PPOOL EXTRA_ENV="$(echo "$LENV" | tr ' ' '\n' | grep -v '^EXL3_SHARED_EXPERT_OVERLAP=' | tr '\n' ' ' | sed 's/ $//')" || { rc=1; break; } ;;
  esac
  f0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0); f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  img=$(sudo docker inspect -f '{{.Config.Image}}' flashnext); ov=$(sudo docker exec flashnext env | grep -c '^EXL3_SHARED_EXPERT_OVERLAP=1')
  a=$(greedy "$tag"); b=$(fp30)
  log "UP $tag: $img, $(sudo grep -E '^  (cache_size|max_batch_size):' $CFG | awk '{$1=$1; print}' | tr '\n' ' ')overlap env $ov; boot free $f0/$f1; c1 $a / 30k $b"
  [ "$arm" = S ] && [ -z "$S0" ] && { S0=$f0; S1=$f1; }
  [ "$arm" = P ] && [ -n "$S0" ] && { [ "$f0" -ge $(( S0 - 32 )) ] && [ "$f1" -ge $(( S1 - 32 )) ] && log "$tag headroom OK vs S ($S0/$S1)" || log "$tag HEADROOM BELOW S - 32 ($f0/$f1 vs $S0/$S1)"; }
  [ $n -le 2 ] && { log "$tag cold prefill: $(prefill "$tag")"; python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$tag-c8" --kind code --tokens 1024 --warmup-runs 1 --conc 8 --runs 2 \
      --out "$R/c8.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | cut -c1-200 | sed "s/^/[$tag c8] /" | tee -a "$R/audit.log"; }
  python3 "$MP" run --url "$API" --model "$NEWM" --tag "$tag" --tokens 512 --out "$R/mp.jsonl" 2>&1 | tail -3 | tee -a "$R/audit.log"
  sudo docker logs flashnext > "$R/docker-$tag.log" 2>&1
  grep -aE 'out of memory|GPU assert|Traceback' "$R/docker-$tag.log" | head -3 | tee -a "$R/audit.log"
  alive || { log "$tag: SERVER NOT ALIVE"; rc=1; break; }
done
for pair in "S P" "P C"; do set -- $pair
  python3 "$MP" compare --a $1 --b $2 "$R/mp.jsonl" 2>&1 | sed "s/^/[$1 vs $2] /" | tee -a "$R/analysis.txt" | tee -a "$R/audit.log"; done
[ $rc = 0 ] && finish DONE || finish FAILED
