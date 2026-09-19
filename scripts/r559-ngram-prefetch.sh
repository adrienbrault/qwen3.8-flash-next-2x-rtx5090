#!/usr/bin/env bash
# R559 — n-gram (PLE) row prefetch r1 screen (omp round, patches/exllamav3/ngram-prefetch/r1): EXL3_NGRAM_PREFETCH2=1 starts each
# verify forward's PLE row gather right after the MTP draft readback (today decode-sized gathers are staged inline: R465
# 0.72 ms/step at c4 d3) and moves prefill staging acquisition onto the worker; EXL3_NGRAM_TIMING=1 measures the exposed
# wait. Timing-only by construction: flag-on output must equal flag-off (fingerprints are the quality gate).
# Built on tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16 (R548; the overlay's four files are untouched by ring/bf16).
#   harness both cards (in-image, synthetic table) -> OFF boot + TIMING (fingerprints = REF, prefill 60k, c1/c4 rounds:
#   exposed wait today) -> ON boot + TIMING (fingerprints = REF, same rounds: wait after) -> ON boot (prefill 60k/120k salted,
#   mp_decode) -> OFF boot (same). Pool = live, tier off. Promotion (tier sequence, agentic-edit, long identity) only if it wins.
# GPU TIMEBOX 25 min. RUN (queued): popped by r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r559-ngram-prefetch; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/ngram-prefetch-r1
MP=/srv/qwen5090/probes/mp_decode.py
FB=/srv/qwen5090/probes/fn_bench.py
CFG=/srv/qwen5090/flashnext-config.yml
BIMG=tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16
NIMG=tabbyapi:ngram-prefetch-r1-gdnbf16
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SALT=$(( $(date +%s) % 100000 ))
BOOTED=0
log(){ echo "$(date -Is) [r559] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){
  if [ "$BOOTED" = 1 ]; then sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done; log "daily: $(served_id)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R559 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$MP" "$FB" "$SRC/Dockerfile.box"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
sudo docker image inspect "$BIMG" >/dev/null 2>&1 || { log "ABORT: base image $BIMG missing"; exit 3; }
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE")
BENV=$(echo "$LENV" | tr ' ' '\n' | grep -vE '^EXL3_(EMBED_MIRROR_DEVICE|ADAPTIVE_DRAFT2)' | tr '\n' ' ' | sed 's/ $//')
df -h / | tail -1 | tee -a "$R/audit.log"
log "S1 building $NIMG on $BIMG (before the lock)"
(cd "$SRC" && sudo docker build --build-arg BASE="$BIMG" -f Dockerfile.box -t "$NIMG" . ) > "$R/build.log" 2>&1 \
  || { log "ABORT: build: $(grep -aiE 'error|mismatch|abort|refus' "$R/build.log" | head -3 | tr '\n' ' ' | cut -c1-300)"; exit 3; }
export GPU_QUEUE_NAME=r559-ngram-prefetch
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
for G in 0 1; do
  timeout 600 sudo docker run --rm --gpus all --entrypoint python3 -v "$R":/out "$NIMG" /opt/ngram-prefetch-r1/tests/gpu_ngram_prefetch.py \
    --device cuda:$G --json /out/gpu$G.json > "$R/harness-gpu$G.log" 2>&1; h=$?
  log "harness cuda:$G exit $h: $(tail -1 "$R/harness-gpu$G.log" | cut -c1-160)"; [ $h = 0 ] || { finish FAILED; exit 1; }
done
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$NEWM" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
greedy30k(){ python3 - "$API" "$NEWM" "$R/greedy30k-$1.json" <<'PY' 2>"$R/greedy30k-$1.err" || echo none
import json,sys,hashlib,urllib.request,random
api,model,out=sys.argv[1:4]
rng=random.Random(4242); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
body=" ".join(rng.choice(words) for _ in range(23000))
msg=body+"\n\nSummarize the passage above in one sentence, then write a haiku about it."
req=json.dumps({"model":model,"temperature":0,"max_tokens":128,"min_tokens":128,"messages":[{"role":"user","content":msg}]}).encode()
d=json.loads(urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=req,headers={"Content-Type":"application/json"}),timeout=900).read())
open(out,"w").write(json.dumps(d)); m=d["choices"][0]["message"]
print(hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16])
PY
}
up(){ local tag=$1 i st lp; shift
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" NVME_TIER= IMG="$NIMG" "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 ) & lp=$!
  for i in $(seq 200); do
    [ "$(served_id)" = "$NEWM" ] && { wait $lp; return 0; }
    st=$(cstatus); case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 40 ] && { sleep 3; continue; }
      log "NO BOOT $tag ($st): $(sudo docker logs --tail 80 flashnext 2>&1 | grep -aE 'RuntimeError|Error|memory' | tail -1 | cut -c1-160)"
      kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1;; esac
    [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 0)" -ge 1 ] && {
      log "NO BOOT $tag (restart loop)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1; }
    sleep 3; done
  log "NO BOOT $tag (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }
prefill(){ local tag=$1 c
  for c in 60000 120000; do
    python3 "$FB" --url "$API" --model "$NEWM" --tag "pf-$tag-$c" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c --unique \
      --salt $(( SALT + ${#tag}*13 + c/1000 + RANDOM )) --out "$R/prefill.jsonl" > /dev/null 2>&1
    python3 -c 'import json,sys; r=[x for x in map(json.loads, open(sys.argv[1])) if x.get("tag")==sys.argv[2]][-1]; pt=r["prompt_tokens"]; print("cold prefill %s: %d tokens, %.0f tok/s" % (sys.argv[2], pt, pt/r["ttft_s"]))' \
      "$R/prefill.jsonl" "pf-$tag-$c" 2>&1 | tee -a "$R/audit.log"; done; }
timing(){ local tag=$1
  python3 "$FB" --url "$API" --model "$NEWM" --tag "t-$tag-c1" --kind code --tokens 512 --conc 1 --runs 2 --out "$R/timing-rounds.jsonl" > /dev/null 2>&1
  python3 "$FB" --url "$API" --model "$NEWM" --tag "t-$tag-c4" --kind code --tokens 512 --conc 4 --runs 2 --out "$R/timing-rounds.jsonl" > /dev/null 2>&1
  sleep 12; sudo docker logs flashnext > "$R/docker-$tag.log" 2>&1
  grep -a "ngram timing" "$R/docker-$tag.log" | tail -4 | cut -c1-300 | sed "s/^/[$tag] /" | tee -a "$R/audit.log"; }
fp(){ local tag=$1 a b; a=$(greedy "$tag"); b=$(greedy30k "$tag"); echo "$a $b"; }
log "=== T-OFF: flag off + timing instrument ==="
up T-OFF EXTRA_ENV="$BENV EXL3_NGRAM_TIMING=1 EXL3_NGRAM_TIMING_INTERVAL=10" || { finish FAILED; exit 3; }
read REF1 REF30 < <(fp T-OFF); log "T-OFF: VRAM free $(vram); fingerprints $REF1 / $REF30"
[ "$REF1" = f4add302e176d78e ] && [ "$REF30" = 4a255910dee2d9c5 ] || log "NOTE: T-OFF fingerprints differ from the R548 canonical f4add302 / 4a255910"
prefill T-OFF; timing T-OFF
log "=== T-ON: flag on (timing implied) ==="
up T-ON EXTRA_ENV="$BENV EXL3_NGRAM_PREFETCH2=1 EXL3_NGRAM_TIMING_INTERVAL=10" || { finish FAILED; exit 3; }
read a b < <(fp T-ON); log "T-ON: VRAM free $(vram); fingerprints $a / $b"
[ "$a" = "$REF1" ] && [ "$b" = "$REF30" ] || { log "STOP: flag-on fingerprints differ from flag-off: a bug, not a trade-off"; finish FAILED; exit 1; }
prefill T-ON; timing T-ON
log "=== AB: one boot per arm, ON then OFF: salted prefill + mp_decode ==="
up B1 EXTRA_ENV="$BENV EXL3_NGRAM_PREFETCH2=1" || { finish FAILED; exit 3; }
prefill B1; python3 "$MP" run --url "$API" --model "$NEWM" --tag B1 --tokens 512 --out "$R/mp.jsonl" 2>&1 | tail -2 | tee -a "$R/audit.log"
up A2 EXTRA_ENV="$BENV" || { finish FAILED; exit 3; }
prefill A2; python3 "$MP" run --url "$API" --model "$NEWM" --tag A2 --tokens 512 --out "$R/mp.jsonl" 2>&1 | tail -2 | tee -a "$R/audit.log"
python3 "$MP" compare "$R/mp.jsonl" 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
finish DONE
