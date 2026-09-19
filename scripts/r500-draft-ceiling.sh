#!/usr/bin/env bash
# R500 — deeper MTP draft ceilings (4 / 5) with and without exllamav3 dynamic draft (confidence 0.6), vs the served depth 3.
#
# WHY (user 2026-09-18: "Higher draft ceiling under dynamic draft ... try these"): vcruz305's native exllamav3 recipe runs
# draft depth 5 with dynamic draft at confidence 0.6 (GB10). Constraints on our stack: (1) verify rows = jobs x (depth + 1)
# must stay <= 16 (MoE decode tier MAX_BSZN 16) — depth 5 only up to 2 jobs (12 rows), depth 4 up to 3 jobs (15 rows), c4 keeps
# depth 3 (16 rows); (2) the GDN recurrent-state history is sized by max(DRAFT, policy depths): depth 5 = 6 fp32 copies per slot
# instead of 4, ~+0.9 GiB, so 360,448 will not fit. The unit first finds the largest pool the deepest arm boots at (360,448
# down in 16,384 steps) and runs EVERY arm at that pool, so arms differ only in the draft policy / dynamic flag.
# IMAGE tabbyapi:draftconf (R497: daily image + EXL3_DRAFT_CONFIDENCE), launcher-r498 (= live r491 + DYN + CKPT_NAME).
# ARMS (policy | dynamic | confidence):
#   FIX3   [[4,3],[8,1]]        off        (served)
#   FIX5   [[2,5],[4,3],[8,1]]  off
#   D60C4  [[3,4],[4,3],[8,1]]  on  0.6
#   D60C5  [[2,5],[4,3],[8,1]]  on  0.6
#   FIX3b / D60C5b repeats
# each: c1 greedy fingerprint (recorded; FIX3 at the served pool placement is the canonical control), fn_bench code + prose
# c1/c4 2,048 x 2, multiprompt 12 x kind 2,048 c1/c4 (the served sampler; dynamic draft calibrates on it). c2 is where the
# depth-5 policy also applies: fn_bench adds --conc 2.
# RUN: sudo systemd-run --unit=r500-draft-ceiling --collect -p RuntimeMaxSec=86400 -E HOME=$HOME /bin/bash /srv/qwen5090/r500-draft-ceiling.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r500-draft-ceiling; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r498.sh
CFG=/srv/qwen5090/flashnext-config.yml
CTX=/srv/qwen5090/build/draftconf
IMG=tabbyapi:draftconf
BASE_ENV="EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r500] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
finish(){ sudo docker rm -f r500-test >/dev/null 2>&1
  if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged live launcher)"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
    log "daily: $(served_id) $(cfgline) VRAM free $(vram)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R500 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$LM" "$CTX/Dockerfile.box" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/multiprompt.py; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r500-draft-ceiling
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"

if ! sudo docker image inspect "$IMG" >/dev/null 2>&1; then
log "=== build $IMG (daily keeps serving) ==="
( cd "$CTX" && sudo docker build -f Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "BUILD FAILED: $(grep -aE 'error|Error' "$R/build.log" | tail -3 | cut -c1-240)"; finish ABORTED; exit 3; }
log "built $IMG $(sudo docker image inspect -f '{{.Id}}' "$IMG" | cut -c8-19)"
fi
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
for i in $(seq 30); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 2; done
# boot TAG REPLAY CACHE -> 0 serving with that config and image, 1 no boot, 2 mismatch
boot(){ local tag=$1 dyn=$2 conf=$3 pol=$4 cache=${5:-$POOL} i st lp got
  log "boot $tag: policy $pol dynamic_draft $dyn confidence $conf cache $cache"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" IMG="$IMG" CACHE=$cache DYN=$dyn DRAFT_POLICY="$pol" EXTRA_ENV="$BASE_ENV EXL3_DRAFT_CONFIDENCE=$conf" bash "$CAND" >> "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
  for i in $(seq 360); do
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp; got=$(cfgline)
      case "$got" in *"cache_size: $cache cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;; *) log "ABORT: config '$got'"; return 2;; esac
      [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = "$IMG" ] || { log "ABORT: container image is not $IMG"; return 2; }
      [ "$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c "^EXL3_DRAFT_CONFIDENCE=$conf$")" = 1 ] || { log "ABORT: container env lacks EXL3_DRAFT_CONFIDENCE=$conf"; return 2; }
      sudo grep -q "^  dynamic_draft: $dyn " $CFG || { log "ABORT: config lacks dynamic_draft: $dyn"; return 2; }
      sudo grep -qF "draft_num_tokens_by_batch: $pol" $CFG || { log "ABORT: config lacks policy $pol"; return 2; }
      log "UP $tag @ $cache: VRAM free MiB $(vram)"; return 0; fi
    st=$(cstatus); case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 60 ] && { sleep 3; continue; }
      log "NO BOOT $tag @ $cache ($st): $(sudo docker logs --tail 60 flashnext 2>&1 | grep -aE 'RuntimeError|Error|VRAM' | tail -1 | cut -c1-220)"
      sudo docker logs flashnext > "$R/boot-$tag-$cache.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1;; esac
    [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 0)" -ge 1 ] && {
      log "NO BOOT $tag @ $cache (restart loop): $(sudo docker logs --tail 60 flashnext 2>&1 | grep -aE 'RuntimeError|Error|VRAM' | tail -1 | cut -c1-220)"
      sudo docker logs flashnext > "$R/boot-$tag-$cache.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1; }
    sleep 3; done
  log "NO BOOT $tag @ $cache (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
greedy30k(){ python3 - "$API" "$MODEL" "$R/greedy30k-$1.json" <<'PY' 2>/dev/null || echo none
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
fp(){ local a b; a=$(greedy "$1"); b=$(greedy30k "$1"); log "[$1] fingerprints c1 $a / 30k $b (canonical 1474eee2f5945248 / 4a255910dee2d9c5)"
  [ "$a" = 1474eee2f5945248 ] && [ "$b" = 4a255910dee2d9c5 ]; }
speed(){ local t=$1 kind
  for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$t-$kind" --kind $kind --tokens 2048 --conc 1 2 4 --runs 2 \
      --out "$R/records-$t.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$t $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done
  python3 /srv/qwen5090/probes/multiprompt.py --url "$API" --model "$MODEL" --tag "$t" --tokens 2048 --conc 1 2 4 --out "$R/multiprompt.jsonl" 2>&1 | tee -a "$R/audit.log"
}

P3='[[4, 3], [8, 1]]'; P4='[[3, 4], [4, 3], [8, 1]]'; P5='[[2, 5], [4, 3], [8, 1]]'
log "=== pool for the deepest arm (policy $P5 -> GDN history 6) ==="
POOL=0
for c in 360448 344064 327680 311296 294912 278528 262144; do
  boot PROBE true 0.6 "$P5" $c; rc=$?; [ $rc = 2 ] && { finish ABORTED; exit 3; }; [ $rc = 0 ] && { POOL=$c; break; }; done
[ "$POOL" = 0 ] && { log "ABORT: depth-5 policy does not boot at >= 262,144"; finish ABORTED; exit 3; }
log "=== all arms at pool $POOL (served 360,448) ==="
for arm in "FIX3|false|0.4|$P3" "FIX5|false|0.4|$P5" "D60C4|true|0.6|$P4" "D60C5|true|0.6|$P5" "FIX3b|false|0.4|$P3" "D60C5b|true|0.6|$P5"; do
  IFS='|' read -r tag dyn conf pol <<< "$arm"
  boot "$tag" "$dyn" "$conf" "$pol" || { finish ABORTED; exit 3; }
  fp "$tag" || log "[$tag] greedy differs from canonical (pool $POOL placement / verify shapes; recorded)"
  speed "$tag"
done
finish DONE
