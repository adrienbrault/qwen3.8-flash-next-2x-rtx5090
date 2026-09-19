#!/usr/bin/env bash
# R506 — GDN state replay (codex gdn-state r1+r3) on TOP of the current daily (decode-kernels-r2, shared-expert overlap): isolate
# the replay cost from the pool change and collect the promotion evidence. R496 (on the older moecoopv2 base): every identity gate
# PASS (harness OFF=OFF, OFF=ON locked, served canonical at 360,448 AND at the unlocked 425,984 with an unchanged 26/26 placement),
# pool 360,448 -> 425,984 (+18 %), GSM8K 0.925, needles 5/5, prefill equal — but fn_bench prose c1 158 vs 170 t/s and c4
# 396 vs 427 (−7..8 %) while multiprompt prose was flat/up. Greedy output is identical, so a prose gap is per-step cost.
# IMAGE tabbyapi:gdn-state-r3-r2 = decode-kernels-r2 + gdnstate r1 overlay + r3 overlay (built by this unit). Launcher-r498.
# ARMS: OFF@360,448 / ON@360,448 / ON@425,984 / OFF2@360,448 / ON2@425,984 — fingerprints canonical on EVERY arm, fn_bench code +
# prose c1/c4 2,048 x 3 runs, multiprompt 12 x kind 2,048 c1/c4; then tool-eval 69 x 4 on ON2@425,984 (daily band 83–85, R491b).
# RUN: sudo systemd-run --unit=r506-gdn-replay-r2 --collect -p RuntimeMaxSec=86400 -E HOME=$HOME /bin/bash /srv/qwen5090/r506-gdn-replay-r2.sh
# 2026-09-18 23:30 CEST (user: "Stop using 3.05. From now on stay on 2.5"): moved to the 2.50bpw daily — checkpoint
# qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab, pool 786,432, launcher-r511, canonical c1 ae890c45d1000582 / 30k 2aa8d1024daece5c (R511).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r506-gdn-replay-r2; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r511.sh
CFG=/srv/qwen5090/flashnext-config.yml
CTX=/srv/qwen5090/build/draftconf
IMG=tabbyapi:gdn-state-r3-r2
BASE_ENV="EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r506] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
finish(){ sudo docker rm -f r506-test >/dev/null 2>&1
  if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged live launcher)"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done   # any daily (R511 changes the served id)
    log "daily: $(served_id) $(cfgline) VRAM free $(vram)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R506 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$LM" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/multiprompt.py; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r506-gdn-replay-r2
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"

log "=== build $IMG (r1 then r3 overlays over the daily image; daily keeps serving) ==="
( cd /srv/qwen5090/build/gdnstate-r1 && sudo docker build -f Dockerfile.box --build-arg BASE=tabbyapi:decode-kernels-r2 -t tabbyapi:gdn-state-r1-r2 . ) > "$R/build-r1.log" 2>&1 || { log "BUILD r1 FAILED: $(grep -aE 'error|Error' "$R/build-r1.log" | tail -3 | cut -c1-240)"; finish ABORTED; exit 3; }
( cd /srv/qwen5090/build/gdnstate-r3 && sudo docker build -f Dockerfile.box --build-arg BASE=tabbyapi:gdn-state-r1-r2 -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "BUILD r3 FAILED: $(grep -aE 'error|Error' "$R/build.log" | tail -3 | cut -c1-240)"; finish ABORTED; exit 3; }
log "built $IMG $(sudo docker image inspect -f '{{.Id}}' "$IMG" | cut -c8-19)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
for i in $(seq 30); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 2; done
# boot TAG REPLAY CACHE -> 0 serving with that config and image, 1 no boot, 2 mismatch
boot(){ local tag=$1 rp=$2 cache=$3 i st lp got
  log "boot $tag: replay $rp cache $cache"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" IMG="$IMG" CACHE=$cache EXTRA_ENV="$BASE_ENV EXL3_GDN_STATE_REPLAY=$rp EXL3_GDN_STATE_REPLAY_PLACEMENT_LOCK=0" bash "$CAND" >> "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
  for i in $(seq 360); do
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp; got=$(cfgline)
      case "$got" in *"cache_size: $cache cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;; *) log "ABORT: config '$got'"; return 2;; esac
      [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = "$IMG" ] || { log "ABORT: container image is not $IMG"; return 2; }
      [ "$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c "^EXL3_GDN_STATE_REPLAY=$rp$")" = 1 ] || { log "ABORT: container env lacks EXL3_GDN_STATE_REPLAY=$rp"; return 2; }
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
fp(){ local a b; a=$(greedy "$1"); b=$(greedy30k "$1"); log "[$1] fingerprints c1 $a / 30k $b (canonical ae890c45d1000582 / 2aa8d1024daece5c)"
  [ "$a" = ae890c45d1000582 ] && [ "$b" = 2aa8d1024daece5c ]; }
speed(){ local t=$1 kind
  for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$t-$kind" --kind $kind --tokens 2048 --conc 1 4 --runs 3 \
      --out "$R/records-$t.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$t $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done
  python3 /srv/qwen5090/probes/multiprompt.py --url "$API" --model "$MODEL" --tag "$t" --tokens 2048 --conc 4 --out "$R/multiprompt.jsonl" 2>&1 | tee -a "$R/audit.log"
}

# the replay releases ~0.67 GB per card (R496: +65,536 tokens on 3.05); find the largest ON pool above the 2.50 daily's 786,432
BIG=0
for c in 851968 835584 819200 802816; do boot "PROBE$c" 1 $c; rc=$?; [ $rc = 2 ] && { finish ABORTED; exit 3; }; [ $rc = 0 ] && { BIG=$c; break; }; done
[ "$BIG" = 0 ] && { log "ABORT: replay ON boots at no pool above 786,432"; finish ABORTED; exit 3; }
log "replay ON boots at $BIG (daily 786,432)"
for arm in "OFF|0|786432" "ON|1|786432" "ONBIG|1|$BIG" "OFF2|0|786432" "ONBIGb|1|$BIG"; do
  IFS='|' read -r tag rp cache <<< "$arm"
  boot "$tag" "$rp" "$cache" || { finish ABORTED; exit 3; }
  sudo docker logs flashnext 2>&1 | grep -a "EXL3 layer placement:" | tail -1 | python3 -c 'import sys,json,collections
l=sys.stdin.read(); i=l.find("{"); d=json.loads(l[i:]) if i>=0 else {}; print("placement", dict(collections.Counter(m.get("device") for m in d.get("modules",[]))))' 2>&1 | sed "s/^/[$tag] /" | tee -a "$R/audit.log"
  fp "$tag" || { log "[$tag] FINGERPRINT NOT CANONICAL"; finish ABORTED; exit 3; }
  speed "$tag"
done
log "=== tool-eval on ONBIGb ==="
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$MODEL" --temperature 0.6 --top-p 0.95 --top-k 20 \
    --trials 4 --parallel 8 --json-file "$R/tooleval-ONBIGb.json" > "$R/tooleval-ONBIGb.log" 2>&1 )
python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval-ONBIGb.json" ONBIGb 2>&1 | tee -a "$R/audit.log"
finish DONE
