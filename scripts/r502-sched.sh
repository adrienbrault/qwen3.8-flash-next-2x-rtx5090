#!/usr/bin/env bash
# R502 — codex sched-r1 (after MiaAI-Lab's GLM-5.3-Flash kit): acceptance-driven draft depth (EXL3_ADAPTIVE_DRAFT=1, per-job
# EMA of accepted/proposed, alpha 0.25, down/up 0.45/0.75, depth <= the policy ceiling) and decode-first fair prefill
# (EXL3_DECODE_FLOOR=1, at most EXL3_DECODE_FLOOR_MAX_PREFILL_CHUNKS=1 prefill chunk between decode steps while decodes wait).
# WHY (user 2026-09-18: adaptive depth "try these"; decode-first fair prefill "worth trying").
# IMAGE tabbyapi:sched-r1 = the daily decode-kernels-r2 + Python overlay (flan/patches/exllamav3/sched/r1/Dockerfile.box).
# Launcher-r498 (= live r491 + DYN + CKPT_NAME), overlap on in every arm.
#   block A  adaptive OFF / ON / OFF2 / ON2 (dynamic_draft off): fingerprints canonical on EVERY arm, fn_bench code + prose
#            c1/c4 2,048 x 2, multiprompt 12 x kind 2,048 c1/c4; then a composition smoke (dynamic_draft on + adaptive):
#            fingerprints recorded
#   block B  decode floor OFF / ON / OFF2 / ON2: probe_decode_floor.py (3 sampled 2,048-token streams + a ~100k-token prompt
#            arriving mid-stream; per-stream inter-token latency p50/p99/max during the prefill window, long-request TTFT),
#            fingerprints canonical (the floor only reorders work)
# RUN: sudo systemd-run --unit=r502-sched --collect -p RuntimeMaxSec=129600 -E HOME=$HOME /bin/bash /srv/qwen5090/r502-sched.sh
# 2026-09-18 23:30 CEST (user: "Stop using 3.05. From now on stay on 2.5"): moved to the 2.50bpw daily — checkpoint
# qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab, pool 786,432, launcher-r511, canonical c1 ae890c45d1000582 / 30k 2aa8d1024daece5c (R511).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r502-sched; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r511.sh
CFG=/srv/qwen5090/flashnext-config.yml
CTX=/srv/qwen5090/build/sched-r1
IMG=tabbyapi:sched-r1
BASE_ENV="EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r502] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
finish(){ sudo docker rm -f r502-test >/dev/null 2>&1
  if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged live launcher)"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done   # any daily (R511 changes the served id)
    log "daily: $(served_id) $(cfgline) VRAM free $(vram)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R502 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$CTX/probe_decode_floor.py" "$LM" "$CTX/Dockerfile.box" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/multiprompt.py; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r502-sched
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"

log "=== build $IMG (daily keeps serving) ==="
( cd "$CTX" && sudo docker build -f Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "BUILD FAILED: $(grep -aE 'error|Error' "$R/build.log" | tail -3 | cut -c1-240)"; finish ABORTED; exit 3; }
log "built $IMG $(sudo docker image inspect -f '{{.Id}}' "$IMG" | cut -c8-19)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
for i in $(seq 30); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 2; done
# boot TAG REPLAY CACHE -> 0 serving with that config and image, 1 no boot, 2 mismatch
boot(){ local tag=$1 dyn=$2 sel=$3 cache=786432 i st lp got
  log "boot $tag: dynamic_draft $dyn env: ${sel:-none}"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" IMG="$IMG" CACHE=$cache DYN=$dyn EXTRA_ENV="$BASE_ENV $sel" bash "$CAND" >> "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
  for i in $(seq 360); do
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp; got=$(cfgline)
      case "$got" in *"cache_size: $cache cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;; *) log "ABORT: config '$got'"; return 2;; esac
      [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = "$IMG" ] || { log "ABORT: container image is not $IMG"; return 2; }
      [ "$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c "^EXL3_SHARED_EXPERT_OVERLAP=1$")" = 1 ] || { log "ABORT: container env lacks the overlap flag"; return 2; }
      for kv in $sel; do sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -qx "$kv" || { log "ABORT: container env lacks $kv"; return 2; }; done
      sudo grep -q "^  dynamic_draft: $dyn " $CFG || { log "ABORT: config lacks dynamic_draft: $dyn"; return 2; }
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
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$t-$kind" --kind $kind --tokens 2048 --conc 1 4 --runs 2 \
      --out "$R/records-$t.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$t $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done
  python3 /srv/qwen5090/probes/multiprompt.py --url "$API" --model "$MODEL" --tag "$t" --tokens 2048 --conc 4 --out "$R/multiprompt.jsonl" 2>&1 | tee -a "$R/audit.log"
}

AD="EXL3_ADAPTIVE_DRAFT=1"; DF="EXL3_DECODE_FLOOR=1 EXL3_DECODE_FLOOR_MAX_PREFILL_CHUNKS=1"
log "=== block A: acceptance-driven draft depth ==="
for arm in "A-OFF|" "A-ON|$AD" "A-OFF2|" "A-ON2|$AD"; do
  tag=${arm%%|*}; sel=${arm#*|}
  boot "$tag" false "$sel" || { finish ABORTED; exit 3; }
  # OFF arms must be canonical; ON varies the per-step draft depth, so its verify shapes (and greedy tokens) move, as with
  # dynamic draft in R497 — recorded, not required (first try aborted on A-ON 512c74fcc7204cff, 2026-09-18 20:10 UTC)
  if [ -z "$sel" ]; then fp "$tag" || { log "[$tag] FINGERPRINT NOT CANONICAL (OFF arm)"; finish ABORTED; exit 3; }
  else fp "$tag" || log "[$tag] greedy differs from canonical (variable draft depth changes verify shapes; recorded)"; fi
  speed "$tag"
  sudo docker logs flashnext > "$R/docker-$tag.log" 2>&1
  python3 - "$R/docker-$tag.log" "$tag" <<'PY' 2>&1 | tee -a "$R/audit.log"
import re, sys
t = re.sub(r"\s+", " ", open(sys.argv[1], errors="replace").read())
m = re.findall(r"draft (\d+)/(\d+) accepted", t)
a, b = sum(int(x) for x, _ in m), sum(int(y) for _, y in m)
print(f"[{sys.argv[2]}] draft acceptance over {len(m)} requests: {a}/{b} = {a / b:.3f}" if b else f"[{sys.argv[2]}] no draft lines")
PY
  grep -aE "GPU assert|out of memory" "$R/docker-$tag.log" | head -2 | sed "s/^/[$tag] /" | tee -a "$R/audit.log"
  [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ] && [ "$(served_id)" = "$MODEL" ] \
    && log "[$tag] alive after the arm" || log "[$tag] CRASHED during the arm (R497-class graph OOM if the assert line above says so)"
done
log "=== composition smoke: dynamic_draft on + adaptive ==="
boot A-DYN true "$AD" || { finish ABORTED; exit 3; }
fp A-DYN || log "[A-DYN] fingerprint differs from canonical (dynamic draft changes verify shapes; recorded)"
log "=== block B: decode floor (3 streams + one ~100k prompt mid-stream) ==="
for arm in "B-OFF|" "B-ON|$DF" "B-OFF2|" "B-ON2|$DF"; do
  tag=${arm%%|*}; sel=${arm#*|}
  boot "$tag" false "$sel" || { finish ABORTED; exit 3; }
  fp "$tag" || { log "[$tag] FINGERPRINT NOT CANONICAL"; finish ABORTED; exit 3; }
  python3 "$CTX/probe_decode_floor.py" --url "$API" --model "$MODEL" --tag "$tag" --short-tokens 2048 --long-prompt-tokens 100000 \
    --out "$R/decode-floor.jsonl" > "$R/decode-floor-$tag.log" 2>&1 || log "[$tag] probe exited non-zero"
  tail -8 "$R/decode-floor-$tag.log" | cut -c1-240 | tee -a "$R/audit.log"
done
finish DONE
