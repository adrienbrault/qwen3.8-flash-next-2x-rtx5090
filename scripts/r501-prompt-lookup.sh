#!/usr/bin/env bash
# R501 — prompt lookup beside the MTP draft (codex prompt-lookup r1, EXL3_PROMPT_LOOKUP=1), after peonist-ai/halogen 0.6.0
# (coding-agent turns +13–15 %, function calling +9 %, prose flat; greedy byte-identical).
# WHY (user 2026-09-18: prompt lookup "still worth trying"). Design (flan/patches/exllamav3/prompt-lookup/r1/design-source.md):
# in-window replacement — the MTP draft opens the chain, a 3-token suffix match in the request's own context fills the
# remaining positions of the EXISTING depth (policy [[4,3],[8,1]]), so c4 stays at 16 verify rows and the GDN history is
# unchanged; verification unchanged (lossless for greedy and sampled, codex's proof); counters in TabbyAPI's metrics log.
# IMAGE tabbyapi:prompt-lookup-r1 = the daily decode-kernels-r2 + Python overlay (Dockerfile.box). Launcher-r498 (= live r491).
#   gate H  in-process real-checkpoint harness (model_prompt_lookup_probe.py: coding-agent file edit, exact-path tool call,
#           prose): greedy c1 OFF / OFF-b / ON — OFF vs OFF-b is the determinism control (R489b: fresh containers without a
#           frozen tune cache are not deterministic); if OFF == OFF-b, every ON (greedy c1/c4, sampled c1/c4) must equal
#           its OFF, else the harness is inconclusive and the served fingerprints decide
#   serve   OFF / ON / OFF2 / ON2: fingerprints canonical on EVERY arm, fn_bench code + prose c1/c4 2,048 x 2, multiprompt
#           12 x kind 2,048 c1/c4, probes/agentic-edit.py (6 real files rewritten with a small edit, greedy + sampled, c1/c4)
#           + the lookup counters from TabbyAPI's log
# RUN: sudo systemd-run --unit=r501-prompt-lookup --collect -p RuntimeMaxSec=86400 -E HOME=$HOME /bin/bash /srv/qwen5090/r501-prompt-lookup.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r501-prompt-lookup; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r498.sh
CFG=/srv/qwen5090/flashnext-config.yml
CTX=/srv/qwen5090/build/prompt-lookup-r1
IMG=tabbyapi:prompt-lookup-r1
BASE_ENV="EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r501] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
finish(){ sudo docker rm -f r501-test >/dev/null 2>&1
  if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged live launcher)"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
    log "daily: $(served_id) $(cfgline) VRAM free $(vram)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R501 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$LM" "$CTX/Dockerfile.box" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/multiprompt.py /srv/qwen5090/probes/agentic-edit.py; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r501-prompt-lookup
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"

log "=== build $IMG (daily keeps serving) ==="
( cd "$CTX" && sudo docker build -f Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "BUILD FAILED: $(grep -aE 'error|Error' "$R/build.log" | tail -3 | cut -c1-240)"; finish ABORTED; exit 3; }
log "built $IMG $(sudo docker image inspect -f '{{.Id}}' "$IMG" | cut -c8-19)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
for i in $(seq 30); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 2; done
log "=== gate H: in-process harness, OFF vs OFF-b determinism control, then OFF vs ON ==="
EV=(); for kv in $BASE_ENV; do EV+=(-e "$kv"); done
mkdir -p "$R/h"; sudo chmod 777 "$R/h"
harness(){ local tag=$1 flag=$2 mode=$3 conc=$4
  sudo docker run --rm --name r501-test --gpus all --ipc=host --entrypoint bash -v "$CKPT":/models:ro -v "$R/h":/results "${EV[@]}" \
    -e EXL3_PROMPT_LOOKUP=$flag -e EXL3_PROMPT_LOOKUP_MATCH=3 -e EXL3_PROMPT_LOOKUP_CONTINUATION=3 "$IMG" \
    -lc "python3 /opt/prompt-lookup-r1/tests/model_prompt_lookup_probe.py --model /models --vision --mode $mode --concurrency $conc --tokens 256 --cache-size 8192 --max-batch-size 4 --max-history 3 --chunk-size 2048 --target-cache-mode 8,8 --draft-cache-mode FP16 --target-use 30,30 --draft-use 30,30 --vision-use 30,30 --output /results/$tag.json" \
    > "$R/h/$tag.log" 2>&1 || { log "harness $tag exited non-zero: $(tail -2 "$R/h/$tag.log" | tr '\n' ' ' | cut -c1-240)"; return 1; }; }
hcmp(){ python3 "$CTX/tests/compare_prompt_lookup.py" "$R/h/$1.json" "$R/h/$2.json" > "$R/h/cmp-$1-$2.log" 2>&1; local rc=$?
  log "compare $1 vs $2: rc $rc $(tail -2 "$R/h/cmp-$1-$2.log" | tr '\n' ' ' | cut -c1-240)"; return $rc; }
harness greedy-c1-off 0 greedy 1 && harness greedy-c1-offb 0 greedy 1 && harness greedy-c1-on 1 greedy 1 || { finish ABORTED; exit 3; }
grep -aE "lookup|hit|accept" "$R/h/greedy-c1-on.log" | tail -3 | cut -c1-200 | tee -a "$R/audit.log"
if hcmp greedy-c1-off greedy-c1-offb; then
  log "harness deterministic: ON must equal OFF everywhere"
  hcmp greedy-c1-off greedy-c1-on || { log "GATE H FAIL greedy c1"; finish ABORTED; exit 3; }
  for mc in "greedy 4" "sampled 1" "sampled 4"; do set -- $mc
    harness $1-c$2-off 0 $1 $2 && harness $1-c$2-on 1 $1 $2 || { finish ABORTED; exit 3; }
    hcmp $1-c$2-off $1-c$2-on || { log "GATE H FAIL $1 c$2"; finish ABORTED; exit 3; }; done
else
  log "harness NOT deterministic OFF vs OFF (R489b class): gate H inconclusive, the served fingerprints decide"
fi
# boot TAG REPLAY CACHE -> 0 serving with that config and image, 1 no boot, 2 mismatch
boot(){ local tag=$1 rp=$2 cache=$3 i st lp got
  log "boot $tag: prompt lookup $rp cache $cache"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" IMG="$IMG" CACHE=$cache EXTRA_ENV="$BASE_ENV EXL3_PROMPT_LOOKUP=$rp" bash "$CAND" >> "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
  for i in $(seq 360); do
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp; got=$(cfgline)
      case "$got" in *"cache_size: $cache cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;; *) log "ABORT: config '$got'"; return 2;; esac
      [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = "$IMG" ] || { log "ABORT: container image is not $IMG"; return 2; }
      [ "$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c "^EXL3_PROMPT_LOOKUP=$rp$")" = 1 ] || { log "ABORT: container env lacks EXL3_PROMPT_LOOKUP=$rp"; return 2; }
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
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$t-$kind" --kind $kind --tokens 2048 --conc 1 4 --runs 2 \
      --out "$R/records-$t.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$t $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done
  python3 /srv/qwen5090/probes/multiprompt.py --url "$API" --model "$MODEL" --tag "$t" --tokens 2048 --conc 1 4 --out "$R/multiprompt.jsonl" 2>&1 | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/agentic-edit.py --url "$API" --model "$MODEL" --tag "$t" --conc 1 4 --modes greedy sampled --out "$R/agentic-edit.jsonl" 2>&1 | tee -a "$R/audit.log"
  sudo docker logs flashnext 2>&1 | grep -aE "lookup [0-9]+/[0-9]+ rounds" | tail -200 > "$R/lookup-$t.log"
  log "[$t] lookup counter lines: $(wc -l < "$R/lookup-$t.log"); last: $(tail -1 "$R/lookup-$t.log" | cut -c1-200)"
}

for arm in "OFF|0" "ON|1" "OFF2|0" "ON2|1"; do
  IFS='|' read -r tag f <<< "$arm"
  boot "$tag" "$f" 360448 || { finish ABORTED; exit 3; }
  fp "$tag" || { log "FINGERPRINT FAIL on $tag"; finish ABORTED; exit 3; }
  speed "$tag"
done
finish DONE
