#!/usr/bin/env bash
# R511 — promote r0b0tlab's 2.50bpw pack to the Flash-Next daily (:8022) at a 786,432-token pool (user 2026-09-18: "Alright
# lets do 2.5 bpw"). Evidence so far: R495b (decode +3-8 % except prose c4 flat, needles 5/5, 4 x 141.6k in flight, tool-eval
# 85.5 +- 1.3 = control band) and R509 (GSM8K without stop strings 0.978 vs 0.980).
# This unit boots the exact candidate launcher (launch-flashnext-r511.sh with NO env = the full daily config), records the new
# canonical fingerprints, and gates before installing it:
#   G1 boots at 786,432 with the served image / flags / 4 slots / vision / chunk 2048
#   G2 agentic-edit (6 real files rewritten with a small edit) greedy + sampled, c1 + c4: every request ok, and the server alive
#   G3 needles 131k / 240k: 5/5 each
#   G4 tool-eval 69 x 4 at parallel 8: mean >= 82.0 (today's controls on this box: R491b 83.2 / 85.0 / 84.8 / 82.2; R495b 2.50bpw 85.5)
# PASS -> launch-flashnext.sh := r511 (old live kept as launch-flashnext.sh.pre-r511; rollback = launch-flashnext-r491.sh),
# daily restarted from it and asserted (id, cache 786,432). FAIL -> the 3.05bpw daily is restored unchanged.
# RUN: sudo systemd-run --unit=r511-promote-2p50 --collect -p RuntimeMaxSec=172800 -E HOME=$HOME /bin/bash /srv/qwen5090/r511-promote-2p50.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r511-promote-2p50; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r511.sh
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0; PROMOTED=0
log(){ echo "$(date -Is) [r511] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
finish(){
  if [ "$BOOTED" = 1 ] && [ "$PROMOTED" = 0 ]; then log "restoring the unchanged 3.05bpw daily"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done
    log "daily: $(served_id) $(cfgline)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R511 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" /srv/qwen5090/models/$NEWM/config.json /srv/qwen5090/probes/agentic-edit.py /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/tooleval_summary.py; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r511-promote-2p50
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"

log "=== G1: boot the candidate launcher with no env (= the full daily config) ==="
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
"${CLEAN_ENV[@]}" bash "$CAND" > "$R/boot.log" 2>&1 || { log "G1 FAIL: launcher exit $(tail -2 "$R/boot.log" | tr '\n' ' ' | cut -c1-200)"; finish ABORTED; exit 3; }
for i in $(seq 120); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done
got=$(cfgline)
case "$got" in *"cache_size: 786432 cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;;
  *) log "G1 FAIL: served '$(served_id)' config '$got'"; finish ABORTED; exit 3;; esac
log "G1 PASS: $(served_id) $got; image $(sudo docker ps --format '{{.Image}}' -f name=flashnext); VRAM free MiB $(vram)"

greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$NEWM" "$PROMPT")" > "$R/greedy.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy.json" 2>/dev/null || echo none; }
greedy30k(){ python3 - "$API" "$NEWM" "$R/greedy30k.json" <<'PY' 2>/dev/null || echo none
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
a=$(greedy); b=$(greedy30k); log "NEW CANONICAL fingerprints (2.50bpw @786,432, daily image/flags): c1 $a / 30k $b"
b2=$(greedy30k); [ "$b" = "$b2" ] && log "30k fingerprint repeats ($b2)" || log "WARNING: 30k fingerprint not repeatable ($b vs $b2)"

log "=== G2: agentic-edit greedy + sampled, c1 + c4 ==="
python3 /srv/qwen5090/probes/agentic-edit.py --url "$API" --model "$NEWM" --tag P250 --conc 1 4 --modes greedy sampled --out "$R/agentic-edit.jsonl" 2>&1 | tee "$R/agentic-edit.log" | grep -a "agentic-edit" | tee -a "$R/audit.log"
n_ok=$(grep -ac "6/6 ok" "$R/agentic-edit.log")
if [ "$n_ok" = 4 ] && alive; then log "G2 PASS (4/4 modes 6/6 ok, server alive)"; else log "G2 FAIL ($n_ok/4 modes fully ok; alive: $(alive && echo yes || echo no))"; finish ABORTED; exit 3; fi

log "=== G3: needles 131k / 240k ==="
python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$NEWM" --tag needle --ctx-tokens 131072 240000 \
  --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee "$R/needle.log" | tee -a "$R/audit.log"
[ "$(grep -c '5/5 retrieved' "$R/needle.log")" = 2 ] && log "G3 PASS" || { log "G3 FAIL"; finish ABORTED; exit 3; }

log "=== G4: tool-eval 69 x 4, parallel 8 ==="
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$NEWM" --temperature 0.6 --top-p 0.95 --top-k 20 \
    --trials 4 --parallel 8 --json-file "$R/tooleval.json" > "$R/tooleval.log" 2>&1 )
python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval.json" P250 2>&1 | tee "$R/tooleval-summary.txt" | tee -a "$R/audit.log"
mean=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trial_statistics"]["final_score_mean"])' "$R/tooleval.json" 2>/dev/null)
log "tool-eval mean: ${mean:-unparsed}"
if [ -n "$mean" ] && python3 -c "import sys; sys.exit(0 if float('$mean') >= 82.0 else 1)"; then log "G4 PASS"; else log "G4 FAIL (mean '${mean:-unparsed}')"; finish ABORTED; exit 3; fi
alive || { log "server not alive after the gates"; finish ABORTED; exit 3; }

log "=== PROMOTE: launch-flashnext.sh := launch-flashnext-r511.sh ==="
sudo cp -p "$LIVE" /srv/qwen5090/launch-flashnext.sh.pre-r511
cp "$CAND" /srv/qwen5090/launch-flashnext.sh.new && mv /srv/qwen5090/launch-flashnext.sh.new "$LIVE"
cmp -s "$CAND" "$LIVE" || { log "PROMOTE FAIL: live launcher differs from r511"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r511 "$LIVE"; finish ABORTED; exit 3; }
PROMOTED=1
sudo docker rm -f flashnext >/dev/null 2>&1
"${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || { log "LIVE LAUNCH FAILED — rolling back"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r511 "$LIVE"; PROMOTED=0; finish ABORTED; exit 3; }
for i in $(seq 120); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done
a2=$(greedy); log "daily now: $(served_id) $(cfgline); c1 fingerprint $a2 (promotion-boot canonical $a)"
[ "$(served_id)" = "$NEWM" ] || { log "daily did not come up as $NEWM — rolling back"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r511 "$LIVE"; PROMOTED=0; finish ABORTED; exit 3; }
finish PROMOTED
