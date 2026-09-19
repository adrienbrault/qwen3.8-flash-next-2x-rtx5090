#!/usr/bin/env bash
# R517 — promote the stacked image tabbyapi:stack-r4-e3r2 with E3 grouped MoE prefill r2 ON (R513) on the Flash-Next daily.
# Candidate: launch-flashnext-r517.sh (= R514 launcher + stack image + EXL3_MOE_PREFILL_E3=1) if R514 promoted decode r4 group I,
# else launch-flashnext-r517e.sh (= R511 + stack image + E3). Gates:
#   G0 sampler fallbacks byte-identical to the live daily (short prompts: E3 is not engaged below 512 rows)
#   G1 boot, image stack-r4-e3r2, EXL3_MOE_PREFILL_E3=1 in the container; fingerprints c1 ae890c45d1000582 (canonical) and
#      30k 4a255910dee2d9c5 (E3's hash, identical on R507 / R513 ON arms)
#   G1b salted cold prefill 60k >= 9,200 and 120k >= 9,500 tok/s (R513 ON 9,814 / 10,163; OFF 8,115 / 8,543)
#   G2 agentic-edit greedy + sampled c1 + c4 6/6; G3 needles 131k / 240k 5/5; G4 tool-eval 69 x 4 mean >= 82
# PASS -> launch-flashnext.sh := candidate (old live kept as launch-flashnext.sh.pre-r517).
# RUN: sudo systemd-run --unit=r517-promote-stack --collect -p RuntimeMaxSec=172800 -E HOME=$HOME /bin/bash /srv/qwen5090/r517-promote-stack.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r517-promote-stack; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
if cmp -s /srv/qwen5090/launch-flashnext.sh /srv/qwen5090/launch-flashnext-r514.sh; then CAND=/srv/qwen5090/launch-flashnext-r517.sh
else CAND=/srv/qwen5090/launch-flashnext-r517e.sh; fi
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0; PROMOTED=0
log(){ echo "$(date -Is) [r517] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
finish(){
  if [ "$BOOTED" = 1 ] && [ "$PROMOTED" = 0 ]; then log "restoring the unchanged daily"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done
    log "daily: $(served_id) $(cfgline)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R517 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" /srv/qwen5090/models/$NEWM/config.json /srv/qwen5090/probes/agentic-edit.py /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/tooleval_summary.py; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r517-promote-stack
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"

fallback(){ python3 - "$API" "$NEWM" "$R/fallback-$1.json" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,hashlib,urllib.request
api,model,out=sys.argv[1:4]; res={}
cases={"presence":{"temperature":0,"presence_penalty":0.5,"max_tokens":200,"min_tokens":200},
       "rep":{"temperature":0,"repetition_penalty":1.1,"max_tokens":200,"min_tokens":200},
       "stop":{"temperature":0,"max_tokens":400,"stop":["\n\n\n","def test_"]}}
for name,extra in cases.items():
    body={"model":model,"messages":[{"role":"user","content":"Write a Python function that merges overlapping intervals, then tests for it."}],**extra}
    d=json.loads(urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600).read())
    m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or "")
    res[name]=hashlib.sha256(t.encode()).hexdigest()[:16]+f" len={len(t)} fin={d['choices'][0].get('finish_reason')}"
json.dump(res,open(out,"w"),sort_keys=True); print("fallback", json.dumps(res,sort_keys=True))
PY
}
log "=== G0 reference: sampler-fallback probes on the live daily ==="
[ "$(served_id)" = "$NEWM" ] || { log "ABORT: live daily is '$(served_id)', not $NEWM"; finish ABORTED; exit 3; }
fallback live >/dev/null
[ -s "$R/fallback-live.json" ] || { log "G0 FAIL: no reference"; finish ABORTED; exit 3; }
log "fallback reference: $(cat "$R/fallback-live.json")"
log "=== G1: boot the candidate launcher with no env (= the full daily config) ==="
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
"${CLEAN_ENV[@]}" bash "$CAND" > "$R/boot.log" 2>&1 || { log "G1 FAIL: launcher exit $(tail -2 "$R/boot.log" | tr '\n' ' ' | cut -c1-200)"; finish ABORTED; exit 3; }
for i in $(seq 120); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done
got=$(cfgline)
case "$got" in *"cache_size: 786432 cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;;
  *) log "G1 FAIL: served '$(served_id)' config '$got'"; finish ABORTED; exit 3;; esac
log "G1 PASS: $(served_id) $got; image $(sudo docker ps --format '{{.Image}}' -f name=flashnext); VRAM free MiB $(vram)"
img=$(sudo docker ps --format '{{.Image}}' -f name=flashnext); envs=$(sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' flashnext)
[ "$img" = tabbyapi:stack-r4-e3r2 ] || { log "G1 FAIL: image $img"; finish ABORTED; exit 3; }
need="EXL3_MOE_PREFILL_E3=1 EXL3_SHARED_EXPERT_OVERLAP=1"; case $CAND in *r517.sh) need="$need EXL3_DRAFT_PINNED_STAGING=1 EXL3_BATCH_VERIFY=1 EXL3_MTP_HEAD_N=65536";; esac
log "candidate $(basename $CAND); required env: $need"
for f in $need; do
  echo "$envs" | grep -qx "$f" || { log "G1 FAIL: container env lacks $f"; finish ABORTED; exit 3; }; done

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
a=$(greedy); b=$(greedy30k); log "fingerprints c1 $a / 30k $b (canonical ae890c45d1000582 / 2aa8d1024daece5c)"
[ "$a" = ae890c45d1000582 ] && [ "$b" = 4a255910dee2d9c5 ] || { log "G1 FAIL: fingerprints (want c1 ae890c45d1000582 / 30k 4a255910dee2d9c5)"; finish ABORTED; exit 3; }
log "=== G1b: salted cold prefill 60k / 120k ==="
ok=1; for c in 60000 120000; do
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "prefill-$c" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c --unique \
    --salt $(( (RANDOM << 15 | RANDOM) + c )) --out "$R/prefill.jsonl" > "$R/prefill-$c.log" 2>&1
  tps=$(python3 -c 'import json,sys; r=[x for x in map(json.loads, open(sys.argv[1])) if x.get("ttft_s") and x.get("prompt_tokens") and x.get("ctx_requested") == int(sys.argv[2])][-1]; print(int(r["prompt_tokens"]/r["ttft_s"]))' "$R/prefill.jsonl" $c 2>/dev/null || echo 0)
  floor=$([ $c = 60000 ] && echo 9200 || echo 9500); log "cold prefill ctx $c: $tps tok/s (floor $floor)"; [ "$tps" -ge $floor ] || ok=0; done
[ $ok = 1 ] || { log "G1b FAIL"; finish ABORTED; exit 3; }
log "=== G0: sampler-fallback probes on the candidate vs the live reference ==="
fallback cand >/dev/null
if cmp -s "$R/fallback-live.json" "$R/fallback-cand.json"; then log "G0 PASS: $(cat "$R/fallback-cand.json")"
else log "G0 FAIL: live $(cat "$R/fallback-live.json") vs cand $(cat "$R/fallback-cand.json")"; finish ABORTED; exit 3; fi

log "=== G2: agentic-edit greedy + sampled, c1 + c4 ==="
python3 /srv/qwen5090/probes/agentic-edit.py --url "$API" --model "$NEWM" --tag STK --conc 1 4 --modes greedy sampled --out "$R/agentic-edit.jsonl" 2>&1 | tee "$R/agentic-edit.log" | grep -a "agentic-edit" | tee -a "$R/audit.log"
n_ok=$(grep -ac "6/6 ok" "$R/agentic-edit.log")
if [ "$n_ok" = 4 ] && alive; then log "G2 PASS (4/4 modes 6/6 ok, server alive)"; else log "G2 FAIL ($n_ok/4 modes fully ok; alive: $(alive && echo yes || echo no))"; finish ABORTED; exit 3; fi

log "=== G3: needles 131k / 240k ==="
python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$NEWM" --tag needle --ctx-tokens 131072 240000 \
  --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee "$R/needle.log" | tee -a "$R/audit.log"
[ "$(grep -c '5/5 retrieved' "$R/needle.log")" = 2 ] && log "G3 PASS" || { log "G3 FAIL"; finish ABORTED; exit 3; }

log "=== G4: tool-eval 69 x 4, parallel 8 ==="
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$NEWM" --temperature 0.6 --top-p 0.95 --top-k 20 \
    --trials 4 --parallel 8 --json-file "$R/tooleval.json" > "$R/tooleval.log" 2>&1 )
python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval.json" STK 2>&1 | tee "$R/tooleval-summary.txt" | tee -a "$R/audit.log"
mean=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trial_statistics"]["final_score_mean"])' "$R/tooleval.json" 2>/dev/null)
log "tool-eval mean: ${mean:-unparsed}"
if [ -n "$mean" ] && python3 -c "import sys; sys.exit(0 if float('$mean') >= 82.0 else 1)"; then log "G4 PASS"; else log "G4 FAIL (mean '${mean:-unparsed}')"; finish ABORTED; exit 3; fi
alive || { log "server not alive after the gates"; finish ABORTED; exit 3; }

log "=== PROMOTE: launch-flashnext.sh := $(basename $CAND) ==="
sudo cp -p "$LIVE" /srv/qwen5090/launch-flashnext.sh.pre-r517
cp "$CAND" /srv/qwen5090/launch-flashnext.sh.new && chmod 755 /srv/qwen5090/launch-flashnext.sh.new && mv /srv/qwen5090/launch-flashnext.sh.new "$LIVE"
cmp -s "$CAND" "$LIVE" || { log "PROMOTE FAIL: live launcher differs from the candidate"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r517 "$LIVE"; finish ABORTED; exit 3; }
PROMOTED=1
sudo docker rm -f flashnext >/dev/null 2>&1
"${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || { log "LIVE LAUNCH FAILED — rolling back"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r517 "$LIVE"; PROMOTED=0; finish ABORTED; exit 3; }
for i in $(seq 120); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done
a2=$(greedy); log "daily now: $(served_id) $(cfgline); c1 fingerprint $a2 (promotion-boot canonical $a)"
[ "$(served_id)" = "$NEWM" ] || { log "daily did not come up as $NEWM — rolling back"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r517 "$LIVE"; PROMOTED=0; finish ABORTED; exit 3; }
finish PROMOTED
