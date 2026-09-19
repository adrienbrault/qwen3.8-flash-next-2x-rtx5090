#!/usr/bin/env bash
# R530 — promote ple-ckpt-clone r1 (patches/exllamav3/ple-ckpt-clone/r1): a one-line correctness fix in the base exllamav3
# (upstream master too, 2026-09-19): PLELayerState.stash() returned id_state[slot, :ctx].cpu(), but id_state is a host tensor,
# so .cpu() is a no-op and every stored PLE checkpoint (prompt-end and decode-interval) aliased the live slot; later writes and
# the next job's eos fill changed it, and a job resuming from it got a wrong n-gram token context for its first ctx tokens.
# Found by the recurrent-tip round (R524 try 4 phase D: float state bit-exact, PLE id_state (-2,0)[1] differed). Fix: .clone().
# Cold requests are unaffected, so G1 requires the live fingerprints; resumed requests change (that is the fix), so the quality
# gates decide. Built on the LIVE image (stacks on the pruned-draft / tool_choice images). No VRAM change, so no G1v.
# PASS -> launch-flashnext.sh := candidate (old live kept as launch-flashnext.sh.pre-r530).
# RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r530-promote-plefix; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/$NEWM
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r530.sh
SRC=/srv/qwen5090/overlay-src/ple-ckpt-clone-r1
SALT=$(( $(date +%s) % 100000 ))
CFG=/srv/qwen5090/flashnext-config.yml
POOL=819200
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0; PROMOTED=0
log(){ echo "$(date -Is) [r530] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
finish(){
  if [ "$BOOTED" = 1 ] && [ "$PROMOTED" = 0 ]; then
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the unchanged daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done; log "daily: $(served_id) $(cfgline)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R530 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CKPT/config.json" "$LM" /srv/qwen5090/probes/agentic-edit.py /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/tooleval_summary.py \
         /srv/qwen5090/probes/nostop_proxy.py /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
grep -q '^MAXBS=${MAXBS:-4}$' "$LIVE" && grep -q '^CACHE=${CACHE:-819200}' "$LIVE" || { log "ABORT: live launcher is not 4 slots @ 819,200"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE"); [ -n "$LIMG" ] || { log "ABORT: no live IMG"; exit 3; }
case "$LIMG" in *-plefix) log "ABORT: live image $LIMG already has the fix"; exit 3;; esac
NIMG="$LIMG-plefix"
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE")
# live references: fingerprints are the same for both possible live images (R522 byte-identical); decode floors follow R522b
C1=e7fb377c987d685c; C30=4a255910dee2d9c5
case " $LENV " in *" EXL3_EMBED_GPU_PRUNED=1 "*) REFS="221.8 194.5 508.4 484.4";; *) REFS="217.4 194.5 500.9 484.4";; esac
log "building $NIMG from $LIMG (before the lock)"
(cd "$SRC" && sudo docker build -f Dockerfile.box --build-arg BASE="$LIMG" -t "$NIMG" . ) > "$R/build.log" 2>&1 || { log "ABORT: build failed: $(tail -3 "$R/build.log" | tr '\n' ' ' | cut -c1-240)"; exit 3; }
grep -a "ple-ckpt-clone" "$R/build.log" | tail -3 | tee -a "$R/audit.log"
grep -aq "ple-ckpt-clone test OK" "$R/build.log" || { log "ABORT: in-image test did not pass"; exit 3; }
sed -e "s/^IMG=\${IMG:-$LIMG}/IMG=\${IMG:-$NIMG}/" "$LIVE" > "$CAND.new" && chmod 755 "$CAND.new" && mv "$CAND.new" "$CAND"
diff "$LIVE" "$CAND" | tee -a "$R/audit.log"
[ "$(diff "$LIVE" "$CAND" | grep -c '^>')" = 1 ] || { log "ABORT: candidate did not get exactly one changed line"; exit 3; }
export GPU_QUEUE_NAME=r530-promote-plefix
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
log "=== G1: boot the candidate ==="
"${CLEAN_ENV[@]}" bash "$CAND" > "$R/boot.log" 2>&1 || { log "G1 FAIL: launcher exit $(tail -2 "$R/boot.log" | tr '\n' ' ' | cut -c1-200)"; finish ABORTED; exit 3; }
for i in $(seq 120); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done
got=$(cfgline)
case "$got" in *"cache_size: $POOL cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;;
  *) log "G1 FAIL: served '$(served_id)' config '$got'"; finish ABORTED; exit 3;; esac
img=$(sudo docker ps --format '{{.Image}}' -f name=flashnext); envs=$(sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' flashnext)
[ "$img" = "$NIMG" ] || { log "G1 FAIL: image $img"; finish ABORTED; exit 3; }
for f in EXL3_HC_MIX_V2_INT8=1 EXL3_MOE_PREFILL_E3=1 EXL3_SHARED_EXPERT_OVERLAP=1 EXL3_MTP_HEAD_N=65536; do
  echo "$envs" | grep -qx "$f" || { log "G1 FAIL: container env lacks $f"; finish ABORTED; exit 3; }; done
log "G1 PASS: $got; image $img; VRAM free MiB $(vram)"
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$NEWM" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
greedy30k(){ python3 - "$API" "$NEWM" "$R/greedy30k-$1.json" <<'PY' 2>/dev/null || echo none
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
a=$(greedy boot); b=$(greedy30k boot); log "candidate fingerprints c1 $a / 30k $b (live $C1 / $C30)"
[ "$a" = "$C1" ] && [ "$b" = "$C30" ] || { log "G1 FAIL: not byte-identical to the live daily"; finish ABORTED; exit 3; }
log "=== G1b: cold prefill + decode ==="
ok=1; for c in 60000 120000; do
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "prefill-$c" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c --unique --salt $((SALT+c/1000)) \
    --out "$R/prefill.jsonl" 2>&1 | grep -E "FAILED|Traceback" | tee -a "$R/audit.log"
  tps=$(python3 -c 'import json,sys; r=[x for x in map(json.loads, open(sys.argv[1])) if x.get("ttft_s") and x.get("prompt_tokens") and x.get("ctx_requested") == int(sys.argv[2])][-1]; print(int(r["prompt_tokens"]/r["ttft_s"]))' "$R/prefill.jsonl" $c 2>/dev/null || echo 0)
  floor=$([ $c = 60000 ] && echo 9200 || echo 9500); log "cold prefill ctx $c: $tps tok/s (floor $floor)"; [ "$tps" -ge $floor ] || ok=0; done
for kind in code prose; do
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "C-$kind" --kind $kind --tokens 2048 --warmup-runs 1 --conc 1 4 --runs 2 \
    --out "$R/records.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$kind]/" | cut -c1-220 | tee -a "$R/audit.log"; done
python3 - "$R/records.jsonl" $REFS <<'PY' 2>&1 | tee "$R/decode.txt" | tee -a "$R/audit.log"
import json,sys,statistics as st,collections
v=[float(x) for x in sys.argv[2:6]]; ref={("code",1):v[0],("prose",1):v[1],("code",4):v[2],("prose",4):v[3]}  # live refs (R522b arms, R525 prose)
rounds=collections.defaultdict(list)
for l in open(sys.argv[1]):
    r=json.loads(l); rounds[(r["tag"].split("-",1)[1],r["conc"],r["run"])].append(r)
agg=collections.defaultdict(list)
for (k,c,run),rs in rounds.items(): agg[(k,c)].append(sum(x["completion_tokens"] for x in rs)/max(x["round_wall_s"] for x in rs))
bad=0
for k in sorted(ref):
    m=st.mean(agg[k]); d=(m/ref[k]-1)*100; bad+= d < -3.0
    print(f"{k[0]} c{k[1]}: {m:.1f} vs {ref[k]} ({d:+.1f} %)")
print("DECODE", "OK" if not bad else "REGRESSED")
PY
grep -q "DECODE OK" "$R/decode.txt" || ok=0
[ $ok = 1 ] && log "G1b PASS" || { log "G1b FAIL"; finish ABORTED; exit 3; }
log "=== G2: agentic-edit ==="
python3 /srv/qwen5090/probes/agentic-edit.py --url "$API" --model "$NEWM" --tag PF --conc 1 4 --modes greedy sampled --out "$R/agentic-edit.jsonl" 2>&1 | tee "$R/agentic-edit.log" | grep -a "agentic-edit" | tee -a "$R/audit.log"
[ "$(grep -ac "6/6 ok" "$R/agentic-edit.log")" = 4 ] && alive && log "G2 PASS" || { log "G2 FAIL"; finish ABORTED; exit 3; }
log "=== G3: needles ==="
python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$NEWM" --tag needle --ctx-tokens 131072 240000 \
  --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee "$R/needle.log" | tee -a "$R/audit.log"
[ "$(grep -c '5/5 retrieved' "$R/needle.log")" = 2 ] && log "G3 PASS" || { log "G3 FAIL"; finish ABORTED; exit 3; }
log "=== G4: tool-eval 69 x 4 ==="
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$NEWM" --temperature 0.6 --top-p 0.95 --top-k 20 \
    --trials 4 --parallel 8 --json-file "$R/tooleval.json" > "$R/tooleval.log" 2>&1 )
python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval.json" PF 2>&1 | tee "$R/tooleval-summary.txt" | tee -a "$R/audit.log"
mean=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trial_statistics"]["final_score_mean"])' "$R/tooleval.json" 2>/dev/null)
[ -n "$mean" ] && python3 -c "import sys; sys.exit(0 if float('$mean') >= 82.0 else 1)" && log "G4 PASS ($mean)" || { log "G4 FAIL (${mean:-unparsed})"; finish ABORTED; exit 3; }
log "=== G5: GSM8K n=500 (nostop proxy) ==="
python3 /srv/qwen5090/probes/nostop_proxy.py --listen 127.0.0.1:8031 --upstream http://127.0.0.1:8022 >> "$R/proxy.log" 2>&1 & pxp=$!; sleep 2
timeout 10800 "$LM" --model local-chat-completions \
  --model_args "base_url=http://127.0.0.1:8031/v1/chat/completions,model=$NEWM,tokenizer=$CKPT,num_concurrent=4,max_retries=1,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 5 --limit 500 --apply_chat_template --gen_kwargs temperature=0,max_gen_toks=8192 --log_samples \
  --output_path "$R/ev-gsm8k" > "$R/ev-gsm8k.log" 2>&1
kill $pxp 2>/dev/null
g=$(python3 -c '
import json,pathlib,sys
f=list(pathlib.Path(sys.argv[1]).rglob("results_*.json")); print(json.loads(f[0].read_text())["results"]["gsm8k"]["exact_match,flexible-extract"] if len(f)==1 else "none")' "$R/ev-gsm8k" 2>/dev/null)
log "GSM8K n=500 flexible: ${g:-none} (R525 0.974)"
[ -n "$g" ] && [ "$g" != none ] && python3 -c "import sys; sys.exit(0 if float('$g') >= 0.970 else 1)" && log "G5 PASS" || { log "G5 FAIL"; finish ABORTED; exit 3; }
alive || { log "server not alive after the gates"; finish ABORTED; exit 3; }
log "=== PROMOTE: launch-flashnext.sh := launch-flashnext-r530.sh ==="
sudo cp -p "$LIVE" /srv/qwen5090/launch-flashnext.sh.pre-r530
cp "$CAND" "$LIVE.new" && chmod 755 "$LIVE.new" && mv "$LIVE.new" "$LIVE"
cmp -s "$CAND" "$LIVE" || { log "PROMOTE FAIL"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r530 "$LIVE"; finish ABORTED; exit 3; }
PROMOTED=1
sudo docker rm -f flashnext >/dev/null 2>&1
"${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || { log "LIVE LAUNCH FAILED — rolling back"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r530 "$LIVE"; PROMOTED=0; finish ABORTED; exit 3; }
for i in $(seq 120); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done
a2=$(greedy live); log "daily now: $(served_id) $(cfgline); c1 fingerprint $a2 (promotion-boot $a); VRAM free MiB $(vram)"
[ "$(served_id)" = "$NEWM" ] || { log "daily did not come up — rolling back"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r530 "$LIVE"; PROMOTED=0; finish ABORTED; exit 3; }
finish PROMOTED
