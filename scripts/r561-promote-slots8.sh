#!/usr/bin/env bash
# R561 — promote 8 slots on the R548 daily (user 2026-09-19: "8 slots with 900k KV would be nice"). R558 (same image, tier off):
# the highest 8-slot pool passing the headroom rule is 966,656 (-65,536, -6.3 %); c1 fingerprint = the 4-slot REF (slot count
# does not change single-stream numerics); boot free 2,085 / 867 MiB (REF 2,013 / 737); min free under 120k + c8 1,299 / 353
# (REF 1,243 / 195); fn_bench 2,048 forced: c1 code 200.9 / prose 199.6, c4 566 / 523, c6 544 / 523, c8 674 / 639 (4 slots at c8:
# 549 / 549, the other 4 queue); 8-agent replay wall 408.6 s (R557 at 4 slots 413.2 / 395.1), p50 3.76 s (4.23 / 4.11), queue wait
# p50 0.12 s (1.39 / 1.52).
#   S0 reference: live launcher, tier off: fingerprints c1 / 30k, boot free, cold prefill 60k / 120k
#   S1 candidate = live + MAXBS 8 + CACHE 966,656 (exactly 2 changed lines), tier off: fingerprints = REF (both), boot free >= REF - 32
#      on both cards, cold prefill >= 0.95x REF (user rule), then 120k + c8 survival
#   G1 tier ON (config, fingerprints = REF), GT crash-restart restore, G2 agentic-edit, G3 needles 131k/240k, G4 tool-eval >= 82,
#   G5 GSM8K >= 0.970
# S1 + G1 PASS -> launch-flashnext.sh := candidate (old live kept as launch-flashnext.sh.pre-r561), user 2026-09-19 "Lets promote/keep 8 slots";
# GT, G2-G5 run on the promoted daily and roll back on failure. The draft policy stays [[4, 3], [8, 1]]
# (R560 sweeps c5-8 policies separately; a policy change is its own promotion). GPU TIMEBOX ~2.5 h. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r561-promote-slots8; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/$NEWM
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r561.sh
MP=/srv/qwen5090/probes/mp_decode.py
TT=/srv/qwen5090/overlay-src/nvme-tier-r4/tests/gpu_nvme_ab.py
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SALT=$(( $(date +%s) % 100000 ))
BOOTED=0; PROMOTED=0
log(){ echo "$(date -Is) [r561] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
free0(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0; }
free1(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
finish(){
  if [ "$BOOTED" = 1 ] && [ "$PROMOTED" = 0 ]; then
    sudo docker logs flashnext > "$R/docker-final.log" 2>&1   # the container log is lost once it is removed (R548 G5)
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the unchanged daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done; log "daily: $(served_id) $(cfgline)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R561 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CKPT/config.json" "$LM" "$TT" "$MP" /srv/qwen5090/probes/agentic-edit.py /srv/qwen5090/probes/fn_needle_oai.py \
         /srv/qwen5090/probes/tooleval_summary.py /srv/qwen5090/probes/nostop_proxy.py /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
grep -q '^MAXBS=${MAXBS:-4}$' "$LIVE" || { log "ABORT: live launcher is not 4 slots"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")
[ "$LIMG" = tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16 ] || { log "ABORT: live image '$LIMG' is not R548"; exit 3; }
LPOOL=$(sed -n 's/^CACHE=\${CACHE:-\([0-9]*\)}.*/\1/p' "$LIVE"); [ "$LPOOL" = 1032192 ] || { log "ABORT: live pool '$LPOOL' is not R548's 1,032,192"; exit 3; }
POOL8=966656   # R558: highest 8-slot pool with c1 = REF, boot free >= REF - 32 and min free under 120k + c8 >= REF_MIN - 32 on both cards
export GPU_QUEUE_NAME=r561-promote-slots8
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
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
# up LAUNCHER TAG [ENV...] -> 0 up, 1 no boot
up(){ local L=$1 tag=$2 i st lp; shift 2
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" "$@" bash "$L" > "$R/boot-$tag.log" 2>&1 ) & lp=$!
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
prefill(){ local tag=$1 c tps out=""
  for c in 60000 120000; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$tag-prefill-$c" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c --unique \
      --salt $(( SALT + c/1000 + ${#tag}*7 + RANDOM )) --out "$R/prefill.jsonl" > /dev/null 2>&1
    tps=$(python3 -c 'import json,sys; r=[x for x in map(json.loads, open(sys.argv[1])) if x.get("ttft_s") and x.get("prompt_tokens") and x.get("tag") == sys.argv[2]][-1]; print(int(r["prompt_tokens"]/r["ttft_s"]))' "$R/prefill.jsonl" "$tag-prefill-$c" 2>/dev/null || echo 0)
    out="$out $tps"; done; echo $out; }
log "=== S0 reference: live launcher, tier off ==="
up "$LIVE" REF NVME_TIER= || { finish FAILED; exit 3; }
REF_F0=$(free0); REF_F1=$(free1)
REF1=$(greedy REF); REF30=$(greedy30k REF)
read RP60 RP120 < <(prefill REF)
log "REF @ $LPOOL x 4: boot free $REF_F0/$REF_F1; fingerprints c1 $REF1 / 30k $REF30; cold prefill 60k $RP60 / 120k $RP120 tok/s"
[ "$REF1" != none ] && [ "$REF30" != none ] && [ "$RP60" -gt 0 ] && [ "$RP120" -gt 0 ] || { log "S0 FAIL"; finish FAILED; exit 3; }
python3 - "$LIVE" "$CAND.new" "$LPOOL" "$POOL8" <<'PY' || { log "ABORT: candidate edit"; finish FAILED; exit 3; }
import sys,re; src,dst,pa,pb=sys.argv[1:5]; s=open(src).read()
assert s.count("\nMAXBS=${MAXBS:-4}\n")==1; s=s.replace("\nMAXBS=${MAXBS:-4}\n","\nMAXBS=${MAXBS:-8}\n")
m=re.search(r"^CACHE=\$\{CACHE:-%s\}( *# *)(.*)$" % pa, s, re.M); assert m
s=s[:m.start()]+"CACHE=${CACHE:-%s}%sR561: 8 slots (R558 ladder top at 8 slots); was %s at 4 slots; %s" % (pb, m.group(1), pa, m.group(2))+s[m.end():]
open(dst,"w").write(s)
PY
chmod 755 "$CAND.new" && mv "$CAND.new" "$CAND"
diff "$LIVE" "$CAND" | tee -a "$R/audit.log"
[ "$(diff "$LIVE" "$CAND" | grep -c '^>')" = 2 ] || { log "ABORT: candidate should change exactly 2 lines"; finish FAILED; exit 3; }
log "=== S1 candidate, tier off ==="
up "$CAND" S1 NVME_TIER= || { log "S1 FAIL: no boot"; finish FAILED; exit 3; }
got=$(cfgline)
case "$got" in *"cache_size: $POOL8 cache_mode: 8,8 max_batch_size: 8 "*) ;; *) log "S1 FAIL: config '$got'"; finish FAILED; exit 3;; esac
f0=$(free0); f1=$(free1); a=$(greedy S1); b=$(greedy30k S1)
read P60 P120 < <(prefill S1)
log "S1 @ $POOL8 x 8: boot free $f0/$f1 (floors $((REF_F0-32))/$((REF_F1-32))); fingerprints c1 $a / 30k $b; cold prefill 60k $P60 / 120k $P120 tok/s"
[ "$f0" -ge $(( REF_F0 - 32 )) ] && [ "$f1" -ge $(( REF_F1 - 32 )) ] || { log "S1 FAIL: headroom"; finish FAILED; exit 3; }
[ "$a" = "$REF1" ] && [ "$b" = "$REF30" ] || { log "S1 FAIL: fingerprints $a / $b != REF $REF1 / $REF30"; finish FAILED; exit 3; }
python3 -c 'import sys; a,b,c,d=map(float,sys.argv[1:5]); r=(c/a, d/b); print(f"prefill ratio 60k {r[0]:.3f} 120k {r[1]:.3f}"); sys.exit(0 if min(r) >= 0.95 else 1)' $RP60 $RP120 $P60 $P120 \
  | tee -a "$R/audit.log" || { log "S1 FAIL: prefill below 0.95x REF"; finish FAILED; exit 3; }
python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag S1-c8 --kind code --tokens 2048 --conc 8 --runs 1 --out "$R/records.jsonl" 2>&1 \
  | grep -E "^  c=|FAILED|Traceback" | cut -c1-200 | tee -a "$R/audit.log"
alive && log "S1 PASS" || { log "S1 FAIL: server died under c8"; finish FAILED; exit 3; }
log "=== G1: candidate with the NVMe tier ON ==="
up "$CAND" G1 || { log "G1 FAIL: no boot"; finish ABORTED; exit 3; }
got=$(cfgline)
case "$got" in *"cache_size: $POOL8 cache_mode: 8,8 max_batch_size: 8 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;;
  *) log "G1 FAIL: config '$got'"; finish ABORTED; exit 3;; esac
tl=$(sudo docker logs flashnext 2>&1 | grep -a -- ' -- nvme tier: ON' | tail -1 | cut -c1-200); [ -n "$tl" ] || { log "G1 FAIL: NVMe tier not ON"; finish ABORTED; exit 3; }
v0=$(vram); a=$(greedy G1); b=$(greedy30k G1); log "G1: $got; tier: $tl; VRAM free at boot $v0; fingerprints c1 $a / 30k $b"
[ "$a" = "$REF1" ] && [ "$b" = "$REF30" ] || { log "G1 FAIL: fingerprints $a / $b != REF $REF1 / $REF30"; finish ABORTED; exit 3; }
# user 2026-09-19 17:1x: "Lets promote/keep 8 slots" -> promote once S1 + G1 hold (config, fingerprints = REF, headroom, prefill,
# c8 survival, tier ON); GT and G2-G5 then run on the live 8-slot daily and roll back to .pre-r561 on any failure
log "=== PROMOTE (early): launch-flashnext.sh := launch-flashnext-r561.sh ==="
sudo cp -p "$LIVE" /srv/qwen5090/launch-flashnext.sh.pre-r561
cp "$CAND" "$LIVE.new" && chmod 755 "$LIVE.new" && mv "$LIVE.new" "$LIVE"
cmp -s "$CAND" "$LIVE" || { log "PROMOTE FAIL"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r561 "$LIVE"; finish ABORTED; exit 3; }
PROMOTED=1; log "promoted: live launcher = 8 slots @ $POOL8 (the G1 container is that launcher's boot); post-promotion gates follow"
rollback(){ log "POST-PROMOTION GATE FAILED ($1): rolling back to .pre-r561"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r561 "$LIVE"; PROMOTED=0; finish "ROLLED-BACK ($1)"; exit 3; }
log "=== GT: crash-restart restore ==="
timeout 600 python3 "$TT" fill --out "$R" --sizes 30000 2>&1 | tail -3 | cut -c1-220 | tee -a "$R/audit.log"
timeout 300 python3 "$TT" wait-drained --out "$R" --container flashnext 2>&1 | tail -2 | cut -c1-240 | tee -a "$R/audit.log"
sudo docker logs flashnext > "$R/docker-pre-restart.log" 2>&1
up "$CAND" GT-restart || rollback "GT FAIL: relaunch"
sudo docker logs flashnext 2>&1 | grep -a "open scan" | tail -1 | cut -c1-200 | tee -a "$R/audit.log"
timeout 600 python3 "$TT" verify --out "$R" 2>&1 | tail -4 | cut -c1-240 | tee "$R/verify.txt" | tee -a "$R/audit.log"
grep -aq "VERIFY PASS" "$R/verify.txt" && log "GT PASS" || rollback "GT FAIL"
log "=== G2: agentic-edit ==="
python3 /srv/qwen5090/probes/agentic-edit.py --url "$API" --model "$NEWM" --tag BR --conc 1 4 --modes greedy sampled --out "$R/agentic-edit.jsonl" 2>&1 | tee "$R/agentic-edit.log" | grep -a "agentic-edit" | tee -a "$R/audit.log"
[ "$(grep -ac "6/6 ok" "$R/agentic-edit.log")" = 4 ] && alive && log "G2 PASS" || rollback "G2 FAIL"
log "=== G3: needles ==="
python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$NEWM" --tag needle --ctx-tokens 131072 240000 \
  --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee "$R/needle.log" | tee -a "$R/audit.log"
[ "$(grep -c '5/5 retrieved' "$R/needle.log")" = 2 ] && log "G3 PASS" || rollback "G3 FAIL"
log "=== G4: tool-eval 69 x 4 ==="
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$NEWM" --temperature 0.6 --top-p 0.95 --top-k 20 \
    --trials 4 --parallel 8 --json-file "$R/tooleval.json" > "$R/tooleval.log" 2>&1 )
python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval.json" BR 2>&1 | tee "$R/tooleval-summary.txt" | tee -a "$R/audit.log"
mean=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trial_statistics"]["final_score_mean"])' "$R/tooleval.json" 2>/dev/null)
[ -n "$mean" ] && python3 -c "import sys; sys.exit(0 if float('$mean') >= 82.0 else 1)" && log "G4 PASS ($mean)" || rollback "G4 FAIL (${mean:-unparsed})"
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
log "GSM8K n=500 flexible: ${g:-none}"
[ -n "$g" ] && [ "$g" != none ] && python3 -c "import sys; sys.exit(0 if float('$g') >= 0.970 else 1)" && log "G5 PASS" || rollback "G5 FAIL"
alive || rollback "server not alive after the gates"
alive && log "daily now: $(served_id) $(cfgline); canonical c1 $REF1 / 30k $REF30; VRAM free $(vram)"
finish PROMOTED
