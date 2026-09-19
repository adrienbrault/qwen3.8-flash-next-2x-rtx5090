#!/usr/bin/env bash
# R565 — confirm and promote the n-gram (PLE) row prefetch (patches/exllamav3/ngram-prefetch/r1, EXL3_NGRAM_PREFETCH2=1, image
# tabbyapi:ngram-prefetch-r1-gdnbf16 built by R559). Timing-only: output byte-identical (R559: c1 / 30k fingerprints canonical on
# the flag-on boots). R559, one boot per arm: mp_decode B/A code c1 +1.13 % [+0.58, +1.69], code c4 +2.01 % [-0.02, +4.19],
# prose c1 +0.74 % [+0.38, +1.12], prose c4 +1.58 % [+0.57, +2.63]; cold prefill 60k 0.971x / 120k 1.012x; free VRAM equal.
# One boot per arm is inside the ~1.7 % boot spread, so this unit re-measures over 4 boots before promoting.
#   candidate = live + IMG (2 lines: IMG default + tier-default condition) + EXL3_NGRAM_PREFETCH2=1 in EXTRA_ENV (3 changed lines)
#   AB 4 boots A B B A (A = live, B = candidate), tier off, 8 slots at the live pool: boot free per card, c1 / 30k fingerprints,
#      salted cold prefill 60k / 120k, mp_decode 24 code + 24 prose, c1 + c4, 512 tokens
#   Rule (fixed now): B fingerprints = A on every boot; min B boot free >= min A boot free - 32 MiB per card; mean B prefill
#      >= 0.95x mean A at 60k and at 120k; every mp_decode shape's paired geo-mean B/A >= 0 % and at least two shapes with the
#      95 % CI lower bound > 0 %. Otherwise: no promotion (finish DONE).
#   G1 tier ON (config, image, env, fingerprints = REF) -> PROMOTE early (launch-flashnext.sh.pre-r565 kept) -> GT crash-restart
#      restore, G2 agentic-edit, G3 needles 131k/240k, G4 tool-eval >= 82, G5 GSM8K >= 0.970; any failure rolls back.
# EXL3_NGRAM_TIMING is implied by PREFETCH2 (a wall-clock instrument, printed every 30 s); R559's gain was measured with it on.
# GPU TIMEBOX ~1 h 15. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r565-promote-ngram-prefetch; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/$NEWM
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r565.sh
MP=/srv/qwen5090/probes/mp_decode.py
TT=/srv/qwen5090/overlay-src/nvme-tier-r4/tests/gpu_nvme_ab.py
CFG=/srv/qwen5090/flashnext-config.yml
NIMG=tabbyapi:ngram-prefetch-r1-gdnbf16
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SALT=$(( $(date +%s) % 100000 ))
BOOTED=0; PROMOTED=0
log(){ echo "$(date -Is) [r565] $*" | tee -a "$R/audit.log"; }
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
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R565 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CKPT/config.json" "$LM" "$TT" "$MP" /srv/qwen5090/probes/agentic-edit.py /srv/qwen5090/probes/fn_needle_oai.py \
         /srv/qwen5090/probes/tooleval_summary.py /srv/qwen5090/probes/nostop_proxy.py /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
sudo docker image inspect "$NIMG" >/dev/null 2>&1 || { log "ABORT: $NIMG missing (built by R559)"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")
[ "$LIMG" = tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16 ] || { log "ABORT: live image '$LIMG' is not R548/R561's"; exit 3; }
grep -q '^MAXBS=${MAXBS:-8}$' "$LIVE" || { log "ABORT: live launcher is not 8 slots (R561)"; exit 3; }
LPOOL=$(sed -n 's/^CACHE=\${CACHE:-\([0-9]*\)}.*/\1/p' "$LIVE")
python3 - "$LIVE" "$CAND.new" "$LIMG" "$NIMG" <<'PY' || { log "ABORT: candidate edit"; exit 3; }
import sys,re; src,dst,limg,nimg=sys.argv[1:5]; s=open(src).read()
pat=re.escape(limg)+r"(?![-\w])"
assert len(re.findall(pat, s))==2, "image name count"
s=re.sub(pat, nimg, s)
m=re.search(r"^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$", s, re.M); assert m and "EXL3_NGRAM_PREFETCH2" not in m.group(1)
s=s[:m.start(1)]+m.group(1)+" EXL3_NGRAM_PREFETCH2=1"+s[m.end(1):]
open(dst,"w").write(s)
PY
chmod 755 "$CAND.new" && mv "$CAND.new" "$CAND"
diff "$LIVE" "$CAND" | tee -a "$R/audit.log"
[ "$(diff "$LIVE" "$CAND" | grep -c '^>')" = 3 ] || { log "ABORT: candidate should change exactly 3 lines"; exit 3; }
export GPU_QUEUE_NAME=r565-promote-ngram-prefetch
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
log "=== AB: 4 boots A B B A, A = live, B = candidate, tier off, pool $LPOOL x 8 ==="
rc=0; n=0; REF1=; REF30=
: > "$R/ab.tsv"
for arm in A B B A; do
  n=$((n+1)); tag="$arm$n"; L="$LIVE"; [ "$arm" = B ] && L="$CAND"
  up "$L" "$tag" NVME_TIER= || { rc=1; break; }
  f0=$(free0); f1=$(free1)
  ev=$(sudo docker exec flashnext env | grep -c '^EXL3_NGRAM_PREFETCH2=1')
  a=$(greedy "$tag"); b=$(greedy30k "$tag")
  [ -z "$REF1" ] && { REF1=$a; REF30=$b; }
  read P60 P120 < <(prefill "$tag")
  log "UP $tag: $(sudo docker inspect -f '{{.Config.Image}}' flashnext), prefetch env $ev; boot free $f0/$f1; c1 $a / 30k $b; cold prefill 60k $P60 / 120k $P120"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$f0" "$f1" "$a" "$b" "$P60" "$P120" >> "$R/ab.tsv"
  [ "$a" != none ] && [ "$b" != none ] || { log "FAIL $tag: greedy failed"; rc=1; break; }
  python3 "$MP" run --url "$API" --model "$NEWM" --tag "$arm$n" --tokens 512 --out "$R/mp.jsonl" 2>&1 | tail -3 | tee -a "$R/audit.log"
  alive || { log "FAIL $tag: server not alive"; rc=1; break; }
done
[ $rc = 0 ] || { finish FAILED; exit 1; }
python3 "$MP" compare --a A --b B "$R/mp.jsonl" 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
python3 - "$R/ab.tsv" "$R/analysis.txt" <<'PY' | tee -a "$R/audit.log"
import re,sys
rows=[l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
A=[r for r in rows if r[0]=="A"]; B=[r for r in rows if r[0]=="B"]
why=[]
if {(r[3],r[4]) for r in rows}!={(A[0][3],A[0][4])}: why.append("fingerprints differ across boots")
for i,card in ((1,"cuda:0"),(2,"cuda:1")):
    if min(int(r[i]) for r in B) < min(int(r[i]) for r in A)-32: why.append("headroom "+card)
for i,c in ((5,"60k"),(6,"120k")):
    ma=sum(float(r[i]) for r in A)/len(A); mb=sum(float(r[i]) for r in B)/len(B)
    print(f"prefill {c}: A {ma:.0f} B {mb:.0f} B/A {mb/ma:.3f}")
    if mb < 0.95*ma: why.append("prefill "+c)
t=open(sys.argv[2]).read()
g=[(float(a),float(lo)) for a,lo in re.findall(r"paired geo-mean B/A ([+-][0-9.]+) %, 95 % CI \[([+-][0-9.]+),", t)]
print("shapes", g)
if len(g)!=4: why.append(f"{len(g)} shapes parsed")
elif min(a for a,_ in g) < 0: why.append("a shape below 0")
elif sum(lo>0 for _,lo in g) < 2: why.append("fewer than 2 shapes with CI above 0")
print("AB OK" if not why else "AB STOP: "+"; ".join(why))
PY
grep -q "AB OK" "$R/audit.log" || { log "AB rule not met; not promoting"; finish DONE; exit 0; }
log "=== G1: candidate with the NVMe tier ON ==="
up "$CAND" G1 || { log "G1 FAIL: no boot"; finish ABORTED; exit 3; }
got=$(cfgline)
case "$got" in *"cache_size: $LPOOL cache_mode: 8,8 max_batch_size: 8 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;;
  *) log "G1 FAIL: config '$got'"; finish ABORTED; exit 3;; esac
img=$(sudo docker ps --format '{{.Image}}' -f name=flashnext); envs=$(sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' flashnext)
[ "$img" = "$NIMG" ] && echo "$envs" | grep -qx EXL3_NGRAM_PREFETCH2=1 || { log "G1 FAIL: image $img / prefetch env"; finish ABORTED; exit 3; }
tl=$(sudo docker logs flashnext 2>&1 | grep -a -- ' -- nvme tier: ON' | tail -1 | cut -c1-200); [ -n "$tl" ] || { log "G1 FAIL: NVMe tier not ON"; finish ABORTED; exit 3; }
v0=$(vram); a=$(greedy G1); b=$(greedy30k G1); log "G1: $got; tier: $tl; VRAM free at boot $v0; fingerprints c1 $a / 30k $b"
[ "$a" = "$REF1" ] && [ "$b" = "$REF30" ] || { log "G1 FAIL: fingerprints $a / $b != REF $REF1 / $REF30"; finish ABORTED; exit 3; }
log "=== PROMOTE (early): launch-flashnext.sh := launch-flashnext-r565.sh ==="
sudo cp -p "$LIVE" /srv/qwen5090/launch-flashnext.sh.pre-r565
cp "$CAND" "$LIVE.new" && chmod 755 "$LIVE.new" && mv "$LIVE.new" "$LIVE"
cmp -s "$CAND" "$LIVE" || { log "PROMOTE FAIL"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r565 "$LIVE"; finish ABORTED; exit 3; }
PROMOTED=1; log "promoted: live launcher = $NIMG + EXL3_NGRAM_PREFETCH2=1 (the G1 container is that launcher's boot); post-promotion gates follow"
rollback(){ log "POST-PROMOTION GATE FAILED ($1): rolling back to .pre-r565"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r565 "$LIVE"; PROMOTED=0; finish "ROLLED-BACK ($1)"; exit 3; }
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
