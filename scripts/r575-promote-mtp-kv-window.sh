#!/usr/bin/env bash
# R575 — promote the MTP draft-cache window (patches/exllamav3/mtp-kv-window/r2 = r1 rebased onto the R565 image; glm round
# finished by an Opus subagent): EXL3_MTP_KV_WINDOW=16384 gives the draft layer a per-slot ring (sink page + 64 pages) instead
# of a full-length draft cache. R569: at the served [30, 30] the loader then moves a layer onto cuda:0 (boot free 1,241 / 2,751
# vs 2,085 / 867) and the pool ladder climbs to 1,015,808 (+49,152, +5.1 %); 1,032,192 fails the cuda:0 floor by 14 MiB.
# R573 at the served pool: mp_decode code c1 +1.89 %, code c4 +2.13 %, prose c1 +1.48 %, prose c4 +1.45 %; deep c1 at 100k
# +4.7 %; c8 at 16k context +2.3 %; needles 5/5 at 131k and 240k. Split [29.5, 30.5] / [29, 31] changed nothing.
# The c1 / 30k fingerprints CHANGE (18238d63 = the moved-layer layout, R567): greedy text depends on the verify layout, so the
# rule below fixes the new pair from the B boots and requires every B boot to agree.
#   candidate = live + IMG (2 lines) + EXL3_MTP_KV_WINDOW=16384 in EXTRA_ENV + CACHE default 1,015,808 (4 changed lines)
#   AB 4 boots A B B A (A = live @ 966,656, B = candidate @ 1,015,808), tier off, 8 slots: boot free per card, fingerprints,
#      salted cold prefill 60k / 120k, mp_decode 24 code + 24 prose at c1 + c4, 512 tokens
#   Rule (fixed now): every A boot canonical (f4add302 / 4a255910) and every B boot equal to the first B boot; B boot free
#      >= 835 MiB on both cards (the tightest A card - 32, absolute, placement-aware per the round's spec); mean B prefill
#      >= 0.95x mean A at 60k and 120k; every mp_decode shape's paired geo-mean B/A >= -1.5 %. Otherwise no promotion.
#   G1 tier ON (config incl. cache_size 1,015,808, image, env, fingerprints = the B pair) -> PROMOTE early (.pre-r575) ->
#      GT crash-restart restore (the tier namespace changes with the new EXL3_* variable, so the tier is rebuilt), G2
#      agentic-edit, G3 needles 131k/240k, G4 tool-eval >= 82, G5 GSM8K >= 0.970; any failure rolls back.
# GPU TIMEBOX ~1 h 15. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r575-promote-mtp-kv-window; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/$NEWM
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r575.sh
SRC=/srv/qwen5090/overlay-src/mtp-kv-window-r2
CPOOL=1015808; WIN=16384; FLOOR=835
MP=/srv/qwen5090/probes/mp_decode.py
TT=/srv/qwen5090/overlay-src/nvme-tier-r4/tests/gpu_nvme_ab.py
CFG=/srv/qwen5090/flashnext-config.yml
NIMG=tabbyapi:mtpwin-r2
LIMGX=tabbyapi:ngram-prefetch-r1-gdnbf16
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SALT=$(( $(date +%s) % 100000 ))
BOOTED=0; PROMOTED=0
log(){ echo "$(date -Is) [r575] $*" | tee -a "$R/audit.log"; }
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
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R575 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CKPT/config.json" "$LM" "$TT" "$MP" /srv/qwen5090/probes/agentic-edit.py /srv/qwen5090/probes/fn_needle_oai.py \
         /srv/qwen5090/probes/tooleval_summary.py /srv/qwen5090/probes/nostop_proxy.py /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
[ -e "$SRC/Dockerfile.box" ] || { log "ABORT: missing $SRC"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")
[ "$LIMG" = "$LIMGX" ] || { log "ABORT: live image '$LIMG' is not R565's"; exit 3; }
grep -q '^MAXBS=${MAXBS:-8}$' "$LIVE" || { log "ABORT: live launcher is not 8 slots"; exit 3; }
LPOOL=$(sed -n 's/^CACHE=\${CACHE:-\([0-9]*\)}.*/\1/p' "$LIVE")
log "building $NIMG on $LIMG (python overlay, rebased manifest)"
(cd "$SRC" && sudo docker build --build-arg BASE="$LIMG" -f Dockerfile.box -t "$NIMG" . ) > "$R/build.log" 2>&1 \
  || { log "ABORT: build: $(grep -aiE 'error|mismatch|abort|refus|assert' "$R/build.log" | head -3 | tr '\n' ' ' | cut -c1-300)"; exit 3; }
log "build OK: $(grep -a 'imports through' "$R/build.log" | tail -1 | cut -c1-160)"
python3 - "$LIVE" "$CAND.new" "$LIMG" "$NIMG" "$LPOOL" "$CPOOL" "$WIN" <<'PY' || { log "ABORT: candidate edit"; exit 3; }
import sys,re; src,dst,limg,nimg,lpool,cpool,win=sys.argv[1:8]; s=open(src).read()
pat=re.escape(limg)+r"(?![-\w])"
assert len(re.findall(pat, s))==2, "image name count"
s=re.sub(pat, nimg, s)
m=re.search(r"^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$", s, re.M); assert m and "EXL3_MTP_KV_WINDOW" not in m.group(1)
s=s[:m.start(1)]+m.group(1)+f" EXL3_MTP_KV_WINDOW={win}"+s[m.end(1):]
a=f"CACHE=${{CACHE:-{lpool}}}"; assert s.count(a)==1, "cache line"
s=s.replace(a, f"CACHE=${{CACHE:-{cpool}}}")
open(dst,"w").write(s)
PY
chmod 755 "$CAND.new" && mv "$CAND.new" "$CAND"
diff "$LIVE" "$CAND" | tee -a "$R/audit.log"
[ "$(diff "$LIVE" "$CAND" | grep -c '^>')" = 4 ] || { log "ABORT: candidate should change exactly 4 lines"; exit 3; }
export GPU_QUEUE_NAME=r575-promote-mtp-kv-window
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
  ev=$(sudo docker exec flashnext env | grep -c "^EXL3_MTP_KV_WINDOW=$WIN")
  a=$(greedy "$tag"); b=$(greedy30k "$tag")
  [ "$arm" = B ] && [ -z "$REF1" ] && { REF1=$a; REF30=$b; }
  read P60 P120 < <(prefill "$tag")
  log "UP $tag: $(sudo docker inspect -f '{{.Config.Image}}' flashnext), window env $ev, pool $(sudo grep -E '^  cache_size:' $CFG | awk '{print $2}'); boot free $f0/$f1; c1 $a / 30k $b; cold prefill 60k $P60 / 120k $P120"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$f0" "$f1" "$a" "$b" "$P60" "$P120" >> "$R/ab.tsv"
  [ "$a" != none ] && [ "$b" != none ] || { log "FAIL $tag: greedy failed"; rc=1; break; }
  python3 "$MP" run --url "$API" --model "$NEWM" --tag "$arm$n" --tokens 512 --out "$R/mp.jsonl" 2>&1 | tail -3 | tee -a "$R/audit.log"
  alive || { log "FAIL $tag: server not alive"; rc=1; break; }
done
[ $rc = 0 ] || { finish FAILED; exit 1; }
python3 "$MP" compare --a A --b B "$R/mp.jsonl" 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
python3 - "$R/ab.tsv" "$R/analysis.txt" "$FLOOR" <<'PY' | tee -a "$R/audit.log"
import re,sys
rows=[l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
floor=int(sys.argv[3])
A=[r for r in rows if r[0]=="A"]; B=[r for r in rows if r[0]=="B"]
why=[]
if len(A)!=2 or len(B)!=2: why.append("boots missing")
if {(r[3],r[4]) for r in A}!={("f4add302e176d78e","4a255910dee2d9c5")}: why.append("A fingerprints not canonical")
if len({(r[3],r[4]) for r in B})!=1: why.append("B fingerprints differ between boots")
else: print("B fingerprints:", B[0][3], "/", B[0][4])
for i,card in ((1,"cuda:0"),(2,"cuda:1")):
    mb=min(int(r[i]) for r in B); print(f"min B free {card}: {mb} (floor {floor})")
    if mb < floor: why.append("headroom "+card)
for i,c in ((5,"60k"),(6,"120k")):
    ma=sum(float(r[i]) for r in A)/len(A); mb=sum(float(r[i]) for r in B)/len(B)
    print(f"prefill {c}: A {ma:.0f} B {mb:.0f} B/A {mb/ma:.3f}")
    if mb < 0.95*ma: why.append("prefill "+c)
t=open(sys.argv[2]).read()
g=[float(a) for a in re.findall(r"paired geo-mean B/A ([+-][0-9.]+) %", t)]
print("shapes", g)
if len(g)!=4: why.append(f"{len(g)} shapes parsed")
elif min(g) < -1.5: why.append(f"a shape at {min(g):+.2f} % < -1.5 %")
print("AB OK" if not why else "AB STOP: "+"; ".join(why))
PY
grep -q "AB OK" "$R/audit.log" || { log "AB rule not met; not promoting"; finish DONE; exit 0; }
log "=== G1: candidate with the NVMe tier ON ==="
up "$CAND" G1 || { log "G1 FAIL: no boot"; finish ABORTED; exit 3; }
got=$(cfgline)
case "$got" in *"cache_size: $CPOOL cache_mode: 8,8 max_batch_size: 8 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;;
  *) log "G1 FAIL: config '$got'"; finish ABORTED; exit 3;; esac
img=$(sudo docker ps --format '{{.Image}}' -f name=flashnext); envs=$(sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' flashnext)
[ "$img" = "$NIMG" ] && echo "$envs" | grep -qx "EXL3_MTP_KV_WINDOW=$WIN" || { log "G1 FAIL: image $img / window env"; finish ABORTED; exit 3; }
tl=$(sudo docker logs flashnext 2>&1 | grep -a -- ' -- nvme tier: ON' | tail -1 | cut -c1-200); [ -n "$tl" ] || { log "G1 FAIL: NVMe tier not ON"; finish ABORTED; exit 3; }
v0=$(vram); a=$(greedy G1); b=$(greedy30k G1); log "G1: $got; tier: $tl; VRAM free at boot $v0; fingerprints c1 $a / 30k $b"
[ "$a" = "$REF1" ] && [ "$b" = "$REF30" ] || { log "G1 FAIL: fingerprints $a / $b != REF $REF1 / $REF30"; finish ABORTED; exit 3; }
log "=== PROMOTE (early): launch-flashnext.sh := launch-flashnext-r575.sh ==="
sudo cp -p "$LIVE" /srv/qwen5090/launch-flashnext.sh.pre-r575
cp "$CAND" "$LIVE.new" && chmod 755 "$LIVE.new" && mv "$LIVE.new" "$LIVE"
cmp -s "$CAND" "$LIVE" || { log "PROMOTE FAIL"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r575 "$LIVE"; finish ABORTED; exit 3; }
PROMOTED=1; log "promoted: live launcher = $NIMG + EXL3_MTP_KV_WINDOW=$WIN @ $CPOOL (the G1 container is that launcher's boot); post-promotion gates follow"
rollback(){ log "POST-PROMOTION GATE FAILED ($1): rolling back to .pre-r575"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r575 "$LIVE"; PROMOTED=0; finish "ROLLED-BACK ($1)"; exit 3; }
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
