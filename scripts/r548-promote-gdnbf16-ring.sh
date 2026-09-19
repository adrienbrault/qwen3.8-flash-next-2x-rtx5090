#!/usr/bin/env bash
# R548 — stack bf16 GDN recurrent state (patches/exllamav3/gdn-state-bf16/r1, EXL3_GDN_STATE_BF16=1) on the daily, expected to
# be the R546 raw_k-ring daily (…-r6-rawk @ 983,040; also runs on the R540 image). bf16 frees 864 MiB (R543: +49,152 tokens on
# R540) and is not bit-exact: its greedy text differs, so the decode criterion is the multi-prompt paired A/B (R547 on R540:
# code c1 +3.0 % [−0.3, +6.2], code c4 +2.3 % [+0.2, +4.3], prose c1 +1.4 % [−0.3, +3.0], prose c4 +3.3 % [+1.3, +5.1];
# the single-prompt −13 % of R545 was content, per-prompt range −11 … +22 %).
#   S0 reference boot (live, tier off): fingerprints and VRAM recorded
#   S1 build $LIMG-gdnbf16 (manifest-on-r6.json; the ring touches disjoint files)
#   S2 ladder bf16 on, tier off, from live pool + 16,384 (max 20) until no boot: normal placement (cuda:0 free >= 1,200 MiB at
#      boot), c1/30k fingerprints equal across normal-placement steps (BF1/BF30), cold 120k + c4 survival; POOL_ON = highest
#   S3 candidate = live + IMG (2 lines) + EXL3_GDN_STATE_BF16=1 + CACHE=POOL_ON (exactly 4 changed lines)
#   AB mp_decode 4 boots A B B A, tier off (24 code + 24 prose prompts, c1 + c4, 512 tokens): STOP if any shape's paired
#      geo-mean B/A is below -1.5 %
#   G1 tier ON (fingerprints = BF1/BF30, placement at boot), G1b prefill floors + 120k + c4, GT crash-restart restore,
#      G2 agentic-edit, G3 needles 131k/240k, G4 tool-eval >= 82, G5 GSM8K >= 0.970
# PASS -> launch-flashnext.sh := candidate (old live kept as launch-flashnext.sh.pre-r548). New canonical fingerprints = BF1/BF30.
# GPU TIMEBOX: ladder 30 min, AB 25 min, gates ~2 h. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r548-promote-gdnbf16-ring; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/$NEWM
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r548.sh
SRC=/srv/qwen5090/overlay-src/gdn-state-bf16-r1
MP=/srv/qwen5090/probes/mp_decode.py
TT=/srv/qwen5090/overlay-src/nvme-tier-r4/tests/gpu_nvme_ab.py
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SALT=$(( $(date +%s) % 100000 ))
BOOTED=0; PROMOTED=0
log(){ echo "$(date -Is) [r548] $*" | tee -a "$R/audit.log"; }
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
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R548 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CKPT/config.json" "$LM" "$SRC/Dockerfile.box" "$TT" "$MP" /srv/qwen5090/probes/agentic-edit.py /srv/qwen5090/probes/fn_needle_oai.py \
         /srv/qwen5090/probes/tooleval_summary.py /srv/qwen5090/probes/nostop_proxy.py /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
grep -q '^MAXBS=${MAXBS:-4}$' "$LIVE" || { log "ABORT: live launcher is not 4 slots"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")
case "$LIMG" in tabbyapi:nvme-tier-r4-e3det-r6|tabbyapi:nvme-tier-r4-e3det-r6-rawk) ;; *) log "ABORT: live image '$LIMG' is not R540/R546"; exit 3;; esac
LPOOL=$(sed -n 's/^CACHE=\${CACHE:-\([0-9]*\)}.*/\1/p' "$LIVE"); [ -n "$LPOOL" ] || { log "ABORT: no live pool"; exit 3; }
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE")
case " $LENV " in *" EXL3_GDN_STATE_BF16=1 "*) log "ABORT: bf16 already live"; exit 3;; esac
NIMG="$LIMG-gdnbf16"
log "S1 building $NIMG on $LIMG (live pool $LPOOL, before the lock)"
(cd "$SRC" && sudo docker build --build-arg BASE="$LIMG" --build-arg MANIFEST=manifest-on-r6.json -f Dockerfile.box -t "$NIMG" . ) > "$R/build.log" 2>&1 \
  || { log "ABORT: build: $(grep -aiE 'error|mismatch|abort' "$R/build.log" | head -3 | tr '\n' ' ' | cut -c1-300)"; exit 3; }
export GPU_QUEUE_NAME=r548-promote-gdnbf16-ring
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
log "=== S0 reference: live launcher, tier off ==="
up "$LIVE" REF NVME_TIER= || { finish FAILED; exit 3; }
# headroom rule (R548 try 2: bf16 @ 868,352 booted with cuda:1 free 637 MiB vs the daily 737, and a CUDA graph capture hit
# "GPU assert: out of memory graph.cu 51" on the first GSM8K requests after tool-eval): each card at boot must keep >= the
# live pool's free VRAM at boot minus 32 MiB
REF_F0=$(free0); REF_F1=$(free1); log "REF free at boot $REF_F0/$REF_F1 MiB (headroom floors $((REF_F0-32))/$((REF_F1-32)))"
REF1=$(greedy REF); REF30=$(greedy30k REF); log "REF @ $LPOOL: VRAM free $(vram); fingerprints c1 $REF1 / 30k $REF30"
[ "$REF1" != none ] && [ "$REF30" != none ] || { log "S0 FAIL: reference greedy failed"; finish FAILED; exit 3; }
ENVS="$LENV EXL3_GDN_STATE_BF16=1"; POOL_ON=0; BF1=; BF30=
log "=== S2 bf16 on, ladder from $LPOOL ==="
c=$(( LPOOL + 16384 )); LEND=$(( $(date +%s) + 1800 ))
for k in $(seq 1 20); do
  [ $(( LEND - $(date +%s) )) -lt 120 ] && { log "ladder stopped: timebox"; break; }
  up "$LIVE" "L-$c" NVME_TIER= IMG="$NIMG" EXTRA_ENV="$ENVS" CACHE=$c || break
  sudo grep -q "cache_size: $c$" $CFG || { log "ABORT: config lacks cache $c"; finish FAILED; exit 3; }
  f0=$(free0); f1=$(free1); v0=$(vram); a=$(greedy "L-$c")
  if [ "$f0" -lt 1200 ]; then log "[L] $c: moved placement (VRAM free at boot $v0), c1 $a"; c=$(( c + 16384 )); continue; fi
  [ "$f1" -ge $(( REF_F1 - 32 )) ] && [ "$f0" -ge $(( REF_F0 - 32 )) ] || { log "[L] $c: normal placement but free at boot $f0/$f1 < headroom floors $((REF_F0-32))/$((REF_F1-32)): ladder stops"; break; }
  b=$(greedy30k "L-$c")
  [ "$a" != none ] && [ "$b" != none ] || { log "[L] $c: greedy failed: STOP"; finish FAILED; exit 1; }
  [ -z "$BF1" ] && { BF1=$a; BF30=$b; }
  [ "$a" = "$BF1" ] && [ "$b" = "$BF30" ] || { log "[L] $c: normal placement but fingerprints $a / $b != $BF1 / $BF30: STOP"; finish FAILED; exit 1; }
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "L$c-pf" --kind prose --tokens 64 --conc 1 --runs 1 --ctx 120000 \
    --unique --salt $(( SALT + k*101 )) --out "$R/ladder.jsonl" > /dev/null 2>&1
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "L$c-c4" --kind code --tokens 2048 --conc 4 --runs 1 --out "$R/ladder.jsonl" > /dev/null 2>&1
  alive || { log "[L] $c boots but dies under 120k + c4"; break; }
  POOL_ON=$c; log "[L] $c: normal placement (boot $v0), fingerprints $a / $b, survives 120k + c4"; c=$(( c + 16384 ))
done
log "POOL_ON = $POOL_ON (live $LPOOL); bf16 fingerprints $BF1 / $BF30 (REF $REF1 / $REF30)"
[ "$POOL_ON" -gt "$LPOOL" ] || { log "no bf16 pool above the live pool passes"; finish DONE; exit 0; }
python3 - "$LIVE" "$CAND.new" "$LIMG" "$NIMG" "$LPOOL" "$POOL_ON" <<'PY' || { log "ABORT: candidate edit"; finish FAILED; exit 3; }
import sys,re; src,dst,limg,nimg,pa,pb=sys.argv[1:7]; s=open(src).read()
pat=re.escape(limg)+r"(?![-\w])"   # the exact name, not a longer tag that starts with it
assert len(re.findall(pat, s))==2, "image name count"   # IMG default + tier-default condition
s=re.sub(pat, nimg, s)
m=re.search(r"^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$", s, re.M); assert m
s=s[:m.start(1)]+m.group(1)+" EXL3_GDN_STATE_BF16=1"+s[m.end(1):]
m=re.search(r"^CACHE=\$\{CACHE:-%s\}( *# *)(.*)$" % pa, s, re.M); assert m
s=s[:m.start()]+"CACHE=${CACHE:-%s}%sR548: bf16 GDN state (ladder top %s at normal placement); was %s; %s" % (pb, m.group(1), pb, pa, m.group(2))+s[m.end():]
open(dst,"w").write(s)
PY
chmod 755 "$CAND.new" && mv "$CAND.new" "$CAND"
diff "$LIVE" "$CAND" | tee -a "$R/audit.log"
[ "$(diff "$LIVE" "$CAND" | grep -c '^>')" = 4 ] || { log "ABORT: candidate should change exactly 4 lines"; finish FAILED; exit 3; }
log "=== AB: mp_decode 4 boots, A = live @ $LPOOL, B = candidate @ $POOL_ON, tier off ==="
rc=0; n=0
for arm in A B B A; do
  n=$((n+1)); tag="$arm$n"; L="$LIVE"; [ "$arm" = B ] && L="$CAND"
  up "$L" "$tag" NVME_TIER= || { rc=1; break; }
  f0=$(free0); v0=$(vram); [ "$f0" -ge 1200 ] || { log "FAIL $tag: placement at boot $v0"; rc=1; break; }
  log "UP $tag: $(sudo docker inspect -f '{{.Config.Image}}' flashnext); VRAM free at boot $v0"
  python3 "$MP" run --url "$API" --model "$NEWM" --tag "$tag" --tokens 512 --out "$R/mp.jsonl" 2>&1 | tail -3 | tee -a "$R/audit.log"
done
python3 "$MP" compare "$R/mp.jsonl" 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
[ $rc = 0 ] || { finish FAILED; exit 1; }
python3 - "$R/analysis.txt" <<'PY' | tee -a "$R/audit.log"
import re,sys
g=[float(m) for m in re.findall(r"paired geo-mean B/A ([+-][0-9.]+) %", open(sys.argv[1]).read())]
print("AB OK" if len(g)==4 and min(g) >= -1.5 else f"AB STOP {g}")
PY
grep -q "AB OK" "$R/audit.log" || { log "AB: bf16 is slower than -1.5 % on a shape (or missing shapes); not promoting"; finish DONE; exit 0; }
log "=== G1: candidate with the NVMe tier ON ==="
up "$CAND" G1 || { log "G1 FAIL: no boot"; finish ABORTED; exit 3; }
got=$(cfgline)
case "$got" in *"cache_size: $POOL_ON cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;;
  *) log "G1 FAIL: config '$got'"; finish ABORTED; exit 3;; esac
img=$(sudo docker ps --format '{{.Image}}' -f name=flashnext); envs=$(sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' flashnext)
[ "$img" = "$NIMG" ] && echo "$envs" | grep -qx EXL3_GDN_STATE_BF16=1 || { log "G1 FAIL: image $img / bf16 env"; finish ABORTED; exit 3; }
tl=$(sudo docker logs flashnext 2>&1 | grep -a -- ' -- nvme tier: ON' | tail -1 | cut -c1-200); [ -n "$tl" ] || { log "G1 FAIL: NVMe tier not ON"; finish ABORTED; exit 3; }
f0=$(free0); v0=$(vram)   # placement is read at idle, before any request (a 30k prefill leaves cuda:0 ~800 MiB lower: R546 try 1)
a=$(greedy G1); b=$(greedy30k G1); log "G1: $got; tier: $tl; VRAM free at boot $v0; fingerprints c1 $a / 30k $b"
[ "$f0" -ge 1200 ] || { log "G1 FAIL: placement (cuda:0 free $f0 MiB at boot)"; finish ABORTED; exit 3; }
[ "$a" = "$BF1" ] && [ "$b" = "$BF30" ] || { log "G1 FAIL: fingerprints $a / $b != ladder $BF1 / $BF30"; finish ABORTED; exit 3; }
ok=1; for c in 60000 120000; do
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "prefill-$c" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c --unique --salt $((SALT+c/1000)) \
    --out "$R/prefill.jsonl" 2>&1 | grep -E "FAILED|Traceback" | tee -a "$R/audit.log"
  tps=$(python3 -c 'import json,sys; r=[x for x in map(json.loads, open(sys.argv[1])) if x.get("ttft_s") and x.get("prompt_tokens") and x.get("ctx_requested") == int(sys.argv[2])][-1]; print(int(r["prompt_tokens"]/r["ttft_s"]))' "$R/prefill.jsonl" $c 2>/dev/null || echo 0)
  floor=$([ $c = 60000 ] && echo 9000 || echo 9500); log "cold prefill ctx $c: $tps tok/s (floor $floor)"; [ "$tps" -ge $floor ] || ok=0; done
python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag G1b-c4 --kind code --tokens 2048 --conc 4 --runs 1 --out "$R/records.jsonl" 2>&1 \
  | grep -E "^  c=|FAILED|Traceback" | cut -c1-200 | tee -a "$R/audit.log"
alive || ok=0
[ $ok = 1 ] && log "G1b PASS" || { log "G1b FAIL"; finish ABORTED; exit 3; }
log "=== GT: crash-restart restore ==="
timeout 600 python3 "$TT" fill --out "$R" --sizes 30000 2>&1 | tail -3 | cut -c1-220 | tee -a "$R/audit.log"
timeout 300 python3 "$TT" wait-drained --out "$R" --container flashnext 2>&1 | tail -2 | cut -c1-240 | tee -a "$R/audit.log"
sudo docker logs flashnext > "$R/docker-pre-restart.log" 2>&1
up "$CAND" GT-restart || { log "GT FAIL: relaunch"; finish ABORTED; exit 3; }
sudo docker logs flashnext 2>&1 | grep -a "open scan" | tail -1 | cut -c1-200 | tee -a "$R/audit.log"
timeout 600 python3 "$TT" verify --out "$R" 2>&1 | tail -4 | cut -c1-240 | tee "$R/verify.txt" | tee -a "$R/audit.log"
grep -aq "VERIFY PASS" "$R/verify.txt" && log "GT PASS" || { log "GT FAIL"; finish ABORTED; exit 3; }
log "=== G2: agentic-edit ==="
python3 /srv/qwen5090/probes/agentic-edit.py --url "$API" --model "$NEWM" --tag BR --conc 1 4 --modes greedy sampled --out "$R/agentic-edit.jsonl" 2>&1 | tee "$R/agentic-edit.log" | grep -a "agentic-edit" | tee -a "$R/audit.log"
[ "$(grep -ac "6/6 ok" "$R/agentic-edit.log")" = 4 ] && alive && log "G2 PASS" || { log "G2 FAIL"; finish ABORTED; exit 3; }
log "=== G3: needles ==="
python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$NEWM" --tag needle --ctx-tokens 131072 240000 \
  --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee "$R/needle.log" | tee -a "$R/audit.log"
[ "$(grep -c '5/5 retrieved' "$R/needle.log")" = 2 ] && log "G3 PASS" || { log "G3 FAIL"; finish ABORTED; exit 3; }
log "=== G4: tool-eval 69 x 4 ==="
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$NEWM" --temperature 0.6 --top-p 0.95 --top-k 20 \
    --trials 4 --parallel 8 --json-file "$R/tooleval.json" > "$R/tooleval.log" 2>&1 )
python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval.json" BR 2>&1 | tee "$R/tooleval-summary.txt" | tee -a "$R/audit.log"
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
log "GSM8K n=500 flexible: ${g:-none}"
[ -n "$g" ] && [ "$g" != none ] && python3 -c "import sys; sys.exit(0 if float('$g') >= 0.970 else 1)" && log "G5 PASS" || { log "G5 FAIL"; finish ABORTED; exit 3; }
alive || { log "server not alive after the gates"; finish ABORTED; exit 3; }
log "=== PROMOTE: launch-flashnext.sh := launch-flashnext-r548.sh ==="
sudo cp -p "$LIVE" /srv/qwen5090/launch-flashnext.sh.pre-r548
cp "$CAND" "$LIVE.new" && chmod 755 "$LIVE.new" && mv "$LIVE.new" "$LIVE"
cmp -s "$CAND" "$LIVE" || { log "PROMOTE FAIL"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r548 "$LIVE"; finish ABORTED; exit 3; }
PROMOTED=1
up "$LIVE" live || { log "LIVE LAUNCH FAILED — rolling back"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r548 "$LIVE"; PROMOTED=0; finish ABORTED; exit 3; }
a2=$(greedy live); log "daily now: $(served_id) $(cfgline); c1 $a2 (new canonical $BF1 / $BF30); VRAM free $(vram)"
finish PROMOTED
