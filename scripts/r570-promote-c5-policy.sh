#!/usr/bin/env bash
# R570 — confirm and promote the c5 draft policy P5 = [[4, 3], [5, 2], [8, 1]] (launcher-only: DRAFT_POLICY default line; no
# image or env change). R566 try 3 (one boot per arm, fn_bench aggregate, 1,024 tokens x 2 after a warm-up): c5 code 566 vs 478
# t/s (+18.5 %), c5 prose 546 vs 472 (+15.8 %); c4 / c6 / c8 within -1.8 .. +0.7 %; c1 / 30k fingerprints canonical (the policy is
# unchanged at 1..4 jobs); boot free 2,085 / 867 both. Why: served [8, 1] runs 5 jobs at depth 1 (10 verify rows) while 5 x (2 + 1)
# = 15 rows still fits the coop MoE path (MAX_BSZN 16); the served c5 aggregate sat below c4 (478 vs 528).
#   candidate = live launcher with the DRAFT_POLICY default changed (exactly 1 line)
#   AB 4 boots A B B A (A = live, B = candidate), tier off, 8 slots at the live pool: boot free per card, c1 / 30k fingerprints,
#      fn_bench aggregate c4 / c5 / c6 / c8 x code / prose, 1,024 tokens, warm-up 1 + 2 runs
#   Rule (fixed now): B fingerprints = A on every boot; min B boot free >= min A boot free - 32 MiB per card; mean B/A c5 >= +8 %
#      on code AND prose; every other shape's mean B/A >= -2 %. Otherwise: no promotion (finish DONE).
#   G1 tier ON (config, policy, image, fingerprints = REF) -> PROMOTE early (launch-flashnext.sh.pre-r570 kept) -> G2 agentic-edit
#      at c1 + c5, G3 needles 131k/240k, G4 tool-eval >= 82 (parallel 8), G5 GSM8K >= 0.970 at 5 concurrent (exercises the new
#      depth-2 rows; R565 ran 4); any failure rolls back. No GT: the policy does not touch the NVMe tier (R565 GT passed).
# GPU TIMEBOX ~1 h. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r570-promote-c5-policy; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/$NEWM
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r570.sh
MP=/srv/qwen5090/probes/mp_decode.py
TT=/srv/qwen5090/overlay-src/nvme-tier-r4/tests/gpu_nvme_ab.py
CFG=/srv/qwen5090/flashnext-config.yml
NIMG=tabbyapi:ngram-prefetch-r1-gdnbf16
OLDP="[[4, 3], [8, 1]]"; NEWP="[[4, 3], [5, 2], [8, 1]]"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SALT=$(( $(date +%s) % 100000 ))
BOOTED=0; PROMOTED=0
log(){ echo "$(date -Is) [r570] $*" | tee -a "$R/audit.log"; }
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
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R570 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CKPT/config.json" "$LM" "$TT" "$MP" /srv/qwen5090/probes/agentic-edit.py /srv/qwen5090/probes/fn_needle_oai.py \
         /srv/qwen5090/probes/tooleval_summary.py /srv/qwen5090/probes/nostop_proxy.py /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")
[ "$LIMG" = "$NIMG" ] || { log "ABORT: live image '$LIMG' is not R565's"; exit 3; }
grep -q '^MAXBS=${MAXBS:-8}$' "$LIVE" || { log "ABORT: live launcher is not 8 slots (R561)"; exit 3; }
LPOOL=$(sed -n 's/^CACHE=\${CACHE:-\([0-9]*\)}.*/\1/p' "$LIVE")
python3 - "$LIVE" "$CAND.new" "$OLDP" "$NEWP" <<'PY' || { log "ABORT: candidate edit"; exit 3; }
import sys; src,dst,old,new=sys.argv[1:5]; s=open(src).read()
a="DRAFT_POLICY=${DRAFT_POLICY-"+old+"}"; assert s.count(a)==1, "policy line"
open(dst,"w").write(s.replace(a,"DRAFT_POLICY=${DRAFT_POLICY-"+new+"}"))
PY
chmod 755 "$CAND.new" && mv "$CAND.new" "$CAND"
diff "$LIVE" "$CAND" | tee -a "$R/audit.log"
[ "$(diff "$LIVE" "$CAND" | grep -c '^>')" = 1 ] || { log "ABORT: candidate should change exactly 1 line"; exit 3; }
export GPU_QUEUE_NAME=r570-promote-c5-policy
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
bench(){ local tag=$1 conc kind
  for conc in 4 5 6 8; do for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$tag-c$conc-$kind" --kind $kind --tokens 1024 --warmup-runs 1 --conc $conc --runs 2 --out "$R/records.jsonl" 2>&1 \
      | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag c$conc $kind]/" | cut -c1-240 | tee -a "$R/audit.log"; done; done; }
log "=== AB: 4 boots A B B A, A = live, B = candidate, tier off, pool $LPOOL x 8 ==="
rc=0; n=0; REF1=; REF30=
: > "$R/ab.tsv"
for arm in A B B A; do
  n=$((n+1)); tag="$arm$n"; L="$LIVE"; [ "$arm" = B ] && L="$CAND"
  up "$L" "$tag" NVME_TIER= || { rc=1; break; }
  f0=$(free0); f1=$(free1)
  cp_=$(sudo grep -E 'draft_num_tokens_by_batch' $CFG | awk '{$1=$1; print}')
  a=$(greedy "$tag"); b=$(greedy30k "$tag")
  [ -z "$REF1" ] && { REF1=$a; REF30=$b; }
  log "UP $tag: $cp_; boot free $f0/$f1; c1 $a / 30k $b"
  printf "%s\t%s\t%s\t%s\t%s\n" "$arm" "$f0" "$f1" "$a" "$b" >> "$R/ab.tsv"
  [ "$a" != none ] && [ "$b" != none ] || { log "FAIL $tag: greedy failed"; rc=1; break; }
  bench "$tag"
  alive || { log "FAIL $tag: server not alive"; rc=1; break; }
done
[ $rc = 0 ] || { finish FAILED; exit 1; }
python3 - "$R/ab.tsv" "$R/records.jsonl" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
rows=[l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
A=[r for r in rows if r[0]=="A"]; B=[r for r in rows if r[0]=="B"]
why=[]
if len(A)!=2 or len(B)!=2: why.append("boots missing")
if {(r[3],r[4]) for r in rows}!={(rows[0][3],rows[0][4])}: why.append("fingerprints differ across boots")
for i,card in ((1,"cuda:0"),(2,"cuda:1")):
    if B and A and min(int(r[i]) for r in B) < min(int(r[i]) for r in A)-32: why.append("headroom "+card)
tok=collections.defaultdict(int); wall={}
for l in open(sys.argv[2]):
    r=json.loads(l)
    if r.get("ok"): tok[(r["tag"],r["run"])]+=r["completion_tokens"] or 0
    wall[(r["tag"],r["run"])]=r["round_wall_s"]
agg=collections.defaultdict(list)
for (t,run),v in tok.items(): agg[t].append(v/wall[(t,run)])
for conc in (4,5,6,8):
    for kind in ("code","prose"):
        ma=[x for t,v in agg.items() if t[0]=="A" and t.endswith(f"-c{conc}-{kind}") for x in v]
        mb=[x for t,v in agg.items() if t[0]=="B" and t.endswith(f"-c{conc}-{kind}") for x in v]
        if not ma or not mb: why.append(f"c{conc} {kind} missing"); continue
        d=st.mean(mb)/st.mean(ma)-1
        print(f"c{conc} {kind}: A {st.mean(ma):.1f} (n {len(ma)}) B {st.mean(mb):.1f} (n {len(mb)}) B/A {d*100:+.1f} %")
        if conc==5 and d < 0.08: why.append(f"c5 {kind} {d*100:+.1f} % < +8 %")
        if conc!=5 and d < -0.02: why.append(f"c{conc} {kind} {d*100:+.1f} % < -2 %")
print("AB OK" if not why else "AB STOP: "+"; ".join(why))
PY
grep -q "AB OK" "$R/audit.log" || { log "AB rule not met; not promoting"; finish DONE; exit 0; }
log "=== G1: candidate with the NVMe tier ON ==="
up "$CAND" G1 || { log "G1 FAIL: no boot"; finish ABORTED; exit 3; }
got=$(cfgline)
case "$got" in *"cache_size: $LPOOL cache_mode: 8,8 max_batch_size: 8 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;;
  *) log "G1 FAIL: config '$got'"; finish ABORTED; exit 3;; esac
img=$(sudo docker ps --format '{{.Image}}' -f name=flashnext); envs=$(sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' flashnext)
pl=$(sudo grep -E 'draft_num_tokens_by_batch' $CFG | awk '{$1=$1; print}')
[ "$img" = "$NIMG" ] && echo "$envs" | grep -qx EXL3_NGRAM_PREFETCH2=1 && [ "$pl" = "draft_num_tokens_by_batch: $NEWP" ] || { log "G1 FAIL: image $img / prefetch env / policy '$pl'"; finish ABORTED; exit 3; }
tl=$(sudo docker logs flashnext 2>&1 | grep -a -- ' -- nvme tier: ON' | tail -1 | cut -c1-200); [ -n "$tl" ] || { log "G1 FAIL: NVMe tier not ON"; finish ABORTED; exit 3; }
v0=$(vram); a=$(greedy G1); b=$(greedy30k G1); log "G1: $got; tier: $tl; VRAM free at boot $v0; fingerprints c1 $a / 30k $b"
[ "$a" = "$REF1" ] && [ "$b" = "$REF30" ] || { log "G1 FAIL: fingerprints $a / $b != REF $REF1 / $REF30"; finish ABORTED; exit 3; }
log "=== PROMOTE (early): launch-flashnext.sh := launch-flashnext-r570.sh ==="
sudo cp -p "$LIVE" /srv/qwen5090/launch-flashnext.sh.pre-r570
cp "$CAND" "$LIVE.new" && chmod 755 "$LIVE.new" && mv "$LIVE.new" "$LIVE"
cmp -s "$CAND" "$LIVE" || { log "PROMOTE FAIL"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r570 "$LIVE"; finish ABORTED; exit 3; }
PROMOTED=1; log "promoted: live launcher = R565 + draft policy $NEWP (the G1 container is that launcher's boot); post-promotion gates follow"
rollback(){ log "POST-PROMOTION GATE FAILED ($1): rolling back to .pre-r570"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r570 "$LIVE"; PROMOTED=0; finish "ROLLED-BACK ($1)"; exit 3; }
log "=== G2: agentic-edit ==="
python3 /srv/qwen5090/probes/agentic-edit.py --url "$API" --model "$NEWM" --tag BR --conc 1 5 --modes greedy sampled --out "$R/agentic-edit.jsonl" 2>&1 | tee "$R/agentic-edit.log" | grep -a "agentic-edit" | tee -a "$R/audit.log"
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
  --model_args "base_url=http://127.0.0.1:8031/v1/chat/completions,model=$NEWM,tokenizer=$CKPT,num_concurrent=5,max_retries=1,tokenized_requests=False" \
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
