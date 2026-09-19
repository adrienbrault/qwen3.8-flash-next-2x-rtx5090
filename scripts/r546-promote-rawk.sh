#!/usr/bin/env bash
# R546 — promote the QSA raw-key ring (patches/exllamav3/qsa-rawk-ring/r1, EXL3_QSA_RAWK_RING=1) on top of whatever the
# daily is when this runs (R540 image, or R545's -gdnbf16 if that promoted: the ring touches disjoint files).
# R544b: the ring is bit-exact end to end at equal layer placement; the greedy output depends on placement only (flag off
# with layer 23 moved to cuda:0 at 753,664 = ring on at 851k-917k = 4fad2dbf; ring on at 950,272 with normal placement =
# canonical e7fb377c / 4a255910). Normal placement = cuda:0 free >= 1,200 MiB at idle.
#   S0 reference: boot the live launcher (tier off), record its fingerprints REF1 / REF30 and VRAM
#   S1 build $LIMG-rawk; harness both cards
#   S2 ladder ring on, tier off: from live pool + 16,384 in +16,384 steps (max 20) until no boot; at each step record
#      placement + c1; at normal placement require c1 = REF1 and 30k = REF30 and survive a cold 120k prefill + c4 round;
#      POOL_ON = highest such pool
#   S3 candidate = live + IMG (2 lines) + EXL3_QSA_RAWK_RING=1 + CACHE=POOL_ON (exactly 4 changed lines)
#   AB 8 boots A B B A B A A B, tier off: A = live, B = candidate. Bit-exact, so both arms decode the same text: STOP (no
#      promotion) if any shape's B-A is below -1.5 % on round throughput, or salted cold prefill (60k x 2, 120k x 1 per boot)
#      is below 0.95x A; G1b floors = 0.95 x A's prefill means (R546b failed a fixed 9,000 floor at 8,901; user 2026-09-19:
#      +20 % KV is worth a small prefill dip)
#   G1 candidate with the tier ON (fingerprints = REF), G1b prefill floors + 120k + c4, GT tier crash-restart restore,
#      G2 agentic-edit, G3 needles 131k/240k, G4 tool-eval >= 82, G5 GSM8K >= 0.970
# PASS -> launch-flashnext.sh := candidate (old live kept as launch-flashnext.sh.pre-r546).
# GPU TIMEBOX: ladder 30 min, AB 40 min, gates ~2 h. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r546-promote-rawk; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/$NEWM
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r546.sh
SRC=/srv/qwen5090/overlay-src/qsa-rawk-ring-r1
TT=/srv/qwen5090/overlay-src/nvme-tier-r4/tests/gpu_nvme_ab.py
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SALT=$(( $(date +%s) % 100000 ))
BOOTED=0; PROMOTED=0
log(){ echo "$(date -Is) [r546] $*" | tee -a "$R/audit.log"; }
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
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R546 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CKPT/config.json" "$LM" "$SRC/Dockerfile.box" "$TT" /srv/qwen5090/probes/agentic-edit.py /srv/qwen5090/probes/fn_needle_oai.py \
         /srv/qwen5090/probes/tooleval_summary.py /srv/qwen5090/probes/nostop_proxy.py /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
grep -q '^MAXBS=${MAXBS:-4}$' "$LIVE" || { log "ABORT: live launcher is not 4 slots"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")
case "$LIMG" in tabbyapi:nvme-tier-r4-e3det-r6|tabbyapi:nvme-tier-r4-e3det-r6-gdnbf16) ;; *) log "ABORT: live image '$LIMG' is not R540/R545"; exit 3;; esac
LPOOL=$(sed -n 's/^CACHE=\${CACHE:-\([0-9]*\)}.*/\1/p' "$LIVE"); [ -n "$LPOOL" ] || { log "ABORT: no live pool"; exit 3; }
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE")
case " $LENV " in *" EXL3_QSA_RAWK_RING=1 "*) log "ABORT: ring already live"; exit 3;; esac
NIMG="$LIMG-rawk"
log "S1 building $NIMG on $LIMG (live pool $LPOOL, before the lock)"
(cd "$SRC" && sudo docker build --build-arg BASE="$LIMG" -f Dockerfile.box -t "$NIMG" . ) > "$R/build.log" 2>&1 \
  || { log "ABORT: build: $(grep -aiE 'error|mismatch|abort' "$R/build.log" | head -3 | tr '\n' ' ' | cut -c1-300)"; exit 3; }
export GPU_QUEUE_NAME=r546-promote-rawk
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
for i in $(seq 30); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 2; done
h=0
for G in 0 1; do
  sudo docker run --rm --gpus all --ipc=host --entrypoint python3 -v "$R":/out "$NIMG" \
    /opt/qsa-rawk-ring-r1/tests/gpu_rawk_ring.py --device cuda:$G --json /out/harness-gpu$G.json > "$R/harness-gpu$G.log" 2>&1 || h=1
  log "[harness gpu$G] $(tail -2 "$R/harness-gpu$G.log" | tr '\n' ' ' | cut -c1-200)"
done
[ $h = 0 ] || { log "S1 FAIL: harness"; finish FAILED; exit 3; }
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
ENVS="$LENV EXL3_QSA_RAWK_RING=1"; POOL_ON=0
# GATES_ONLY=<pool>: resume at G1 with a pool whose ladder and AB already passed in an earlier try of this unit (the try's
# analysis.txt must say AB OK for that pool); S0 still runs so REF comes from this boot.
if [ -n "${GATES_ONLY:-}" ]; then
  T1=${GATES_FROM:?GATES_FROM=<earlier results dir>}
  grep -q "POOL_ON = $GATES_ONLY " "$T1/audit.log" && grep -q "AB OK" "$T1/analysis.txt" || { log "ABORT: $T1 has no passing ladder+AB at $GATES_ONLY"; finish FAILED; exit 3; }
  POOL_ON=$GATES_ONLY; log "GATES_ONLY: ladder and AB from $T1 (POOL_ON $POOL_ON, AB OK)"; cp "$T1/analysis.txt" "$R/analysis-from-try.txt"
fi
if [ -z "${GATES_ONLY:-}" ]; then
log "=== S2 ring on, ladder from $LPOOL ==="
c=$(( LPOOL + 16384 )); LEND=$(( $(date +%s) + 1800 ))
for k in $(seq 1 20); do
  [ $(( LEND - $(date +%s) )) -lt 120 ] && { log "ladder stopped: timebox"; break; }
  up "$LIVE" "L-$c" NVME_TIER= IMG="$NIMG" EXTRA_ENV="$ENVS" CACHE=$c || break
  sudo grep -q "cache_size: $c$" $CFG || { log "ABORT: config lacks cache $c"; finish FAILED; exit 3; }
  f0=$(free0); f1=$(free1); a=$(greedy "L-$c")
  if [ "$f0" -lt 1200 ]; then log "[L] $c: moved placement (VRAM free $(vram)), c1 $a"; c=$(( c + 16384 )); continue; fi
  [ "$f1" -ge $(( REF_F1 - 32 )) ] && [ "$f0" -ge $(( REF_F0 - 32 )) ] || { log "[L] $c: normal placement but free at boot $f0/$f1 < headroom floors $((REF_F0-32))/$((REF_F1-32)): ladder stops"; break; }
  b=$(greedy30k "L-$c")
  [ "$a" = "$REF1" ] && [ "$b" = "$REF30" ] || { log "[L] $c: normal placement but fingerprints $a / $b != REF: STOP"; finish FAILED; exit 1; }
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "L$c-pf" --kind prose --tokens 64 --conc 1 --runs 1 --ctx 120000 \
    --unique --salt $(( SALT + k*101 )) --out "$R/ladder.jsonl" > /dev/null 2>&1
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "L$c-c4" --kind code --tokens 2048 --conc 4 --runs 1 --out "$R/ladder.jsonl" > /dev/null 2>&1
  alive || { log "[L] $c boots but dies under 120k + c4"; break; }
  POOL_ON=$c; log "[L] $c: normal placement, fingerprints = REF, survives 120k + c4; VRAM free $(vram)"; c=$(( c + 16384 ))
done
log "POOL_ON = $POOL_ON (live $LPOOL)"
fi
[ "$POOL_ON" -gt "$LPOOL" ] || { log "no ring pool above the live pool passes"; finish DONE; exit 0; }
python3 - "$LIVE" "$CAND.new" "$LIMG" "$NIMG" "$LPOOL" "$POOL_ON" <<'PY' || { log "ABORT: candidate edit"; finish FAILED; exit 3; }
import sys,re; src,dst,limg,nimg,pa,pb=sys.argv[1:7]; s=open(src).read()
pat=re.escape(limg)+r"(?![-\w])"   # the exact name, not a longer tag that starts with it
assert len(re.findall(pat, s))==2, "image name count"   # IMG default + tier-default condition
s=re.sub(pat, nimg, s)
m=re.search(r"^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$", s, re.M); assert m
s=s[:m.start(1)]+m.group(1)+" EXL3_QSA_RAWK_RING=1"+s[m.end(1):]
m=re.search(r"^CACHE=\$\{CACHE:-%s\}( *# *)(.*)$" % pa, s, re.M); assert m
s=s[:m.start()]+"CACHE=${CACHE:-%s}%sR546: QSA raw-key ring (R544b ladder top %s at normal placement); was %s; %s" % (pb, m.group(1), pb, pa, m.group(2))+s[m.end():]
open(dst,"w").write(s)
PY
chmod 755 "$CAND.new" && mv "$CAND.new" "$CAND"
diff "$LIVE" "$CAND" | tee -a "$R/audit.log"
[ "$(diff "$LIVE" "$CAND" | grep -c '^>')" = 4 ] || { log "ABORT: candidate should change exactly 4 lines"; finish FAILED; exit 3; }
if [ -z "${GATES_ONLY:-}" ]; then
log "=== AB: 8 boots, A = live @ $LPOOL, B = candidate @ $POOL_ON, tier off ==="
AB_END=$(( $(date +%s) + 2400 )); rc=0; n=0
for arm in A B B A B A A B; do
  n=$((n+1)); tag="$arm$n"; L="$LIVE"; [ "$arm" = B ] && L="$CAND"
  [ $(( AB_END - $(date +%s) )) -lt 180 ] && { log "SKIP $tag and later boots: timebox"; break; }
  up "$L" "$tag" NVME_TIER= || { rc=1; break; }
  a=$(greedy $tag); log "UP $tag: VRAM free $(vram); c1 $a"
  [ "$a" = "$REF1" ] || { log "FAIL $tag: c1 $a != REF"; rc=1; break; }
  for c in 60000 60000 120000; do   # salted cold prefills (R546b: G1b 60k 8,901 vs a 9,000 floor; the ring also updates planes in prefill)
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$tag-pf$c" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c \
      --unique --salt $(( SALT + n*1009 + c/1000 + RANDOM )) --out "$R/prefill-ab.jsonl" > /dev/null 2>&1; done
  for spec in "code 1 3" "code 4 2" "prose 1 2" "prose 4 2"; do set -- $spec
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$tag-${1:0:1}$2" --kind $1 --tokens 2048 --warmup-runs 1 \
      --conc $2 --runs $3 --out "$R/ab.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-200 | tee -a "$R/audit.log"; done
done
python3 - "$R" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,math,statistics as st,collections,pathlib
R=pathlib.Path(sys.argv[1]); rounds=collections.defaultdict(list)
for l in open(R/"ab.jsonl"):
    r=json.loads(l); boot,shape=r["tag"].rsplit("-",1); rounds[(boot,shape,r["run"])].append(r)
thr=collections.defaultdict(list)
for (boot,shape,run),rs in rounds.items(): thr[(boot,shape)].append(sum(x["completion_tokens"] for x in rs)/max(x["round_wall_s"] for x in rs))
boots=sorted({b for b,_ in thr}, key=lambda b:int(b[1:]))
T={1:12.71,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306}
stop=0
for shape in ("c1","c4","p1","p4"):
    bm={b:st.mean(thr[(b,shape)]) for b in boots if thr.get((b,shape))}
    A=[v for b,v in bm.items() if b[0]=="A"]; B=[v for b,v in bm.items() if b[0]=="B"]
    if len(A)>1 and len(B)>1:
        d=st.mean(B)-st.mean(A); se=math.sqrt(st.variance(A)/len(A)+st.variance(B)/len(B)); ci=T.get(min(len(A),len(B))-1,2.0)*se
        pct=d/st.mean(A)*100; stop|= pct < -1.5
        print(f"{shape}: A {st.mean(A):.1f}, B {st.mean(B):.1f} -> {pct:+.2f} %, 95 % CI [{(d-ci)/st.mean(A)*100:+.2f}, {(d+ci)/st.mean(A)*100:+.2f}] % (n {len(A)}+{len(B)} boots)")
pf=collections.defaultdict(list)
for l in open(R/"prefill-ab.jsonl"):
    r=json.loads(l)
    if r.get("ttft_s") and r.get("prompt_tokens"): pf[(r["tag"].split("-")[0][0], r["ctx_requested"])].append(r["prompt_tokens"]/r["ttft_s"])
floors={}
for c in (60000,120000):
    A,B=pf.get(("A",c),[]),pf.get(("B",c),[])
    if A and B:
        ratio=st.mean(B)/st.mean(A); floors[c]=int(0.95*st.mean(A)); stop|= ratio < 0.95
        print(f"prefill {c}: A {st.mean(A):,.0f} (n {len(A)}, range {min(A):,.0f}-{max(A):,.0f}), B {st.mean(B):,.0f} (n {len(B)}, range {min(B):,.0f}-{max(B):,.0f}) -> {ratio:.3f}x")
    else: stop=1; print(f"prefill {c}: missing samples")
open(R/"prefill-floors.txt","w").write(" ".join(f"{c}:{v}" for c,v in floors.items()))
print("AB STOP" if stop else "AB OK")
PY
[ $rc = 0 ] || { finish FAILED; exit 1; }
grep -q "AB OK" "$R/analysis.txt" || { log "AB: the ring is slower than -1.5 % on a shape; not promoting"; finish DONE; exit 0; }
fi
log "=== G1: candidate with the NVMe tier ON ==="
up "$CAND" G1 || { log "G1 FAIL: no boot"; finish ABORTED; exit 3; }
got=$(cfgline)
case "$got" in *"cache_size: $POOL_ON cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;;
  *) log "G1 FAIL: config '$got'"; finish ABORTED; exit 3;; esac
img=$(sudo docker ps --format '{{.Image}}' -f name=flashnext); envs=$(sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' flashnext)
[ "$img" = "$NIMG" ] && echo "$envs" | grep -qx EXL3_QSA_RAWK_RING=1 || { log "G1 FAIL: image $img / ring env"; finish ABORTED; exit 3; }
tl=$(sudo docker logs flashnext 2>&1 | grep -a -- ' -- nvme tier: ON' | tail -1 | cut -c1-200); [ -n "$tl" ] || { log "G1 FAIL: NVMe tier not ON"; finish ABORTED; exit 3; }
f0=$(free0); v0=$(vram)   # placement is read at idle, before any request (a 30k prefill leaves cuda:0 ~800 MiB lower: R546 try 1)
a=$(greedy G1); b=$(greedy30k G1); log "G1: $got; tier: $tl; VRAM free at boot $v0; fingerprints c1 $a / 30k $b"
[ "$f0" -ge 1200 ] || { log "G1 FAIL: placement (cuda:0 free $f0 MiB at boot)"; finish ABORTED; exit 3; }
[ "$a" = "$REF1" ] && [ "$b" = "$REF30" ] || { log "G1 FAIL: fingerprints"; finish ABORTED; exit 3; }
PF="$R/prefill-floors.txt"; [ -n "${GATES_ONLY:-}" ] && PF="${GATES_FROM:-/nonexistent}/prefill-floors.txt"
ok=1; for c in 60000 120000; do
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "prefill-$c" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c --unique --salt $((SALT+c/1000)) \
    --out "$R/prefill.jsonl" 2>&1 | grep -E "FAILED|Traceback" | tee -a "$R/audit.log"
  tps=$(python3 -c 'import json,sys; r=[x for x in map(json.loads, open(sys.argv[1])) if x.get("ttft_s") and x.get("prompt_tokens") and x.get("ctx_requested") == int(sys.argv[2])][-1]; print(int(r["prompt_tokens"]/r["ttft_s"]))' "$R/prefill.jsonl" $c 2>/dev/null || echo 0)
  floor=$(tr ' ' '\n' < "$PF" | sed -n "s/^$c://p"); [ -n "$floor" ] || floor=$([ $c = 60000 ] && echo 9000 || echo 9500)
  log "cold prefill ctx $c: $tps tok/s (floor $floor = 0.95 x the AB's A arm)"; [ "$tps" -ge $floor ] || ok=0; done
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
python3 /srv/qwen5090/probes/agentic-edit.py --url "$API" --model "$NEWM" --tag RK --conc 1 4 --modes greedy sampled --out "$R/agentic-edit.jsonl" 2>&1 | tee "$R/agentic-edit.log" | grep -a "agentic-edit" | tee -a "$R/audit.log"
[ "$(grep -ac "6/6 ok" "$R/agentic-edit.log")" = 4 ] && alive && log "G2 PASS" || { log "G2 FAIL"; finish ABORTED; exit 3; }
log "=== G3: needles ==="
python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$NEWM" --tag needle --ctx-tokens 131072 240000 \
  --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee "$R/needle.log" | tee -a "$R/audit.log"
[ "$(grep -c '5/5 retrieved' "$R/needle.log")" = 2 ] && log "G3 PASS" || { log "G3 FAIL"; finish ABORTED; exit 3; }
log "=== G4: tool-eval 69 x 4 ==="
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$NEWM" --temperature 0.6 --top-p 0.95 --top-k 20 \
    --trials 4 --parallel 8 --json-file "$R/tooleval.json" > "$R/tooleval.log" 2>&1 )
python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval.json" RK 2>&1 | tee "$R/tooleval-summary.txt" | tee -a "$R/audit.log"
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
log "=== PROMOTE: launch-flashnext.sh := launch-flashnext-r546.sh ==="
sudo cp -p "$LIVE" /srv/qwen5090/launch-flashnext.sh.pre-r546
cp "$CAND" "$LIVE.new" && chmod 755 "$LIVE.new" && mv "$LIVE.new" "$LIVE"
cmp -s "$CAND" "$LIVE" || { log "PROMOTE FAIL"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r546 "$LIVE"; finish ABORTED; exit 3; }
PROMOTED=1
up "$LIVE" live || { log "LIVE LAUNCH FAILED — rolling back"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r546 "$LIVE"; PROMOTED=0; finish ABORTED; exit 3; }
a2=$(greedy live); log "daily now: $(served_id) $(cfgline); c1 $a2 (REF $REF1); VRAM free $(vram)"
finish PROMOTED
