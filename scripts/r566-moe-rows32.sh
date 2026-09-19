#!/usr/bin/env bash
# R566 — moe-rows32 k3-r1 (Kimi K3 round, patches/exllamav3/moe-rows32/k3-r1): cooperative MoE decode for 17-32 verify rows,
# EXL3_MOE_COOP_ROWS32=1 (default off = served bytes). split (default) = two <=16-row coop spans per layer, bit-identical to
# try 2 (19:12 UTC): split bit-identical on both cards for 17..32 rows (router + hotspot); wide NOT repeatable (odd tails,
# rel L2 0.7-1.4, a race) -> Q2w dropped, harness runs --skip-wide; the timing section's 2-D fallback forward crashed (fixed).
# the chunked <=16-row path by construction; MODE=wide = one call (perf arm, not bit-identical for most tails). R560 / R562:
# above 16 rows the MoE layers leave the coop kernels (2.4-2.9x kernel time, +6-10 ms non-kernel, 48 blocking syncs at 32
# rows), which is why the served policy [[4, 3], [8, 1]] drops to depth 1 above 4 jobs (c6 513, c8 640 t/s).
#   build tabbyapi:moe-rows32-k3-r1 on the R561 image (pinned; extension rebuilt) before the lock
#   H  harness per card (tests/gpu_rows32.py): torch.equal 17..32 split vs chunked, <=16 untouched, refusals, dispatch
#      proofs; exit 0 on both cards required, else stop
#   boots at 8 slots @ 966,656, tier off, R561 env (prefetch flag stripped): boot free per card, c1 / 30k fingerprints
#      (must be canonical: c1 = 4 rows), fn_bench code + prose at c4 / c6 / c8, 1,024 forced x 2 after a warm-up
#        P1 flag off [[4, 3], [8, 1]] (served)   Q2 flag on [[4, 3], [8, 2]]   Q3 flag on [[8, 3]]   Q2w wide [[4, 3], [8, 2]]
#        P5 flag off [[4, 3], [5, 2], [8, 1]]: depth 2 at exactly 5 jobs = 15 rows, on the coop path today (K3 review S1:
#           never run; projected c5 +25 %). Rule for P5: c5 >= +12 % on code and prose with c4 / c6 / c8 within +-2 %.
#      the ladder is c4 / c5 / c6 / c8 on every arm (c5 = the replay's modal band)
# Decision rule (fixed, from the round's spec): a flagged policy wins if c6 and c8 aggregate are each >= +5 % over P1 on code
# and prose with c4 within +-1 %, flag-on boot free >= P1 - 32 MiB per card and fingerprints canonical; prefer Q2 over Q3, split
# over wide on a tie. A win -> promotion unit (replay + gates). Measurement only. GPU TIMEBOX 55 min. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r566-moe-rows32-try3; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
FB=/srv/qwen5090/probes/fn_bench.py
SRC=/srv/qwen5090/overlay-src/moe-rows32-k3-r1
BIMG=tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16
NIMG=tabbyapi:moe-rows32-k3-r1
C1=f4add302e176d78e; C30=4a255910dee2d9c5
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SALT=$(( $(date +%s) % 100000 ))
BOOTED=0
log(){ echo "$(date -Is) [r566] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
free0(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0; }
free1(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
finish(){
  if [ "$BOOTED" = 1 ]; then sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done; log "daily: $(served_id)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R566 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$FB" "$SRC/Dockerfile.box" "$SRC/tests/gpu_rows32.py"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
sudo docker image inspect "$BIMG" >/dev/null 2>&1 || { log "ABORT: $BIMG missing"; exit 3; }
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE" | tr ' ' '\n' | grep -v '^EXL3_NGRAM_PREFETCH2=' | tr '\n' ' ' | sed 's/ $//')
LPOOL=$(sed -n 's/^CACHE=\${CACHE:-\([0-9]*\)}.*/\1/p' "$LIVE"); grep -q '^MAXBS=${MAXBS:-8}$' "$LIVE" || { log "ABORT: live is not 8 slots"; exit 3; }
log "building $NIMG on $BIMG (extension rebuild)"
(cd "$SRC" && sudo docker build --build-arg BASE="$BIMG" -f Dockerfile.box -t "$NIMG" . ) > "$R/build.log" 2>&1 \
  || { log "ABORT: build: $(grep -aiE 'error|assert|abort' "$R/build.log" | head -4 | tr '\n' ' ' | cut -c1-400)"; exit 3; }
log "build OK: $(grep -aE 'rows32 defaults off|split default|mode wide|k3-r1 built' "$R/build.log" | tr '\n' ' ' | cut -c1-200)"
export GPU_QUEUE_NAME=r566-moe-rows32
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; live pool $LPOOL"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$NEWM" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
up(){ local tag=$1 i st lp; shift
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 ) & lp=$!
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
greedy30k(){ python3 - "$API" "$NEWM" <<'PY' 2>/dev/null || echo none
import json,sys,hashlib,urllib.request,random
api,model=sys.argv[1:3]
rng=random.Random(4242); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
msg=" ".join(rng.choice(words) for _ in range(23000))+"\n\nSummarize the passage above in one sentence, then write a haiku about it."
req=json.dumps({"model":model,"temperature":0,"max_tokens":128,"min_tokens":128,"messages":[{"role":"user","content":msg}]}).encode()
d=json.loads(urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=req,headers={"Content-Type":"application/json"}),timeout=900).read())
m=d["choices"][0]["message"]; print(hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16])
PY
}
END=$(( $(date +%s) + 3300 ))
log "=== H: harness per card ==="
hok=1
for d in 0 1; do
  sudo timeout 900 docker run --rm --gpus all -e EXL3_MOE_COOP_V2=1 -v /srv/qwen5090/models:/models:ro -v "$R":/out --entrypoint python3 "$NIMG" \
    /opt/moe-rows32-k3-r1/tests/gpu_rows32.py --device cuda:$d --json /out/harness-cuda$d.json --skip-wide > "$R/harness-cuda$d.log" 2>&1; hr=$?
  log "harness cuda:$d exit $hr: $(grep -aE 'PASS|FAIL|Error|equal' "$R/harness-cuda$d.log" | tail -2 | tr '\n' ' ' | cut -c1-240)"
  [ $hr = 0 ] || hok=0
done
[ $hok = 1 ] || { log "harness FAILED: no serving A/B"; finish FAILED; exit 1; }
bench(){ local tag=$1 conc kind
  for conc in 4 5 6 8; do for kind in code prose; do
    python3 "$FB" --url "$API" --model "$NEWM" --tag "$tag-c$conc-$kind" --kind $kind --tokens 1024 --warmup-runs 1 --conc $conc --runs 2 --out "$R/records.jsonl" 2>&1 \
      | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag c$conc $kind]/" | cut -c1-240 | tee -a "$R/audit.log"; done; done; }
: > "$R/boots.tsv"
for arm in P1 P5 Q2 Q3; do
  [ $(( END - $(date +%s) )) -lt 360 ] && { log "timebox: stopping before $arm"; break; }
  case $arm in
    P1) pol='[[4, 3], [8, 1]]'; ev="$LENV";;
    P5) pol='[[4, 3], [5, 2], [8, 1]]'; ev="$LENV";;
    Q2) pol='[[4, 3], [8, 2]]'; ev="$LENV EXL3_MOE_COOP_ROWS32=1";;
    Q3) pol='[[8, 3]]'; ev="$LENV EXL3_MOE_COOP_ROWS32=1";;
    Q2w) pol='[[4, 3], [8, 2]]'; ev="$LENV EXL3_MOE_COOP_ROWS32=1 EXL3_MOE_COOP_ROWS32_MODE=wide";;
  esac
  up $arm NVME_TIER= IMG="$NIMG" EXTRA_ENV="$ev" DRAFT_POLICY="$pol" || continue
  f0=$(free0); f1=$(free1)
  cp_=$(sudo grep -E 'draft_num_tokens_by_batch' $CFG | awk '{$1=$1; print}')
  fl=$(sudo docker exec flashnext env | grep -E '^EXL3_MOE_COOP_ROWS32' | tr '\n' ' ')
  a=$(greedy $arm); b=$(greedy30k)
  log "UP $arm: $cp_; env: ${fl:-none}; boot free $f0/$f1; c1 $a / 30k $b$([ "$a" = "$C1" ] && [ "$b" = "$C30" ] || echo ' (NOT CANONICAL)')"
  printf "%s\t%s\t%s\t%s\t%s\n" $arm $f0 $f1 $a $b >> "$R/boots.tsv"
  bench $arm
  sudo docker logs flashnext > "$R/docker-$arm.log" 2>&1
  grep -aE 'out of memory|GPU assert|Traceback|RuntimeError' "$R/docker-$arm.log" | head -3 | tee -a "$R/audit.log"
  alive && log "$arm done: alive" || log "$arm done: NOT ALIVE"
done
python3 - "$R/records.jsonl" "$R/boots.tsv" "$C1" "$C30" <<'PY' 2>&1 | tee -a "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
tok=collections.defaultdict(int); wall={}
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("ok"): tok[(r["tag"],r["run"])]+=r["completion_tokens"] or 0
    wall[(r["tag"],r["run"])]=r["round_wall_s"]
agg=collections.defaultdict(list)
for k,v in tok.items(): agg[k[0]].append(v/wall[k])
m={t:st.mean(v) for t,v in agg.items()}
for t in sorted(m): print(f"{t}: aggregate {m[t]:.1f} t/s (runs {len(agg[t])})")
boots={l.split()[0]:l.split() for l in open(sys.argv[2]) if l.strip()}
p1=boots.get("P1")
for arm in ("Q2","Q3"):
    if arm not in boots or not p1: continue
    why=[]; rows=[]
    b=boots[arm]
    if b[3]!=sys.argv[3] or b[4]!=sys.argv[4]: why.append("fingerprints")
    if int(b[1])<int(p1[1])-32 or int(b[2])<int(p1[2])-32: why.append("headroom")
    for c in (4,6,8):
        for k in ("code","prose"):
            x=m.get(f"{arm}-c{c}-{k}"); y=m.get(f"P1-c{c}-{k}")
            if not x or not y: why.append(f"missing c{c} {k}"); continue
            r=x/y; rows.append(f"c{c} {k} {x:.0f}/{y:.0f} {100*(r-1):+.1f}%")
            if c==4 and abs(r-1)>0.01: why.append(f"c4 {k} {100*(r-1):+.1f}%")
            if c in (6,8) and r<1.05: why.append(f"c{c} {k} {100*(r-1):+.1f}%")
    for k in ("code","prose"):
        c6=m.get(f"{arm}-c6-{k}"); c8=m.get(f"{arm}-c8-{k}")
        if c6 and c8: rows.append(f"{k} c8/c6 {c8/c6:.2f}")
    print(f"{arm}: "+"; ".join(rows))
    print(f"{arm}: "+("WINS" if not why else "no win ("+"; ".join(why)+")"))
b=boots.get("P5")
if b and p1:
    why=[]; rows=[]
    if b[3]!=sys.argv[3] or b[4]!=sys.argv[4]: rows.append("fingerprints differ (15-row verify is a new batch shape; gate on quality if it wins)")
    for c in (4,5,6,8):
        for k in ("code","prose"):
            x=m.get(f"P5-c{c}-{k}"); y=m.get(f"P1-c{c}-{k}")
            if not x or not y: why.append(f"missing c{c} {k}"); continue
            r=x/y; rows.append(f"c{c} {k} {x:.0f}/{y:.0f} {100*(r-1):+.1f}%")
            if c==5 and r<1.12: why.append(f"c5 {k} {100*(r-1):+.1f}%")
            if c!=5 and abs(r-1)>0.02: why.append(f"c{c} {k} {100*(r-1):+.1f}%")
    print("P5: "+"; ".join(rows))
    print("P5: "+("WINS" if not why else "no win ("+"; ".join(why)+")"))
PY
finish DONE
