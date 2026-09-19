#!/usr/bin/env bash
# R536 — K=3 MoE coop V2 no-spill (codex decode-kernels r3, EXL3_MOE_COOP_V2_NOSPILL=1) rebased onto the SERVED chain as
# decode-kernels r3b (image $LIMG-nospill). R494 never produced a measurement (build patch header, then parked on the stale
# 3.05/R511 base). Bit-identical by construction: only the prefetch ring moves from registers to a shared-memory ring.
# On the 2.50bpw daily routed experts are K=3 on layers 0-11 and 37-47 (23 of 48); K=2 layers keep the served kernels.
#   gate 0  SASS: 12 nospill symbols, LOCAL:0 on each (LDL/STL counts recorded)
#   gate 1  torch.equal served V2 vs nospill: real layer 0 (K=3) on both cards at split-k 1 and 2, synthetic wide 0/1 x cb 0/1/2
#   A/B     8 boots A B B A B A A B, one image; A = daily env, B = daily env + EXL3_MOE_COOP_V2_NOSPILL=1. Per boot: c1 and 30k
#           fingerprints must be canonical (a mismatch fails the run), fn_bench code c1 x3 and c4 x2 (warm-up 1).
# Decision: B/A decode at c1 and c4 with a 95 % CI. Promote only if the CI lower bound at c1 or c4 is above 0 and neither is below.
# Try 1 (07:33) failed gate 1 in the harness, not the kernels: the per-stage timing walk (cudaGraphNodeGetDependencies) returned
# CUDA error 1 on every real run, and the synthetic runs read shapes from the old 3.05bpw default path. All gate-1 runs now pass
# --skip-stage-events (full-call CUDA-event timing and every torch.equal stay) and --model.
# GPU TIMEBOX 40 min. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r536-nospill; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/$MODEL
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/decode-kernels-r3b
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
CANON1=e7fb377c987d685c; CANON30=4a255910dee2d9c5
BOOTED=0
log(){ echo "$(date -Is) [r536] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){ sudo docker rm -f r536-test >/dev/null 2>&1
  if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R536 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$SRC/Dockerfile.box" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
ENVS=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE"); [ -n "$ENVS" ] || { log "ABORT: no daily EXTRA_ENV"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")
case " $ENVS " in *" EXL3_MOE_COOP_V2=1 "*) ;; *) log "ABORT: live env has no MoE coop V2"; exit 3;; esac
IMG="$LIMG-nospill"
log "building $IMG on $LIMG (before the lock); env: $ENVS"
(cd "$SRC" && sudo docker build --build-arg BASE="$LIMG" -f Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "ABORT: build: $(grep -aiE 'error' "$R/build.log" | head -3 | tr '\n' ' ' | cut -c1-300)"; exit 3; }
sudo docker run --rm --entrypoint python3 "$IMG" -c "import os; from exllamav3.ext import exllamav3_ext as e; assert hasattr(e,'exl3_moe_prefill_e3_det'); assert os.environ.get('EXL3_MOE_COOP_V2_NOSPILL')=='0'; print('image OK', e.__file__)" 2>&1 | tail -1 | tee -a "$R/audit.log" | grep -q '^image OK' || { log "ABORT: image check"; exit 3; }
export GPU_QUEUE_NAME=r536-nospill
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 2400 ))
log "lock held; timebox ends $(date -Is -d @$END)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
for i in $(seq 30); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 2; done
EV=(); for kv in $ENVS; do EV+=(-e "$kv"); done
T=(sudo docker run --rm --name r536-test --gpus all --ipc=host -v "$CKPT":/models/$MODEL:ro -v "$R":/results "${EV[@]}")
log "=== gate 0: SASS audit (12 nospill symbols, LOCAL:0) ==="
"${T[@]}" --entrypoint bash "$IMG" -c 'set -euo pipefail
SO=$(python3 -c "from exllamav3.ext import exllamav3_ext as e; print(e.__file__)" 2>/dev/null | tail -1); echo "so: $SO"
OBJ=$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump); A=/results/sass; mkdir -p $A/functions
$OBJ -res-usage "$SO" > $A/res.txt; $OBJ -symbols "$SO" > $A/symbols.txt
awk "/exl3_moe_coop_[ab]_nospill_kernel/ {print \$NF}" $A/symbols.txt | sort -u > $A/names.txt
echo "nospill symbols: $(wc -l < $A/names.txt)"; test "$(wc -l < $A/names.txt)" -eq 12
while IFS= read -r s; do $OBJ -sass -fun "$s" "$SO" > "$A/functions/$s.sass"; done < $A/names.txt
python3 /opt/decode-kernels-r3/analyze_sass.py --nospill $A/functions | cut -f1-3 | tee $A/counts.tsv || echo "LDL/STL present (recorded; served kernels also carry STACK:8, so this is not the gate)"
test "$(grep -A1 nospill $A/res.txt | grep -oE "LOCAL:[0-9]+" | grep -vc "LOCAL:0$")" -eq 0 || { echo "FAIL: a nospill instance uses LOCAL memory"; exit 1; }
echo "resource usage of nospill instances:"; grep -A1 nospill $A/res.txt | grep -oE "REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+" | sort | uniq -c' > "$R/gate0-sass.log" 2>&1; rc=$?
tail -22 "$R/gate0-sass.log" | cut -c1-200 | tee -a "$R/audit.log"
[ $rc = 0 ] || { log "GATE 0 FAIL (rc $rc)"; finish FAILED; exit 3; }
log "=== gate 1: torch.equal served V2 vs nospill (real layer 0 both cards, split-k 1/2; synthetic wide x cb) ==="
g1fail=0
for ks in 1 2; do for gpu in 0 1; do
  "${T[@]}" -e EXL3_MOE_COOP_KSPLIT=$ks --entrypoint python3 "$IMG" /opt/decode-kernels-r3/tests/test_moe_coop_v2_nospill.py \
    --model /models/$MODEL --layer 0 --device cuda:$gpu --iterations 500 --warmup 50 --skip-stage-events --json /results/real-ks$ks-gpu$gpu.json > "$R/gate1-real-ks$ks-gpu$gpu.log" 2>&1 || g1fail=1
  log "[real ks$ks gpu$gpu] $(grep -aE '^(PASS|FAIL)' "$R/gate1-real-ks$ks-gpu$gpu.log" | tail -1) $(grep -aE 'Error|Traceback' "$R/gate1-real-ks$ks-gpu$gpu.log" | tail -1 | cut -c1-160)"
done; done
for w in 0 1; do for cb in 0 1 2; do
  "${T[@]}" -e EXL3_MOE_COOP_WIDE=$w --entrypoint python3 "$IMG" /opt/decode-kernels-r3/tests/test_moe_coop_v2_nospill.py \
    --model /models/$MODEL --synthetic --bits 3 --cb $cb --device cuda:0 --iterations 200 --warmup 30 --skip-stage-events --json /results/synth-w$w-cb$cb.json \
    > "$R/gate1-synth-w$w-cb$cb.log" 2>&1 || g1fail=1
  log "[synth wide$w cb$cb] $(grep -aE '^(PASS|FAIL)' "$R/gate1-synth-w$w-cb$cb.log" | tail -1) $(grep -aE 'Error|Traceback' "$R/gate1-synth-w$w-cb$cb.log" | tail -1 | cut -c1-160)"
done; done
python3 - "$R" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,glob,os
def walk(o,key,acc):
    if isinstance(o,dict):
        for k,v in o.items():
            if k==key: acc.append(v)
            else: walk(v,key,acc)
    elif isinstance(o,list):
        for v in o: walk(v,key,acc)
    return acc
for f in sorted(glob.glob(sys.argv[1]+"/real-ks*-gpu*.json"))+sorted(glob.glob(sys.argv[1]+"/synth-*.json")):
    try: d=json.load(open(f))
    except Exception as e: print(os.path.basename(f),"unreadable",e); continue
    res=sorted({(g.get("stage"),g.get("resident_blocks_per_sm")) for gs in walk(d,"launch_geometry",[]) for g in (gs or [])},key=str)
    t=[]
    for r in d.get("results",[]):
        if "speedup" in r and r.get("pattern")=="distinct":
            t.append(f"R{r['R']} {r['off']['call_us_cuda_events']:.1f}->{r['on']['call_us_cuda_events']:.1f}us x{r['speedup']:.3f}")
    print(f"[gate1 {os.path.basename(f)}] failures {len(d.get('failures',[]))}; distinct: {'; '.join(t)} | resident {res[:8]}")
PY
[ $g1fail = 0 ] || { log "GATE 1 FAIL: see $R/gate1-*.log"; finish FAILED; exit 3; }
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
greedy30k(){ python3 - "$API" "$MODEL" "$R/greedy30k-$1.json" <<'PY' 2>/dev/null || echo none
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
log "=== A/B: 8 boots, A = daily env, B = + EXL3_MOE_COOP_V2_NOSPILL=1 (image $IMG both arms; NVMe tier off: IMG is not the served image) ==="
rc=0; n=0
for arm in A B B A B A A B; do
  n=$((n+1)); tag="$arm$n"; flag=0; [ "$arm" = B ] && flag=1
  [ $(( END - $(date +%s) )) -lt 180 ] && { log "SKIP $tag and later boots: timebox"; break; }
  sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
  "${CLEAN_ENV[@]}" IMG="$IMG" EXTRA_ENV="$ENVS EXL3_MOE_COOP_V2_NOSPILL=$flag" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 || { log "NO BOOT $tag"; rc=1; break; }
  for i in $(seq 150); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
  [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.Config.Image}}' flashnext)" = "$IMG" ] || { log "NO BOOT $tag"; rc=1; break; }
  has=$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c "^EXL3_MOE_COOP_V2_NOSPILL=$flag$")
  [ "$has" = 1 ] || { log "ABORT $tag: container env lacks EXL3_MOE_COOP_V2_NOSPILL=$flag"; rc=1; break; }
  a=$(greedy $tag); b=$(greedy30k $tag); log "UP $tag (nospill $flag) fingerprints c1 $a / 30k $b; VRAM free $(vram)"
  [ "$a" = "$CANON1" ] && [ "$b" = "$CANON30" ] || { log "FAIL $tag: fingerprints not canonical ($CANON1 / $CANON30)"; rc=1; break; }
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-c1" --kind code --tokens 2048 --warmup-runs 1 \
    --conc 1 --runs 3 --out "$R/ab.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-200 | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-c4" --kind code --tokens 2048 --warmup-runs 1 \
    --conc 4 --runs 2 --out "$R/ab.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-200 | tee -a "$R/audit.log"
done
python3 - "$R" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,math,statistics as st,collections,pathlib
R=pathlib.Path(sys.argv[1]); rounds=collections.defaultdict(list)
for l in open(R/"ab.jsonl"):
    r=json.loads(l); boot,shape=r["tag"].rsplit("-",1); rounds[(boot,shape,r["run"])].append(r)
runs=collections.defaultdict(list)
for (boot,shape,run),rs in rounds.items(): runs[(boot,shape)].append(sum(x["completion_tokens"] for x in rs)/max(x["round_wall_s"] for x in rs))
boots=sorted({b for b,_ in runs}, key=lambda b:int(b[1:]))
T={1:12.71,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306}
for shape in ("c1","c4"):
    bm={b:st.mean(runs[(b,shape)]) for b in boots if (b,shape) in runs}
    A=[v for b,v in bm.items() if b[0]=="A"]; B=[v for b,v in bm.items() if b[0]=="B"]
    print(f"{shape}: boot means " + " ".join(f"{b}={bm[b]:.1f}" for b in bm))
    if len(A)>1 and len(B)>1:
        d=st.mean(B)-st.mean(A); se=math.sqrt(st.variance(A)/len(A)+st.variance(B)/len(B)); ci=T.get(min(len(A),len(B))-1,2.0)*se
        print(f"{shape}: served V2 (A) {st.mean(A):.1f}, nospill (B) {st.mean(B):.1f} -> B-A {d/st.mean(A)*100:+.2f} %, 95 % CI [{(d-ci)/st.mean(A)*100:+.2f}, {(d+ci)/st.mean(A)*100:+.2f}] % (n {len(A)}+{len(B)} boots)")
PY
[ $rc = 0 ] && finish DONE || finish FAILED
