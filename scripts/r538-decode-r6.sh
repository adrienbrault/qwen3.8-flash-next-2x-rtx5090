#!/usr/bin/env bash
# R538 — decode-kernels r6 (r5 rebased onto the served chain by an Opus round): EXL3_GDN_BA_WARP1 (one-warp GDN B/A GEMV at
# c1; r5 planning estimate -72.7 us/step at c1/d3, c4 falls back by construction), EXL3_HC_APPLY_WARP1 (one-warp hc_apply
# column tiles), EXL3_GR_STATE_REGRID (V2 state re-grid, ported to the int8 mixer the daily runs). All default off, bit-order
# preserving. EXL3_GR_FREE_UP_H is not carried (the int8 mixer already released that memory).
#   gate 1  tests/test_r6_kernels.py on both cards: torch.equal rows 1..16 for every selector, launch check (candidate kernel ran)
#   A/B     8 boots A B B A B A A B, one image; A = daily env, B = + the three selectors. Per boot: c1 and 30k fingerprints must
#           be canonical, fn_bench code c1 x3 and c4 x2, prose c1 x2 (warm-up 1).
# Decision: B/A decode per cell with a 95 % CI; attribution run (GDN only) only if c1 improves with the CI above 0.
# GPU TIMEBOX 45 min. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r538-decode-r6; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/$MODEL
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/decode-kernels-r6
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
CANON1=e7fb377c987d685c; CANON30=4a255910dee2d9c5
BOOTED=0
log(){ echo "$(date -Is) [r538] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){ sudo docker rm -f r538-test >/dev/null 2>&1
  if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R538 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$SRC/overlay/Dockerfile.box" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
ENVS=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE"); [ -n "$ENVS" ] || { log "ABORT: no daily EXTRA_ENV"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")
case " $ENVS " in *" EXL3_MOE_COOP_V2=1 "*) ;; *) log "ABORT: live env has no MoE coop V2"; exit 3;; esac
IMG="$LIMG-r6"
log "building $IMG on $LIMG (before the lock); env: $ENVS"
(cd "$SRC" && sudo docker build --build-arg BASE="$LIMG" -f overlay/Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "ABORT: build: $(grep -aiE 'error' "$R/build.log" | head -3 | tr '\n' ' ' | cut -c1-300)"; exit 3; }
sudo docker run --rm --entrypoint python3 "$IMG" -c "import os; from exllamav3.ext import exllamav3_ext as e; assert hasattr(e,'exl3_moe_prefill_e3_det'); assert hasattr(e,'gr_mix_v2_int8_regrid'); print('image OK', e.__file__)" 2>&1 | tail -1 | tee -a "$R/audit.log" | grep -q '^image OK' || { log "ABORT: image check"; exit 3; }
export GPU_QUEUE_NAME=r538-decode-r6
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 2700 ))
log "lock held; timebox ends $(date -Is -d @$END)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
for i in $(seq 30); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 2; done
EV=(); for kv in $ENVS; do EV+=(-e "$kv"); done
T=(sudo docker run --rm --name r538-test --gpus all --ipc=host -v "$CKPT":/models/$MODEL:ro -v "$R":/results "${EV[@]}")
log "=== gate 1: r6 kernel equality on both cards ==="
g1fail=0
for gpu in 0 1; do
  "${T[@]}" --entrypoint python3 "$IMG" /opt/decode-kernels-r6/tests/test_r6_kernels.py --device cuda:$gpu --iterations 200 \
    --json /results/kernels-gpu$gpu.json > "$R/gate1-gpu$gpu.log" 2>&1 || g1fail=1
  log "[gpu$gpu] exit $([ $g1fail = 0 ] && echo 0 || echo nonzero): $(tail -3 "$R/gate1-gpu$gpu.log" | tr '\n' ' ' | cut -c1-300)"
done
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
SEL="EXL3_GDN_BA_WARP1=1 EXL3_HC_APPLY_WARP1=1 EXL3_GR_STATE_REGRID=1"
log "=== A/B: 8 boots, A = daily env, B = + $SEL (image $IMG both arms; NVMe tier off: IMG is not the served image) ==="
rc=0; n=0
for arm in A B B A B A A B; do
  n=$((n+1)); tag="$arm$n"; extra=""; [ "$arm" = B ] && extra="$SEL"
  [ $(( END - $(date +%s) )) -lt 180 ] && { log "SKIP $tag and later boots: timebox"; break; }
  sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
  "${CLEAN_ENV[@]}" IMG="$IMG" EXTRA_ENV="$ENVS $extra" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 || { log "NO BOOT $tag"; rc=1; break; }
  for i in $(seq 150); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
  [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.Config.Image}}' flashnext)" = "$IMG" ] || { log "NO BOOT $tag"; rc=1; break; }
  has=$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -cE '^EXL3_(GDN_BA_WARP1|HC_APPLY_WARP1|GR_STATE_REGRID)=1$')
  want=0; [ "$arm" = B ] && want=3; [ "$has" = "$want" ] || { log "ABORT $tag: $has selectors in the container env, want $want"; rc=1; break; }
  a=$(greedy $tag); b=$(greedy30k $tag); log "UP $tag (selectors $has) fingerprints c1 $a / 30k $b; VRAM free $(vram)"
  [ "$a" = "$CANON1" ] && [ "$b" = "$CANON30" ] || { log "FAIL $tag: fingerprints not canonical ($CANON1 / $CANON30)"; rc=1; break; }
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-c1" --kind code --tokens 2048 --warmup-runs 1 \
    --conc 1 --runs 3 --out "$R/ab.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-200 | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-c4" --kind code --tokens 2048 --warmup-runs 1 \
    --conc 4 --runs 2 --out "$R/ab.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-200 | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-p1" --kind prose --tokens 2048 --warmup-runs 1 \
    --conc 1 --runs 2 --out "$R/ab.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-200 | tee -a "$R/audit.log"
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
for shape in ("c1","c4","p1"):
    bm={b:st.mean(runs[(b,shape)]) for b in boots if (b,shape) in runs}
    A=[v for b,v in bm.items() if b[0]=="A"]; B=[v for b,v in bm.items() if b[0]=="B"]
    print(f"{shape}: boot means " + " ".join(f"{b}={bm[b]:.1f}" for b in bm))
    if len(A)>1 and len(B)>1:
        d=st.mean(B)-st.mean(A); se=math.sqrt(st.variance(A)/len(A)+st.variance(B)/len(B)); ci=T.get(min(len(A),len(B))-1,2.0)*se
        print(f"{shape}: daily (A) {st.mean(A):.1f}, r6 selectors (B) {st.mean(B):.1f} -> B-A {d/st.mean(A)*100:+.2f} %, 95 % CI [{(d-ci)/st.mean(A)*100:+.2f}, {(d+ci)/st.mean(A)*100:+.2f}] % (n {len(A)}+{len(B)} boots)")
PY
[ $rc = 0 ] && finish DONE || finish FAILED
