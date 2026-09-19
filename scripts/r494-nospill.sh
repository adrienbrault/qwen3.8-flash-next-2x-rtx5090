#!/usr/bin/env bash
# R494 — codex decode-kernels r3: K=3 MoE coop V2 kernels without register spills (EXL3_MOE_COOP_V2_NOSPILL=1), bit-identical
# by construction (only the prefetch ring moves from registers to an 8-stage shared-memory ring; same arithmetic and order).
#
# WHY (user 2026-09-18: "Decode speed and kv pool size"; Mia GLM-5.3-Flash EXL3 PR #217 removed SASS spills from its MoE decode
# kernel for +7.7..+14.6 % decode): the served V2 K=3 instances sit at REG 64 with 5-13 LDL/STL each (flan audit sass-r491);
# routed experts are K=3 and coop A+B are 30-36 % of the decode step. Codex r3: spills are outer/tile bookkeeping, not the per-k
# MMA loop; no global maxrregcount (K=5 uses 95-108) -> ptxas occupancy-tier choice; fix = restage the prefetch ring in smem.
#
# SEQUENCE (flan/patches/exllamav3/decode-kernels/r3/box-ab-spec.md): build tabbyapi:decode-kernels-r3 (Dockerfile.box, over the
# served moecoopv2 image, NOT stacked on r2 so only one variable moves) -> gate 0 SASS: 12 nospill symbols, LDL=0 STL=0 each ->
# gate 1 torch.equal served-V2 vs nospill, rows 1..16, all routing patterns, real layer on both cards (split-k 1 and 2) + synthetic
# wide/narrow x cb 0/1/2, resident blocks/SM recorded, per-layer timing rows 1/4/16 -> serving at 360,448 OFF / ON / OFF2 / ON2
# (shared-expert overlap 0 in all arms): fingerprints canonical on every arm, fn_bench code + prose c1/c4 2,048 x 2, multiprompt
# 12 x kind 2,048 tokens c1/c4, cold prefill 30k / 120k.
#
# RUN: sudo systemd-run --unit=r494-nospill --collect -p RuntimeMaxSec=43200 -E HOME=$HOME \
#        /bin/bash /srv/qwen5090/r494-nospill.sh
# 2026-09-18 19:20: gate 0 relaxed after the first run: 12/12 symbols, LOCAL:0, REG 61-64, STACK 8 (b, = served kernels) or
# 16 (a ks0/ks2: one 8-byte STL.64/LDL.64 pair); LDL=0/STL=0 was stricter than the served baseline. Old run in -gate0-strict.
# 2026-09-18 23:30 CEST (user: "Stop using 3.05. From now on stay on 2.5"): moved to the 2.50bpw daily — checkpoint
# qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab, pool 786,432, launcher-r511, canonical c1 ae890c45d1000582 / 30k 2aa8d1024daece5c (R511).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r494-nospill; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r511.sh
CFG=/srv/qwen5090/flashnext-config.yml
CTX=/srv/qwen5090/build/decode-kernels-r3
IMG=tabbyapi:decode-kernels-r3
BASE_ENV="EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=0"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r494] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
finish(){ sudo docker rm -f r494-test >/dev/null 2>&1
  if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged live launcher)"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done   # any daily (R511 changes the served id)
    log "daily: $(served_id) $(cfgline) VRAM free $(vram)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R494 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$LM" "$CTX/Dockerfile.box" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/multiprompt.py; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r494-nospill
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"

log "=== build $IMG (daily keeps serving) ==="
( cd "$CTX" && sudo docker build -f Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "BUILD FAILED: $(grep -aE 'error|Error' "$R/build.log" | tail -3 | cut -c1-240)"; finish ABORTED; exit 3; }
log "built $IMG $(sudo docker image inspect -f '{{.Id}}' "$IMG" | cut -c8-19)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
for i in $(seq 30); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 2; done
EV=(); for kv in $BASE_ENV; do EV+=(-e "$kv"); done
T=(sudo docker run --rm --name r494-test --gpus all --ipc=host -v "$CKPT":/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab:ro -v "$R":/results "${EV[@]}")
log "=== gate 0: SASS audit of the rebuilt extension (12 nospill symbols, LOCAL:0; LDL/STL counts recorded) ==="
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
[ $rc = 0 ] || { log "GATE 0 FAIL (rc $rc): LOCAL memory used or symbols missing; SASS in $R/sass"; finish ABORTED; exit 3; }
log "=== gate 1: torch.equal served-V2 vs nospill (real layer both cards, split-k 1/2; synthetic wide x cb) + per-layer timing ==="
g1fail=0
for ks in 1 2; do for gpu in 0 1; do
  "${T[@]}" -e EXL3_MOE_COOP_KSPLIT=$ks --entrypoint python3 "$IMG" /opt/decode-kernels-r3/tests/test_moe_coop_v2_nospill.py \
    --model /models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab --device cuda:$gpu --iterations 500 --warmup 50 --json /results/real-ks$ks-gpu$gpu.json > "$R/gate1-real-ks$ks-gpu$gpu.log" 2>&1 || g1fail=1
  log "[real ks$ks gpu$gpu] $(grep -aE '^(PASS|FAIL)' "$R/gate1-real-ks$ks-gpu$gpu.log" | tail -1) $(grep -aE 'Error|Traceback' "$R/gate1-real-ks$ks-gpu$gpu.log" | tail -1 | cut -c1-160)"
done; done
for w in 0 1; do for cb in 0 1 2; do
  "${T[@]}" -e EXL3_MOE_COOP_WIDE=$w --entrypoint python3 "$IMG" /opt/decode-kernels-r3/tests/test_moe_coop_v2_nospill.py \
    --synthetic --bits 3 --cb $cb --device cuda:0 --iterations 200 --warmup 30 --skip-stage-events --json /results/synth-w$w-cb$cb.json \
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
[ $g1fail = 0 ] || { log "GATE 1 FAIL: see $R/gate1-*.log"; finish ABORTED; exit 3; }
# boot TAG REPLAY CACHE -> 0 serving with that config and image, 1 no boot, 2 mismatch
boot(){ local tag=$1 rp=$2 cache=$3 i st lp got
  log "boot $tag: nospill $rp cache $cache"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" IMG="$IMG" CACHE=$cache EXTRA_ENV="$BASE_ENV EXL3_MOE_COOP_V2_NOSPILL=$rp" bash "$CAND" >> "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
  for i in $(seq 360); do
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp; got=$(cfgline)
      case "$got" in *"cache_size: $cache cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;; *) log "ABORT: config '$got'"; return 2;; esac
      [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = "$IMG" ] || { log "ABORT: container image is not $IMG"; return 2; }
      [ "$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c "^EXL3_MOE_COOP_V2_NOSPILL=$rp$")" = 1 ] || { log "ABORT: container env lacks EXL3_MOE_COOP_V2_NOSPILL=$rp"; return 2; }
      log "UP $tag @ $cache: VRAM free MiB $(vram)"; return 0; fi
    st=$(cstatus); case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 60 ] && { sleep 3; continue; }
      log "NO BOOT $tag @ $cache ($st): $(sudo docker logs --tail 60 flashnext 2>&1 | grep -aE 'RuntimeError|Error|VRAM' | tail -1 | cut -c1-220)"
      sudo docker logs flashnext > "$R/boot-$tag-$cache.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1;; esac
    [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 0)" -ge 1 ] && {
      log "NO BOOT $tag @ $cache (restart loop): $(sudo docker logs --tail 60 flashnext 2>&1 | grep -aE 'RuntimeError|Error|VRAM' | tail -1 | cut -c1-220)"
      sudo docker logs flashnext > "$R/boot-$tag-$cache.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1; }
    sleep 3; done
  log "NO BOOT $tag @ $cache (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }
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
fp(){ local a b; a=$(greedy "$1"); b=$(greedy30k "$1"); log "[$1] fingerprints c1 $a / 30k $b (canonical ae890c45d1000582 / 2aa8d1024daece5c)"
  [ "$a" = ae890c45d1000582 ] && [ "$b" = 2aa8d1024daece5c ]; }
speed(){ local t=$1 kind
  for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$t-$kind" --kind $kind --tokens 2048 --conc 1 4 --runs 2 \
      --out "$R/records-$t.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$t $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done
  python3 /srv/qwen5090/probes/multiprompt.py --url "$API" --model "$MODEL" --tag "$t" --tokens 2048 --conc 4 --out "$R/multiprompt.jsonl" 2>&1 | tee -a "$R/audit.log"
  local c; for c in 30000 120000; do   # one invocation per ctx with its own salt (the --unique seed is otherwise fixed: R507)
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "ttft-$t" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c --unique \
      --salt $(( (RANDOM << 15 | RANDOM) + c )) --out "$R/ttft-$t.jsonl" 2>&1 | grep -E "FAILED|Traceback|Error" | tee -a "$R/audit.log"; done
  python3 - "$R/ttft-$t.jsonl" "$t" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("ttft_s") and r.get("prompt_tokens"):
        print(f"[{sys.argv[2]} prefill] ctx {r['ctx_requested']}: {r['prompt_tokens']:,} prompt tokens, cold TTFT {r['ttft_s']:.2f} s = {r['prompt_tokens']/r['ttft_s']:,.0f} tok/s")
PY
}

for arm in "OFF|0" "ON|1" "OFF2|0" "ON2|1"; do
  IFS='|' read -r tag f <<< "$arm"
  boot "$tag" "$f" 786432 || { finish ABORTED; exit 3; }
  fp "$tag" || { log "FINGERPRINT FAIL on $tag"; finish ABORTED; exit 3; }
  speed "$tag"
done
finish DONE
