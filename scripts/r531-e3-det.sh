#!/usr/bin/env bash
# R531 — deterministic E3 grouped MoE prefill (Opus round e3-det r1, patches/exllamav3/e3-det/r1): EXL3_MOE_PREFILL_E3_DET=1
# gives every (token, expert) assignment its own slot (fat experts: E3 down kernel stores instead of atomics; thin experts: the
# fused kernel's slot mode) and one fixed-order reduction in router top-k order. R524 try 4 proved atomic E3 is the daily's
# non-determinism (cold 8k reps diverge at token 128). Measurement unit, no promotion: image tabbyapi:e3-det-r1 on
# stack-r4-e3r2; the pruned-draft flags are stripped from the live env so B0 (atomic E3, stack-r4-e3r2) and the DET boots
# differ only by the DET flag. Gates per box-ab-spec.md: G0 build; G1 kernel determinism (20 reps, side-stream GEMM, NRMSE vs
# atomic < 1e-5); G2 cold 8k/30k x3 identical; G3 cross-process; G4 c1 = live canonical on two fresh boots, 30k equal across
# the boots; G5 salted cold prefill >= 0.97 x B0 at 60k and 120k; G5b free VRAM within 50 MiB of B0; G6 decode within +-2 %.
# GPU TIMEBOX 20 min. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r531-e3-det; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/e3-det-r1
IMG=tabbyapi:e3-det-r1
BIMG=tabbyapi:stack-r4-e3r2
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SALT=$(( $(date +%s) % 100000 ))
BOOTED=0
log(){ echo "$(date -Is) [r531] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){ sudo docker rm -f r531-k >/dev/null 2>&1 || true
  if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R531 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$SRC/Dockerfile.box" "$SRC/tests/gpu_e3_det.py" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
ENVS=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE"); [ -n "$ENVS" ] || { log "ABORT: no daily EXTRA_ENV"; exit 3; }
ENVS=$(echo " $ENVS " | sed -e 's/ EXL3_MTP_DEVICE_DRAFT=1 / /; s/ EXL3_EMBED_GPU=1 / /; s/ EXL3_EMBED_GPU_PRUNED=1 / /' | xargs)
case " $ENVS " in *" EXL3_MOE_PREFILL_E3=1 "*) ;; *) log "ABORT: live env has no E3"; exit 3;; esac
C1=e7fb377c987d685c
log "G0: building $IMG (before the lock); env: $ENVS"
(cd "$SRC" && sudo docker build --build-arg BASE=$BIMG -f Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "G0 FAIL (NO-GO): build: $(grep -aiE 'error' "$R/build.log" | head -3 | tr '\n' ' ' | cut -c1-300)"; exit 3; }
log "G0 PASS: $(sudo docker image inspect -f '{{.Id}}' "$IMG" | cut -c1-19)"
export GPU_QUEUE_NAME=r531-e3-det
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 1200 ))
log "lock held; timebox ends $(date -Is -d @$END)"
BOOTED=1
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
boot(){ local tag=$1 img=$2 extra=$3 i
  sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
  "${CLEAN_ENV[@]}" IMG="$img" EXTRA_ENV="$ENVS $extra" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 || { log "NO BOOT $tag"; return 1; }
  for i in $(seq 150); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
  [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.Config.Image}}' flashnext)" = "$img" ] || { log "NO BOOT $tag"; return 1; }
  local has; has=$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c '^EXL3_MOE_PREFILL_E3_DET=1$')
  log "UP $tag ($img, DET env lines $has); VRAM free MiB $(vram)"; }
prefill(){ local tag=$1 c k
  for c in 60000 120000; do for k in 1 2; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-pf$c-$k" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c \
      --unique --salt $(( SALT + c/1000 + k*7 + ${#tag}*1000 )) --out "$R/prefill.jsonl" 2>&1 | grep -E "FAILED|Traceback" | tee -a "$R/audit.log"
    [ $c = 120000 ] && [ $k = 1 ] && log "[$tag] VRAM free during/after 120k: $(vram)"; done; done; }
rc=0
# B0: atomic E3 baseline on the base image
boot B0 "$BIMG" "" || { finish ABORTED; exit 3; }
log "[B0] fingerprints c1 $(greedy B0) / 30k $(greedy30k B0)"
prefill B0
sudo docker rm -f flashnext >/dev/null 2>&1; sleep 3
# K1 + E1 and K2 (in-process, DET flipped by the script)
DENV=(); for kv in $ENVS; do DENV+=(-e "$kv"); done
RUN=(sudo docker run --rm --name r531-k --gpus all --ipc=host --shm-size=16g -v /srv/qwen5090/models:/models:ro -v /srv/qwen5090/.exl3cache:/exl3-cache
  -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache -e EXL3_AUTOSPLIT_MARGIN_MB=2048 "${DENV[@]}" -v "$R":/out --entrypoint python3 "$IMG"
  /opt/e3-det-r1/tests/gpu_e3_det.py --model /models/$MODEL)
# EXL3_AUTOSPLIT_MARGIN_MB=2048: the loader packs cuda:0 to one transient + 256 MiB whatever the split (R524 tip
# subagent, model_ls.py _load_autosplit), and the det slot buffer (A x 2560 fp32, 200 MiB at 2048 rows) OOMed at
# ~128 MiB free (R531 try 2); placement only. The served boots below keep the real config.
log "K1+E1: kernel determinism + cold 8k/30k reps"
timeout 420 "${RUN[@]}" --kernel --e2e --out /out/gpu_e3_det_p1.json > "$R/gpu_e3_det_p1.log" 2>&1; k1=$?
tail -25 "$R/gpu_e3_det_p1.log" | cut -c1-240 | tee -a "$R/audit.log"
[ $k1 = 0 ] || { log "K1/E1 FAIL (rc $k1): NO-GO"; rc=1; }
if [ $rc = 0 ]; then
  log "K2: cross-process"
  timeout 240 "${RUN[@]}" --kernel --repeats 3 --iters 5 --warmup 2 --compare /out/gpu_e3_det_p1.json --out /out/gpu_e3_det_p2.json > "$R/gpu_e3_det_p2.log" 2>&1; k2=$?
  tail -10 "$R/gpu_e3_det_p2.log" | cut -c1-240 | tee -a "$R/audit.log"
  [ $k2 = 0 ] || { log "K2 FAIL (rc $k2)"; rc=1; }
fi
if [ $rc = 0 ]; then
  boot A "$IMG" "EXL3_MOE_PREFILL_E3_DET=1" || { finish ABORTED; exit 3; }
  a1=$(greedy A); a30=$(greedy30k A); log "[A] fingerprints c1 $a1 / 30k $a30"
  boot B "$IMG" "EXL3_MOE_PREFILL_E3_DET=1" || { finish ABORTED; exit 3; }
  b1=$(greedy B); b30=$(greedy30k B); log "[B] fingerprints c1 $b1 / 30k $b30"
  [ "$a1" = "$C1" ] && [ "$b1" = "$C1" ] && [ "$a30" = "$b30" ] && [ "$a30" != none ] && log "G4 PASS (DET 30k canonical $a30)" || { log "G4 FAIL"; rc=1; }
  prefill B
  for c in 1 4; do python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "B-c$c" --kind code --tokens 2048 --warmup-runs 1 \
    --conc $c --runs 2 --out "$R/records.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[B]/" | cut -c1-200 | tee -a "$R/audit.log"; done
  sudo docker logs flashnext > "$R/docker-B.log" 2>&1
  python3 - "$R/prefill.jsonl" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("ttft_s") and r.get("prompt_tokens"): d[(r["tag"].split("-pf")[0], r["ctx_requested"])].append(r["prompt_tokens"]/r["ttft_s"])
for c in (60000,120000):
    b0=d.get(("B0",c)); b=d.get(("B",c))
    if b0 and b: print(f"G5 prefill {c}: B0 atomic {st.mean(b0):.0f} {[round(x) for x in b0]} / B DET {st.mean(b):.0f} {[round(x) for x in b]} -> {st.mean(b)/st.mean(b0):.3f}x ({'PASS' if st.mean(b)>=0.97*st.mean(b0) else 'FAIL'})")
PY
fi
[ $rc = 0 ] && finish DONE || finish FAILED
