#!/usr/bin/env bash
# R496 — codex gdn-state r3: GDN state replay (EXL3_GDN_STATE_REPLAY=1) re-gated with a deterministic harness and a serving
# placement lock. R489 gate 2 failed OFF vs ON at token 87, but R489b showed OFF vs OFF also differs (token 14): codex r3
# traced the nondeterminism to fresh-container timing-selected kernels (coop GEMM autotune without a frozen tune cache, plus
# the HGEMM fp16-accumulator probe), and the placement change to replay freeing memory the layer-split loader then used
# (GDN layer 26 moved cards). r3 adds EXL3_COOP_AUTOTUNE_REQUIRE_CACHE=1 (any tune-cache miss is fatal), a harness contract
# (frozen cache digest, PYTHONHASHSEED, EXL3_HGEMM_F16ACC, CUBLAS_WORKSPACE_CONFIG, full placement signature), and
# EXL3_GDN_STATE_REPLAY_PLACEMENT_LOCK=1 (load with the OFF footprint, release the padding after placement), plus an
# "EXL3 layer placement: {json}" boot line. Records: flan/patches/exllamav3/gdnstate/r3 (build-and-ab.md = codex runbook).
# WHY (user 2026-09-18: "Decode speed and kv pool size"): ~1.3 GB of fp32 history copies -> ~+83k tokens of pool.
#
# SEQUENCE (every gate aborts on failure; the daily is restored at the end):
#   build   tabbyapi:gdn-state-r3 FROM tabbyapi:gdn-state-r1 (Dockerfile.box, JIT rebuild for the one .cu change)
#   gate A  r1 kernel identity (PASS: 2485), r2 compact-batch shapes, r3 placement-lock unit test
#   gate B  harness OFF vs OFF, frozen tune cache (primed once in a discarded run), strict cache gate: 512/512 equal
#   gate C  harness OFF vs ON with the placement lock: 512/512 equal and identical placement
#   gate D  serving 360,448 OFF vs ON locked: identical "EXL3 layer placement" (text component) + lock-release line, BOTH
#           canonical (1474eee2f5945248 / 4a255910dee2d9c5)
#   ladder  ON unlocked at 458,752 / 425,984 / 409,600 / 393,216; measure the winner + a paired OFF control (as R489)
#
# RUN: sudo systemd-run --unit=r496-gdn-state-r3 --collect -p RuntimeMaxSec=86400 -E HOME=$HOME \
#        /bin/bash /srv/qwen5090/r496-gdn-state-r3.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r496-gdn-state-r3; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r484.sh
CFG=/srv/qwen5090/flashnext-config.yml
CTX=/srv/qwen5090/build/gdnstate-r3
IMG=tabbyapi:gdn-state-r3
BASE_ENV="EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r496] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
finish(){ sudo docker rm -f r496-test >/dev/null 2>&1
  if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged live launcher)"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
    log "daily: $(served_id) $(cfgline) VRAM free $(vram)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R496 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$LM" "$CTX/Dockerfile.box" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/multiprompt.py; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r496-gdn-state-r3
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"

log "=== build $IMG (daily keeps serving) ==="
( cd "$CTX" && sudo docker build -f Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "BUILD FAILED: $(grep -aE 'error|Error' "$R/build.log" | tail -3 | cut -c1-240)"; finish ABORTED; exit 3; }
log "built $IMG $(sudo docker image inspect -f '{{.Id}}' "$IMG" | cut -c8-19); $(grep -a 'exllamav3_ext' "$R/build.log" | tail -1 | cut -c1-160)"

BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
for i in $(seq 30); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 2; done

HX=(-e PYTHONHASHSEED=0 -e EXL3_HGEMM_F16ACC=1 -e CUBLAS_WORKSPACE_CONFIG=:4096:8)
EV=(); for kv in $BASE_ENV; do EV+=(-e "$kv"); done
H="python3 /opt/gdn-state-r3/tests/model_decode_ids.py --model /models --cache-size 8192 --max-batch-size 4 --max-history 3 --chunk-size 2048 --target-use 30,30 --draft-use 30,30 --vision-use 30,30 --target-cache-mode 8,8 --draft-cache-mode FP16 --vision --scenario single --tokens 512 --seed 0"
mkdir -p "$R/g" "$R/g/tune"; sudo chmod -R 777 "$R/g"

log "=== gate A: kernel identity (r1), compact-batch shapes (r2), placement lock (r3) ==="
sudo docker run --rm --name r496-test --gpus all --ipc=host --entrypoint bash "$IMG" -lc \
  'python3 /opt/gdn-state-r3/tests/test_gdn_replay_kernel.py --device cuda:0 && python3 /opt/gdn-state-r3/tests/test_gdn_replay_model_shapes.py --device cuda:0 && python3 /opt/gdn-state-r3/tests/test_placement_lock.py' \
  > "$R/gateA.log" 2>&1; rc=$?
grep -aE "^PASS|^FAIL|Error" "$R/gateA.log" | tail -6 | cut -c1-200 | tee -a "$R/audit.log"
[ $rc = 0 ] && grep -q "^PASS: 2485 torch.equal checks" "$R/gateA.log" || { log "GATE A FAIL (rc $rc)"; finish ABORTED; exit 3; }

log "=== gate B: harness determinism OFF vs OFF with a frozen tune cache ==="
sudo docker run --rm --name r496-test --gpus all --ipc=host --entrypoint bash -v "$CKPT":/models:ro -v "$R/g":/results -v "$R/g/tune":/tune \
  "${EV[@]}" "${HX[@]}" -e EXLLAMAV3_TUNE_CACHE=/tune/coop-autotune.bin -e EXL3_COOP_AUTOTUNE_REQUIRE_CACHE=0 \
  -e EXL3_GDN_STATE_REPLAY=0 -e EXL3_GDN_STATE_REPLAY_PLACEMENT_LOCK=0 "$IMG" \
  -lc "$H --allow-unfrozen-runtime --output /results/discarded-tune-prime.json" > "$R/gateB-prime.log" 2>&1 \
  || { log "GATE B prime failed: $(tail -2 "$R/gateB-prime.log" | tr '\n' ' ' | cut -c1-240)"; finish ABORTED; exit 3; }
[ -s "$R/g/tune/coop-autotune.bin" ] || { log "GATE B: priming wrote no tune cache"; finish ABORTED; exit 3; }
TUNE_SHA=$(sha256sum "$R/g/tune/coop-autotune.bin" | awk '{print $1}'); sudo chmod a-w "$R/g/tune/coop-autotune.bin"
log "frozen tune cache $TUNE_SHA"
harness(){ local out=$1 rp=$2 lock=$3
  sudo docker run --rm --name r496-test --gpus all --ipc=host --entrypoint bash -v "$CKPT":/models:ro -v "$R/g":/results \
    -v "$R/g/tune/coop-autotune.bin":/frozen/coop-autotune.bin:ro "${EV[@]}" "${HX[@]}" \
    -e EXLLAMAV3_TUNE_CACHE=/frozen/coop-autotune.bin -e EXL3_COOP_AUTOTUNE_REQUIRE_CACHE=1 \
    -e EXL3_GDN_STATE_REPLAY=$rp -e EXL3_GDN_STATE_REPLAY_PLACEMENT_LOCK=$lock "$IMG" \
    -lc "$H --expected-tune-cache-sha256 $TUNE_SHA --output /results/$out.json" > "$R/g/$out.log" 2>&1; }
cmpids(){ python3 "$CTX/tests/compare_model_ids.py" --label "$1" "$R/g/$2.json" "$R/g/$3.json" > "$R/g/cmp-$1.log" 2>&1; local rc=$?
  log "compare $1: rc $rc $(tail -3 "$R/g/cmp-$1.log" | tr '\n' ' ' | cut -c1-260)"; return $rc; }
harness model-single-off-a 0 0 || { log "GATE B: OFF-a exited non-zero: $(tail -2 "$R/g/model-single-off-a.log" | tr '\n' ' ' | cut -c1-240)"; finish ABORTED; exit 3; }
harness model-single-off-b 0 0 || { log "GATE B: OFF-b exited non-zero: $(tail -2 "$R/g/model-single-off-b.log" | tr '\n' ' ' | cut -c1-240)"; finish ABORTED; exit 3; }
cmpids OFF-vs-OFF model-single-off-a model-single-off-b || { log "GATE B FAIL: the harness is still nondeterministic — OFF/ON would prove nothing"; finish ABORTED; exit 3; }

log "=== gate C: harness OFF vs locked ON ==="
harness model-single-on-locked 1 1 || { log "GATE C: ON exited non-zero: $(tail -2 "$R/g/model-single-on-locked.log" | tr '\n' ' ' | cut -c1-240)"; finish ABORTED; exit 3; }
cmpids OFF-vs-ON-locked model-single-off-a model-single-on-locked || { log "GATE C FAIL: replay changes token ids with placement locked"; finish ABORTED; exit 3; }

# boot TAG REPLAY CACHE -> 0 serving with that config and image, 1 no boot, 2 mismatch
boot(){ local tag=$1 rp=$2 cache=$3 i st lp got
  log "boot $tag: replay $rp lock ${LOCK:-0} cache $cache"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" IMG="$IMG" CACHE=$cache EXTRA_ENV="$BASE_ENV EXL3_GDN_STATE_REPLAY=$rp EXL3_GDN_STATE_REPLAY_PLACEMENT_LOCK=${LOCK:-0}" bash "$CAND" >> "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
  for i in $(seq 360); do
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp; got=$(cfgline)
      case "$got" in *"cache_size: $cache cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;; *) log "ABORT: config '$got'"; return 2;; esac
      [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = "$IMG" ] || { log "ABORT: container image is not $IMG"; return 2; }
      [ "$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c "^EXL3_GDN_STATE_REPLAY=$rp$")" = 1 ] || { log "ABORT: container env lacks EXL3_GDN_STATE_REPLAY=$rp"; return 2; }
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
fp(){ local a b; a=$(greedy "$1"); b=$(greedy30k "$1"); log "[$1] fingerprints c1 $a / 30k $b (canonical 1474eee2f5945248 / 4a255910dee2d9c5)"
  [ "$a" = 1474eee2f5945248 ] && [ "$b" = 4a255910dee2d9c5 ]; }
speed(){ local t=$1 kind
  for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$t-$kind" --kind $kind --tokens 2048 --conc 1 4 --runs 2 \
      --out "$R/records-$t.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$t $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done
  python3 /srv/qwen5090/probes/multiprompt.py --url "$API" --model "$MODEL" --tag "$t" --tokens 1024 --conc 1 4 --out "$R/multiprompt.jsonl" 2>&1 | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "ttft-$t" --kind prose --tokens 64 --conc 1 --runs 1 --ctx 30000 120000 --unique \
    --out "$R/ttft-$t.jsonl" 2>&1 | grep -E "FAILED|Traceback|Error" | tee -a "$R/audit.log"
  python3 - "$R/ttft-$t.jsonl" "$t" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("ttft_s") and r.get("prompt_tokens"):
        print(f"[{sys.argv[2]} prefill] ctx {r['ctx_requested']}: {r['prompt_tokens']:,} prompt tokens, cold TTFT {r['ttft_s']:.2f} s = {r['prompt_tokens']/r['ttft_s']:,.0f} tok/s")
PY
}

placement(){ sudo docker logs flashnext 2>&1 | python3 -c '
import json,sys
for l in sys.stdin:
    i=l.find("EXL3 layer placement: ")
    if i<0: continue
    d=json.loads(l[i+len("EXL3 layer placement: "):])
    if d.get("component")=="text": print(json.dumps(d.get("modules"),sort_keys=True))
' | tail -1 > "$R/placement-$1.json"; log "[$1] placement: $(python3 -c 'import json,sys,collections; m=json.load(open(sys.argv[1])); print(len(m), "modules", dict(collections.Counter(x.get("device") for x in m)))' "$R/placement-$1.json" 2>&1 | cut -c1-200)"; }
log "=== gate D: serving at 360,448, OFF then ON with the placement lock ==="
LOCK=0; boot OFF 0 360448 || { finish ABORTED; exit 3; }
placement OFF
fp OFF || { log "GATE D FAIL (OFF on the new image is not canonical)"; finish ABORTED; exit 3; }
LOCK=1; boot ON360L 1 360448 || { finish ABORTED; exit 3; }
placement ON360L
sudo docker logs flashnext 2>&1 | grep -a "EXL3 GDN replay placement lock released:" | tail -1 | cut -c1-240 | tee -a "$R/audit.log" | grep -q . \
  || { log "GATE D FAIL (no placement-lock release line)"; finish ABORTED; exit 3; }
[ -s "$R/placement-OFF.json" ] && cmp -s "$R/placement-OFF.json" "$R/placement-ON360L.json" || { log "GATE D FAIL (placement differs OFF vs locked ON, or no placement line)"; finish ABORTED; exit 3; }
fp ON360L || { log "GATE D FAIL (replay changes served greedy output at identical placement)"; finish ABORTED; exit 3; }
LOCK=0

log "=== ladder ==="
POOL=0
for c in 458752 425984 409600 393216; do boot ON 1 $c; rc=$?; [ $rc = 2 ] && { finish ABORTED; exit 3; }; [ $rc = 0 ] && { POOL=$c; break; }; done
log "replay ON: largest pool that serves = $POOL"
[ "$POOL" = 0 ] && { log "no pool above 360,448 boots with replay"; POOL=360448; boot ON 1 360448 || { finish ABORTED; exit 3; }; }
placement "ON$POOL"
fp "ON$POOL" || log "NOTE: ON@$POOL fingerprint differs (placement; identity was proven at 360,448)"
log "=== measure ON@$POOL: VRAM free $(vram) ==="
speed "ON$POOL"
python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "c4x60k" --kind code --tokens 256 --conc 4 --runs 1 --ctx 60000 --unique \
  --out "$R/c4x60k.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | cut -c1-260 | tee -a "$R/audit.log"
log "c4x60k: $(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get("completion_tokens",0)>=256))' "$R/c4x60k.jsonl" 2>/dev/null)/4; engine $(served_id); VRAM free $(vram)"
python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$MODEL" --tag "needle" --ctx-tokens 131072 240000 \
  --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee -a "$R/audit.log"
timeout 10800 "$LM" --model local-chat-completions \
  --model_args "base_url=$API/chat/completions,model=$MODEL,tokenizer=$CKPT,num_concurrent=4,max_retries=1,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 5 --limit 200 --apply_chat_template --gen_kwargs temperature=0,max_gen_toks=8192 --output_path "$R/ev-gsm8k" > "$R/ev-gsm8k.log" 2>&1
python3 - "$R/ev-gsm8k" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json, pathlib, sys
files = list(pathlib.Path(sys.argv[1]).rglob("results_*.json"))
if len(files) != 1: raise SystemExit(f"GSM8K: expected one result file, got {len(files)}")
d = json.loads(files[0].read_text()); s = d["results"]["gsm8k"]
print(f"GSM8K n=200 c4: flexible {s['exact_match,flexible-extract']:.3f} (SE {s['exact_match_stderr,flexible-extract']:.4f})   [served 0.925]")
PY
log "=== measure OFF@360,448 (paired control) ==="
boot OFF2 0 360448 || { finish ABORTED; exit 3; }
speed OFF2
finish DONE
