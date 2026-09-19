#!/usr/bin/env bash
# R513 — E3 grouped MoE prefill, codex round 2: K = 2 / 3 / 4 (independent gate / up / down K), the served 2.50bpw pack's widths
# (38,400 K=2 / 35,328 K=3 / 1,536 K=4 expert trellis tensors; round 1 = R507 was K=3 only, about half the MoE layers).
# IMAGE tabbyapi:prefill-e3-r2 = refbase + flan/patches/exllamav3/prefill-e3/r2 (replaces r1; GNU dry-run clean 2026-09-18 23:50).
# Not bit-identical (accumulation order): c1 fingerprint must stay canonical (short prompt < 512 rows never enters E3), 30k
# recorded on ON. Codex estimate at 2048 rows: K2 1.05-1.20x, K3 1.10-1.30x, K4 1.05-1.25x per layer.
#   gate 0  checkpoint K layout (inspect_checkpoint_k_layout.py) + per-layer bench, one layer per K per card, rows 512 / 2048
#   arms    OFF / ON / OFF2 / ON2 at 786,432 (the 2.50 daily): cold prefill 30k / 60k / 120k x 2, each its own --salt;
#           decode fn_bench code + prose c1/c4 2,048 x 2 (must be unchanged)
#   quality GSM8K n=200 on OFF2 and ON2 THROUGH probes/nostop_proxy.py (R509: lm-eval's stop strings cut the reasoning;
#           R507's GSM8K ran without it), needles 131k / 240k on ON2
# RUN: sudo systemd-run --unit=r513-prefill-e3-r2 --collect -p RuntimeMaxSec=172800 -E HOME=$HOME /bin/bash /srv/qwen5090/r513-prefill-e3-r2.sh
# 2026-09-18 23:30 CEST (user: "Stop using 3.05. From now on stay on 2.5"): moved to the 2.50bpw daily — checkpoint
# qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab, pool 786,432, launcher-r511, canonical c1 ae890c45d1000582 / 30k 2aa8d1024daece5c (R511).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r513-prefill-e3-r2; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r511.sh
CFG=/srv/qwen5090/flashnext-config.yml
CTX=/srv/qwen5090/build/prefill-e3-r2
IMG=tabbyapi:prefill-e3-r2
BASE_ENV="EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r513] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
finish(){ sudo docker rm -f r513-test >/dev/null 2>&1
  if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged live launcher)"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done   # any daily (R511 changes the served id)
    log "daily: $(served_id) $(cfgline) VRAM free $(vram)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R513 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$LM" /srv/qwen5090/probes/nostop_proxy.py "$CTX/Dockerfile.box" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/multiprompt.py; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r513-prefill-e3-r2
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"

log "=== build $IMG (daily keeps serving) ==="
( cd "$CTX" && sudo docker build -f Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "BUILD FAILED: $(grep -aE 'error|Error' "$R/build.log" | tail -3 | cut -c1-240)"; finish ABORTED; exit 3; }
log "built $IMG $(sudo docker image inspect -f '{{.Id}}' "$IMG" | cut -c8-19)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
for i in $(seq 30); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 2; done
# boot TAG REPLAY CACHE -> 0 serving with that config and image, 1 no boot, 2 mismatch
boot(){ local tag=$1 e3=$2 img=$IMG cache=786432 i st lp got
  log "boot $tag: EXL3_MOE_PREFILL_E3=$e3"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" IMG="$img" CACHE=$cache EXTRA_ENV="$BASE_ENV EXL3_MOE_PREFILL_E3=$e3" bash "$CAND" >> "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
  for i in $(seq 360); do
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp; got=$(cfgline)
      case "$got" in *"cache_size: $cache cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;; *) log "ABORT: config '$got'"; return 2;; esac
      [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = "$img" ] || { log "ABORT: container image is not $img"; return 2; }
      sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -qx "EXL3_MOE_PREFILL_E3=$e3" || { log "ABORT: container env lacks EXL3_MOE_PREFILL_E3=$e3"; return 2; }
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
  [ "$a" = ae890c45d1000582 ] || return 1
  [ "${2:-}" = strict ] && { [ "$b" = 2aa8d1024daece5c ] || return 1; }
  [ "$b" = 2aa8d1024daece5c ] || log "[$1] 30k fingerprint moved (expected possible under E3: prefill arithmetic differs)"; return 0; }
speed(){ local t=$1 kind
  for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$t-$kind" --kind $kind --tokens 2048 --conc 1 4 --runs 2 \
      --out "$R/records-$t.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$t $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done
  # one invocation per (ctx, rep), each with its own --salt: the --unique filler seed is otherwise fixed, so run 2 hit the
  # prefix cache and 60k / 120k shared the shorter prompt's opening in try 1 (0.19 s "cold" TTFT)
  local c rep
  for c in 30000 60000 120000; do for rep in 1 2; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "ttft-$t" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c --unique \
      --salt $(( (RANDOM << 15 | RANDOM) + c )) --out "$R/ttft-$t.jsonl" 2>&1 | grep -E "FAILED|Traceback|Error" | tee -a "$R/audit.log"
  done; done
  python3 - "$R/ttft-$t.jsonl" "$t" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("ttft_s") and r.get("prompt_tokens"):
        print(f"[{sys.argv[2]} prefill] ctx {r['ctx_requested']}: {r['prompt_tokens']:,} prompt tokens, cold TTFT {r['ttft_s']:.2f} s = {r['prompt_tokens']/r['ttft_s']:,.0f} tok/s")
PY
}

gsm(){ local t=$1 pxp
  python3 /srv/qwen5090/probes/nostop_proxy.py --listen 127.0.0.1:8031 --upstream http://127.0.0.1:8022 >> "$R/proxy-$t.log" 2>&1 & pxp=$!; sleep 2
  timeout 10800 "$LM" --model local-chat-completions \
    --model_args "base_url=http://127.0.0.1:8031/v1/chat/completions,model=$MODEL,tokenizer=$CKPT,num_concurrent=4,max_retries=1,tokenized_requests=False" \
    --tasks gsm8k --num_fewshot 5 --limit 200 --apply_chat_template --gen_kwargs temperature=0,max_gen_toks=8192 --log_samples \
    --output_path "$R/ev-gsm8k-$t" > "$R/ev-gsm8k-$t.log" 2>&1
  kill $pxp 2>/dev/null
  python3 - "$R/ev-gsm8k-$t" "$t" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json, pathlib, sys
files = list(pathlib.Path(sys.argv[1]).rglob("results_*.json"))
if len(files) != 1: raise SystemExit(f"GSM8K: expected one result file, got {len(files)}")
s = json.loads(files[0].read_text())["results"]["gsm8k"]
print(f"[{sys.argv[2]}] GSM8K n=200 c4: flexible {s['exact_match,flexible-extract']:.3f} (SE {s['exact_match_stderr,flexible-extract']:.4f})   [R509 no-stop: 2.50 0.978]")
PY
}
log "=== gate 0a: checkpoint expert K layout ==="
sudo docker run --rm -v "$CKPT":/checkpoint:ro -v "$R":/results --entrypoint python3 "$IMG" /opt/prefill-e3-r2/tests/inspect_checkpoint_k_layout.py \
  --model /checkpoint --out /results/k-layout.json --strict-served-pack 2>&1 | tail -15 | cut -c1-220 | tee -a "$R/audit.log"
log "=== gate 0: real-weight per-layer microbenchmark, one layer per K per card, rows 512 / 2048 ==="
sudo docker run --rm --name r513-test --gpus all --ipc=host -v "$CKPT":/checkpoint:ro -v "$R":/results -e EXL3_MOE_PREFILL_E3=1 \
  --entrypoint python3 "$IMG" /opt/prefill-e3-r2/tests/bench_prefill_e3_layer.py --model /checkpoint --use-per-device 30,30 \
  --rows 512,2048 --warmup 5 --iterations 20 --out /results/prefill-e3-layer.json > "$R/gate0.log" 2>&1; rc=$?
tail -20 "$R/gate0.log" | cut -c1-240 | tee -a "$R/audit.log"
[ $rc = 0 ] || { log "GATE 0 FAIL (bench rc $rc)"; finish ABORTED; exit 3; }
python3 - "$R/prefill-e3-layer.json" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json, sys, statistics
d = json.load(open(sys.argv[1]))
def walk(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from walk(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from walk(v, f"{path}[{i}]")
    else: yield path, x
flat = dict(walk(d))
for k, v in flat.items():
    if any(w in k.lower() for w in ("median", "max_abs", "nrmse", "device", "rows")) and not isinstance(v, list): print(f"  {k} = {v}")
PY
log "gate 0: bench completed; the GO check (2048 faster on both cards, 512 <= +3 %) is read from the lines above and prefill-e3-layer.json"
for arm in "OFF|0" "ON|1" "OFF2|0" "ON2|1"; do
  tag=${arm%%|*}; e3=${arm#*|}
  boot "$tag" "$e3" || { finish ABORTED; exit 3; }
  if [ "$e3" = 0 ]; then fp "$tag" strict || { log "[$tag] FINGERPRINT NOT CANONICAL (OFF arm)"; finish ABORTED; exit 3; }
  else fp "$tag" || { log "[$tag] c1 FINGERPRINT NOT CANONICAL (decode-only prompt must not change)"; finish ABORTED; exit 3; }; fi
  sudo docker logs flashnext 2>&1 | grep -aiE "e3|prefill_e3" | tail -3 | cut -c1-200 > "$R/e3-log-$tag.txt"
  speed "$tag"
  [ "$tag" = OFF2 ] && gsm OFF2
  if [ "$tag" = ON2 ]; then gsm ON2
    python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$MODEL" --tag "needle" --ctx-tokens 131072 240000 \
      --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee -a "$R/audit.log"; fi
done
finish DONE
