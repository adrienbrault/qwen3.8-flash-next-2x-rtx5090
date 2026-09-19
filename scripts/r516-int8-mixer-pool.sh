#!/usr/bin/env bash
# R516 — int8 V2 mixer weights (codex decode r4 group M, EXL3_HC_MIX_V2_INT8=1) as a POOL lever on the 2.50bpw daily (user
# 2026-09-19: "the free vram sounds great if there no perf con"). R499: +218 MiB free on cuda:0 / +258 on cuda:1 (0.47 GiB),
# decode code c1 0 %, prose c1 -1.4 %, multiprompt c4 flat, synthetic c4 -1.5..-2.9 % (partly early EOS: ON outputs ended at a
# median 1,954 of 2,048 tokens); numerics change (fingerprints e7fb377c / 4a255910). This unit answers the two open questions:
#   1. how much pool the freed VRAM buys: ON ladder 835,584 / 819,200 / 802,816 (daily 786,432 is R495b's largest OFF boot),
#      each candidate must boot AND survive a c4 stress pass with RestartCount 0;
#   2. whether quality holds: GSM8K n=500 through the no-stop proxy on OFF and ONBIG, PAIRED per question (R509 2.50: 0.978);
#      needles 131k / 240k 5/5 on ONBIG; tool-eval 69 x 4 on ONBIG2 (R511 daily 86.0 +- 2.6, gate 82).
#   ARMS OFF@786,432 / ONBIG / OFF2@786,432 / ONBIG2: fn_bench code + prose c1/c4 2,048 x 2, multiprompt 12 x kind c4.
# Isolated from group I: launcher-r511 with IMG=tabbyapi:decode-kernels-r4 (r4 flags default-off = r2 bytes) and the R511
# EXTRA_ENV, + EXL3_HC_MIX_V2_INT8=1 on ON arms (exactly R499's group M arms). Promotion is a separate step (stacked with R514).
# RUN: sudo systemd-run --unit=r516-int8-mixer-pool --collect -p RuntimeMaxSec=86400 -E HOME=$HOME /bin/bash /srv/qwen5090/r516-int8-mixer-pool.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r516-int8-mixer-pool; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r511.sh
CFG=/srv/qwen5090/flashnext-config.yml
IMG=tabbyapi:decode-kernels-r4
BASE_ENV="EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r516] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
finish(){   if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged live launcher)"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done   # any daily (R511 changes the served id)
    log "daily: $(served_id) $(cfgline) VRAM free $(vram)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R516 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$LM" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/multiprompt.py; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r516-int8-mixer-pool
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"

sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: $IMG missing (built by R499)"; finish ABORTED; exit 3; }
log "image $IMG $(sudo docker image inspect -f '{{.Id}}' "$IMG" | cut -c8-19)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
for i in $(seq 30); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 2; done
# boot TAG REPLAY CACHE -> 0 serving with that config and image, 1 no boot, 2 mismatch
boot(){ local tag=$1 rp=$2 cache=$3 i st lp got
  log "boot $tag: int8 mixer $rp cache $cache"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" IMG="$IMG" CACHE=$cache EXTRA_ENV="$BASE_ENV EXL3_HC_MIX_V2_INT8=$rp" bash "$CAND" >> "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
  for i in $(seq 360); do
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp; got=$(cfgline)
      case "$got" in *"cache_size: $cache cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;; *) log "ABORT: config '$got'"; return 2;; esac
      [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = "$IMG" ] || { log "ABORT: container image is not $IMG"; return 2; }
      [ "$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c "^EXL3_HC_MIX_V2_INT8=$rp$")" = 1 ] || { log "ABORT: container env lacks EXL3_HC_MIX_V2_INT8=$rp"; return 2; }
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
}

alive(){ [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
stress(){ local kind; for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "stress-$1-$kind" --kind $kind --tokens 2048 --conc 4 --runs 2 \
      --out "$R/stress-$1.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[stress $1 $kind]/" | cut -c1-200 | tee -a "$R/audit.log"; done
  alive; }
gsm(){ local t=$1 pxp
  python3 /srv/qwen5090/probes/nostop_proxy.py --listen 127.0.0.1:8031 --upstream http://127.0.0.1:8022 >> "$R/proxy-$t.log" 2>&1 & pxp=$!; sleep 2
  timeout 10800 "$LM" --model local-chat-completions \
    --model_args "base_url=http://127.0.0.1:8031/v1/chat/completions,model=$MODEL,tokenizer=$CKPT,num_concurrent=4,max_retries=1,tokenized_requests=False" \
    --tasks gsm8k --num_fewshot 5 --limit 500 --apply_chat_template --gen_kwargs temperature=0,max_gen_toks=8192 --log_samples \
    --output_path "$R/ev-gsm8k-$t" > "$R/ev-gsm8k-$t.log" 2>&1
  kill $pxp 2>/dev/null
  python3 - "$R/ev-gsm8k-$t" "$t" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json, pathlib, sys
files = list(pathlib.Path(sys.argv[1]).rglob("results_*.json"))
if len(files) != 1: raise SystemExit(f"GSM8K: expected one result file, got {len(files)}")
s = json.loads(files[0].read_text())["results"]["gsm8k"]
print(f"[{sys.argv[2]}] GSM8K n=500: flexible {s['exact_match,flexible-extract']:.3f} (SE {s['exact_match_stderr,flexible-extract']:.4f})   [R509 no-stop: 2.50 0.978]")
PY
}
gsmpair(){ python3 - "$R/ev-gsm8k-$1" "$R/ev-gsm8k-$2" "$1" "$2" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json, pathlib, sys
def load(d):
    out = {}
    for f in pathlib.Path(d).rglob("samples_gsm8k_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("filter", "flexible-extract") != "flexible-extract": continue
            out[r["doc_id"]] = bool(r.get("exact_match", 0))
    return out
a, b = load(sys.argv[1]), load(sys.argv[2]); ids = sorted(set(a) & set(b))
ab = sum(1 for i in ids if a[i] and not b[i]); ba = sum(1 for i in ids if b[i] and not a[i])
print(f"GSM8K paired n={len(ids)}: {sys.argv[3]} {sum(a[i] for i in ids)} vs {sys.argv[4]} {sum(b[i] for i in ids)}; "
      f"only-{sys.argv[3]} {ab}, only-{sys.argv[4]} {ba}" + (f", z {(ab-ba)/((ab+ba)**0.5):.2f}" if ab + ba else ""))
PY
}

log "=== ladder: largest int8 pool above 786,432 that boots and survives c4 stress ==="
BIG=0
for c in 835584 819200 802816; do boot "PROBE$c" 1 $c; rc=$?; [ $rc = 2 ] && { finish ABORTED; exit 3; }
  [ $rc = 0 ] || continue
  if stress "P$c"; then BIG=$c; log "int8 pool $c boots and survives c4 stress (VRAM free MiB $(vram))"; break
  else log "int8 pool $c boots but does NOT survive c4 stress: $(sudo docker logs --tail 80 flashnext 2>&1 | grep -aE 'out of memory|Error' | tail -1 | cut -c1-200)"
    sudo docker logs flashnext > "$R/stress-$c.docker.log" 2>&1; fi; done
if [ "$BIG" = 0 ]; then log "no int8 pool above 786,432 survives; running the arms at 786,432 (headroom only)"; BIG=786432; fi
log "int8 pool: $BIG (daily 786,432, +$(( (BIG - 786432) * 100 / 786432 )) %)"

for arm in "OFF|0|786432" "ONBIG|1|$BIG" "OFF2|0|786432" "ONBIG2|1|$BIG"; do
  IFS='|' read -r tag rp cache <<< "$arm"
  boot "$tag" "$rp" "$cache" || { finish ABORTED; exit 3; }
  if [ "$rp" = 0 ]; then fp "$tag" || { log "[$tag] OFF FINGERPRINT NOT CANONICAL"; finish ABORTED; exit 3; }
  else fp "$tag"; log "[$tag] fingerprint change expected (int8; R499 e7fb377c987d685c / 4a255910dee2d9c5)"; fi
  speed "$tag"
  alive || { log "[$tag] server not alive after the speed runs"; finish ABORTED; exit 3; }
  case $tag in
    OFF) gsm OFF;;
    ONBIG) gsm ONBIG; gsmpair OFF ONBIG
      log "=== needles 131k / 240k on ONBIG ==="
      python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$MODEL" --tag needle-ONBIG --ctx-tokens 131072 240000 \
        --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee -a "$R/audit.log";;
    ONBIG2) log "=== tool-eval 69 x 4 on ONBIG2 ==="
      ( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$MODEL" --temperature 0.6 --top-p 0.95 --top-k 20 \
          --trials 4 --parallel 8 --json-file "$R/tooleval-ONBIG2.json" > "$R/tooleval-ONBIG2.log" 2>&1 )
      python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval-ONBIG2.json" ONBIG2 2>&1 | tee -a "$R/audit.log"
      log "tool-eval mean: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trial_statistics"]["final_score_mean"])' "$R/tooleval-ONBIG2.json" 2>/dev/null || echo unparsed) (gate 82; R511 86.0)";;
  esac
done
finish DONE
