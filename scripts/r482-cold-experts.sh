#!/usr/bin/env bash
# R482 — cold-expert offload on the exllamav3 daily: N routed experts per MoE layer on the CPU, hot/cold swapping on.
#
# WHY (user 2026-09-18, "yes try this lol", after "what I'm describing is cold expert offloading to cpu"): R480 showed a
# whole CPU-offloaded MoE layer buys 425,984 tokens but costs 20-25 % decode, because every token runs every layer. The
# engine we serve already splits EXPERTS instead: TabbyAPI cpu_moe_split_experts N -> exllamav3 moe_cpu_split keeps N of
# each layer's 512 routed experts in a CPU worker, overlapped with the GPU experts, and (EXL3_MOE_CPU_SWAP=1, the default)
# every 128 decode steps swaps the hottest CPU expert with the coldest GPU one. One routed expert is ~1.7 MiB of VRAM, so
# N=16 over 48 layers frees ~1.2 GiB (~+80k tokens at 8,8), N=32 ~2.4 GiB.
#
# RISK = HOST RAM. flan has 60 GiB; the 30 GiB PLE table lives in the page cache and is gathered every step; k3s, Home
# Assistant and storage share the box. In dynamic mode the worker may hold every expert of a split layer (~0.85 GiB per
# layer), not only the tail. So: (1) a PROBE arm splits only the first 8 layers at N=8 and measures the anon-memory growth;
# (2) the per-layer host cost decides how many layers the real arms may split within a 10 GiB anon budget; (3) a watchdog
# removes the engine if MemAvailable falls under 26 GiB at any time. Each real arm: 4 slots, 8,8, pool ladder 557,056 ->
# 360,448, warm-up pass (lets the swapper converge), greedy fingerprints (a CPU expert changes numerics, so a divergence is
# expected), code + prose c1/c4 x2, cold prefill (ONE request per context: the R480 instrument repeated the prompt),
# needle x5 at 131k and 240k, GSM8K n=200 c4. Escalation N = 8 -> 16 -> 32 stops once code c1 falls below 0.75x the
# served 215. The served daily (4 slots, 360,448, 8,8, no split) is restored at the end.
#
# RUN: sudo systemd-run --unit=r482-cold-experts --collect -p RuntimeMaxSec=43200 -E HOME=$HOME \
#        /bin/bash /srv/qwen5090/r482-cold-experts.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r482-cold-experts; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r482.sh
CFG=/srv/qwen5090/flashnext-config.yml
DAILY_ENV="EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
RAM_FLOOR_KB=$((26*1024*1024))
ANON_BUDGET_MIB=10240
BOOTED=0; WD=""
log(){ echo "$(date -Is) [r482] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
mem(){ awk '/^MemAvailable|^AnonPages|^Cached:/{printf "%s %d MiB  ", $1, $2/1024}' /proc/meminfo; }
anon_mib(){ awk '/^AnonPages/{print int($2/1024)}' /proc/meminfo; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|cpu_moe_offload_layers|cpu_moe_split_experts|vision):' $CFG | awk '{print $1 $2}' | tr '\n' ' '; }
watchdog_start(){ rm -f "$R/RAM_ABORT"; ( while :; do
    a=$(awk '/^MemAvailable/{print $2}' /proc/meminfo); echo "$(date +%s) $a $(awk '/^AnonPages/{print $2}' /proc/meminfo)" >> "$R/ram.log"
    if [ "$a" -lt $RAM_FLOOR_KB ]; then echo "$(date -Is) MemAvailable $((a/1024)) MiB < floor" > "$R/RAM_ABORT"; sudo docker rm -f flashnext >/dev/null 2>&1; fi
    sleep 2; done ) & WD=$!; }
watchdog_stop(){ [ -n "$WD" ] && kill "$WD" 2>/dev/null; WD=""; }
finish(){ watchdog_stop
  if [ "$BOOTED" = 1 ]; then log "restoring the served daily"
    env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
    log "daily: $(served_id) $(cfgline) | $(mem)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R482 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$LM" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/fn_needle_oai.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r482-cold-experts
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
ANON0=$(anon_mib)
log "lock held; served at entry: $(served_id) $(cfgline) | $(mem)"
watchdog_start

# try_boot TAG CACHE N LAYERS -> 0 when the candidate serves exactly that config; 1 no boot; 2 abort
try_boot(){ local tag=$1 cache=$2 n=$3 layers=$4 i st lp env=$DAILY_ENV
  [ "$layers" != 0 ] && env="$env EXL3_MOE_CPU_SPLIT_LAYERS=$layers"
  BOOTED=1; log "boot $tag: cache $cache split $n experts x $([ "$layers" = 0 ] && echo all || echo $layers) layers | $(mem)"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      MAXBS=4 CACHE=$cache CACHE_MODE=8,8 MOE_OFFLOAD=0 MOE_SPLIT=$n EXTRA_ENV="$env" bash "$CAND" >> "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
  for i in $(seq 400); do
    [ -e "$R/RAM_ABORT" ] && { log "RAM ABORT during boot $tag: $(cat $R/RAM_ABORT)"; kill $lp 2>/dev/null; return 2; }
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp
      local got; got=$(cfgline)
      case "$got" in *"cache_size:$cache cache_mode:8,8 max_batch_size:4 cpu_moe_offload_layers:0 cpu_moe_split_experts:$n vision:true"*) ;; *) log "ABORT: config '$got'"; return 2;; esac
      sudo docker logs flashnext > "$R/boot-$tag-$cache.docker.log" 2>&1
      log "UP $tag @ $cache: VRAM free $(vram) | $(mem) | anon +$(( $(anon_mib) - ANON0 )) MiB vs served | split lines: $(grep -ac 'CPU split experts' "$R/boot-$tag-$cache.docker.log") ($(grep -a 'CPU split experts' "$R/boot-$tag-$cache.docker.log" | head -1 | cut -c1-120))"
      return 0; fi
    st=$(cstatus); [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 0)" -gt 0 ] && st="Restarting (restart count > 0)"
    case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 60 ] && { sleep 3; continue; }
      log "NO BOOT $tag @ $cache ($st): $(sudo docker logs --tail 60 flashnext 2>&1 | grep -aE 'RuntimeError|Error|VRAM|memory' | tail -1 | cut -c1-200)"
      sudo docker logs flashnext > "$R/boot-$tag-$cache.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1;; esac
    sleep 3; done
  log "NO BOOT $tag @ $cache (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }
ladder(){ local tag=$1 n=$2 layers=$3 c rc; shift 3; POOL=0
  for c in "$@"; do try_boot "$tag" "$c" "$n" "$layers"; rc=$?
    [ $rc = 0 ] && { POOL=$c; return 0; }; [ $rc = 2 ] && return 2; done; return 1; }

bench(){ python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" "$@" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | cut -c1-260; }
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(sys.argv[2], "c1 greedy sha", hashlib.sha256(t.encode()).hexdigest()[:16], "(served 1474eee2f5945248)")' "$R/greedy-$1.json" "$1" 2>&1 | tee -a "$R/audit.log"; }
greedy30k(){ python3 - "$API" "$MODEL" "$R/greedy30k-$1.json" "$1" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,hashlib,urllib.request,random
api,model,out,tag=sys.argv[1:5]
rng=random.Random(4242); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
body=" ".join(rng.choice(words) for _ in range(23000))
msg=body+"\n\nSummarize the passage above in one sentence, then write a haiku about it."
req=json.dumps({"model":model,"temperature":0,"max_tokens":128,"min_tokens":128,"messages":[{"role":"user","content":msg}]}).encode()
d=json.loads(urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=req,headers={"Content-Type":"application/json"}),timeout=900).read())
open(out,"w").write(json.dumps(d)); m=d["choices"][0]["message"]
print(tag,"30k greedy sha",hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16],"(served 4a255910dee2d9c5)")
PY
}
code_c1(){ python3 -c '
import json,sys,statistics
v=[r["wall_tps"] for r in map(json.loads,open(sys.argv[1])) if r.get("tag","").endswith("-code") and r.get("conc")==1 and r.get("wall_tps")]
print(round(statistics.median(v),1) if v else 0)' "$1" 2>/dev/null || echo 0; }
measure(){ local t=$1 full=$2
  log "=== measure $t: $(cfgline) VRAM free $(vram) | $(mem) ==="
  log "warm-up (swapper convergence): code + prose c4 x 1,024"
  bench --tag "$t-warm" --kind code --tokens 1024 --conc 4 --runs 1 --out "$R/warm-$t.jsonl" >/dev/null
  bench --tag "$t-warm" --kind prose --tokens 1024 --conc 4 --runs 1 --out "$R/warm-$t.jsonl" >/dev/null
  [ -e "$R/RAM_ABORT" ] && { log "RAM ABORT: $(cat $R/RAM_ABORT)"; return 2; }
  greedy "$t"; greedy30k "$t"
  for kind in code prose; do bench --tag "$t-$kind" --kind $kind --tokens 2048 --conc 1 4 --runs 2 --out "$R/records-$t.jsonl" | sed "s/^/[$t $kind]/" | tee -a "$R/audit.log"; done
  log "$t after decode: $(mem) | anon +$(( $(anon_mib) - ANON0 )) MiB vs served | split swaps logged: $(sudo docker logs flashnext 2>&1 | grep -aic 'swap')"
  [ -e "$R/RAM_ABORT" ] && { log "RAM ABORT: $(cat $R/RAM_ABORT)"; return 2; }
  [ "$full" = 1 ] || return 0
  bench --tag "ttft-$t" --kind prose --tokens 64 --conc 1 --runs 1 --ctx 30000 120000 --unique --out "$R/ttft-$t.jsonl" >/dev/null
  python3 - "$R/ttft-$t.jsonl" "$t" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys
for r in map(json.loads,open(sys.argv[1])):
    if r.get("ttft_s") and r.get("prompt_tokens"): print(f"[{sys.argv[2]} prefill] ctx {r['ctx_requested']}: {r['prompt_tokens']:,} prompt tokens, cold TTFT {r['ttft_s']:.2f} s = {r['prompt_tokens']/r['ttft_s']:,.0f} tok/s")
PY
  python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$MODEL" --tag "$t-needle" --ctx-tokens 131072 240000 \
    --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle-$t.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | sed "s/^/[$t needle]/" | tee -a "$R/audit.log"
  timeout 10800 "$LM" --model local-chat-completions \
    --model_args "base_url=$API/chat/completions,model=$MODEL,tokenizer=$CKPT,num_concurrent=4,max_retries=1,tokenized_requests=False" \
    --tasks gsm8k --num_fewshot 5 --limit 200 --apply_chat_template --gen_kwargs temperature=0,max_gen_toks=8192 --output_path "$R/ev-gsm8k-$t" > "$R/ev-gsm8k-$t.log" 2>&1
  python3 - "$R/ev-gsm8k-$t" "$t" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json, pathlib, sys
files = list(pathlib.Path(sys.argv[1]).rglob("results_*.json"))
if len(files) != 1: raise SystemExit(f"[{sys.argv[2]}] GSM8K: expected one result file, got {len(files)}")
d = json.loads(files[0].read_text()); s = d["results"]["gsm8k"]
print(f"[{sys.argv[2]}] GSM8K n={d.get('n-samples',{}).get('gsm8k',{}).get('effective')} c4: flexible {s['exact_match,flexible-extract']:.3f} (SE {s['exact_match_stderr,flexible-extract']:.4f}) strict {s['exact_match,strict-match']:.3f}   [served 0.925]")
PY
  log "$t done: $(mem) | anon +$(( $(anon_mib) - ANON0 )) MiB vs served"; }

# 1. PROBE: N=8 on the first 8 MoE layers at the served pool — host-RAM cost per split layer, first decode read
# R482 first launch (13:14 CEST): the probe did NOT boot at 360,448 — the layer-split loader's budget check failed at the
# last module (52/53) although 8 experts x 8 layers had left the GPU, so the split adds a GPU transient or buffer that
# outweighs the ~110 MiB of experts it removes. The probe now walks down to find where it fits; the VRAM free at that size
# against the unsplit engine (R480: 895 / 2,581 MiB at 360,448) prices the split.
ladder probe 8 8 360448 327680 294912 262144; rc=$?
[ $rc = 2 ] && { finish ABORTED; exit 3; }
[ "$POOL" = 0 ] && { log "probe did not serve at any size down to 262,144"; finish DONE; exit 0; }
log "probe (8 experts x 8 layers): largest pool that serves = $POOL"
measure probe 0; [ $? = 2 ] && { finish ABORTED; exit 3; }
PROBE_MIB=$(( $(anon_mib) - ANON0 )); [ $PROBE_MIB -lt 1 ] && PROBE_MIB=1
PER_LAYER=$(( PROBE_MIB / 8 ))
if [ $PROBE_MIB -lt 1024 ]; then LAYERS=0; log "probe: anon +$PROBE_MIB MiB for 8 layers x 8 experts -> the worker holds the tail only; real arms split ALL layers"
else LAYERS=$(( ANON_BUDGET_MIB / (PER_LAYER > 0 ? PER_LAYER : 1) )); [ $LAYERS -gt 48 ] && LAYERS=0
  log "probe: anon +$PROBE_MIB MiB for 8 layers (~$PER_LAYER MiB per split layer) -> the worker holds whole layers; real arms split the first $LAYERS layers (budget $ANON_BUDGET_MIB MiB)"; fi
PROBE_C1=$(code_c1 "$R/records-probe.jsonl"); log "probe code c1 median $PROBE_C1 (served 215)"

# 2. REAL ARMS, escalating N while code c1 >= 0.75 x 215
for N in 8 16 32; do
  ladder "sp$N" $N $LAYERS 557056 524288 491520 458752 425984 393216 360448; rc=$?
  [ $rc = 2 ] && { finish ABORTED; exit 3; }
  [ "$POOL" = 0 ] && { log "sp$N: no pool size served"; continue; }
  log "sp$N ($N experts x $([ $LAYERS = 0 ] && echo all || echo $LAYERS) layers on CPU): largest pool that serves = $POOL"
  measure "sp$N" 1; [ $? = 2 ] && { finish ABORTED; exit 3; }
  C1=$(code_c1 "$R/records-sp$N.jsonl"); log "sp$N code c1 median $C1"
  python3 -c "import sys; sys.exit(0 if float('$C1') >= 0.75*215 else 1)" || { log "sp$N code c1 $C1 < 161: stop escalating"; break; }
done
finish DONE
