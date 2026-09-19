#!/usr/bin/env bash
# R480 — exllamav3 KV pool at 4 slots: how far the pool grows, and what K8V4 and one CPU-offloaded MoE layer cost.
#
# WHY (user 2026-09-18): "stop anything vllm related and go back to focusing on exllamav3"; "vision must stay on. then
# let's do 4 slots to optimize for c4"; "dont touch mtp depth"; "K8V4 let's def try this"; "offload 1 MoE layer's
# experts to CPU sounds worth trying?". VRAM on the daily: weights ~45.7 GiB, GDN recurrent state ~3.4 GiB (8 slots x 36
# layers x (draft depth + 1) fp32 copies, exllamav3 cache/recurrent.py), KV 262,144 tokens at 8,8 ~4 GiB (~65,536 tokens
# per GiB, R452), MTP head ~1 GiB, vision ~0.5 GiB; 327,680 did not boot at 8 slots under any split (R337/R452). 4 slots
# free ~1.7 GiB of recurrent state (~+110k tokens). K8V4 costs 832 B per token per attention layer against 1,088 at 8,8
# (= 6,6, which did not boot at 393,216 with 8 slots, R448). MoE offload moves the routed experts of the FIRST N MoE layers
# to the CPU (exllamav3 moe_cpu_offload), i.e. frees cuda:0, the card that binds (947 MiB free vs 2,511 on cuda:1).
#
# ARMS (vision on, MTP policy unchanged [[4, 3], [8, 1]] = depth 3 at up to 4 requests, EXTRA_ENV = the R460 daily's):
#   ctl     the served daily as found (8 slots, 262,144, 8,8) — measured without a reboot
#   s4      4 slots, 8,8, largest of 393,216 / 360,448 / 327,680 / 294,912 / 262,144 that boots
#   s4k84   4 slots, 8,4, largest of 524,288 / 491,520 / 458,752 / 425,984 / 393,216 / 360,448 / 327,680 that boots
#   s4off1  4 slots, 8,8, MOE_OFFLOAD=1, largest of s4 + 131,072 / + 65,536 / + 32,768 / s4 that boots
# MEASURE per arm: VRAM free; c1 greedy fingerprint (expect 1474eee2f5945248 at 8,8) and 30k greedy (4a255910dee2d9c5);
# fn_bench code AND prose c1/c4, 2,048 forced, 2 runs; prefill tok/s at the 30k/120k fillers (unique, 64 forced, 3 runs);
# needle five positions at 131,072 and 240,000; GSM8K n=200 (R355 parameters) at num_concurrent=4 for every arm.
# Every per-request window stays 262,144 (MAXLEN). The daily is restored UNCHANGED at the end (8 slots): the 4-slot
# promotion is decided on these numbers, with the tool-eval and agent-replay gates run on the chosen config.
#
# RUN: sudo systemd-run --unit=r480-exl3-pool --collect -p RuntimeMaxSec=43200 -E HOME=$HOME \
#        /bin/bash /srv/qwen5090/r480-exl3-pool.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r480-exl3-pool; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r480.sh
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
BOOTED=0
log(){ echo "$(date -Is) [r480] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){ if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged R460 launcher)"
    env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
    log "daily: $(served_id), config: $(sudo grep -E '^  (cache_size|cache_mode|max_batch_size|cpu_moe_offload_layers):' $CFG | tr -s ' ' | tr '\n' ' ')"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R480 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$LM" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/fn_needle_oai.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r480-exl3-pool
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id); VRAM free $(vram)"

# try_boot TAG SLOTS CACHE MODE OFFLOAD -> 0 when the candidate serves with exactly that config. The launcher waits up to
# 450 s for /health, so it runs in the background and a restart-looping container (insufficient VRAM) ends the try early.
try_boot(){ local tag=$1 bs=$2 cache=$3 mode=$4 off=$5 i st lp
  BOOTED=1; log "boot $tag: slots $bs cache $cache mode $mode moe_offload $off"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      MAXBS=$bs CACHE=$cache CACHE_MODE=$mode MOE_OFFLOAD=$off bash "$CAND" >> "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
  for i in $(seq 360); do
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp
      local got; got=$(sudo grep -E '^  (cache_size|cache_mode|max_batch_size|cpu_moe_offload_layers):' $CFG | awk '{print $2}' | tr '\n' ' ')
      [ "$got" = "$cache $mode $bs $off " ] || { log "ABORT: generated config is '$got', wanted '$cache $mode $bs $off '"; return 2; }
      log "UP $tag @ $cache: VRAM free MiB $(vram)"; return 0; fi
    st=$(cstatus); case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 60 ] && { sleep 3; continue; }
      log "NO BOOT $tag @ $cache ($st): $(sudo docker logs --tail 40 flashnext 2>&1 | grep -aE 'RuntimeError|Error|VRAM' | tail -1 | cut -c1-200)"
      sudo docker logs flashnext > "$R/boot-$tag-$cache.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1;; esac
    sleep 3; done
  log "NO BOOT $tag @ $cache (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }
# ladder TAG SLOTS MODE OFFLOAD SIZE... -> sets POOL to the first size that serves (engine left running), 0 if none
ladder(){ local tag=$1 bs=$2 mode=$3 off=$4 c rc; shift 4; POOL=0
  for c in "$@"; do try_boot "$tag" "$bs" "$c" "$mode" "$off"; rc=$?
    [ $rc = 0 ] && { POOL=$c; return 0; }; [ $rc = 2 ] && return 2; done; return 1; }

greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(sys.argv[2], "c1 greedy sha", hashlib.sha256(t.encode()).hexdigest()[:16], "(8,8 canonical 1474eee2f5945248)")' "$R/greedy-$1.json" "$1" 2>&1 | tee -a "$R/audit.log"; }
greedy30k(){ python3 - "$API" "$MODEL" "$R/greedy30k-$1.json" "$1" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,hashlib,urllib.request,random
api,model,out,tag=sys.argv[1:5]
rng=random.Random(4242); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
body=" ".join(rng.choice(words) for _ in range(23000))
msg=body+"\n\nSummarize the passage above in one sentence, then write a haiku about it."
req=json.dumps({"model":model,"temperature":0,"max_tokens":128,"min_tokens":128,"messages":[{"role":"user","content":msg}]}).encode()
d=json.loads(urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=req,headers={"Content-Type":"application/json"}),timeout=900).read())
open(out,"w").write(json.dumps(d)); m=d["choices"][0]["message"]
print(tag,"30k greedy sha",hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16],"(8,8 canonical 4a255910dee2d9c5)")
PY
}
measure(){ local t=$1
  log "=== measure $t: $(sudo grep -E '^  (cache_size|cache_mode|max_batch_size|cpu_moe_offload_layers|vision):' $CFG | tr -s ' ' | tr '\n' ' ') VRAM free $(vram) ==="
  greedy "$t"; greedy30k "$t"
  for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$t-$kind" --kind $kind --tokens 2048 --conc 1 4 --runs 2 \
      --out "$R/records-$t.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$t $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "ttft-$t" --kind prose --tokens 64 --conc 1 --runs 3 --ctx 30000 120000 --unique \
    --out "$R/ttft-$t.jsonl" 2>&1 | grep -E "FAILED|Traceback|Error" | tee -a "$R/audit.log"
  python3 - "$R/ttft-$t.jsonl" "$t" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,statistics
by={}
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("ttft_s") and r.get("prompt_tokens"): by.setdefault(r["ctx_requested"],[]).append((r["prompt_tokens"],r["ttft_s"]))
for ctx,v in sorted(by.items()):
    pt=v[0][0]; med=statistics.median(t for _,t in v)
    print(f"[{sys.argv[2]} prefill] ctx {ctx}: {pt:,} prompt tokens, TTFT median {med:.2f} s (n={len(v)}) = {pt/med:,.0f} tok/s")
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
print(f"[{sys.argv[2]}] GSM8K n={d.get('n-samples',{}).get('gsm8k',{}).get('effective')} c4: flexible {s['exact_match,flexible-extract']:.3f} (SE {s['exact_match_stderr,flexible-extract']:.4f}) strict {s['exact_match,strict-match']:.3f}   [daily c8 0.935]")
PY
}

# ctl: the served daily as found (no reboot) — the paired reference for every column
[ "$(served_id)" = "$MODEL" ] || { log "ABORT: the daily is not serving at entry"; finish ABORTED; exit 3; }
measure ctl
ladder s4 4 8,8 0 393216 360448 327680 294912 262144; rc=$?; P88=$POOL
[ $rc = 2 ] && { finish ABORTED; exit 3; }
log "s4 (4 slots, 8,8): largest pool that serves = $P88"; [ "$P88" != 0 ] && measure s4
ladder s4k84 4 8,4 0 524288 491520 458752 425984 393216 360448 327680; rc=$?
[ $rc = 2 ] && { finish ABORTED; exit 3; }
log "s4k84 (4 slots, 8,4): largest pool that serves = $POOL"; [ "$POOL" != 0 ] && measure s4k84
if [ "$P88" != 0 ]; then
  ladder s4off1 4 8,8 1 $((P88+131072)) $((P88+65536)) $((P88+32768)) $P88; rc=$?
  [ $rc = 2 ] && { finish ABORTED; exit 3; }
  log "s4off1 (4 slots, 8,8, first MoE layer's experts on CPU): largest pool that serves = $POOL"; [ "$POOL" != 0 ] && measure s4off1
fi
finish DONE
