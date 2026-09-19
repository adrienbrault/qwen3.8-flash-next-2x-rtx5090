#!/usr/bin/env bash
# R495b — audition r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw against the served 3.05bpw: pool frontier, decode, prefill, quality.
#
# WHY (user 2026-09-18: "what about this checkpoint? https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw"; standing
# ask "Decode speed and kv pool size"): same exllamav3 v1.5.0 format and architecture (config diff = quantization only: bits
# 2.52 vs 3.05, head 6 vs 5, mtp 4 vs 3, vision 6 vs 5; tokenizer + chat template byte-identical). GPU weights 41.5 GiB vs
# 49 GiB, and the served 360,448-token pool is only ~5.4 GiB of VRAM (15.75 KiB/token), so the freed ~7.5 GiB is up to ~+500k
# tokens of pool on paper; routed experts stream ~17 % fewer bytes per token (decode); the 4-bit MTP head may accept more.
# The card's quality numbers (GSM8K 79/80, HumanEval 40/40, IFEval 34/39, NIAH 262k) were taken at 3-bit KV on a 3090 with
# CPU experts, so they are not comparable: this unit reruns OUR gates at OUR settings (8,8 KV — the floor —, 4 slots, vision,
# [30, 30], chunk 2048, the served image and five flags) with launcher-r495 (= r484 + CKPT_NAME).
#
# SEQUENCE: ctl = the served 3.05 booted through the SAME launcher (r495, image + flags defaults = R481) at 360,448: c1 greedy, fn_bench code + prose c1/c4 2,048 x 2, cold prefill 30k / 120k,
# multiprompt 12 x kind 2,048 c1/c4 (ctl GSM8K 0.925 and tool-eval 83.2 are today's R487 / R491b numbers on the same
# harness). Ladder on 2.50: pools 1,048,576 (= 4 x 262,144) down in 65,536 steps to 360,448; the first that boots gets one
# +32,768 refinement. Winner: greedy (recorded; a different checkpoint cannot match the canonical hashes), fn_bench, prefill,
# 4 x 60k concurrent, 4 x (pool/4 - 8k, <= 200k) concurrent prose (the whole pool in flight), multiprompt, needle 131k /
# 240k, GSM8K n=200 c4, tool-eval 69 x 4. The daily is restored UNCHANGED at the end; promotion is a separate decision.
#
# RUN: sudo systemd-run --unit=r495b-2p50-audition --collect -p RuntimeMaxSec=43200 -E HOME=$HOME \
#        /bin/bash /srv/qwen5090/r495b-2p50-audition.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r495b-2p50-audition; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
CTLM=qwen3.8-flash-next-exl3-3.05bpw
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
MODEL=$CTLM
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r495.sh
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SERVED=360448
BOOTED=0
log(){ echo "$(date -Is) [r495b] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
finish(){ if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged live launcher)"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    MODEL=$CTLM; for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
    log "daily: $(served_id) $(cfgline) VRAM free $(vram)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R495b $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$LM" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/multiprompt.py /srv/qwen5090/probes/tooleval_summary.py /srv/qwen5090/models/$NEWM/config.json; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r495b-2p50-audition
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline) VRAM free $(vram)"
[ "$(served_id)" = "$MODEL" ] || { log "ABORT: the daily is not serving at entry"; finish ABORTED; exit 3; }

# try_boot TAG SPLIT CHUNK CACHE -> 0 when the candidate serves with exactly that config, 1 no boot, 2 config mismatch
try_boot(){ local tag=$1 split=$2 chunk=$3 cache=$4 i st lp got
  BOOTED=1; MODEL=$CK; log "boot $tag: split [$split] chunk $chunk cache $cache"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" CKPT_NAME=$CK GPU_SPLIT="$split" CHUNK=$chunk CACHE=$cache bash "$CAND" >> "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
  for i in $(seq 360); do
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp
      got=$(cfgline)
      case "$got" in *"cache_size: $cache cache_mode: 8,8 max_batch_size: 4 gpu_split: [$split] "*"chunk_size: $chunk "*"vision: true"*) ;;
        *) log "ABORT: generated config is '$got'"; return 2;; esac
      log "UP $tag @ $cache: VRAM free MiB $(vram)"; return 0; fi
    st=$(cstatus); case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 60 ] && { sleep 3; continue; }
      log "NO BOOT $tag @ $cache ($st): $(sudo docker logs --tail 60 flashnext 2>&1 | grep -aE 'RuntimeError|OutOfMemory|headroom|VRAM' | tail -1 | cut -c1-220)"
      sudo docker logs flashnext > "$R/boot-$tag-$cache.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1;; esac
    # a container that restarts in a loop reports Up between attempts: count restarts too
    [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 0)" -ge 1 ] && {
      log "NO BOOT $tag @ $cache (restart loop): $(sudo docker logs --tail 60 flashnext 2>&1 | grep -aE 'RuntimeError|OutOfMemory|headroom|VRAM' | tail -1 | cut -c1-220)"
      sudo docker logs flashnext > "$R/boot-$tag-$cache.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1; }
    sleep 3; done
  log "NO BOOT $tag @ $cache (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }

greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(sys.argv[2], "c1 greedy sha", hashlib.sha256(t.encode()).hexdigest()[:16], "(canonical 1474eee2f5945248)")' "$R/greedy-$1.json" "$1" 2>&1 | tee -a "$R/audit.log"; }
greedy30k(){ python3 - "$API" "$MODEL" "$R/greedy30k-$1.json" "$1" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,hashlib,urllib.request,random
api,model,out,tag=sys.argv[1:5]
rng=random.Random(4242); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
body=" ".join(rng.choice(words) for _ in range(23000))
msg=body+"\n\nSummarize the passage above in one sentence, then write a haiku about it."
req=json.dumps({"model":model,"temperature":0,"max_tokens":128,"min_tokens":128,"messages":[{"role":"user","content":msg}]}).encode()
d=json.loads(urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=req,headers={"Content-Type":"application/json"}),timeout=900).read())
open(out,"w").write(json.dumps(d)); m=d["choices"][0]["message"]
print(tag,"30k greedy sha",hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16],"(canonical 4a255910dee2d9c5)")
PY
}
prefill(){ local t=$1
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
decode(){ local t=$1 kind
  for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$t-$kind" --kind $kind --tokens 2048 --conc 1 4 --runs 2 \
      --out "$R/records-$t.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$t $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done; }
measure(){ local t=$1
  log "=== measure $t: $(cfgline) VRAM free $(vram) ==="
  greedy "$t"; greedy30k "$t"; decode "$t"; prefill "$t"
  log "[$t] four concurrent 60k prefills (runtime peak)"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "c4x60k-$t" --kind code --tokens 256 --conc 4 --runs 1 --ctx 60000 --unique \
    --out "$R/c4x60k-$t.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$t c4x60k]/" | cut -c1-260 | tee -a "$R/audit.log"
  N4=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get("completion_tokens",0)>=256))' "$R/c4x60k-$t.jsonl" 2>/dev/null || echo 0)
  log "[$t] c4x60k: $N4/4 completed at 256 tokens; engine $(served_id); VRAM free $(vram)"
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
print(f"[{sys.argv[2]}] GSM8K n={d.get('n-samples',{}).get('gsm8k',{}).get('effective')} c4: flexible {s['exact_match,flexible-extract']:.3f} (SE {s['exact_match_stderr,flexible-extract']:.4f}) strict {s['exact_match,strict-match']:.3f}   [served R480 s4 0.925]")
PY
}


mp(){ python3 /srv/qwen5090/probes/multiprompt.py --url "$API" --model "$MODEL" --tag "$1" --tokens 2048 --conc 1 4 --out "$R/multiprompt.jsonl" 2>&1 | tee -a "$R/audit.log"; }
te(){ local tag=$1
  ( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$MODEL" --temperature 0.6 --top-p 0.95 --top-k 20 \
      --trials 4 --parallel 8 --json-file "$R/tooleval-$tag.json" > "$R/tooleval-$tag.log" 2>&1 )
  python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval-$tag.json" "$tag" 2>&1 | tee -a "$R/audit.log"; }

[ "$(served_id)" = "$MODEL" ] || { log "ABORT: the daily is not serving at entry"; finish ABORTED; exit 3; }
CK=$CTLM; try_boot C "30, 30" 2048 360448 || { log "ABORT: ctl did not boot"; finish ABORTED; exit 3; }
log "=== measure ctl (3.05 through launcher-r495): $(cfgline) VRAM free $(vram) ==="
greedy ctl; decode ctl; prefill ctl; mp ctl

CK=$NEWM
log "=== ladder: $NEWM, split [30, 30], chunk 2048, 8,8 ==="
BEST=0
for pool in 1048576 983040 917504 851968 786432 720896 655360 589824 524288 458752 393216 360448; do
  try_boot L "30, 30" 2048 $pool; rc=$?
  [ $rc = 2 ] && { finish ABORTED; exit 3; }
  [ $rc = 0 ] && { BEST=$pool; break; }
done
[ "$BEST" = 0 ] && { log "ABORT: $NEWM does not boot at 360,448"; finish ABORTED; exit 3; }
if [ "$BEST" -lt 1048576 ]; then
  up=$((BEST + 32768)); try_boot L "30, 30" 2048 $up; rc=$?
  [ $rc = 2 ] && { finish ABORTED; exit 3; }
  if [ $rc = 0 ]; then BEST=$up; else try_boot W "30, 30" 2048 $BEST || { log "ABORT: winner $BEST does not re-boot"; finish ABORTED; exit 3; }; fi
fi
log "=== frontier: $NEWM boots at $BEST tokens ($(python3 -c "print(f'{$BEST/360448:.2f}')")x the served 360,448) ==="
sudo docker logs flashnext 2>&1 | grep -aiE "pipeline|placement" | tail -2 | cut -c1-400 >> "$R/audit.log"
measure N
mp N
BIG=$(( BEST / 4 - 8192 )); [ $BIG -gt 200000 ] && BIG=200000
log "[N] four concurrent prose prefills at ctx $BIG (the pool in flight: ~$((4 * BIG)) of $BEST)"
python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "c4big-N" --kind prose --tokens 256 --conc 4 --runs 1 --ctx $BIG --unique \
  --out "$R/c4big-N.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[N c4big]/" | cut -c1-260 | tee -a "$R/audit.log"
NB=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get("completion_tokens",0)>=256))' "$R/c4big-N.jsonl" 2>/dev/null || echo 0)
log "[N] c4big: $NB/4 completed at 256 tokens; engine $(served_id); VRAM free $(vram)"
te N
log "summary: ctl GSM8K 0.925 (R487) / tool-eval 83.2 (R491b CTL) vs N above; pool $BEST vs 360,448"
finish DONE
