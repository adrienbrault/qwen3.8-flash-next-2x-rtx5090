#!/usr/bin/env bash
# R488 — is the 393,216 candidate's −17 % code c1 a speed loss or one greedy text's acceptance luck? 12 prompts per kind, sampled.
#
# WHY (user 2026-09-18): "Keep experimenting and pushing the setup. Decode speed and kv pool size. Prefill speed seems more
# than enough, 8k/s?". The served daily (R481: 4 slots, 360,448 at 8,8, [30, 30], chunk 2048) leaves 895 MiB free on cuda:0
# and 2,581 MiB on cuda:1; 393,216 did not boot (R480). The layer-split loader (exllamav3 model/model_ls.py:230-270) runs
# every module on a dummy chunk of chunk_size tokens and closes a device once its headroom no longer covers the largest
# transient measured on it plus EXL3_AUTOSPLIT_MARGIN_MB (256). So a smaller chunk shrinks the reserve on both cards, and a
# larger cuda:1 budget lets the layers cuda:0 sheds land on the card with 2.5 GiB idle. R452's [29, 32] / [29.5, 32] failures
# were at 8 slots with a 32 GiB budget on a 31.8 GiB card; this grid stays under the physical size.
#
# R487: [30, 31] / 2048 / 393,216 (a layer moves to cuda:1) = GSM8K 0.925, needles 10/10, prefill / prose c1 / c4 = served,
# but code c1 175–177 vs 212–215 and the c1 greedy flips to 7812e069df67f255 (as R483 D). fn_bench code c1 is ONE greedy
# text; if the moved layer shifts rounding, the text and its MTP acceptance change. Arms S0 (served geometry) / C / S1 / C1:
# probes/multiprompt.py, 12 distinct code + 12 prose prompts, temp 0.6 / top_p 0.95 / top_k 20, per-prompt seeds, 1,024
# forced, c1 and c4. Same speed = content luck (candidate promotable after gates); lower = a real placement cost.
#
# (R483 header follows)
# PHASE 1 (boot only): combos in order of prefill cost, each tries only pools ABOVE the best found so far, largest first:
#   C  split [30, 31]    chunk 2048      A  split [30, 30]  chunk 1024     D  split [30, 31]  chunk 1024
#   F  split [30.5, 31]  chunk 1024      B  split [30, 30]  chunk 512      E  split [30, 31]  chunk 512
#   pools 458,752 / 425,984 / 409,600 / 393,216 / 376,832.
# PHASE 2 (if a combo beats 360,448): the served daily (ctl, no reboot, decode + prefill only; R480 has its needle/GSM8K)
# and the winner: c1 + 30k greedy fingerprints, code + prose c1/c4 (2,048 forced, 2 runs), cold prefill 30k / 120k (one run
# per size: repeats hit the prefix cache), a four-way 60k concurrent prefill (the runtime peak the budget check does not see),
# needle 131k / 240k, GSM8K n=200 c4. The daily is restored UNCHANGED at the end; promotion is a separate gated unit.
#
# RUN: sudo systemd-run --unit=r488-multiprompt --collect -p RuntimeMaxSec=43200 -E HOME=$HOME \
#        /bin/bash /srv/qwen5090/r488-multiprompt.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r488-multiprompt; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r483.sh
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SERVED=360448
BOOTED=0
log(){ echo "$(date -Is) [r488] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
finish(){ if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged live launcher)"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
    log "daily: $(served_id) $(cfgline) VRAM free $(vram)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R488 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$LM" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/fn_needle_oai.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r488-multiprompt
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline) VRAM free $(vram)"
[ "$(served_id)" = "$MODEL" ] || { log "ABORT: the daily is not serving at entry"; finish ABORTED; exit 3; }

# try_boot TAG SPLIT CHUNK CACHE -> 0 when the candidate serves with exactly that config, 1 no boot, 2 config mismatch
try_boot(){ local tag=$1 split=$2 chunk=$3 cache=$4 i st lp got
  BOOTED=1; log "boot $tag: split [$split] chunk $chunk cache $cache"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" GPU_SPLIT="$split" CHUNK=$chunk CACHE=$cache bash "$CAND" >> "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
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


[ -e /srv/qwen5090/probes/multiprompt.py ] || { log "ABORT: missing probes/multiprompt.py"; finish ABORTED; exit 3; }
for arm in "S0|30, 30|360448" "C0|30, 31|393216" "S1|30, 30|360448" "C1|30, 31|393216"; do
  IFS='|' read -r tag split cache <<< "$arm"
  try_boot "$tag" "$split" 2048 "$cache"; rc=$?
  [ $rc = 0 ] || { log "ABORT: $tag did not boot (rc $rc)"; finish ABORTED; exit 3; }
  greedy "$tag"
  python3 /srv/qwen5090/probes/multiprompt.py --url "$API" --model "$MODEL" --tag "$tag" --tokens 1024 --conc 1 4 \
    --out "$R/multiprompt.jsonl" 2>&1 | tee -a "$R/audit.log"
done
finish DONE
