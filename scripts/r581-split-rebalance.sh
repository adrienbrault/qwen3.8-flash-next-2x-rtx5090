#!/usr/bin/env bash
# R581 — does moving the layer boundary buy KV? Screen only.
# GPU_SPLIT='30, 30' is a per-card budget in GiB for the WEIGHTS, not a layer count: the splitter fills cuda:0 to its
# budget and spills the rest to cuda:1, which also takes lm_head and the 320 MiB pruned draft-embedding mirror. The
# checkpoint has 48 layers with full_attention_interval 4, so the 12 attention layers are 3, 7, ... 47, and KV pages
# exist only on those. Before R579 the map was cuda:0 = 0-22 (5 attention) and cuda:1 = 23-47 (7 attention), so
# cuda:1 paid about 58 % of the pool and its 867 MiB set the ceiling while 2,085 MiB sat stranded on cuda:0.
# R579 promoted the draft-cache window, which frees draft cache on cuda:0, lets the splitter pull the boundary
# attention layer 23 across, and hands cuda:1 back that layer's weights AND its share of the pool: boot free went
# from 2,085 / 867 to 1,041 / 2,531 and the pool from 966,656 to 999,424. The imbalance did not go away, it INVERTED
# -- cuda:0 is now the tight card and 2.5 GB sits stranded on cuda:1.
# So the open question is whether the budget knob can move the boundary back the other way by one layer and buy the
# pool a second step. R573 swept [29, 31] and [29.5, 30.5] against the pre-window daily and found free VRAM
# identical, but that was the other layout and a smaller pool; the knob has never been swept on this one.
#   Arms: S = '30, 30' (control, the served value), A = '29, 31' (push a layer to cuda:1), B = '31, 29' (pull one to
#   cuda:0). Per arm, walk the pool up from the served 999,424 until a boot fails or a card falls below the 835 MiB
#   floor, and at every pool that boots run a c1 -> c8 decode ramp so every CUDA graph is captured before the pool is
#   called good (R575 died on a graph first captured under load; the ramp is what let R579 through).
#   Record per boot: the layer map from the container log, boot free per card, free after the ramp, c1 / 30k
#   fingerprints, and any graph.cu OOM line.
# Measurement only: nothing is promoted and no launcher is edited. A promotion unit follows if an arm wins.
# GPU TIMEBOX ~50 min. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-20-r581-split-rebalance; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
FLOOR=835
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r581] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
free0(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0; }
free1(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
finish(){
  if [ "$BOOTED" = 1 ]; then
    sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done; log "daily: $(served_id) $(cfgline)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R581 $1 ==="; }
trap 'log signal; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done

export GPU_QUEUE_NAME=r581-split-rebalance
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1

greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$NEWM" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
greedy30k(){ python3 - "$API" "$NEWM" "$R/greedy30k-$1.json" <<'PY' 2>/dev/null || echo none
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
# up TAG [ENV...] -> 0 up, 1 no boot
up(){ local tag=$1 i st lp; shift
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 ) & lp=$!
  for i in $(seq 200); do
    [ "$(served_id)" = "$NEWM" ] && { wait $lp; return 0; }
    st=$(cstatus); case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 40 ] && { sleep 3; continue; }
      log "NO BOOT $tag ($st): $(sudo docker logs --tail 80 flashnext 2>&1 | grep -aE 'RuntimeError|Error|memory' | tail -1 | cut -c1-160)"
      kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1;; esac
    [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 0)" -ge 1 ] && {
      log "NO BOOT $tag (restart loop): $(sudo docker logs --tail 80 flashnext 2>&1 | grep -aE 'RuntimeError|Error|memory' | tail -1 | cut -c1-160)"
      kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1; }
    sleep 3; done
  log "NO BOOT $tag (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }

# layermap -> "cuda:0 0-22 (5 attn) | cuda:1 23-47 (7 attn) head=cuda:1"
layermap(){ sudo docker logs flashnext 2>&1 | grep -a 'LS prefill pipeline' | head -1 | python3 -c "
import sys,re
s=sys.stdin.read()
# 'stages' is one list per device, in the order of the 'devices' list; the tuples are (module, stage, order).
stages=re.findall(r'\[(\(.*?\))\](?=,|\]\})', s)
out=[]
for d,st in enumerate(stages):
    nums=[int(m) for m in re.findall(r'layers\.(\d+)\'', st)]
    nums=sorted(set(nums))
    attn=[n for n in nums if n % 4 == 3]
    head=' head' if 'lm_head' in st else ''
    out.append(f\"cuda:{d} {min(nums)}-{max(nums)} ({len(nums)} layers, {len(attn)} attn){head}\" if nums else f'cuda:{d} empty')
print(' | '.join(out) if out else 'unparsed')
" 2>/dev/null || echo unparsed; }

# ramp -> 0 if c1..c8 all capture with no graph OOM
ramp(){ local tag=$1 c bad=0
  for c in 1 2 3 4 5 6 7 8; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$tag-ramp$c" --kind code --tokens 128 \
      --conc $c --runs 1 --out "$R/ramp.jsonl" > "$R/ramp-$tag-$c.log" 2>&1 || bad=1
    alive || { log "RAMP $tag: server not alive after c$c"; return 1; }
  done
  [ "$(sudo docker logs flashnext 2>&1 | grep -ac 'graph.cu')" = 0 ] || { log "RAMP $tag: graph.cu OOM in the container log"; return 1; }
  [ $bad = 0 ]; }

printf "arm\tsplit\tpool\tboot_free0\tboot_free1\tramp_free0\tramp_free1\tc1\tc30k\tlayers\tverdict\n" > "$R/sweep.tsv"
BEST=""
for arm in S A B; do
  case $arm in S) SPLIT='30, 30';; A) SPLIT='29, 31';; B) SPLIT='31, 29';; esac
  log "=== arm $arm: GPU_SPLIT='$SPLIT' ==="
  for POOL in 999424 1032192 1064960 1097728 1130496; do
    tag="$arm-$POOL"
    if ! up "$tag" GPU_SPLIT="$SPLIT" CACHE=$POOL NVME_TIER=; then
      log "$arm pool $POOL: NO BOOT -- arm stops here"
      printf "%s\t%s\t%s\t\t\t\t\t\t\t\tno-boot\n" "$arm" "$SPLIT" "$POOL" >> "$R/sweep.tsv"; break; fi
    b0=$(free0); b1=$(free1); lm=$(layermap)
    a=$(greedy "$tag"); b=$(greedy30k "$tag")
    log "$arm pool $POOL: boot free $b0/$b1; $lm; c1 $a / 30k $b"
    if ramp "$tag"; then r0=$(free0); r1=$(free1); v=ok
      # The floor is judged on free VRAM AFTER the ramp, not at boot: the graphs are what R575 ran out of memory
      # capturing, and they are not resident yet when the boot number is read (1,041 -> 251 MiB on cuda:0 at 999,424).
      { [ "$r0" -lt $FLOOR ] || [ "$r1" -lt $FLOOR ]; } && v="below-floor"
      log "$arm pool $POOL: ramp clean; free after ramp $r0/$r1; verdict $v"
      [ "$v" = ok ] && BEST="$BEST $arm:$POOL"
    else r0=$(free0); r1=$(free1); v=ramp-fail
      log "$arm pool $POOL: RAMP FAILED; free $r0/$r1 -- arm stops here"; fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$SPLIT" "$POOL" "$b0" "$b1" "$r0" "$r1" "$a" "$b" "$lm" "$v" >> "$R/sweep.tsv"
    [ "$v" = ok ] || break
  done
done
log "pools that booted, ramped clean and held the ${FLOOR} MiB floor:${BEST:- none}"
cat "$R/sweep.tsv" | tee -a "$R/audit.log"
finish DONE
