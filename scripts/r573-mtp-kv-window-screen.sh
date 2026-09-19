#!/usr/bin/env bash
# R573 — mtp-kv-window r1, the full screen R569 earned (k = 3 steps). R569: EXL3_MTP_KV_WINDOW=16384 frees ~930 MiB of draft
# cache on cuda:0, and at the served [30, 30] split the loader then moves a layer onto cuda:0 (boot free 1,241 / 2,751 vs OFF
# 2,085 / 867; c1 18238d63 = R567's moved-layer fingerprint). Ladder (floor 835 MiB both cards, 120k-class prefill + c8 survive):
# 983,040 / 999,424 / 1,015,808 pass, 1,032,192 fails the floor on cuda:0 (821 / 2,311). cuda:1 keeps ~1.6 GB above the floor,
# so a split that moves weight back to cuda:1 should buy more. Image tabbyapi:mtpwin-r1 (R561 base), tier off, 8 slots.
#   A  acceptance / decode A/B at the served pool 966,656, [30, 30]: OFF vs W16 (different placement, so fingerprints differ by
#      design; decode is compared): mp_decode 24 code + 24 prose c1 + c4 512 tokens; deep c1 fn_bench code --ctx 100000 1,024
#      tokens salted; c8 at --ctx 16000 unique salted 1,024 tokens. W16 also: needles 131k / 240k (G3 set, 5 fracs each).
#   B  split sweep with W16: "29.5, 30.5" then "29, 31", each laddered from 1,015,808 up in 16,384 steps (<= 8) with the same
#      floor (835 MiB both cards) and survival; records the top pool per split.
# Rule (fixed now): W16 is a promotion candidate iff mp_decode c1 / c4 paired geo-means >= -1.5 % (code and prose), deep c1
# and c8@16k decode >= 0.97 x OFF, needles 5/5 at both depths, and the best (split, pool) is >= 1,015,808. Then: glm round
# rebases the overlay onto the R565 image; promotion unit = that image + W16 + the best split and pool, full gates.
# GPU TIMEBOX 40 min.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r573-mtp-kv-window-screen; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
MP=/srv/qwen5090/probes/mp_decode.py
WIMG=tabbyapi:mtpwin-r1
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r573] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){
  if [ "$BOOTED" = 1 ]; then sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done; log "daily: $(served_id)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R573 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
sudo docker image inspect "$WIMG" >/dev/null 2>&1 || { log "ABORT: $WIMG missing (built by R569)"; exit 3; }
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE" | tr ' ' '\n' | grep -v '^EXL3_NGRAM_PREFETCH2=' | tr '\n' ' ' | sed 's/ $//')
[ -n "$LENV" ] || { log "ABORT: EXTRA_ENV parse"; exit 3; }
export GPU_QUEUE_NAME=r573-mtp-kv-window-screen
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id)"
BOOTED=1
SALT=$(( $(date +%s) % 100000 ))
END=$(( $(date +%s) + 2400 ))
FLOOR=835
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$NEWM" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
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
free0(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0; }
free1(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1; }
survive(){ python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "surv-$1" --kind prose --tokens 64 --conc 1 --runs 1 --ctx 120000 --unique \
    --salt $(( SALT + RANDOM )) --out "$R/survive.jsonl" > /dev/null 2>&1
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "c8-$1" --kind code --tokens 1024 --conc 8 --runs 1 --out "$R/survive.jsonl" 2>&1 \
    | grep -E "^  c=|FAILED|Traceback" | cut -c1-200 | sed "s/^/[$1] /" | tee -a "$R/audit.log"
  alive; }
boot(){ local tag=$1 sp=$2 pool=$3 w=$4 ev="$LENV"
  [ "$w" != 0 ] && ev="$LENV EXL3_MTP_KV_WINDOW=$w"
  up "$tag" NVME_TIER= IMG="$WIMG" EXTRA_ENV="$ev" MAXBS=8 CACHE=$pool GPU_SPLIT="$sp" || return 1
  F0=$(free0); F1=$(free1); FP=$(greedy "$tag")
  log "UP $tag: split $sp pool $pool W $w; free at boot $F0/$F1; c1 $FP; $(sudo docker logs flashnext 2>&1 | grep -a 'EXL3_MTP_KV_WINDOW' | tail -1 | cut -c1-120)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$sp" "$pool" "$w" "$F0" "$F1" "$FP" >> "$R/boots.tsv"; }
deep(){ local tag=$1
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$tag-deep100k" --kind code --tokens 1024 --conc 1 --runs 1 --ctx 100000 --unique \
    --salt $(( SALT + 11 )) --out "$R/deep.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | cut -c1-220 | sed "s/^/[$tag deep c1] /" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$tag-c8ctx16k" --kind code --tokens 1024 --conc 8 --runs 1 --ctx 16000 --unique \
    --salt $(( SALT + 23 )) --out "$R/deep.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | cut -c1-220 | sed "s/^/[$tag c8@16k] /" | tee -a "$R/audit.log"; }
: > "$R/boots.tsv"
log "=== A: OFF vs W16 at [30, 30] @ 966,656 ==="
for arm in A1 B1; do
  w=0; [ $arm = B1 ] && w=16384
  boot $arm '30, 30' 966656 $w || { log "$arm: no boot"; finish FAILED; exit 1; }
  python3 "$MP" run --url "$API" --model "$NEWM" --tag $arm --tokens 512 --out "$R/mp.jsonl" 2>&1 | tail -2 | tee -a "$R/audit.log"
  deep $arm
  if [ $arm = B1 ]; then
    python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$NEWM" --tag needle --ctx-tokens 131072 240000 \
      --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee "$R/needle.log" | tee -a "$R/audit.log"
  fi
  sudo docker logs flashnext > "$R/docker-$arm.log" 2>&1
  alive || { log "$arm: NOT ALIVE"; finish FAILED; exit 1; }
done
python3 "$MP" compare --a A --b B "$R/mp.jsonl" 2>&1 | tee "$R/mp-compare.txt" | tee -a "$R/audit.log"
python3 - "$R/deep.jsonl" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys
d={}
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("ok"): d.setdefault(r["tag"],[]).append(r.get("decode_tps") or 0)
for k in ("deep100k","c8ctx16k"):
    a=d.get(f"A1-{k}"); b=d.get(f"B1-{k}")
    if a and b: print(f"{k}: OFF {sum(a)/len(a):.1f} W16 {sum(b)/len(b):.1f} decode_tps per stream, W16/OFF {sum(b)/len(b)/(sum(a)/len(a)):.3f}")
PY
log "=== B: split sweep with W16 ==="
for sp in '29.5, 30.5' '29, 31'; do
  top=none
  for c in 1015808 1032192 1048576 1064960 1081344 1097728 1114112 1130496; do
    [ $(( END - $(date +%s) )) -lt 150 ] && { log "timebox: sweep stops at [$sp] $c"; break 2; }
    t="S${sp%%,*}-$c"; t=${t//./p}
    boot "$t" "$sp" $c 16384 || { log "[$sp] $c: no boot"; break; }
    [ "$F0" -ge $FLOOR ] && [ "$F1" -ge $FLOOR ] || { log "[$sp] $c: free $F0/$F1 below floor $FLOOR"; break; }
    survive "$t" || { log "[$sp] $c: died under prefill + c8"; break; }
    top=$c; log "[$sp] $c: PASS"
  done
  log "split [$sp]: top pool $top"
done
finish DONE
