#!/usr/bin/env bash
# R556 — adaptive MTP draft depth r2 screen (omp round, patches/exllamav3/adaptive-draft/r2, EXL3_ADAPTIVE_DRAFT2=1): per round
# the depth maximizing E[accepted + 1 | d] / T(d, bsz), E from per-job per-position acceptance, T from an online round-time
# table (R542's threshold rule lost 12 % on code). Python only (generator.py, job.py, new adaptive_draft.py).
# Built on tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16 (the R548 daily; gdn-bf16 and the ring do not touch generator.py/job.py,
# the overlay's install.py hash-checks them) regardless of what is live, so the screen stays valid if R555 promotes.
# Arms, one boot each, tier off, all at pool 1,015,808 (= the R548 pool - 16,384 so A4's extra depth fits):
#   S  = bf16 image, served policy [[4, 3], [8, 1]]
#   A3 = r2 image, flag on, same policy (ceiling 3: can it go shallower on prose and gain?)
#   A4 = r2 image, flag on, policy [[1, 4], [4, 3], [8, 1]] (ceiling 4 at one job: code +5 % without the prose loss?)
# Per boot: c1 fingerprint, mp_decode run (24 code + 24 prose, c1 + c4 groups, 512 tokens), docker log kept.
# Decision (spec, fixed before the run): A3 interesting if prose c1 CI > 0 and code c1 >= -1 %; A4 if code c1 >= +3.0 % and
# prose c1 CI excludes < -2.0 %; any c4 < -2 % kills an arm. A screen (1 boot per arm): a pass becomes a promotion unit.
# GPU TIMEBOX 20 min. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r556-adaptive-draft-r2; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/adaptive-draft-r2
MP=/srv/qwen5090/probes/mp_decode.py
CFG=/srv/qwen5090/flashnext-config.yml
BIMG=tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16
AIMG=tabbyapi:adaptive-draft-r2-gdnbf16
POOL=1015808
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r556] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){
  if [ "$BOOTED" = 1 ]; then sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done; log "daily: $(served_id)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R556 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$MP" "$SRC/Dockerfile.box" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
sudo docker image inspect "$BIMG" >/dev/null 2>&1 || { log "ABORT: base image $BIMG missing"; exit 3; }
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE")
case " $LENV " in *" EXL3_GDN_STATE_BF16=1 "*) ;; *) log "ABORT: live EXTRA_ENV lacks bf16 (expected the R548 daily or later)"; exit 3;; esac
BENV=$(echo "$LENV" | tr ' ' '\n' | grep -v '^EXL3_EMBED_MIRROR_DEVICE=' | tr '\n' ' ' | sed 's/ $//')   # R555's flag needs its own image
ADAPT="EXL3_ADAPTIVE_DRAFT2=1 EXL3_ADAPTIVE_DRAFT2_MAX_BATCH=1 EXL3_ADAPTIVE_DRAFT2_SWEEP=1 EXL3_ADAPTIVE_DRAFT2_LOG=1"
df -h / | tail -1 | tee -a "$R/audit.log"
log "S1 building $AIMG on $BIMG (before the lock)"
(cd "$SRC" && sudo docker build --build-arg BASE="$BIMG" -f Dockerfile.box -t "$AIMG" . ) > "$R/build.log" 2>&1 \
  || { log "ABORT: build: $(grep -aiE 'error|mismatch|abort|refus' "$R/build.log" | head -3 | tr '\n' ' ' | cut -c1-300)"; exit 3; }
export GPU_QUEUE_NAME=r556-adaptive-draft-r2
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id)"
BOOTED=1
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
rc=0
for arm in S A3 A4; do
  case $arm in
    S)  up S  NVME_TIER= IMG="$BIMG" EXTRA_ENV="$BENV" CACHE=$POOL || { rc=1; break; } ;;
    A3) up A3 NVME_TIER= IMG="$AIMG" EXTRA_ENV="$BENV $ADAPT" CACHE=$POOL || { rc=1; break; } ;;
    A4) up A4 NVME_TIER= IMG="$AIMG" EXTRA_ENV="$BENV $ADAPT" CACHE=$POOL DRAFT_POLICY='[[1, 4], [4, 3], [8, 1]]' || { rc=1; break; } ;;
  esac
  sudo grep -q "cache_size: $POOL$" $CFG || { log "FAIL $arm: config lacks cache $POOL"; rc=1; break; }
  pol=$(sudo grep -E 'draft_num_tokens_by_batch' $CFG | awk '{$1=$1; print}')
  ae=$(sudo docker exec flashnext env | grep -c '^EXL3_ADAPTIVE_DRAFT2')
  bl=$(sudo docker logs flashnext 2>&1 | grep -a 'EXL3_ADAPTIVE_DRAFT2=1:' | tail -1 | cut -c1-220)
  if [ $arm = S ]; then [ "$ae" = 0 ] && [ -z "$bl" ] || { log "FAIL S: adaptive env/boot line present"; rc=1; break; }
  else [ "$ae" = 4 ] && [ -n "$bl" ] || { log "FAIL $arm: adaptive env ($ae keys) / boot line missing"; rc=1; break; }; fi
  fp=$(greedy "$arm")
  log "UP $arm: $(sudo docker inspect -f '{{.Config.Image}}' flashnext); $pol; VRAM free $(vram); c1 $fp; ${bl:-no adaptive line}"
  python3 "$MP" run --url "$API" --model "$NEWM" --tag "${arm}x" --tokens 512 --out "$R/mp.jsonl" 2>&1 | tail -3 | tee -a "$R/audit.log"
  sudo docker logs flashnext > "$R/docker-$arm.log" 2>&1
  grep -aE 'out of memory|GPU assert|Traceback' "$R/docker-$arm.log" | head -3 | tee -a "$R/audit.log"
  log "$arm done; VRAM free $(vram); new verify shapes: $(grep -ac 'new verify shape' "$R/docker-$arm.log")"
done
for b in A3 A4; do
  python3 "$MP" compare --a Sx --b "${b}x" "$R/mp.jsonl" 2>&1 | sed "s/^/[S vs $b] /" | tee -a "$R/analysis.txt" | tee -a "$R/audit.log"
  python3 - "$R/docker-$b.log" "$b" <<'PY' 2>&1 | tee -a "$R/analysis.txt" | tee -a "$R/audit.log"
import re,sys,statistics as st
d=[float(m) for m in re.findall(r"mean_depth[=: ]+([0-9.]+)", open(sys.argv[1], errors="replace").read())]
print(f"[{sys.argv[2]}] per-request mean depth: n {len(d)}, mean {st.mean(d):.2f}, min {min(d):.2f}, max {max(d):.2f}" if d else f"[{sys.argv[2]}] no mean_depth lines")
PY
done
[ $rc = 0 ] && finish DONE || finish FAILED
