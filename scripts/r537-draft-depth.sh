#!/usr/bin/env bash
# R537 — deeper MTP draft ceilings at c1 on the served 2.50bpw stack (user 2026-09-18: deeper MTP ceilings "try these"; R500
# never ran: superseded on the 3.05 base, then parked). Policy [[1,D],[4,3],[8,1]]: one decoding job drafts D, 2-4 jobs keep
# the served depth 3. Constraints: verify rows = jobs x (depth+1) <= 16 (1 job x 7 rows fine); the GDN recurrent-state history
# is sized by max(policy depths) + 1: 36 layers x 48 heads x 128 x 128 x 4 B = 113 MB per copy per slot, x 4 slots = ~0.45 GiB
# per extra depth, so the pool shrinks (1 GiB of 8-bit KV = ~87k tokens at 12 KB/token).
#   ladder  for D = 6, 5, 4: largest pool that boots (819,200 down in 16,384 steps, at most 8 steps) -> recorded (the pool cost)
#   A/B     all arms at the pool D6 boots at (arms differ only in the policy), order S D4 D5 D6 D6 D5 D4 S. Per boot: c1 greedy
#           fingerprint (recorded, depth may change numerics), fn_bench code c1 x3 + prose c1 x3 (warm-up 1), code c4 x1 (crash
#           check: R497's variable-depth graph OOM).
# NVMe tier off in every arm (NVME_TIER=): repeated fn_bench prompts would otherwise go through the tier.
# GPU TIMEBOX 45 min. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r537-draft-depth; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r537] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){ if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R537 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
SERVED_POLICY=$(sed -n 's/^DRAFT_POLICY=\${DRAFT_POLICY-\(.*\)}$/\1/p' "$LIVE")
[ "$SERVED_POLICY" = "[[4, 3], [8, 1]]" ] || { log "ABORT: live policy is '$SERVED_POLICY', expected [[4, 3], [8, 1]]"; exit 3; }
LCACHE=$(sed -n 's/^CACHE=\${CACHE:-\([0-9]*\)}.*/\1/p' "$LIVE"); [ -n "$LCACHE" ] || { log "ABORT: no CACHE default in the live launcher"; exit 3; }
pol(){ case $1 in S) echo "[[4, 3], [8, 1]]";; D*) echo "[[1, ${1#D}], [4, 3], [8, 1]]";; esac; }
export GPU_QUEUE_NAME=r537-draft-depth
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 2700 ))
log "lock held; live pool $LCACHE, served policy $SERVED_POLICY; timebox ends $(date -Is -d @$END)"
BOOTED=1
# boot TAG ARM CACHE -> 0 up with that policy and pool, 1 no boot
boot(){ local tag=$1 arm=$2 cache=$3 p i st lp; p=$(pol "$arm")
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" NVME_TIER= CACHE=$cache DRAFT_POLICY="$p" bash "$LIVE" > "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
  for i in $(seq 200); do
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp
      sudo grep -q "cache_size: $cache$" $CFG && sudo grep -qF "draft_num_tokens_by_batch: $p" $CFG || { log "ABORT: config lacks cache $cache / policy $p"; return 2; }
      log "UP $tag ($arm, policy $p) @ $cache: VRAM free MiB $(vram)"; return 0; fi
    st=$(cstatus); case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 40 ] && { sleep 3; continue; }
      log "NO BOOT $tag @ $cache ($st): $(sudo docker logs --tail 80 flashnext 2>&1 | grep -aE 'RuntimeError|Error|memory' | tail -1 | cut -c1-200)"
      sudo docker logs flashnext > "$R/boot-$tag-$cache.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1;; esac
    [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 0)" -ge 1 ] && {
      log "NO BOOT $tag @ $cache (restart loop): $(sudo docker logs --tail 80 flashnext 2>&1 | grep -aE 'RuntimeError|Error|memory' | tail -1 | cut -c1-200)"
      sudo docker logs flashnext > "$R/boot-$tag-$cache.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1; }
    sleep 3; done
  log "NO BOOT $tag @ $cache (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
log "=== pool ladder: largest pool per depth ==="
declare -A MAXPOOL
for arm in D6 D5 D4; do
  c=$LCACHE; MAXPOOL[$arm]=0
  for k in $(seq 0 8); do
    [ $(( END - $(date +%s) )) -lt 1500 ] && { log "ladder stopped: timebox"; break 2; }
    boot "L$arm" $arm $c; r=$?; [ $r = 2 ] && { finish FAILED; exit 3; }
    [ $r = 0 ] && { MAXPOOL[$arm]=$c; break; }
    c=$(( c - 16384 ))
  done
  log "LADDER $arm: largest pool ${MAXPOOL[$arm]} (served S = $LCACHE; $(( (LCACHE - ${MAXPOOL[$arm]}) )) tokens fewer)"
  [ "${MAXPOOL[$arm]}" = "$LCACHE" ] && { for a2 in D5 D4; do [ -z "${MAXPOOL[$a2]+x}" ] && MAXPOOL[$a2]=$LCACHE; done; log "LADDER: $arm fits the served pool, so every shallower depth does"; break; }
done
POOL=${MAXPOOL[D6]:-0}
[ "$POOL" -gt 0 ] || { log "D6 never booted; A/B at the D5 pool without D6"; POOL=${MAXPOOL[D5]:-0}; }
[ "$POOL" -gt 0 ] || { log "ABORT: no deep arm boots"; finish FAILED; exit 1; }
ARMS="S D4 D5 D6 D6 D5 D4 S"; [ "${MAXPOOL[D6]:-0}" = 0 ] && ARMS="S D4 D5 D5 D4 S"
log "=== A/B at pool $POOL: $ARMS ==="
rc=0; n=0
for arm in $ARMS; do
  n=$((n+1)); tag="$arm-$n"
  [ $(( END - $(date +%s) )) -lt 200 ] && { log "SKIP $tag and later boots: timebox"; break; }
  boot "$tag" $arm $POOL || { log "NO BOOT $tag at the common pool"; rc=1; break; }
  log "[$tag] c1 fingerprint $(greedy $tag)"
  for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-$kind-c1" --kind $kind --tokens 2048 --warmup-runs 1 \
      --conc 1 --runs 3 --out "$R/ab.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-200 | tee -a "$R/audit.log"; done
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-code-c4" --kind code --tokens 2048 \
    --conc 4 --runs 1 --out "$R/ab.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-200 | tee -a "$R/audit.log"
  [ "$(served_id)" = "$MODEL" ] || { log "FAIL $tag: server gone after the c4 run: $(sudo docker logs --tail 80 flashnext 2>&1 | grep -aE 'assert|Error|memory' | tail -1 | cut -c1-200)"; rc=1; break; }
done
python3 - "$R" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,statistics as st,collections,pathlib
R=pathlib.Path(sys.argv[1]); rounds=collections.defaultdict(list)
for l in open(R/"ab.jsonl"):
    r=json.loads(l); arm,n,kind,shape=r["tag"].split("-"); rounds[(arm,n,kind,shape,r["run"])].append(r)
v=collections.defaultdict(list)
for (arm,n,kind,shape,run),rs in rounds.items(): v[(arm,kind,shape)].append(sum(x["completion_tokens"] for x in rs)/max(x["round_wall_s"] for x in rs))
for kind,shape in (("code","c1"),("prose","c1"),("code","c4")):
    base=st.mean(v[("S",kind,shape)]) if v.get(("S",kind,shape)) else None
    row=[]
    for arm in ("S","D4","D5","D6"):
        x=v.get((arm,kind,shape))
        if x: row.append(f"{arm} {st.mean(x):.1f} (n {len(x)}, sd {st.pstdev(x):.1f}{'' if base is None or arm=='S' else f', {st.mean(x)/base-1:+.1%}'})")
    print(f"{kind} {shape}: " + " | ".join(row))
PY
[ $rc = 0 ] && finish DONE || finish FAILED
