#!/usr/bin/env bash
# R524 — recurrent tip checkpoints (Opus round recurrent-tip-checkpoint r1, patches/exllamav3/recurrent-tip-checkpoint/r1).
# Image tabbyapi:recurrent-tip-r1 = stack-r4-e3r2 + Python overlay behind EXL3_RECURRENT_TIP_STASH=1: a D2H snapshot of the
# GDN state at each page boundary a decoding job crosses, committed at job end as a checkpoint keyed by that page's hash, and
# tip-aware checkpoint eviction (stranded, then superseded, then LRU). Aim: an agent's next call resumes at its previous
# answer's last page instead of re-prefilling generated tokens or, after LRU loss, its whole context. Spec box-ab-spec.md:
# step 0 gpu_tip_replay.py (A tip / B base / C cold / D state distance); ON arm boot + stash line; fingerprints (canonical
# c1 ae890c45 / 30k 4a255910); agent replay in ECHO mode (the server's own answers sent back; 8 agents x 8 convs x 15 calls)
# with the per-request tip log; fn_bench code c1/c4. The OFF echo baseline is R524b (daily image, same replay).
# GPU TIMEBOX 15 min. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME
TAG=${TAG:-ON}
R=/srv/qwen5090/results/2026-09-19-r524-recurrent-tip; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/recurrent-tip-r1
IMG=tabbyapi:recurrent-tip-r1
TRAJS="/srv/qwen5090/results/2026-09-16-r359-swebench-10/out/*/*.traj.json /srv/qwen5090/results/2026-09-16-r360-swebench-strat/out/*/*.traj.json /srv/qwen5090/results/2026-09-16-r361-swebench-failed/out/*/*.traj.json /srv/qwen5090/results/2026-09-16-r369-swebench-30/out/*/*.traj.json"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
ONENV="EXL3_RECURRENT_TIP_STASH=1 EXL3_RECURRENT_TIP_LOG_EACH=1 EXL3_RECURRENT_TIP_LOG_SECS=30"
BOOTED=0
log(){ echo "$(date -Is) [r524$([ "$TAG" = OFF ] && echo b)] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
finish(){ sudo docker rm -f r524-probe >/dev/null 2>&1 || true; [ -n "${LP:-}" ] && kill $LP 2>/dev/null
  if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R524 $TAG $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$SRC/Dockerfile.box" /srv/qwen5090/probes/agent_replay.py /srv/qwen5090/probes/tabby_log_stats.py /srv/qwen5090/probes/fn_bench.py; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
grep -q -- "--echo" /srv/qwen5090/probes/agent_replay.py || { log "ABORT: agent_replay.py has no --echo"; exit 3; }
ENVS=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE"); [ -n "$ENVS" ] || { log "ABORT: no daily EXTRA_ENV"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")
# The tip overlay is built on stack-r4-e3r2. If the daily moved to the pruned-draft image (R528) and/or the tool_choice
# overlay (R529), both arms here still run on stack-r4-e3r2-based images with the pruned-draft flags stripped, so ON vs OFF
# stays internally consistent (R522: the pruned draft is byte-identical).
case "$LIMG" in tabbyapi:stack-r4-e3r2|tabbyapi:stack-r4-e3r2-*|tabbyapi:mtp-pruned-r1|tabbyapi:mtp-pruned-r1-*) ;; *) log "ABORT: live image '$LIMG' unknown to this unit"; exit 3;; esac
ENVS=$(echo " $ENVS " | sed -e 's/ EXL3_MTP_DEVICE_DRAFT=1 / /; s/ EXL3_EMBED_GPU=1 / /; s/ EXL3_EMBED_GPU_PRUNED=1 / /' | xargs)
if [ "$TAG" = ON ]; then log "building $IMG (before the lock)"
  (cd "$SRC" && sudo docker build -t "$IMG" -f Dockerfile.box . ) > "$R/build.log" 2>&1 || { log "ABORT: build failed: $(tail -3 "$R/build.log" | tr '\n' ' ' | cut -c1-240)"; exit 3; }; fi
export GPU_QUEUE_NAME=r524-recurrent-tip-$TAG
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 1500 ))
log "lock held; arm $TAG; timebox ends $(date -Is -d @$END)"
BOOTED=1
sudo docker stop -t 30 flashnext >/dev/null 2>&1; sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
if [ "$TAG" = ON ]; then
  DENV=(); for kv in $ENVS; do DENV+=(-e "$kv"); done
  # cuda:0 headroom: the harness loads the target with a 2048 MiB autosplit margin and exits 4 (SETUP) below 1.5 GiB;
  # --split does not bind at 29-30 GB (R524 tries 6-7 OOMed on cuda:0 either way)
  log "step 0: gpu_tip_replay.py"
  timeout 720 sudo docker run --rm --name r524-probe --gpus all --ipc=host --shm-size=16g "${DENV[@]}" \
    -e EXL3_RECURRENT_TIP_STASH=1 -e EXL3_RECURRENT_TIP_LOG_EACH=1 -e EXL3_RECURRENT_TIP_LOG_SECS=0 \
    -v /srv/qwen5090/models:/models:ro -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v "$R":/out --entrypoint python3 "$IMG" /opt/recurrent-tip-r1/tests/gpu_tip_replay.py --model /models/$MODEL --out /out/gpu_tip_replay.json \
    > "$R/gpu_tip_replay.log" 2>&1; rc=$?
  log "step 0 exit $rc: $(tail -4 "$R/gpu_tip_replay.log" | tr '\n' ' ' | cut -c1-300)"
  grep -a "verdict\|control_deterministic\|e3_on_rep_vs_rep\|phase D" "$R/gpu_tip_replay.log" | tail -6 | cut -c1-240 | tee -a "$R/audit.log"
  onlyD=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); r=d.get("reasons") or []; print(1 if r and all(x.startswith("phase D") for x in r) else 0)' "$R/gpu_tip_replay.json" 2>/dev/null || echo 0)
  case $rc in 0|2) ;; 3) log "step 0 INDETERMINATE (control not reproducible): continuing to the served A/B";;
    *) if [ "$onlyD" = 1 ]; then log "step 0 FAIL on phase D only (identity phases pass): continuing to the served A/B for throughput"; else log "step 0 FAIL"; finish FAILED; exit 1; fi;; esac
  BIMG=$IMG; BENV="$ENVS $ONENV"
else BIMG=tabbyapi:stack-r4-e3r2; BENV="$ENVS"; fi
"${CLEAN_ENV[@]}" IMG="$BIMG" EXTRA_ENV="$BENV" bash "$LIVE" > "$R/boot-$TAG.log" 2>&1 || { log "NO BOOT $TAG"; finish ABORTED; exit 3; }
for i in $(seq 120); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
[ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.Config.Image}}' flashnext)" = "$BIMG" ] || { log "NO BOOT $TAG"; finish ABORTED; exit 3; }
if [ "$TAG" = ON ]; then
  sudo docker logs flashnext 2>&1 | grep -a "recurrent tip stash: ON" | head -1 | tee -a "$R/audit.log" | grep -q . || { log "G1 FAIL: no stash-ON line"; finish FAILED; exit 1; }; fi
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
C1=ae890c45d1000582; case " $ENVS " in *" EXL3_HC_MIX_V2_INT8=1 "*) C1=e7fb377c987d685c;; esac   # R525 int8 mixer daily
a=$(greedy $TAG); b=$(greedy30k $TAG); log "[$TAG] fingerprints c1 $a / 30k $b (canonical $C1 / 4a255910dee2d9c5)"
[ "$a" = "$C1" ] && [ "$b" = 4a255910dee2d9c5 ] || { log "G2 FAIL"; finish FAILED; exit 1; }
sudo docker logs -f --since "$(date -Is)" flashnext > "$R/replay-$TAG.docker.log" 2>&1 & LP=$!
timeout 360 python3 /srv/qwen5090/probes/agent_replay.py --url "$API" --model "$MODEL" --trajs $TRAJS --agents 8 --convs 8 --calls 15 \
  --tool-gap 2 --echo --tag "echo-$TAG" --out "$R/replay.jsonl" 2>&1 | tail -1 | sed "s/^/[replay $TAG] /" | cut -c1-300 | tee -a "$R/audit.log"
sleep 2; kill $LP 2>/dev/null; LP=
python3 /srv/qwen5090/probes/tabby_log_stats.py "$R/replay-$TAG.docker.log" 2>&1 | tee -a "$R/audit.log"
python3 - "$R/replay-$TAG.docker.log" <<'PY' 2>&1 | tee -a "$R/audit.log"
import re,sys,collections
hits=collections.Counter(); rep=[]
for l in open(sys.argv[1], errors="replace"):
    m=re.search(r"-- recurrent tip: .*?replay (\d+).*?hit (\S+)", l)
    if m: rep.append(int(m[1])); hits[m[2]]+=1
    if "recurrent tip stash:" in l: last=l.strip()
print("tip lines", sum(hits.values()), "hits", dict(hits), "replay mean", round(sum(rep)/len(rep),1) if rep else None)
try: print("last summary:", last[-400:])
except NameError: print("no stash summary line")
PY
if [ $(( END - $(date +%s) )) -ge 120 ]; then
  timeout $(( END - $(date +%s) )) python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$TAG-code" --kind code --tokens 2048 --warmup-runs 1 \
    --conc 1 4 --runs 2 --out "$R/records-$TAG.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$TAG code]/" | cut -c1-220 | tee -a "$R/audit.log"
else log "SKIP fn_bench: timebox"; fi
sudo docker logs flashnext > "$R/docker-$TAG.log" 2>&1
log "[$TAG] errors in log: $(grep -acE 'Traceback|CUDA error|out of memory' "$R/docker-$TAG.log")"
finish DONE
