#!/usr/bin/env bash
# R421 — coopwide image A/B: stage-B wide tile at >= 128 slots (flan/docker/coopwide.patch) vs the promoted bszn16 image.
# R419 measured wide stage B at 16 rows = c8 +7 % / c4 +5 % and forcing it at 4 rows = c1 -11 %; the patch moves the
# heuristic threshold (slots >= 128) so only >= 13-row batches change. Expected: c1 greedy IDENTICAL to BASE (gate), c4/c8
# up 5-7 %. Arms BASE1 / COOPWIDE / BASE2 (drift control), fn_bench c1/c4/c8 code 2048 x2 runs. Quality at c8 = R420.
set -uo pipefail
export HOME=${HOME:?}
R=/srv/qwen5090/results/2026-09-17-r421-coopwide-ab; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-3.05bpw
API=http://127.0.0.1:8022/v1
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
ARMS=${ARMS:-"BASE1=tabbyapi:qsa-cid-pr337-bszn16 COOPWIDE=tabbyapi:qsa-cid-pr337-bszn16-coopwide BASE2=tabbyapi:qsa-cid-pr337-bszn16"}
# arm syntax: NAME=IMAGE or NAME=IMAGE|KEY=VAL (one extra container env)
BOOTED=0
log(){ echo "$(date -Is) [r421] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
wait_id(){ local want=$1 i st; for i in $(seq 240); do [ "$(served_id)" = "$want" ] && return 0
  st=$(cstatus); case "$st" in Restarting*|Exited*) log "container $st"; sudo docker logs --tail 12 flashnext 2>&1 | grep -aE "Error|error|Traceback|CUDA" | tail -4 | cut -c1-240 | tee -a "$R/audit.log"; return 1;; esac
  sleep 2; done; return 1; }
boot(){ # $1 = arm name, $2 = image[|KEY=VAL]
  local img=${2%%|*} env=""; [ "$2" != "${2%%|*}" ] && env=${2#*|}
  BOOTED=1; log "booting arm $1 IMG=$img EXTRA_ENV='$env' (other launcher defaults = promoted config)"
  IMG="$img" EXTRA_ENV="$env" bash "$LIVE" >> "$R/audit.log" 2>&1 || { log "BOOT FAILED"; return 1; }
  wait_id "$MODEL" || { log "BOOT UNVERIFIED"; return 1; }
  [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = "$img" ] || { log "ABORT: container image is not $img"; return 1; }
  if [ -n "$env" ]; then [ "$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c "^$env$")" = 1 ] || { log "ABORT: container env lacks $env"; return 1; }; fi
  log "up: $1 on $img $env"; }
ladder(){ python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1" --kind code --tokens 2048 --conc 1 4 8 --runs 2 \
  --out "$R/records-$1.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$1]/" | tee -a "$R/audit.log"; }
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(sys.argv[2], "c1 greedy sha", hashlib.sha256(t.encode()).hexdigest()[:16], "usage", d.get("usage"))' "$R/greedy-$1.json" "$1" | tee -a "$R/audit.log"; }
finish(){ if [ "$BOOTED" = 1 ]; then log "restoring the daily (promoted config)"; env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    wait_id "$MODEL" && log "RESTORED: serving $MODEL on $(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" || log "RESTORE UNVERIFIED: $(served_id)"; fi; log "=== R421 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
for arm in $ARMS; do im=${arm#*=}; im=${im%%|*}; sudo docker image inspect "$im" >/dev/null 2>&1 || { log "ABORT: image $im missing"; exit 3; }; done
grep -qE '^IMG=\$\{IMG:-tabbyapi:qsa-cid-pr337-bszn16\}' "$LIVE" || { log "ABORT: live launcher default is not the promoted bszn16 image"; exit 3; }
# swebench-rearchive holds the GPU lock per image (since 2026-09-17 03:35), so it cannot stream through the page cache while this unit holds the lock.
GPU_QUEUE_NAME=r421-coopwide-ab
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id)"
for arm in $ARMS; do name=${arm%%=*}; env=${arm#*=}
  boot "$name" "$env" || { log "arm $name skipped (boot failed)"; continue; }
  greedy "$name"; ladder "$name"
done
python3 - "$R" <<'PY' | tee -a "$R/audit.log"
import json,sys,pathlib,hashlib,glob
R=pathlib.Path(sys.argv[1])
def sha(p):
    d=json.load(open(p)); m=d["choices"][0]["message"]; return hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16]
shas={p.stem.split("-",1)[1]:sha(p) for p in sorted(R.glob("greedy-*.json"))}
print("fingerprints:", shas)
for p in sorted(R.glob("records-*.jsonl")):
    rows=[json.loads(l) for l in open(p) if l.strip()]
    out={}
    for c in sorted({r.get("conc") for r in rows if r.get("ok")}):
        g=[r for r in rows if r.get("ok") and r.get("conc")==c]
        runs=sorted({r.get("run") for r in g})
        aggs=[]
        for run in runs:
            gr=[r for r in g if r.get("run")==run]; wall=gr[0].get("round_wall_s") or 0
            aggs.append(round(sum(r["completion_tokens"] for r in gr)/wall,1) if wall else None)
        dec=sorted(r["decode_tps"] for r in g if r.get("decode_tps"))
        out[f"c{c}"]={"agg_per_run":aggs,"decode_med":dec[len(dec)//2] if dec else None}
    print(p.stem, json.dumps(out))
PY
finish DONE
