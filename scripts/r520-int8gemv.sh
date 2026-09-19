#!/usr/bin/env bash
# R520 — EXL3_INT8_GEMV A/B on the daily. exllamav3 defaults EXL3_INT8_GEMV=2: single-row (mul1) quantized linears run a fused
# Hadamard + int8-activation dp4a GEMV instead of the fp16 kernel (doc/env_vars.md:125). On GB10 turning it OFF (=0) was worth
# +3 % decode (vcruz305 recipe); it has never been measured here. OFF changes numerics (the fp16 path does not quantize
# activations), so a win needs a quality gate before it can serve. Arms ON / OFF / ON2 / OFF2 on the live launcher with its own
# EXTRA_ENV (+ EXL3_INT8_GEMV=0 on OFF arms): fingerprints, fn_bench code + prose c1/c4 2,048 x 2. GPU TIMEBOX 15 min.
# RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r520-int8gemv; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r520] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
finish(){
  if [ "$BOOTED" = 1 ]; then log "restoring the unchanged daily"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R520 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
ENVS=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE")
[ -n "$ENVS" ] || { log "ABORT: cannot read the daily EXTRA_ENV"; exit 3; }
case " $ENVS " in *" EXL3_INT8_GEMV="*) log "ABORT: the daily already sets EXL3_INT8_GEMV"; exit 3;; esac
export GPU_QUEUE_NAME=r520-int8gemv
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 900 ))
log "lock held; timebox ends $(date -Is -d @$END); served $(served_id)"
BOOTED=1
boot(){ local tag=$1 extra=$2 i
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  "${CLEAN_ENV[@]}" EXTRA_ENV="$ENVS $extra" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 || { log "NO BOOT $tag"; return 1; }
  for i in $(seq 120); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
  [ "$(served_id)" = "$MODEL" ] || { log "NO BOOT $tag (id $(served_id))"; return 1; }
  if [ -n "$extra" ]; then sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -qx "$extra" || { log "ABORT: container lacks $extra"; return 1; }; fi
  log "UP $tag ${extra:-default}"; }
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
for arm in "ON|" "OFF|EXL3_INT8_GEMV=0" "ON2|" "OFF2|EXL3_INT8_GEMV=0"; do
  IFS='|' read -r tag extra <<< "$arm"
  [ $(( END - $(date +%s) )) -lt 200 ] && { log "SKIP $tag: timebox"; continue; }
  boot "$tag" "$extra" || { finish ABORTED; exit 3; }
  log "[$tag] c1 fingerprint $(greedy "$tag") (canonical ae890c45d1000582)"
  for kind in code prose; do
    timeout $(( END - $(date +%s) )) python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-$kind" --kind $kind --tokens 2048 \
      --conc 1 4 --runs 2 --out "$R/records.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$tag $kind]/" | cut -c1-220 | tee -a "$R/audit.log"; done
done
python3 - "$R/records.jsonl" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,statistics as st
d={}
for l in open(sys.argv[1]):
    r=json.loads(l); arm,kind=r["tag"].rsplit("-",1); d.setdefault((arm.rstrip("2"),kind,r["conc"]),[]).append(r["aggregate_tps"])
for kind in ("code","prose"):
    for c in (1,4):
        a,b=d.get(("ON",kind,c)),d.get(("OFF",kind,c))
        if a and b: print(f"{kind} c{c}: ON {st.mean(a):.1f} (n={len(a)}) / OFF {st.mean(b):.1f} (n={len(b)}) -> {(st.mean(b)/st.mean(a)-1)*100:+.1f} %")
PY
finish DONE
