#!/usr/bin/env bash
# R491 — promotion gates for the shared-expert side-stream overlap (codex decode-kernels r2) on the exllamav3 daily, then promote on PASS.
#
# WHY (user 2026-09-18: "Keep experimenting and pushing the setup. Decode speed and kv pool size"): R490 measured the shared
# expert on a side stream byte-identical (canonical fingerprints on OFF/ON/OFF2/ON2) and faster: multi-prompt paired code c1
# +6.3 %, code c4 +5.5 %, prose c1 +2.7 %, prose c4 +6.9 %; one MoE layer −9..−16 %. This unit runs the remaining gates on the
# candidate (flan/launch-flashnext-r491.sh = the R481 daily + image tabbyapi:decode-kernels-r2 + EXL3_SHARED_EXPERT_OVERLAP=1):
# canonical c1 fingerprint, the R356 agent replay (must end in tool_calls), c8 on four slots (8/8), tool-eval 69x4 >= 80.
# PASS -> it becomes /srv/qwen5090/launch-flashnext.sh; the R481 launcher stays as the rollback.
#
# RUN: sudo systemd-run --unit=r491-shared-promote --collect -p RuntimeMaxSec=43200 -E HOME=$HOME \
#        /bin/bash /srv/qwen5090/r491-shared-promote.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r491-shared-promote; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r491.sh
ROLLBACK=/srv/qwen5090/launch-flashnext-r481-s4.sh
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0; PROMOTED=0; FAILS=0
log(){ echo "$(date -Is) [r491] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
wait_up(){ local i; for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && return 0; sleep 2; done; return 1; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|cpu_moe_offload_layers|vision):' $CFG | awk '{print $1 $2}' | tr '\n' ' '; }
finish(){ if [ "$BOOTED" = 1 ] && [ "$PROMOTED" = 0 ]; then log "restoring the R481 daily unchanged"
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"; wait_up; log "daily: $(served_id) $(cfgline)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R491 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" /srv/qwen5090/r356-replay.json /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/tooleval_summary.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
command -v tool-eval-bench >/dev/null 2>&1 || { log "ABORT: tool-eval-bench not on PATH"; exit 3; }
export GPU_QUEUE_NAME=r491-shared-promote
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"

BOOTED=1
"${CLEAN_ENV[@]}" bash "$CAND" >> "$R/audit.log" 2>&1 || { log "BOOT FAILED"; finish ABORTED; exit 3; }
wait_up || { log "BOOT UNVERIFIED"; finish ABORTED; exit 3; }
got=$(cfgline); log "candidate up: $got VRAM free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/')"
case "$got" in *"cache_size:360448 cache_mode:8,8 max_batch_size:4 cpu_moe_offload_layers:0 vision:true"*) ;; *) log "ABORT: config is not the candidate"; finish ABORTED; exit 3;; esac
[ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = tabbyapi:decode-kernels-r2 ] || { log "ABORT: container image is not tabbyapi:decode-kernels-r2"; finish ABORTED; exit 3; }
sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -qx EXL3_SHARED_EXPERT_OVERLAP=1 || { log "ABORT: container env lacks EXL3_SHARED_EXPERT_OVERLAP=1"; finish ABORTED; exit 3; }

log "=== gate 1: c1 greedy fingerprint (canonical 1474eee2f5945248) ==="
curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy.json"
FP=$(python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; print(hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16])' "$R/greedy.json" 2>/dev/null || echo none)
[ "$FP" = 1474eee2f5945248 ] && log "GATE 1 PASS: $FP" || { log "GATE 1 FAIL: $FP"; FAILS=$((FAILS+1)); }

log "=== gate 2: the original failing agent request (R356 replay) ==="
curl -sN -m 1800 "$API/chat/completions" -H 'Content-Type: application/json' --data-binary @/srv/qwen5090/r356-replay.json > "$R/replay.sse" 2>&1
G2=$(python3 - "$R/replay.sse" <<'PY'
import json, sys
r = c = ""; tools = []; finish = None
for line in open(sys.argv[1], errors="replace"):
    line = line.strip()
    if not line.startswith("data:"): continue
    p = line[5:].strip()
    if p == "[DONE]": break
    try: o = json.loads(p)
    except Exception: continue
    for ch in o.get("choices") or []:
        d = ch.get("delta") or ch.get("message") or {}
        r += d.get("reasoning_content") or d.get("reasoning") or ""; c += d.get("content") or ""
        for call in d.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name"): tools.append(fn["name"])
        if ch.get("finish_reason"): finish = ch["finish_reason"]
print(f"GATE 2 {'PASS' if tools and finish == 'tool_calls' else 'FAIL'}: reasoning {len(r)} chars, content {len(c)} chars, finish_reason={finish}, tools={tools}")
PY
); log "$G2"; case "$G2" in "GATE 2 PASS"*) ;; *) FAILS=$((FAILS+1));; esac

log "=== gate 3: c8 on four slots (all eight must complete; four queue) ==="
python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag s4-c8 --kind code --tokens 1024 --conc 8 --runs 1 \
  --out "$R/records-c8.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | cut -c1-300 | tee -a "$R/audit.log"
N8=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get("completion_tokens",0)>=1024))' "$R/records-c8.jsonl" 2>/dev/null || echo 0)
[ "$N8" = 8 ] && log "GATE 3 PASS: 8/8 completed at 1,024 tokens" || { log "GATE 3 FAIL: $N8/8 completed"; FAILS=$((FAILS+1)); }

log "=== gate 4: tool-eval 69x4, sampler 0.6/0.95/20, parallel 8 (served band 83.5-86.5) ==="
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$MODEL" \
    --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval.json" > "$R/tooleval.log" 2>&1 )
if [ -s "$R/tooleval.json" ]; then
  SUM=$(python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval.json" s4 2>&1); echo "$SUM" | tee -a "$R/audit.log"
  SC=$(echo "$SUM" | grep -oE '[0-9]+\.[0-9]+ \+-' | head -1 | cut -d' ' -f1)
  python3 -c "import sys; sys.exit(0 if float('${SC:-0}') >= 80.0 else 1)" && log "GATE 4 PASS: $SC (floor 80.0 = served band minus its spread)" || { log "GATE 4 FAIL: '$SC'"; FAILS=$((FAILS+1)); }
else log "GATE 4 FAIL: no JSON; $(tail -2 "$R/tooleval.log" | cut -c1-160)"; FAILS=$((FAILS+1)); fi

if [ "$FAILS" = 0 ]; then
  [ -e "$ROLLBACK" ] || sudo cp "$LIVE" "$ROLLBACK"
  cmp -s "$LIVE" "$ROLLBACK" && log "rollback launcher kept at $ROLLBACK" || log "NOTE: $ROLLBACK differs from the live launcher; live kept as $LIVE.pre-r491"
  sudo cp "$LIVE" "$LIVE.pre-r491"; sudo cp "$CAND" "$LIVE.new" && sudo mv "$LIVE.new" "$LIVE"
  PROMOTED=1; log "PROMOTED: $LIVE = $CAND (shared-expert overlap, image tabbyapi:decode-kernels-r2); serving: $(served_id) $(cfgline)"
else log "NOT PROMOTED: $FAILS gate(s) failed"; fi
finish DONE
