#!/usr/bin/env bash
# R492 — decode and prefill at depth on the served daily (no reboot; runs after R491 on whatever it leaves serving).
# User 2026-09-18: "what about decode/prefill at depth? @100k ? @200k ?". fn_bench `--ctx` is filler words (~0.75 tokens
# each: ctx 120000 -> 90,135 tokens), so ctx 133000 ≈ 100k tokens and 266000 ≈ 200k tokens (MAXLEN 262,144). Every prompt is
# unique (cold prefill); 1,024 forced tokens; decode_tps = tokens after the first / time after the first token.
# Arms: c1 code + prose at ≈0 / 100k / 200k; c4 code + prose at ≈60k each (4 × 60k = 240k of the 360,448 pool; c4 × 100k
# does not fit). The ≈0 arm is the same-session reference.
# RUN: sudo systemd-run --unit=r492-depth --collect -p RuntimeMaxSec=43200 -E HOME=$HOME /bin/bash /srv/qwen5090/r492-depth.sh
set -uo pipefail
R=/srv/qwen5090/results/2026-09-18-r492-depth; mkdir -p "$R"
API=http://127.0.0.1:8022/v1; MODEL=qwen3.8-flash-next-exl3-3.05bpw; CFG=/srv/qwen5090/flashnext-config.yml
log(){ echo "$(date -Is) [r492] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
trap 'log SIGTERM; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; exit 4' TERM
export GPU_QUEUE_NAME=r492-depth
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
[ "$(served_id)" = "$MODEL" ] || { log "ABORT: daily not serving"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; exit 3; }
log "serving: image $(sudo docker ps --format '{{.Image}}' -f name=flashnext) env $(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^EXL3_' | tr '\n' ' ') $(sudo grep -E '^  (cache_size|max_batch_size):' $CFG | tr -s ' ' | tr '\n' ' ')"
run(){ local tag=$1 kind=$2 conc=$3 ctx=$4
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag" --kind "$kind" --tokens 1024 --conc "$conc" --runs 1 --ctx "$ctx" --unique \
    --out "$R/records.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$tag]/" | cut -c1-300 | tee -a "$R/audit.log"; }
for kind in code prose; do
  run "$kind-c1-0" $kind 1 0
  run "$kind-c1-100k" $kind 1 133000
  run "$kind-c1-200k" $kind 1 266000
  run "$kind-c4-60k" $kind 4 80000
done
python3 - "$R/records.jsonl" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json, sys, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1])]
by = {}
for r in rows:
    if r.get("ok") and r.get("completion_tokens", 0) >= 1024: by.setdefault(r["tag"], []).append(r)
print(f"{'arm':16} {'n':>2} {'prompt tok':>10} {'TTFT s':>7} {'prefill tok/s':>13} {'decode t/s per stream':>22} {'aggregate':>9}")
for tag, v in by.items():
    pt = st.median(x["prompt_tokens"] for x in v); tt = st.median(x["ttft_s"] for x in v)
    dec = st.median(x["decode_tps"] for x in v); agg = sum(x["completion_tokens"] for x in v) / v[0]["round_wall_s"]
    print(f"{tag:16} {len(v):>2} {pt:>10,.0f} {tt:>7.2f} {pt/tt if len(v)==1 else float('nan'):>13,.0f} {dec:>22.1f} {agg:>9.1f}")
PY
rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R492 DONE ==="
