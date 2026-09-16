#!/usr/bin/env bash
# R362 — upstream PR #337 A/B: does the layer-split device context change anything measurable here?
#
# The patch is a robustness fix (keep the process-wide CUDA current device on the module's device across
# forward_ls/prefill_ls). Its author found it via an out-of-tree kernel whose fault was misattributed to autotune;
# stock wrappers self-guard, so the expected effect on stock code ranges from nothing to "removes accidental P2P
# traffic". Either answer is worth having: if it is flat, our layer-split path is clean; if it wins, we have a lever
# nobody has measured on this model.
#
# THE CORRECTNESS GATE COMES FIRST. The patch moves device placement, so greedy output must be byte-identical to the
# baseline. If it is not, the patch changes numerics and the throughput columns are not worth reading.
#
# RUN: sudo systemd-run --unit=r362-pr337 --collect -p User=adrienbrault -p RuntimeMaxSec=14400 \
#        bash /srv/qwen5090/r362-pr337.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r362-pr337; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
L=/srv/qwen5090/launch-flashnext-r340.sh
log(){ echo "$(date -Is) [r362] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r362-pr337
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R362 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

greedy(){  # <out-prefix>
  curl -sN -m 900 "$API/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":512,\"min_tokens\":512,\"temperature\":0,\"stream\":true,
         \"messages\":[{\"role\":\"user\",\"content\":\"Write a Python LRU cache with type hints. Code only.\"}]}" \
    | python3 -c '
import sys, json
r, c = [], []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data:"): continue
    p = line[5:].strip()
    if p == "[DONE]": break
    try: o = json.loads(p)
    except Exception: continue
    for ch in o.get("choices") or []:
        d = ch.get("delta") or ch.get("message") or {}
        r.append(d.get("reasoning_content") or d.get("reasoning") or "")
        c.append(d.get("content") or "")
sys.stdout.write("R:" + "".join(r) + "\nC:" + "".join(c))' > "$1.txt"
  local n; n=$(wc -c < "$1.txt"); log "  greedy capture $n bytes -> $1.txt"
  [ "$n" -ge 200 ] || { log "  ERROR: capture too short; gate would be vacuous"; rm -f "$1.txt"; }
}

arm(){  # arm <tag> <image>
  log "booting $1: IMG=$2"
  IMG="$2" bash "$L" >> "$R/audit.log" 2>&1 || { log "$1 BOOT FAILED"; return 1; }
  curl -sf -m 8 "$API/model" >/dev/null || { log "$1: no server"; return 1; }
  sudo docker exec flashnext python3 - <<'PY' >> "$R/audit.log" 2>&1
import exllamav3, inspect, pathlib
p = pathlib.Path(inspect.getfile(exllamav3)).parent / "model" / "model_ls.py"
print("marker _LSDeviceContext present:", "_LSDeviceContext" in p.read_text())
PY
  greedy "$R/greedy-$1"
  log "  narrow context, 2048 forced, c1/c4/c8"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-narrow" --kind code \
     --tokens 2048 --conc 1 4 8 --runs 1 --out "$R/records-$1.jsonl" 2>&1 | tee -a "$R/audit.log"
  log "  deep context, 1024 forced, c4 at 152k-token prompts (where extra P2P traffic would show)"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-deep" --kind code \
     --tokens 1024 --ctx 120000 --conc 4 --runs 1 --out "$R/records-$1.jsonl" 2>&1 | tee -a "$R/audit.log"
}

arm baseline tabbyapi:53da7919-rqcount
arm pr337    tabbyapi:53da7919-rqcount-pr337

log "=== correctness: greedy output across arms ==="
for f in greedy-baseline greedy-pr337; do [ -s "$R/$f.txt" ] || log "GATE INVALID: $f.txt missing"; done
if [ -s "$R/greedy-baseline.txt" ] && cmp -s "$R/greedy-baseline.txt" "$R/greedy-pr337.txt"; then
  log "PASS baseline == pr337 (byte-identical); throughput columns are read in the summaries above"
else
  log "FAIL baseline != pr337 — the patch changes output; do not read the throughput columns"
fi

# Explicit image: the launcher's default moved to the enabled configuration on 2026-09-16, and a bare call here
# would silently boot the *enabled* config while this script's log claimed it had restored the improvement-free one.
log "restoring the improvement-free baseline (explicitly)"
IMG=tabbyapi:53da7919-rqcount bash "$L" >> "$R/audit.log" 2>&1 || log "BASELINE BOOT FAILED"
finish DONE
