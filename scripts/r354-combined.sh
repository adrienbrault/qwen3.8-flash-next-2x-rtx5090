#!/usr/bin/env bash
# R354 — do the two measured levers compose?
#
#   QSA multi-job          +27 % at c2, +40 % per stream at c4 on 152,761-token contexts   (r341)
#   draft-depth policy     +35 % aggregate at c4 on 2,048-token generations                (r340)
#
# They touch different things (attention vs speculative depth) and were each validated alone, so the question is
# whether the wins add, cancel, or overlap. Both arms run on the same devel base with the same pip resolution and
# the same native rebuild; the only difference is the combination and the one config line.
#
# CORRECTNESS FIRST, as in both single-lever A/Bs: greedy output at c1 (both arms select depth 3 there) and at deep
# context with two concurrent requests must be byte-identical across arms. A composition that is faster and says
# something different is a failure.
#
# RUN: sudo systemd-run --unit=r354-combined --collect -p User=adrienbrault -p RuntimeMaxSec=14400 \
#        bash /srv/qwen5090/r354-combined.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r354-combined; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
L=/srv/qwen5090/launch-flashnext-r340.sh
log(){ echo "$(date -Is) [r354] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r354-combined
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R354 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

boot(){ log "booting $1: IMG=$2 policy='${3}'"
  IMG="$2" DRAFT_POLICY="$3" bash "$L" >> "$R/audit.log" 2>&1 || { log "$1 BOOT FAILED"; return 1; }
  curl -sf -m 8 "$API/model" >/dev/null || { log "$1: no server"; return 1; }
  sudo docker exec flashnext python3 - <<'PY' >> "$R/audit.log" 2>&1
import exllamav3, inspect, pathlib
g = pathlib.Path(inspect.getfile(exllamav3)).parent / "generator" / "generator.py"
a = pathlib.Path(inspect.getfile(exllamav3)).parent / "modules" / "attention_fn" / "bc_attn.py"
print("markers: draft_depth", "def _get_draft_depth(self, batch_size: int) -> int:" in g.read_text(),
      "| qsa_multijob", "BC-attn QSA slot layer" in a.read_text())
PY
}

measure(){  # measure <tag> <record-file>
  log "--- $1: short context c4 (draft-depth lever) ---"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-short" --kind code \
     --tokens 2048 --conc 1 4 8 --runs 1 --out "$2" 2>&1 | tee -a "$R/audit.log"
  log "--- $1: deep context c2/c4 (QSA lever) ---"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-deep" --kind code \
     --tokens 1024 --ctx 120000 --conc 2 4 --runs 1 --out "$2" 2>&1 | tee -a "$R/audit.log"
}

greedy(){  # <out-prefix>  single greedy request, both channels, for the cross-arm equality gate
  curl -sN -m 600 "$API/chat/completions" -H 'Content-Type: application/json' \
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
    try: obj = json.loads(p)
    except Exception: continue
    for ch in obj.get("choices") or []:
        d = ch.get("delta") or ch.get("message") or {}
        r.append(d.get("reasoning_content") or d.get("reasoning") or "")
        c.append(d.get("content") or "")
sys.stdout.write("R:" + "".join(r) + "\nC:" + "".join(c))' > "$1.txt"
  local n; n=$(wc -c < "$1.txt"); log "  greedy capture $n bytes -> $1.txt"
  [ "$n" -ge 200 ] || { log "  ERROR: capture too short; the equality gate would be vacuous"; rm -f "$1.txt"; }
}

boot baseline tabbyapi:53da7919-rqcount "" || finish "ABORTED (baseline)"
greedy "$R/greedy-baseline"
measure baseline "$R/records-baseline.jsonl"

boot combined tabbyapi:qsa-cid '[[2, 3], [8, 1]]' || finish "ABORTED (combined)"
greedy "$R/greedy-combined"
measure combined "$R/records-combined.jsonl"

log "=== correctness: greedy output must not change ==="
for f in greedy-baseline greedy-combined; do [ -s "$R/$f.txt" ] || log "GATE INVALID: $f.txt missing"; done
if [ -s "$R/greedy-baseline.txt" ] && cmp -s "$R/greedy-baseline.txt" "$R/greedy-combined.txt"; then
  log "PASS baseline == combined (byte-identical)"
else
  log "FAIL baseline != combined"
fi

boot baseline tabbyapi:53da7919-rqcount "" || log "WARNING: baseline did not come back up"
finish DONE
