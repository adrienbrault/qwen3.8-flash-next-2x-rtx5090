#!/usr/bin/env bash
# R340 — concurrency-indexed draft depth: control, parity arm, treatment arm.
#
# WHY. R335 measured draft depth x concurrency on this engine and the launcher carries the operator's note that
# depth 1 is better at four concurrent requests. Today that is a per-boot choice; the patch makes it a load-time
# policy over the decode-ready job count. The question this answers is whether a single boot can hold c1's depth-3
# advantage and c4/c8's depth-1 advantage at once — and whether enabling it changes output.
#
# THREE ARMS, ONE VARIABLE EACH:
#   control    unpatched engine, no policy            the current served baseline
#   parity     patched engine, no policy              must reproduce control, or the patch is a regression
#   treatment  patched engine, policy [[2,3],[8,1]]   depth 3 up to two jobs, 1 above
#
# The correctness gate is not optional: greedy output at c1 is compared byte-for-byte between control and
# treatment. Both select depth 3 there, so identical output is the only acceptable result; a faster wrong answer
# is a failure. The comparison uses the same prompt and a fixed seed-free greedy request.
#
# RUN: sudo systemd-run --unit=r340-cid --collect -p User=adrienbrault -p RuntimeMaxSec=14400 \
#        bash /srv/qwen5090/r340-ci-depth.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r340-ci-depth; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
L=/srv/qwen5090/launch-flashnext-r340.sh          # the launcher, copied here by the caller
PROBE="python3 /srv/qwen5090/probes/fn_bench.py --url $API --model $MODEL --kind code --tokens 2048 --conc 1 4 8 --runs 2"
log(){ echo "$(date -Is) [r340] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r340-cid
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R340 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

boot(){   # boot <tag> <image> <policy>
  local tag=$1 img=$2 pol=$3
  log "booting $tag: IMG=$img DRAFT_POLICY='${pol}'"
  IMG="$img" DRAFT_POLICY="$pol" bash "$L" >> "$R/audit.log" 2>&1 || { log "$tag BOOT FAILED"; return 1; }
  local served; served=$(curl -s -m 5 "$API/model" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])' 2>/dev/null)
  [ "$served" = "$MODEL" ] || { log "$tag: server did not come up ($served)"; return 1; }
  # Prove which engine is actually running: the patch's marker is in the module the process imported.
  sudo docker exec flashnext python3 - <<'PY' >> "$R/audit.log" 2>&1
import exllamav3, inspect, pathlib
g = pathlib.Path(inspect.getfile(exllamav3)).parent / "generator" / "generator.py"
src = g.read_text()
print("engine marker _get_draft_depth:", "def _get_draft_depth(self, batch_size: int) -> int:" in src)
PY
}

greedy_text(){  # one greedy request, text captured for the byte-equality gate
  curl -sN -m 600 "$API/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":512,\"min_tokens\":512,\"temperature\":0,\"stream\":true,
         \"messages\":[{\"role\":\"user\",\"content\":\"Write a Python LRU cache with type hints. Code only.\"}]}" \
    | sed -n 's/^data: //p' | grep -v '^\[DONE\]$' \
    | python3 -c '
import sys, json
out = []
for line in sys.stdin:
    try: c = json.loads(line)
    except Exception: continue
    for ch in c.get("choices") or []:
        out.append((ch.get("delta") or {}).get("content") or "")
sys.stdout.write("".join(out))' > "$1"
  wc -c < "$1" | xargs -I{} log "  greedy text captured: {} bytes -> $1"
}

# --- arm 1: control (current baseline) -----------------------------------------------------------
boot control tabbyapi:53da7919-rqcount "" || finish "ABORTED (control boot)"
greedy_text "$R/greedy-control.txt"
$PROBE --tag control --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"

# --- arm 2: parity (patched engine, policy unset) ------------------------------------------------
boot parity tabbyapi:53da7919-rqcount-cid "" || finish "ABORTED (parity boot)"
greedy_text "$R/greedy-parity.txt"
$PROBE --tag parity --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"

# --- arm 3: treatment (patched engine, policy on) ------------------------------------------------
boot treatment tabbyapi:53da7919-rqcount-cid '[[2, 3], [8, 1]]' || finish "ABORTED (treatment boot)"
greedy_text "$R/greedy-treatment.txt"
$PROBE --tag treatment --out "$R/records.jsonl" 2>&1 | tee -a "$R/audit.log"

# --- correctness gates ---------------------------------------------------------------------------
log "=== byte-equality of greedy output (both arms select depth 3 at c1) ==="
if cmp -s "$R/greedy-control.txt" "$R/greedy-treatment.txt"; then log "PASS control == treatment";
else log "FAIL control != treatment ($(cmp -l "$R/greedy-control.txt" "$R/greedy-treatment.txt" | wc -l | tr -d ' ') differing bytes)"; fi
if cmp -s "$R/greedy-control.txt" "$R/greedy-parity.txt"; then log "PASS control == parity (patch alone changes nothing)";
else log "FAIL control != parity -- the patch alone changes output"; fi

# --- back to the baseline ------------------------------------------------------------------------
boot baseline tabbyapi:53da7919-rqcount "" || log "WARNING: could not restore the baseline image"
finish DONE
