#!/usr/bin/env bash
# R356 — the promoted configuration, validated end to end on the request that started all of this.
#
# WHY. Every measurement so far compares the combined engine (QSA multi-job + concurrency-indexed draft depth) on
# synthetic probes. This boots it as the served configuration and runs the DSH agent request that failed on
# 2026-09-16 — the exact payload rebuilt from the session archive — plus the retrieval gate at depth, so the
# promoted configuration is validated on the workload, not only on the instrument.
#
# The baseline is restored at the end: promoting the served configuration is the operator's call, and a session
# should hand the box back in the state it found it (or better, with the better option one line away).
#
# RUN: sudo systemd-run --unit=r356-promoted --collect -p User=adrienbrault -p RuntimeMaxSec=10800 \
#        bash /srv/qwen5090/r356-promoted.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r356-promoted; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
L=/srv/qwen5090/launch-flashnext-r340.sh
log(){ echo "$(date -Is) [r356] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r356-promoted
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R356 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

log "booting the promoted configuration: IMG=tabbyapi:qsa-cid DRAFT_POLICY='[[2, 3], [8, 1]]'"
IMG=tabbyapi:qsa-cid DRAFT_POLICY='[[2, 3], [8, 1]]' bash "$L" >> "$R/audit.log" 2>&1 \
  || { log "BOOT FAILED"; finish ABORTED; exit 1; }
curl -sf -m 8 "$API/model" >/dev/null || { log "no server"; finish ABORTED; exit 1; }
sudo docker exec flashnext python3 - <<'PY' >> "$R/audit.log" 2>&1
import exllamav3, inspect, pathlib
p = pathlib.Path(inspect.getfile(exllamav3)).parent
print("markers: draft_depth",
      "def _get_draft_depth(self, batch_size: int) -> int:" in (p / "generator" / "generator.py").read_text(),
      "| qsa_multijob",
      "BC-attn QSA slot layer" in (p / "modules" / "attention_fn" / "bc_attn.py").read_text())
PY
log "policy line in the served config: $(sudo grep -c 'draft_num_tokens_by_batch' /srv/qwen5090/flashnext-config.yml)"

# --- 1. the DSH agent request that failed before R338, byte-for-byte -------------------------------
log "=== the original failing agent request (tools + system + 3 user messages, max_tokens 32768) ==="
ssh_copy=/srv/qwen5090/r356-replay.json
if [ ! -s "$ssh_copy" ]; then log "ABORT: replay payload missing at $ssh_copy"; finish ABORTED; exit 3; fi
curl -sN -m 1800 "$API/chat/completions" -H 'Content-Type: application/json' --data-binary "@$ssh_copy" \
  > "$R/replay.sse" 2>&1
python3 - "$R/replay.sse" "$R/audit.log" <<'PY'
import json, sys
path, audit = sys.argv[1], sys.argv[2]
r, c, tools, finish = [], [], [], None
for line in open(path, errors="replace"):
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
        if d.get("tool_calls"): tools.append(d["tool_calls"])
        if ch.get("finish_reason"): finish = ch["finish_reason"]
R, C = "".join(r), "".join(c)
def weird(s):
    import unicodedata
    return sum(1 for ch in s if ord(ch) > 0x2000 and not unicodedata.category(ch).startswith("P"))
names = []
for t in tools:
    for call in t:
        fn = call.get("function") or {}
        if fn.get("name"): names.append(fn["name"])
lines = [
  f"  reasoning {len(R)} chars ({weird(R)/max(len(R),1)*100:.2f}% non-latin) | content {len(C)} chars",
  f"  finish_reason={finish} | tool calls parsed: {names or 'NONE'}",
]
if C.strip():
    lines.append(f"  visible text head: {C.strip()[:180]!r}")
open(audit, "a").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
PY

# --- 2. retrieval at depth on the promoted engine ---------------------------------------------------
log "=== needle at 131072 (the deep-context gate, on the promoted engine) ==="
python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$MODEL" --tag promoted-needle \
   --ctx-tokens 131072 --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | tee -a "$R/audit.log"

sudo docker logs --since 60m flashnext > "$R/server-log.txt" 2>&1
log "server log captured: $(wc -l < "$R/server-log.txt") lines"

log "restoring the baseline as the served configuration"
STOP=1 bash "$L" >/dev/null 2>&1 || true
bash "$L" >> "$R/audit.log" 2>&1 || log "BASELINE BOOT FAILED"
finish DONE
