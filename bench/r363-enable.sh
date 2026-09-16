#!/usr/bin/env bash
# R363 — enable the improvements on the served configuration, in the order that keeps every measurement honest.
#
# WHAT "ENABLED" MEANS. The launcher's defaults are the artifact of record: since 2026-09-16 it serves
# `tabbyapi:qsa-cid` with `DRAFT_POLICY='[[2, 3], [8, 1]]'`, i.e. the two improvements that were measured on matched
# arms — QSA multi-job and concurrency-indexed draft depth — plus the R338 requeue token-count fix that is in every
# image since. The improvement-free baseline stays one variable away (`IMG=tabbyapi:53da7919-rqcount`).
#
# THIS CHAIN, in order:
#   1. PR #337's A/B (r362) — the only upstream patch of the twenty open ones that applies cleanly to v1.5.0 and
#      touches the layer-split topology this seat runs. Its correctness gate decides whether it is eligible at all.
#   2. If and only if PR #337 is byte-identical to the baseline, build `tabbyapi:qsa-cid-pr337` (PR #337 applied on
#      top of the already-built combined image) and serve THAT; otherwise serve `tabbyapi:qsa-cid`.
#   3. Prove what is running: markers for each patch inside the live container, the policy line in the served config,
#      and — the strongest check — a greedy capture whose sha256 must equal the fingerprint every earlier run
#      produced for the same request (`750e1459e177c47e…`, 1,989 bytes). If the enabled configuration changes one
#      byte of output, this fails.
#   4. Confirm the workload still works: the original failing agent request must return parsed tool calls, and the
#      needle gate must read 5/5 at 105,680 prompt tokens.
#
# RUN: sudo systemd-run --unit=r363-enable --collect -p User=adrienbrault -p RuntimeMaxSec=21600 \
#        bash /srv/qwen5090/r363-enable.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r363-enable; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
L=/srv/qwen5090/launch-flashnext-r340.sh
CANON_SHA=750e1459e177c47e
log(){ echo "$(date -Is) [r363] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r363-enable
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R363 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

# --- 1. PR #337's A/B -----------------------------------------------------------------------------
log "=== 1. PR #337 A/B (its own gate decides eligibility) ==="
bash /srv/qwen5090/r362-pr337.sh >> "$R/audit.log" 2>&1
if grep -qa "PASS baseline == pr337" /srv/qwen5090/results/2026-09-16-r362-pr337/audit.log; then
  PR337=1; log "PR #337 passed its correctness gate; it becomes eligible"
else
  PR337=0; log "PR #337 did NOT pass its gate (or did not run); it is NOT enabled"
fi

# --- 2. the image to serve -----------------------------------------------------------------------
IMG=tabbyapi:qsa-cid
if [ "$PR337" = 1 ]; then
  log "=== 2. building tabbyapi:qsa-cid-pr337 (PR #337 on the combined image) ==="
  if (cd /srv/qwen5090/docker && sudo docker build -f Dockerfile.tabbyapi-pr337 \
        --build-arg BASE=tabbyapi:qsa-cid -t tabbyapi:qsa-cid-pr337 . >> "$R/build.log" 2>&1); then
    IMG=tabbyapi:qsa-cid-pr337; log "combined image built"
  else
    log "combined build FAILED; serving tabbyapi:qsa-cid (the validated pair) instead"
    tail -4 "$R/build.log" | cut -c1-160 | tee -a "$R/audit.log"
  fi
fi
log "SERVING: IMG=$IMG with the launcher's default policy"

# --- 3. boot and prove what is running -----------------------------------------------------------
STOP=1 bash "$L" >/dev/null 2>&1 || true
IMG="$IMG" bash "$L" >> "$R/audit.log" 2>&1 || { log "BOOT FAILED"; finish ABORTED; exit 1; }
curl -sf -m 8 "$API/model" >/dev/null || { log "no server"; finish ABORTED; exit 1; }
log "served: $(curl -s -m 5 "$API/model" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')"
log "policy line in the served config: $(sudo grep -c 'draft_num_tokens_by_batch' /srv/qwen5090/flashnext-config.yml)"
sudo docker exec flashnext python3 - <<'PY' >> "$R/audit.log" 2>&1
import exllamav3, inspect, pathlib
p = pathlib.Path(inspect.getfile(exllamav3)).parent
gen = (p / "generator" / "generator.py").read_text()
attn = (p / "modules" / "attention_fn" / "bc_attn.py").read_text()
ls = (p / "model" / "model_ls.py").read_text()
job = (p / "generator" / "job.py").read_text()
print("markers: draft_depth", "def _get_draft_depth(self, batch_size: int) -> int:" in gen,
      "| qsa_multijob", "BC-attn QSA slot layer" in attn,
      "| pr337_device_ctx", "_LSDeviceContext" in ls,
      "| r338_requeue_fix", '"rq_new_tokens": self.rq_new_tokens + self.new_tokens' in job)
PY

# --- the strongest check: output must equal the fingerprint every earlier run produced -------------
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
sys.stdout.write("R:" + "".join(r) + "\nC:" + "".join(c))' > "$R/greedy-enabled.txt"
GOT=$(shasum -a 256 "$R/greedy-enabled.txt" | cut -c1-16)
log "greedy fingerprint: $GOT (canonical $CANON_SHA, $(wc -c < "$R/greedy-enabled.txt") bytes)"
[ "$GOT" = "$CANON_SHA" ] && log "PASS: the enabled configuration is byte-identical to every configuration measured before" \
                          || log "FAIL: output differs from the canonical fingerprint — investigate before trusting the enablement"

# --- 4. the workload still works ------------------------------------------------------------------
log "=== 4. the original failing agent request + the retrieval gate, on the ENABLED configuration ==="
curl -sN -m 1800 "$API/chat/completions" -H 'Content-Type: application/json' \
  --data-binary @/srv/qwen5090/r356-replay.json > "$R/replay.sse" 2>&1
python3 - "$R/replay.sse" <<'PY' | tee -a "$R/audit.log"
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
        r += d.get("reasoning_content") or d.get("reasoning") or ""
        c += d.get("content") or ""
        if d.get("tool_calls"):
            for call in d["tool_calls"]:
                fn = call.get("function") or {}
                if fn.get("name"): tools.append(fn["name"])
        if ch.get("finish_reason"): finish = ch["finish_reason"]
print(f"  agent request: reasoning {len(r)} chars, content {len(c)} chars, finish_reason={finish}, tools={tools}")
PY
python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$MODEL" --tag enabled-needle \
   --ctx-tokens 131072 --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved" | tee -a "$R/audit.log"

log "left serving: $IMG with DRAFT_POLICY='[[2, 3], [8, 1]]'"
finish DONE
