#!/usr/bin/env bash
# R414 — promotion gates for the MAX_BSZN candidate (default: tabbyapi:qsa-cid-pr337-bszn16 with policy [[4,3],[8,1]],
# the best R412 combination: c1 203 / c4 375 / c8 443). The c1 path is byte-identical to the served image (R412); what
# changed is every round with 9..16 flattened rows (c3-c4 at depth 3, c5-c8 at depth 1), which now run the fused decode
# tier instead of reconstruct -- FP16 MMA partials folded to FP32, so bitwise identity at c8 is not expected and the
# gates must be quality gates under concurrency:
#   1. c1 greedy fingerprint == the canonical 750e1459e177c47e (r363 request, 512 tokens)      [identity, unchanged path]
#   2. the original failing agent request returns parsed tool_calls (r356 replay)               [workload]
#   3. needle 5/5 at 131k, five positions (fn_needle_oai)                                        [long context]
#   4. GSM8K n=200, R355 parameters, num_concurrent=8                                           [quality on the fused path]
#   5. tool-eval 69x4, sampler 0.6/0.95/20, --parallel 8 (r357 invocation; served read 85.8 +- 3.1)  [quality under c8]
# Runs as the login user (tool-eval-bench lives in ~/.local/bin), like r357. Restores the daily at the end.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
IMG=${IMG:-tabbyapi:qsa-cid-pr337-bszn16}
POLICY=${POLICY:-[[4, 3], [8, 1]]}
TAG=${TAG:-bszn16-4-3-8-1}
R=/srv/qwen5090/results/2026-09-17-r414-gates-$TAG; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-3.05bpw
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw
API=http://127.0.0.1:8022/v1
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
CANON_SHA=750e1459e177c47e
BOOTED=0
log(){ echo "$(date -Is) [r414] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
wait_id(){ local want=$1 i st; for i in $(seq 240); do [ "$(served_id)" = "$want" ] && return 0
  st=$(cstatus); case "$st" in Restarting*|Exited*) log "container $st"; sudo docker logs --tail 8 flashnext 2>&1 | grep -aE "Error|error|Value|Input should" | tail -3 | tee -a "$R/audit.log"; return 1;; esac; sleep 2; done; return 1; }
finish(){ if [ "$BOOTED" = 1 ]; then log "restoring the daily"; bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    wait_id "$MODEL" && log "RESTORED: serving $MODEL on $(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" || log "RESTORE UNVERIFIED: $(served_id)"; fi; log "=== R414 $TAG $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$LM" "$CKPT/config.json" /srv/qwen5090/r356-replay.json /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/tooleval_summary.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
command -v tool-eval-bench >/dev/null 2>&1 || { log "ABORT: tool-eval-bench not on PATH ($PATH)"; exit 3; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
GPU_QUEUE_NAME=r414-gates-$TAG
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id); candidate $IMG policy $POLICY"
BOOTED=1
IMG="$IMG" DRAFT_POLICY="$POLICY" bash "$LIVE" >> "$R/audit.log" 2>&1 || { log "BOOT FAILED"; finish ABORTED; exit 3; }
wait_id "$MODEL" || { log "BOOT UNVERIFIED"; finish ABORTED; exit 3; }
sudo grep -qF "draft_num_tokens_by_batch: $POLICY" /srv/qwen5090/flashnext-config.yml || { log "ABORT: generated policy is not $POLICY"; finish ABORTED; exit 3; }
[ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = "$IMG" ] || { log "ABORT: served image is not $IMG"; finish ABORTED; exit 3; }
log "up: $(served_id) on $IMG, policy $POLICY"

log "=== gate 1: canonical c1 greedy fingerprint ==="
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
sys.stdout.write("R:" + "".join(r) + "\nC:" + "".join(c))' > "$R/greedy-c1.txt"
GOT=$(shasum -a 256 "$R/greedy-c1.txt" | cut -c1-16)
[ "$GOT" = "$CANON_SHA" ] && log "GATE 1 PASS: c1 greedy fingerprint $GOT == canonical" || log "GATE 1 FAIL: c1 greedy fingerprint $GOT != canonical $CANON_SHA ($(wc -c < "$R/greedy-c1.txt") bytes)"

log "=== gate 2: the original failing agent request ==="
curl -sN -m 1800 "$API/chat/completions" -H 'Content-Type: application/json' --data-binary @/srv/qwen5090/r356-replay.json > "$R/replay.sse" 2>&1
python3 - "$R/replay.sse" <<'PY' 2>&1 | tee -a "$R/audit.log"
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
print(f"GATE 2 {'PASS' if tools and finish == 'tool_calls' else 'FAIL'}: agent request reasoning {len(r)} chars, content {len(c)} chars, finish_reason={finish}, tools={tools}")
PY

log "=== gate 3: needle 131k, five positions ==="
python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$MODEL" --tag "$TAG-needle" \
   --ctx-tokens 131072 --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee -a "$R/audit.log"

log "=== gate 4: GSM8K n=200, R355 parameters, num_concurrent=8 ==="
timeout 10800 "$LM" --model local-chat-completions \
  --model_args "base_url=$API/chat/completions,model=$MODEL,tokenizer=$CKPT,num_concurrent=8,max_retries=1,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 5 --limit 200 --apply_chat_template --gen_kwargs temperature=0,max_gen_toks=8192 --output_path "$R/ev-gsm8k" > "$R/ev-gsm8k.log" 2>&1
log "lm_eval exit=$?"
python3 - "$R/ev-gsm8k" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json, pathlib, sys
files = list(pathlib.Path(sys.argv[1]).rglob("results_*.json"))
if len(files) != 1: raise SystemExit(f"GATE 4 FAIL: expected one result file, got {len(files)}")
d = json.loads(files[0].read_text()); s = d["results"]["gsm8k"]
if d.get("n-samples", {}).get("gsm8k", {}).get("effective") != 200: raise SystemExit("GATE 4 FAIL: not n=200")
v = s['exact_match,flexible-extract']; e = s['exact_match_stderr,flexible-extract']
print(f"GATE 4 {'PASS' if v >= 0.935 - 2*0.0175 else 'FAIL'}: GSM8K c8 flexible {v:.3f} (SE {e:.4f}) strict {s['exact_match,strict-match']:.3f}   [served control r396: 0.935 (SE 0.0175); pass = within 2 SE]")
PY

log "=== gate 5: tool-eval 69x4, sampler 0.6/0.95/20, parallel 8 ==="
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$MODEL" \
    --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 \
    --json-file "$R/tooleval.json" > "$R/tooleval.log" 2>&1 ); log "tool-eval harness exit=$?"
if [ -s "$R/tooleval.json" ]; then python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval.json" "$TAG" 2>&1 | tee -a "$R/audit.log"; log "GATE 5: compare with the served 85.8 +- 3.1 (r357)"; else log "GATE 5 FAIL: no JSON; $(tail -2 "$R/tooleval.log" | cut -c1-160)"; fi
finish DONE
