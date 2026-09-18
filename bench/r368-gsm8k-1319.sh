#!/usr/bin/env bash
# R368 — GSM8K at n=1319 as served: the same instrument and parameters as R355, full test split, tighter interval.
#
# WHY. R355 measured the served seat at n=200, which gives ±0.019. The full test split gives ±0.0077 on the same
# configuration, so the figure can be quoted without the interval swallowing the differences of interest.
#
# WHAT DIFFERS FROM R300, ON PURPOSE. R300 booted llama.cpp and removed its think-off flag. Here the engine is
# already running and thinking is on by configuration (`reasoning: true`), so nothing about the route is being
# changed for the measurement: the arm measures the seat as served rather than a variant of it.
#
# No restart, no reboot: it takes the GPU lock so nothing else disturbs the run, and leaves the server alone.
#
# RUN: sudo systemd-run --unit=r368-gsm8k-1319 --collect -p User=adrienbrault -p RuntimeMaxSec=14400 \
#        bash /srv/qwen5090/r368-gsm8k-1319.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r368-gsm8k-1319; mkdir -p "$R"
U=http://127.0.0.1:8022
MODEL=qwen3.8-flash-next-exl3-3.05bpw
TOKENIZER=/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
log(){ echo "$(date -Is) [r355] $*" | tee -a "$R/audit.log"; }

[ -x "$LM" ] || { log "ABORT: lm-eval venv missing at $LM"; exit 3; }
[ -f "$TOKENIZER/chat_template.jinja" ] || { log "ABORT: tokenizer/template missing"; exit 3; }
export GPU_QUEUE_NAME=r355-gsm8k
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R355 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

curl -sf -m 8 "$U/v1/model" >/dev/null || { log "ABORT: no server on $U"; exit 3; }
SERVED=$(curl -s -m 5 "$U/v1/model" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
log "served: $SERVED | image: $(sudo docker inspect flashnext --format '{{.Config.Image}}')"
log "harness: gsm8k, 5-shot, apply_chat_template, temperature 0, max_gen_toks 8192, limit 1319, num_concurrent 4"
log "(this line said num_concurrent 8 until 2026-09-16 while the invocation below asked for 4; the invocation is"
log " authoritative, and a record that disagrees with the measurement is a defect)"
log "(R355 ran these same parameters at n=200; this run is n=1319 for a tighter interval on this seat's own figure)"
# lm-eval reads `message.content`, so this measurement is only meaningful if the model puts the ANSWER there and
# keeps the thinking in reasoning_content. Checked against the live server on this configuration before trusting the
# run: a GSM8K item came back with 126 chars of reasoning and 131 chars of content ending "Answer: 72 clips" --
# the right answer in the right field, and the brief reasoning the CONFIG notes describe.

# Thinking-on costs generations that run to the 8192 budget, and c4 is what the harness asks for. A timeout here
# is a measurement that did not finish, not a score.
timeout 10800 "$LM" --model local-chat-completions \
  --model_args "base_url=$U/v1/chat/completions,model=$MODEL,tokenizer=$TOKENIZER,num_concurrent=4,max_retries=1,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 5 --limit 1319 --apply_chat_template \
  --gen_kwargs temperature=0,max_gen_toks=8192 --output_path "$R/ev-gsm8k" > "$R/ev-gsm8k.log" 2>&1
rc=$?
log "lm_eval exit code: $rc"

grep -arhoE '"exact_match[^"]*": *[0-9.]+' "$R/ev-gsm8k" 2>/dev/null | sort -u | sed 's/^/  [gsm8k] /' | tee -a "$R/audit.log"
grep -aiE "error|exception|timeout" "$R/ev-gsm8k.log" 2>/dev/null | tail -5 | cut -c1-160 | tee -a "$R/audit.log"

# Did thinking actually happen? A score from an arm that never thought is a different measurement.
python3 - <<'PY' | tee -a "$R/audit.log"
import json, glob, statistics
samples = []
for path in glob.glob("/srv/qwen5090/results/2026-09-16-r368-gsm8k-1319/ev-gsm8k/**/samples_gsm8k_*.jsonl", recursive=True):
    for line in open(path):
        try: samples.append(json.loads(line))
        except Exception: pass
if not samples:
    print("  no samples captured to inspect")
else:
    lens = [len(s.get("resps", [[""]])[0][0] or "") for s in samples]
    print(f"  samples={len(samples)} answer chars median={int(statistics.median(lens))} "
          f"max={max(lens)} (a thinking run should be well above the ~40 chars of a bare answer)")
PY
finish DONE
