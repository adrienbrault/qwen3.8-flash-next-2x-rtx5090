#!/usr/bin/env bash
# R509 — GSM8K n=500 on 3.05bpw and 2.50bpw with lm-eval's stop strings removed (probes/nostop_proxy.py).
# WHY (R503, 2026-09-18): 36 of 3.05bpw's 43 and 89 of 2.50bpw's 98 wrong answers were EMPTY. Replaying doc 8 on the daily:
# the model restates the problem as "Question: ..." inside its reasoning, TabbyAPI applies lm-eval's server-side stop string
# "Question:" to the reasoning, and the reply ends mid-thought with content null (the reasoning already had the right answer).
# So R503's 0.914 vs 0.804 and every earlier Flash-Next GSM8K gate measured how often the model writes "Question:" while
# thinking. Same image / launcher / pools / flags / greedy / max_gen_toks 8192 / samples as R503; only the stop field is
# dropped by a local proxy (content never contains the few-shot continuation, so scoring is otherwise unchanged).
# RUN: sudo systemd-run --unit=r509-gsm8k-nostop --collect -p RuntimeMaxSec=172800 -E HOME=$HOME /bin/bash /srv/qwen5090/r509-gsm8k-nostop.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r509-gsm8k-nostop; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
PROXY=http://127.0.0.1:8031/v1
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r498.sh
CTLM=qwen3.8-flash-next-exl3-3.05bpw; NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab; MODEL=$CTLM
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r509] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
finish(){ [ -n "${PXP:-}" ] && kill $PXP 2>/dev/null; if [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    MODEL=$CTLM; for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R509 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$LM" /srv/qwen5090/probes/nostop_proxy.py /srv/qwen5090/models/$NEWM/config.json; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r509-gsm8k-nostop
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
python3 /srv/qwen5090/probes/nostop_proxy.py --listen 127.0.0.1:8031 --upstream http://127.0.0.1:8022 > "$R/proxy.log" 2>&1 & PXP=$!
sleep 2; kill -0 $PXP 2>/dev/null || { log "ABORT: proxy did not start"; finish ABORTED; exit 3; }
boot(){ local ck=$1 cache=$2 i; BOOTED=1; MODEL=$ck; log "boot $ck @ $cache"
  sudo docker rm -f flashnext >/dev/null 2>&1; sleep 3
  "${CLEAN_ENV[@]}" CKPT_NAME=$ck CACHE=$cache bash "$CAND" >> "$R/boot.log" 2>&1
  for i in $(seq 240); do [ "$(served_id)" = "$ck" ] && { log "UP $ck @ $cache"; return 0; }; sleep 2; done; return 1; }
gsm(){ local t=$1
  timeout 14400 "$LM" --model local-chat-completions \
    --model_args "base_url=$PROXY/chat/completions,model=$MODEL,tokenizer=/srv/qwen5090/models/$CTLM,num_concurrent=4,max_retries=1,tokenized_requests=False" \
    --tasks gsm8k --num_fewshot 5 --limit 500 --apply_chat_template --gen_kwargs temperature=0,max_gen_toks=8192 --log_samples \
    --output_path "$R/ev-gsm8k-$t" > "$R/ev-gsm8k-$t.log" 2>&1
  python3 - "$R/ev-gsm8k-$t" "$t" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json, pathlib, sys, statistics
d = pathlib.Path(sys.argv[1]); t = sys.argv[2]
res = json.loads(next(d.rglob("results_*.json")).read_text())["results"]["gsm8k"]
print(f"[{t}] GSM8K n=500: flexible {res['exact_match,flexible-extract']:.3f} (SE {res['exact_match_stderr,flexible-extract']:.4f}) strict {res['exact_match,strict-match']:.3f}")
rs = [json.loads(l) for f in d.rglob("samples_gsm8k*.jsonl") for l in open(f)]
rs = [r for r in rs if r.get("filter", "flexible-extract") == "flexible-extract"]  # one row per doc (R503 counted both filters)
def resp(r):
    x = r["resps"][0]; return x[0] if isinstance(x, list) else x
wrong = [r for r in rs if not r.get("exact_match,flexible-extract", 0) and not r.get("exact_match", 0)] if rs and "exact_match,flexible-extract" not in rs[0] else [r for r in rs if not r.get("exact_match,flexible-extract", r.get("exact_match", 0))]
L = [len(resp(r)) for r in wrong]; allL = [len(resp(r)) for r in rs]
trunc = sum(1 for r in wrong if len(resp(r)) > 20000); empty = sum(1 for r in wrong if len(resp(r).strip()) < 5)
print(f"[{t}] samples {len(rs)}, wrong {len(wrong)}: truncated-like (>20k chars) {trunc}, empty {empty}, other {len(wrong)-trunc-empty}; "
      f"resp chars median all {statistics.median(allL) if allL else 0:.0f} / wrong {statistics.median(L) if L else 0:.0f}")
json.dump([{"doc_id": r.get("doc_id"), "target": r.get("target"), "resp_tail": resp(r)[-400:], "chars": len(resp(r))} for r in wrong],
          open(f"{sys.argv[1]}-wrong.json", "w"), indent=1)
PY
}
boot $CTLM 360448 || { finish ABORTED; exit 3; }; gsm C305
boot $NEWM 786432 || { finish ABORTED; exit 3; }; gsm N250
python3 - "$R" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys
r=sys.argv[1]; a={x["doc_id"] for x in json.load(open(f"{r}/ev-gsm8k-C305-wrong.json"))}; b={x["doc_id"] for x in json.load(open(f"{r}/ev-gsm8k-N250-wrong.json"))}
print(f"paired: wrong on both {len(a&b)}, only 3.05 {len(a-b)}, only 2.50 {len(b-a)}")
PY
finish DONE
