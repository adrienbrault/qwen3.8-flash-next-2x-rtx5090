#!/usr/bin/env bash
# R523 — TabbyAPI tool_choice "required"/named enforcement (Opus round tool-choice r1, patches/tabbyapi/tool-choice/r1).
# Image tabbyapi:stack-r4-e3r2-tc1 = daily image + Python overlay: for required/named requests a Lark grammar (tags by token
# id) switches on at THINK_END through the existing filter_trigger; a bounded second job covers reasoning that never ends.
# auto/none untouched. Today TC-45 "tool_choice=required Compliance" fails 12/12. Spec box-ab-spec.md, cut to the 15-min
# timebox: in-container tests with the real tokenizer (0 skipped) → A (daily image) identity requests → B: box_probe matrix
# (identity, required 48, named 4, validation 4, cap 2, concurrent 8), A/B identity compare, fingerprints, TC-45 x12. The
# full 69x4 tool-eval regression gate runs only if everything here is green (a later unit). Daily EXTRA_ENV unchanged.
# RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=${HOME:?} PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r523-tool-choice; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/tool-choice-r1
IMG=tabbyapi:stack-r4-e3r2-tc1
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r523] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
finish(){ sudo docker rm -f r523-tests >/dev/null 2>&1 || true
  if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R523 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$SRC/Dockerfile.box" "$SRC/box_probe.py"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE"); [ "$LIMG" = tabbyapi:stack-r4-e3r2 ] || { log "ABORT: live image '$LIMG' is not the overlay base"; exit 3; }
log "building $IMG (before the lock)"
(cd "$SRC" && sudo docker build -t "$IMG" -f Dockerfile.box . ) > "$R/build.log" 2>&1 || { log "ABORT: build failed: $(tail -3 "$R/build.log" | tr '\n' ' ' | cut -c1-240)"; exit 3; }
grep -a "overlay verified\|llguidance\|^OK" "$R/build.log" | cut -c1-160 | tee -a "$R/audit.log"
export GPU_QUEUE_NAME=r523-tool-choice
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 900 ))
log "lock held; timebox ends $(date -Is -d @$END)"
BOOTED=1
sudo docker stop -t 30 flashnext >/dev/null 2>&1; sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
log "step 1: in-container tests with the Flash-Next tokenizer"
timeout 240 sudo docker run --rm --name r523-tests --gpus all --entrypoint python3 -w /app \
  -v /srv/qwen5090/models/$MODEL:/model:ro -e TABBY_TOOLCHOICE_TOKENIZER=/model/tokenizer.json -e TABBY_TOOLCHOICE_REQUIRE_LLG=1 \
  "$IMG" -m unittest -v tests.test_tool_choice tests.test_tool_choice_collector > "$R/tests.log" 2>&1; rc=$?
log "tests rc=$rc: $(tail -3 "$R/tests.log" | tr '\n' ' ' | cut -c1-200)"
[ $rc = 0 ] && ! grep -aq "skipped=[1-9]" "$R/tests.log" || { log "step 1 FAIL (rc $rc or skips)"; finish FAILED; exit 1; }
boot(){ local tag=$1 img=$2 i
  sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
  "${CLEAN_ENV[@]}" IMG="$img" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 || { log "NO BOOT $tag"; return 1; }
  for i in $(seq 120); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
  [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.Config.Image}}' flashnext)" = "$img" ] || { log "NO BOOT $tag"; return 1; }
  log "UP $tag ($img)"; }
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
boot A tabbyapi:stack-r4-e3r2 || { finish ABORTED; exit 3; }
timeout 240 python3 "$SRC/box_probe.py" --url http://127.0.0.1:8022 --out "$R/A.json" --identity-only 2>&1 | tail -4 | tee -a "$R/audit.log"
boot B "$IMG" || { finish ABORTED; exit 3; }
log "[B] c1 fingerprint $(greedy B) (canonical ae890c45d1000582)"
timeout $(( END - $(date +%s) - 150 > 60 ? END - $(date +%s) - 150 : 60 )) python3 "$SRC/box_probe.py" --url http://127.0.0.1:8022 --out "$R/B.json" --trials 12 > "$R/probe-B.log" 2>&1
tail -15 "$R/probe-B.log" | cut -c1-200 | tee -a "$R/audit.log"
python3 "$SRC/box_probe.py" --compare "$R/A.json" "$R/B.json" 2>&1 | tail -6 | tee -a "$R/audit.log"
if [ $(( END - $(date +%s) )) -ge 100 ]; then
  ( cd "$HOME" && timeout $(( END - $(date +%s) )) tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$MODEL" --temperature 0.6 --top-p 0.95 --top-k 20 \
      --scenarios TC-45 --trials 12 --parallel 4 --json-file "$R/tc45.json" > "$R/tc45.log" 2>&1 )
  python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tc45.json" TC45 2>&1 | tail -4 | tee -a "$R/audit.log"
else log "SKIP TC-45: timebox"; fi
sudo docker logs flashnext > "$R/docker-B.log" 2>&1
log "B log: ToolChoiceNotHonoured $(grep -ac ToolChoiceNotHonoured "$R/docker-B.log"), grammar skipped $(grep -ac 'Skipping because the grammar' "$R/docker-B.log"), phase-2 jobs $(grep -ac 'tool_choice phase 2' "$R/docker-B.log"), Traceback $(grep -ac Traceback "$R/docker-B.log")"
grep -a "reasoning reached\|tool_choice phase 2" "$R/docker-B.log" | head -4 | cut -c1-220 | tee -a "$R/audit.log"
finish DONE
