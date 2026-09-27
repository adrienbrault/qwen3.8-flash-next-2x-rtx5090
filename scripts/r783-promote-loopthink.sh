#!/usr/bin/env bash
# R783 (2026-09-27): promote tabbyapi:stack-r3-rows32-tokcount-loopthink3 (the served daily image + loop-think r3,
# TabbyAPI only) as the Flash-Next daily on :8022. A loop inside the thinking used to end the request as
# finish_reason "stop" with no content, which Hermes shows as the reply (the daily did it 2026-09-26 in two turns);
# r3 forces </think> after W=800 looping tokens instead. Same promotion class as R747: the change is in the chat
# endpoint only and leaves non-looping output bit-identical, so the gates are identity + mechanism + headroom.
#   1. registers in the GPU queue and takes the GPU-exclusive lock: it waits behind R782 (a finetune on :8029),
#      whose finish_restore then skips the daily because this unit is queued;
#   2. boots the CURRENT daily (live launcher, unchanged) and records chat greedy (bench/chat_greedy.py, tag OLD):
#      fn_greedy.py covers /v1/completions only, which never enters the patched collector;
#   3. installs the staged launcher (DAILY_IMG flipped, md5-checked; rollback launch-flashnext.sh.pre-r783) and boots it;
#   4. gates (all must hold, else the old launcher goes back and the old daily is booted):
#      G1 serves the loopthink3 image, 41 env keys, pool 983,040, UP free per card >= 1125-32 / 1573-32 MiB;
#      G2 fn_greedy (6 prompts incl. ~100k) identical to R747's served reference (tag P there);
#      G3 chat greedy (5 prompts, thinking on, a tool call, two turns) identical to step 2 (tag NEW vs OLD);
#      G4 loop mechanism on the daily template (R782 review): forced x2 -> content "391" and 2 collector injections;
#         nothink x2 -> finish stop at exactly 800 completion tokens (engine (W, 2), no injection);
#         p107 x2 (a Hermes agent turn, thinking prefilled with 2 copies of the finetune's real looping paragraph)
#         -> tool call or content;
#      G5 OOM / traceback / restart count 0.
#   5. points Hermes at :8022 (on promotion and on rollback).
# The Olla gateway is drained for the unit's lifetime (lib/gateway-drain.sh): the greedy gates need c1 with no foreign
# requests in the batch (R782 review: a concurrent warm-up request changed a T=0 answer).
# Staging: scripts/launch-flashnext.sh -> /srv/qwen5090/launch-flashnext.sh.r783 ; probes bench/chat_greedy.py,
#   bench/replay_hermes_turn.py and the prefix builder loop_prefixes.py ; R782's prefix-p107.txt and the R781 control
#   snapshot (a private Hermes session, not published) are the inputs. loop_prefixes.py and hermes-tools-r779.json read
#   that private session and are not in this repository; without them the p107 probe cannot be reproduced.
# Launch: sudo systemd-run --unit=r783-promote-loopthink --collect -p RuntimeMaxSec=86400 -p Environment=HOME=$HOME -E EXPECT_NEW=<md5> /usr/bin/bash /srv/qwen5090/r783-promote-loopthink.sh
set -uo pipefail
UNIT=r783-promote-loopthink
D=/srv/qwen5090
LIVE=$D/launch-flashnext.sh
NEW=$D/launch-flashnext.sh.r783
EXPECT_OLD=e0ea5c22b428f075cfee372faff188f4          # R747's launcher
EXPECT_NEW=${EXPECT_NEW:?set EXPECT_NEW to the md5 of the staged launcher}
IMG_OLD=tabbyapi:stack-r3-rows32-tokcount
IMG_NEW=tabbyapi:stack-r3-rows32-tokcount-loopthink3
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
API=http://127.0.0.1:8022/v1
REF_GREEDY=$D/results/2026-09-26-r747-promote-tokcount-1136/greedy.jsonl
PR=$D/probes
CTL=$D/results/2026-09-27-r781-control
P107=$(ls -d $D/results/2026-09-27-r782-loopthink2-*/ 2>/dev/null | tail -1)prefix-p107.txt
HSET=$D/hermes-set-model.sh
R=$D/results/$(date +%F)-$UNIT-$(date +%H%M); mkdir -p "$R"
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$UNIT
. $D/lib/gpu-queue.sh
. $D/lib/serve-ctl.sh
. $D/lib/gateway-drain.sh
SCTL_LOG="$R/audit.log"
cp "$0" "$R/"
for f in "$NEW" "$REF_GREEDY" "$PR/fn_greedy.py" "$PR/chat_greedy.py" "$PR/replay_hermes_turn.py" "$P107" "$CTL/session.json" "$CTL/state.db" "$HSET"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 2; }; done
[ "$(md5sum < "$NEW" | cut -c1-32)" = "$EXPECT_NEW" ] || { log "ABORT: staged launcher md5 != $EXPECT_NEW"; exit 2; }
bash -n "$NEW" || { log "ABORT: staged launcher does not parse"; exit 2; }
[ "$(grep -oE '^DAILY_IMG=\S+' "$NEW" | cut -d= -f2)" = "$IMG_NEW" ] || { log "ABORT: staged launcher DAILY_IMG is not $IMG_NEW"; exit 2; }
diff <(grep -v '^DAILY_IMG=' "$LIVE" | grep -vE '^\s*#') <(grep -v '^DAILY_IMG=' "$NEW" | grep -vE '^\s*#') > "$R/launcher-code-diff.txt" \
  || { log "ABORT: staged launcher differs from the live one in more than DAILY_IMG and comments (launcher-code-diff.txt)"; exit 2; }
sudo docker image inspect "$IMG_NEW" --format '{{json .Config.Labels}}' 2>/dev/null | grep -q '"local.loopthink.round":"r3"' \
  || { log "ABORT: $IMG_NEW missing or not loop-think r3"; exit 2; }
[ "$(grep -c '"tag": "P"' "$REF_GREEDY")" = 6 ] || { log "ABORT: R747 reference does not have 6 tag-P rows"; exit 2; }

log "queued; waiting for the GPU-exclusive lock (R782 serves the finetune until it is stopped)"
gpu_lock
log "lock held; served $(served_id); VRAM free $(vram_free)"
gateway_drain   # the greedy identity gates run on the live :8022 port; Olla routes nothing to it until this unit exits
cur=$(md5sum < "$LIVE" | cut -c1-32)
[ "$cur" = "$EXPECT_OLD" ] || { log "ABORT: live launcher md5 $cur != $EXPECT_OLD; nothing installed"; exit 3; }
hermes_daily(){ PORT=8022 MODEL_ID=$MODEL bash "$HSET" >> "$R/audit.log" 2>&1 && log "Hermes on :8022" || log "WARN: Hermes not moved"; }
boot(){ served_stop; wait_unserved 45 || true
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" > "$R/boot-$1.log" 2>&1 \
    && wait_served_id "$MODEL" 240 10; }

# ---- the current daily: chat greedy reference ----
boot old || { log "ABORT: the current daily did not boot"; hermes_daily; exit 3; }
log "old daily up: $(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'), VRAM free $(vram_free)"
gateway_wait_idle 900 || log "WARN: gateway not idle after 900 s"
python3 $PR/chat_greedy.py --url $API --model $MODEL --tag OLD --out "$R/chat-greedy.jsonl" > "$R/chat-greedy-OLD.log" 2>&1 \
  || { log "ABORT: chat greedy on the old daily failed"; hermes_daily; exit 3; }

# ---- install and boot the candidate ----
cp -p "$LIVE" $D/launch-flashnext.sh.pre-r783
cp -p "$NEW" "$LIVE.new" && mv "$LIVE.new" "$LIVE"
log "launcher installed: $cur -> $(md5sum < "$LIVE" | cut -c1-32) (rollback launch-flashnext.sh.pre-r783)"
rollback(){ log "ROLLBACK: $1"; cp -p $D/launch-flashnext.sh.pre-r783 "$LIVE.new" && mv "$LIVE.new" "$LIVE"
  sudo docker logs flashnext > "$R/container-failed.log" 2>&1
  boot rollback; log "rolled back: serving $(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}')"
  hermes_daily; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== $UNIT ROLLED BACK ==="; exit 1; }
trap 'rollback "signal"' TERM INT HUP
boot new || rollback "the new daily did not boot"
generation_state "$MODEL" | grep -q GEN_SANE || rollback "generation not sane"

got=$(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}')
keys=$(grep -aoE 'env keys \([0-9]+\)' "$R/boot-new.log" | tail -1 | tr -dc '0-9')
pool=$(grep -aoE 'cache [0-9]+ @' "$R/boot-new.log" | tail -1 | tr -dc '0-9')
free=$(grep -aoE 'VRAM free MiB [0-9/]+' "$R/boot-new.log" | tail -1 | awk '{print $4}'); free=${free%/}
f0=${free%/*}; f1=${free#*/}
log "G1: image $got, keys $keys, pool $pool, UP free $free (REF 1125/1573, floor 1093/1541)"
[ "$got" = "$IMG_NEW" ] && [ "$keys" = 41 ] && [ "$pool" = 983040 ] && [ "${f0:-0}" -ge 1093 ] && [ "${f1:-0}" -ge 1541 ] || rollback "G1 failed"

grep '"tag": "P"' "$REF_GREEDY" | sed 's/"tag": "P"/"tag": "A"/' > "$R/greedy.jsonl"
python3 $PR/fn_greedy.py --url http://127.0.0.1:8022 --tag P --out "$R/greedy.jsonl" > "$R/greedy-P.log" 2>&1
python3 $PR/fn_greedy.py --compare --ref A --out "$R/greedy.jsonl" > "$R/greedy-compare.txt" 2>&1
grep -a '^GREEDY' "$R/greedy-compare.txt" | tee -a "$R/audit.log"
grep -aq 'IDENTICAL' "$R/greedy-compare.txt" && ! grep -aqE 'DIVERGE|MISSING' "$R/greedy-compare.txt" || rollback "G2 fn_greedy not identical to R747's served reference"

python3 $PR/chat_greedy.py --url $API --model $MODEL --tag NEW --out "$R/chat-greedy.jsonl" > "$R/chat-greedy-NEW.log" 2>&1
python3 $PR/chat_greedy.py --compare --ref OLD --cand NEW --out "$R/chat-greedy.jsonl" > "$R/chat-greedy-compare.txt" 2>&1
g3=$?; tee -a "$R/audit.log" < "$R/chat-greedy-compare.txt"
[ "$g3" = 0 ] || rollback "G3 chat greedy not identical to the old daily"

RP="python3 $PR/replay_hermes_turn.py --url $API --model $MODEL"
python3 $PR/loop_prefixes.py "$CTL/session.json" "$R" "$API" >> "$R/audit.log" 2>&1   # nothink prefix (p107 comes from R782)
$RP --forced-loop --n 2 --conc 2 --max-tokens 12000 --tag forced --out "$R/forced.jsonl" > "$R/forced.txt" 2>&1
$RP --forced-loop --no-think --prompt "Write the line 'the quick brown fox jumps over the lazy dog' 400 times, one per line." \
    --prefix-file "$R/prefix-nothink.txt" --n 2 --conc 2 --max-tokens 5000 --tag nothink --out "$R/nothink.jsonl" > "$R/nothink.txt" 2>&1
$RP --session $CTL/session.json --db $CTL/state.db --upto 16 --tools $PR/hermes-tools-r779.json --prefix-file "$P107" \
    --n 2 --conc 2 --tag p107 --out "$R/p107.jsonl" > "$R/p107.txt" 2>&1
sudo docker logs flashnext > "$R/container.log" 2>&1
inj=$(grep -ac 'reasoning loop detected' "$R/container.log"); eng=$(grep -ac 'token loop was detected' "$R/container.log")
cat "$R/forced.txt" "$R/nothink.txt" "$R/p107.txt" | grep -aE '^(forced|nothink|p107) #|RESULT' | sed -E 's/ [|] C:.*//' | tee -a "$R/audit.log"
log "G4: collector injections $inj, engine loop stops $eng"
python3 - "$R" "$inj" <<'EOF' || rollback "G4 loop mechanism failed"
import json, sys
R, inj = sys.argv[1], int(sys.argv[2])
rows = lambda t: [json.loads(l) for l in open(f"{R}/{t}.jsonl")]
f, n, p = rows("forced"), rows("nothink"), rows("p107")
assert len(f) == 2 and all(r["class"] == "content" and "391" in r["content"] for r in f), "forced"
assert len(n) == 2 and all(r["finish"] == "stop" and r["completion_tokens"] == 800 for r in n), "nothink"
assert len(p) == 2 and all(r["class"] in ("tool_call", "content") for r in p), "p107"
assert inj >= 2, f"injections {inj}"
print("G4 ok: forced 2/2 391, nothink 2/2 stop@800, p107", [r["class"] for r in p], "injections", inj)
EOF
oom=$(grep -acE 'OutOfMemoryError|out of memory' "$R/container.log"); tb=$(grep -ac Traceback "$R/container.log")
rs=$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo ?)
log "G5: OOM $oom tracebacks $tb restarts $rs"
[ "$oom" = 0 ] && [ "$tb" = 0 ] && [ "$rs" = 0 ] || rollback "G5 failed"

trap - TERM INT HUP
hermes_daily
rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
sudo chown -R "$(stat -c %U $D/results)" "$R" 2>/dev/null || true
log "=== $UNIT PROMOTED: daily = $IMG_NEW (rollback: launch-flashnext.sh.pre-r783, image $IMG_OLD) ==="
