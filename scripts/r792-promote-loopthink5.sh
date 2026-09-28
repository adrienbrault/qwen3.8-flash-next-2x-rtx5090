#!/usr/bin/env bash
# R792 (2026-09-28): promote TabbyAPI loop-think r5 on the Flash-Next daily on :8022 (gate and serve r5). The
# candidate launcher is the live one (R789, 39 keys) with ONE code change: DAILY_IMG tabbyapi:rebase-dev-r3 ->
# tabbyapi:rebase-dev-r3-loopthink5 (= rebase-dev-r3 + docker/overlays/loop-think-r5/r4-to-r5.patch, TabbyAPI /app only):
#   - long-period reasoning-loop rungs LoopDetector(6000, 2000) + (12000, 4000) beside r4's (3000, 1000) (R791: exact loops
#     with 1,199- and 3,713-token periods ran to max_tokens);
#   - finish_reason "length" (was "tool_calls") when max_tokens cuts a tool call that did not parse (R791 u10 #2).
# Engine, the 39 keys, pool 901,120, split [30, 30] and TUNEDIR unchanged; the NVMe tier stays on (the launcher's tier
# default tests IMG against DAILY_IMG, which now names the new image).
# Expected: greedy byte-identical to R785c's reference (fn_greedy 6/6 + chat_greedy 6/6 incl. --long): the loop watch only
# injects on a detected loop and the finish change only touches a cut tool call. Autonomous promotion on all gates PASS.
# Derived from r789-promote-dropkeys.sh (PROMOTED 2026-09-28). Changes vs R789:
#   - structural guard: template = live outside comments except the DAILY_IMG line (live rebase-dev-r3, template
#     rebase-dev-r3-loopthink5); EXTRA_ENV identical, 39 keys;
#   - preflight: new image id pinned, its labels local.loopthink.base_id = rebase-dev-r3's id and local.loopthink.round = r5,
#     the rollback image rebase-dev-r3 still present at its id, the r5 test file + fixture installed;
#   - G1: 39 keys (no removed-key checks), plus landing_loopthink.py run inside the served container (env -u EXL3_NVME_TIER,
#     no GPU visible) must pass;
#   - G5: R789's G5 unchanged (pinned 0.6, same bars), plus (a) GATED: the repo's test_loop_think.py + fixtures run inside
#     the served container -> FAILS 0; and three REPORT-ONLY arms after the G5 decision, on their own log window: (b) the
#     forced-loop probe n 8 at the served fallback (no temperature; R789 review), (c) the ladder on R781's turn 16 with a
#     prefix of the R791 u16 row's reasoning cut where its 3rd loop copy starts, 0.6, n 4, 16k, (d) finish_reason of a
#     tool call cut by max_tokens 300 (thinking off);
#   - AFTER empty by default (nothing queued ahead).
# Gates, in order G1 G2 G3 (+ one 120k cold prefill) G5 G6 G4 G7 G8 G9 G10 (bars as R789); any miss -> .pre-r792 goes back,
# the R789 daily boots, DECISION: ROLLED BACK (<gate>).
# REFERENCE: unchanged. R785c's greedy.jsonl / chat-greedy.jsonl stay the daily's reference.
# UNIT HYGIENE: preflight before the queue and again under the lock; every post-lock abort leaves the daily serving from
#   $LIVE on :8022 (nothing else restores :8022). summary.txt ends with exactly one of: DECISION: PROMOTED |
#   DECISION: ROLLED BACK (<gate>) | DECISION: NOT PROMOTED (<why>).
# GPU ~125 min (+25 on a GSM8K c4 re-run, +5 on a rollback).
# Install (operator, .new + mv each): scripts/launchers/launch-flashnext-r792-loopthink5.sh -> /srv/qwen5090/launch-flashnext-r792-loopthink5.sh
#   (mode 755); docker/overlays/loop-think-r5/test_loop_think.py -> /srv/qwen5090/loopthink-r5/test_loop_think.py;
#   the private fixtures/r791-rows.json.gz (re-tokenized text; the published one is synthetic, other md5) -> /srv/qwen5090/loopthink-r5/fixtures/r791-rows.json.gz;
#   this file -> /srv/qwen5090/r792-promote-loopthink5.sh.
#   EXPECT_TPL = md5 of the deployed template:
#   sudo systemd-run --unit=r792-promote-loopthink5 --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=1800 \
#     -p Environment=HOME=$HOME -E EXPECT_TPL=<md5> /usr/bin/bash /srv/qwen5090/r792-promote-loopthink5.sh
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
UNIT=r792-promote-loopthink5
D=/srv/qwen5090
R=$D/results/$(date +%F)-$UNIT-$(date +%H%M)
[ -e "$R" ] && R=$R-$(date +%S)-$$
mkdir -p "$R"
SUM=$R/summary.txt; : > "$SUM"
LIVE=$D/launch-flashnext.sh
TPL=$D/launch-flashnext-r792-loopthink5.sh
PRE=$D/launch-flashnext.sh.pre-r792
EXPECT_OLD=${EXPECT_OLD:-e50283721c8b101d899f0a9d87ea0d16}   # R789's launcher (39 keys, fallback temperature 1.0), md5 of the host's copy
EXPECT_TPL=${EXPECT_TPL:?set EXPECT_TPL to the md5 of the deployed template}
IMG=${IMG:-tabbyapi:rebase-dev-r3-loopthink5}
IMG_ID=${IMG_ID:-sha256:9f7b66f022092ad45e5c7361103e9a65269cf24664aa3c15a79e16059e9c743b}   # built FROM rebase-dev-r3 (Dockerfile.box, r4-to-r5.patch)
BASE_IMG=tabbyapi:rebase-dev-r3                           # the served image (R785 / R789) = the rollback launcher's DAILY_IMG
BASE_ID=${BASE_ID:-sha256:30ed33eb60365a3aca89515162739374832c645378a34e5e131da9ef7317f98b}   # R784 P / R785 / R786 N / R788 / R789
TUNE=$D/.exl3cache-rebase-dev-r3
LT5=$D/loopthink-r5                                     # the repo's loop-think r5 test_loop_think.py + fixtures/ (G5a)
LT5_TEST_MD5=${LT5_TEST_MD5:-5ce047a6d4200c8e703e0187fcd33e1a}   # docker/overlays/loop-think-r5/test_loop_think.py (89 checks)
LT5_FIX_MD5=${LT5_FIX_MD5:-8fff3995f684b71c6c1ea671b145786c}     # the private fixture with text; the published docker/overlays/loop-think-r5/fixtures/ file is synthetic and has another md5
LT5_PATCH_SHA=2794fd6b744f0dbd91f59314855367133670a00a084551ebe94693ca36d2f40d   # sha256(fix.patch + r4-to-r5.patch) in the repo: label reported, not gated
NKEYS=39
POOL=901120
SPLITSP="30, 30"
REF_UP=${REF_UP:-1181/1759}                                # R785c G1 (= R784 P1/P2 UP) at 901,120 [30, 30]
REF_DIR=${REF_DIR:-$D/results/2026-09-27-r785-promote-rebase-r3-1223}   # R785c: the daily's greedy reference (tag R785)
REF_TAG=R785
TAG=R792
AFTER=${AFTER-}
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=$D/models/$MODEL
BASEURL=http://127.0.0.1:8022
API=$BASEURL/v1
P=$D/probes
GREEDY=$P/fn_greedy.py
BENCH=$P/fn_bench.py
CHAT=$P/chat_greedy.py
RHT=$P/replay_hermes_turn.py
LPX=$P/loop_prefixes.py
HTOOLS=$P/hermes-tools-r779.json
REPLAY=$P/agent_replay.py
AGG=$P/tabby_log_agg.py
RSTEPS=$P/replay_steps.py
LM=$D/venv-lmeval/bin/lm_eval
CTL=$D/results/2026-09-27-r781-control                    # R781's Hermes session snapshot (R782 / R783 / R785 inputs)
P107=$(ls -d $D/results/2026-09-27-r782-loopthink2-*/ 2>/dev/null | tail -1)prefix-p107.txt
HSET=$D/hermes-set-model.sh
LOOPMSG="I am repeating myself"                           # loop-think r4's LOOP_THINK_MESSAGE (fix.patch), unchanged in r5
R722_W=122.75                                             # R722 W1/W2 per-stream mean (r728)
REPLAY_BAR=$(python3 -c "print(round(0.98 * $R722_W, 2))")
MIN_ACC=0.5
R785_REPLAY="123.7 all / 124.6 acc>=0.5 (3.117 tok/step, 25.20 ms/step); R789 132.3 all (3.269 tok/step, 24.70 ms/step)"   # R785c / R789 G7 through replay_steps.py
TE_BAR=82.0
GSM_BAR=0.970; GSM_RERUN=0.960
PROBE_TO=${PROBE_TO:-1800}
NONCE=$(( $(date +%s) % 100000 ))
SALT_RAMP=$(( (NONCE + 4099) % 100000 )); SALT_S4=$(( (NONCE + 1231) % 100000 )); SALT_S8=$(( (NONCE + 2462) % 100000 ))
SALT_120=$(( (NONCE + 6007) % 100000 ))
CLEAN_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PROMOTED=0; DECIDED=0; HERMES_STOPPED=
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
note(){ log "$*"; echo "$*" >> "$SUM"; }
decide(){ [ "$DECIDED" = 1 ] && return 0; DECIDED=1; echo "DECISION: $1" >> "$SUM"; log "DECISION: $1"; }
iid(){ sudo docker image inspect "$1" --format '{{.Id}}' 2>/dev/null; }
md5of(){ md5sum < "$1" | cut -c1-32; }
envline(){ sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$1" | head -1; }
defval(){ sed -n "s/^$1=\\\${$1:-\\([^}]*\\)}.*/\\1/p" "$2" | head -1; }   # the default of KEY=${KEY:-...}
cp "$0" "$R/" 2>/dev/null
trap 'log "signal before the queue: nothing mutated"; exit 4' TERM INT HUP

# ---------------- preflight (before the queue and again under the lock) ----------------
# structural A B: B = A outside comments except the DAILY_IMG line (A: $BASE_IMG, B: $IMG); the EXTRA_ENV line is inside the
# diff, so it must be identical (39 tokens, 39 distinct keys). Anything else means the live launcher moved since the template
# was cut, and installing the template would revert it.
structural(){ local a=$1 b=$2 ea
  [ "$(grep -c '^EXTRA_ENV=' "$b")" = 1 ] && [ "$(grep -c '^EXTRA_ENV=' "$a")" = 1 ] || { log "structural: want one 'EXTRA_ENV=' line in each of $a and $b"; return 1; }
  [ "$(grep -c '^DAILY_IMG=' "$b")" = 1 ] && [ "$(grep -c '^DAILY_IMG=' "$a")" = 1 ] || { log "structural: want one 'DAILY_IMG=' line in each of $a and $b"; return 1; }
  grep -vE '^[[:space:]]*#' "$a" | grep -vE '^DAILY_IMG=' > "$R/.code-a"
  grep -vE '^[[:space:]]*#' "$b" | grep -vE '^DAILY_IMG=' > "$R/.code-b"
  diff "$R/.code-a" "$R/.code-b" > "$R/launcher-code-diff.txt" || { log "structural: $b differs from $a outside comments and DAILY_IMG (launcher-code-diff.txt)"; return 1; }
  [ "$(grep -oE '^DAILY_IMG=\S+' "$a" | cut -d= -f2)" = "$BASE_IMG" ] || { log "structural: $a DAILY_IMG is not $BASE_IMG"; return 1; }
  [ "$(grep -oE '^DAILY_IMG=\S+' "$b" | cut -d= -f2)" = "$IMG" ] || { log "structural: $b DAILY_IMG is not $IMG"; return 1; }
  ea=$(envline "$b")
  [ "$(echo $ea | wc -w | tr -dc 0-9)" = "$NKEYS" ] && [ "$(for t in $ea; do echo "${t%%=*}"; done | sort -u | wc -l | tr -dc 0-9)" = "$NKEYS" ] \
    || { log "structural: $b EXTRA_ENV is not $NKEYS tokens / $NKEYS distinct keys"; return 1; }
  [ "$(defval CACHE "$b")" = "$POOL" ] || { log "structural: $b CACHE default '$(defval CACHE "$b")' != $POOL"; return 1; }
  [ "$(defval GPU_SPLIT "$b")" = "$SPLITSP" ] || { log "structural: $b GPU_SPLIT default '$(defval GPU_SPLIT "$b")' != '$SPLITSP'"; return 1; }
  [ "$(defval TUNEDIR "$b")" = "$TUNE" ] || { log "structural: $b TUNEDIR default '$(defval TUNEDIR "$b")' != $TUNE"; return 1; }
  return 0; }
refrows(){ python3 -c 'import json,sys
rs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(sum(1 for r in rs if r.get("tag")==sys.argv[2]), len(rs))' "$1" "$2" 2>/dev/null || echo "0 0"; }
preflight(){ local f why= k
  for f in "$LIVE" "$TPL" "$CKPT/config.json" "$LM" "$GREEDY" "$BENCH" "$CHAT" "$RHT" "$LPX" "$HTOOLS" "$REPLAY" "$AGG" "$RSTEPS" \
           "$P/agentic-edit.py" "$P/fn_needle_oai.py" "$P/tooleval_summary.py" "$P/nostop_proxy.py" "$P107" \
           "$CTL/session.json" "$CTL/state.db" "$HSET" "$D/lib/gpu-queue.sh" "$D/lib/serve-ctl.sh" "$D/lib/gateway-drain.sh" \
           "$REF_DIR/greedy.jsonl" "$REF_DIR/chat-greedy.jsonl" "$TUNE" "$LT5/test_loop_think.py" "$LT5/fixtures/r791-rows.json.gz"; do
    [ -e "$f" ] || why="$why missing $f;"; done
  [ -n "$why" ] && { log "preflight:$why"; return 1; }
  command -v tool-eval-bench >/dev/null || { log "preflight: tool-eval-bench not on PATH"; return 1; }
  grep -q 'LADDER_PROD' "$REPLAY" && grep -q 'max_fail_streak' "$REPLAY" || { log "preflight: agent_replay.py lacks the ladder / failure guard"; return 1; }
  grep -q -- '--long' "$CHAT" && grep -q 'tokens_source' "$CHAT" || { log "preflight: chat_greedy.py lacks --long / the usage:null fallback"; return 1; }
  grep -q 'prefix-p700' "$LPX" || { log "preflight: loop_prefixes.py lacks the p700 prefix"; return 1; }
  grep -q 'SCORED per_stream=' "$RSTEPS" || { log "preflight: $RSTEPS is not the R789 replay scorer"; return 1; }
  [ "$(md5of "$LT5/test_loop_think.py")" = "$LT5_TEST_MD5" ] || { log "preflight: $LT5/test_loop_think.py md5 $(md5of "$LT5/test_loop_think.py") != $LT5_TEST_MD5 (the repo's F1 revision, 89 checks)"; return 1; }
  [ "$(md5of "$LT5/fixtures/r791-rows.json.gz")" = "$LT5_FIX_MD5" ] || { log "preflight: $LT5/fixtures/r791-rows.json.gz md5 != $LT5_FIX_MD5"; return 1; }
  grep -q -- '--temperature' "$RHT" || { log "preflight: $RHT lacks --temperature (G5 pins 0.6)"; return 1; }
  [ "$(refrows "$REF_DIR/greedy.jsonl" $REF_TAG)" = "6 6" ] || { log "preflight: $REF_DIR/greedy.jsonl is not 6 rows tagged $REF_TAG ($(refrows "$REF_DIR/greedy.jsonl" $REF_TAG))"; return 1; }
  [ "$(refrows "$REF_DIR/chat-greedy.jsonl" $REF_TAG)" = "6 6" ] || { log "preflight: $REF_DIR/chat-greedy.jsonl is not 6 rows tagged $REF_TAG"; return 1; }
  [ "$(md5of "$LIVE")" = "$EXPECT_OLD" ] || { log "preflight: live launcher md5 $(md5of "$LIVE") != $EXPECT_OLD"; return 1; }
  [ "$(grep -oE '^DAILY_IMG=\S+' "$LIVE" | cut -d= -f2)" = "$BASE_IMG" ] || { log "preflight: live DAILY_IMG is not $BASE_IMG"; return 1; }
  [ "$(envline "$LIVE" | wc -w | tr -dc 0-9)" = "$NKEYS" ] || { log "preflight: live EXTRA_ENV is not $NKEYS keys"; return 1; }
  grep -q '^CKPT_NAME=${CKPT_NAME:-qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab}' "$LIVE" && grep -q '^CKPT_NAME=${CKPT_NAME:-qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab}' "$TPL" \
    || { log "preflight: live / template CKPT_NAME is not the r0b0tlab 2.50 daily"; return 1; }
  [ "$(md5of "$TPL")" = "$EXPECT_TPL" ] || { log "preflight: template md5 $(md5of "$TPL") != $EXPECT_TPL"; return 1; }
  bash -n "$TPL" || { log "preflight: template does not parse"; return 1; }
  structural "$LIVE" "$TPL" || { log "preflight: template is not live with only DAILY_IMG changed"; return 1; }
  [ "$(iid "$IMG")" = "$IMG_ID" ] || { log "preflight: $IMG id $(iid "$IMG") != $IMG_ID (the image the operator built and tested)"; return 1; }
  [ "$(iid "$BASE_IMG")" = "$BASE_ID" ] || { log "preflight: rollback image $BASE_IMG id $(iid "$BASE_IMG") != $BASE_ID"; return 1; }
  [ "$(sudo docker image inspect "$IMG" -f '{{index .Config.Labels "local.loopthink.base_id"}}' 2>/dev/null)" = "$BASE_ID" ] \
    || { log "preflight: $IMG label local.loopthink.base_id is not $BASE_ID (not built FROM the served image)"; return 1; }
  [ "$(sudo docker image inspect "$IMG" -f '{{index .Config.Labels "local.loopthink.round"}}' 2>/dev/null)" = r5 ] \
    || { log "preflight: $IMG label local.loopthink.round is not r5"; return 1; }
  return 0; }
preflight || { log "ABORT (pre-queue preflight): nothing touched"; decide "NOT PROMOTED (pre-queue preflight)"; exit 3; }
log "start: live md5 $EXPECT_OLD ($BASE_IMG), template md5 $EXPECT_TPL ($IMG $IMG_ID, base_id label $BASE_ID, patch_sha256 label $(sudo docker image inspect "$IMG" -f '{{index .Config.Labels "local.loopthink.patch_sha256"}}' 2>/dev/null) vs repo $LT5_PATCH_SHA: reported); $NKEYS keys both; AFTER '${AFTER:-}'; results $R"
cp -p "$TPL" "$R/launch-flashnext-r792-loopthink5.sh"; cp -p "$LIVE" "$R/launch-flashnext.sh.at-start"

# ---------------- queue: register now, wait for the units in AFTER (default none), then the lock ----------------
export GPU_QUEUE_NAME=$UNIT
. $D/lib/gpu-queue.sh
. $D/lib/serve-ctl.sh
. $D/lib/gateway-drain.sh
SCTL_LOG="$R/audit.log"
SCTL_API=$API
DAILY_IMG=$IMG
# daily_ok: :8022 serves THE daily: the model id, the live launcher's DAILY_IMG and the NVMe tier on (r787-chain / r788).
daily_ok(){ local want; want=$(grep -oE '^DAILY_IMG=\S+' "$LIVE" | cut -d= -f2)
  [ "$(SCTL_API=http://127.0.0.1:8022/v1 served_id)" = "$MODEL" ] \
  && [ "$(sudo docker inspect -f '{{.Config.Image}}' flashnext 2>/dev/null)" = "$want" ] \
  && sudo docker exec flashnext env 2>/dev/null > "$R/.daily-env" && grep -q '^EXL3_NVME_TIER=' "$R/.daily-env"; }
# a signal before the lock: the unit holding the GPUs may have skipped its daily restore because THIS unit was
# registered; if the lock is free and :8022 does not serve the daily, boot it from $LIVE (untouched while queued).
queued_trap(){ log "signal while queued (before the lock)"
  exec 9>/srv/qwen5090/gpu-exclusive.lock
  if flock -n 9 && ! daily_ok; then
    log "GPUs free and :8022 does not serve the daily: booting it from $LIVE"
    env -i HOME="$HOME" PATH="$CLEAN_PATH" bash "$LIVE" > "$R/boot-queued-restore.log" 2>&1
  fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; decide "NOT PROMOTED (signal while queued)"; exit 4; }
trap queued_trap TERM INT HUP
log "registered in the GPU queue ($GPU_QUEUE_MARK); others: $(gpu_queue_others | tr '\n' ' ')"
# after_wait (r788-common.sh): while a unit named in AFTER is registered with a live PID, do not queue on the lock (flock
# has no FIFO order). This unit's registration is live during the wait, so that unit's finish_restore skips the daily.
after_wait(){ local a m pid waited=0
  for a in ${AFTER:-}; do
    m=$GPU_QUEUE_DIR/$a
    while pid=$(cat "$m" 2>/dev/null) && [ -n "$pid" ] && [ "$pid" != "$GPU_QUEUE_SELF" ] && kill -0 "$pid" 2>/dev/null; do
      [ $((waited % 600)) = 0 ] && log "waiting for $a (pid $pid) to finish before queueing on the GPU lock (${waited}s)"
      sleep 30; waited=$((waited + 30))
    done
  done; }
after_wait

# ---------------- exits ----------------
running_img(){ sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'; }
hermes_off(){ local c
  for c in hermes hermes-webui; do
    [ "$(sudo docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" = true ] || continue
    sudo docker stop -t 30 "$c" >/dev/null 2>&1 && HERMES_STOPPED="$HERMES_STOPPED $c"; done
  log "Hermes stopped for the gates:${HERMES_STOPPED:- none was running (left as is)}"; }
# Repoint (hermes-set-model.sh: config + restart + smoke) ONLY when :8022 serves the production daily (R786 repointed Hermes to
# an experiment container). Otherwise restart what this unit stopped, config untouched, and leave it for the restoring unit.
hermes_back(){ local was=$HERMES_STOPPED; HERMES_STOPPED=
  [ -n "$was" ] && sudo docker start $was >/dev/null 2>&1
  if [ -z "$was" ] && [ "$(sudo docker inspect -f '{{.State.Running}}' hermes 2>/dev/null)" != true ]; then log "Hermes was not running before the unit: left stopped"; return 0; fi
  if daily_ok; then
    PORT=8022 MODEL_ID=$MODEL bash "$HSET" >> "$R/audit.log" 2>&1 && log "Hermes on :8022 (production daily: image + tier checked)" || log "WARN: Hermes not repointed"
  else
    log "WARN: :8022 does not serve the production daily (id $(served_id || echo none), image $(running_img || echo none)): Hermes${was:+ restarted with its config untouched,} not repointed; left for the unit that restores the daily"
  fi; }
wrapup(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; sudo chown -R "$(stat -c %U $D/results)" "$R" 2>/dev/null || true; log "=== $UNIT $1 ==="; }
# The launcher was NOT swapped: make sure the daily serves from $LIVE on :8022 (tier on), without a bounce when it already does.
ensure_daily(){
  if daily_ok; then log "daily already serving from $LIVE ($(running_img), tier on)"; return 0; fi
  served_stop; wait_unserved 45 || true
  env -i HOME="$HOME" PATH="$CLEAN_PATH" bash "$LIVE" > "$R/boot-restore.log" 2>&1 && wait_served_id "$MODEL" 240 10 \
    && log "daily restored from $LIVE: $(running_img)" || log "DAILY RESTORE FAILED: $(tail -3 "$R/boot-restore.log" | tr '\n' ' ' | cut -c1-200)"; }
not_promoted(){ trap 'log "signal during the abort: ignored"' TERM INT HUP
  log "NOT PROMOTED ($1): live launcher untouched"
  [ -n "$(served_id)" ] && sudo docker logs flashnext > "$R/docker-final.log" 2>&1
  ensure_daily; hermes_back; decide "NOT PROMOTED ($1)"; wrapup "NOT PROMOTED"; exit 3; }
rollback(){ trap 'log "signal during the rollback: ignored"' TERM INT HUP
  log "GATE FAILED ($1): rolling back to $PRE"
  sudo docker logs flashnext > "$R/docker-final.log" 2>&1
  if [ "$PROMOTED" = 1 ]; then
    cp -p "$PRE" "$LIVE.new" && mv -f "$LIVE.new" "$LIVE"
    [ "$(md5of "$LIVE")" = "$EXPECT_OLD" ] && log "live launcher restored from $PRE (md5 $EXPECT_OLD)" || log "ROLLBACK COPY FAILED: restore $LIVE from $PRE by hand"
    PROMOTED=0
  fi
  # always a fresh boot: the candidate container would otherwise keep serving under the rolled-back launcher
  served_stop; wait_unserved 45 || true
  env -i HOME="$HOME" PATH="$CLEAN_PATH" bash "$LIVE" > "$R/boot-rollback.log" 2>&1 && wait_served_id "$MODEL" 240 10 \
    && log "daily restarted on the rolled-back launcher: $(running_img); $(grep -aoE 'env keys \([0-9]+\)' "$R/boot-rollback.log" | tail -1); $(grep -aoE 'VRAM free MiB [0-9/]+' "$R/boot-rollback.log" | tail -1)" \
    || log "ROLLBACK BOOT FAILED: $(tail -3 "$R/boot-rollback.log" | tr '\n' ' ' | cut -c1-200)"
  hermes_back; decide "ROLLED BACK ($1)"; wrapup "ROLLED BACK"; exit 3; }
on_signal(){ log "signal"; [ "$PROMOTED" = 1 ] && rollback "signal (needrestart / operator stop): VOID"; not_promoted "aborted by signal"; }

gpu_lock
trap on_signal TERM INT HUP
gateway_drain   # the gates run on the live :8022 port; Olla routes nothing to it until this unit exits
log "lock held; served $(served_id || echo none) ($(running_img)); VRAM free $(vram_free)"
preflight || not_promoted "post-lock preflight"

# ---------------- helpers (r728 / r785) ----------------
errcounts(){ echo "OOM $(grep -acE 'OutOfMemoryError|out of memory|graph\.cu' "$1"); alloc-retry $(grep -acE 'retries\+[1-9]|num_alloc_retries[^0-9]*[1-9]' "$1"); TORCH_CHECK $(grep -acE 'TORCH_CHECK|c10::Error' "$1"); tracebacks $(grep -ac Traceback "$1")"; }
errsum(){ grep -acE 'OutOfMemoryError|out of memory|graph\.cu|retries\+[1-9]|TORCH_CHECK|c10::Error|Traceback' "$1"; }
oomsum(){ grep -acE 'OutOfMemoryError|out of memory|graph\.cu|retries\+[1-9]|num_alloc_retries[^0-9]*[1-9]' "$1"; }
alive(){ [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
okrows(){ python3 -c 'import json,sys,os
rs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()] if os.path.exists(sys.argv[1]) else []
print(sum(1 for r in rs if r.get("ok")), len(rs))' "$1" 2>/dev/null || echo "0 0"; }
clog(){ sudo docker logs flashnext > "$R/container.log" 2>&1; }
TO="timeout $PROBE_TO"

# ---------------- install + boot (the production config: tier on) ----------------
hermes_off
[ "$(md5of "$LIVE")" = "$EXPECT_OLD" ] || not_promoted "guard: live launcher changed before the swap"
[ "$(md5of "$TPL")" = "$EXPECT_TPL" ] || not_promoted "guard: template changed before the swap"
cp -p "$LIVE" "$PRE" && [ "$(md5of "$PRE")" = "$EXPECT_OLD" ] || not_promoted "guard: could not write the rollback copy $PRE"
PROMOTED=1   # before the copy: a signal mid-swap must roll back, never boot an ungated launcher (R785 review F5)
# Safety net (PRELAUNCH-R789 S2): if the script dies after the swap without a decision, roll back on EXIT instead of
# leaving the loop-think r5 launcher live ungated; keeps gpu-queue.sh's marker removal.
trap 'rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; if [ "$PROMOTED" = 1 ] && [ "$DECIDED" = 0 ]; then rollback "unit exited without a decision"; fi' EXIT
cp -p "$TPL" "$LIVE.new" && mv -f "$LIVE.new" "$LIVE"
cmp -s "$TPL" "$LIVE" || rollback "install copy (live != template)"
note "launcher installed: $EXPECT_OLD -> $EXPECT_TPL (DAILY_IMG $BASE_IMG -> $IMG, $NKEYS keys unchanged; rollback $PRE)"
served_stop; wait_unserved 45 || true
env -i HOME="$HOME" PATH="$CLEAN_PATH" bash "$LIVE" > "$R/boot-new.log" 2>&1 && wait_served_id "$MODEL" 240 10 \
  || { sudo docker logs flashnext > "$R/container-noboot.log" 2>&1; rollback "G1 the new daily did not boot"; }
generation_state "$MODEL" | grep -q GEN_SANE || rollback "G1 generation not sane"
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || log "WARN: gateway not idle after 900 s"

# ---- G1 ----
S0=${REF_UP%/*}; S1=${REF_UP#*/}
sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$R/env-new.txt"
got=$(running_img); gotid=$(sudo docker inspect -f '{{.Image}}' flashnext 2>/dev/null)
pool=$(grep -aoE 'cache [0-9]+ @' "$R/boot-new.log" | tail -1 | tr -dc '0-9')
spl=$(grep -aoE 'split \[[^]]*\]' "$R/boot-new.log" | tail -1)
up=$(grep -aoE 'VRAM free MiB [0-9]+/[0-9]+' "$R/boot-new.log" | tail -1 | grep -oE '[0-9]+/[0-9]+'); f0=${up%/*}; f1=${up#*/}
keys=$(assert_env_keys "$R/boot-new.log" $NKEYS EXL3_GR_MIX_TILED 2>&1); krc=$?
verb=0; for kv in $(envline "$LIVE"); do grep -qxF "$kv" "$R/env-new.txt" || { log "G1: container env lacks $kv"; verb=1; }; done
sudo docker logs flashnext > "$R/container-boot.log" 2>&1
stale=$(grep -aoE '[0-9]+ stale namespaces removed' "$R/container-boot.log" | tail -1)
# landing: loop-think r5's import-time proof (the Dockerfile.box build step) re-run inside the SERVED container. env -u
# EXL3_NVME_TIER as in Dockerfile.box (the served container has the tier var; a second process must not open the tier);
# CUDA_VISIBLE_DEVICES empty so the extra python never opens a CUDA context on the serving GPUs (the build ran with no GPU).
timeout 300 sudo docker exec -e CUDA_VISIBLE_DEVICES= flashnext env -u EXL3_NVME_TIER python3 /opt/loopthink-r5/landing_loopthink.py > "$R/landing.txt" 2>&1; lrc=$?
sudo docker exec flashnext sh -c 'md5sum /opt/loopthink-r5/* /opt/loopthink-r4/* 2>/dev/null' > "$R/opt-loopthink-md5.txt" 2>&1
landed=$(grep -a 'loop-think r5 landed:' "$R/landing.txt" | tail -1)
note "G1 landing: rc $lrc, ${landed:-no 'loop-think r5 landed' line ($(tail -2 "$R/landing.txt" | tr '\n' ' ' | cut -c1-200))}; in-image /opt/loopthink-r5 md5s in opt-loopthink-md5.txt (reported)"
note "G1: image $got ($gotid), keys rc $krc ${keys:+($keys)}, $NKEYS present verbatim: $([ $verb = 0 ] && echo yes || echo NO), tier $(grep -c '^EXL3_NVME_TIER=' "$R/env-new.txt") (${stale:-no 'stale namespaces' line}: reported, a non-zero count means the tier restarted cold), pool $pool, $spl, served $(served_id), UP free $up (floor $(( S0 - 32 ))/$(( S1 - 32 )) = R785c $REF_UP - 32)"
[ "$got" = "$IMG" ] && [ "$gotid" = "$IMG_ID" ] && [ "$krc" = 0 ] && [ "$verb" = 0 ] && grep -q '^EXL3_NVME_TIER=' "$R/env-new.txt" \
  && [ "$lrc" = 0 ] && [ -n "$landed" ] \
  && [ "$pool" = "$POOL" ] && [ "$spl" = "split [$SPLITSP]" ] && grep -q "tunedir $TUNE," "$R/boot-new.log" && [ "$(served_id)" = "$MODEL" ] \
  && [ "${f0:-0}" -ge $(( S0 - 32 )) ] 2>/dev/null && [ "${f1:-0}" -ge $(( S1 - 32 )) ] 2>/dev/null \
  && note "G1 PASS" || { note "G1 FAIL"; rollback "G1"; }

# ---- G2: greedy identity vs R785c (fn_greedy /v1/completions + chat_greedy incl. the long requeue-crossing answer) ----
cp "$REF_DIR/greedy.jsonl" "$R/greedy.jsonl"; cp "$REF_DIR/chat-greedy.jsonl" "$R/chat-greedy.jsonl"   # tag R785 rows, then R792 appended
$TO python3 "$GREEDY" --url "$BASEURL" --tag $TAG --out "$R/greedy.jsonl" > "$R/greedy-$TAG.log" 2>&1
gn=$(grep -c "\"tag\": \"$TAG\"" "$R/greedy.jsonl" 2>/dev/null); ge=$(grep -ac ' ERROR ' "$R/greedy-$TAG.log")
python3 "$GREEDY" --compare --out "$R/greedy.jsonl" --ref $REF_TAG > "$R/greedy-compare.txt" 2>&1; grc=$?
gid=$(grep -c "^GREEDY $TAG vs $REF_TAG: 6 identical, IDENTICAL$" "$R/greedy-compare.txt")
$TO python3 "$CHAT" --url "$API" --model "$MODEL" --tag $TAG --long --out "$R/chat-greedy.jsonl" > "$R/chat-greedy-$TAG.log" 2>&1; crc=$?
python3 "$CHAT" --compare --out "$R/chat-greedy.jsonl" --ref $REF_TAG --cand $TAG --long > "$R/chat-compare.txt" 2>&1; ccrc=$?
cid=$(grep -c "^CHAT-GREEDY $TAG vs $REF_TAG: 6/6 identical$" "$R/chat-compare.txt")
cg=$(python3 - "$R/chat-greedy.jsonl" "$TAG" <<'PY' 2>&1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if r["tag"] == sys.argv[2]]
pids = [r["pid"] for r in rows]
lg = [r for r in rows if r["pid"] == "long"]
ok = sorted(pids) == sorted(["code", "math", "prose", "tool", "multi", "long"]) and len(lg) == 1 \
     and lg[0].get("crossed_requeue") and lg[0]["finish"] in ("stop", "length")
print(("OK" if ok else "BAD") + f" rows {len(rows)} pids {pids} long tokens {lg[0]['completion_tokens'] if lg else None} "
      f"finish {lg[0]['finish'] if lg else None}")
PY
)
grep -aE '^(GREEDY|  DIVERGE|  MISSING|     )' "$R/greedy-compare.txt" | head -20 | sed 's/^/  [greedy] /' | tee -a "$R/audit.log" > /dev/null
grep -a '^CHAT-GREEDY' "$R/chat-compare.txt" | sed 's/^/  [chat] /' | tee -a "$R/audit.log" > /dev/null
note "G2: fn_greedy $gn records / $ge errors, compare rc $grc ($(grep -a "^GREEDY $TAG vs" "$R/greedy-compare.txt" | head -1)); chat_greedy rc $crc: $cg; compare rc $ccrc ($(grep -a "^CHAT-GREEDY $TAG vs" "$R/chat-compare.txt" | head -1)); reference $REF_DIR (tag $REF_TAG)"
[ "$gn" = 6 ] && [ "$ge" = 0 ] && [ "$grc" = 0 ] && [ "$gid" = 1 ] && [ "$crc" = 0 ] && [ "$ccrc" = 0 ] && [ "$cid" = 1 ] && [ "${cg:0:2}" = OK ] && alive \
  && note "G2 PASS (byte-identical to R785c: fn_greedy 6/6, chat_greedy 6/6 incl. long)" || { note "G2 FAIL (identity is mandatory)"; rollback "G2 greedy identity"; }

# ---- G3: ramp + stress (r728 sequence) ----
$TO python3 "$BENCH" --url "$API" --model "$MODEL" --tag "ramp-$TAG" --kind prose --ctx 4000 --tokens 256 \
  --conc 1 2 3 4 5 6 7 8 --runs 1 --warmup-runs 0 --unique --distinct --salt "$SALT_RAMP" --out "$R/ramp.jsonl" > "$R/ramp.log" 2>&1
clog; read -r okn tot <<< "$(okrows "$R/ramp.jsonl")"; e=$(errsum "$R/container.log")
note "G3 ramp c1..c8: $okn/$tot ok, $(errcounts "$R/container.log"), free $(vram_free)"
[ "$okn" = 36 ] && [ "$tot" = 36 ] && [ "$e" = 0 ] && alive || { note "G3 ramp FAIL"; rollback "G3 ramp"; }
for c in 4 8; do
  [ "$c" = 4 ] && s=$SALT_S4 || s=$SALT_S8
  $TO python3 "$BENCH" --url "$API" --model "$MODEL" --tag "stress-c$c-26k" --kind prose --tokens 256 --conc $c \
    --runs 1 --ctx 26000 --unique --salt "$s" --out "$R/stress-c$c.jsonl" > "$R/stress-c$c.log" 2>&1
  clog; read -r okn tot <<< "$(okrows "$R/stress-c$c.jsonl")"; o=$(oomsum "$R/container.log")
  note "G3 stress c$c @26k: $okn/$tot ok (want $c), $(errcounts "$R/container.log"), free $(vram_free)"
  [ "$okn" = "$c" ] && [ "$tot" = "$c" ] && [ "$o" = 0 ] && alive || { note "G3 stress c$c FAIL"; rollback "G3 stress c$c"; }
done
note "G3 PASS"
$TO python3 "$BENCH" --url "$API" --model "$MODEL" --tag "pf-$TAG-120000" --kind prose --tokens 64 --conc 1 --runs 1 --ctx 120000 --unique \
  --salt "$SALT_120" --out "$R/prefill-120k.jsonl" > "$R/prefill-120k.log" 2>&1
note "120k cold prefill (production boot, tier on; G4 input, speed reported; R785c 12009, R789 12464 t/s): $(python3 -c 'import json,sys
r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(" ".join("ok=%s %d tok in %.2f s = %.0f t/s" % (x.get("ok"), x["prompt_tokens"], x["ttft_s"], x["prompt_tokens"]/x["ttft_s"]) for x in r if x.get("ttft_s")) or "no record")' "$R/prefill-120k.jsonl" 2>/dev/null); free $(vram_free)"

# ---- G5a (GATED): loop-think r5's own test file inside the served container ----
# test_loop_think.py (89 checks: r4's 27 unchanged + r5's 62: the ladder on the R791 fixture rows, false-positive texts,
# r4-identical firing, finish_reason of a cut tool call, streaming / usage-chunk / n=2 cases) drives the served /app's
# collector with a fake backend: CPU only, no request reaches the server, its WARNING lines go to its own stdout. The image
# carries only fix.patch, r4-to-r5.patch and landing_loopthink.py in /opt/loopthink-r5, so the repo's test file and fixture
# ($LT5, md5-pinned in preflight) are copied into the container's /tmp. Same env hygiene as the landing check.
sudo docker exec flashnext rm -rf /tmp/loopthink-r5-test > /dev/null 2>&1
sudo docker cp "$LT5/." flashnext:/tmp/loopthink-r5-test >> "$R/audit.log" 2>&1
timeout 1200 sudo docker exec -w /app -e PYTHONPATH=/app -e CUDA_VISIBLE_DEVICES= flashnext \
  env -u EXL3_NVME_TIER python3 /tmp/loopthink-r5-test/test_loop_think.py > "$R/test-in-image.txt" 2>&1; trc=$?
tfails=$(grep -a '^FAILS ' "$R/test-in-image.txt" | tail -1 | cut -d' ' -f2); tpass=$(grep -ac '^PASS ' "$R/test-in-image.txt"); tfail=$(grep -ac '^FAIL ' "$R/test-in-image.txt")
grep -a '^FAIL ' "$R/test-in-image.txt" | head -10 | sed 's/^/  [test] /' | tee -a "$R/audit.log" > /dev/null
note "G5a in-image test_loop_think.py: rc $trc, FAILS ${tfails:-none}, PASS lines $tpass (want 89), FAIL lines $tfail; CPU lines (report): $(grep -aiE 'us/token|µs' "$R/test-in-image.txt" | head -4 | tr '\n' ';' | cut -c1-400)"
[ "$trc" = 0 ] && [ "$tfails" = 0 ] && [ "$tpass" = 89 ] && [ "$tfail" = 0 ] && alive && note "G5a PASS" || { note "G5a FAIL"; rollback "G5a in-image test_loop_think.py"; }

# ---- G5: loop-think on the daily template (R789's G5 unchanged: r4's mechanism, sampler pinned at 0.6) ----
T5=$(date -u +%Y-%m-%dT%H:%M:%SZ)
$TO python3 "$LPX" "$CTL/session.json" "$R" "$API" > "$R/loop-prefixes.txt" 2>&1; tee -a "$R/audit.log" < "$R/loop-prefixes.txt"
[ -s "$R/prefix-nothink.txt" ] && [ -s "$R/prefix-p700.txt" ] && [ -s "$R/prefix-p700.json" ] || { note "G5 FAIL: prefixes not built"; rollback "G5 prefixes"; }
# Sampler pinned at 0.6 (R789 re-run, 2026-09-28): the probe used to omit the temperature, so G5 ran at the server's
# fallback -- 0.6 for R785c's reference, 1.0 since 2026-09-27 20:18 UTC. At 1.0 the first R789 failed G5 (1 of 2 forced
# rows carried the loop into the answer) and the served daily itself does the same on 7/16 forced rows vs 0/16 at 0.6
# (results/2026-09-28-loopthink-temp-0011): a sampler effect, not the key drop (G2 byte-identical). G5 gates the
# loop-think mechanism under R785c's conditions; the 1.0 behaviour is reported separately.
RP="$TO python3 $RHT --url $API --model $MODEL --temperature 0.6"
$RP --forced-loop --n 2 --conc 2 --max-tokens 12000 --tag forced --out "$R/forced.jsonl" > "$R/forced.txt" 2>&1
$RP --forced-loop --no-think --prompt "Write the line 'the quick brown fox jumps over the lazy dog' 400 times, one per line." \
    --prefix-file "$R/prefix-nothink.txt" --n 2 --conc 2 --max-tokens 5000 --tag nothink --out "$R/nothink.jsonl" > "$R/nothink.txt" 2>&1
$RP --session "$CTL/session.json" --db "$CTL/state.db" --upto 16 --tools "$HTOOLS" --prefix-file "$P107" \
    --n 2 --conc 2 --tag p107 --out "$R/p107.jsonl" > "$R/p107.txt" 2>&1
P700Q="Restate your plan in your thinking, then answer: what is 17*23? Reply with the number only."
$RP --forced-loop --prompt "$P700Q" --prefix-file "$R/prefix-p700.txt" --n 4 --conc 4 --max-tokens 8000 --tag p700 \
    --out "$R/p700.jsonl" > "$R/p700.txt" 2>&1
g5check(){ python3 - "$R" "$LOOPMSG" "$@" <<'PY'
import json, os, sys
R, msg = sys.argv[1], sys.argv[2]
counts = dict(a.split("=") for a in sys.argv[3:])
rows = lambda t: [json.loads(l) for l in open(f"{R}/{t}.jsonl")] if os.path.exists(f"{R}/{t}.jsonl") else []
has = lambda r: msg in (r.get("reasoning") or "") + (r.get("content") or "")
anchor = json.load(open(f"{R}/prefix-p700.json"))["anchor"]
prefix = open(f"{R}/prefix-p700.txt").read()
def echoed(r):   # does the stream carry the prefilled text itself (response_prefix echoed back), in full?
    return (r.get("reasoning") or "").lstrip().startswith(prefix.strip())
def copies(r):   # GENERATED copies of the p700 block: subtract the prefilled copies only when the stream echoes the prefix
    # (R785 review: R785 subtracted 2 whenever the stream started with the header, which the model can regenerate)
    t = r.get("reasoning") or ""
    return t.count(anchor) - (prefix.count(anchor) if echoed(r) else 0)
out, bad = [], []
f, n, p = rows("forced"), rows("nothink"), rows("p107")
L = rows("p700") + rows("p700b")
# Is the forced injection text surfaced in the stream? R783 gated on the log count only; if no forced row carries it, the
# rows cannot be attributed by text and the checks fall back to content + counts (reported as such, never a rollback cause).
visible = any(has(r) for r in f)
answered = lambda r: r["class"] in ("tool_call", "content")
if not (len(f) == 2 and all(r["class"] == "content" and "391" in r["content"] and (has(r) or not visible) for r in f)):
    bad.append("forced: want 2 x content '391'" + (" with the injection message" if visible else "") + ", got "
               + str([(r["class"], "391" in (r.get("content") or ""), has(r)) for r in f]))
if not (len(n) == 2 and all(r.get("finish") == "stop" and r.get("completion_tokens") == 800 and not has(r) for r in n)):
    bad.append("nothink: want 2 x stop at exactly 800 without the message, got "
               + str([(r.get("finish"), r.get("completion_tokens"), has(r)) for r in n]))
if not (len(p) == 2 and all(answered(r) for r in p)):
    bad.append("p107: want 2 x tool call or content, got " + str([r["class"] for r in p]))
p107cls = ["injected" if has(r) else ("escaped" if (r.get("reasoning") or "").count("Better approach: make the hill a cone") >= 3
           else "no-loop") for r in p]
if visible:
    inj_rows = [r for r in L if has(r)]
    esc_rows = [r for r in L if not has(r) and copies(r) >= 5]
else:   # by behaviour: looped (>= 3 generated copies) and then answered = caught; >= 5 copies and no answer = escaped
    inj_rows = [r for r in L if copies(r) >= 3 and answered(r) and "391" in (r.get("content") or "")]
    esc_rows = [r for r in L if copies(r) >= 5 and r not in inj_rows]
looped = [r for r in L if r in inj_rows or copies(r) >= 3]
if any(r["class"] == "error" for r in f + n + p + L):
    bad.append("error rows")
if any(not answered(r) for r in inj_rows):
    bad.append("p700: an injected row did not answer: " + str([r["class"] for r in inj_rows]))
if esc_rows:
    bad.append(f"p700: {len(esc_rows)} row(s) escaped (>= 5 generated copies, no injection): " + str([copies(r) for r in esc_rows]))
if any(copies(r) < 0 for r in L):
    bad.append("p700: negative generated-copy count (echo detection wrong): " + str([copies(r) for r in L]))
if visible:
    tagged = sum(1 for r in f + p + L if has(r))
    if int(counts["inj"]) != tagged:
        bad.append(f"collector injections {counts['inj']} in the window != rows carrying the message {tagged}")
else:
    lo, hi = 2 + len(inj_rows), 2 + len(p) + len(L)
    out.append(f"injection text not surfaced in the stream: attribution by content and counts only (inj in [{lo}, {hi}])")
    if not lo <= int(counts["inj"]) <= hi:
        bad.append(f"collector injections {counts['inj']} in the window outside [{lo}, {hi}]")
if int(counts["nof"]) != 0:
    bad.append(f"'could not force' lines {counts['nof']}")
if int(counts["eng"]) != 2:
    bad.append(f"engine loop stops {counts['eng']} (want 2 = nothink)")
state = "EXERCISED" if inj_rows else ("NOT-EXERCISED" if not looped else "LOOPED-NOT-CAUGHT")
if state == "LOOPED-NOT-CAUGHT" and not esc_rows:
    state = "NOT-EXERCISED"   # looped 3-4 copies then left: the 3,000-token window never filled
for o in out:
    print("G5 note: " + o)
print(f"G5 p107 rows: {p107cls}; p700 rows {len(L)}: injected {len(inj_rows)}, escaped {len(esc_rows)}, looped {len(looped)}, "
      f"copies {[copies(r) for r in L]}, prefix echoed {sum(echoed(r) for r in L)}/{len(L)}, classes {[r['class'] for r in L]}; "
      f"window inj {counts['inj']} nof {counts['nof']} eng {counts['eng']}")
print(("G5 FAIL: " + "; ".join(bad)) if bad else f"G5 OK p700 {state}")
PY
}
g5counts(){ sudo docker logs --since "$T5" flashnext > "$R/container-g5.log" 2>&1
  echo "inj=$(grep -ac 'reasoning loop detected' "$R/container-g5.log") nof=$(grep -ac 'could not force the end' "$R/container-g5.log") eng=$(grep -ac 'token loop was detected' "$R/container-g5.log")"; }
g5=$(g5check $(g5counts) 2>&1); echo "$g5" > "$R/g5.txt"
if echo "$g5" | grep -q '^G5 OK p700 NOT-EXERCISED'; then
  log "G5: p700 not exercised (no row kept looping): one re-run, x4"
  $RP --forced-loop --prompt "$P700Q" --prefix-file "$R/prefix-p700.txt" --n 4 --conc 4 --max-tokens 8000 --tag p700b \
      --out "$R/p700b.jsonl" > "$R/p700b.txt" 2>&1
  g5=$(g5check $(g5counts) 2>&1); echo "$g5" > "$R/g5.txt"
fi
cat "$R/forced.txt" "$R/nothink.txt" "$R/p107.txt" "$R/p700.txt" $( [ -e "$R/p700b.txt" ] && echo "$R/p700b.txt" ) \
  | grep -aE '^(forced|nothink|p107|p700b?) #|RESULT' | sed -E 's/ [|] C:.*//' | tee -a "$R/audit.log" > /dev/null
note "$g5"
echo "$g5" | grep -q '^G5 OK' && alive || { note "G5 FAIL"; rollback "G5 loop-think"; }
echo "$g5" | grep -q '^G5 OK p700 NOT-EXERCISED' && note "G5 PASS with r4's long-period case NOT EXERCISED (as in R785b/c: the model leaves the loop): reported, not a rollback" \
  || note "G5 PASS"

# ---- G5 REPORT-ONLY arms (never a rollback cause), after the G5 decision, each on its own log window ----
# (b) R789's forced-loop probe at the SERVED fallback (no temperature sent = 1.0; R789 review), n 8. R790: 6/16 answers
#     carry the looped line at 1.0 vs 0/16 at 0.6 -- a sampler effect loop-think r5 does not change; reported for the record.
# (c) the r5 ladder live: R781's Hermes turn 16 (R791's u16 = the p107 turn) with a prefix = the R791 u16 t0.6 row's reasoning
#     cut where the 3rd copy of its 3,575-char (~1,199-token) loop block starts (pre-loop text + 2 copies; p107's
#     construction, which R790 saw continued 8/8 at 0.6). Period > L = 1,000, so r4's detectors cannot fire; the (6000, 2000)
#     rung can, after ~6,000 generated looping tokens. Built at run time from the md5-pinned fixture; 0.6, n 4, 16k.
#     Session mode instead of --forced-loop: a block out of its conversation reads NOT-EXERCISED (R785 p700).
# (d) finish_reason of a tool call cut by max_tokens (r5 change 2): write_file of a long file, thinking off, temperature 0,
#     max_tokens 300; "length" with 0 tool calls expected (base TabbyAPI said "tool_calls").
logwin(){ sudo docker logs --since "$1" flashnext > "$2" 2>&1; }
T5B=$(date -u +%Y-%m-%dT%H:%M:%SZ)
$TO python3 "$RHT" --url "$API" --model "$MODEL" --forced-loop --n 8 --conc 4 --max-tokens 12000 --tag forced-fallback \
    --out "$R/forced-fallback.jsonl" > "$R/forced-fallback.txt" 2>&1
logwin "$T5B" "$R/container-g5b.log"
rb=$(python3 - "$R/forced-fallback.jsonl" "$LOOPMSG" <<'PY' 2>&1
import json, os, sys
f, msg = sys.argv[1], sys.argv[2]
rs = [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []
line = "the quick brown fox jumps over the lazy dog"
clean = [r for r in rs if r["class"] == "content" and "391" in r["content"] and line not in r["content"]]
looped = [r for r in rs if line in (r.get("content") or "")]
print(f"rows {len(rs)}: clean '391' {len(clean)}, looped line in the answer {len(looped)}, injection message in the stream "
      f"{sum(msg in (r.get('reasoning') or '') + (r.get('content') or '') for r in rs)}, errors {sum(r['class'] == 'error' for r in rs)}; "
      f"(class, finish, tokens) {[(r['class'], r.get('finish'), r.get('completion_tokens')) for r in rs]} (R790 at 1.0: 6/16 looped)")
PY
)
note "G5-REPORT (b) forced loop at the served fallback (no temperature), n 8: $rb; window inj $(grep -ac 'reasoning loop detected' "$R/container-g5b.log") eng $(grep -ac 'token loop was detected' "$R/container-g5b.log")"

T5C=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 - "$LT5/fixtures/r791-rows.json.gz" u16_t06_len "$R" > "$R/prefix-ladder.log" 2>&1 <<'PY'
import gzip, json, sys
fx, name, out = sys.argv[1:]
t = {r["name"]: r for r in json.load(gzip.open(fx, "rt"))["rows"]}[name]["reasoning"]
p = next(q for q in range(200, len(t) // 4) if t[-3 * q:] == t[-4 * q:-q])   # smallest char period of the tail
s = len(t) - 1
while s - p >= 0 and t[s] == t[s - p]:
    s -= 1
first = max(0, s + 1 - p)          # start of the first copy (the tail match begins at the second)
cut = first + 2 * p                # where the 3rd copy starts
block = t[first:first + p]
open(f"{out}/prefix-ladder.txt", "w").write(t[:cut])
json.dump({"row": name, "period_chars": p, "loop_start_char": first, "cut_at": cut, "copies_in_row": round((len(t) - first) / p, 2),
           "anchor": block[:120]}, open(f"{out}/prefix-ladder.json", "w"))
print(f"ladder prefix: {name} period {p} chars, loop from char {first}, {(len(t) - first) / p:.2f} copies in the row, "
      f"cut at {cut} (pre-loop {first} chars + 2 copies)")
PY
if [ -s "$R/prefix-ladder.txt" ] && [ -s "$R/prefix-ladder.json" ]; then
  tee -a "$R/audit.log" < "$R/prefix-ladder.log" > /dev/null
  $TO python3 "$RHT" --url "$API" --model "$MODEL" --temperature 0.6 --session "$CTL/session.json" --db "$CTL/state.db" --upto 16 \
      --tools "$HTOOLS" --prefix-file "$R/prefix-ladder.txt" --n 4 --conc 4 --max-tokens 16000 --tag ladder \
      --out "$R/ladder.jsonl" > "$R/ladder.txt" 2>&1
  logwin "$T5C" "$R/container-g5c.log"
  rc5=$(python3 - "$R" "$LOOPMSG" "$API" <<'PY' 2>&1
import json, os, re, sys, urllib.request
R, msg, api = sys.argv[1:]
meta = json.load(open(f"{R}/prefix-ladder.json")); prefix = open(f"{R}/prefix-ladder.txt").read(); anchor = meta["anchor"]
rs = [json.loads(l) for l in open(f"{R}/ladder.jsonl") if l.strip()] if os.path.exists(f"{R}/ladder.jsonl") else []
def ntok(t):
    try:
        q = urllib.request.Request(api.rsplit("/v1", 1)[0] + "/v1/token/encode", json.dumps({"text": t}).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(q, timeout=60))["length"]
    except Exception:
        return None
out = []
for r in rs:
    t = r.get("reasoning") or ""
    echoed = t.lstrip().startswith(prefix.strip()[:2000])
    gen = t.lstrip()[len(prefix.strip()):] if echoed else t
    k = gen.find(msg)
    out.append({"class": r["class"], "finish": r.get("finish"), "tok": r.get("completion_tokens"), "msg": k >= 0,
                "msg_at_tok": ntok(gen[:k]) if k >= 0 else None, "gen_copies": gen.count(anchor), "echoed": echoed})
fires = [int(x) for x in re.findall(r"reasoning loop detected after (\d+) reasoning tokens", open(f"{R}/container-g5c.log", errors="replace").read())]
print(f"rows {len(rs)}: injection message in {sum(o['msg'] for o in out)}, answered (tool call / content) "
      f"{sum(o['class'] in ('tool_call', 'content') for o in out)}; server fired after {sorted(fires)} reasoning tokens; "
      f"per row {[(o['class'], o['finish'], o['tok'], o['msg_at_tok'], o['gen_copies']) for o in out]} "
      f"(class, finish, completion tokens, reasoning tokens before the message, generated copies of the block); "
      f"prefix echoed {sum(o['echoed'] for o in out)}/{len(out)}")
PY
)
  note "G5-REPORT (c) ladder, turn 16 + u16 prefix ($(head -1 "$R/prefix-ladder.log" | cut -c1-160)), 0.6, n 4, 16k: $rc5; window inj $(grep -ac 'reasoning loop detected' "$R/container-g5c.log") nof $(grep -ac 'could not force the end' "$R/container-g5c.log") eng $(grep -ac 'token loop was detected' "$R/container-g5c.log")"
else
  note "G5-REPORT (c) ladder: SKIPPED, prefix not built ($(tail -2 "$R/prefix-ladder.log" | tr '\n' ' ' | cut -c1-200))"
fi

T5D=$(date -u +%Y-%m-%dT%H:%M:%SZ)
$TO python3 - "$API" "$MODEL" > "$R/finish-probe.txt" 2>&1 <<'PY'
import json, sys, urllib.request
api, model = sys.argv[1:]
tool = {"type": "function", "function": {"name": "write_file", "description": "Write a text file to disk.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "file path"},
                       "content": {"type": "string", "description": "the full file content"}}, "required": ["path", "content"]}}}
body = {"model": model, "max_tokens": 300, "temperature": 0, "tools": [tool],
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "Call the write_file tool right away to create numbers.txt containing every "
                      "integer from 1 to 1000, one per line, written out in full. Do not reply with anything else."}]}
def post(b, timeout=300):
    q = urllib.request.Request(api + "/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(q, timeout=timeout)
res = []
try:
    d = json.load(post(dict(body, stream=False)))
    c = d["choices"][0]
    res.append(("non-stream", c.get("finish_reason"), len(c["message"].get("tool_calls") or []),
                len(c["message"].get("content") or ""), (d.get("usage") or {}).get("completion_tokens")))
except Exception as e:
    res.append(("non-stream", "ERROR " + repr(e)[:120], None, None, None))
try:
    fin, calls, content, usage = None, set(), 0, None
    for line in post(dict(body, stream=True, stream_options={"include_usage": True})):
        line = line.strip()
        if not line.startswith(b"data:") or line == b"data: [DONE]":
            continue
        ch = json.loads(line[5:]); usage = ch.get("usage") or usage
        for c in ch.get("choices", []):
            fin = c.get("finish_reason") or fin
            dl = c.get("delta") or {}
            content += len(dl.get("content") or "")
            for t in dl.get("tool_calls") or []:
                calls.add(t.get("index", 0))
    res.append(("stream", fin, len(calls), content, (usage or {}).get("completion_tokens")))
except Exception as e:
    res.append(("stream", "ERROR " + repr(e)[:120], None, None, None))
for r in res:
    print(f"{r[0]}: finish_reason {r[1]}, tool_calls {r[2]}, content chars {r[3]}, completion_tokens {r[4]}"
          + (" (cut at max_tokens)" if r[4] == 300 else " (NOT cut: the call fit, probe not exercised)" if r[4] else ""))
PY
logwin "$T5D" "$R/container-g5d.log"
note "G5-REPORT (d) finish_reason of a write_file call cut at max_tokens 300 (want length, 0 calls; base said tool_calls): $(tr '\n' ';' < "$R/finish-probe.txt" | cut -c1-400) server WARNING 'did not parse ... finish_reason length' x$(grep -ac 'did not parse into a tool call; finish_reason length' "$R/container-g5d.log"), '... finish_reason tool_calls' x$(grep -ac 'did not parse into a tool call; finish_reason tool_calls' "$R/container-g5d.log")"
alive || log "WARN: server not alive after the G5 report arms (G6 checks it)"

# ---- G6: needles (R728 P2) ----
$TO python3 "$P/fn_needle_oai.py" --url "$API" --model "$MODEL" --tag needle --ctx-tokens 131072 240000 \
  --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" > "$R/needle-full.log" 2>&1
grep -E "retrieved|FAIL|Error" "$R/needle-full.log" | tee "$R/needle.log" | tee -a "$R/audit.log" > /dev/null
[ "$(grep -c '5/5 retrieved' "$R/needle.log")" = 2 ] && alive && note "G6 needles PASS (5/5 at 131k and 240k)" || { note "G6 needles FAIL"; rollback "G6 needles"; }

# ---- G4: functional headroom over ramp + stress + 120k + needles ----
clog; af=$(vram_free); a0=$(echo $af | cut -d' ' -f1); a1=$(echo $af | cut -d' ' -f2)
o=$(oomsum "$R/container.log")
note "G4 headroom: $(errcounts "$R/container.log") since boot; restarts $(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null); free after the sequence $a0/$a1 MiB (UP $up; R785c 383/899, R789 367/903, reported not compared); cuda:1 floor 500 (R728). Allocator-retry: no logger in this stack, so effectively OOM-only"
fl=met; [ "${a1:-0}" -ge 500 ] || fl="NOT met"
[ "$o" = 0 ] && alive && note "G4 PASS (cuda:1 500 MiB floor REPORT-ONLY: $fl)" || { note "G4 FAIL"; rollback "G4 headroom"; }

# ---- G7: agent replay (r728 G7, R722 flags), scored over the replay window; per session instance (R786 review) ----
T7=$(date -u +%Y-%m-%dT%H:%M:%SZ)
timeout 1300 python3 "$REPLAY" --url "$API" --model "$MODEL" \
  --prod-ladder --respawn --drain --max-fail-streak 3 --gen 700 --tool 2020 --think 11 --stagger 3 --temp 0.6 \
  --seed 722 --max-seconds 600 --out "$R/replay.jsonl" > "$R/replay.log" 2>&1
rc=$?
sudo docker logs --since "$T7" flashnext > "$R/container-replay.log" 2>&1
python3 "$AGG" "$R/container-replay.log" > "$R/agg-replay.txt" 2>&1
sed 's/^/  [replay] /' "$R/agg-replay.txt" | grep -E 'window|AGGREGATE|PER STREAM|mean streams|prompt tokens|prefix cached|MTP acceptance' | tee -a "$R/audit.log"
python3 "$RSTEPS" "$R/container-replay.log" "$R/replay.jsonl" --min-acc $MIN_ACC --label $TAG > "$R/replay-steps.txt" 2>&1
tee -a "$R/audit.log" < "$R/replay-steps.txt" > /dev/null
sv(){ python3 -c 'import sys
L=[l for l in open(sys.argv[1]) if l.startswith("SCORED ")]
d=dict(t.split("=",1) for t in L[-1].split()[1:] if "=" in t) if L else {}
print(d.get(sys.argv[2], ""))' "$R/replay-steps.txt" "$1" 2>/dev/null; }
ps_acc=$(sv per_stream); ps_all=$(sv all_per_stream); tps_acc=$(sv tps); ms_acc=$(sv ms); tps_all=$(sv all_tps); ms_all=$(sv all_ms); basis=$(sv basis); nexc=$(sv excluded_sessions)
pstream=$(python3 -c 'import re,sys
m=re.search(r"PER STREAM\s*:\s*([\d,.]+)", open(sys.argv[1]).read()); print(m.group(1).replace(",","") if m else "")' "$R/agg-replay.txt" 2>/dev/null)
e=$(errsum "$R/container-replay.log"); loops=$(grep -aci 'token loop was detected' "$R/container-replay.log"); linj=$(grep -ac 'reasoning loop detected' "$R/container-replay.log")
[ "$rc" = 0 ] || note "G7 replay rc $rc, the replay's own last lines: $(tail -3 "$R/replay.log" | tr '\n' ' ' | cut -c1-300)"
note "G7 replay: rc $rc; SCORED (sessions with acceptance >= $MIN_ACC, basis ${basis:-none}, ${nexc:-?} excluded) per-stream ${ps_acc:-none} tok/s, ${tps_acc:-?} tok/step, ${ms_acc:-?} ms/step | all requests per-stream ${ps_all:-none} (tabby_log_agg ${pstream:-none}), ${tps_all:-?} tok/step, ${ms_all:-?} ms/step | bar $REPLAY_BAR = 0.98 x R722 W $R722_W on the scored set; R785c $R785_REPLAY; $(errcounts "$R/container-replay.log"); engine loop stops $loops, collector injections $linj"
# coverage bound (PRELAUNCH-R789 S1): the scored set must keep >= 2/3 of the joined requests
[ "$rc" = 0 ] && [ "$e" = 0 ] && [ -n "$ps_acc" ] && [ "$ps_acc" != none ] && python3 -c "import sys; sys.exit(0 if float('$ps_acc') >= $REPLAY_BAR else 1)" \
  && python3 -c "import sys; x, n = int('$(sv excluded_requests)' or 10**9), int('$(sv n)' or 0); sys.exit(0 if 2 * x <= n else 1)" && alive \
  && note "G7 replay PASS" || { note "G7 replay FAIL"; rollback "G7 replay"; }

# ---- G8: agentic-edit (r728 P1) ----
$TO python3 "$P/agentic-edit.py" --url "$API" --model "$MODEL" --tag r792 --conc 1 4 --modes greedy sampled --out "$R/agentic-edit.jsonl" > "$R/agentic-edit.log" 2>&1
grep -a "agentic-edit" "$R/agentic-edit.log" | tee -a "$R/audit.log" > /dev/null
[ "$(grep -ac "6/6 ok" "$R/agentic-edit.log")" = 4 ] && alive && note "G8 agentic-edit PASS (4 x 6/6)" || { note "G8 agentic-edit FAIL"; rollback "G8 agentic-edit"; }

# ---- G9: tool-eval 69 x 4, parallel 8 (r728 P3) ----
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "$BASEURL" --model "$MODEL" --temperature 0.6 --top-p 0.95 --top-k 20 \
    --trials 4 --parallel 8 --json-file "$R/tooleval.json" > "$R/tooleval.log" 2>&1 )
python3 "$P/tooleval_summary.py" "$R/tooleval.json" r792 2>&1 | tee "$R/tooleval-summary.txt" | tee -a "$R/audit.log"
read -r mean nsc ntr <<< "$(python3 -c 'import json,sys
t=json.load(open(sys.argv[1]))["trial_statistics"]; ps=t.get("per_scenario",{})
print(t["final_score_mean"], len(ps), len(next(iter(ps.values()))["points"]) if ps else 0)' "$R/tooleval.json" 2>/dev/null || echo "none 0 0")"
[ "$nsc" = 69 ] && [ "$ntr" = 4 ] || { note "G9 tool-eval VOID: $nsc scenarios x $ntr trials (want 69 x 4)"; rollback "G9 tool-eval VOID (records $nsc x $ntr)"; }
python3 -c "import sys; sys.exit(0 if float('$mean') >= $TE_BAR else 1)" 2>/dev/null && alive \
  && note "G9 tool-eval PASS ($mean >= $TE_BAR; R785b/c 85.5 / 86.5, R789 85.2, R728 band 84.5-88.0)" || { note "G9 tool-eval FAIL (${mean:-unparsed})"; rollback "G9 tool-eval (${mean:-unparsed})"; }

# ---- G10: GSM8K n=500 (r728 P4), c8; 0.960-0.970 -> c4 re-run decides ----
# Sample count = unique doc_id: lm-eval writes one samples line per doc per filter (R785b VOID'd a 0.982 on the line count).
gsm(){ local c=$1 out=$R/ev-gsm8k-c$1 pxp
  python3 "$P/nostop_proxy.py" --listen 127.0.0.1:8031 --upstream "$BASEURL" >> "$R/proxy.log" 2>&1 & pxp=$!; sleep 2
  timeout 10800 "$LM" --model local-chat-completions \
    --model_args "base_url=http://127.0.0.1:8031/v1/chat/completions,model=$MODEL,tokenizer=$CKPT,num_concurrent=$c,max_retries=1,tokenized_requests=False" \
    --tasks gsm8k --num_fewshot 5 --limit 500 --apply_chat_template --gen_kwargs temperature=0,max_gen_toks=8192 --log_samples \
    --output_path "$out" > "$out.log" 2>&1
  kill $pxp 2>/dev/null; wait $pxp 2>/dev/null
  python3 -c '
import json,pathlib,sys
d=pathlib.Path(sys.argv[1]); f=list(d.rglob("results_*.json")); s={json.loads(l)["doc_id"] for x in d.rglob("samples_gsm8k*.jsonl") for l in open(x) if l.strip()}
print(json.loads(f[0].read_text())["results"]["gsm8k"]["exact_match,flexible-extract"] if len(f)==1 else "none", len(s))' "$out" 2>/dev/null || echo "none 0"; }
read -r g8 n8 <<< "$(gsm 8)"
note "G10 GSM8K c8: flexible-extract ${g8} on $n8 samples (bar $GSM_BAR; R785b/c 0.982 / 0.980, R789 0.976)"
[ "$n8" = 500 ] || rollback "G10 GSM8K VOID (c8 samples $n8)"
if python3 -c "import sys; sys.exit(0 if float('$g8') >= $GSM_BAR else 1)" 2>/dev/null; then note "G10 GSM8K PASS (c8 $g8)"
elif python3 -c "import sys; sys.exit(0 if float('$g8') >= $GSM_RERUN else 1)" 2>/dev/null; then
  log "G10: c8 $g8 in [$GSM_RERUN, $GSM_BAR): re-running at c4 (the shape with a same-config reference)"
  read -r g4 n4 <<< "$(gsm 4)"
  note "G10 GSM8K c4 re-run: flexible-extract ${g4} on $n4 samples"
  [ "$n4" = 500 ] || rollback "G10 GSM8K VOID (c4 samples $n4)"
  python3 -c "import sys; sys.exit(0 if float('$g4') >= $GSM_BAR else 1)" 2>/dev/null \
    && note "G10 GSM8K PASS on the c4 re-run ($g4; c8 $g8 borderline: report it)" || { note "G10 GSM8K FAIL (c8 $g8, c4 $g4)"; rollback "G10 GSM8K (c8 $g8, c4 $g4)"; }
else note "G10 GSM8K FAIL (c8 $g8 < $GSM_RERUN)"; rollback "G10 GSM8K (c8 $g8)"; fi

# ---- done ----
alive || rollback "server not alive after the gates"
[ "$(md5of "$LIVE")" = "$EXPECT_TPL" ] || rollback "VOID: live launcher changed under the unit"
daily_ok || rollback "VOID: :8022 is not the production daily at the end (image / tier)"
sudo docker logs flashnext > "$R/docker-final.log" 2>&1
note "promoted daily over the whole run (reported): $(errcounts "$R/docker-final.log"); free $(vram_free)"
note "reference: unchanged, R785c $REF_DIR/greedy.jsonl + chat-greedy.jsonl (tag $REF_TAG); R792's rows are byte-identical. Repo: flan/launch-flashnext-r792-loopthink5.sh -> flan/launch-flashnext-tabby.sh (md5 $EXPECT_TPL); public mirror: DAILY_IMG bump + the loop-think r5 overlay (fixture reasoning stripped)"
trap 'log "signal after the decision: ignored"' TERM INT HUP
decide "PROMOTED"
hermes_back
wrapup "PROMOTED: daily = $IMG @ $POOL [$SPLITSP], $NKEYS keys (rollback $PRE = R789's launcher, $BASE_IMG)"
