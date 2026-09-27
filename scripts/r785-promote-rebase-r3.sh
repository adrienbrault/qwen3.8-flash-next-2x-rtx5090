#!/usr/bin/env bash
# R785 (2026-09-27): gate and promote tabbyapi:rebase-dev-r3 as the Flash-Next daily on :8022, the second half of the
# promotion chain (the plan's prose: docs/PROMOTION.md, "A rebase onto a new upstream engine").
# Derived from r783-promote-loopthink.sh (queue/lock, install-then-gate, Hermes) and r728-promote-window-off.sh (gate
# helpers, post-promotion gates, rollback). The promotion runs without a further approval once the gates pass: R784 must
# read CANDIDATE and every gate below must pass; any miss rolls back.
#   1. waits (unregistered) until R784 has registered in the GPU queue (after its image build), then registers, waits for
#      R784's unit to end and takes the GPU-exclusive lock; R784's finish_restore then skips the daily (this unit queued);
#   2. preflight again under the lock; reads R784's promote.env: aborts (the old daily serving on :8022) unless
#      DECISION=CANDIDATE, FOUND >= 884,736 on the 16,384 grid, the image id / tree sha / live launcher R784 measured;
#   3. renders the launcher from the repo template scripts/launchers/launch-flashnext-r785-rebase-r3.sh (@FOUND@ / @SPLIT@ from
#      promote.env; the template already carries DAILY_IMG=tabbyapi:rebase-dev-r3, EXTRA_ENV + EXL3_GR_MIX_TILED=1 = 42 keys,
#      TUNEDIR=/srv/qwen5090/.exl3cache-rebase-dev-r3 and the header), records its md5, copies it into $R for the repo;
#      structural guard: the rendered file differs from the live one only in those five lines and comments;
#   4. stops Hermes (hermes, hermes-webui: it calls :8022 directly, the Olla drain does not cover it) and drains Olla for
#      the unit's lifetime: the c1 gates need no foreign request in the batch (R782 review);
#   5. installs the rendered launcher (md5 guard right before the swap, rollback copy launch-flashnext.sh.pre-r785) and
#      boots it = the production config (NVMe tier on by default: IMG == DAILY_IMG);
#   6. gates, in order G1 G2 G3 (+ one 120k cold prefill) G5 G6 G4 G7 G8 G9 G10; any miss -> the old launcher goes back,
#      the old daily boots, Hermes returns to :8022, DECISION: ROLLED BACK (<gate>):
#      G1  image rebase-dev-r3 (label loop-think r4), 42 env keys incl. EXL3_GR_MIX_TILED, tier on, pool = FOUND, split =
#          SPLIT, served id = the r0b0tlab 2.50 daily, UP free per card >= R784's S1 UP - 32 MiB
#      G2  fn_greedy 6/6 records, 0 errors + chat_greedy 5 prompts + --long (one T=0 answer capped at 5,000 tokens that
#          crosses the 2,048-token requeue): complete. RECORDED AS THE NEW REFERENCE (tag R785); identity vs R747 N/A
#          (numerics change by design)
#      G3  ramp c1..c8 36/36 ok, 0 errors; stress c4 and c8 @ ~26k all ok, 0 OOM (R661/R667/R717 shape, r728 helpers)
#      G5  loop-think (R783 G4 + its review + r4): forced x2 -> content "391" with the injection message; nothink x2 ->
#          stop at exactly 800 tokens, no message; p107 x2 -> tool call or content (rows classified injected / escaped /
#          no loop); p700 x4 (a ~700-token block prefilled twice: a period only r4's LoopDetector(3L, L) catches) -> >= 1
#          row injected and answering, 0 rows escaped (>= 5 generated copies, no injection); window counts tie out:
#          collector injections == rows carrying the message, 0 "could not force", engine loop stops == 2 (nothink).
#          p700 not exercised (no row kept looping) -> one re-run; still none -> reported NOT EXERCISED, not a rollback
#      G6  needles 5/5 at 131,072 and 5/5 at 240,000 (fn_needle_oai, R728 P2 invocation)
#      G4  functional headroom over ramp + stress + 120k + both needles: 0 OOM / graph.cu / allocator-retry lines, server
#          alive (0 restarts); post-sequence free per card reported (not compared to S; R741, R754); cuda:1 >= 500 MiB REPORT-ONLY
#      G7  agent replay, R722's flags (r728 G7): rc 0, 0 errors, per-stream >= 120.3 tok/s (0.98 x R722 W); vs R728 128.6
#      G8  agentic-edit 4 x "6/6 ok" (r728 P1)
#      G9  tool-eval 69 x 4 @ parallel 8: final_score_mean >= 82.0 (r728 P3); record count 69 x 4 else VOID
#      G10 GSM8K n=500, c8, nostop proxy: flexible-extract >= 0.970; 0.960 <= v < 0.970 -> re-run at c4 (the shape with a
#          same-config reference, 0.974) and decide on it; 500 samples else VOID
#   7. PASS -> DECISION: PROMOTED (the launcher is already live), Hermes back on :8022.
# Reported (R784): F5 lp_margin, F6 corpus fidelity, the speed verdict; copied into summary.txt.
# REFERENCE ROLLOVER: after PROMOTED, R747's greedy.jsonl is no longer the daily's reference; queued units must compare
#   to $R/greedy.jsonl (tag R785) and $R/chat-greedy.jsonl (tag R785).
# UNIT HYGIENE (R783 review): preflight before the lock and again after it; the trap is set right after the lock; every
#   post-lock abort leaves the daily serving from $LIVE on :8022 and Hermes on :8022 (nothing else restores :8022).
# summary.txt ends with exactly one of: DECISION: PROMOTED | DECISION: ROLLED BACK (<gate>) | DECISION: NOT PROMOTED (<why>).
# GPU ~115 min (+25 on a GSM8K c4 re-run, +5 on a rollback).
# Install (operator): scripts/launchers/launch-flashnext-r785-rebase-r3.sh -> /srv/qwen5090/launch-flashnext-r785-rebase-r3.sh
#   (the template, mode 755); probes bench/chat_greedy.py and loop_prefixes.py -> /srv/qwen5090/probes/; this file ->
#   /srv/qwen5090/r785-promote-rebase-r3.sh (.new + mv each). loop_prefixes.py and hermes-tools-r779.json read a private
#   agent session (R781's snapshot, the p107 and p700 prefixes) and are not in this repository; without them G5's p107
#   and p700 probes cannot be reproduced. fn_greedy.py is not in this repository either; the host's probe names map to
#   bench/ as fn_bench.py = probe.py, fn_needle_oai.py = needle.py, the others by the same name.
# Launch (right after R784; EXPECT_TPL = md5 of the deployed template):
#   sudo systemd-run --unit=r785-promote-rebase-r3 --collect -p RuntimeMaxSec=86400 -p TimeoutStopSec=900 \
#     -p Environment=HOME=$HOME -E EXPECT_TPL=<md5> /usr/bin/bash /srv/qwen5090/r785-promote-rebase-r3.sh
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
UNIT=r785-promote-rebase-r3
D=/srv/qwen5090
R=$D/results/$(date +%F)-$UNIT-$(date +%H%M)
[ -e "$R/audit.log" ] && R=$R-$(date +%S)
mkdir -p "$R"
SUM=$R/summary.txt; : > "$SUM"
LIVE=$D/launch-flashnext.sh
TPL=$D/launch-flashnext-r785-rebase-r3.sh
NEW=$D/launch-flashnext.sh.r785
PRE=$D/launch-flashnext.sh.pre-r785
EXPECT_OLD=${EXPECT_OLD:-3f8c18109f1c6dc16cf0701761993f38}   # R783's launcher file on the host (md5 of the host's copy)
EXPECT_TPL=${EXPECT_TPL:?set EXPECT_TPL to the md5 of the deployed template}
IMG_OLD=${IMG_OLD:-tabbyapi:stack-r3-rows32-tokcount-loopthink3}
IMG_NEW=${IMG_NEW:-tabbyapi:rebase-dev-r3}
TUNE_NEW=$D/.exl3cache-rebase-dev-r3
TILED=EXL3_GR_MIX_TILED=1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=$D/models/$MODEL
BASEURL=http://127.0.0.1:8022
API=$BASEURL/v1
FLOOR=884736
STEP=16384
R784_UNIT=${R784_UNIT:-r784-rebase-dev-r3}
R784_DIR=${R784_DIR:-}
P=$D/probes
GREEDY=$P/fn_greedy.py
BENCH=$P/fn_bench.py
CHAT=$P/chat_greedy.py
RHT=$P/replay_hermes_turn.py
LPX=$P/loop_prefixes.py
HTOOLS=$P/hermes-tools-r779.json
REPLAY=$P/agent_replay.py
AGG=$P/tabby_log_agg.py
LM=$D/venv-lmeval/bin/lm_eval
CTL=$D/results/2026-09-27-r781-control                    # R781's Hermes session snapshot (R782 / R783 inputs)
P107=$(ls -d $D/results/2026-09-27-r782-loopthink2-*/ 2>/dev/null | tail -1)prefix-p107.txt
HSET=$D/hermes-set-model.sh
LOOPMSG="I am repeating myself"                           # loop-think r4's LOOP_THINK_MESSAGE (fix.patch)
R722_W=122.75                                             # R722 W1/W2 per-stream mean (r728)
REPLAY_BAR=$(python3 -c "print(round(0.98 * $R722_W, 2))")
R728_REPLAY=128.6
TE_BAR=82.0
GSM_BAR=0.970; GSM_RERUN=0.960
NONCE=$(( $(date +%s) % 100000 ))
SALT_RAMP=$(( (NONCE + 4099) % 100000 )); SALT_S4=$(( (NONCE + 1231) % 100000 )); SALT_S8=$(( (NONCE + 2462) % 100000 ))
SALT_120=$(( (NONCE + 6007) % 100000 ))
CLEAN_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PROMOTED=0; DECIDED=0; HERMES_STOPPED=
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
note(){ log "$*"; echo "$*" >> "$SUM"; }
decide(){ [ "$DECIDED" = 1 ] && return 0; DECIDED=1; echo "DECISION: $1" >> "$SUM"; log "DECISION: $1"; }
iid(){ sudo docker image inspect "$1" --format '{{.Id}}' 2>/dev/null; }
ilbl(){ sudo docker image inspect "$1" --format "{{ index .Config.Labels \"$2\" }}" 2>/dev/null; }
md5of(){ md5sum < "$1" | cut -c1-32; }
envline(){ sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$1" | head -1; }
defval(){ sed -n "s/^$1=\\\${$1:-\\([^}]*\\)}.*/\\1/p" "$2" | head -1; }   # the default of KEY=${KEY:-...}
cp "$0" "$R/" 2>/dev/null
trap 'log "signal before the lock: nothing mutated"; exit 4' TERM INT HUP

# ---------------- preflight (before the lock and again under it) ----------------
# structural A B EXPECT_CACHE EXPECT_SPLIT: B = A with exactly DAILY_IMG / CACHE / GPU_SPLIT / EXTRA_ENV / TUNEDIR changed
# (+ comments): anything else means the live launcher moved since the template was cut, and promoting would revert it.
structural(){ local a=$1 b=$2 k
  for k in DAILY_IMG CACHE GPU_SPLIT EXTRA_ENV TUNEDIR; do
    [ "$(grep -c "^$k=" "$b")" = 1 ] || { log "structural: $b has $(grep -c "^$k=" "$b") '$k=' lines (want 1)"; return 1; }; done
  grep -vE '^[[:space:]]*#' "$a" | grep -vE '^(DAILY_IMG|CACHE|GPU_SPLIT|EXTRA_ENV|TUNEDIR)=' > "$R/.code-a"
  grep -vE '^[[:space:]]*#' "$b" | grep -vE '^(DAILY_IMG|CACHE|GPU_SPLIT|EXTRA_ENV|TUNEDIR)=' > "$R/.code-b"
  diff "$R/.code-a" "$R/.code-b" > "$R/launcher-code-diff.txt" || { log "structural: $b differs from $a outside the five lines (launcher-code-diff.txt)"; return 1; }
  [ "$(grep -oE '^DAILY_IMG=\S+' "$b" | cut -d= -f2)" = "$IMG_NEW" ] || { log "structural: $b DAILY_IMG is not $IMG_NEW"; return 1; }
  [ "$(defval CACHE "$b")" = "$3" ] || { log "structural: $b CACHE default '$(defval CACHE "$b")' != '$3'"; return 1; }
  [ "$(defval GPU_SPLIT "$b")" = "$4" ] || { log "structural: $b GPU_SPLIT default '$(defval GPU_SPLIT "$b")' != '$4'"; return 1; }
  [ "$(defval TUNEDIR "$b")" = "$TUNE_NEW" ] || { log "structural: $b TUNEDIR default '$(defval TUNEDIR "$b")' != $TUNE_NEW"; return 1; }
  [ "$(envline "$b")" = "$(envline "$a") $TILED" ] || { log "structural: $b EXTRA_ENV != live + $TILED"; return 1; }
  [ "$(envline "$b" | wc -w | tr -dc 0-9)" = 42 ] || { log "structural: $b EXTRA_ENV has $(envline "$b" | wc -w | tr -dc 0-9) keys (want 42)"; return 1; }
  return 0; }
preflight(){ local f why=
  for f in "$LIVE" "$TPL" "$CKPT/config.json" "$LM" "$GREEDY" "$BENCH" "$CHAT" "$RHT" "$LPX" "$HTOOLS" "$REPLAY" "$AGG" \
           "$P/agentic-edit.py" "$P/fn_needle_oai.py" "$P/tooleval_summary.py" "$P/nostop_proxy.py" "$P107" \
           "$CTL/session.json" "$CTL/state.db" "$HSET" "$D/lib/gpu-queue.sh" "$D/lib/serve-ctl.sh" "$D/lib/gateway-drain.sh"; do
    [ -e "$f" ] || why="$why missing $f;"; done
  [ -n "$why" ] && { log "preflight:$why"; return 1; }
  command -v tool-eval-bench >/dev/null || { log "preflight: tool-eval-bench not on PATH"; return 1; }
  grep -q 'LADDER_PROD' "$REPLAY" && grep -q 'max_fail_streak' "$REPLAY" || { log "preflight: agent_replay.py lacks the ladder / failure guard"; return 1; }
  grep -q -- '--long' "$CHAT" || { log "preflight: chat_greedy.py lacks --long (install the R785 probe)"; return 1; }
  grep -q 'prefix-p700' "$LPX" || { log "preflight: loop_prefixes.py lacks the p700 prefix (install the R785 probe)"; return 1; }
  [ "$(md5of "$LIVE")" = "$EXPECT_OLD" ] || { log "preflight: live launcher md5 $(md5of "$LIVE") != $EXPECT_OLD"; return 1; }
  [ "$(grep -oE '^DAILY_IMG=\S+' "$LIVE" | cut -d= -f2)" = "$IMG_OLD" ] || { log "preflight: live DAILY_IMG is not $IMG_OLD"; return 1; }
  grep -q '^CKPT_NAME=${CKPT_NAME:-qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab}' "$LIVE" || { log "preflight: live CKPT_NAME is not the r0b0tlab 2.50 daily"; return 1; }
  grep -q '^CKPT_NAME=${CKPT_NAME:-qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab}' "$TPL" || { log "preflight: template CKPT_NAME is not the r0b0tlab 2.50 daily (a finetune checkpoint is not promoted)"; return 1; }
  [ "$(md5of "$TPL")" = "$EXPECT_TPL" ] || { log "preflight: template md5 $(md5of "$TPL") != $EXPECT_TPL"; return 1; }
  bash -n "$TPL" || { log "preflight: template does not parse"; return 1; }
  structural "$LIVE" "$TPL" @FOUND@ @SPLIT@ || { log "preflight: template is not live + the five R785 lines"; return 1; }
  return 0; }
preflight || { log "ABORT (pre-lock preflight): nothing touched"; decide "NOT PROMOTED (pre-lock preflight)"; exit 3; }
log "start: live md5 $EXPECT_OLD ($IMG_OLD), template md5 $EXPECT_TPL; results $R"

# ---------------- queue: register only once R784 has (after its build), then wait for R784 to end ----------------
r784_registered(){ local m=$D/gpu-queue/$R784_UNIT; [ -e "$m" ] && kill -0 "$(cat "$m" 2>/dev/null)" 2>/dev/null; }
if systemctl is-active -q "$R784_UNIT"; then
  log "waiting for $R784_UNIT to register in the GPU queue (after its image build): registering earlier would make a finishing unit skip the daily restore for the whole build"
  while systemctl is-active -q "$R784_UNIT" && ! r784_registered; do sleep 20; done
fi
export GPU_QUEUE_NAME=$UNIT
. $D/lib/gpu-queue.sh
. $D/lib/serve-ctl.sh
. $D/lib/gateway-drain.sh
SCTL_LOG="$R/audit.log"
SCTL_API=$API
# review F4: registered but not yet holding the lock. R784 skips its restore because we are queued; if this unit is
# stopped now and the GPUs are free, bring the daily back before exiting (keyed on the lock, not on R784 being gone).
trap 'log "signal while queued"; exec 9>/srv/qwen5090/gpu-exclusive.lock; if flock -n 9 && [ -z "$(served_id)" ]; then log "GPUs free and :8022 empty: booting the daily from $LIVE"; env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" > "$R/boot-queued-restore.log" 2>&1; fi; exit 4' TERM INT HUP
if systemctl is-active -q "$R784_UNIT"; then
  log "registered; waiting for $R784_UNIT to end (its finish_restore skips the daily: this unit is queued)"
  while systemctl is-active -q "$R784_UNIT"; do sleep 30; done
fi

# ---------------- exits ----------------
running_img(){ sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'; }
hermes_off(){ local c
  for c in hermes hermes-webui; do
    [ "$(sudo docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" = true ] || continue
    sudo docker stop -t 30 "$c" >/dev/null 2>&1 && HERMES_STOPPED="$HERMES_STOPPED $c"; done
  log "Hermes stopped for the gates:${HERMES_STOPPED:- none was running (left as is)}"; }
hermes_back(){ local was=$HERMES_STOPPED; HERMES_STOPPED=
  [ -n "$was" ] && sudo docker start $was >/dev/null 2>&1
  if [ -z "$was" ] && [ "$(sudo docker inspect -f '{{.State.Running}}' hermes 2>/dev/null)" != true ]; then log "Hermes was not running before the unit: left stopped"; return 0; fi
  [ "$(served_id)" = "$MODEL" ] || { log "WARN: :8022 does not serve $MODEL: Hermes${was:+ restarted but} not repointed"; return 0; }
  PORT=8022 MODEL_ID=$MODEL bash "$HSET" >> "$R/audit.log" 2>&1 && log "Hermes on :8022" || log "WARN: Hermes not repointed"; }
wrapup(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; sudo chown -R "$(stat -c %U $D/results)" "$R" 2>/dev/null || true; log "=== $UNIT $1 ==="; }
# The launcher was NOT swapped: make sure the (old) daily serves from $LIVE on :8022, without a bounce when it already does.
ensure_daily(){
  if [ "$(served_id)" = "$MODEL" ] && [ "$(running_img)" = "$(grep -oE '^DAILY_IMG=\S+' "$LIVE" | cut -d= -f2)" ]; then
    log "daily already serving from $LIVE ($(running_img))"; return 0; fi
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
    && log "daily restarted on the rolled-back launcher: $(running_img); $(grep -aoE 'VRAM free MiB [0-9/]+' "$R/boot-rollback.log" | tail -1)" \
    || log "ROLLBACK BOOT FAILED: $(tail -3 "$R/boot-rollback.log" | tr '\n' ' ' | cut -c1-200)"
  hermes_back; decide "ROLLED BACK ($1)"; wrapup "ROLLED BACK"; exit 3; }
on_signal(){ log "signal"; [ "$PROMOTED" = 1 ] && rollback "signal (needrestart / operator stop): VOID"; not_promoted "aborted by signal"; }

gpu_lock
trap on_signal TERM INT HUP
gateway_drain   # the gates run on the live :8022 port; Olla routes nothing to it until this unit exits
log "lock held; served $(served_id || echo none) ($(running_img)); VRAM free $(vram_free)"
preflight || not_promoted "post-lock preflight"

# ---------------- R784's hand-off ----------------
[ -n "$R784_DIR" ] || R784_DIR=$(ls -d $D/results/*-r784-rebase-dev-r3-* 2>/dev/null | sort | tail -1)
PENV=$R784_DIR/promote.env
[ -n "$R784_DIR" ] && [ -f "$PENV" ] || not_promoted "no R784 promote.env (${R784_DIR:-no R784 results dir})"
[ -n "$(find "$PENV" -mmin -1440)" ] || not_promoted "R784 promote.env older than 24 h ($PENV)"
pe(){ sed -n "s/^$1=//p" "$PENV" | tail -1; }
cp "$PENV" "$R/r784-promote.env"; cp "$R784_DIR/summary.txt" "$R/r784-summary.txt" 2>/dev/null
DEC=$(pe DECISION); FOUND=$(pe FOUND); SPLIT=$(pe SPLIT); S1_UP=$(pe S1_UP)
log "R784 ($R784_DIR): DECISION=$DEC FOUND=$FOUND SPLIT=$SPLIT S1_UP=$S1_UP"
[ "$(pe R784_FINISHED)" = 1 ] || not_promoted "R784 did not finish ($PENV)"
# USER_ACCEPT_DECODE=1 (2026-09-27, after R784: decode accepted as flat, promotion on the quality gates): accept a NOT-A-CANDIDATE whose ONLY
# failing clause is decode (the RULE line names "decode"; pool and prefill passed), and run every quality gate as usual.
# Any other failing clause (pool, prefill, health, VOID) still aborts.
if [ "$DEC" != CANDIDATE ]; then
  rule=$(grep -a '^RULE' "$R784_DIR/summary.txt" 2>/dev/null | tr '\n' ' ')
  if [ "${USER_ACCEPT_DECODE:-0}" = 1 ] && [ "$DEC" = NOT-A-CANDIDATE ] && [ "$(grep -ac '^RULE' "$R784_DIR/summary.txt")" = 1 ] \
     && python3 -c 'import sys; r=sys.argv[1].strip(); p="RULE NOT-A-CANDIDATE ("; assert r.startswith(p) and r.endswith(")"); cl=r[len(p):-1].split("; "); assert cl and all(c.startswith("decode ") for c in cl)' "$rule" 2>/dev/null \
     && [ -n "$FOUND" ] && [ "$FOUND" -ge 884736 ]; then
    note "R784 decision $DEC accepted by the user on decode only (USER_ACCEPT_DECODE=1): $rule"
  else
    not_promoted "R784 decision $DEC: $(grep -a '^DECISION' "$R784_DIR/summary.txt" 2>/dev/null | cut -c1-200)"
  fi
fi
[[ "$FOUND" =~ ^[0-9]+$ ]] && [ "$FOUND" -ge "$FLOOR" ] && [ $(( FOUND % STEP )) = 0 ] || not_promoted "R784 FOUND '$FOUND' not >= $FLOOR on the $STEP grid"
[[ "$SPLIT" =~ ^[0-9.]+,[0-9.]+$ ]] || not_promoted "R784 SPLIT '$SPLIT' unreadable"
[[ "$S1_UP" =~ ^[0-9]+/[0-9]+$ ]] || not_promoted "R784 S1_UP '$S1_UP' unreadable"
[ "$(pe P_IMG)" = "$IMG_NEW" ] || not_promoted "R784 measured $(pe P_IMG), not $IMG_NEW"
[ "$(iid "$IMG_NEW")" = "$(pe P_IMG_ID)" ] || not_promoted "$IMG_NEW id $(iid "$IMG_NEW") != the image R784 measured $(pe P_IMG_ID)"
[ "$(ilbl "$IMG_NEW" local.rebase.tree_sha256)" = "$(pe P_TREE_SHA)" ] || not_promoted "$IMG_NEW tree label != R784's P_TREE_SHA"
[ "$(ilbl "$IMG_NEW" local.loopthink.round)" = r4 ] || not_promoted "$IMG_NEW does not carry loop-think r4"
[ "$(ilbl "$IMG_NEW" local.rebase.round)" = rebase-dev-r3 ] || not_promoted "$IMG_NEW is not the rebase-dev-r3 build (label local.rebase.round)"
[ "$(pe LIVE_MD5)" = "$EXPECT_OLD" ] || not_promoted "R784 measured against launcher md5 $(pe LIVE_MD5), live is $EXPECT_OLD"
iid "$IMG_OLD" >/dev/null || not_promoted "rollback image $IMG_OLD missing"
S0=${S1_UP%/*}; S1=${S1_UP#*/}
SPLITSP=${SPLIT//,/, }
{ echo "R784 ($R784_DIR):"; grep -aE '^(P vs S|P pool|A/A|report|RULE|DECISION)' "$R784_DIR/summary.txt" | sed 's/^/  /'; } >> "$SUM"

# ---------------- render + guard ----------------
sed -e "s/@FOUND@/$FOUND/g" -e "s/@SPLIT@/$SPLITSP/g" "$TPL" > "$NEW.tmp" && chmod 755 "$NEW.tmp" && mv -f "$NEW.tmp" "$NEW" \
  || not_promoted "could not render $NEW"
grep -q '@FOUND@\|@SPLIT@' "$NEW" && not_promoted "rendered launcher still holds a placeholder"
bash -n "$NEW" || not_promoted "rendered launcher does not parse"
structural "$LIVE" "$NEW" "$FOUND" "$SPLITSP" || not_promoted "guard: rendered launcher is not live + the five R785 lines"
NEW_MD5=$(md5of "$NEW")
cp -p "$NEW" "$R/launch-flashnext-r785-rendered.sh"
note "rendered $NEW (md5 $NEW_MD5) from the template (md5 $EXPECT_TPL): pool $FOUND, split [$SPLITSP], 42 keys, $IMG_NEW; repo copy $R/launch-flashnext-r785-rendered.sh"

# ---------------- helpers (r728) ----------------
errcounts(){ echo "OOM $(grep -acE 'OutOfMemoryError|out of memory|graph\.cu' "$1"); alloc-retry $(grep -acE 'retries\+[1-9]|num_alloc_retries[^0-9]*[1-9]' "$1"); TORCH_CHECK $(grep -acE 'TORCH_CHECK|c10::Error' "$1"); tracebacks $(grep -ac Traceback "$1")"; }
errsum(){ grep -acE 'OutOfMemoryError|out of memory|graph\.cu|retries\+[1-9]|TORCH_CHECK|c10::Error|Traceback' "$1"; }
oomsum(){ grep -acE 'OutOfMemoryError|out of memory|graph\.cu|retries\+[1-9]|num_alloc_retries[^0-9]*[1-9]' "$1"; }
alive(){ [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
okrows(){ python3 -c 'import json,sys,os
rs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()] if os.path.exists(sys.argv[1]) else []
print(sum(1 for r in rs if r.get("ok")), len(rs))' "$1" 2>/dev/null || echo "0 0"; }
clog(){ sudo docker logs flashnext > "$R/container.log" 2>&1; }

# ---------------- install + boot (the production config: tier on) ----------------
hermes_off
[ "$(md5of "$LIVE")" = "$EXPECT_OLD" ] || not_promoted "guard: live launcher changed before the swap"
[ "$(md5of "$NEW")" = "$NEW_MD5" ] || not_promoted "guard: rendered launcher changed before the swap"
cp -p "$LIVE" "$PRE" && [ "$(md5of "$PRE")" = "$EXPECT_OLD" ] || not_promoted "guard: could not write the rollback copy $PRE"
PROMOTED=1   # before the copy: a signal mid-swap must roll back, never boot an ungated launcher (review F5)
cp -p "$NEW" "$LIVE.new" && mv -f "$LIVE.new" "$LIVE"
cmp -s "$NEW" "$LIVE" || rollback "install copy (live != rendered)"
note "launcher installed: $EXPECT_OLD -> $NEW_MD5 (rollback $PRE)"
served_stop; wait_unserved 45 || true
env -i HOME="$HOME" PATH="$CLEAN_PATH" bash "$LIVE" > "$R/boot-new.log" 2>&1 && wait_served_id "$MODEL" 240 10 \
  || { sudo docker logs flashnext > "$R/container-noboot.log" 2>&1; rollback "G1 the new daily did not boot"; }
BOOT_T=$(date -u +%Y-%m-%dT%H:%M:%SZ)
generation_state "$MODEL" | grep -q GEN_SANE || rollback "G1 generation not sane"
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || log "WARN: gateway not idle after 900 s"

# ---- G1 ----
sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$R/env-new.txt"
got=$(running_img)
pool=$(grep -aoE 'cache [0-9]+ @' "$R/boot-new.log" | tail -1 | tr -dc '0-9')
spl=$(grep -aoE 'split \[[^]]*\]' "$R/boot-new.log" | tail -1)
up=$(grep -aoE 'VRAM free MiB [0-9]+/[0-9]+' "$R/boot-new.log" | tail -1 | grep -oE '[0-9]+/[0-9]+'); f0=${up%/*}; f1=${up#*/}
keys=$(assert_env_keys "$R/boot-new.log" 42 EXL3_GR_MIX_TILED 2>&1); krc=$?
note "G1: image $got ($(ilbl "$got" local.loopthink.round)), keys rc $krc ${keys:+($keys)}, tiled $(grep -c "^$TILED\$" "$R/env-new.txt"), tier $(grep -c '^EXL3_NVME_TIER=' "$R/env-new.txt"), pool $pool, $spl, served $(served_id), UP free $up (floor $(( S0 - 32 ))/$(( S1 - 32 )) = R784 S1 $S1_UP - 32)"
[ "$got" = "$IMG_NEW" ] && [ "$krc" = 0 ] && grep -qx "$TILED" "$R/env-new.txt" && grep -q '^EXL3_NVME_TIER=' "$R/env-new.txt" \
  && [ "$pool" = "$FOUND" ] && [ "$spl" = "split [$SPLITSP]" ] && grep -q "tunedir $TUNE_NEW," "$R/boot-new.log" && [ "$(served_id)" = "$MODEL" ] \
  && [ "${f0:-0}" -ge $(( S0 - 32 )) ] 2>/dev/null && [ "${f1:-0}" -ge $(( S1 - 32 )) ] 2>/dev/null \
  && note "G1 PASS" || { note "G1 FAIL"; rollback "G1"; }

# ---- G2: new greedy reference (fn_greedy /v1/completions + chat_greedy incl. the long requeue-crossing answer) ----
python3 "$GREEDY" --url "$BASEURL" --tag R785 --out "$R/greedy.jsonl" > "$R/greedy-R785.log" 2>&1
gn=$(grep -c '"tag": "R785"' "$R/greedy.jsonl" 2>/dev/null); ge=$(grep -ac ' ERROR ' "$R/greedy-R785.log")
python3 "$CHAT" --url "$API" --model "$MODEL" --tag R785 --long --out "$R/chat-greedy.jsonl" > "$R/chat-greedy-R785.log" 2>&1; crc=$?
cg=$(python3 - "$R/chat-greedy.jsonl" <<'PY' 2>&1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if r["tag"] == "R785"]
pids = [r["pid"] for r in rows]
lg = [r for r in rows if r["pid"] == "long"]
ok = sorted(pids) == sorted(["code", "math", "prose", "tool", "multi", "long"]) and len(lg) == 1 \
     and lg[0].get("crossed_requeue") and lg[0]["finish"] in ("stop", "length")
print(("OK" if ok else "BAD") + f" rows {len(rows)} pids {pids} long tokens {lg[0]['completion_tokens'] if lg else None} "
      f"finish {lg[0]['finish'] if lg else None}")
PY
)
note "G2: fn_greedy $gn records / $ge errors; chat_greedy rc $crc: $cg (new reference: $R/greedy.jsonl + chat-greedy.jsonl, tag R785)"
[ "$gn" = 6 ] && [ "$ge" = 0 ] && [ "$crc" = 0 ] && [ "${cg:0:2}" = OK ] && alive && note "G2 PASS (recorded; identity vs R747 N/A)" \
  || { note "G2 FAIL"; rollback "G2"; }

# ---- G3: ramp + stress (r728 sequence) ----
python3 "$BENCH" --url "$API" --model "$MODEL" --tag "ramp-R785" --kind prose --ctx 4000 --tokens 256 \
  --conc 1 2 3 4 5 6 7 8 --runs 1 --warmup-runs 0 --unique --distinct --salt "$SALT_RAMP" --out "$R/ramp.jsonl" > "$R/ramp.log" 2>&1
clog; read -r okn tot <<< "$(okrows "$R/ramp.jsonl")"; e=$(errsum "$R/container.log")
note "G3 ramp c1..c8: $okn/$tot ok, $(errcounts "$R/container.log"), free $(vram_free)"
[ "$okn" = 36 ] && [ "$tot" = 36 ] && [ "$e" = 0 ] && alive || { note "G3 ramp FAIL"; rollback "G3 ramp"; }
for c in 4 8; do
  [ "$c" = 4 ] && s=$SALT_S4 || s=$SALT_S8
  python3 "$BENCH" --url "$API" --model "$MODEL" --tag "stress-c$c-26k" --kind prose --tokens 256 --conc $c \
    --runs 1 --ctx 26000 --unique --salt "$s" --out "$R/stress-c$c.jsonl" > "$R/stress-c$c.log" 2>&1
  clog; read -r okn tot <<< "$(okrows "$R/stress-c$c.jsonl")"; o=$(oomsum "$R/container.log")
  note "G3 stress c$c @26k: $okn/$tot ok (want $c), $(errcounts "$R/container.log"), free $(vram_free)"
  [ "$okn" = "$c" ] && [ "$tot" = "$c" ] && [ "$o" = 0 ] && alive || { note "G3 stress c$c FAIL"; rollback "G3 stress c$c"; }
done
note "G3 PASS"
python3 "$BENCH" --url "$API" --model "$MODEL" --tag "pf-R785-120000" --kind prose --tokens 64 --conc 1 --runs 1 --ctx 120000 --unique \
  --salt "$SALT_120" --out "$R/prefill-120k.jsonl" > "$R/prefill-120k.log" 2>&1
note "120k cold prefill (production boot, tier on; G4 input, speed reported): $(python3 -c 'import json,sys
r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(" ".join("ok=%s %d tok in %.2f s = %.0f t/s" % (x.get("ok"), x["prompt_tokens"], x["ttft_s"], x["prompt_tokens"]/x["ttft_s"]) for x in r if x.get("ttft_s")) or "no record")' "$R/prefill-120k.jsonl" 2>/dev/null); free $(vram_free)"

# ---- G5: loop-think r4 on the daily template ----
T5=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 "$LPX" "$CTL/session.json" "$R" "$API" > "$R/loop-prefixes.txt" 2>&1; tee -a "$R/audit.log" < "$R/loop-prefixes.txt"
[ -s "$R/prefix-nothink.txt" ] && [ -s "$R/prefix-p700.txt" ] && [ -s "$R/prefix-p700.json" ] || { note "G5 FAIL: prefixes not built"; rollback "G5 prefixes"; }
RP="python3 $RHT --url $API --model $MODEL"
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
def copies(r):   # generated copies of the p700 block (the stream may or may not echo the 2 prefilled copies)
    t = r.get("reasoning") or ""
    return t.count(anchor) - (2 if t.lstrip().startswith("Let me restate the plan.") else 0)
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
      f"copies {[copies(r) for r in L]}, classes {[r['class'] for r in L]}; window inj {counts['inj']} nof {counts['nof']} eng {counts['eng']}")
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
echo "$g5" | grep -q '^G5 OK p700 NOT-EXERCISED' && note "G5 PASS with r4's long-period case NOT EXERCISED (the model left the loop in every row, twice): reported to the user, not a rollback" \
  || note "G5 PASS"

# ---- G6: needles (R728 P2) ----
python3 "$P/fn_needle_oai.py" --url "$API" --model "$MODEL" --tag needle --ctx-tokens 131072 240000 \
  --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee "$R/needle.log" | tee -a "$R/audit.log"
[ "$(grep -c '5/5 retrieved' "$R/needle.log")" = 2 ] && alive && note "G6 needles PASS (5/5 at 131k and 240k)" || { note "G6 needles FAIL"; rollback "G6 needles"; }

# ---- G4: functional headroom over ramp + stress + 120k + needles ----
clog; af=$(vram_free); a0=$(echo $af | cut -d' ' -f1); a1=$(echo $af | cut -d' ' -f2)
o=$(oomsum "$R/container.log")
note "G4 headroom: $(errcounts "$R/container.log") since boot; restarts $(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null); free after the sequence $a0/$a1 MiB (UP $up; R741: P 277/631 vs S 313/847 after a 120k, reported not compared); cuda:1 floor 500 (R728). Allocator-retry: no logger in this stack (pattern kept for decode-floor r7's 'retries+N'), so effectively OOM-only"
fl=met; [ "${a1:-0}" -ge 500 ] || fl="NOT met"
[ "$o" = 0 ] && alive && note "G4 PASS (cuda:1 500 MiB floor REPORT-ONLY: $fl)" || { note "G4 FAIL"; rollback "G4 headroom"; }

# ---- G7: agent replay (r728 G7, R722 flags), scored over the replay window only ----
T7=$(date -u +%Y-%m-%dT%H:%M:%SZ)
timeout 1300 python3 "$REPLAY" --url "$API" --model "$MODEL" \
  --prod-ladder --respawn --drain --max-fail-streak 3 --gen 700 --tool 2020 --think 11 --stagger 3 --temp 0.6 \
  --seed 722 --max-seconds 600 --out "$R/replay.jsonl" > "$R/replay.log" 2>&1
rc=$?
sudo docker logs --since "$T7" flashnext > "$R/container-replay.log" 2>&1
python3 "$AGG" "$R/container-replay.log" > "$R/agg-replay.txt" 2>&1
sed 's/^/  [replay] /' "$R/agg-replay.txt" | grep -E 'window|AGGREGATE|PER STREAM|mean streams|prompt tokens|prefix cached|MTP acceptance' | tee -a "$R/audit.log"
pstream=$(python3 -c 'import re,sys
m=re.search(r"PER STREAM\s*:\s*([\d,.]+)", open(sys.argv[1]).read()); print(m.group(1).replace(",","") if m else "")' "$R/agg-replay.txt" 2>/dev/null)
e=$(errsum "$R/container-replay.log"); loops=$(grep -aci 'token loop was detected' "$R/container-replay.log"); linj=$(grep -ac 'reasoning loop detected' "$R/container-replay.log")
[ "$rc" = 0 ] || note "G7 replay rc $rc, the replay's own last lines (a capacity-driven fail streak at the smaller pool is not a speed regression): $(tail -3 "$R/replay.log" | tr '\n' ' ' | cut -c1-300)"
note "G7 replay: rc $rc; per-stream ${pstream:-none} tok/s (bar $REPLAY_BAR = 0.98 x R722 W $R722_W; R728 read $R728_REPLAY); $(errcounts "$R/container-replay.log"); engine loop stops $loops, collector injections $linj"
[ "$rc" = 0 ] && [ "$e" = 0 ] && [ -n "$pstream" ] && python3 -c "import sys; sys.exit(0 if float('$pstream') >= $REPLAY_BAR else 1)" && alive \
  && note "G7 replay PASS" || { note "G7 replay FAIL"; rollback "G7 replay"; }

# ---- G8: agentic-edit (r728 P1) ----
python3 "$P/agentic-edit.py" --url "$API" --model "$MODEL" --tag r785 --conc 1 4 --modes greedy sampled --out "$R/agentic-edit.jsonl" 2>&1 \
  | tee "$R/agentic-edit.log" | grep -a "agentic-edit" | tee -a "$R/audit.log"
[ "$(grep -ac "6/6 ok" "$R/agentic-edit.log")" = 4 ] && alive && note "G8 agentic-edit PASS (4 x 6/6)" || { note "G8 agentic-edit FAIL"; rollback "G8 agentic-edit"; }

# ---- G9: tool-eval 69 x 4, parallel 8 (r728 P3) ----
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "$BASEURL" --model "$MODEL" --temperature 0.6 --top-p 0.95 --top-k 20 \
    --trials 4 --parallel 8 --json-file "$R/tooleval.json" > "$R/tooleval.log" 2>&1 )
python3 "$P/tooleval_summary.py" "$R/tooleval.json" r785 2>&1 | tee "$R/tooleval-summary.txt" | tee -a "$R/audit.log"
read -r mean nsc ntr <<< "$(python3 -c 'import json,sys
t=json.load(open(sys.argv[1]))["trial_statistics"]; ps=t.get("per_scenario",{})
print(t["final_score_mean"], len(ps), len(next(iter(ps.values()))["points"]) if ps else 0)' "$R/tooleval.json" 2>/dev/null || echo "none 0 0")"
[ "$nsc" = 69 ] && [ "$ntr" = 4 ] || { note "G9 tool-eval VOID: $nsc scenarios x $ntr trials (want 69 x 4)"; rollback "G9 tool-eval VOID (records $nsc x $ntr)"; }
python3 -c "import sys; sys.exit(0 if float('$mean') >= $TE_BAR else 1)" 2>/dev/null && alive \
  && note "G9 tool-eval PASS ($mean >= $TE_BAR; daily 87.8, band 84.5-88.0)" || { note "G9 tool-eval FAIL (${mean:-unparsed})"; rollback "G9 tool-eval (${mean:-unparsed})"; }

# ---- G10: GSM8K n=500 (r728 P4), c8; 0.960-0.970 -> c4 re-run decides ----
# Sample count = unique doc_id: lm-eval writes one samples line per doc per filter (strict-match + flexible-extract),
# so 500 docs are 1,000 lines (R785b VOID'd a 0.982 on that).
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
note "G10 GSM8K c8: flexible-extract ${g8} on $n8 samples (bar $GSM_BAR; c4 reference 0.974)"
[ "$n8" = 500 ] || rollback "G10 GSM8K VOID (c8 samples $n8)"
if python3 -c "import sys; sys.exit(0 if float('$g8') >= $GSM_BAR else 1)" 2>/dev/null; then note "G10 GSM8K PASS (c8 $g8)"
elif python3 -c "import sys; sys.exit(0 if float('$g8') >= $GSM_RERUN else 1)" 2>/dev/null; then
  log "G10: c8 $g8 in [$GSM_RERUN, $GSM_BAR): re-running at c4 (the shape with a same-config reference)"
  read -r g4 n4 <<< "$(gsm 4)"
  note "G10 GSM8K c4 re-run: flexible-extract ${g4} on $n4 samples"
  [ "$n4" = 500 ] || rollback "G10 GSM8K VOID (c4 samples $n4)"
  python3 -c "import sys; sys.exit(0 if float('$g4') >= $GSM_BAR else 1)" 2>/dev/null \
    && note "G10 GSM8K PASS on the c4 re-run ($g4; c8 $g8 borderline: tell the user)" || { note "G10 GSM8K FAIL (c8 $g8, c4 $g4)"; rollback "G10 GSM8K (c8 $g8, c4 $g4)"; }
else note "G10 GSM8K FAIL (c8 $g8 < $GSM_RERUN)"; rollback "G10 GSM8K (c8 $g8)"; fi

# ---- done ----
alive || rollback "server not alive after the gates"
[ "$(md5of "$LIVE")" = "$NEW_MD5" ] || rollback "VOID: live launcher changed under the unit"
sudo docker logs flashnext > "$R/docker-final.log" 2>&1
note "promoted daily over the whole run (reported): $(errcounts "$R/docker-final.log"); free $(vram_free)"
note "reference rollover: queued units comparing to R747's greedy.jsonl must now use $R/greedy.jsonl (tag R785); chat: $R/chat-greedy.jsonl (tag R785). Repo: copy $R/launch-flashnext-r785-rendered.sh to flan/launch-flashnext-tabby.sh"
trap 'log "signal after the decision: ignored"' TERM INT HUP
decide "PROMOTED"
hermes_back
wrapup "PROMOTED: daily = $IMG_NEW @ $FOUND [$SPLITSP] (rollback $PRE, $IMG_OLD @ 983,040)"
