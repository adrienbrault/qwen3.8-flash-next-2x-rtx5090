#!/usr/bin/env bash
# R789 (2026-09-27): promote a bitwise, same-pool simplification of the Flash-Next daily on :8022: three EXTRA_ENV keys
# leave the served launcher (42 -> 39), same image tabbyapi:rebase-dev-r3, same pool 901,120, same split [30, 30], same
# TUNEDIR:
#   EXL3_HOST_GAP_REWIND=1  R788a R0 DROP (null within +-1-2 %); upstream 5783a93's flag-off path builds identical rewind
#                           descriptors (R788a review). The flag-off job keeps OUR _state_words bf16 word count, not
#                           upstream's raw stride: a rebase must keep that hunk (port hazard, R788a review).
#   EXL3_GR_STATE_REGRID=1  never read on the served path (HC_MIX_V2_INT8=1 + MIX_V3 + STATE_IN_UP make its branches dead)
#   EXL3_HC_MIX_V3_PDL=0    the literal default (parsed == "1", default "0")
# The image bakes none of the three (docker image inspect, 2026-09-27: its EXL3_* ENV is MOE_COOP_V2 / SHARED_EXPERT_OVERLAP
# / DECODE_OVERLAP / DECODE_FUSE / MOE_PREFILL_E3 / EMBED_GPU_PRUNED = 0), so "removed" means ABSENT from the container env.
# Expected: greedy byte-identical to R785c's reference (fn_greedy 6/6 + chat_greedy 6/6 incl. --long). Autonomous promotion
# class (bitwise + unchanged pool): PROMOTED only if every gate passes; G2 identity is mandatory.
# Derived from r785-promote-rebase-r3.sh (its -1223 run = R785c, both harness fixes included: chat_greedy's usage:null
# token-count fallback, G10's unique doc_id counter). Changes vs R785:
#   - no R784 hand-off / no rendering: the repo template scripts/launchers/launch-flashnext-r789-dropkeys.sh IS the new launcher; structural
#     guard = the template equals the live launcher outside comments except EXTRA_ENV, and EXTRA_ENV = live's sequence with
#     exactly the three tokens deleted (39 keys);
#   - queue: registers at start, waits while r788b-ablation-mem (AFTER) is registered alive (R788b's finish_restore then
#     skips the daily), takes the lock, boots the new daily itself; queued-signal trap with the daily_ok check (served id +
#     DAILY_IMG + NVMe tier on; r787-chain / r788 pattern);
#   - G1: 39 keys, the three ABSENT, every one of the 39 present verbatim in the container, image id pinned, pool 901,120,
#     split [30, 30], tier on, UP free per card >= R785c's 1181/1759 - 32 MiB; tier "stale namespaces" line reported;
#   - G2: identity REQUIRED vs R785c's greedy.jsonl + chat-greedy.jsonl rows (tag R785, copied into $R); any divergence or a
#     missing row -> rollback;
#   - G5: copies() detects the prefix echo explicitly (R785 review) instead of assuming it from the header text;
#   - G7: probes/replay_steps.py joins the server log to replay.jsonl per session instance: per-stream, tokens/step, ms/step
#     for all requests and for sessions with acceptance >= 0.5 (R786 review); the bar (0.98 x R722 W = 120.3) is scored on
#     the latter so a degenerate low-acceptance trajectory cannot fail the gate; both reported;
#   - Hermes: repointed only when :8022 serves the production daily (daily_ok), else only restarted as it was (config
#     untouched) and left for the unit that restores the daily;
#   - probe calls under `timeout $PROBE_TO` (a hung engine cannot outlive the unit); TimeoutStopSec 1800.
# Gates, in order G1 G2 G3 (+ one 120k cold prefill) G5 G6 G4 G7 G8 G9 G10 (bars as R785); any miss -> .pre-r789 goes back,
# the R785 daily boots, DECISION: ROLLED BACK (<gate>).
# REFERENCE: unchanged. R785c's greedy.jsonl / chat-greedy.jsonl stay the daily's reference (R789's rows are byte-identical
#   when PROMOTED, so either is valid; queued units keep the R785c path).
# UNIT HYGIENE: preflight before the queue and again under the lock; every post-lock abort leaves the daily serving from
#   $LIVE on :8022 (nothing else restores :8022). summary.txt ends with exactly one of: DECISION: PROMOTED |
#   DECISION: ROLLED BACK (<gate>) | DECISION: NOT PROMOTED (<why>).
# GPU ~115 min (+25 on a GSM8K c4 re-run, +5 on a rollback).
# Install (operator, .new + mv each): scripts/launchers/launch-flashnext-r789-dropkeys.sh -> /srv/qwen5090/launch-flashnext-r789-dropkeys.sh
#   (mode 755); replay_steps.py (a host probe, not in this repository) -> /srv/qwen5090/probes/; this file -> /srv/qwen5090/r789-promote-dropkeys.sh.
# Launch WHILE R788b is still running and registered (`ls /srv/qwen5090/gpu-queue/r788b-ablation-mem` exists and
#   `systemctl is-active r788b-ablation-mem`): otherwise R788b restores the daily and R789 bounces it again.
#   EXPECT_TPL = md5 of the deployed template:
#   sudo systemd-run --unit=r789-promote-dropkeys --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=1800 \
#     -p Environment=HOME=$HOME -E EXPECT_TPL=<md5> /usr/bin/bash /srv/qwen5090/r789-promote-dropkeys.sh
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
UNIT=r789-promote-dropkeys
D=/srv/qwen5090
R=$D/results/$(date +%F)-$UNIT-$(date +%H%M)
[ -e "$R" ] && R=$R-$(date +%S)-$$
mkdir -p "$R"
SUM=$R/summary.txt; : > "$SUM"
LIVE=$D/launch-flashnext.sh
TPL=$D/launch-flashnext-r789-dropkeys.sh
PRE=$D/launch-flashnext.sh.pre-r789
EXPECT_OLD=${EXPECT_OLD:-fddd41529039da2135a0f743b7c51270}   # R785's launcher + sampler temperature 1.0 (md5 of the host's copy)
EXPECT_TPL=${EXPECT_TPL:?set EXPECT_TPL to the md5 of the deployed template}
IMG=${IMG:-tabbyapi:rebase-dev-r3}
IMG_ID=${IMG_ID:-sha256:30ed33eb60365a3aca89515162739374832c645378a34e5e131da9ef7317f98b}   # R784 P / R785 / R786 N / R788
TUNE=$D/.exl3cache-rebase-dev-r3
DROP="EXL3_HOST_GAP_REWIND EXL3_GR_STATE_REGRID EXL3_HC_MIX_V3_PDL"
DROPKV="EXL3_HOST_GAP_REWIND=1 EXL3_GR_STATE_REGRID=1 EXL3_HC_MIX_V3_PDL=0"
NKEYS=39
POOL=901120
SPLITSP="30, 30"
REF_UP=${REF_UP:-1181/1759}                                # R785c G1 (= R784 P1/P2 UP) at 901,120 [30, 30]
REF_DIR=${REF_DIR:-$D/results/2026-09-27-r785-promote-rebase-r3-1223}   # R785c: the daily's greedy reference (tag R785)
REF_TAG=R785
TAG=R789
AFTER=${AFTER-r788b-ablation-mem}
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
LOOPMSG="I am repeating myself"                           # loop-think r4's LOOP_THINK_MESSAGE (fix.patch)
R722_W=122.75                                             # R722 W1/W2 per-stream mean (r728)
REPLAY_BAR=$(python3 -c "print(round(0.98 * $R722_W, 2))")
MIN_ACC=0.5
R785_REPLAY="123.7 all / 124.6 acc>=0.5 (3.117 tok/step, 25.20 ms/step)"   # R785c G7 through replay_steps.py
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
# structural A B: B = A outside comments except the EXTRA_ENV line, and B's EXTRA_ENV = A's token sequence with exactly the
# three DROPKV tokens deleted (each present once in A, absent from B, order kept), 39 tokens. Anything else means the live
# launcher moved since the template was cut, and installing the template would revert it.
structural(){ local a=$1 b=$2
  [ "$(grep -c '^EXTRA_ENV=' "$b")" = 1 ] && [ "$(grep -c '^EXTRA_ENV=' "$a")" = 1 ] || { log "structural: want one 'EXTRA_ENV=' line in each of $a and $b"; return 1; }
  grep -vE '^[[:space:]]*#' "$a" | grep -vE '^EXTRA_ENV=' > "$R/.code-a"
  grep -vE '^[[:space:]]*#' "$b" | grep -vE '^EXTRA_ENV=' > "$R/.code-b"
  diff "$R/.code-a" "$R/.code-b" > "$R/launcher-code-diff.txt" || { log "structural: $b differs from $a outside comments and EXTRA_ENV (launcher-code-diff.txt)"; return 1; }
  python3 - "$(envline "$a")" "$(envline "$b")" "$DROPKV" "$NKEYS" <<'PY' >> "$R/audit.log" 2>&1 || { log "structural: EXTRA_ENV is not live minus the three keys (audit.log)"; return 1; }
import sys
a, b, drop, n = sys.argv[1].split(), sys.argv[2].split(), sys.argv[3].split(), int(sys.argv[4])
keys = [kv.split("=")[0] for kv in drop]
ok = True
for kv in drop:
    if a.count(kv) != 1: print(f"structural: live has {a.count(kv)} x {kv} (want 1)"); ok = False
for k in keys:
    if any(t.split("=")[0] == k for t in b): print(f"structural: template still sets {k}"); ok = False
if [t for t in a if t not in drop] != b: print("structural: template EXTRA_ENV != live minus the three tokens (order kept)"); ok = False
if len(b) != n or len(set(t.split("=")[0] for t in b)) != n: print(f"structural: template has {len(b)} tokens / {len(set(t.split('=')[0] for t in b))} distinct keys (want {n})"); ok = False
print("structural EXTRA_ENV: OK" if ok else "structural EXTRA_ENV: BAD")
sys.exit(0 if ok else 1)
PY
  [ "$(grep -oE '^DAILY_IMG=\S+' "$b" | cut -d= -f2)" = "$IMG" ] || { log "structural: $b DAILY_IMG is not $IMG"; return 1; }
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
           "$REF_DIR/greedy.jsonl" "$REF_DIR/chat-greedy.jsonl" "$TUNE"; do
    [ -e "$f" ] || why="$why missing $f;"; done
  [ -n "$why" ] && { log "preflight:$why"; return 1; }
  command -v tool-eval-bench >/dev/null || { log "preflight: tool-eval-bench not on PATH"; return 1; }
  grep -q 'LADDER_PROD' "$REPLAY" && grep -q 'max_fail_streak' "$REPLAY" || { log "preflight: agent_replay.py lacks the ladder / failure guard"; return 1; }
  grep -q -- '--long' "$CHAT" && grep -q 'tokens_source' "$CHAT" || { log "preflight: chat_greedy.py lacks --long / the usage:null fallback"; return 1; }
  grep -q 'prefix-p700' "$LPX" || { log "preflight: loop_prefixes.py lacks the p700 prefix"; return 1; }
  grep -q 'SCORED per_stream=' "$RSTEPS" || { log "preflight: $RSTEPS is not the R789 replay scorer"; return 1; }
  grep -q -- '--temperature' "$RHT" || { log "preflight: $RHT lacks --temperature (G5 pins 0.6)"; return 1; }
  [ "$(refrows "$REF_DIR/greedy.jsonl" $REF_TAG)" = "6 6" ] || { log "preflight: $REF_DIR/greedy.jsonl is not 6 rows tagged $REF_TAG ($(refrows "$REF_DIR/greedy.jsonl" $REF_TAG))"; return 1; }
  [ "$(refrows "$REF_DIR/chat-greedy.jsonl" $REF_TAG)" = "6 6" ] || { log "preflight: $REF_DIR/chat-greedy.jsonl is not 6 rows tagged $REF_TAG"; return 1; }
  [ "$(md5of "$LIVE")" = "$EXPECT_OLD" ] || { log "preflight: live launcher md5 $(md5of "$LIVE") != $EXPECT_OLD"; return 1; }
  [ "$(grep -oE '^DAILY_IMG=\S+' "$LIVE" | cut -d= -f2)" = "$IMG" ] || { log "preflight: live DAILY_IMG is not $IMG"; return 1; }
  [ "$(envline "$LIVE" | wc -w | tr -dc 0-9)" = 42 ] || { log "preflight: live EXTRA_ENV is not 42 keys"; return 1; }
  grep -q '^CKPT_NAME=${CKPT_NAME:-qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab}' "$LIVE" && grep -q '^CKPT_NAME=${CKPT_NAME:-qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab}' "$TPL" \
    || { log "preflight: live / template CKPT_NAME is not the r0b0tlab 2.50 daily"; return 1; }
  [ "$(md5of "$TPL")" = "$EXPECT_TPL" ] || { log "preflight: template md5 $(md5of "$TPL") != $EXPECT_TPL"; return 1; }
  bash -n "$TPL" || { log "preflight: template does not parse"; return 1; }
  structural "$LIVE" "$TPL" || { log "preflight: template is not live minus the three keys"; return 1; }
  [ "$(iid "$IMG")" = "$IMG_ID" ] || { log "preflight: $IMG id $(iid "$IMG") != $IMG_ID (the image R785 promoted)"; return 1; }
  for k in $DROP; do
    sudo docker image inspect "$IMG" -f '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep -q "^$k=" \
      && { log "preflight: $IMG bakes $k (removal would not mean absent): re-plan"; return 1; }; done
  return 0; }
preflight || { log "ABORT (pre-queue preflight): nothing touched"; decide "NOT PROMOTED (pre-queue preflight)"; exit 3; }
log "start: live md5 $EXPECT_OLD (42 keys), template md5 $EXPECT_TPL (39 keys), $IMG ($IMG_ID); AFTER '${AFTER:-}'; results $R"
cp -p "$TPL" "$R/launch-flashnext-r789-dropkeys.sh"; cp -p "$LIVE" "$R/launch-flashnext.sh.at-start"

# ---------------- queue: register now, wait for R788b (AFTER), then the lock ----------------
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
# a signal before the lock: the unit holding the GPUs (R788b) may have skipped its daily restore because THIS unit was
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
# has no FIFO order). This unit's registration is live during the wait, so R788b's finish_restore skips the daily.
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
# leaving the 39-key launcher live ungated; keeps gpu-queue.sh's marker removal.
trap 'rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; if [ "$PROMOTED" = 1 ] && [ "$DECIDED" = 0 ]; then rollback "unit exited without a decision"; fi' EXIT
cp -p "$TPL" "$LIVE.new" && mv -f "$LIVE.new" "$LIVE"
cmp -s "$TPL" "$LIVE" || rollback "install copy (live != template)"
note "launcher installed: $EXPECT_OLD -> $EXPECT_TPL (42 -> 39 keys; rollback $PRE)"
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
neg=; for k in $DROP; do neg="$neg -$k"; done
keys=$(assert_env_keys "$R/boot-new.log" $NKEYS EXL3_GR_MIX_TILED $neg 2>&1); krc=$?
absent=0; for k in $DROP; do grep -q "^$k=" "$R/env-new.txt" && { log "G1: removed key present in the container: $(grep "^$k=" "$R/env-new.txt")"; absent=1; }; done
verb=0; for kv in $(envline "$LIVE"); do grep -qxF "$kv" "$R/env-new.txt" || { log "G1: container env lacks $kv"; verb=1; }; done
sudo docker logs flashnext > "$R/container-boot.log" 2>&1
stale=$(grep -aoE '[0-9]+ stale namespaces removed' "$R/container-boot.log" | tail -1)
note "G1: image $got ($gotid), keys rc $krc ${keys:+($keys)}, three removed keys absent: $([ $absent = 0 ] && echo yes || echo NO), 39 present verbatim: $([ $verb = 0 ] && echo yes || echo NO), tier $(grep -c '^EXL3_NVME_TIER=' "$R/env-new.txt") (${stale:-no 'stale namespaces' line}: reported, a non-zero count means the tier restarted cold), pool $pool, $spl, served $(served_id), UP free $up (floor $(( S0 - 32 ))/$(( S1 - 32 )) = R785c $REF_UP - 32)"
[ "$got" = "$IMG" ] && [ "$gotid" = "$IMG_ID" ] && [ "$krc" = 0 ] && [ "$absent" = 0 ] && [ "$verb" = 0 ] && grep -q '^EXL3_NVME_TIER=' "$R/env-new.txt" \
  && [ "$pool" = "$POOL" ] && [ "$spl" = "split [$SPLITSP]" ] && grep -q "tunedir $TUNE," "$R/boot-new.log" && [ "$(served_id)" = "$MODEL" ] \
  && [ "${f0:-0}" -ge $(( S0 - 32 )) ] 2>/dev/null && [ "${f1:-0}" -ge $(( S1 - 32 )) ] 2>/dev/null \
  && note "G1 PASS" || { note "G1 FAIL"; rollback "G1"; }

# ---- G2: greedy identity vs R785c (fn_greedy /v1/completions + chat_greedy incl. the long requeue-crossing answer) ----
cp "$REF_DIR/greedy.jsonl" "$R/greedy.jsonl"; cp "$REF_DIR/chat-greedy.jsonl" "$R/chat-greedy.jsonl"   # tag R785 rows, then R789 appended
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
note "120k cold prefill (production boot, tier on; G4 input, speed reported; R785c 12009 t/s): $(python3 -c 'import json,sys
r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(" ".join("ok=%s %d tok in %.2f s = %.0f t/s" % (x.get("ok"), x["prompt_tokens"], x["ttft_s"], x["prompt_tokens"]/x["ttft_s"]) for x in r if x.get("ttft_s")) or "no record")' "$R/prefill-120k.jsonl" 2>/dev/null); free $(vram_free)"

# ---- G5: loop-think r4 on the daily template ----
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

# ---- G6: needles (R728 P2) ----
$TO python3 "$P/fn_needle_oai.py" --url "$API" --model "$MODEL" --tag needle --ctx-tokens 131072 240000 \
  --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" > "$R/needle-full.log" 2>&1
grep -E "retrieved|FAIL|Error" "$R/needle-full.log" | tee "$R/needle.log" | tee -a "$R/audit.log" > /dev/null
[ "$(grep -c '5/5 retrieved' "$R/needle.log")" = 2 ] && alive && note "G6 needles PASS (5/5 at 131k and 240k)" || { note "G6 needles FAIL"; rollback "G6 needles"; }

# ---- G4: functional headroom over ramp + stress + 120k + needles ----
clog; af=$(vram_free); a0=$(echo $af | cut -d' ' -f1); a1=$(echo $af | cut -d' ' -f2)
o=$(oomsum "$R/container.log")
note "G4 headroom: $(errcounts "$R/container.log") since boot; restarts $(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null); free after the sequence $a0/$a1 MiB (UP $up; R785c 383/899, reported not compared); cuda:1 floor 500 (R728). Allocator-retry: no logger in this stack, so effectively OOM-only"
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
$TO python3 "$P/agentic-edit.py" --url "$API" --model "$MODEL" --tag r789 --conc 1 4 --modes greedy sampled --out "$R/agentic-edit.jsonl" > "$R/agentic-edit.log" 2>&1
grep -a "agentic-edit" "$R/agentic-edit.log" | tee -a "$R/audit.log" > /dev/null
[ "$(grep -ac "6/6 ok" "$R/agentic-edit.log")" = 4 ] && alive && note "G8 agentic-edit PASS (4 x 6/6)" || { note "G8 agentic-edit FAIL"; rollback "G8 agentic-edit"; }

# ---- G9: tool-eval 69 x 4, parallel 8 (r728 P3) ----
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "$BASEURL" --model "$MODEL" --temperature 0.6 --top-p 0.95 --top-k 20 \
    --trials 4 --parallel 8 --json-file "$R/tooleval.json" > "$R/tooleval.log" 2>&1 )
python3 "$P/tooleval_summary.py" "$R/tooleval.json" r789 2>&1 | tee "$R/tooleval-summary.txt" | tee -a "$R/audit.log"
read -r mean nsc ntr <<< "$(python3 -c 'import json,sys
t=json.load(open(sys.argv[1]))["trial_statistics"]; ps=t.get("per_scenario",{})
print(t["final_score_mean"], len(ps), len(next(iter(ps.values()))["points"]) if ps else 0)' "$R/tooleval.json" 2>/dev/null || echo "none 0 0")"
[ "$nsc" = 69 ] && [ "$ntr" = 4 ] || { note "G9 tool-eval VOID: $nsc scenarios x $ntr trials (want 69 x 4)"; rollback "G9 tool-eval VOID (records $nsc x $ntr)"; }
python3 -c "import sys; sys.exit(0 if float('$mean') >= $TE_BAR else 1)" 2>/dev/null && alive \
  && note "G9 tool-eval PASS ($mean >= $TE_BAR; R785b/c 85.5 / 86.5, R728 band 84.5-88.0)" || { note "G9 tool-eval FAIL (${mean:-unparsed})"; rollback "G9 tool-eval (${mean:-unparsed})"; }

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
note "G10 GSM8K c8: flexible-extract ${g8} on $n8 samples (bar $GSM_BAR; R785b/c 0.982 / 0.980)"
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
note "reference: unchanged, R785c $REF_DIR/greedy.jsonl + chat-greedy.jsonl (tag $REF_TAG); R789's rows are byte-identical. Repo: flan/launch-flashnext-r789-dropkeys.sh -> flan/launch-flashnext-tabby.sh (md5 $EXPECT_TPL); public mirror: the same 3-key drop"
trap 'log "signal after the decision: ignored"' TERM INT HUP
decide "PROMOTED"
hermes_back
wrapup "PROMOTED: daily = $IMG @ $POOL [$SPLITSP], 39 keys (rollback $PRE = R785's 42-key launcher)"
