#!/usr/bin/env bash
# R728 (2026-09-25) — promotion gate: the Flash-Next daily (:8022) with the MTP draft-KV window OFF, pool unchanged at
#   983,040. The user approved the promotion on 2026-09-25; Flash-Next promotions run autonomously once the gates pass.
# Candidate = flan/launch-flashnext-r728-window-off.sh (repo-first) = the live launcher (stack-r3-rows32, 42 keys, +4500
#   memory-offset re-apply) minus EXL3_MTP_KV_WINDOW=16384 in the default EXTRA_ENV line, plus an R728 header block.
#   No image change and no patch: without the window the draft K/V lives in the page-indexed draft cache.
# EVIDENCE (FINDINGS R721-R723, all reviewed):
#   R721  a prompt revived from the prompt cache in a later request group drafts 0.655 accepted/proposed under the window
#         vs 0.868 without it (fresh 0.911 in both); c8 4k code per-stream +12.5 %.
#   R722  agent-shaped replay (2 pairs): window off +2.0 % per-stream decode (W 123.3 / 122.2, N 125.8 / 124.7 tok/s).
#   R723  fits at 983,040: cuda:0 identical (1,125 MiB free at boot, 229 after the heavy sequence), layout identical,
#         greedy 6/6 identical, 0 errors, leg B 4/4; cuda:1 -940 MiB (1,573 at boot, 763 after); ramp c7/c8 +16-19 %.
#         Its cuda:1 clause ("post-sequence >= D - 32") fails any planned relocation by construction; replaced below.
# BOOTS (same unit, one queue lock; env -i; the three probes' shapes are r677's / r723's / r722's):
#   REF    the live launcher, NVME_TIER= : fn_greedy (incl. ~100k) -> ramp c1..c8 (prose 4k, 256 tokens) -> stress c4 and
#          c8 @ ~26k (256 tokens, unique) -> canonical fn_gate RUNS=2 -> free VRAM after the sequence.
#   CAND   the candidate, NVME_TIER= : the same sequence, then the 600 s agent replay on the same boot (gate 7).
#   CANDT  the candidate with the launcher's default tier (NVME_TIER unset: IMG == DAILY_IMG turns the daily's NVMe
#          prefix tier on): fn_greedy -> ramp -> leg B. This boot is what the promoted launcher boots, so PROMOTE is a
#          file swap (.new + mv, rollback copy .pre-r728) and the post-promotion gates run on it without a reboot.
# PRE-GATES (pre-registered; any miss -> DECISION: NOT PROMOTED (<gate>), live launcher untouched, daily restored):
#   G1 greedy     CAND fn_greedy identical to REF on all 6 prompts (REF must itself complete 6/6).
#   G2 ramp       36/36 ramp requests ok, server alive, 0 OOM / TORCH_CHECK / tracebacks in the container so far.
#   G3 stress     c4 and c8 @ ~26k (R661/R667's failure shape): every request ok, 0 OOM lines, server alive.
#   G4 gate       canonical fn_gate RUNS=2, every row ok; per-stream decode >= 0.97x REF in each gated cell: leg A c4/4k,
#                 c8/4k, leg B c4/26k (c1 is reported, not gated: 20 % boot-to-boot noise floor).
#   G5 headroom   cuda:0 free at UP and after the sequence >= REF - 32 MiB (each). cuda:1 draw during the sequence
#                 (UP - after) <= REF's draw + 32 MiB, and cuda:1 free after the sequence >= 500 MiB (R723 review's
#                 replacement clause for a planned cuda:1 relocation).
#   G6 tier       CANDT boots with EXL3_NVME_TIER in the container env; greedy 6 records and no ERROR line, ramp 36/36 ok,
#                 leg B 4/4 ok, 0 OOM / TORCH_CHECK / tracebacks, alive. (Greedy identity vs REF is reported, not gated.)
#   G7 replay     agent_replay.py with R722's exact flags (--prod-ladder --respawn --drain --max-fail-streak 3 --gen 700
#                 --tool 2020 --think 11 --stagger 3 --temp 0.6 --seed 722, 600 s) on the CAND boot, scored by
#                 tabby_log_agg.py over the replay window only (docker logs --since). rc 0, 0 OOM / TORCH_CHECK /
#                 tracebacks, PER STREAM >= 0.98 x R722's W mean 122.75 = 120.3 tok/s. A sanity check, not an A/B: R722
#                 ran before R726 restored the +4500 memory offset, and loop-detected stops alone move an arm by several %
#                 (the count is logged).
# RUN ORDER: REF sequence; CAND sequence -> G1-G5 -> G7 on the same boot; CANDT -> G6; PROMOTE; P1-P4.
# POST-PROMOTION GATES on the promoted daily (r677's bars; any miss -> roll back to .pre-r728, restart the daily with it,
#   DECISION: ROLLED BACK (<gate>)):
#   P1 agentic-edit 4 x "6/6 ok" (c1/c4 x greedy/sampled), alive.   P2 needles 5/5 at 131k and at 240k.
#   P3 tool-eval 69 x 4 @ parallel 8, final_score_mean >= 82.0.     P4 GSM8K n=500 c4 (nostop proxy), flexible >= 0.970.
# GUARDS: the live launcher's md5 is recorded at unit start and re-checked after the queue lock and right before the swap
#   (a concurrent change aborts); the candidate must equal the live launcher minus the window key, plus comment lines only.
# summary.txt ends with exactly one of: DECISION: PROMOTED | DECISION: ROLLED BACK (<gate>) | DECISION: NOT PROMOTED (<pre-gate>).
# GPU ~2 h 15: REF ~20 min, CAND ~20 + replay ~12-17, CANDT ~12, post-promotion ~60 (tool-eval and GSM8K ~20-25 each);
#   +5 min on a rollback. Queue-chained (lib/gpu-queue.sh).
# DEPLOY: the candidate launcher goes to /srv/qwen5090/launch-flashnext-r728-window-off.sh (mode 755); this script to
#   /srv/qwen5090/r728-promote-window-off.sh; run e.g.
#   systemd-run --unit=r728-promote-window-off --collect -p RuntimeMaxSec=43200 bash /srv/qwen5090/r728-promote-window-off.sh
#   Probes used (already on flan): fn_greedy.py fn_bench.py fn_gate.sh agent_replay.py tabby_log_agg.py agentic-edit.py
#   fn_needle_oai.py tooleval_summary.py nostop_proxy.py; lm_eval at /srv/qwen5090/venv-lmeval/bin/lm_eval.
# AFTER PROMOTED: the repo launcher (flan/launch-flashnext-tabby.sh) takes the same edit, and the public Flash-Next repo
#   mirrors it in the same session (CLAUDE.md).
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
UNIT=${UNIT:-$(basename "$0" .sh)}
R=${R:-/srv/qwen5090/results/$(date +%F)-$UNIT}
# fn_greedy / fn_bench append to their --out files: a same-day rerun gets its own directory, or its counts double.
[ -e "$R/audit.log" ] && R=$R-$(date +%H%M)
mkdir -p "$R"
SUM=$R/summary.txt; : > "$SUM"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/$MODEL
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
PRE=/srv/qwen5090/launch-flashnext.sh.pre-r728
CAND=/srv/qwen5090/launch-flashnext-r728-window-off.sh
P=/srv/qwen5090/probes
GREEDY=$P/fn_greedy.py
BENCH=$P/fn_bench.py
GATE=$P/fn_gate.sh
REPLAY=$P/agent_replay.py
AGG=$P/tabby_log_agg.py
R722_W=122.75                       # R722 W1/W2 per-stream mean (123.3 + 122.2) / 2
REPLAY_BAR=$(python3 -c "print(round(0.98 * $R722_W, 2))")
NONCE=$(( $(date +%s) % 100000 ))
SALT_RAMP=$(( (NONCE + 4099) % 100000 ))    # shared by REF and CAND: paired prompts, cold per boot (tier off)
SALT_GATE=$(( (NONCE + 7919) % 100000 ))
SALT_S4=$(( (NONCE + 1231) % 100000 )); SALT_S8=$(( (NONCE + 2462) % 100000 ))
SALT_TRAMP=$(( (NONCE + 3571) % 100000 )); SALT_TB=$(( (NONCE + 5381) % 100000 ))   # CANDT: never seen by the tier
CLEAN_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PROMOTED=0
BOOTED=0
DECIDED=0
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
note(){ log "$*"; echo "$*" >> "$SUM"; }              # a gate line: audit + summary
export GPU_QUEUE_NAME=$UNIT
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"

# ---- exits: summary.txt's last line is the DECISION on every path ----
decide(){ [ "$DECIDED" = 1 ] && return 0; DECIDED=1; echo "DECISION: $1" >> "$SUM"; log "DECISION: $1"; }
wrapup(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== r728 $1 ==="
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; }
not_promoted(){ log "PRE-GATE FAILED ($1): live launcher untouched"
  sudo docker logs flashnext > "$R/docker-final.log" 2>&1
  decide "NOT PROMOTED ($1)"; finish_restore "$LIVE"; wrapup "NOT PROMOTED"; exit 3; }
# A rollback always restarts the daily from the restored launcher, even with a queued unit waiting: finish_restore would
# skip the boot then and leave the candidate container serving under a rolled-back launcher.
rollback(){ log "GATE FAILED ($1): rolling back to $PRE"
  sudo docker logs flashnext > "$R/docker-final.log" 2>&1
  if [ "$PROMOTED" = 1 ]; then
    sudo cp -p "$PRE" "$LIVE.new" && sudo mv -f "$LIVE.new" "$LIVE"
    cmp -s "$PRE" "$LIVE" && log "live launcher restored from $PRE (md5 $(md5sum < "$LIVE" | cut -c1-8))" \
      || log "ROLLBACK COPY FAILED: restore $LIVE from $PRE by hand"
    PROMOTED=0
  fi
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH="$CLEAN_PATH" bash "$LIVE" > "$R/boot-rollback.log" 2>&1 && wait_served_id "$MODEL" 200 8 \
    && log "daily restarted on the rolled-back launcher: $(served_id); $(grep -aoE 'VRAM free MiB [0-9/]+' "$R/boot-rollback.log" | tail -1)" \
    || log "ROLLBACK BOOT FAILED: $(tail -3 "$R/boot-rollback.log" | tr '\n' ' ' | cut -c1-200)"
  BOOTED=0
  decide "ROLLED BACK ($1)"; wrapup "ROLLED BACK"; exit 3; }
on_signal(){ log "signal"
  [ "$PROMOTED" = 1 ] && rollback "signal"
  decide "NOT PROMOTED (aborted by signal)"; finish_restore "$LIVE"; wrapup ABORTED; exit 4; }
trap on_signal TERM INT HUP

# ---- preconditions and guards (before the lock: cheap) ----
for f in "$LIVE" "$CAND" "$CKPT/config.json" "$LM" "$GREEDY" "$BENCH" "$GATE" "$REPLAY" "$AGG" "$P/agentic-edit.py" \
         "$P/fn_needle_oai.py" "$P/tooleval_summary.py" "$P/nostop_proxy.py"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; decide "NOT PROMOTED (missing $f)"; wrapup ABORTED; exit 3; }; done
command -v tool-eval-bench >/dev/null \
  || { log "ABORT: tool-eval-bench not on PATH"; decide "NOT PROMOTED (missing tool-eval-bench)"; wrapup ABORTED; exit 3; }
grep -q 'LADDER_PROD' "$REPLAY" && grep -q 'max_fail_streak' "$REPLAY" \
  || { log "ABORT: agent_replay.py lacks the ladder / failure guard"; decide "NOT PROMOTED (agent_replay.py too old)"; wrapup ABORTED; exit 3; }
MD5_START=$(md5sum < "$LIVE" | cut -d' ' -f1)
envline(){ sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$/\1/p' "$1" | head -1; }
# The candidate is the live launcher minus the window key plus comment lines: exactly one line leaves (live's EXTRA_ENV
# default), exactly one non-comment line arrives, and it is that line without EXL3_MTP_KV_WINDOW. Anything else means the
# live launcher moved since the candidate was cut, and promoting would silently revert that change.
structural(){ local d rem add
  d=$(diff "$LIVE" "$CAND")
  rem=$(printf '%s\n' "$d" | grep -c '^<'); add=$(printf '%s\n' "$d" | grep '^>' | grep -vc '^> #')
  [ "$rem" = 1 ] && [ "$add" = 1 ] || { log "structural: $rem line(s) removed, $add non-comment line(s) added (want 1 / 1)"; return 1; }
  local W N
  W=$(envline "$LIVE"); N=$(envline "$CAND")
  printf '%s\n' "$W" | tr ' ' '\n' | grep -q '^EXL3_MTP_KV_WINDOW=' || { log "structural: live EXTRA_ENV has no EXL3_MTP_KV_WINDOW"; return 1; }
  [ "$(printf '%s\n' "$W" | tr ' ' '\n' | grep -v '^EXL3_MTP_KV_WINDOW=' | xargs)" = "$(printf '%s\n' "$N" | xargs)" ] \
    || { log "structural: candidate EXTRA_ENV != live minus the window key"; return 1; }; }
structural || { diff "$LIVE" "$CAND" | grep -v '^[<>] #' | cut -c1-200 >> "$R/audit.log"
  decide "NOT PROMOTED (guard: candidate is not live minus the window)"; wrapup ABORTED; exit 3; }
NKEYS_LIVE=$(envline "$LIVE" | wc -w | tr -d " ")
log "start: live md5 $MD5_START ($NKEYS_LIVE keys), candidate md5 $(md5sum < "$CAND" | cut -c1-8); nonce $NONCE; replay bar $REPLAY_BAR"

gpu_lock
log "lock held; served at entry: $(served_id || echo none)"
[ "$(md5sum < "$LIVE" | cut -d' ' -f1)" = "$MD5_START" ] \
  || { log "ABORT: live launcher changed while queued"; decide "NOT PROMOTED (guard: live launcher changed while queued)"; wrapup ABORTED; exit 3; }
structural || { decide "NOT PROMOTED (guard: candidate is not live minus the window)"; wrapup ABORTED; exit 3; }
LIVE_IMG=$(grep -oE '^DAILY_IMG=\S+' "$LIVE" | cut -d= -f2)

# ---- helpers ----
declare -A UP0 UP1 AF0 AF1 RAMP STRESS
errcounts(){ local f=$1   # "OOM n; TORCH_CHECK n; tracebacks n" over a container log
  echo "OOM $(grep -acE 'OutOfMemoryError|out of memory|graph\.cu' "$f"); TORCH_CHECK $(grep -acE 'TORCH_CHECK|c10::Error' "$f"); tracebacks $(grep -ac Traceback "$f")"; }
errsum(){ grep -acE 'OutOfMemoryError|out of memory|graph\.cu|TORCH_CHECK|c10::Error|Traceback' "$1"; }
alive(){ [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
okrows(){ python3 -c 'import json,sys
rs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()] if __import__("os").path.exists(sys.argv[1]) else []
print(sum(1 for r in rs if r.get("ok")), len(rs))' "$1" 2>/dev/null || echo "0 0"; }
# boot TAG LAUNCHER TIER(off|on). off passes NVME_TIER= (no tier); on leaves NVME_TIER unset, i.e. the launcher's default.
boot(){ local tag=$1 L=$2 tier=$3 up
  served_stop; wait_unserved 45
  if [ "$tier" = off ]; then
    env -i HOME="$HOME" PATH="$CLEAN_PATH" NVME_TIER= bash "$L" > "$R/boot-$tag.log" 2>&1
  else
    env -i HOME="$HOME" PATH="$CLEAN_PATH" bash "$L" > "$R/boot-$tag.log" 2>&1
  fi && wait_served_id "$MODEL" 200 8 \
    || { log "[$tag] NO BOOT: $(grep -aE 'Insufficient VRAM|out of memory|Error|NO BOOT|ABORT' "$R/boot-$tag.log" | tail -1 | cut -c1-200)"
         sudo docker logs flashnext > "$R/container-$tag-noboot.log" 2>&1; return 1; }
  generation_state "$MODEL" | grep -q GEN_SANE || { log "[$tag] NO BOOT: generation not sane"; return 1; }
  sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$R/env-$tag.txt"
  up=$(grep -aoE 'VRAM free MiB [0-9]+/[0-9]+' "$R/boot-$tag.log" | tail -1 | grep -oE '[0-9]+/[0-9]+')
  UP0[$tag]=${up%/*}; UP1[$tag]=${up#*/}
  log "[$tag] UP: image $(sudo docker ps --format '{{.Image}}' -f 'name=^flashnext$'); $(grep -aoE 'env keys \([0-9]+\)' "$R/boot-$tag.log" | tail -1); window $(grep -c '^EXL3_MTP_KV_WINDOW=' "$R/env-$tag.txt"); tier $(grep -c '^EXL3_NVME_TIER=' "$R/env-$tag.txt"); $(grep -aoE 'cache [0-9]+' "$R/boot-$tag.log" | tail -1); $(grep -aoE "policy '[^']*'" "$R/boot-$tag.log" | tail -1); $(grep -aoE 'memory clock offset: .*' "$R/boot-$tag.log" | tail -1); VRAM free MiB ${UP0[$tag]}/${UP1[$tag]}"
  [ -n "${UP0[$tag]}" ] && [ -n "${UP1[$tag]}" ] || { log "[$tag] no UP-line VRAM reading"; return 1; }
  [ "$(sudo docker ps --format '{{.Image}}' -f 'name=^flashnext$')" = "$LIVE_IMG" ] || { log "[$tag] image is not the live DAILY_IMG $LIVE_IMG"; return 1; }
  if [ "$tier" = off ]; then grep -q '^EXL3_NVME_TIER=' "$R/env-$tag.txt" && { log "[$tag] NVMe tier on in a tier-off boot"; return 1; }
  else grep -q '^EXL3_NVME_TIER=' "$R/env-$tag.txt" || { log "[$tag] NVMe tier OFF in the tier-on boot"; return 1; }; fi
  return 0; }
greedy(){ python3 "$GREEDY" --url http://127.0.0.1:8022 --tag "$1" --out "$R/greedy.jsonl" > "$R/greedy-$1.log" 2>&1
  log "[$1] greedy: $(grep -c "\"tag\": \"$1\"" "$R/greedy.jsonl") records, $(grep -ac ' ERROR ' "$R/greedy-$1.log") errors"; }
ramp(){ local tag=$1 salt=$2
  python3 "$BENCH" --url "$API" --model "$MODEL" --tag "ramp-$tag" --kind prose --ctx 4000 --tokens 256 \
    --conc 1 2 3 4 5 6 7 8 --runs 1 --warmup-runs 0 --unique --distinct --salt "$salt" \
    --out "$R/ramp-$tag.jsonl" > "$R/ramp-$tag.log" 2>&1; }
# The pre-gate sequence, identical for REF and CAND. Leaves AF0/AF1 = free VRAM right after it. $2 = the fn_greedy tag.
sequence(){ local tag=$1 c s okn tot af A
  greedy "$2"
  ramp "$tag" "$SALT_RAMP"
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  read -r okn tot <<< "$(okrows "$R/ramp-$tag.jsonl")"
  alive && A=alive || A=DEAD
  log "[$tag] ramp c1..c8: $okn/$tot ok, $A, $(errcounts "$R/container-$tag.log"), free $(vram_free)"
  RAMP[$tag]="$okn $tot $A $(errsum "$R/container-$tag.log")"
  [ "$A" = alive ] || return 0
  for c in 4 8; do
    [ "$c" = 4 ] && s=$SALT_S4 || s=$SALT_S8
    python3 "$BENCH" --url "$API" --model "$MODEL" --tag "stress-c$c-26k" --kind prose --tokens 256 --conc $c \
      --runs 1 --ctx 26000 --unique --salt "$s" --out "$R/stress-$tag-c$c.jsonl" > "$R/stress-$tag-c$c.log" 2>&1
    sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
    read -r okn tot <<< "$(okrows "$R/stress-$tag-c$c.jsonl")"
    alive && A=alive || A=DEAD
    log "[$tag] stress c$c @26k: $okn/$tot ok (want $c), $A, $(errcounts "$R/container-$tag.log"), free $(vram_free)"
    STRESS[$tag-$c]="$okn $tot $A $(grep -acE 'OutOfMemoryError|out of memory|graph\.cu' "$R/container-$tag.log")"
    [ "$A" = alive ] || return 0
  done
  RUNS=2 bash "$GATE" "$R/gate-$tag" "$API" "$MODEL" "$SALT_GATE" > "$R/gate-$tag.out" 2>&1
  af=$(vram_free); AF0[$tag]=$(echo $af | cut -d' ' -f1); AF1[$tag]=$(echo $af | cut -d' ' -f2)
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  grep -aE "^ +[AB] " "$R/gate-$tag.out" | sed "s/^/  [$tag gate] /" | tee -a "$R/audit.log"
  log "[$tag] sequence done: FREE-AFTER ${AF0[$tag]} ${AF1[$tag]} MiB; gate rows A $(okrows "$R/gate-$tag/bench-A.jsonl") B $(okrows "$R/gate-$tag/bench-B.jsonl") (ok total); $(errcounts "$R/container-$tag.log"); alive $(alive && echo yes || echo NO)"; }

# ================= REF: the live daily, tier off =================
boot REF "$LIVE" off || not_promoted "REF did not boot"
grep -q '^EXL3_MTP_KV_WINDOW=' "$R/env-REF.txt" || not_promoted "REF container has no EXL3_MTP_KV_WINDOW"
assert_env_keys "$R/boot-REF.log" "$NKEYS_LIVE" EXL3_MTP_KV_WINDOW >> "$R/audit.log" || not_promoted "REF env keys"
sequence REF ref
[ "$(grep -c '"tag": "ref"' "$R/greedy.jsonl" 2>/dev/null)" = 6 ] || not_promoted "REF greedy incomplete"
# the REF sequence must itself be complete, or the comparisons below compare against nothing
[ -n "${AF0[REF]:-}" ] && [ -n "${AF1[REF]:-}" ] && grep -aqE '^ +B +4 ' "$R/gate-REF.out" || not_promoted "REF sequence incomplete"

# ================= CAND: the candidate, tier off =================
boot CAND "$CAND" off || not_promoted "G0 candidate did not boot"
grep -q '^EXL3_MTP_KV_WINDOW=' "$R/env-CAND.txt" && not_promoted "G0 candidate container still has the window"
assert_env_keys "$R/boot-CAND.log" $((NKEYS_LIVE - 1)) -EXL3_MTP_KV_WINDOW >> "$R/audit.log" || not_promoted "G0 candidate env keys"
sequence CAND CAND

# G1 greedy identity (run before CANDT's greedy lands in the same file)
python3 "$GREEDY" --compare --ref ref --out "$R/greedy.jsonl" > "$R/greedy-compare-CAND.txt" 2>&1
grep -aE '^GREEDY |DIVERGE|MISSING' "$R/greedy-compare-CAND.txt" | sed 's/^/  [greedy] /' | tee -a "$R/audit.log"
grep -aq '^GREEDY CAND vs ref: 6 identical, IDENTICAL' "$R/greedy-compare-CAND.txt" \
  && note "G1 greedy PASS: 6/6 identical to REF" || { note "G1 greedy FAIL"; not_promoted "G1 greedy"; }

# G2 ramp
read -r okn tot A e <<< "${RAMP[CAND]:-0 0 DEAD 1}"
[ "$okn" = 36 ] && [ "$tot" = 36 ] && [ "$A" = alive ] && [ "$e" = 0 ] \
  && note "G2 ramp PASS: 36/36 ok, 0 errors" || { note "G2 ramp FAIL: $okn/$tot ok, $A, $e error lines"; not_promoted "G2 ramp"; }

# G3 stress
for c in 4 8; do
  read -r okn tot A oom <<< "${STRESS[CAND-$c]:-0 0 DEAD 1}"
  [ "$okn" = "$c" ] && [ "$tot" = "$c" ] && [ "$A" = alive ] && [ "$oom" = 0 ] \
    && note "G3 stress c$c @26k PASS: $okn/$c ok, 0 OOM" || { note "G3 stress c$c FAIL: $okn/$tot ok, $A, OOM lines $oom"; not_promoted "G3 stress c$c"; }
done

# G4 canonical gate
for leg in A B; do
  read -r okn tot <<< "$(okrows "$R/gate-CAND/bench-$leg.jsonl")"
  [ "$tot" -gt 0 ] && [ "$okn" = "$tot" ] || { note "G4 gate FAIL: leg $leg rows $okn/$tot ok"; not_promoted "G4 gate leg $leg errors"; }
done
python3 - "$R/gate-REF.out" "$R/gate-CAND.out" <<'PY' > "$R/g4.txt"
import re, sys
def table(p):   # columns: leg conc n prompt tokens decode wall ttft acc it_ms finishes
    return {(r[0], r[1]): r for r in (l.split() for l in open(p)) if len(r) > 9 and re.fullmatch(r"[AB]", r[0])}
ref, cand = table(sys.argv[1]), table(sys.argv[2])
why = []
for key, name, gated in ((("A", "1"), "c1/4k", False), (("A", "4"), "c4/4k", True), (("A", "8"), "c8/4k", True), (("B", "4"), "c4/26k", True)):
    if key not in ref or key not in cand: why.append(f"{name} missing"); continue
    a, b = float(ref[key][5]), float(cand[key][5]); ratio = b / a
    print(f"  {name}: REF {a:.1f} CAND {b:.1f} -> {ratio:.3f}x  (acc {ref[key][8]} -> {cand[key][8]}, it_ms {ref[key][9]} -> {cand[key][9]}){'' if gated else '  [not gated]'}")
    if gated and ratio < 0.97: why.append(f"{name} {ratio:.3f}x")
print("GATE OK" if not why else "GATE STOP: " + "; ".join(why))
PY
tee -a "$R/audit.log" < "$R/g4.txt" >> "$SUM"
grep -q '^GATE OK' "$R/g4.txt" && note "G4 gate PASS (>= 0.97x REF in c4/4k, c8/4k, c4/26k)" \
  || { note "G4 gate FAIL: $(grep -a 'GATE STOP' "$R/g4.txt")"; not_promoted "G4 gate"; }

# G5 headroom
[ -n "${AF0[CAND]:-}" ] && [ -n "${AF1[CAND]:-}" ] || { note "G5 headroom FAIL: no after-sequence reading"; not_promoted "G5 headroom"; }
u0r=${UP0[REF]}; u1r=${UP1[REF]}; a0r=${AF0[REF]}; a1r=${AF1[REF]}; u0c=${UP0[CAND]}; u1c=${UP1[CAND]}; a0c=${AF0[CAND]}; a1c=${AF1[CAND]}
note "G5 headroom: REF up $u0r/$u1r after $a0r/$a1r (cuda:1 draw $(( u1r - a1r ))); CAND up $u0c/$u1c after $a0c/$a1c (cuda:1 draw $(( u1c - a1c ))) MiB"
why=""
[ "$u0c" -ge $(( u0r - 32 )) ] || why="$why cuda:0 UP $u0c < $u0r-32;"
[ "$a0c" -ge $(( a0r - 32 )) ] || why="$why cuda:0 after $a0c < $a0r-32;"
[ $(( u1c - a1c )) -le $(( u1r - a1r + 32 )) ] || why="$why cuda:1 draw $(( u1c - a1c )) > REF draw $(( u1r - a1r ))+32;"
[ "$a1c" -ge 500 ] || why="$why cuda:1 after $a1c < 500;"
[ -z "$why" ] && note "G5 headroom PASS" || { note "G5 headroom FAIL:$why"; not_promoted "G5 headroom"; }

# G7 agent replay on the CAND boot (after the sequence; scored over the replay window only)
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
timeout 1300 python3 "$REPLAY" --url "$API" --model "$MODEL" \
  --prod-ladder --respawn --drain --max-fail-streak 3 --gen 700 --tool 2020 --think 11 --stagger 3 --temp 0.6 \
  --seed 722 --max-seconds 600 --out "$R/replay-CAND.jsonl" > "$R/replay-CAND.log" 2>&1
rc=$?
sudo docker logs --since "$T0" flashnext > "$R/container-replay.log" 2>&1
python3 "$AGG" "$R/container-replay.log" > "$R/agg-replay.txt" 2>&1
sed 's/^/  [replay] /' "$R/agg-replay.txt" | grep -E 'window|AGGREGATE|PER STREAM|mean streams|prompt tokens|prefix cached|MTP acceptance' | tee -a "$R/audit.log"
pstream=$(python3 -c 'import re,sys
m=re.search(r"PER STREAM\s*:\s*([\d,.]+)", open(sys.argv[1]).read()); print(m.group(1).replace(",","") if m else "")' "$R/agg-replay.txt" 2>/dev/null)
e=$(errsum "$R/container-replay.log"); loops=$(grep -aci 'token loop was detected' "$R/container-replay.log")
note "G7 replay: rc $rc; per-stream ${pstream:-none} tok/s (bar $REPLAY_BAR = 0.98 x R722 W $R722_W); $(errcounts "$R/container-replay.log"); loop-detected stops $loops"
[ "$rc" = 0 ] && [ "$e" = 0 ] && [ -n "$pstream" ] && python3 -c "import sys; sys.exit(0 if float('$pstream') >= $REPLAY_BAR else 1)" && alive \
  && note "G7 replay PASS" || { note "G7 replay FAIL"; not_promoted "G7 replay"; }

# ================= CANDT: the candidate with the launcher's default NVMe tier =================
boot CANDT "$CAND" on || { note "G6 tier FAIL: no boot (or tier not on)"; not_promoted "G6 tier boot"; }
greedy CANDT
ramp CANDT "$SALT_TRAMP"
python3 "$BENCH" --url "$API" --model "$MODEL" --tag legB-CANDT --kind prose --ctx 26000 --conc 4 --tokens 1024 \
  --warmup-runs 1 --runs 1 --unique --distinct --salt "$SALT_TB" --out "$R/legB-CANDT.jsonl" > "$R/legB-CANDT.log" 2>&1
sudo docker logs flashnext > "$R/container-CANDT.log" 2>&1
python3 "$GREEDY" --compare --ref ref --out "$R/greedy.jsonl" > "$R/greedy-compare-CANDT.txt" 2>&1
log "[CANDT] greedy vs REF (reported, not gated): $(grep -a '^GREEDY CANDT vs ref' "$R/greedy-compare-CANDT.txt")"
read -r rok rtot <<< "$(okrows "$R/ramp-CANDT.jsonl")"; read -r bok btot <<< "$(okrows "$R/legB-CANDT.jsonl")"
gn=$(grep -c '"tag": "CANDT"' "$R/greedy.jsonl"); ge=$(grep -ac ' ERROR ' "$R/greedy-CANDT.log"); e=$(errsum "$R/container-CANDT.log")
alive && A=alive || A=DEAD
note "G6 tier: greedy $gn records / $ge errors; ramp $rok/$rtot ok; leg B $bok/$btot ok; $(errcounts "$R/container-CANDT.log"); $A; free $(vram_free)"
[ "$gn" = 6 ] && [ "$ge" = 0 ] && [ "$rok" = 36 ] && [ "$rtot" = 36 ] && [ "$bok" = 4 ] && [ "$btot" = 4 ] && [ "$e" = 0 ] && [ "$A" = alive ] \
  && note "G6 tier PASS" || { note "G6 tier FAIL"; not_promoted "G6 tier"; }

# ================= PROMOTE (the served CANDT container is byte-for-byte what the promoted launcher boots) =================
[ "$(md5sum < "$LIVE" | cut -d' ' -f1)" = "$MD5_START" ] \
  || { log "ABORT: live launcher changed during the unit"; not_promoted "guard: live launcher changed during the unit"; }
sudo cp -p "$LIVE" "$PRE" && [ "$(md5sum < "$PRE" | cut -d' ' -f1)" = "$MD5_START" ] \
  || not_promoted "guard: could not write the rollback copy $PRE"
cp "$CAND" "$LIVE.new" && chmod 755 "$LIVE.new" && mv "$LIVE.new" "$LIVE"
cmp -s "$CAND" "$LIVE" || { log "PROMOTE FAIL: live != candidate after the swap"; PROMOTED=1; rollback "promote copy"; }
PROMOTED=1
note "PROMOTED (provisional): live md5 $(md5sum < "$LIVE" | cut -c1-8) = candidate, rollback $PRE (md5 ${MD5_START:0:8}); post-promotion gates follow"

log "=== P1: agentic-edit ==="
python3 "$P/agentic-edit.py" --url "$API" --model "$MODEL" --tag r728 --conc 1 4 --modes greedy sampled --out "$R/agentic-edit.jsonl" 2>&1 \
  | tee "$R/agentic-edit.log" | grep -a "agentic-edit" | tee -a "$R/audit.log"
[ "$(grep -ac "6/6 ok" "$R/agentic-edit.log")" = 4 ] && alive && note "P1 agentic-edit PASS (4 x 6/6)" || { note "P1 agentic-edit FAIL"; rollback "P1 agentic-edit"; }
log "=== P2: needles 131k / 240k ==="
python3 "$P/fn_needle_oai.py" --url "$API" --model "$MODEL" --tag needle --ctx-tokens 131072 240000 \
  --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee "$R/needle.log" | tee -a "$R/audit.log"
[ "$(grep -c '5/5 retrieved' "$R/needle.log")" = 2 ] && alive && note "P2 needles PASS (5/5 at 131k and 240k)" || { note "P2 needles FAIL"; rollback "P2 needles"; }
log "=== P3: tool-eval 69 x 4, parallel 8 ==="
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$MODEL" --temperature 0.6 --top-p 0.95 --top-k 20 \
    --trials 4 --parallel 8 --json-file "$R/tooleval.json" > "$R/tooleval.log" 2>&1 )
python3 "$P/tooleval_summary.py" "$R/tooleval.json" r728 2>&1 | tee "$R/tooleval-summary.txt" | tee -a "$R/audit.log"
mean=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trial_statistics"]["final_score_mean"])' "$R/tooleval.json" 2>/dev/null)
[ -n "$mean" ] && python3 -c "import sys; sys.exit(0 if float('$mean') >= 82.0 else 1)" && alive \
  && note "P3 tool-eval PASS ($mean >= 82.0)" || { note "P3 tool-eval FAIL (${mean:-unparsed})"; rollback "P3 tool-eval (${mean:-unparsed})"; }
log "=== P4: GSM8K n=500, c4 (nostop proxy) ==="
python3 "$P/nostop_proxy.py" --listen 127.0.0.1:8031 --upstream http://127.0.0.1:8022 >> "$R/proxy.log" 2>&1 & pxp=$!; sleep 2
timeout 10800 "$LM" --model local-chat-completions \
  --model_args "base_url=http://127.0.0.1:8031/v1/chat/completions,model=$MODEL,tokenizer=$CKPT,num_concurrent=4,max_retries=1,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 5 --limit 500 --apply_chat_template --gen_kwargs temperature=0,max_gen_toks=8192 --log_samples \
  --output_path "$R/ev-gsm8k" > "$R/ev-gsm8k.log" 2>&1
kill $pxp 2>/dev/null
g=$(python3 -c '
import json,pathlib,sys
f=list(pathlib.Path(sys.argv[1]).rglob("results_*.json")); print(json.loads(f[0].read_text())["results"]["gsm8k"]["exact_match,flexible-extract"] if len(f)==1 else "none")' "$R/ev-gsm8k" 2>/dev/null)
[ -n "$g" ] && [ "$g" != none ] && python3 -c "import sys; sys.exit(0 if float('$g') >= 0.970 else 1)" \
  && note "P4 GSM8K PASS ($g >= 0.970)" || { note "P4 GSM8K FAIL (${g:-none})"; rollback "P4 GSM8K (${g:-none})"; }
alive || { note "server not alive after the post-promotion gates"; rollback "server not alive after the gates"; }
sudo docker logs flashnext > "$R/docker-final.log" 2>&1
note "promoted daily over the whole run (reported): $(errcounts "$R/docker-final.log"); free $(vram_free)"
BOOTED=0   # the promoted daily is serving; nothing to restore
decide "PROMOTED"
wrapup PROMOTED
