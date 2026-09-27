#!/usr/bin/env bash
# R787 chain (2026-09-27): re-measure the public README's four figure inputs on the R785 daily (tabbyapi:rebase-dev-r3,
# pool 901,120, split [30, 30], 42 EXL3 keys incl. EXL3_GR_MIX_TILED=1, TUNEDIR /srv/qwen5090/.exl3cache-rebase-dev-r3,
# live launcher md5 e3db755f), same instruments and output formats as the rounds the figures draw today, so the public
# bench/plot.py only needs its paths changed (bench/plot.py in this repository reads the four results directories):
#   R787a  r787a-decode-curve.sh   decode curve c1..c8 code/prose, 2 boots     (was R719b)  decode-scaling.svg, std-bench.svg solid
#   R787b  r787b-prefill-curve.sh  cold prefill at 30k..240k true tokens       (was R580)   prefill.svg prefill line
#   R787c  r787c-depth-decode.sh   c1 decode at 0 / ~100k / ~200k depth         (was R554)   prefill.svg dashed lines
#   R787d  r787d-std-bench.sh      vllm bench serve ShareGPT + Spec-Bench, A/B  (was R731b)  std-bench.svg dashed lines
# Estimated GPU time: a ~18 min, b ~6 min, c ~6 min, d ~90 min; + one daily restore (~1 min) = ~2 h 5 min.
# Measurement only: nothing is promoted, no launcher or clock is written (the launcher applies memory +4500, core 0 and
# stock power at every boot; every unit reads and asserts them).
#
# QUEUE / LOCK / RESTORE (the box's GPU-queue contract, lib/gpu-queue.sh). The chain registers itself in the GPU queue
# (gpu-queue/r787-chain = its PID) and takes /srv/qwen5090/gpu-exclusive.lock ONCE for the whole chain. The units run as
# its children, in the foreground, in order a b c d: they inherit fd 9 (the lock), so their own gpu_lock returns at once
# (gpu_lock's "lock-holder invoked me" path) and nothing else can take the GPUs between two units; each unit's
# finish_restore sees the chain's live registration in gpu_queue_others and skips the daily restore. So the daily goes
# down once (at R787a's first boot) and comes back once, here, after R787d (or on any abort: trap). If another unit has
# registered behind the chain by then, finish_restore leaves the GPUs to it (the §12 contract), and that unit's chain
# restores the daily.
# A unit that fails (VOID, NO BOOT, bad provenance) does not stop the chain: the four measurements are independent. Each
# unit's rc and last VERDICT/DECISION line go to $R/chain.tsv and audit.log.
#
# TRAFFIC. After the lock: gateway_drain + gateway_wait_idle (Olla, the API gateway in front of the box's agent clients) for the chain's lifetime,
# and the direct :8022 clients stopped (r787-common.sh R787_QUIESCE: hermes, hermes-webui, owui-proxy; they bypass Olla).
# The units find them stopped and so stop / restart nothing. The chain restarts exactly what it stopped, AFTER the
# daily restore, without touching Hermes' config (no hermes-set-model.sh: nothing is promoted). Open WebUI shows no
# models while owui-proxy is stopped (~2 h); R787_QUIESCE="hermes hermes-webui" keeps it on (its requests would then be
# counted as foreign: R787d NOT-PUBLISHABLE on any, a-c report-only).
# ON ABORT (TERM / INT / HUP; stop it with `sudo systemctl stop r787-chain`, which signals the running unit too): bash runs the trap after the running unit exits (systemd TERMs the
# whole cgroup, so the unit runs its own finish first, restore skipped because the chain is still registered); the trap
# then restores the daily and restarts the stopped clients. TimeoutStopSec=1800 covers a unit's finish + the launcher's worst-case waits + one daily boot (PRELAUNCH-R787 #3).
# Signal while still queued behind another lock-holder: that holder skipped its restore because this chain was
# registered, so if the lock is then free and :8022 is empty the trap boots the daily (r785's queued-trap pattern).
#
# INSTALL (operator; .new + mv each, never overwrite a running script): r787-common.sh, r787a-decode-curve.sh,
#   r787b-prefill-curve.sh, r787c-depth-decode.sh, r787d-std-bench.sh, r787-chain.sh -> /srv/qwen5090/ (mode 755).
#   Probes: nothing to deploy (fn_bench.py, vllm_bench_tabby.py, parse_container.py, std_bench_summary.py on flan ==
#   repo, md5 checked 2026-09-27). The client image vllm/vllm-openai:v0.30.0 and the std-bench datasets must be on flan
#   (R787d checks; it never pulls).
# LAUNCH:
#   sudo systemd-run --unit=r787-chain --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=1800 \
#     -p Environment=HOME=$HOME /usr/bin/bash /srv/qwen5090/r787-chain.sh
#   Subset / order: UNITS="a d" (default "a b c d"). The results dirs are $R787_DATE-r787<x>-..., R787_DATE fixed at the
#   chain's start (UTC), so a chain that crosses midnight keeps one date; r787-plot.patch assumes 2026-09-27.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
D=/srv/qwen5090
export R787_DATE=${R787_DATE:-$(date +%F)}
R=$D/results/$R787_DATE-r787-chain; [ -e "$R/audit.log" ] && R=$R-$(date +%H%M)   # a same-day re-run gets its own dir
mkdir -p "$R"
UNITS=${UNITS:-"a b c d"}
declare -A SCRIPT=([a]=r787a-decode-curve.sh [b]=r787b-prefill-curve.sh [c]=r787c-depth-decode.sh [d]=r787d-std-bench.sh)
log(){ echo "$(date -Is) [r787-chain] $*" | tee -a "$R/audit.log"; }
. $D/lib/serve-ctl.sh
. $D/r787-common.sh
SCTL_LOG="$R/audit.log"
LIVE=$R787_LIVE
MODEL=$R787_MODEL
cp "$0" "$R/" 2>/dev/null

# ---------------- preflight (nothing touched) ----------------
why=
for f in $D/lib/gpu-queue.sh $D/lib/serve-ctl.sh $D/lib/gateway-drain.sh $D/r787-common.sh "$LIVE" $D/probes/fn_bench.py; do
  [ -e "$f" ] || why="$why missing $f;"; done
for u in $UNITS; do
  s=${SCRIPT[$u]:-}; [ -n "$s" ] || { why="$why unknown unit '$u';"; continue; }
  [ -e "$D/$s" ] || { why="$why missing $D/$s;"; continue; }
  bash -n "$D/$s" || why="$why $s does not parse;"; done
w=$(r787_launcher_ok) || why="$why $w;"
[ -z "$why" ] || { log "ABORT (preflight, nothing touched):$why"; exit 3; }
log "start: units $UNITS; date $R787_DATE; live launcher $R787_MD5 ($R787_IMG, pool $R787_POOL, split [$R787_SPLIT], $R787_KEYS keys); stock power '$R787_WANT_PWR' W; quiesce '$R787_QUIESCE'"

# ---------------- queue + lock ----------------
export GPU_QUEUE_NAME=r787-chain
. $D/lib/gpu-queue.sh
. $D/lib/gateway-drain.sh
trap 'log "signal while queued"; exec 9>/srv/qwen5090/gpu-exclusive.lock; if flock -n 9 && [ -z "$(served_id)" ]; then log "GPUs free and :8022 empty: booting the daily from $LIVE"; env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" > "$R/boot-queued-restore.log" 2>&1; fi; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; exit 4' TERM INT HUP
gpu_lock

FINISHED=0 RAN=0
finish(){ [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  trap 'log "signal during the chain finish: ignored"' TERM INT HUP
  # The chain itself never stopped the served container (the units did), so finish_restore's BOOTED guard would read
  # "never booted": set it once a unit has run. No served_stop first: if another unit is queued, finish_restore leaves
  # the GPUs to it. Always a fresh boot, even when :8022 answers: the last unit booted the live launcher with NVME_TIER=
  # (tier off), and the daily serves tier on (IMG == DAILY_IMG); finish_restore stops and re-boots it. Before any unit
  # ran: restore only when :8022 is empty (a previous lock-holder skipped its restore because this chain was queued).
  # Also re-boot when :8022 serves a tier-off container (e.g. the previous lock-holder's last arm): not the production config.
  if [ "$RAN" = 1 ] || [ "$(served_id)" != "$MODEL" ] || ! sudo docker exec flashnext env 2>/dev/null | grep -q '^EXL3_NVME_TIER='; then BOOTED=1; else BOOTED=0; fi
  if [ "$BOOTED" = 0 ]; then log "restore: not needed (no unit ran and the daily serves)"
  else log "restore: $([ -n "$(gpu_queue_others)" ] && echo "skipped, the GPU queue continues ($(gpu_queue_others))" || echo "booting the daily from $LIVE (production config, tier on)")"; fi
  finish_restore "$LIVE" > "$R/boot-restore.log" 2>&1   # launcher stdout ends with the LAN address: kept out of audit.log
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"   # only now: while it existed, queue-aware restore helpers kept off the GPUs
  log "daily: $(served_id || echo '<no answer>') ($(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}' 2>/dev/null)); $(grep -aoE 'VRAM free MiB [0-9/]+' "$R/boot-restore.log" | tail -1)"
  if [ -n "$R787_STOPPED" ] && [ -z "$(served_id)" ]; then log "WARN: :8022 not serving; restarting the direct clients anyway"; fi
  r787_unquiesce; log "$R787_QMSG"
  sudo chown -R "$(stat -c %U $D/results)" "$R" 2>/dev/null || true
  log "=== R787 chain $1 ==="; }
trap 'log "signal: waiting for the running unit to finish, then restoring"; finish ABORTED; exit 4' TERM INT HUP
gateway_drain
r787_quiesce
log "lock held; served $(served_id || echo none); $R787_QMSG"
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || log "WARN: Olla still has requests in flight after 900 s"
w=$(r787_launcher_ok) || { log "ABORT under the lock: $w"; finish "ABORTED (launcher changed)"; exit 3; }

# ---------------- the units ----------------
printf 'unit\tscript\trc\tstart\tend\tresults\tlast\n' > "$R/chain.tsv"
for u in $UNITS; do
  s=${SCRIPT[$u]}; t0=$(date -Is)
  w=$(r787_launcher_ok) || { log "[$u] SKIPPED: $w"; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$u" "$s" skip "$t0" "$t0" - "$w" >> "$R/chain.tsv"; continue; }
  log "[$u] start $s"; RAN=1
  bash "$D/$s" > "$R/unit-$u.out" 2>&1; rc=$?
  ud=$(ls -d $D/results/$R787_DATE-r787$u-* 2>/dev/null | tail -1)
  last=$(grep -aE '(VERDICT|DECISION): ' "$ud/audit.log" 2>/dev/null | tail -1 | sed -E 's/^.*(VERDICT|DECISION): //' | cut -c1-200)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$u" "$s" "$rc" "$t0" "$(date -Is)" "${ud:--}" "${last:--}" >> "$R/chain.tsv"
  log "[$u] rc $rc: ${last:-no verdict line} ($ud)"
done
finish DONE
