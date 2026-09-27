#!/usr/bin/env bash
# R786 (2026-09-27): is the rebase-dev r3 daily slower on long-context agent traffic? Same-session A/B of the agent replay.
# R785 review: G7 read 123.0 / 123.7 tok/s per stream on the new image vs R728's 128.6 on the old one (−4 to −6 %;
# non-loop 121.6 / 123.2 vs 129.2), a different day and tier on vs off; G7's bar (0.98 x R722 W) cannot see it and R784's
# decode reading is short-prompt only. The replay (median ~29k cached prompts) is the only long-context decode reading.
#   O = the previous daily, /srv/qwen5090/launch-flashnext.sh.pre-r785 (stack-r3-rows32-tokcount-loopthink3, 983,040,
#       41 keys, TUNEDIR .exl3cache); N = the served launcher (rebase-dev-r3, 901,120, 42 keys, own TUNEDIR). Both launchers
#       unmodified, NVME_TIER= (tier off, as R722 / R728), env -i, one fresh boot per arm, order O1 N1 N2 O2 O3 N3.
#   Traffic: r722's replay exactly (probes/agent_replay.py --prod-ladder --respawn --drain, gen 700, tool 2020, think 11,
#   stagger 3, temp 0.6, seed 722), ARM_SECS per arm; scored server-side by probes/tabby_log_agg.py over the arm's
#   container log (PER STREAM) and by probes/replay_agg_nonloop.py (non-loop-stopped requests, acceptance).
# DECISION (pre-registered; per-stream decode per pair N/O, pairs (O1,N1) (N2,O2) (O3,N3), m = mean of the 3 ratios):
#   SLOWER     m <= 0.97 and all 3 pairs < 1.00  -> report to the operator: a long-context decode trade vs x1.13 prefill
#                                                   (rollback is the operator's call, per the R785 review)
#   SLIGHTLY   m <  0.99 and all 3 pairs < 1.00
#   PARITY     otherwise
# The same rule is printed for the non-loop scores (report-only). GPU ~70 min. Hermes stopped for the arms; the daily
# (served launcher) is restored and Hermes repointed at the end, on abort too. Olla drained for the unit's lifetime.
# Launch: sudo systemd-run --unit=r786-replay-abba --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#   -p Environment=HOME=$HOME /usr/bin/bash /srv/qwen5090/r786-replay-abba.sh
# Install: bench/agent_replay.py, bench/tabby_log_agg.py and bench/replay_agg_nonloop.py -> /srv/qwen5090/probes/;
#   scripts/lib/serve-ctl.sh -> /srv/qwen5090/lib/. lib/gpu-queue.sh, lib/gateway-drain.sh (the box's GPU queue and API gateway)
#   and hermes-set-model.sh (repoints the box's agent client) are host scripts that are not in this repository; the unit
#   aborts without the last two. Write-up: bench/results/r786-replay-abba.md.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
UNIT=${UNIT:-$(basename "$0" .sh)}
R=${R:-/srv/qwen5090/results/$(date +%F)-$UNIT-$(date +%H%M)}; mkdir -p "$R"
D=/srv/qwen5090
LIVE=$D/launch-flashnext.sh
OLD=$D/launch-flashnext.sh.pre-r785
EXPECT_LIVE=${EXPECT_LIVE:-e3db755f24a24192711e7edb96347d83}
EXPECT_OLD=${EXPECT_OLD:-3f8c18109f1c6dc16cf0701761993f38}
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
API=http://127.0.0.1:8022/v1
REPLAY=$D/probes/agent_replay.py
AGG=$D/probes/tabby_log_agg.py
AGG2=$D/probes/replay_agg_nonloop.py
HSET=$D/hermes-set-model.sh
ORDER=${ORDER:-"O1 N1 N2 O2 O3 N3"}
ARM_SECS=${ARM_SECS:-600}
HERMES_STOPPED=
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$UNIT
. $D/lib/gpu-queue.sh
. $D/lib/serve-ctl.sh
. $D/lib/gateway-drain.sh
SCTL_LOG="$R/audit.log"
cp "$0" "$R/" 2>/dev/null
hermes_off(){ local c
  for c in hermes hermes-webui; do
    [ "$(sudo docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" = true ] || continue
    sudo docker stop -t 30 "$c" >/dev/null 2>&1 && HERMES_STOPPED="$HERMES_STOPPED $c"; done
  log "Hermes stopped for the arms:${HERMES_STOPPED:- none was running (left as is)}"; }
hermes_back(){ local was=$HERMES_STOPPED; HERMES_STOPPED=
  [ -z "$was" ] && { log "Hermes was not running before the unit: left as is"; return 0; }
  [ "$(served_id)" = "$MODEL" ] || { log "WARN: :8022 does not serve $MODEL: Hermes not repointed (start it by hand)"; return 0; }
  PORT=8022 MODEL_ID=$MODEL bash "$HSET" >> "$R/audit.log" 2>&1 && log "Hermes on :8022" || log "WARN: Hermes not repointed"; }
finish(){ finish_restore "$LIVE"; hermes_back; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U $D/results)" "$R" 2>/dev/null || true; log "=== r786 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$OLD" "$REPLAY" "$AGG" "$AGG2" "$HSET" "$D/lib/gateway-drain.sh"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
grep -q 'LADDER_PROD' "$REPLAY" && grep -q 'max_fail_streak' "$REPLAY" || { log "ABORT: agent_replay.py lacks the ladder / failure guard"; exit 3; }
[ "$(md5sum < "$LIVE" | cut -c1-32)" = "$EXPECT_LIVE" ] || { log "ABORT: live launcher md5 is not $EXPECT_LIVE"; exit 3; }
[ "$(md5sum < "$OLD" | cut -c1-32)" = "$EXPECT_OLD" ] || { log "ABORT: old launcher md5 is not $EXPECT_OLD"; exit 3; }
log "O = $OLD ($(grep -m1 '^DAILY_IMG=' "$OLD")); N = $LIVE ($(grep -m1 '^DAILY_IMG=' "$LIVE")); order $ORDER; $ARM_SECS s per arm; tier off"

gpu_lock
log "lock held; served at entry: $(served_id || echo none)"
gateway_drain   # the arms serve on :8022; Olla (omp) routes nothing to it until this unit exits
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || log "WARN: gateway not idle after 900 s"
hermes_off
try_boot(){ local tag=$1 l=$2
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= bash "$l" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { sudo docker logs flashnext > "$R/container-$tag-noboot.log" 2>&1; return 1; }
  sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$R/env-$tag.txt"
  grep -q '^EXL3_NVME_TIER=' "$R/env-$tag.txt" && { log "[$tag] ABORT: NVMe tier on"; return 2; }
  log "[$tag] booted $(sudo docker inspect -f '{{.Config.Image}}' flashnext 2>/dev/null); keys $(wc -l < "$R/env-$tag.txt"); $(grep -aoE "cache [0-9]+" "$R/boot-$tag.log" | tail -1); $(grep -aoE 'VRAM free MiB [0-9/]+' "$R/boot-$tag.log" | tail -1)"; }
run_arm(){ local tag=$1 rc
  timeout $((ARM_SECS + 700)) python3 "$REPLAY" --url "$API" --model "$MODEL" \
    --prod-ladder --respawn --drain --max-fail-streak 3 --gen 700 --tool 2020 --think 11 --stagger 3 --temp 0.6 \
    --seed 722 --max-seconds "$ARM_SECS" --out "$R/replay-$tag.jsonl" > "$R/replay-$tag.log" 2>&1
  rc=$?
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  log "[$tag] replay rc=$rc; OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$tag.log"); tracebacks $(grep -ac Traceback "$R/container-$tag.log"); engine loop stops $(grep -aci 'token loop was detected' "$R/container-$tag.log")"
  python3 "$AGG" "$R/container-$tag.log" > "$R/agg-$tag.txt" 2>&1
  python3 "$AGG2" "$R/container-$tag.log" > "$R/agg2-$tag.txt" 2>&1
  sed "s/^/  [$tag] /" "$R/agg-$tag.txt" | grep -E 'window|AGGREGATE|PER STREAM|mean streams|prompt tokens|prefix cached|MTP acceptance' | tee -a "$R/audit.log"
  sed "s/^/  [$tag] /" "$R/agg2-$tag.txt" | tee -a "$R/audit.log"; }
for t in $ORDER; do
  case $t in O*) l=$OLD ;; N*) l=$LIVE ;; *) log "bad arm $t"; finish ABORTED; exit 3 ;; esac
  try_boot $t "$l"; rc=$?
  [ $rc = 0 ] || { log "[$t] NO BOOT (rc $rc)"; finish NO-BOOT; exit 3; }
  run_arm $t
done

python3 - "$R" <<'EOF' | tee "$R/summary.txt" | tee -a "$R/audit.log"
import re, sys
R = sys.argv[1]
def ps(tag):
    try: txt = open(f"{R}/agg-{tag}.txt").read()
    except OSError: return None
    m = re.search(r"PER STREAM\s*:\s*([\d,.]+)", txt)
    return float(m.group(1).replace(",", "")) if m else None
def nl(tag):
    try: txt = open(f"{R}/agg2-{tag}.txt").read()
    except OSError: return None
    m = re.search(r"non-loop n \d+ ([\d.]+)", txt)
    return float(m.group(1)) if m else None
PAIRS = [("O1", "N1"), ("O2", "N2"), ("O3", "N3")]
def decide(name, f):
    v = {t: f(t) for p in PAIRS for t in p}
    print(f"{name} per-stream decode t/s:", {k: (round(x, 1) if x else x) for k, x in v.items()})
    if None in v.values():
        print(f"{name} DECISION: VOID (an arm has no score)"); return
    r = [v[n] / v[o] for o, n in PAIRS]; m = sum(r) / len(r)
    print(f"{name} N/O pairs " + " ".join(f"{x:.3f}" for x in r) + f" mean {m:.3f}")
    if m <= 0.97 and max(r) < 1.0: d = "SLOWER"
    elif m < 0.99 and max(r) < 1.0: d = "SLIGHTLY"
    else: d = "PARITY"
    print(f"{name} DECISION: {d} (N/O {m:.3f})")
decide("all", ps)
decide("non-loop (report-only)", nl)
EOF
finish DONE
