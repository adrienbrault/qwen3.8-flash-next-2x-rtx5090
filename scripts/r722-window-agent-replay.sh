#!/usr/bin/env bash
# R722 (2026-09-25): what is R721's draft-window revive loss worth on agent-shaped traffic?
# R721 (DECISION WINDOW): a prompt revived from the prompt cache drafts 0.72 accepted/proposed under the served
# EXL3_MTP_KV_WINDOW vs 0.88 with the window unset (fresh 0.91 either way); c8 4k per-stream 111 vs 125 tok/s.
# Agent sessions re-send a growing conversation every step (R586 production: 97 % prefix-cached, median prompt
# 26k tokens, 72 % MTP acceptance), so nearly every agent request is a revive. This round measures the prize on
# that traffic shape before the fix lands, and becomes the fix's reference.
#   W = the daily (window on, pool 983,040); N = the same env minus EXL3_MTP_KV_WINDOW at the first pool that fits
#   from POOLS_N (R721: 950,272). Order W1 N1 W2 N2, one boot per arm, NVME_TIER=.
#   Traffic: probes/agent_replay.py --prod-ladder --respawn --drain (R607's production-shaped settings: gen 700,
#   tool 2020, think 11 s, stagger 3, temp 0.6), ARM_SECS per arm; scored server-side by probes/tabby_log_agg.py
#   over the arm's container log (the instrument of the production numbers).
# DECISION (pre-registered, per-stream decode = tokens / summed per-request decode seconds, mean of the 2 pairs N/W):
#   BIG       N/W >= 1.08 and both pairs >= 1.04  -> the fix is worth the most of anything on the list for agents
#   SOME      N/W >= 1.03
#   NONE      otherwise (revives on real traffic mostly fall outside the window's span, e.g. prompts > 16k tokens)
# Also reports MTP acceptance, aggregate, prefix-cached median, requests, errors per arm. GPU ~50 min. Daily restored.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
UNIT=${UNIT:-$(basename "$0" .sh)}
R=${R:-/srv/qwen5090/results/$(date +%F)-$UNIT}; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
API=http://127.0.0.1:8022/v1
REPLAY=/srv/qwen5090/probes/agent_replay.py
AGG=/srv/qwen5090/probes/tabby_log_agg.py
POOLS_N=${POOLS_N:-"950272 917504"}
ORDER=${ORDER:-"W1 N1 W2 N2"}
ARM_SECS=${ARM_SECS:-600}
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$UNIT
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== r722 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$REPLAY" "$AGG"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
grep -q 'LADDER_PROD' "$REPLAY" && grep -q 'max_fail_streak' "$REPLAY" || { log "ABORT: agent_replay.py lacks the ladder / failure guard"; exit 3; }
WENV=$(sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$/\1/p' "$LIVE" | head -1)
echo "$WENV" | grep -q 'EXL3_MTP_KV_WINDOW=' || { log "ABORT: served env has no EXL3_MTP_KV_WINDOW"; exit 3; }
NENV=$(echo "$WENV" | tr ' ' '\n' | grep -v '^EXL3_MTP_KV_WINDOW=' | xargs)
log "W env $(echo $WENV | wc -w) keys; N env $(echo $NENV | wc -w) keys; N pools $POOLS_N; order $ORDER; $ARM_SECS s per arm"

gpu_lock
log "lock held; served at entry: $(served_id || echo none)"
try_boot(){ local tag=$1; shift
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { sudo docker logs flashnext > "$R/container-$tag-noboot.log" 2>&1; return 1; }
  sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$R/env-$tag.txt"
  grep -q '^EXL3_NVME_TIER=' "$R/env-$tag.txt" && { log "[$tag] ABORT: NVMe tier on"; return 2; }
  log "[$tag] booted; window $(grep -c '^EXL3_MTP_KV_WINDOW=' "$R/env-$tag.txt"); $(grep -aoE "cache [0-9]+" "$R/boot-$tag.log" | tail -1); $(grep -aoE 'VRAM free MiB [0-9/]+' "$R/boot-$tag.log" | tail -1)"; }
run_arm(){ local tag=$1 rc
  timeout $((ARM_SECS + 700)) python3 "$REPLAY" --url "$API" --model "$MODEL" \
    --prod-ladder --respawn --drain --max-fail-streak 3 --gen 700 --tool 2020 --think 11 --stagger 3 --temp 0.6 \
    --seed 722 --max-seconds "$ARM_SECS" --out "$R/replay-$tag.jsonl" > "$R/replay-$tag.log" 2>&1
  rc=$?
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  log "[$tag] replay rc=$rc; OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$tag.log"); tracebacks $(grep -ac Traceback "$R/container-$tag.log")"
  python3 "$AGG" "$R/container-$tag.log" > "$R/agg-$tag.txt" 2>&1
  sed "s/^/  [$tag] /" "$R/agg-$tag.txt" | grep -E 'window|AGGREGATE|PER STREAM|mean streams|prompt tokens|prefix cached|MTP acceptance' | tee -a "$R/audit.log"; }
for t in $ORDER; do
  case $t in
    W*) try_boot $t || { log "[$t] NO BOOT"; finish NO-BOOT; exit 3; } ;;
    N*) ok=0; for p in $POOLS_N; do
          try_boot $t "EXTRA_ENV=$NENV" CACHE=$p; rc=$?
          [ $rc = 0 ] && { ok=1; log "[$t] pool $p"; break; }
          [ $rc = 2 ] && { finish ABORTED; exit 3; }
          log "[$t] pool $p did not boot"; done
        [ $ok = 1 ] || { log "[$t] NO BOOT at any pool"; finish NO-BOOT; exit 3; } ;;
  esac
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
v = {t: ps(t) for t in ("W1", "N1", "W2", "N2")}
print("per-stream decode t/s:", {k: (round(x, 1) if x else x) for k, x in v.items()})
if None in v.values():
    print("DECISION: VOID (an arm has no score)"); sys.exit()
pairs = [v["N1"] / v["W1"], v["N2"] / v["W2"]]
m = sum(pairs) / 2
print(f"N/W pairs {pairs[0]:.3f} {pairs[1]:.3f} mean {m:.3f}")
if m >= 1.08 and min(pairs) >= 1.04: print(f"DECISION: BIG (N/W {m:.3f})")
elif m >= 1.03: print(f"DECISION: SOME (N/W {m:.3f})")
else: print(f"DECISION: NONE (N/W {m:.3f})")
EOF
finish DONE
