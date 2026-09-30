#!/usr/bin/env bash
# R813 (2026-09-30): the README decode curve (docs/img/decode-scaling.svg and the solid lines of std-bench.svg)
#   re-measured on the SERVED daily, with R787a's instrument, prompts, forced length and boots, so the two curves are
#   comparable shape by shape. Standalone unit (not via r787-chain.sh): its own GPU queue registration + lock, traffic
#   quiesce, Olla drain, and a finish() that restores the daily from the LIVE launcher on every exit after the lock.
# WHAT: two boots of the live launcher (env -i, NVME_TIER= : tier off, as every published curve), fn_bench --distinct
#   (every concurrent request its own suffix), greedy, 1,024 forced tokens, 1 warm-up + 3 recorded rounds per shape,
#   c1..c8, code and prose. Tags NEW1-c<n>-<kind> / NEW2-c<n>-<kind>, records.jsonl / curve.tsv / analysis.txt exactly
#   as R787a (the measurement loop and the analysis heredoc are r787a-decode-curve.sh's, byte for byte), so the public
#   bench/plot.py decode_rates(<dir>/records.jsonl, "NEW") reads the new dir unchanged: only its R787A path moves.
# THE SERVED DAILY (R809p, promoted 2026-09-29; every boot must BE it, else the unit ends VOID, never a mixed curve):
#   live launcher /srv/qwen5090/launch-flashnext.sh = repo flan/launch-flashnext-r809-merge.sh, md5
#   6429dfa2035a37dba51bb09651d374e8 (checked before and after the lock, before each boot, at the end; never modified);
#   image tabbyapi:merge-tok-r1, id sha256:ac16920f72cf41864ed7f4151bd18bc4ce593015dc0bc263284a8fde8c111454 (the
#   launcher's GATED IMAGE ID line); 41 EXL3 keys incl. EXL3_GR_MIX_TILED=1 EXL3_PREFILL_MERGE=1 EXL3_STASH_ASYNC=1;
#   in-container readback `prefill-merge-r1 1 1`; pool 901,120 @ 8,8; split [30, 30]; TUNEDIR
#   /srv/qwen5090/.exl3cache-rebase-dev-r3 (the launcher's own default, line 406 of the repo copy: the port's cache since
#   R785, unchanged by R808/R809); tier off; power == stock default limits, core offset 0, memory offset +4500 (the
#   launcher's POWER / MEMOC blocks apply them at every boot; the unit only reads them).
# WHY: the public README's decode figure was measured by R787a on rebase-dev-r3 (results 2026-09-27-r787a-decode-curve),
#   and says the served image's later layers do not change the decode kernels. R811 (paired std-bench, results
#   2026-09-29-r811-std-bench-ab, REVIEW-R811.md section 5) measured the served image's c1 decode step (ITL p50) +2.1 to
#   +3.4 % over the previous daily, same session, equal tau: the figure has to be re-measured on the served image before
#   the std-bench series is refreshed next to it (REVIEW-R811 option (a)).
# EXPECTATION (2026-09-30, before any R813 data; NOT a gate, nothing is promoted or rolled back on it): ms/step at c1
#   ~+2-3 % over R787a and tokens/step within ~1 % (R811: step +2.1..+3.4 % paired vs OLD, tau equal within 0.3 % at c1);
#   c2 ~+1-4 %; c4-c8 within ~+-2 %. R813 vs R787a is CROSS-SESSION (two boots each, one session per side, three days
#   apart). The boot-spread column is only the WITHIN-session noise (0.1-2.3 % in R787a); the cross-session reference is
#   R787a/R719b on the same instrument two days apart: ms/step 1.000-1.022 over all 16 shapes, median ~1.015, c1 +1.7 %
#   read as "cross-day step time within noise" (REVIEW-R787 section 4). So a c1 ms/step near 1.02 is not separable from
#   day-to-day drift by this design. The unit's job is the served-state curve for the README; R811's paired measurement
#   stays the evidence for the step cost.
# COMPARE (report only): probes/r813_curve_compare.py (offline self-test probes/test_r813_curve_compare.py, run before
#   the lock; the unit aborts if it fails) prints per concurrency x kind the per-stream decode rate, the decode aggregate,
#   ms per decode step, tokens per step, both rounds and the R813/R787a ratios, the per-round boot spread, and a
#   container-log check that SSE frames == verify steps on both images. It runs in finish(), AFTER the daily restore
#   (REVIEW-R811-code SHOULD-FIX 2), on whatever records exist. Reference: R813_REF (default the R787a dir on flan).
#   Missing reference = a WARN and no table, not an abort (the measurement is the deliverable). Re-run by hand:
#   python3 $R/probes/r813_curve_compare.py --new $R --ref /srv/qwen5090/results/2026-09-27-r787a-decode-curve
# CHANGES AGAINST r787a-decode-curve.sh (provenance and hygiene only; the measurement loop is the same):
#   - Standalone (r811-std-bench-ab.sh's spine): GPU queue + lock, queued-signal trap (boots the daily if the lock is
#     free and :8022 empty), direct :8022 clients stopped (hermes, hermes-webui, owui-proxy) and exactly those that were
#     running restarted, re-stopped after each boot, Olla drain + wait-idle, finish() on TERM/INT/HUP, on every VOID, at
#     the end, and (EXIT trap) on any other exit after the lock. r787a's "RUN VIA r787-chain.sh ONLY" does not apply.
#   - r787-common.sh is not sourced (its R787_* pins are the R785 daily and r787_boot_ok has no image-id, readback or
#     merge-key check); the served daily's checks are inline (arm() below): launcher md5 before the boot, image tag +
#     FULL image id, env-keys line (41 + the three required keys), the container's EXL3_ env (the three keys =1, no
#     EXL3_NVME_TIER), the in-container readback (r811's form, -e CUDA_VISIBLE_DEVICES= so no CUDA context is opened),
#     pool, split, TUNEDIR, power, core and memory offsets; NEW2 must match NEW1's launcher md5 / image id / env sha.
#   - fn_bench.py md5 pinned (b457fa44..., unchanged since 2026-09-25, before R787a): same instrument or abort before the
#     lock. Probes snapshotted into $R/probes and run from there.
#   - Per boot, report-only: foreign requests (r787_foreign's regex) and the prefill-merge self-disable line
#     (REVIEW-R811-code SHOULD-FIX 1: the readback runs in a fresh process and cannot see a self-disable in the served one).
#   - Results dir $R813_DATE-r813-decode-curve (R813_DATE defaults to today's date, as R787_DATE did; a same-day re-run
#     gets -HHMM and never appends to an earlier run's records.jsonl).
# GPU TIME: ~20 min (R787a: two boots x 16 shapes in ~18 min) + the daily restore (~2 min). gateway_wait_idle can add up
#   to 15 min before the first boot if a user stream is in flight (lock held, no GPU used).
# FILES TO DEPLOY (operator; .new + mv, never over a running file):
#   for f in r813-decode-curve.sh probes/r813_curve_compare.py probes/test_r813_curve_compare.py; do
#     ssh flan "cat > /srv/qwen5090/$f.new" < flan/$f && ssh flan "mv /srv/qwen5090/$f.new /srv/qwen5090/$f"; done
#   Must already be on flan (the unit checks each before the lock): probes/fn_bench.py (md5 pinned below),
#   probes/parse_container.py (R811 deployed the repo copy 2026-09-29), lib/gpu-queue.sh, lib/serve-ctl.sh,
#   lib/gateway-drain.sh, the live launcher at md5 6429dfa2, image tabbyapi:merge-tok-r1 at the pinned id, and (for the
#   table only) $R813_REF/records.jsonl + container-NEW{1,2}.log.
# RUN:
#   ssh flan 'sudo systemd-run --unit=r813-decode-curve --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=1800 -p Environment=HOME=$HOME /usr/bin/bash /srv/qwen5090/r813-decode-curve.sh'
#   Stop: sudo systemctl stop r813-decode-curve (the trap ends it VOID and restores the daily from the live launcher).
#   Knobs (-p Environment=...): R813_DATE, R813_REF, QUIESCE ("hermes hermes-webui owui-proxy"; "hermes hermes-webui"
#   keeps Open WebUI on, its requests then show in the foreign counts).
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
UNIT=r813
D=/srv/qwen5090
R813_DATE=${R813_DATE:-$(date +%F)}
R=$D/results/$R813_DATE-r813-decode-curve
[ -e "$R/audit.log" ] && R=$R-$(date +%H%M)   # a re-run never appends to an earlier run's records.jsonl
[ -e "$R/audit.log" ] && { echo "ABORT: $R already holds a run"; exit 3; }
mkdir -p "$R/probes"
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
note(){ log "$*"; echo "$*" >> "$R/summary.txt"; }
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
API=http://127.0.0.1:8022/v1
LIVE=$D/launch-flashnext.sh
CLEAN_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
R813_REF=${R813_REF:-$D/results/2026-09-27-r787a-decode-curve}

# ---------------- the served daily (pinned) ----------------
WANT_MD5=6429dfa2035a37dba51bb09651d374e8
WANT_IMG=tabbyapi:merge-tok-r1
WANT_IMGID=sha256:ac16920f72cf41864ed7f4151bd18bc4ce593015dc0bc263284a8fde8c111454
WANT_KEYS=41
WANT_REQ="EXL3_GR_MIX_TILED EXL3_PREFILL_MERGE EXL3_STASH_ASYNC"
WANT_RB="prefill-merge-r1 1 1"
WANT_POOL=901120
WANT_SPLIT="30, 30"
WANT_TUNEDIR=$D/.exl3cache-rebase-dev-r3
WANT_GPC=${WANT_GPC:-"0 0"}
WANT_MEM=${WANT_MEM:-"4500 4500"}
# Flash-Next publishes at stock (600 / 575 W): power.limit must equal power.default_limit on every card
WANT_PWR=${WANT_PWR:-$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | awk '{printf "%s%.0f", (NR>1?" ":""), $1}')}
# the instrument R787a ran (repo probes/fn_bench.py, last changed 2026-09-25 ead9e55/7b83b8f, before R787a)
FN_MD5=b457fa447eee9919b28dfd088c96bc9c
P=$D/probes
PROBES="fn_bench.py parse_container.py r813_curve_compare.py test_r813_curve_compare.py"
QUIESCE=${QUIESCE-"hermes hermes-webui owui-proxy"}
# r811's readback: `absent <module>` if the module is missing, else `<rev> <merge 0|1> <async 0|1>`
RB_PY=$'try:\n    import exllamav3.cache.prefill_merge as p\nexcept ModuleNotFoundError as e:\n    print("absent", e.name)\nelse:\n    print(p.REVISION, int(p.merge_enabled()), int(p.async_enabled()))'

md5f(){ md5sum < "$1" 2>/dev/null | cut -c1-32; }
iid(){ sudo docker image inspect "$1" -f '{{.Id}}' 2>/dev/null; }
# per-card offsets, NVML index order (0 = ASUS, 1 = HP; r730). The unit only reads them.
gpcoff(){ timeout 30 sudo python3 -c 'import pynvml as N;N.nvmlInit();print(*[N.nvmlDeviceGetGpcClkVfOffset(N.nvmlDeviceGetHandleByIndex(i)) for i in range(N.nvmlDeviceGetCount())])' 2>/dev/null || echo "?"; }
memoff(){ timeout 30 sudo python3 -c 'import pynvml as N;N.nvmlInit();print(*[N.nvmlDeviceGetMemClkVfOffset(N.nvmlDeviceGetHandleByIndex(i)) for i in range(N.nvmlDeviceGetCount())])' 2>/dev/null || echo "?"; }
pwrlim(){ nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%s%.0f", (NR>1?" ":""), $1}'; }
# the live launcher is the R809p one and still names the pinned image (prints the reason, rc 1)
launcher_ok(){ local m; m=$(md5f "$LIVE")
  [ "$m" = "$WANT_MD5" ] || { echo "live launcher md5 '${m:-missing}' != $WANT_MD5 (the daily changed since R809p)"; return 1; }
  [ "$(grep -oE '^DAILY_IMG=\S+' "$LIVE" | cut -d= -f2)" = "$WANT_IMG" ] || { echo "live DAILY_IMG is not $WANT_IMG"; return 1; }
  return 0; }
# the tag still resolves to the pinned id (a retag / prune would measure another build)
image_ok(){ local i; i=$(iid "$WANT_IMG")
  [ "$i" = "$WANT_IMGID" ] || { echo "image $WANT_IMG is '${i:-missing}', not the pinned ${WANT_IMGID:7:12}"; return 1; }; return 0; }
# clocks / power now: prints "power P W; core G; memory M", rc 1 on a drift
clocks_ok(){ local pwr gpc mem; pwr=$(pwrlim); gpc=$(gpcoff); mem=$(memoff)
  echo "power $pwr W; core $gpc; memory $mem"
  [ "$pwr" = "$WANT_PWR" ] && [ "$gpc" = "$WANT_GPC" ] && [ "$mem" = "$WANT_MEM" ]; }
# Report-only (r787_foreign, verbatim): request headers without min_tokens other than the first one (the launcher's
# own /completions warm-up). Every fn_bench request forces its length with min_tokens, so anything else is foreign.
foreign(){ python3 - "$1" <<'PY'
import re, sys
t = open(sys.argv[1], errors="replace").read()
h = re.findall(r"INFO:\s+#(\d+) (?:chat/)?completions(?: \([\w-]+\))?: [\d,]+ prompt tokens ·\s+(.*?)(?=\n\S|\Z)", t, re.S)
if not h:
    print("?"); raise SystemExit
first = min(int(n) for n, _ in h)
print(sum(1 for n, rest in h if int(n) != first and "min_tokens" not in rest))
PY
}

# ---------------- checks before the lock (CPU only; nothing touched) ----------------
for f in "$LIVE" $D/lib/gpu-queue.sh $D/lib/serve-ctl.sh $D/lib/gateway-drain.sh $(for x in $PROBES; do echo "$P/$x"; done); do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
why=$(launcher_ok) || { log "ABORT (before the lock, nothing touched): $why"; exit 3; }
# the launcher defaults this curve depends on (the md5 pins them; checked by name so a reader sees what)
grep -qxF "CACHE=\${CACHE:-$WANT_POOL}" <(grep -oE '^CACHE=\S+' "$LIVE") || { log "ABORT: live launcher CACHE default is not $WANT_POOL"; exit 3; }
grep -qE "^GPU_SPLIT=\\\$\{GPU_SPLIT:-$WANT_SPLIT\}" "$LIVE" || { log "ABORT: live launcher GPU_SPLIT default is not $WANT_SPLIT"; exit 3; }
grep -qE "^TUNEDIR=\\\$\{TUNEDIR:-$WANT_TUNEDIR\}" "$LIVE" || { log "ABORT: live launcher TUNEDIR default is not $WANT_TUNEDIR"; exit 3; }
wenv=$(sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$/\1/p' "$LIVE" | head -1)
[ "$(echo $wenv | wc -w)" = "$WANT_KEYS" ] || { log "ABORT: live launcher EXTRA_ENV default has $(echo $wenv | wc -w) keys, want $WANT_KEYS"; exit 3; }
for k in $WANT_REQ; do echo " $wenv " | grep -q " $k=1 " || { log "ABORT: live launcher does not set $k=1"; exit 3; }; done
why=$(image_ok) || { log "ABORT (before the lock, nothing touched): $why"; exit 3; }
[ "$(md5f "$P/fn_bench.py")" = "$FN_MD5" ] || { log "ABORT: $P/fn_bench.py md5 $(md5f "$P/fn_bench.py") != $FN_MD5 (not R787a's instrument; deploy the repo copy)"; exit 3; }
# snapshot the inputs: a probe edited while the unit is queued must not change what this run measures
for x in $PROBES; do cp "$P/$x" "$R/probes/"; done
cp "$0" "$R/" 2>/dev/null
[ "$(md5f "$R/probes/fn_bench.py")" = "$FN_MD5" ] || { log "ABORT: the fn_bench.py snapshot is not $FN_MD5"; exit 3; }
for x in fn_bench.py parse_container.py r813_curve_compare.py; do
  python3 -m py_compile "$R/probes/$x" 2>/dev/null || { log "ABORT: $x does not compile"; exit 3; }; done
mkdir -p "$R/.selftest-tmp"
TMPDIR=$R/.selftest-tmp python3 "$R/probes/test_r813_curve_compare.py" > "$R/selftest-r813_curve_compare.txt" 2>&1 \
  || { log "ABORT: test_r813_curve_compare.py fails: $(tail -3 "$R/selftest-r813_curve_compare.txt" | tr '\n' ' ')"; exit 3; }
rm -rf "$R/.selftest-tmp" "$R/probes/__pycache__"
log "probes (snapshot md5): $(cd "$R/probes" && for x in $PROBES; do printf '%s %s; ' "$x" "$(md5f "$x" | cut -c1-8)"; done)self-test: $(tail -1 "$R/selftest-r813_curve_compare.txt")"
if [ -s "$R813_REF/records.jsonl" ]; then log "reference: $R813_REF ($(grep -c '"ok": true' "$R813_REF/records.jsonl") ok records; container logs: $(ls "$R813_REF"/container-NEW?.log 2>/dev/null | wc -l))"
else log "WARN: reference $R813_REF/records.jsonl missing: the curve is measured, the comparison table is skipped (re-run the compare by hand)"; fi
note "R813 decode curve on the served daily: $LIVE (${WANT_MD5:0:8}, $WANT_IMG ${WANT_IMGID:7:12}, $WANT_KEYS keys, readback '$WANT_RB', pool $WANT_POOL, split [$WANT_SPLIT], tunedir $WANT_TUNEDIR); tier off; want core $WANT_GPC / memory $WANT_MEM / power $WANT_PWR W; quiesce '$QUIESCE'; reference $R813_REF; results $R"

# ---------------- queue + lock ----------------
export GPU_QUEUE_NAME=r813-decode-curve
. $D/lib/gpu-queue.sh
. $D/lib/serve-ctl.sh
. $D/lib/gateway-drain.sh
SCTL_LOG="$R/audit.log"
# Signal while still queued (r811's, verbatim): a holder that skipped its restore because this unit was registered left
# :8022 empty; boot the daily if the lock is free, :8022 is empty and nobody else is queued.
trap 'log "signal while queued"; exec 9>/srv/qwen5090/gpu-exclusive.lock; if flock -n 9 && [ -z "$(served_id)" ] && [ -z "$(gpu_queue_others)" ]; then log "GPUs free and :8022 empty: booting the Flash-Next daily from $LIVE"; env -i HOME="$HOME" PATH=$CLEAN_PATH bash "$LIVE" > "$R/boot-queued-restore.log" 2>&1; log "daily: $(served_id || echo none)"; fi; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; exit 4' TERM INT HUP
gpu_lock
why=$(launcher_ok) || true
[ -n "$why" ] || why=$(image_ok) || true
if [ -n "$why" ]; then
  log "ABORT after the lock (nothing stopped): $why"
  if [ -z "$(served_id)" ]; then BOOTED=1; finish_restore "$LIVE" > "$R/boot-restore.log" 2>&1; log "daily: $(served_id || echo none)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; exit 3
fi
log "after the lock: live launcher and image id unchanged"

VERDICT="VOID the unit ended before the analysis"
FINISHED=0 WAS_RUNNING= FIRST_CFG=
running(){ [ "$(sudo docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }
quiesce(){ local c s=; for c in $QUIESCE; do running "$c" || continue; sudo docker stop -t 30 "$c" >/dev/null 2>&1 && s="$s $c"; done; echo "${s# }"; }
finish(){ [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  trap 'log "signal during finish: ignored"' TERM INT HUP
  [ -n "$(served_id)" ] && sudo docker logs flashnext > "$R/container-final.log" 2>&1
  # every boot ran tier off: never leave it on :8022. Stop, then restore from the LIVE launcher (tier on: IMG ==
  # DAILY_IMG). finish_restore leaves the GPUs to a queued unit (OPERATIONS §12).
  if [ "${BOOTED:-0}" = 1 ] || [ -z "$(served_id)" ]; then
    served_stop; wait_unserved 45
    finish_restore "$LIVE" > "$R/boot-restore.log" 2>&1   # launcher stdout ends with the LAN address: kept out of audit.log
  else log "restore: not needed (the daily was never stopped)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  local img imgid tier now; img=$(sudo docker inspect -f '{{.Config.Image}}' flashnext 2>/dev/null); imgid=$(sudo docker inspect -f '{{.Image}}' flashnext 2>/dev/null)
  tier=$(sudo docker exec flashnext env 2>/dev/null | grep -c '^EXL3_NVME_TIER=')
  log "after: served $(served_id || echo none); image ${img:-none} ${imgid:7:12}; tier ${tier:-0}; live launcher md5 $(md5f "$LIVE" | cut -c1-8); core $(gpcoff); memory $(memoff); power $(pwrlim) W"
  if [ -n "$(served_id)" ] && { [ "$img" != "$WANT_IMG" ] || [ "$imgid" != "$WANT_IMGID" ] || [ "${tier:-0}" = 0 ]; }; then
    log "WARN: :8022 is not the production daily (want $WANT_IMG ${WANT_IMGID:7:12}, tier on): restore it by hand from $LIVE"; fi
  [ -n "$(served_id)" ] || [ -n "$(gpu_queue_others)" ] || log "WARN: :8022 not serving after the restore and no unit queued: boot $LIVE by hand"
  now=$(quiesce); [ -n "$now" ] && log "stopped after the restore: $now (started below if it ran at the start)"
  if [ -n "$WAS_RUNNING" ]; then
    [ -n "$(served_id)" ] || log "WARN: :8022 not serving (another unit queued?); restarting the direct clients anyway"
    if sudo docker start $WAS_RUNNING >/dev/null 2>&1; then log "restarted: $WAS_RUNNING"
    else log "RESTART FAILED: $WAS_RUNNING (start by hand: sudo docker start $WAS_RUNNING)"; fi
  else log "no direct client to restart"; fi
  # the comparison (report only; CPU, after the restore) on whatever records exist
  if [ -s "$R/records.jsonl" ] && [ -s "$R813_REF/records.jsonl" ]; then
    python3 "$R/probes/r813_curve_compare.py" --new "$R" --ref "$R813_REF" --json "$R/compare.json" > "$R/compare.txt" 2>&1 \
      || log "WARN: r813_curve_compare.py rc $? (compare.txt)"
    { echo; echo "---- R813 vs R787a (probes/r813_curve_compare.py, report only) ----"; cat "$R/compare.txt"; } >> "$R/summary.txt"
    sed 's/^/  /' "$R/compare.txt" | tee -a "$R/audit.log"
  else log "comparison skipped: $([ -s "$R/records.jsonl" ] && echo "no reference at $R813_REF" || echo "no R813 records")"; fi
  sudo chown -R "$(stat -c %U $D/results)" "$R" 2>/dev/null || true
  echo "VERDICT: $VERDICT" >> "$R/summary.txt"
  log "=== R813 $1 ==="; log "VERDICT: $VERDICT"; }
void(){ VERDICT="VOID $*"; finish VOID; exit 3; }
trap 'log "signal"; VERDICT="VOID signal (the run was interrupted)"; finish ABORTED; exit 4' TERM INT HUP
# any other exit from here on (set -u, an unexpected error path): same finish, so :8022 is never left down or tier off
trap 'rc=$?; [ "$FINISHED" = 1 ] || { VERDICT="VOID unexpected exit rc $rc"; finish ABORTED; }' EXIT

for c in $QUIESCE; do running "$c" && WAS_RUNNING="$WAS_RUNNING $c"; done; WAS_RUNNING=${WAS_RUNNING# }
st=$(quiesce)
gateway_drain   # the boots serve on the live :8022 port; Olla routes nothing to it until this unit exits
log "lock held; served at entry: $(served_id || echo none); direct clients stopped: ${st:-none}; offsets now core $(gpcoff) / memory $(memoff), power $(pwrlim) W"
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || log "WARN: Olla still has requests in flight after 900 s"
why=$(launcher_ok) || void "under the lock: $why"
cp "$LIVE" "$R/launcher-at-lock.sh"

# ---------------- one boot = the 16 shapes (R787a's loop) ----------------
arm(){ local tag=$1 bl=$R/boot-$1.log m full img envsha keys rb ck cko why= st
  served_stop; wait_unserved 45
  m=$(md5f "$LIVE"); [ "$m" = "$WANT_MD5" ] || void "[$tag] live launcher md5 '$m' != $WANT_MD5 (changed mid-run)"
  env -i HOME="$HOME" PATH=$CLEAN_PATH NVME_TIER= bash "$LIVE" > "$bl" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
      void "[$tag] NO BOOT: $(grep -aE 'Insufficient VRAM|out of memory|Error|ABORT|NO BOOT' "$bl" | tail -1 | cut -c1-160)"; }
  st=$(quiesce); [ -n "$st" ] && log "[$tag] stopped again after the boot: $st"
  # the served daily's provenance; every reason is collected, then the unit is VOID
  sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$R/env-$tag.txt"
  img=$(sudo docker inspect -f '{{.Config.Image}}' flashnext 2>/dev/null); full=$(sudo docker inspect -f '{{.Image}}' flashnext 2>/dev/null)
  envsha=$(sha256sum "$R/env-$tag.txt" | cut -c1-12)
  keys=$(assert_env_keys "$bl" "$WANT_KEYS" $WANT_REQ 2>&1) || why="$why env keys: $(echo $keys);"
  [ "$img" = "$WANT_IMG" ] || why="$why image '$img' != $WANT_IMG;"
  [ "$full" = "$WANT_IMGID" ] || why="$why image id '${full:7:12}' != pinned ${WANT_IMGID:7:12};"
  for k in $WANT_REQ; do grep -qx "$k=1" "$R/env-$tag.txt" || why="$why container env lacks $k=1;"; done
  grep -q '^EXL3_NVME_TIER=' "$R/env-$tag.txt" && why="$why NVMe tier on;"
  sudo docker exec -e CUDA_VISIBLE_DEVICES= flashnext python3 -c "$RB_PY" > "$R/readback-$tag.txt" 2>&1
  rb=$(grep -aE '^(absent \S+|\S+ [01] [01])$' "$R/readback-$tag.txt" | tail -1)
  rb=${rb:-$(tail -1 "$R/readback-$tag.txt" | tr '\t' ' ' | cut -c1-160)}
  [ "$rb" = "$WANT_RB" ] || why="$why prefill-merge readback '$rb' != '$WANT_RB';"
  grep -aqF "cache $WANT_POOL @" "$bl" || why="$why pool $(grep -aoE 'cache [0-9]+ @' "$bl" | tail -1) != $WANT_POOL;"
  grep -aqF "split [$WANT_SPLIT]" "$bl" || why="$why $(grep -aoE 'split \[[^]]*\]' "$bl" | tail -1) != split [$WANT_SPLIT];"
  grep -aqF "tunedir $WANT_TUNEDIR," "$bl" || why="$why tunedir is not $WANT_TUNEDIR;"
  ck=$(clocks_ok) || why="$why at boot: $ck (want power $WANT_PWR W, core $WANT_GPC, memory $WANT_MEM);"
  # R787a's two boots describe one configuration: NEW2 must match NEW1
  if [ -z "$FIRST_CFG" ]; then FIRST_CFG="$m/$full/$envsha"
  elif [ "$m/$full/$envsha" != "$FIRST_CFG" ]; then why="$why config changed between boots: $m/${full:7:12}/$envsha vs $FIRST_CFG;"; fi
  log "[$tag] booted: image $img (${full:7:12}); launcher md5 ${m:0:8}; env keys $(grep -aoE 'env keys \([0-9]+\)' "$bl" | tail -1 | tr -dc 0-9) (container EXL3_ env $(wc -l < "$R/env-$tag.txt"), sha $envsha); readback '$rb'; $(grep -aoE 'cache [0-9]+ @ [0-9]+,[0-9]+' "$bl" | tail -1); $(grep -aoE 'split \[[^]]*\]' "$bl" | tail -1); $ck; launcher: $(grep -aoE 'memory clock offset: .*' "$bl" | tail -1) / $(grep -aoE 'power policy .*' "$bl" | tail -1); $(grep -aoE 'VRAM free MiB [0-9/]+' "$bl" | tail -1)"
  [ -z "$why" ] || void "[$tag] not the served daily at the published regime:$why"
  for conc in 1 2 3 4 5 6 7 8; do for kind in code prose; do
    python3 "$R/probes/fn_bench.py" --url "$API" --model "$MODEL" --tag "$tag-c$conc-$kind" --kind $kind --distinct \
      --tokens 1024 --warmup-runs 1 --conc $conc --runs 3 --out "$R/records.jsonl" > "$R/bench-$tag-c$conc-$kind.log" 2>&1
  done; done
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  ck=$(clocks_ok) && cko=0 || cko=1
  log "[$tag] done; OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$tag.log"); restarts $(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null); foreign (report-only) $(foreign "$R/container-$tag.log"); prefill-merge lines $(grep -ac 'prefill-merge-r1' "$R/container-$tag.log"), disabled: $(grep -a 'EXL3_PREFILL_MERGE disabled for this process' "$R/container-$tag.log" | tail -1 | cut -c1-160); after: $ck; free after $(vram_free)"
  [ "$cko" = 0 ] || void "[$tag] clocks / power drifted during the arm: $ck"; }
arm NEW1; arm NEW2
# ---------------- R787a's analysis, verbatim (curve.tsv + analysis.txt) ----------------
python3 - "$R/records.jsonl" "$R/curve.tsv" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
tok=collections.defaultdict(int); wall={}; dec=collections.defaultdict(list); rounds=collections.defaultdict(list); ttft=collections.defaultdict(list)
for l in open(sys.argv[1]):
    r=json.loads(l)
    if not r.get("ok"): continue
    k=(r["tag"],r["run"]); tok[k]+=r["completion_tokens"] or 0; wall[k]=r["round_wall_s"]
    if r.get("decode_tps"): dec[r["tag"]].append(r["decode_tps"]); rounds[k].append(r["decode_tps"])
    if r.get("ttft_s") is not None: ttft[r["tag"]].append(r["ttft_s"])
agg=collections.defaultdict(list); dagg=collections.defaultdict(list)
for k,v in tok.items(): agg[k[0]].append(v/wall[k])
for k,v in rounds.items(): dagg[k[0]].append(sum(v))
def m(d, conc, kind, f):
    v=[x for b in (1,2) for x in d.get(f"NEW{b}-c{conc}-{kind}",[])]; return f(v) if v else float("nan")
def per_boot(d, conc, kind, f):
    return [f(d[t]) if d.get(t) else float("nan") for t in (f"NEW1-c{conc}-{kind}", f"NEW2-c{conc}-{kind}")]
out=open(sys.argv[2],"w"); out.write("conc\tkind\twallagg\tdecode_stream\tdecode_agg\tttft\n")
for kind in ("code","prose"):
    for c in range(1,9):
        v=[m(d,c,kind,f) for d,f in ((agg,st.mean),(dec,st.median),(dagg,st.mean),(ttft,st.median))]
        b=per_boot(dagg,c,kind,st.mean)
        print(f"{kind} c{c}: decode/stream {v[1]:.1f} | decode aggregate {v[2]:.0f} (boots {b[0]:.0f} / {b[1]:.0f}) | round-wall aggregate {v[0]:.0f} | TTFT {v[3]:.2f}s")
        out.write(f"{c}\t{kind}\t"+"\t".join(f"{x:.2f}" for x in v)+"\n")
PY
# every boot was md5-checked before it ran, so a change now does not touch the curve: logged, not a VOID
why=$(launcher_ok) && log "at the end: live launcher unchanged" || log "WARN at the end (after the last boot): $why"
VERDICT="DONE $(python3 -c 'import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; print(sum(1 for x in r if x.get("ok")), "/", len(r), "ok records (want 432)")' "$R/records.jsonl" 2>/dev/null)"
finish DONE
