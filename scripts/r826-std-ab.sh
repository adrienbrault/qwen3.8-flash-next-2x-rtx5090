#!/usr/bin/env bash
# R826 (2026-10-01): paired served Flash-Next daily vs R809 daily on BOTH standard instruments.
# OLD = COPY of launch-flashnext.sh.pre-r818, 6429dfa2035a37dba51bb09651d374e8,
#       tabbyapi:merge-tok-r1 / ac16920f72cf (41 selectors; R811 NEW / R813 daily).
# NEW = LIVE launch-flashnext.sh, 262e9c31f714724409635fbac9df8ac2,
#       tabbyapi:r825c-hostprepare / aa04a1cbe94b (46 selectors; R818p + R823p + R825p).
# Never writes either launcher. Boot snapshots with env -i HOME PATH NVME_TIER=;
# restore ONLY from LIVE, without overrides, using lib/serve-ctl.sh's queue-aware finish_restore.
# Both: pool 901120 @ 8,8, split [30, 30], same TUNEDIR, stock power, core 0, memory +4500.
# PRE-REGISTERED EXPECTATION (report only, no promotion/gate): decode c1-c8 within ~+/-2%,
# except c1-c2 where R818p boot warm-up should give NEW the fast state: ~-2.5..-3.3% ms/step.
# ShareGPT c4-c8 TTFT possibly lower for NEW. Decode kernels were not changed.
# GPU TIME: R811 173 min + two R813-sized curves 34 min + c4 repeat 15-17 min
# + restore ~3 min = about 3 h 45 min, excluding queue/drain wait; reserve 4 h.
# PROTOCOL: R811 client/warm-up/measurement matrix byte-identical; 32 fresh boots, pass A NEW OLD,
# pass B OLD NEW. Then a mandatory ShareGPT c4 repeat, same A/B loops, four extra boots.
# c4 composite = all FOUR boots per arm, mean + range + values, never select the faster pair
# (R811b rule); repeat spread <=3%, integrity/identity/foreign checks reported, no publication action.
# Historical R811/R811b cells are NEVER pooled into R826 (their arms differ).
# Decode: R813 fn_bench --distinct c1..c8 x code/prose, 1024 forced tokens, 1 warm-up +3 rounds;
# fresh boots OLD1 NEW1 NEW2 OLD2, each 216 records (864 total), no std-bench warm-ups on these boots.
# IDENTITY per boot: source AND snapshot md5, DAILY_IMG, full image id, resolved selector count,
# exact EXL3 env against archived NEW daily / R811 NEW-as-OLD (tier off), merge + tokenize readbacks,
# NEW /opt/r825c/landing_r825c.py rc AND PASS. A mismatch VOIDs cell and unit before measurement.
# Outputs: R/{NEW,OLD}/results + cells + runs/boots/foreign.tsv (R811); R/c4-repeat/{NEW,OLD}/...
# R/records.jsonl (OLD1/OLD2/NEW1/NEW2), R/curve.tsv + analysis.txt (NEW),
# R/{NEW,OLD}/decode/{records.jsonl,curve.tsv,analysis.txt}; final-table.txt + compare.json.
# Existing std_bench_summary and public plot.py readers need no changes.
# Report-only self-disable/foreign checks: a fresh-process readback cannot see server self-disable;
# absence of the async stash log is NOT evidence that async was inactive (REVIEW-R811).
# DEPLOY to matching /srv/qwen5090 paths (operator, .new + mv):
# r826-std-ab.sh; probes/{r826_report,r826_identity,r811_ab_compare,r813_curve_compare,
# test_r811_ab_compare,test_r813_curve_compare,vllm_bench_tabby,parse_container,std_bench_summary,fn_bench}.py;
# probes/r826-fixtures/{env-NEW,env-OLD}.txt; lib/{gpu-queue,serve-ctl,gateway-drain}.sh.
# Local verification: python3 probes/test_r826_std_ab.py (fixtures and sources stay local).
# RUN (operator): sudo systemd-run --unit=r826-std-ab --collect -p RuntimeMaxSec=43200
#   -p TimeoutStopSec=1800 -p Environment=HOME=$HOME
#   /usr/bin/bash /srv/qwen5090/r826-std-ab.sh
# PASSES/DATASETS/CONCS/SG_N/SB_N/SEED/SB_OUT inherited from R811; overrides = nonstandard run.
# No image pull, gate, promotion, public upload, or launcher modification.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
UNIT=${UNIT:-r826-std-ab}
D=/srv/qwen5090
R826_DATE=${R826_DATE:-$(date +%F)}
R=${R:-$D/results/$R826_DATE-$UNIT}
[ -e "$R/audit.log" ] && R=$R-$(date +%H%M)   # a re-run never mixes into an earlier run's dir
[ -e "$R/audit.log" ] && { echo "ABORT: $R already holds a run"; exit 3; }
mkdir -p "$R/probes" "$R/launchers"
ROOT=$R
declare -A BOOT_L
# Fixtures are archived selector maps, independent of the boot being checked.
FIX=$D/probes/r826-fixtures
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
MDIR=$D/models/$MODEL
LIVE=$D/launch-flashnext.sh
CLEAN_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# ---------------- the two arms (pinned; the launcher files are the only thing booted) ----------------
ARMS="NEW OLD"
declare -A AL AMD5 AIMG AIMGID AKEYS AREQ
AL[NEW]=$LIVE;                    AMD5[NEW]=262e9c31f714724409635fbac9df8ac2
AIMG[NEW]=tabbyapi:r825c-hostprepare; AIMGID[NEW]=sha256:aa04a1cbe94b60bbd91e3d69d15f5e1b6eb679f14885e599404de218d8a897c9
AKEYS[NEW]=46;                    AREQ[NEW]="EXL3_GR_MIX_TILED EXL3_PREFILL_MERGE EXL3_STASH_ASYNC"
AL[OLD]=$D/launch-flashnext.sh.pre-r818; AMD5[OLD]=6429dfa2035a37dba51bb09651d374e8
AIMG[OLD]=tabbyapi:merge-tok-r1; AIMGID[OLD]=sha256:ac16920f72cf41864ed7f4151bd18bc4ce593015dc0bc263284a8fde8c111454
AKEYS[OLD]=41;                    AREQ[OLD]="EXL3_GR_MIX_TILED EXL3_PREFILL_MERGE EXL3_STASH_ASYNC"
WANT_POOL=901120
WANT_SPLIT="30, 30"
WANT_TUNEDIR=$D/.exl3cache-rebase-dev-r3
# Both arms inherit the same merge and tokenizer revisions. NEW source landing verifies replacements.
RB_PY=$'import exllamav3.cache.prefill_merge as p\nprint(p.REVISION, int(p.merge_enabled()), int(p.async_enabled()))'
TOK_PY='import sys; sys.path.insert(0, "/app"); import common.tokenize_offloop as T, exllamav3.tokenizer.tokenizer as E; print(T.REVISION, int(T.encode_once_enabled()), int(T.offloop_enabled()), int(E.tokenize_offloop_enabled()), T.offload_min_chars(), int(T.should_offload(12000)), int(T.should_offload(12001)))'
rb_ok(){ [[ "$1" = NEW || "$1" = OLD ]] && [ "$2" = "prefill-merge-r1 1 1" ]; }
FN_MD5=b457fa447eee9919b28dfd088c96bc9c
API=http://127.0.0.1:8022/v1

# ---------------- the benchmark (R787d's, unchanged) ----------------
DS=$D/datasets/std-bench
SG=$DS/ShareGPT_V3_unfiltered_cleaned_split.json; SG_SHA=35f0e213
SB=$DS/spec_bench_question.jsonl;                 SB_SHA=4b6d33e7
CLIENT_IMG=${CLIENT_IMG:-vllm/vllm-openai:v0.30.0}
P=$D/probes
PROBES="vllm_bench_tabby.py parse_container.py std_bench_summary.py r811_ab_compare.py test_r811_ab_compare.py fn_bench.py r813_curve_compare.py test_r813_curve_compare.py r826_report.py r826_identity.py"
PASSES=${PASSES:-"A B"}
DATASETS=${DATASETS:-"sharegpt specbench"}
CONCS=${CONCS:-"1 2 4 8"}
SG_N=${SG_N:-400}
SB_N=${SB_N:-480}
SEED=${SEED:-7310}
SB_OUT=${SB_OUT:-256}
WANT_GPC=${WANT_GPC:-"0 0"}
WANT_MEM=${WANT_MEM:-"4500 4500"}
# Flash-Next publishes at stock: every boot must read power.limit == power.default_limit on every card
WANT_PWR=${WANT_PWR:-$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | awk '{printf "%s%.0f", (NR>1?" ":""), $1}')}
SPREAD_MAX=${SPREAD_MAX:-3}
SPREAD_EXEMPT=${SPREAD_EXEMPT-1}
CACHE_MAX=${CACHE_MAX:-0.01}
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-1200}
CLIENT_NAME=$UNIT-client
QUIESCE=${QUIESCE-"hermes hermes-webui owui-proxy"}
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
note(){ log "$*"; echo "$*" >> "$R/summary.txt"; }
md5f(){ md5sum < "$1" 2>/dev/null | cut -c1-32; }
iid(){ sudo docker image inspect "$1" -f '{{.Id}}' 2>/dev/null; }
# per-card offsets, NVML index order (0 = ASUS, 1 = HP; r730). The unit only reads them.
gpcoff(){ timeout 30 sudo python3 -c 'import pynvml as N;N.nvmlInit();print(*[N.nvmlDeviceGetGpcClkVfOffset(N.nvmlDeviceGetHandleByIndex(i)) for i in range(N.nvmlDeviceGetCount())])' 2>/dev/null || echo "?"; }
memoff(){ timeout 30 sudo python3 -c 'import pynvml as N;N.nvmlInit();print(*[N.nvmlDeviceGetMemClkVfOffset(N.nvmlDeviceGetHandleByIndex(i)) for i in range(N.nvmlDeviceGetCount())])' 2>/dev/null || echo "?"; }
pwrlim(){ nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%s%.0f", (NR>1?" ":""), $1}'; }
# launchers_ok: both launcher files are the pinned ones (prints the first reason, rc 1)
launchers_ok(){ local a m; for a in $ARMS; do m=$(md5f "${AL[$a]}")
  [ "$m" = "${AMD5[$a]}" ] || { echo "$a launcher ${AL[$a]} md5 '${m:-missing}' != ${AMD5[$a]}"; return 1; }; done; return 0; }
# images_ok: both tags still resolve to the pinned ids (a retag / prune would measure another build)
images_ok(){ local a i; for a in $ARMS; do i=$(iid "${AIMG[$a]}")
  [ "$i" = "${AIMGID[$a]}" ] || { echo "$a image ${AIMG[$a]} is '${i:-missing}', not the pinned ${AIMGID[$a]:7:12}"; return 1; }; done; return 0; }

# ---------------- checks before the lock (CPU only; nothing touched) ----------------
for f in "$LIVE" "${AL[OLD]}" "$D/lib/gpu-queue.sh" "$D/lib/serve-ctl.sh" "$D/lib/gateway-drain.sh" \
         $(for x in $PROBES; do echo "$P/$x"; done) "$FIX/env-NEW.txt" "$FIX/env-OLD.txt" "$SG" "$SB" "$MDIR/tokenizer.json" "$MDIR/tokenizer_config.json"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
why=$(launchers_ok) || { log "ABORT (before the lock, nothing touched): $why"; exit 3; }
for a in $ARMS; do
  cp "${AL[$a]}" "$R/launchers/launch-$a.sh"
  l=$R/launchers/launch-$a.sh; BOOT_L[$a]=$l
  [ "$(md5f "$l")" = "${AMD5[$a]}" ] || { log "ABORT: snapshot $a changed during copy"; exit 3; }
  # the launcher defaults this comparison depends on (the md5 pins them; checked by name so a reader sees what)
  [ "$(grep -oE '^DAILY_IMG=\S+' "$l" | cut -d= -f2)" = "${AIMG[$a]}" ] || { log "ABORT: $a launcher DAILY_IMG is not ${AIMG[$a]}"; exit 3; }
  grep -qxF "CACHE=\${CACHE:-$WANT_POOL}" <(grep -oE '^CACHE=\S+' "$l") || { log "ABORT: $a launcher CACHE default is not $WANT_POOL"; exit 3; }
  grep -qE "^GPU_SPLIT=\\\$\{GPU_SPLIT:-$WANT_SPLIT\}" "$l" || { log "ABORT: $a launcher GPU_SPLIT default is not $WANT_SPLIT"; exit 3; }
  grep -qE "^TUNEDIR=\\\$\{TUNEDIR:-$WANT_TUNEDIR\}" "$l" || { log "ABORT: $a launcher TUNEDIR default is not $WANT_TUNEDIR"; exit 3; }
  wenv=$(sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$/\1/p' "$l" | head -1)
  [ "$(echo $wenv | wc -w)" = "${AKEYS[$a]}" ] || { log "ABORT: $a launcher EXTRA_ENV default has $(echo $wenv | wc -w) keys, want ${AKEYS[$a]}"; exit 3; }
  for k in ${AREQ[$a]}; do case $k in
    -*) echo " $wenv " | grep -q " ${k#-}=" && { log "ABORT: $a launcher sets ${k#-}"; exit 3; } ;;
    *)  echo " $wenv " | grep -q " $k=1 " || { log "ABORT: $a launcher does not set $k=1"; exit 3; } ;; esac; done
done
why=$(images_ok) || { log "ABORT (before the lock, nothing touched): $why"; exit 3; }
sudo docker image inspect "$CLIENT_IMG" >/dev/null 2>&1 || { log "ABORT: $CLIENT_IMG not on flan (the unit never pulls)"; exit 3; }
# snapshot the inputs: a probe edited while the unit is queued must not change what this run measures
for x in $PROBES; do cp "$P/$x" "$R/probes/"; done
cp "$0" "$R/" 2>/dev/null
cp "$FIX"/env-{NEW,OLD}.txt "$R/probes/"
[ "$(md5f "$R/probes/fn_bench.py")" = "$FN_MD5" ] || { log "ABORT: fn_bench is not R813 instrument"; exit 3; }
for x in $PROBES; do
  python3 -m py_compile "$R/probes/$x" 2>/dev/null || { log "ABORT: $x does not compile"; exit 3; }; done
mkdir -p "$R/.selftest-tmp"
TMPDIR=$R/.selftest-tmp python3 "$R/probes/test_r811_ab_compare.py" > "$R/selftest-r811_ab_compare.txt" 2>&1 \
  || { log "ABORT: test_r811_ab_compare.py fails: $(tail -3 "$R/selftest-r811_ab_compare.txt" | tr '\n' ' ')"; exit 3; }
TMPDIR=$R/.selftest-tmp python3 "$R/probes/test_r813_curve_compare.py" > "$R/selftest-r813_curve_compare.txt" 2>&1 \
  || { log "ABORT: test_r813_curve_compare.py fails"; exit 3; }
rm -rf "$R/.selftest-tmp" "$R/probes/__pycache__"
sha(){ sha256sum "$1" | cut -c1-8; }
[ "$(sha "$SG")" = "$SG_SHA" ] || { log "ABORT: ShareGPT sha256 $(sha "$SG") != $SG_SHA"; exit 3; }
[ "$(sha "$SB")" = "$SB_SHA" ] || { log "ABORT: Spec-Bench sha256 $(sha "$SB") != $SB_SHA"; exit 3; }
SB_ROWS=$(grep -c . "$SB")
[ "$SB_N" -le "$SB_ROWS" ] || { log "ABORT: SB_N $SB_N > $SB_ROWS Spec-Bench rows"; exit 3; }
CENV=(-e VLLM_NO_USAGE_STATS=1 -e VLLM_DO_NOT_TRACK=1 -e DO_NOT_TRACK=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e HF_DATASETS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e VLLM_USE_RUST_BENCH=0 -e TABBY_FORCE_LEN=1)
# preflight (R787d's, unchanged): the shim patches this image (manifest recorder included), the tokenizer loads
# offline, and the full Spec-Bench file loads through the shim's pandas-free reader
sudo docker run --rm --network none "${CENV[@]}" -e TABBY_SAMPLES_OUT=/tmp/preflight.samples.tsv \
  -v "$MDIR":/model:ro -v "$DS":/data:ro -v "$R/probes":/probes:ro --entrypoint python3 "$CLIENT_IMG" -c '
import sys; sys.path.insert(0, "/probes")
import vllm_bench_tabby as s; s.install()
import vllm.benchmarks.serve as S
assert S.get_samples.__name__ == "get_samples_recorded", "manifest recorder not installed"
from vllm.tokenizers import get_tokenizer
t = get_tokenizer("/model")
m = t.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True, tokenize=False)
print("preflight: tokenizer", type(t).__name__, "vocab", len(t), "| template tail", repr(m[-48:]))
from vllm.benchmarks.datasets import datasets as D
d = D.SpecBench(dataset_path="/data/spec_bench_question.jsonl")
print("preflight: specbench rows", len(d.data))' > "$R/preflight.log" 2>&1
grep -q '^shim: patched.*sample manifest ON' "$R/preflight.log" && grep -q '^preflight: tokenizer' "$R/preflight.log" \
  && grep -qE "^preflight: specbench rows $SB_ROWS\$" "$R/preflight.log" \
  || { log "ABORT: preflight failed: $(grep -avE '^\s*$' "$R/preflight.log" | tail -2 | cut -c1-200)"; exit 3; }
grep -aE '^(shim|preflight):' "$R/preflight.log" | sed 's/^/  /' | tee -a "$R/audit.log"
log "probes (snapshot md5): $(cd "$R/probes" && for x in $PROBES; do printf '%s %s; ' "$x" "$(md5f "$x" | cut -c1-8)"; done)self-test: $(tail -1 "$R/selftest-r811_ab_compare.txt")"
init_results(){
for a in $ARMS; do
  mkdir -p "$R/$a/results" "$R/$a/cells"
  printf 'tag\tpass\tdataset\tconc\tt_boot\tlauncher_md5\timage\timage_id\tenv_n\tenv_sha\tkeys_n\twindow\tslots\tcache\tpolicy\tpower_limit_w\tgpc_boot\tmem_boot\tgpc_after\tmem_after\tvram_free\tmem_line\tpwr_after\tpwr_line\tarm\treadback\n' > "$R/$a/boots.tsv"
  printf 'tag\tpass\tdataset\tconc\tserial_lo\tserial_hi\twarmups\tcatfile\trc\tcontainer\tn_req\tt0\tt1\n' > "$R/$a/runs.tsv"
  : > "$R/$a/foreign.tsv"   # no header row, as R787d (the end block's awk reads every row as tag<TAB>count)
done
}
init_results
R=$ROOT/c4-repeat; init_results; R=$ROOT
note "R826 paired standard instruments: NEW ${AL[NEW]} (${AMD5[NEW]:0:8}, ${AIMG[NEW]} ${AIMGID[NEW]:7:12}, ${AKEYS[NEW]} keys) vs OLD ${AL[OLD]} (${AMD5[OLD]:0:8}, ${AIMG[OLD]} ${AIMGID[OLD]:7:12}, ${AKEYS[OLD]} keys); passes $PASSES (A: NEW then OLD, B: OLD then NEW); datasets $DATASETS; concs $CONCS; SG_N $SG_N seed $SEED; SB_N $SB_N out $SB_OUT; want core $WANT_GPC / memory $WANT_MEM / power $WANT_PWR W; results $R"

# ---------------- queue + lock ----------------
export GPU_QUEUE_NAME=$UNIT
. $D/lib/gpu-queue.sh
. $D/lib/serve-ctl.sh
. $D/lib/gateway-drain.sh
SCTL_LOG="$R/audit.log"
trap 'log "signal while queued"; exec 9>/srv/qwen5090/gpu-exclusive.lock; if flock -n 9 && [ -z "$(served_id)" ] && [ -z "$(gpu_queue_others)" ]; then log "GPUs free and :8022 empty: booting the Flash-Next daily from $LIVE"; env -i HOME="$HOME" PATH=$CLEAN_PATH bash "$LIVE" > "$R/boot-queued-restore.log" 2>&1; log "daily: $(served_id || echo none)"; fi; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; exit 4' TERM INT HUP
gpu_lock
why=$(launchers_ok) || true
[ -n "$why" ] || why=$(images_ok) || true
if [ -n "$why" ]; then
  log "ABORT after the lock: $why"
  if [ -z "$(served_id)" ]; then BOOTED=1; finish_restore "$LIVE" > "$R/boot-restore.log" 2>&1; log "daily: $(served_id || echo none)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; exit 3
fi
log "after the lock: both launchers and both image ids unchanged"

DECISION="VOID the unit ended before the summary"
FINISHED=0 WAS_RUNNING= REPORTED=0
running(){ [ "$(sudo docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }
quiesce(){ local c s=; for c in $QUIESCE; do running "$c" || continue; sudo docker stop -t 30 "$c" >/dev/null 2>&1 && s="$s $c"; done; echo "${s# }"; }
# per-arm summaries (std_bench_summary.py, unchanged, R787d's arguments) + the paired comparison. On a VOID / signal it
# runs on whatever is on disk (the comparison is then partial; the per-arm decisions say why).
report(){ [ "$REPORTED" = 1 ] && return 0; REPORTED=1; local a section
  R=$ROOT
  for section in "$ROOT" "$ROOT/c4-repeat"; do
    R=$section
    for a in $ARMS; do
      [ "$(tail -n +2 "$R/$a/runs.tsv" 2>/dev/null | wc -l)" -gt 0 ] || continue
    python3 "$ROOT/probes/std_bench_summary.py" --runs "$R/$a/runs.tsv" --results "$R/$a/results" --boots "$R/$a/boots.tsv" \
      --sb-data "$SB" --sb-out "$SB_OUT" --want-gpc "$WANT_GPC" --want-mem "$WANT_MEM" --want-pwr "$WANT_PWR" \
      --cache-max "$CACHE_MAX" --spread-max "$SPREAD_MAX" --spread-exempt "$SPREAD_EXEMPT" --decide "$PASSES" \
      --json "$R/$a/summary.json" > "$R/$a/summary.txt" 2>&1
      sed "s/^/[$section $a] /" "$R/$a/summary.txt" >> "$ROOT/summary.txt"
    done
  done
  R=$ROOT
  python3 "$ROOT/probes/r826_report.py" --root "$ROOT" --sb-data "$SB" --json "$ROOT/compare.json" \
    > "$ROOT/final-table.txt" 2>&1 || log "WARN: R826 report failed (final-table.txt)"
  cat "$ROOT/final-table.txt" | tee -a "$ROOT/summary.txt" | tee -a "$ROOT/audit.log"
}
finish(){ [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  trap 'log "signal during finish: ignored"' TERM INT HUP
  sudo docker rm -f "$CLIENT_NAME" >/dev/null 2>&1
  R=$ROOT
  # Stop measurement boot; restore LIVE with its own defaults only when last in queue (§12).
  if [ "${BOOTED:-0}" = 1 ] || [ -z "$(served_id)" ]; then
    served_stop; wait_unserved 45
    finish_restore "$LIVE" > "$R/boot-restore.log" 2>&1
  else log "restore: not needed (the daily was never stopped)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  local img imgid tier; img=$(sudo docker inspect -f '{{.Config.Image}}' flashnext 2>/dev/null); imgid=$(sudo docker inspect -f '{{.Image}}' flashnext 2>/dev/null)
  tier=$(sudo docker exec flashnext env 2>/dev/null | grep -c '^EXL3_NVME_TIER=')
  log "after: served $(served_id || echo none); image ${img:-none} ${imgid:7:12}; tier ${tier:-0}; launcher md5s now NEW $(md5f "$LIVE" | cut -c1-8) OLD $(md5f "${AL[OLD]}" | cut -c1-8); core $(gpcoff); memory $(memoff); power $(pwrlim) W"
  if [ -n "$(served_id)" ] && { [ "$img" != "${AIMG[NEW]}" ] || [ "$imgid" != "${AIMGID[NEW]}" ]; }; then
    log "WARN: :8022 is not the production daily (want ${AIMG[NEW]} ${AIMGID[NEW]:7:12}): restore it by hand from $LIVE"; fi
  [ -n "$(served_id)" ] || [ -n "$(gpu_queue_others)" ] || log "WARN: :8022 not serving after the restore and no unit queued: boot $LIVE by hand"
  local now=; now=$(quiesce); [ -n "$now" ] && log "stopped after the restore: $now (started below if it ran at the start)"
  if [ -n "$WAS_RUNNING" ]; then
    [ -n "$(served_id)" ] || log "WARN: :8022 not serving (another unit queued?); restarting the direct clients anyway"
    if sudo docker start $WAS_RUNNING >/dev/null 2>&1; then log "restarted: $WAS_RUNNING"
    else log "RESTART FAILED: $WAS_RUNNING (start by hand: sudo docker start $WAS_RUNNING)"; fi
  else log "no direct client to restart"; fi
  log "measurement status before analysis: $DECISION"
  report
  sudo chown -R "$(stat -c %U $D/results)" "$R" 2>/dev/null || true
  echo "DECISION: $DECISION" >> "$R/summary.txt"
  log "=== $UNIT $1 ==="; log "DECISION: $DECISION"; }
void(){ DECISION="VOID $*"; finish VOID; exit 3; }
trap 'log "signal"; DECISION="VOID signal (the run was interrupted)"; finish ABORTED; exit 4' TERM INT HUP
trap 'rc=$?; [ "$FINISHED" = 1 ] || { DECISION="VOID unexpected exit rc $rc"; finish ABORTED; }' EXIT

for c in $QUIESCE; do running "$c" && WAS_RUNNING="$WAS_RUNNING $c"; done; WAS_RUNNING=${WAS_RUNNING# }
st=$(quiesce)
gateway_drain   # the cells run on the live :8022 port; Olla routes nothing to it until this unit exits
log "lock held; served at entry: $(served_id || echo none); direct clients stopped: ${st:-none}; offsets now core $(gpcoff) / memory $(memoff), power $(pwrlim) W"
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || log "WARN: Olla still has requests in flight after 900 s"
why=$(launchers_ok) || void "under the lock: $why"

# ---------------- one cell ----------------
# B_* = the current boot's record; written to the arm's boots.tsv once the cell is over (with the after-cell offsets)
bootrow(){ printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$B_TAG" "$B_PASS" "$B_DS" "$B_CONC" "$B_T" "$B_MD5" "$B_IMG" "$B_IMGID" "$B_ENVN" "$B_ENVSHA" "$B_KEYS" "$B_WIN" \
  "$B_SLOTS" "$B_CACHE" "$B_POLICY" "$B_PWR" "$B_GPC" "$B_MEM" "${1:--}" "${2:--}" "$B_FREE" "$B_MEMLINE" "${3:--}" \
  "$B_PWRLINE" "$B_ARM" "${B_RB:--}" >> "${BOOT_ROWS:-$R/$B_ARM/boots.tsv}"; }
declare -A REF
boot(){ local arm=$1 tag=$2 lt=$1/$2 L C bl why= full pv st
  L=${BOOT_L[$arm]} C=$R/$arm/cells
  full=$(launchers_ok) || void "[$lt] source identity: $full"
  B_ARM=$arm B_TAG=$tag B_PASS=$3 B_DS=$4 B_CONC=$5 B_T=$(date -Is) B_RB= B_IMG= B_IMGID= B_ENVN= B_ENVSHA= B_KEYS= B_WIN=
  B_SLOTS= B_CACHE= B_POLICY= B_PWR= B_FREE= B_MEMLINE= B_PWRLINE= B_GPC= B_MEM=; bl=$C/boot-$tag.log
  served_stop; wait_unserved 45
  B_MD5=$(md5f "$L")
  [ "$B_MD5" = "${AMD5[$arm]}" ] || void "[$lt] launcher $L md5 '$B_MD5' != ${AMD5[$arm]} (changed mid-run)"
  env -i HOME="$HOME" PATH=$CLEAN_PATH NVME_TIER= bash "$L" > "$bl" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { sudo docker logs flashnext > "$C/container-$tag.log" 2>&1
      void "[$lt] NO BOOT: $(grep -aE 'Insufficient VRAM|out of memory|Error|ABORT|NO BOOT' "$bl" | tail -1 | cut -c1-160)"; }
  st=$(quiesce); [ -n "$st" ] && log "[$lt] stopped again after the boot: $st"
  sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$C/env-$tag.txt"
  B_IMG=$(sudo docker inspect -f '{{.Config.Image}}' flashnext 2>/dev/null); full=$(sudo docker inspect -f '{{.Image}}' flashnext 2>/dev/null)
  B_IMGID=$full
  B_ENVN=$(wc -l < "$C/env-$tag.txt"); B_ENVSHA=$(sha256sum "$C/env-$tag.txt" | cut -c1-12)
  B_KEYS=$(grep -aoE 'env keys \([0-9]+\)' "$bl" | tail -1 | grep -oE '[0-9]+'); B_WIN=$(grep -c '^EXL3_MTP_KV_WINDOW=' "$C/env-$tag.txt")
  B_SLOTS=$(grep -aoE 'slots [0-9]+' "$bl" | tail -1 | cut -d' ' -f2); B_CACHE=$(grep -aoE 'cache [0-9]+' "$bl" | tail -1 | cut -d' ' -f2)
  B_POLICY=$(grep -aoE "policy '[^']*'" "$bl" | tail -1 | sed "s/^policy //; s/'//g")
  B_PWR=$(pwrlim)
  B_FREE=$(grep -aoE 'VRAM free MiB [0-9/]+' "$bl" | tail -1 | awk '{print $4}')
  B_MEMLINE=$(grep -aoE 'memory clock offset: .*' "$bl" | tail -1 | tr '\t' ' ')
  B_PWRLINE=$(grep -aoE 'power policy .*readback .*' "$bl" | tail -1 | tr '\t' ' ')
  B_GPC=$(gpcoff); B_MEM=$(memoff)
  # full output kept; the readback is its last `absent <name>` / `<rev> <0|1> <0|1>` line (else the last line, which fails)
  sudo docker exec -e CUDA_VISIBLE_DEVICES= flashnext python3 -c "$RB_PY" > "$C/readback-$tag.txt" 2>&1
  [ "$?" = 0 ] || why="$why merge readback command failed;"
  B_RB=$(grep -aE '^(absent \S+|\S+ [01] [01])$' "$C/readback-$tag.txt" | tail -1)
  B_RB=${B_RB:-$(tail -1 "$C/readback-$tag.txt" | tr '\t' ' ' | cut -c1-160)}
  log "[$lt] UP: image $B_IMG (${full:7:12}); launcher md5 ${B_MD5:0:8}; keys $B_KEYS (container EXL3_ env $B_ENVN, sha $B_ENVSHA, window $B_WIN); readback '$B_RB'; slots $B_SLOTS cache $B_CACHE policy '$B_POLICY'; power $B_PWR W (launcher: ${B_PWRLINE:-no power line}); core offsets $B_GPC; memory offsets $B_MEM (launcher: ${B_MEMLINE:-no memory-offset line}); VRAM free $B_FREE"
  # the arm's provenance (change (3) of the header); every reason is collected, then the cell is VOID
  pv=$(assert_env_keys "$bl" "${AKEYS[$arm]}" ${AREQ[$arm]} 2>&1) || why="$why env keys: $(echo $pv);"
  [ "$B_IMG" = "${AIMG[$arm]}" ] || why="$why image '$B_IMG' != ${AIMG[$arm]};"
  [ "$full" = "${AIMGID[$arm]}" ] || why="$why image id '${full:7:12}' != pinned ${AIMGID[$arm]:7:12};"
  pv=$(python3 "$ROOT/probes/r826_identity.py" --arm "$arm" --launcher "$L" --env "$C/env-$tag.txt" --expected "$ROOT/probes/env-$arm.txt" 2>&1) \
    || why="$why exact identity: $pv;"
  sudo docker exec -e CUDA_VISIBLE_DEVICES= -w /app flashnext python3 -c "$TOK_PY" > "$C/tokenize-$tag.txt" 2>&1
  [ "$?" = 0 ] && [ "$(tail -1 "$C/tokenize-$tag.txt")" = "tokenize-offloop-r2 1 1 1 12000 0 1" ] \
    || why="$why tokenize-offloop readback failed;"
  if [ "$arm" = NEW ]; then
    sudo docker exec -e CUDA_VISIBLE_DEVICES= flashnext python3 /opt/r825c/landing_r825c.py > "$C/landing-$tag.txt" 2>&1
    [ "$?" = 0 ] && grep -qxF 'R825c landing PASS: inherited stack and host preparation delta imported and byte verified' "$C/landing-$tag.txt" \
      || why="$why R825c source landing failed;"
  fi
  grep -qx 'EXL3_GR_MIX_TILED=1' "$C/env-$tag.txt" || why="$why container env lacks EXL3_GR_MIX_TILED=1;"
  rb_ok "$arm" "$B_RB" || why="$why prefill-merge readback '$B_RB';"
  grep -aqF "cache $WANT_POOL @" "$bl" || why="$why pool $(grep -aoE 'cache [0-9]+ @' "$bl" | tail -1) != $WANT_POOL;"
  grep -aqF "split [$WANT_SPLIT]" "$bl" || why="$why $(grep -aoE 'split \[[^]]*\]' "$bl" | tail -1) != split [$WANT_SPLIT];"
  grep -aqF "tunedir $WANT_TUNEDIR," "$bl" || why="$why tunedir is not $WANT_TUNEDIR;"
  grep -q '^EXL3_NVME_TIER=' "$C/env-$tag.txt" && why="$why NVMe tier on;"
  [ "$B_PWR" = "$WANT_PWR" ] || why="$why power limits '$B_PWR' W, want stock '$WANT_PWR' W;"
  [ "$B_GPC" = "$WANT_GPC" ] && [ "$B_MEM" = "$WANT_MEM" ] || why="$why clock offsets core '$B_GPC' memory '$B_MEM', want core '$WANT_GPC' memory '$WANT_MEM';"
  # R787d's "same config at every boot", per arm
  if [ -z "${REF[$arm]:-}" ]; then REF[$arm]="$B_MD5/$full/$B_ENVSHA"
  elif [ "$B_MD5/$full/$B_ENVSHA" != "${REF[$arm]}" ]; then why="$why config changed mid-run: $B_MD5/${full:7:12}/$B_ENVSHA vs the arm's first boot ${REF[$arm]};"; fi
  if [ -n "$why" ]; then bootrow; void "[$lt] not the $arm arm:$why"; fi; }

# conc concurrent non-stream chat requests on short prompts that are in neither dataset (R787d's warm-up, unchanged)
warm(){ python3 - "$MODEL" "$2" "$1" <<'EOF'
import json, sys, threading, urllib.request
model, conc, tag = sys.argv[1], max(int(sys.argv[2]), 1), sys.argv[3]
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
got = []
def one(i):
    body = {"model": model, "stream": False, "temperature": 0, "max_tokens": 64, "min_tokens": 64,
            "messages": [{"role": "user", "content": f"Warm-up {tag} slot {i}: describe a lighthouse at dusk in three sentences."}]}
    try:
        rq = urllib.request.Request("http://127.0.0.1:8022/v1/chat/completions", json.dumps(body).encode(),
                                    {"Content-Type": "application/json"})
        got.append((json.load(op.open(rq, timeout=180)).get("usage") or {}).get("completion_tokens"))
    except Exception as e:
        print(f"warm-up error: {e.__class__.__name__}: {e}"[:200])
ts = [threading.Thread(target=one, args=(i,)) for i in range(conc)]
[t.start() for t in ts]; [t.join() for t in ts]
print(f"{len(got)}/{conc} ok, completion tokens {got}")
EOF
}
# parsed (stream) request lines in a cell's container log: "<count with serial > lo> <max serial>"   (args: arm tag lo)
lines(){ sudo docker logs flashnext > "$R/$1/cells/container-$2.log" 2>&1
  python3 "$R/probes/parse_container.py" "$R/$1/cells/container-$2.log" | python3 -c 'import sys, json
lo = int(sys.argv[1]); s = [json.loads(l)["serial"] for l in sys.stdin if l.strip()]
print(sum(v > lo for v in s), max(s or [lo]))' "$3"; }
# the highest request serial logged so far, any endpoint, stream or not
maxserial(){ sudo docker logs flashnext 2>&1 | grep -aoE '#[0-9]+ (chat/)?completions' | grep -oE '[0-9]+' | sort -n | tail -1; }
# wait (<= 20 s) until the warm-ups' own headers are visible (docker's stdout is block-buffered), then LO = max serial
warm_lo(){ local lt=$1 conc=$2 lo0 k m
  lo0=$(maxserial); lo0=${lo0:-0}
  log "[$lt] warm-up: $(warm "${lt//\//-}" "$conc" 2>&1 | tail -1)"
  for k in $(seq 10); do m=$(maxserial); m=${m:-0}; [ "$m" -ge $(( lo0 + conc )) ] && break; sleep 2; done
  [ "$m" -ge $(( lo0 + conc )) ] || log "[$lt] WARN: warm-up headers not visible after 20 s (max serial $m, before $lo0)"
  LO=$m; }
# requests after `lo` whose header lacks min_tokens (every request of this unit carries it): foreign traffic  (args: arm tag lo)
foreign(){ python3 - "$R/$1/cells/container-$2.log" "$3" <<'PY'
import re, sys
t = open(sys.argv[1], errors="replace").read(); lo = int(sys.argv[2])
h = re.findall(r"INFO:\s+#(\d+) (?:chat/)?completions(?: \([\w-]+\))?: [\d,]+ prompt tokens ·\s+(.*?)(?=\n\S|\Z)", t, re.S)
print(sum(1 for n, rest in h if int(n) > lo and "min_tokens" not in rest))
PY
}

first=1
cell(){ local arm=$1 ps=$2 ds=$3 conc=$4 n=$5 rc lo hi k cnt t0 t1 tag=$2-$3-c$4 lt C; shift 5
  lt=$arm/$tag C=$R/$arm/cells
  boot "$arm" "$tag" "$ps" "$ds" "$conc"
  warm_lo "$lt" "$conc"; lo=$LO
  t0=$(date -Is)
  timeout -k 30 "$CLIENT_TIMEOUT" sudo docker run --rm --name "$CLIENT_NAME" --network host "${CENV[@]}" \
    -e TABBY_SAMPLES_OUT="/out/$tag.samples.tsv" \
    -v "$MDIR":/model:ro -v "$DS":/data:ro -v "$R/probes":/probes:ro -v "$R/$arm/results":/out \
    --entrypoint python3 "$CLIENT_IMG" /probes/vllm_bench_tabby.py bench serve \
    --backend openai-chat --base-url http://127.0.0.1:8022 --endpoint /v1/chat/completions \
    --model "$MODEL" --tokenizer /model --temperature 0 \
    --request-rate inf --max-concurrency "$conc" --num-warmups 0 --num-prompts "$n" --no-oversample \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir /out --result-filename "$tag.json" --disable-tqdm \
    --metadata "unit=$UNIT" "tag=$tag" "arm=$arm" "pass=$ps" "dataset=$ds" "conc=$conc" "num_warmups=0" \
      "launcher_md5=${AMD5[$arm]:0:8}" "image=${AIMG[$arm]}" "image_id=${AIMGID[$arm]:7:12}" \
      "unit_warmup=${conc}x non-stream out-of-set" "boot=fresh per cell" "thinking=template-default" \
      "length_forcing=min_tokens" "client=vllm-v0.30.0+vllm_bench_tabby" \
    "$@" > "$C/client-$tag.log" 2>&1
  rc=$?; t1=$(date -Is); sudo docker rm -f "$CLIENT_NAME" >/dev/null 2>&1
  # docker's stdout is block-buffered: wait (<= 60 s) until the container log holds a line for every request
  for k in $(seq 20); do read -r cnt hi < <(lines "$arm" "$tag" "$lo"); cnt=${cnt:-0} hi=${hi:-$lo}; [ "$cnt" -ge "$n" ] && break; sleep 3; done
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$ps" "$ds" "$conc" "$lo" "$hi" 0 - "$rc" \
    "cells/container-$tag.log" "$n" "$t0" "$t1" >> "$R/$arm/runs.tsv"
  local ga ma pa fr; ga=$(gpcoff); ma=$(memoff); pa=$(pwrlim); bootrow "$ga" "$ma" "$pa"
  fr=$(foreign "$arm" "$tag" "$lo")
  printf '%s\t%s\n' "$tag" "${fr:-?}" >> "$R/$arm/foreign.tsv"
  [ "${fr:-0}" = 0 ] || log "[$lt] WARN: ${fr} foreign (non-benchmark) requests hit the daily during this cell"
  # REVIEW-R811-code SHOULD-FIX 1 (report only): the readback runs in a fresh process and cannot see a self-disable in
  # the served one. Expected: NEW 0 'disabled' lines; OLD 0 'prefill-merge-r1' lines.
  log "[$lt] prefill-merge lines in the container log: $(grep -ac 'prefill-merge-r1' "$C/container-$tag.log"); disabled: $(grep -a 'EXL3_PREFILL_MERGE disabled for this process' "$C/container-$tag.log" | tail -1 | cut -c1-160)"
  log "[$lt] rc $rc; $(grep -c '^shim: patched' "$C/client-$tag.log") shim line; $(grep -aoE '^shim: [0-9]+ samples' "$C/client-$tag.log" | cut -d' ' -f2) sampled; server lines $cnt/$n (serials $lo..$hi); $(grep -aE '^(Successful requests|Failed requests|Benchmark duration|Output token throughput|Mean TTFT|Median TPOT)' "$C/client-$tag.log" | sed -E 's/ {2,}/ /g' | paste -sd';' -); OOM $(grep -acE 'OutOfMemoryError|out of memory' "$C/container-$tag.log") TORCH_CHECK $(grep -acE 'TORCH_CHECK|c10::Error' "$C/container-$tag.log") tracebacks $(grep -ac Traceback "$C/container-$tag.log") unsupported $(grep -aci 'unsupported' "$C/container-$tag.log"); offsets after core $ga memory $ma, power $pa W; foreign ${fr:-?}"
  # the shim's main() path first runs here: a client that produced nothing must not burn 31 more boots
  if [ $first = 1 ]; then first=0
    [ -s "$R/$arm/results/$tag.json" ] && [ -s "$R/$arm/results/$tag.samples.tsv" ] && grep -q '^shim: patched' "$C/client-$tag.log" \
      || void "first client run produced no result/manifest: $(grep -avE '^\s*$' "$C/client-$tag.log" | tail -2 | cut -c1-200)"; fi
  [ "$ga" = "$WANT_GPC" ] && [ "$ma" = "$WANT_MEM" ] && [ "$pa" = "$WANT_PWR" ] || void "[$lt] clocks / power after the cell: core '$ga' memory '$ma' power '$pa' W"; }

# ---------------- the matrix: pass outer (as R787d), arm order NEW OLD in pass A, OLD NEW in pass B ----------------
std_matrix(){
for ps in $PASSES; do
  case $ps in A) order="NEW OLD" ;; B) order="OLD NEW" ;; *) void "unknown pass '$ps' (the arm order is defined for A and B)" ;; esac
  for ds in $DATASETS; do
    for c in $CONCS; do
      for arm in $order; do
        case $ds in
          sharegpt)  cell "$arm" "$ps" sharegpt "$c" "$SG_N" --dataset-name sharegpt --dataset-path "/data/$(basename "$SG")" --seed "$SEED" ;;
          specbench) cell "$arm" "$ps" specbench "$c" "$SB_N" --dataset-name spec_bench --dataset-path "/data/$(basename "$SB")" \
                       --spec-bench-output-len "$SB_OUT" --seed "$SEED" ;;
          *) void "unknown dataset $ds" ;;
        esac
      done
    done
  done
done
}
std_matrix
# R811b composite rule: a second independent c4 pair, regardless of the first pair's spread.
R=$ROOT/c4-repeat
mkdir -p "$R/probes"; cp "$ROOT/probes/"*.py "$R/probes/"
DATASETS_SAVED=$DATASETS CONCS_SAVED=$CONCS
DATASETS=sharegpt CONCS=4
std_matrix
DATASETS=$DATASETS_SAVED CONCS=$CONCS_SAVED
R=$ROOT

# ---------------- decode instrument: R813 measurement loop, unchanged ----------------
decode_arm(){ local tag=$1 arm=${1%?} ck cko decode_lo fr
  BOOT_ROWS=$R/decode-boots.tsv
  boot "$arm" "$tag" decode decode 0
  cp "$R/$arm/cells/boot-$tag.log" "$R/boot-$tag.log"
  cp "$R/$arm/cells/env-$tag.txt" "$R/env-$tag.txt"
  decode_lo=$(maxserial); decode_lo=${decode_lo:-0}
  for conc in 1 2 3 4 5 6 7 8; do for kind in code prose; do
    python3 "$R/probes/fn_bench.py" --url "$API" --model "$MODEL" --tag "$tag-c$conc-$kind" --kind $kind --distinct \
      --tokens 1024 --warmup-runs 1 --conc $conc --runs 3 --out "$R/records.jsonl" > "$R/bench-$tag-c$conc-$kind.log" 2>&1
  done; done
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  cp "$R/container-$tag.log" "$R/$arm/cells/container-$tag.log"
  fr=$(foreign "$arm" "$tag" "$decode_lo")
  printf '%s\t%s\n' "$tag" "${fr:-?}" >> "$R/decode-foreign.tsv"
  bootrow "$(gpcoff)" "$(memoff)" "$(pwrlim)"
  unset BOOT_ROWS
  ck="power $(pwrlim) W; core $(gpcoff); memory $(memoff)"
  [ "$(pwrlim)" = "$WANT_PWR" ] && [ "$(gpcoff)" = "$WANT_GPC" ] && [ "$(memoff)" = "$WANT_MEM" ] \
    || void "[$tag] clocks / power drifted: $ck"
  log "[$tag] done; $ck; OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$tag.log"); merge disabled: $(grep -a 'EXL3_PREFILL_MERGE disabled for this process' "$R/container-$tag.log" | tail -1)"
}
decode_arm OLD1; decode_arm NEW1; decode_arm NEW2; decode_arm OLD2
# ---------------- R813 analysis (only arm identity parameterised) ----------------
for curve_arm in NEW OLD; do
  mkdir -p "$R/$curve_arm/decode"
  python3 - "$R/records.jsonl" "$R/$curve_arm/decode/records.jsonl" "$curve_arm" <<'PY_RECORDS'
import json, sys
with open(sys.argv[2], 'w') as out:
    for line in open(sys.argv[1]):
        if json.loads(line)['tag'].startswith(sys.argv[3]): out.write(line)
PY_RECORDS
python3 - "$R/records.jsonl" "$R/$curve_arm/decode/curve.tsv" "$curve_arm" <<'PY' 2>&1 | tee "$R/$curve_arm/decode/analysis.txt" | tee -a "$R/audit.log"
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
    v=[x for b in (1,2) for x in d.get(f"{sys.argv[3]}{b}-c{conc}-{kind}",[])]; return f(v) if v else float("nan")
def per_boot(d, conc, kind, f):
    return [f(d[t]) if d.get(t) else float("nan") for t in (f"{sys.argv[3]}1-c{conc}-{kind}", f"{sys.argv[3]}2-c{conc}-{kind}")]
out=open(sys.argv[2],"w"); out.write("conc\tkind\twallagg\tdecode_stream\tdecode_agg\tttft\n")
for kind in ("code","prose"):
    for c in range(1,9):
        v=[m(d,c,kind,f) for d,f in ((agg,st.mean),(dec,st.median),(dagg,st.mean),(ttft,st.median))]
        b=per_boot(dagg,c,kind,st.mean)
        print(f"{kind} c{c}: decode/stream {v[1]:.1f} | decode aggregate {v[2]:.0f} (boots {b[0]:.0f} / {b[1]:.0f}) | round-wall aggregate {v[0]:.0f} | TTFT {v[3]:.2f}s")
        out.write(f"{c}\t{kind}\t"+"\t".join(f"{x:.2f}" for x in v)+"\n")
PY
done
cp "$R/NEW/decode/curve.tsv" "$R/curve.tsv"
cp "$R/NEW/decode/analysis.txt" "$R/analysis.txt"
why=$(launchers_ok) && log "at the end: both launchers unchanged" || log "WARN at the end: $why"
DECISION="DONE report only; no promotion or gate"
finish DONE
