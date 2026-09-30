#!/usr/bin/env bash
# R811 (2026-09-29): paired A/B of the standard benchmark (vllm bench serve v0.30.0 + probes/vllm_bench_tabby.py,
#   ShareGPT V3 + Spec-Bench) between the R809p daily and its predecessor, both on :8022, both booted fresh per cell.
#   NEW = /srv/qwen5090/launch-flashnext.sh (the LIVE launcher; repo flan/launch-flashnext-r809-merge.sh, md5 6429dfa2):
#         image tabbyapi:merge-tok-r1 (id ac16920f72cf), 41 EXL3 keys incl. EXL3_GR_MIX_TILED=1, EXL3_PREFILL_MERGE=1,
#         EXL3_STASH_ASYNC=1 (prefill-merge r1, promoted 2026-09-29 as R809p).
#   OLD = /srv/qwen5090/launch-flashnext.sh.pre-r809 (the R809 rollback; repo flan/launch-flashnext-r808-tokoffloop.sh,
#         md5 04e347cc): image tabbyapi:tokenize-offloop-r2 (id 04b06fa97c4d), 39 keys incl. EXL3_GR_MIX_TILED=1, no
#         merge / async keys.
#   Both (checked in the launcher files before the lock and in every boot log): pool 901,120 @ 8,8, split [30, 30],
#   TUNEDIR /srv/qwen5090/.exl3cache-rebase-dev-r3; stock power, core offset 0, memory offset +4500 (the launchers' own
#   POWER / MEMOC blocks; the unit only reads them); NVMe tier OFF (NVME_TIER= as R787d).
# WHY: (1) measure what prefill-merge does on the standard benchmark, paired against the image it replaced, on the same
#   samples; (2) refresh the README std-bench figure / tables (R787d today, the R785 daily) with the NEW arm's numbers.
#   R803 measured agent-turn prefill -13 % / -17 % (ASYNC) and client TTFT -10 % on the house workloads; this is the
#   first measurement on the public benchmark.
#
# PRE-REGISTERED EXPECTATION (2026-09-29, before any R811 data; an expectation, NOT a gate: nothing is promoted or
#   rolled back on it, and probes/r811_ab_compare.py only reports where the measurement landed against it):
#   output tok/s  NEW/OLD +1 .. +3 % at c4-c8 on both datasets (less prefill time interleaved into decode);
#   TTFT          -30 .. -40 % on Spec-Bench's summarization and rag categories (the long prompts);
#                 ~0 on short prompts (ShareGPT overall, Spec-Bench MT-bench / translation / qa / math_reasoning).
#
# PROTOCOL = R787d's (r787d-std-bench.sh, R731b's header there), with these changes:
#   (1) Two arms, 32 cells: datasets {ShareGPT SG_N 400 seed 7310, Spec-Bench 480 x 256 tokens} x c {1, 2, 4, 8} x
#       passes {A, B} x arms {NEW, OLD}. Every cell is a FRESH boot of that arm's launcher (env -i HOME PATH, NVME_TIER=,
#       bash <launcher>), as R787d.
#   (2) Balanced arm order (ABBA across passes). The outer loop is the pass, as R787d, so an arm's A and B cells of one
#       (dataset, c) are two separate boots a pass apart (~90 min here, ~45 min in R787d: a pass now holds both arms),
#       and R787d's 3 % A/B spread rule keeps its meaning (replication across boots and time). Inside pass A each
#       (dataset, c) boots NEW then OLD; inside pass B, OLD then NEW. For every (dataset, c) the two arms' mean position
#       in time is equal, so a linear drift (thermal, fs cache, clock) favours neither arm.
#   (3) Per-arm provenance in place of r787-common.sh's single pinned daily (r787-common is not sourced): at every
#       boot the arm's launcher md5, the container's image TAG and full image ID (pinned, not just the tag), the
#       resolved env-keys line (41 with EXL3_GR_MIX_TILED EXL3_PREFILL_MERGE EXL3_STASH_ASYNC / 39 with
#       EXL3_GR_MIX_TILED and neither merge key), the container's EXL3_ env, the in-container prefill-merge readback
#       (NEW `prefill-merge-r1 1 1`; OLD: the module absent, `absent exllamav3.cache.prefill_merge`, or disabled
#       `<rev> 0 0`), pool, split, TUNEDIR, tier off, power, core and memory offsets. Any failure VOIDs the cell AND
#       ends the unit (R787d semantics: a provenance failure is systematic and would repeat at every cell of that arm;
#       cells already on disk stay readable, and the comparison runs on what exists).
#   (4) Launchers: neither file is modified. Both are snapshotted into $R/launchers/ at the start, their md5s checked
#       before the lock, after the lock, before every boot and at the end. The OLD arm boots the .pre-r809 file BY PATH;
#       it is never copied over the live launcher.
#   (5) Results: one complete R787d-shaped run dir per arm, $R/<ARM>/{runs.tsv, boots.tsv, foreign.tsv, cells/,
#       results/<A|B>-<dataset>-c<N>.json + .samples.tsv, summary.txt, summary.json}. (The brief's $R/<ARM>/<tag>.json
#       gains the results/ level: std_bench_summary.py resolves each runs.tsv container path against dirname(runs.tsv)
#       and checks one boots.tsv for ONE configuration, so it needs one run dir per arm; the public bench/plot.py
#       std_bench() reads a results dir of [AB]-*-c*.json, so $R/NEW/results is a drop-in for R787D.) std_bench_summary.py
#       runs per arm, unchanged, with R787d's arguments.
#   (6) The paired comparison: probes/r811_ab_compare.py (stdlib; offline self-test probes/test_r811_ab_compare.py, run
#       on flan before the lock, the unit aborts if it fails) prints per dataset x c the NEW/OLD ratio of output tok/s,
#       mean / median / p99 TTFT and mean TPOT for each pass and pooled, the within-arm A/B spread beside it as the
#       noise reference, and the Spec-Bench TTFT per task group (MT-bench pooled, translation, summarization, qa,
#       math_reasoning, rag) with a paired per-prompt ratio. Categories need no shim change: the shim's sample manifest
#       (<tag>.samples.tsv, the order sent, sha256[:16] of each prompt) is index-aligned with --save-detailed's arrays,
#       and question.jsonl maps each hash to its category (as std_bench_summary.py does at c1). The compare verifies
#       the alignment per cell (output_lens == manifest output_len; input_lens - manifest prompt_len constant within
#       +-2 tokens = the chat-template overhead) and marks a cell UNVERIFIED (category rows dropped) otherwise.
#   (7) Standalone: the unit owns what r787-chain.sh did for R787d. GPU queue + lock (lib/gpu-queue.sh), a queued-signal
#       trap (r809t's: boots the daily if the lock is free and :8022 empty), Olla drain + wait-idle, the direct :8022
#       clients (hermes, hermes-webui, owui-proxy) stopped and re-stopped after every boot, and finish(): client removed,
#       served container stopped, the daily restored from the LIVE launcher (NEW, tier on) whatever arm ran last,
#       then exactly the clients that were running at the start restarted. (R787d's "RUN VIA r787-chain.sh ONLY" warning
#       does not apply here.)
#   (8) The client's --metadata gains arm=, launcher_md5= (8 hex), image= and image_id= (12 hex; short prefixes keep the
#       public repo's hygiene check quiet) (recorded in the result JSON, never sent);
#       dataset= and conc= are kept as bench/plot.py reads them. boots.tsv gains the columns arm and readback (the
#       summary reads boots.tsv by column name; the extra columns are inert).
#   (9) R-number, UNIT r811-std-bench-ab, results dir $R811_DATE-r811-std-bench-ab (a same-day re-run gets -HHMM).
# SAME SAMPLES, NO CROSS-CELL CACHE: every cell of both arms sends the identical sample (ShareGPT --seed 7310 --num-prompts
#   400; Spec-Bench all 480, vLLM's fixed shuffle seed 0), in the same order; the summary flags any manifest difference
#   inside an arm and the compare flags one between arms. No prefix cache survives between cells: every cell is a fresh
#   boot (empty GPU prefix cache) with the NVMe tier off (nothing on disk to restore), and the unit's warm-ups are short
#   out-of-set prompts. Cached tokens per cell are still counted (CACHE_MAX 1 %).
# DECISION (pre-registered; the last audit.log line is `DECISION: ...`):
#   VOID <why>   a boot or provenance failure, a launcher md5 change, a clock / power drift, preflight, or a signal.
#   otherwise    `NEW <per-arm decision>; OLD <per-arm decision>`, each arm by R787d's rule (std_bench_summary.py
#                --decide "A B": PUBLISHABLE = both passes complete, no integrity flag, cached <= 1 %, A/B spread
#                <= 3 % on output tok/s at c2-c8 (c1 reported), no foreign request).
#   README: only a PUBLISHABLE NEW arm replaces R787d in the std-bench figure / tables (R787's rule). The NEW/OLD
#   comparison is report-only; it is quoted as a paired measurement only where both arms are PUBLISHABLE, otherwise
#   with the failing checks named.
# GPU TIME: R787d / R731b took 89 min for 16 cells (5.6 min per cell incl. boot); 32 cells ~ 178 min, + drain wait,
#   + the final restore (~3 min) ~ 3 h 05 min. The OLD image's boots use the same TUNEDIR (warm).
# NO UPLOADS: as R787d (the client runs offline with telemetry off, --network host only to reach 127.0.0.1:8022, the
#   preflight runs --network none, the unit never pulls an image).
#
# DEPLOY (operator; .new + mv, never over a running file; the three R787d probes are re-deployed from the repo so flan
#   runs exactly the repo copy, and the unit snapshots every probe into $R/probes and logs its md5):
#   for f in r811-std-bench-ab.sh probes/r811_ab_compare.py probes/test_r811_ab_compare.py probes/vllm_bench_tabby.py \
#            probes/parse_container.py probes/std_bench_summary.py; do
#     ssh flan "cat > /srv/qwen5090/$f.new" < flan/$f && ssh flan "mv /srv/qwen5090/$f.new /srv/qwen5090/$f"; done
# RUN:
#   ssh flan 'sudo systemd-run --unit=r811-std-bench-ab --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=1800 -p Environment=HOME=$HOME /usr/bin/bash /srv/qwen5090/r811-std-bench-ab.sh'
#   Stop: sudo systemctl stop r811-std-bench-ab (the trap ends it VOID and restores the daily from the live launcher).
#   Knobs (-p Environment=...): PASSES ("A B") DATASETS ("sharegpt specbench") CONCS ("1 2 4 8") SG_N SB_N SEED SB_OUT
#   QUIESCE ("hermes hermes-webui owui-proxy"). Changing any of them makes the run not the pre-registered one.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
UNIT=${UNIT:-r811-std-bench-ab}
D=/srv/qwen5090
R811_DATE=${R811_DATE:-$(date +%F)}
R=${R:-$D/results/$R811_DATE-$UNIT}
[ -e "$R/audit.log" ] && R=$R-$(date +%H%M)   # a re-run never mixes into an earlier run's dir
[ -e "$R/audit.log" ] && { echo "ABORT: $R already holds a run"; exit 3; }
mkdir -p "$R/probes" "$R/launchers"
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
MDIR=$D/models/$MODEL
LIVE=$D/launch-flashnext.sh
CLEAN_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# ---------------- the two arms (pinned; the launcher files are the only thing booted) ----------------
ARMS="NEW OLD"
declare -A AL AMD5 AIMG AIMGID AKEYS AREQ
AL[NEW]=$LIVE;                    AMD5[NEW]=6429dfa2035a37dba51bb09651d374e8
AIMG[NEW]=tabbyapi:merge-tok-r1;  AIMGID[NEW]=sha256:ac16920f72cf41864ed7f4151bd18bc4ce593015dc0bc263284a8fde8c111454
AKEYS[NEW]=41;                    AREQ[NEW]="EXL3_GR_MIX_TILED EXL3_PREFILL_MERGE EXL3_STASH_ASYNC"
AL[OLD]=$D/launch-flashnext.sh.pre-r809; AMD5[OLD]=04e347cccea238730f04de85807283e5
AIMG[OLD]=tabbyapi:tokenize-offloop-r2;  AIMGID[OLD]=sha256:04b06fa97c4de4e1b8b8ad4862f4b81044b320154d38d3159e562b3a966bd1ca
AKEYS[OLD]=39;                    AREQ[OLD]="EXL3_GR_MIX_TILED -EXL3_PREFILL_MERGE -EXL3_STASH_ASYNC"
WANT_POOL=901120
WANT_SPLIT="30, 30"
WANT_TUNEDIR=$D/.exl3cache-rebase-dev-r3
# the in-container readback: NEW must print `prefill-merge-r1 1 1`; OLD `absent exllamav3.cache.prefill_merge` (the
# module is not in the image) or `<rev> 0 0` (present, disabled). `absent exllamav3` (or anything else) = a broken image.
RB_PY=$'try:\n    import exllamav3.cache.prefill_merge as p\nexcept ModuleNotFoundError as e:\n    print("absent", e.name)\nelse:\n    print(p.REVISION, int(p.merge_enabled()), int(p.async_enabled()))'
rb_ok(){ case $1 in
  NEW) [ "$2" = "prefill-merge-r1 1 1" ] ;;
  OLD) [ "$2" = "absent exllamav3.cache.prefill_merge" ] || [[ "$2" =~ ^[A-Za-z0-9._-]+\ 0\ 0$ ]] ;;
  *) return 1 ;; esac; }

# ---------------- the benchmark (R787d's, unchanged) ----------------
DS=$D/datasets/std-bench
SG=$DS/ShareGPT_V3_unfiltered_cleaned_split.json; SG_SHA=35f0e213
SB=$DS/spec_bench_question.jsonl;                 SB_SHA=4b6d33e7
CLIENT_IMG=${CLIENT_IMG:-vllm/vllm-openai:v0.30.0}
P=$D/probes
PROBES="vllm_bench_tabby.py parse_container.py std_bench_summary.py r811_ab_compare.py test_r811_ab_compare.py"
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
         $(for x in $PROBES; do echo "$P/$x"; done) "$SG" "$SB" "$MDIR/tokenizer.json" "$MDIR/tokenizer_config.json"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
why=$(launchers_ok) || { log "ABORT (before the lock, nothing touched): $why"; exit 3; }
for a in $ARMS; do
  cp "${AL[$a]}" "$R/launchers/launch-$a.sh"
  l=$R/launchers/launch-$a.sh
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
for x in r811_ab_compare.py std_bench_summary.py parse_container.py vllm_bench_tabby.py; do
  python3 -m py_compile "$R/probes/$x" 2>/dev/null || { log "ABORT: $x does not compile"; exit 3; }; done
mkdir -p "$R/.selftest-tmp"
TMPDIR=$R/.selftest-tmp python3 "$R/probes/test_r811_ab_compare.py" > "$R/selftest-r811_ab_compare.txt" 2>&1 \
  || { log "ABORT: test_r811_ab_compare.py fails: $(tail -3 "$R/selftest-r811_ab_compare.txt" | tr '\n' ' ')"; exit 3; }
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
for a in $ARMS; do
  mkdir -p "$R/$a/results" "$R/$a/cells"
  printf 'tag\tpass\tdataset\tconc\tt_boot\tlauncher_md5\timage\timage_id\tenv_n\tenv_sha\tkeys_n\twindow\tslots\tcache\tpolicy\tpower_limit_w\tgpc_boot\tmem_boot\tgpc_after\tmem_after\tvram_free\tmem_line\tpwr_after\tpwr_line\tarm\treadback\n' > "$R/$a/boots.tsv"
  printf 'tag\tpass\tdataset\tconc\tserial_lo\tserial_hi\twarmups\tcatfile\trc\tcontainer\tn_req\tt0\tt1\n' > "$R/$a/runs.tsv"
  : > "$R/$a/foreign.tsv"   # no header row, as R787d (the end block's awk reads every row as tag<TAB>count)
done
note "R811 std-bench A/B: NEW ${AL[NEW]} (${AMD5[NEW]:0:8}, ${AIMG[NEW]} ${AIMGID[NEW]:7:12}, ${AKEYS[NEW]} keys) vs OLD ${AL[OLD]} (${AMD5[OLD]:0:8}, ${AIMG[OLD]} ${AIMGID[OLD]:7:12}, ${AKEYS[OLD]} keys); passes $PASSES (A: NEW then OLD, B: OLD then NEW); datasets $DATASETS; concs $CONCS; SG_N $SG_N seed $SEED; SB_N $SB_N out $SB_OUT; want core $WANT_GPC / memory $WANT_MEM / power $WANT_PWR W; results $R"

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
report(){ [ "$REPORTED" = 1 ] && return 0; REPORTED=1; local a d fr
  for a in $ARMS; do
    [ "$(tail -n +2 "$R/$a/runs.tsv" 2>/dev/null | wc -l)" -gt 0 ] || { ARMDEC[$a]="no cell ran"; continue; }
    python3 "$R/probes/std_bench_summary.py" --runs "$R/$a/runs.tsv" --results "$R/$a/results" --boots "$R/$a/boots.tsv" \
      --sb-data "$SB" --sb-out "$SB_OUT" --want-gpc "$WANT_GPC" --want-mem "$WANT_MEM" --want-pwr "$WANT_PWR" \
      --cache-max "$CACHE_MAX" --spread-max "$SPREAD_MAX" --spread-exempt "$SPREAD_EXEMPT" --decide "$PASSES" \
      --json "$R/$a/summary.json" > "$R/$a/summary.txt" 2>&1
    # prefixed, so the only unprefixed `DECISION:` line in $R/summary.txt is the unit's own (written by finish)
    { echo; echo "---- arm $a: std_bench_summary.py ----"; sed "s/^/[$a] /" "$R/$a/summary.txt"; } >> "$R/summary.txt"
    sed "s/^/  [$a] /" "$R/$a/summary.txt" >> "$R/audit.log"
    d=$(grep -a '^DECISION: ' "$R/$a/summary.txt" | tail -1 | sed 's/^DECISION: //')
    d=${d:-"VOID the summary printed no decision: $(tail -1 "$R/$a/summary.txt" | cut -c1-200)"}
    fr=$(awk -F'\t' '$2 != "0" {printf "%s%s=%s", (n++ ? ", " : ""), $1, $2}' "$R/$a/foreign.tsv" 2>/dev/null)
    if [ -n "$fr" ]; then log "[$a] foreign traffic: $fr"
      case "$d" in PUBLISHABLE*) d="NOT-PUBLISHABLE foreign (non-benchmark) requests in cells: $fr";; esac; fi
    ARMDEC[$a]=$d; log "[$a] arm decision: $d"; done
  python3 "$R/probes/r811_ab_compare.py" --root "$R" --arms $ARMS --sb-data "$SB" --json "$R/compare.json" \
    > "$R/compare.txt" 2>&1 || log "WARN: r811_ab_compare.py rc $? (compare.txt)"
  { echo; echo "---- paired comparison (probes/r811_ab_compare.py) ----"; cat "$R/compare.txt"; } >> "$R/summary.txt"
  sed 's/^/  /' "$R/compare.txt" | tee -a "$R/audit.log"; }
declare -A ARMDEC
finish(){ [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  trap 'log "signal during finish: ignored"' TERM INT HUP
  sudo docker rm -f "$CLIENT_NAME" >/dev/null 2>&1
  report
  # the last cell may be the OLD arm, and every cell ran tier off: never leave either on :8022. Stop, then restore from
  # the LIVE launcher (NEW, tier on: IMG == DAILY_IMG). finish_restore leaves the GPUs to a queued unit (§12).
  if [ "${BOOTED:-0}" = 1 ] || [ -z "$(served_id)" ]; then
    served_stop; wait_unserved 45
    finish_restore "$LIVE" > "$R/boot-restore.log" 2>&1
  else log "restore: not needed (the daily was never stopped)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  local img imgid tier; img=$(sudo docker inspect -f '{{.Config.Image}}' flashnext 2>/dev/null); imgid=$(sudo docker inspect -f '{{.Image}}' flashnext 2>/dev/null)
  tier=$(sudo docker exec flashnext env 2>/dev/null | grep -c '^EXL3_NVME_TIER=')
  log "after: served $(served_id || echo none); image ${img:-none} ${imgid:7:12}; tier ${tier:-0}; launcher md5s now NEW $(md5f "$LIVE" | cut -c1-8) OLD $(md5f "${AL[OLD]}" | cut -c1-8); core $(gpcoff); memory $(memoff); power $(pwrlim) W"
  if [ -n "$(served_id)" ] && { [ "$img" != "${AIMG[NEW]}" ] || [ "$imgid" != "${AIMGID[NEW]}" ] || [ "${tier:-0}" = 0 ]; }; then
    log "WARN: :8022 is not the production daily (want ${AIMG[NEW]} ${AIMGID[NEW]:7:12}, tier on): restore it by hand from $LIVE"; fi
  [ -n "$(served_id)" ] || [ -n "$(gpu_queue_others)" ] || log "WARN: :8022 not serving after the restore and no unit queued: boot $LIVE by hand"
  local now=; now=$(quiesce); [ -n "$now" ] && log "stopped after the restore: $now (started below if it ran at the start)"
  if [ -n "$WAS_RUNNING" ]; then
    [ -n "$(served_id)" ] || log "WARN: :8022 not serving (another unit queued?); restarting the direct clients anyway"
    if sudo docker start $WAS_RUNNING >/dev/null 2>&1; then log "restarted: $WAS_RUNNING"
    else log "RESTART FAILED: $WAS_RUNNING (start by hand: sudo docker start $WAS_RUNNING)"; fi
  else log "no direct client to restart"; fi
  sudo chown -R "$(stat -c %U $D/results)" "$R" 2>/dev/null || true
  echo "DECISION: $DECISION" >> "$R/summary.txt"
  log "=== $UNIT $1 ==="; log "DECISION: $DECISION"; }
void(){ DECISION="VOID $*"; finish VOID; exit 3; }
trap 'log "signal"; DECISION="VOID signal (the run was interrupted)"; finish ABORTED; exit 4' TERM INT HUP

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
  "$B_PWRLINE" "$B_ARM" "${B_RB:--}" >> "$R/$B_ARM/boots.tsv"; }
declare -A REF
boot(){ local arm=$1 tag=$2 lt=$1/$2 L C bl why= full pv st
  L=${AL[$arm]} C=$R/$arm/cells
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
  B_IMGID=${full:0:19}
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
  B_RB=$(grep -aE '^(absent \S+|\S+ [01] [01])$' "$C/readback-$tag.txt" | tail -1)
  B_RB=${B_RB:-$(tail -1 "$C/readback-$tag.txt" | tr '\t' ' ' | cut -c1-160)}
  log "[$lt] UP: image $B_IMG (${full:7:12}); launcher md5 ${B_MD5:0:8}; keys $B_KEYS (container EXL3_ env $B_ENVN, sha $B_ENVSHA, window $B_WIN); readback '$B_RB'; slots $B_SLOTS cache $B_CACHE policy '$B_POLICY'; power $B_PWR W (launcher: ${B_PWRLINE:-no power line}); core offsets $B_GPC; memory offsets $B_MEM (launcher: ${B_MEMLINE:-no memory-offset line}); VRAM free $B_FREE"
  # the arm's provenance (change (3) of the header); every reason is collected, then the cell is VOID
  pv=$(assert_env_keys "$bl" "${AKEYS[$arm]}" ${AREQ[$arm]} 2>&1) || why="$why env keys: $(echo $pv);"
  [ "$B_IMG" = "${AIMG[$arm]}" ] || why="$why image '$B_IMG' != ${AIMG[$arm]};"
  [ "$full" = "${AIMGID[$arm]}" ] || why="$why image id '${full:7:12}' != pinned ${AIMGID[$arm]:7:12};"
  case $arm in
    NEW) grep -qx 'EXL3_PREFILL_MERGE=1' "$C/env-$tag.txt" && grep -qx 'EXL3_STASH_ASYNC=1' "$C/env-$tag.txt" \
           || why="$why container env lacks EXL3_PREFILL_MERGE=1 / EXL3_STASH_ASYNC=1;" ;;
    OLD) grep -qE '^EXL3_(PREFILL_MERGE|STASH_ASYNC)=' "$C/env-$tag.txt" && why="$why container env carries a merge key;" ;;
  esac
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
# every boot was md5-checked before it ran, so a change now does not touch a measured cell: logged, not a VOID
why=$(launchers_ok) && log "at the end: both launchers unchanged" || log "WARN at the end (after the last boot): $why"

report
DECISION="NEW ${ARMDEC[NEW]:-no decision}; OLD ${ARMDEC[OLD]:-no decision}"
case "${ARMDEC[NEW]:-}" in
  PUBLISHABLE*) note "README: the NEW arm is PUBLISHABLE: $R/NEW/results may replace R787d in the std-bench figure / tables" ;;
  *) note "README: the NEW arm is not PUBLISHABLE: R787d stays in the std-bench figure / tables" ;; esac
finish DONE
