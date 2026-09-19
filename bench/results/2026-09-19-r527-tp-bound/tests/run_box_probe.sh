#!/usr/bin/env bash
# Operator wrapper for the TP=2 bounding probe (box-ab-spec.md). Run on the box by the operator's unit: GPU-exclusive
# (source /srv/qwen5090/lib/gpu-queue.sh and take the flock as every GPU unit does), the daily down, both cards free.
# Uses the served image unchanged: nothing is built or installed. ONE docker run; the all-reduce part starts its own
# 2-process launch (torch.distributed.run) inside that container.
#
#   RESULTS=/srv/qwen5090/results/2026-09-19-rNNN-tp-bound out/tp-bound-r1/tests/run_box_probe.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)                 # out/tp-bound-r1 (tests/ + data/)
IMG=${IMG:-tabbyapi:stack-r4-e3r2}
CKPT=${CKPT:-/srv/qwen5090/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab}
TUNEDIR=${TUNEDIR:-/srv/qwen5090/.exl3cache}
RESULTS=${RESULTS:?set RESULTS to the results directory}
mkdir -p "$RESULTS/tunecache"
# a COPY of the daily's kernel caches (coop autotune + Triton): served geometry, and the served files are never written
cp -a "$TUNEDIR"/. "$RESULTS/tunecache/" 2>/dev/null || echo "WARN: could not copy $TUNEDIR"
sha256sum "$RESULTS/tunecache/coop_autotune_v1.bin" 2>/dev/null | tee "$RESULTS/tunecache.sha256" || true
mkdir -p "$RESULTS/probe"
cp -a "$ROOT/tests" "$ROOT/data" "$RESULTS/probe/"
chmod +x "$RESULTS/probe/tests/"*.sh

# the daily EXTRA_ENV: from the environment (the operator unit passes the live launcher's), else
# ref/launch-flashnext-r517.sh:206 verbatim
EXTRA_ENV=${EXTRA_ENV:-"EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1 EXL3_DRAFT_PINNED_STAGING=1 EXL3_BATCH_VERIFY=1 EXL3_MTP_HEAD_N=65536 EXL3_MOE_PREFILL_E3=1"}
echo "EXTRA_ENV: $EXTRA_ENV" | tee "$RESULTS/extra_env.txt"
EV=()
for kv in $EXTRA_ENV; do EV+=(-e "$kv"); done

nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm --format=csv \
  | tee "$RESULTS/host_gpus.csv" || true

set +e
sudo docker run --rm --gpus all --ipc=host --shm-size=16g "${EV[@]}" \
  -e EXLLAMAV3_TUNE_CACHE=/tunecache -e TRITON_CACHE_DIR=/tunecache -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$CKPT":/model:ro -v "$RESULTS/tunecache":/tunecache -v "$RESULTS":/results -v "$RESULTS/probe":/probe:ro \
  --entrypoint bash "$IMG" /probe/tests/box_entry.sh 2>&1 | tee "$RESULTS/run.log"
rc=${PIPESTATUS[0]}
set -e
echo "box_entry rc=$rc (0 = every part ran and passed its checks)"
[ -f "$RESULTS/tp_bound_budget.txt" ] && { echo; cat "$RESULTS/tp_bound_budget.txt"; }
exit "$rc"
