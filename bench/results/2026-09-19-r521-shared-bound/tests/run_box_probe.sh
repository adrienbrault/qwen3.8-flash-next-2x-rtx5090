#!/usr/bin/env bash
# Operator wrapper for the shared-expert bound probe (box-ab-spec.md step 2). Run on flan by the operator's unit
# (GPU-exclusive: source /srv/qwen5090/lib/gpu-queue.sh and take the flock as every GPU unit does; the daily must be
# down, the probe loads four real MoE layers). Uses the served image unchanged: nothing is built or installed.
#
#   RESULTS=/srv/qwen5090/results/2026-09-19-rNNN-shared-bound ./run_box_probe.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=${IMG:-tabbyapi:stack-r4-e3r2}
CKPT=${CKPT:-/srv/qwen5090/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab}
TUNEDIR=${TUNEDIR:-/srv/qwen5090/.exl3cache}
RESULTS=${RESULTS:?set RESULTS to the results directory}
mkdir -p "$RESULTS/tunecache"
# a COPY of the daily's tune cache: the probe runs the served geometry and never appends to the served file
cp -a "$TUNEDIR/coop_autotune_v1.bin" "$RESULTS/tunecache/" 2>/dev/null || echo "WARN: no served tune cache at $TUNEDIR"
sha256sum "$RESULTS/tunecache/coop_autotune_v1.bin" 2>/dev/null | tee "$RESULTS/tunecache.sha256" || true
cp -a "$HERE" "$RESULTS/tests"

# the daily's EXTRA_ENV (ref/launch-flashnext-live.sh:206); EXL3_SHARED_EXPERT_OVERLAP is toggled by the probe itself
ENVS=(-e EXL3_HOST_GAP_REWIND=1 -e EXL3_HC_MIX_V2=1 -e EXL3_HC_MIX_V2_MIN_R=1 -e EXL3_LS_PREFILL_PIPELINE=1
      -e EXL3_MOE_COOP_V2=1 -e EXL3_SHARED_EXPERT_OVERLAP=1 -e EXL3_DRAFT_PINNED_STAGING=1 -e EXL3_BATCH_VERIFY=1
      -e EXL3_MTP_HEAD_N=65536 -e EXL3_MOE_PREFILL_E3=1
      -e EXLLAMAV3_TUNE_CACHE=/tunecache -e TRITON_CACHE_DIR=/tunecache)

sudo docker run --rm --gpus all --ipc=host "${ENVS[@]}" \
  -v "$CKPT":/model:ro -v "$RESULTS/tunecache":/tunecache -v "$RESULTS":/results -v "$RESULTS/tests":/t:ro \
  --entrypoint python3 "$IMG" /t/shared_expert_bound_gpu.py --model /model --out /results/shared_expert_bound.json \
  2>&1 | tee "$RESULTS/probe.log"

# geometry the served tuner picked for the shared GEMMs (CPU only; dims printed by the probe's first JSON line)
python3 - "$RESULTS" <<'PY' || true
import json, subprocess, sys
r = sys.argv[1]
d = json.load(open(f"{r}/shared_expert_bound.json"))
L = d["layers"][0]
k_gu, _, k_d = L["shared_K_gate_up_down"]
cb = 2 if L["shared_mul1"] else (1 if L["shared_mcg"] else 0)
cmd = ["python3", f"{r}/tests/autotune_geometry.py", "--cache", f"{r}/tunecache/coop_autotune_v1.bin",
       "--hidden", str(L["hidden"]), "--interm", str(L["shared_interm"]), "--k-gu", str(k_gu), "--k-d", str(k_d),
       "--cb", str(cb), "--rows", "1,2,4,8,16"]
out = subprocess.run(cmd, capture_output=True, text=True)
open(f"{r}/geometry.txt", "w").write(out.stdout + out.stderr)
print(out.stdout)
PY
