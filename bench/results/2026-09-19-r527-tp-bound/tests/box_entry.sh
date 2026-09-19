#!/usr/bin/env bash
# In-container entry point of the TP=2 bounding probe (started by run_box_probe.sh; do not run on the host).
# Mounts: /probe = out/tp-bound-r1 (ro), /model = the 2.50 bpw checkpoint (ro), /results (rw), /tunecache (rw copy).
# Order: environment record -> P2P gate (fatal) -> NCCL all-reduce (2 processes) -> kernel scaling (cuda:0) -> join.
# Hard ceilings (timeout): 90 + 150 + 570 s = 13.5 min; expected about 9 min in total.
set -uo pipefail
P=/probe
R=/results
T0=$(date +%s)
rc=0
say() { echo "[$(( $(date +%s) - T0 ))s] $*"; }

say "environment"
nvidia-smi topo -m > "$R/topo.txt" 2>&1 || true
nvidia-smi --query-gpu=index,name,pstate,clocks.sm,clocks.max.sm,power.limit,power.default_limit,temperature.gpu,memory.used \
  --format=csv > "$R/gpus_before.csv" 2>&1 || true
python3 - > "$R/versions.txt" 2>&1 <<'PY'
import torch, exllamav3, os
print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())
print("exllamav3", getattr(exllamav3, "__version__", "?"), os.path.dirname(exllamav3.__file__))
try:
    print("nccl", torch.cuda.nccl.version())
except Exception as e:
    print("nccl ?", e)
PY
cat "$R/versions.txt"

say "P2P gate + peer-copy all-reduce"
timeout 90 python3 "$P/tests/tp_bound_p2p.py" --out "$R/tp_bound_p2p.json" 2>&1 | tee "$R/p2p.log"
p2p_rc=${PIPESTATUS[0]}
if [ "$p2p_rc" != 0 ]; then
  say "FATAL: P2P gate failed (rc $p2p_rc) -- see p2p.log; the TP budget premise does not hold, stopping"
  exit 3
fi

say "NCCL all-reduce (2 processes)"
NCCL_P2P_LEVEL=SYS NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P NCCL_DEBUG_FILE="$R/nccl.%h.%p.log" \
  timeout 150 python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$P/tests/tp_bound_ar.py" --out "$R/tp_bound_ar.json" 2>&1 | tee "$R/ar.log"
ar_rc=${PIPESTATUS[0]}
[ "$ar_rc" = 0 ] || { say "ERROR: NCCL all-reduce part failed or was not P2P (rc $ar_rc); continuing"; rc=1; }

say "kernel scaling probe (cuda:0)"
# One step per family (GDN ATTN MOE_K2 MOE_K3 MTP HC HEAD), each independent: a failing step is recorded and the rest
# continue in a fresh process (tp_bound_scaling.from_<STEP>.json, merged into tp_bound_scaling.json); internal budget
# 550 s < this 570 s ceiling. ok / errors / steps are written after every step.
ls -l /tunecache > "$R/tunecache_before.txt" 2>&1 || true
sha256sum /tunecache/coop_autotune_v1.bin > "$R/coop_autotune_before.sha256" 2>/dev/null || true
timeout 570 python3 "$P/tests/tp_bound_gpu.py" --model /model --out "$R/tp_bound_scaling.json" 2>&1 | tee "$R/scaling.log"
sc_rc=${PIPESTATUS[0]}
[ "$sc_rc" = 0 ] || { say "ERROR: scaling probe reported problems (rc $sc_rc); see scaling.log"; rc=1; }
ls -l /tunecache > "$R/tunecache_after.txt" 2>&1 || true
sha256sum /tunecache/coop_autotune_v1.bin > "$R/coop_autotune_after.sha256" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,pstate,clocks.sm,clocks.max.sm,power.limit,temperature.gpu --format=csv \
  > "$R/gpus_after.csv" 2>&1 || true

say "budget join"
if [ -f "$R/tp_bound_scaling.json" ]; then
  python3 "$P/tests/tp_bound_join.py" --scaling "$R/tp_bound_scaling.json" \
    --ar "$R/tp_bound_ar.json" "$R/tp_bound_p2p.json" --out "$R/tp_bound_budget.json" 2>&1 | tee "$R/budget.log"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
else
  say "ERROR: no scaling results, no join"; rc=1
fi
say "done rc=$rc"
exit $rc
