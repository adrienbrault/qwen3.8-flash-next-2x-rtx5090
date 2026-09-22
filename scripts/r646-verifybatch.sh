#!/usr/bin/env bash
# r646-verifybatch.sh — gated A/B for the verifybatch-r1 image (tabbyapi:bverify-r1).
#
# R645 attribution: py-spy showed 43% of c4 wall as a leaf inside receive_sample's
# next_token.cpu() — the serial per-token accept chain. The round-4 batched verifier
# never ran: reqs_past_ids was poisoned by neutral penalty steps before alt()
# simplification (now aggregated over final steps), and device_logit_mask vetoed
# every min_tokens request (now forwarded to the batched sampler).
#
# Pre-registered expectations vs the 2026-09-22 metrics1 gate baseline:
#   * acceptance per verify UNCHANGED (greedy argmax over the same masked logits);
#   * decode t/s UP on drafted shapes — the measured host gap is ~1.4-2.1 ms/iter
#     (r644), i.e. expect roughly +8-15% at c4/c8;
#   * py-spy: job.py:622 leaf share collapses; ready.synchronize() appears.
#
# GPU TIMEBOX ~20 min. Queue-chained; daily restored only when the queue empties.
set -uo pipefail
HOME=${HOME:-/root}
R=/srv/qwen5090/results/2026-09-22-r646-verifybatch
LIVE=/srv/qwen5090/launch-flashnext.sh
IMG=tabbyapi:bverify-r1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
mkdir -p "$R"
LOG="$R/r646.log"
log(){ echo "$(date -Is) $*" | tee -a "$LOG"; }

export GPU_QUEUE_NAME=r646-verifybatch
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh

finish(){ finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R646 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP

gpu_lock
log "lock held; served at entry: $(served_id || echo none)"

env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  IMG="$IMG" bash "$LIVE" > "$R/boot-bverify.log" 2>&1 \
  || { log "NO BOOT: launcher rc=$?"; tail -25 "$R/boot-bverify.log" | tee -a "$LOG"; finish FAILED; exit 1; }
wait_served_id "$MODEL" || { log "NO BOOT: served id mismatch"; finish FAILED; exit 1; }
BOOTED=1
log "booted $IMG, served id OK"

# Mechanism check: profile 20 s under a c4 load — the serial chain should be gone.
log "py-spy mechanism check (20 s under c4)"
PID=$(sudo docker inspect -f '{{.State.Pid}}' flashnext)
python3 /srv/qwen5090/probes/fn_bench.py --url http://127.0.0.1:8022/v1 --model "$MODEL" \
  --kind prose --tokens 2048 --conc 4 --runs 3 --ctx 4096 --salt 424242 \
  --out "$R/pp-instrument.jsonl" > "$R/pp-instrument.log" 2>&1 &
BENCHPID=$!
sleep 8
sudo /srv/qwen5090/bin/py-spy record --pid "$PID" --duration 20 --rate 200 \
  --format speedscope --output "$R/pyspy-c4-bverify.json" --threads --idle 2>&1 | tail -2 | tee -a "$LOG"
wait "$BENCHPID" || true

# Canonical gate on the patched image.
log "canonical gate"
/srv/qwen5090/probes/fn_gate.sh "$R/gate" http://127.0.0.1:8022/v1 "$MODEL" 424242

finish DONE
