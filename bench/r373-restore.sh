#!/usr/bin/env bash
# R373 — put the SERVED configuration back, and prove it.
#
# WHY. The chain ended on `tabbyapi:qsa-cid`, the image from before PR #337 was promoted, and the last line of its log
# said so: "served configuration: tabbyapi:qsa-cid". The cause is not the chain; it is that every experiment script
# boots through `L=/srv/qwen5090/launch-flashnext-r340.sh`, a FROZEN snapshot of the launcher taken for the R340
# experiment. Its `IMG` default is whatever was promoted at the moment it was frozen, so a restore written as "call
# the launcher with no override" restores to a historical default rather than to the served configuration. Fixing
# the call sites is a separate commit; this unit is the immediate repair, and it is also the check that the repair
# worked.
#
# IT VERIFIES RATHER THAN ASSERTS. A restore that boots and does not check is how the box ended up on the wrong
# image while every script printed "restoring the served configuration". The fingerprint below is the identity
# recorded in docs/MEASUREMENTS.md for the served configuration; a mismatch fails the unit.
#
# RUN: sudo systemd-run --unit=r373-restore --collect -p User=adrienbrault \
#        bash /srv/qwen5090/r373-restore.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r373-restore; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
LIVE=/srv/qwen5090/launch-flashnext.sh          # the live launcher, NOT the r340 snapshot
EXPECT_IMG=${EXPECT_IMG:-tabbyapi:qsa-cid-pr337}
EXPECT_PRINT=${EXPECT_PRINT:-750e1459e177c47e}
WAIT_FOR=${WAIT_FOR:-r372-chain2}
log(){ echo "$(date -Is) [r373] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=r373-restore
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R373 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

# The live launcher must be a real file: this session shipped three scripts to this host as 0-byte files, and a
# restore that runs an empty script exits 0 and restores nothing.
[ -s "$LIVE" ] && [ "$(wc -c < "$LIVE")" -gt 1000 ] || { log "ABORT: $LIVE is missing or truncated"; finish ABORTED; exit 3; }

if [ -n "$WAIT_FOR" ] && systemctl is-active --quiet "$WAIT_FOR"; then
  log "waiting for $WAIT_FOR to finish (poll 60 s)"
  while systemctl is-active --quiet "$WAIT_FOR"; do sleep 60; done
fi

log "booting the served configuration from the LIVE launcher (no IMG override)"
bash "$LIVE" >> "$R/audit.log" 2>&1 || { log "BOOT FAILED"; finish ABORTED; exit 1; }

IMG=$(sudo docker inspect flashnext --format '{{.Config.Image}}' 2>/dev/null)
log "serving: $IMG"
[ "$IMG" = "$EXPECT_IMG" ] || log "WARNING: expected $EXPECT_IMG — the live launcher's default is not the promoted image"

# Same greedy fingerprint the enablement gate uses, so this asserts BEHAVIOUR and not just a tag.
FILES=$R/greedy; rm -rf "$FILES"; mkdir -p "$FILES"
python3 /srv/qwen5090/probes/hotvocab-greedy-capture.py --url "$API" --model "$MODEL" --out-dir "$FILES" \
  >> "$R/audit.log" 2>&1 || log "capture FAILED"
. /srv/qwen5090/lib/greedy-compare.sh
GOT=$(greedy_hash "$FILES")
log "greedy dirhash: $GOT"
if [ "$GOT" = "$EXPECT_PRINT" ]; then
  log "PASS: the served configuration is byte-identical to the fingerprinted one"
else
  log "CHECK: dirhash differs from the recorded $EXPECT_PRINT — the served configuration is not the fingerprinted one"
fi
log "policy line in the served config: $(sudo grep -c draft_num_tokens_by_batch /srv/qwen5090/flashnext-config.yml 2>/dev/null)"
finish DONE
