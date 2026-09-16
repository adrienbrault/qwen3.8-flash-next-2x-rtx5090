#!/usr/bin/env bash
# R371 — the #290 identity gate, recomputed from PAIRED captures.
#
# WHY THIS EXISTS. r365 could not answer its own question. Its #290 arms are built and unchanged, but the control
# arm's greedy capture holds ONE prompt's text while the treatment arms hold four: the control ran before
# hotvocab-greedy-capture.py was fixed, and that bug raised AttributeError on `"usage": null` AFTER writing prompt
# 0 and before reaching prompts 1-3. The gate then did `cmp -s dir/* dir/*`, which compares one file per side and
# refuses three operands, so it logged "FAIL/NOT-RUN" for both arms -- a false negative that says nothing about the
# patch, and a false negative is as expensive as a false positive here because it silently discards a candidate.
#
# This re-boots the three images with the fixed probe and compares captures that were taken under identical
# conditions, using greedy_same() from lib/greedy-compare.sh (a hash over sorted file names AND contents, so it is
# agnostic to how many channel files a prompt produced, and an absent capture can never pass).
#
# THE IMAGES ARE NOT REBUILT. Nothing about the binaries changed; only the capture tool did.
#
# RUN: sudo systemd-run --unit=r371-290-identity --collect -p User=adrienbrault -p RuntimeMaxSec=7200 \
#        bash /srv/qwen5090/r371-290-identity.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r371-290-identity; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
L=/srv/qwen5090/launch-flashnext-r340.sh
log(){ echo "$(date -Is) [r371] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=r371-290-identity
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/greedy-compare.sh
gpu_lock
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R371 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

for ARM in unpatched fix reduction; do
  sudo docker image inspect "kernel290:$ARM" >/dev/null 2>&1 || { log "MISSING image kernel290:$ARM — rebuild r365 first"; continue; }
done

# REBUILD the arm whose build failed on container storage rather than on code. r365's `fix` build died with
#   failed to export layer: CreateDiff: mount callback failed on /var/lib/containerd/tmpmounts/containerd-mount...
# while two other builds and a probe run were in flight. Checked before touching anything: the filesystem had 319 GB
# free (82% used) and the tmpmounts named in the error were LIVE mounts of the concurrent build, so this is daemon
# contention, not a leak and not a code error. Retried once here; if it fails again the arm stays missing and the
# gate says so instead of quietly comparing two arms.
if ! sudo docker image inspect kernel290:fix >/dev/null 2>&1; then
  log "=== rebuilding kernel290:fix (previous attempt failed on container storage) ==="
  if (cd /srv/qwen5090/docker/kernel290 && sudo docker build -f kernel290-Dockerfile -t kernel290:fix \
        --build-arg ARM=fix --build-arg MAX_JOBS=4 . >> "$R/build-kernel290-fix.log" 2>&1); then
    log "  BUILD OK kernel290:fix"
  else
    log "  BUILD FAILED again; tail follows:"
    tail -6 "$R/build-kernel290-fix.log" 2>/dev/null | cut -c1-170 | tee -a "$R/audit.log"
  fi
fi

log "=== arm binaries: three arms are only three arms if their extensions differ ==="
declare -A BIN
for ARM in unpatched fix reduction; do
  sudo docker image inspect "kernel290:$ARM" >/dev/null 2>&1 || { log "  $ARM: image missing"; continue; }
  BIN[$ARM]=$(sudo docker run --rm --entrypoint sh "kernel290:$ARM" -c 'cat /app/kernel290-binary.txt 2>/dev/null | cut -d" " -f1' 2>/dev/null)
  log "  $ARM: $(sudo docker run --rm --entrypoint sh "kernel290:$ARM" -c 'cat /app/kernel290-arm.txt 2>/dev/null' 2>/dev/null) binary ${BIN[$ARM]:0:16}"
done
if [ -n "${BIN[unpatched]:-}" ] && [ -n "${BIN[reduction]:-}" ] && [ "${BIN[unpatched]}" = "${BIN[reduction]}" ]; then
  log "  WARNING: unpatched and reduction carry the SAME extension — the arms are not distinguishable, do not read the gate"
fi

log "=== capturing greedy output from each built arm, with the fixed probe ==="
for ARM in unpatched fix reduction; do
  log "--- $ARM"
  if ! IMG="kernel290:$ARM" bash "$L" >> "$R/audit.log" 2>&1; then log "  boot FAILED"; continue; fi
  curl -sf -m 8 "$API/model" >/dev/null || { log "  no server"; continue; }
  # The binary identity travels in the image; record it so a same-binary mistake is visible.
  log "  image env — $(sudo docker exec flashnext sh -c 'cat /app/kernel290-arm.txt 2>/dev/null; cat /app/kernel290-binary.txt 2>/dev/null | cut -c1-20' 2>/dev/null | tr '\n' ' ')"
  python3 /srv/qwen5090/probes/hotvocab-greedy-capture.py --url "$API" --model "$MODEL" --out-dir "$R/greedy-$ARM" \
    >> "$R/audit.log" 2>&1 || log "  capture FAILED"
  n=$(find "$R/greedy-$ARM" -type f ! -name '*.json' 2>/dev/null | wc -l)
  log "  captured $n text files, dirhash $(greedy_hash "$R/greedy-$ARM")"
done

log "=== gate: a memory fix must not change what the model says ==="
for ARM in fix reduction; do
  if greedy_same "$R/greedy-unpatched" "$R/greedy-$ARM"; then
    log "  PASS unpatched == $ARM (byte-identical, dirhash $(greedy_hash "$R/greedy-unpatched"))"
  else
    log "  FAIL unpatched($(greedy_hash "$R/greedy-unpatched")) != $ARM($(greedy_hash "$R/greedy-$ARM"))"
    log "  first difference:"
    diff -r "$R/greedy-unpatched" "$R/greedy-$ARM" 2>&1 | head -5 | tee -a "$R/audit.log"
  fi
done

log "=== restoring the served configuration (launcher default, no override) ==="
bash "$L" >> "$R/audit.log" 2>&1 || log "RESTORE FAILED"
finish DONE
