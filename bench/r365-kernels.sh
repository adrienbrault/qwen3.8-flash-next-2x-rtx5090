#!/usr/bin/env bash
# R365 — the kernel candidates, built and A/B'd on the box.
#
# Builds and tests the ported upstream kernel PRs, in the order their own assessments ranked them:
#   #290 exl3_mgemm capacity/OOB fix — three arms (unpatched / fix / fix+reduction), all with an identical forced
#        native rebuild, per its Dockerfile. Its assessment says the fix is NOT on the served decode path, so the
#        gate is output identity, not speed: a memory fix must not change what the model says.
#   #246 sm120 persistent route-packed MoE — two arms, feature off/on. Its assessment says the claim is PREFILL-only,
#        so the numbers to read are TTFT at depth, not decode.
#
# BUILD FAILURES ARE THE INTERESTING CASE. Each build asserts its own source state; a failure here means the port does
# not compile or does not apply, which is a codex job, not a reason to skip the candidate. Failures are logged with
# the tail of the build log in a form that can be pasted straight into a stream.
#
# RUN: sudo systemd-run --unit=r365-kernels --collect -p User=adrienbrault -p RuntimeMaxSec=28800 \
#        bash /srv/qwen5090/r365-kernels.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r365-kernels; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
L=/srv/qwen5090/launch-flashnext-r340.sh
KC=/srv/qwen5090/docker
log(){ echo "$(date -Is) [r365] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r365-kernels
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/greedy-compare.sh   # greedy_same(): directory-safe identity checks
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R365 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

greedy(){  # <out-prefix>
  python3 /srv/qwen5090/probes/hotvocab-greedy-capture.py --url "$API" --model "$MODEL" --out-dir "$1" \
    >> "$R/audit.log" 2>&1 || log "  capture FAILED for $1"
  local f; f=$(ls "$1"/* 2>/dev/null | head -1)
  [ -n "$f" ] && log "  $(basename "$1"): sha $(shasum -a 256 "$f" | cut -c1-16) ($(wc -c < "$f") bytes)" \
              || log "  $(basename "$1"): no capture"
}

probes(){  # <tag> [prefill-heavy]
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-narrow" --kind code \
     --tokens 2048 --conc 1 4 8 --runs 1 --out "$R/records-$1.jsonl" 2>&1 | grep -E "^  c=|FAILED" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-deep" --kind code \
     --tokens 1024 --ctx 120000 --conc 4 --runs 1 --out "$R/records-$1.jsonl" 2>&1 | grep -E "^  c=|FAILED" | tee -a "$R/audit.log"
  # Prefill ladder: what #246's assessment says its claim is about. TTFT is the column to read.
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-prefill" --kind code \
     --tokens 64 --ctx 0 30000 240000 --conc 1 --runs 1 --out "$R/records-$1.jsonl" 2>&1 | grep -E "^  c=" | tee -a "$R/audit.log"
}

build(){  # build <dir> <dockerfile> <tag> [extra build-args...]
  local dir=$1 df=$2 tag=$3; shift 3
  log "building $tag"
  if (cd "$dir" && sudo docker build -f "$df" -t "$tag" "$@" . >> "$R/build-$tag.log" 2>&1); then
    log "  BUILD OK $tag"; return 0
  fi
  log "  BUILD FAILED $tag — this is a codex job; tail follows"
  tail -8 "$R/build-$tag.log" | cut -c1-180 | sed 's/^/    /' | tee -a "$R/audit.log"
  return 1
}

boot(){ IMG="$1" EXTRA_ENV="${2:-}" bash "$L" >> "$R/audit.log" 2>&1 || { log "  BOOT FAILED for $1"; return 1; }
        curl -sf -m 8 "$API/model" >/dev/null || { log "  no server after boot of $1"; return 1; }; }

# ================================ #290 exl3_mgemm ==================================================
log "##### #290 exl3_mgemm: arms unpatched / fix / reduction #####"
for ARM in unpatched fix reduction; do
  build "$KC/kernel290" kernel290-Dockerfile "kernel290:$ARM" --build-arg "ARM=$ARM" --build-arg MAX_JOBS=4 \
    || { log "#290 arm $ARM not built; skipping its run"; continue; }
  boot "kernel290:$ARM" || continue
  log "  #290 $ARM: image env — $(sudo docker exec flashnext sh -c 'cat /app/kernel290-arm.txt 2>/dev/null; cat /app/kernel290-binary.txt 2>/dev/null | cut -c1-20' 2>/dev/null | tr '\n' ' ')"
  greedy "$R/greedy-290-$ARM"
  probes "k290-$ARM"
done

log "=== #290 gate: a memory fix must not change output ==="
# Compare a HASH OF THE WHOLE DIRECTORY, not `cmp -s dir/* dir/*`. The glob form takes one file per side, so it works
# only while every capture happens to write exactly one channel file -- the model answers in reasoning_content on
# some prompts and in content on others, and `cmp` with three operands fails on "extra operand" rather than
# comparing. A directory hash is agnostic to how many files a capture produced and to their names.
dirhash(){ find "$1" -type f -printf '%P\n' 2>/dev/null | LC_ALL=C sort | while read -r f; do sha256sum "$1/$f"; done | sha256sum | cut -c1-16; }
for ARM in fix reduction; do
  a=$(dirhash "$R/greedy-290-unpatched"); b=$(dirhash "$R/greedy-290-$ARM")
  if [ -n "$a" ] && [ "$a" = "$b" ]; then
    log "  PASS unpatched == $ARM (byte-identical, dirhash $a)"
  else
    log "  FAIL/NOT-RUN unpatched($a) vs $ARM($b)"
  fi
done

# ================================ #246 sm120 route-packed MoE ======================================
log "##### #246 sm120 route-packed MoE: arms APPLY_KERNEL246=0 / =1 #####"
for APPLY in 0 1; do
  build "$KC/kernel246" kernel246-Dockerfile "kernel246:$APPLY" --build-arg "APPLY_KERNEL246=$APPLY" \
    || { log "#246 apply=$APPLY not built; skipping"; continue; }
done
if sudo docker image inspect kernel246:0 >/dev/null 2>&1 && sudo docker image inspect kernel246:1 >/dev/null 2>&1; then
  log "#246 control arm (unpatched): decode + prefill ladder"
  boot kernel246:0 "" && { greedy "$R/greedy-246-control"; probes "k246-control"; }
  log "#246 patched, feature OFF (default): must match the control"
  boot kernel246:1 "EXL3_MOE_ROUTE_PACKED=0" && { greedy "$R/greedy-246-off"; probes "k246-off"; }
  log "#246 patched, feature ON: the treatment — read TTFT, not decode"
  boot kernel246:1 "EXL3_MOE_ROUTE_PACKED=1" && { greedy "$R/greedy-246-on"; probes "k246-on"; }
  log "=== #246 gates ==="
  greedy_same "$R/greedy-246-control" "$R/greedy-246-off" && log "  PASS control == patched-off" || log "  FAIL/NOT-RUN control vs patched-off"
  greedy_same "$R/greedy-246-control" "$R/greedy-246-on" && log "  PASS control == patched-on" || log "  NOTE patched-on differs (expected only if the kernel reorders numerics; investigate)"
else
  log "#246: one or both images missing; the pair cannot be A/B'd"
fi

log "restoring the enabled configuration"
# No IMG override: the launcher default is the served configuration, and pinning the image here is how a
# restore silently reverts a promotion.
bash "$L" >> "$R/audit.log" 2>&1 || log "RESTORE FAILED"
finish DONE
