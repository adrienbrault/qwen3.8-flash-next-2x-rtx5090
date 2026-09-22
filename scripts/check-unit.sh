#!/usr/bin/env bash
# check-unit.sh — static preflight for GPU experiment units (2026-09-22).
#
# WHY: nine of fourteen scripts in the 2026-09-16 session were retries whose faults a dry check
# catches — each retry cost a served-container bounce (~40 s of :8022 down) plus GPU idle. `bash -n`
# is not that check: an empty file is valid bash (GOTCHAS 12), a stale GPU_QUEUE_NAME parses fine
# (r641 ran as r621), and `EXTRA_ENV=EXL3_TP=1` is syntactically perfect while silently dropping all
# 22 tuned keys (R614). Run this BEFORE queueing a unit; a FAIL means the unit would have bounced the
# daily for nothing.
#
#   usage: check-unit.sh rNNN-thing.sh [more.sh ...]
#
# Every check prints PASS/FAIL/WARN. FAILs sum to the exit code. WARNs are judgment calls, not stops.
set -uo pipefail
rc_total=0
fails=0
say(){ printf '%-10s %s\n' "$1" "$2"; [ "$1" = FAIL ] && fails=$((fails+1)); return 0; }

check_one(){ local f=$1 base n
  base=$(basename "$f" .sh)
  echo "== $f"
  [ -f "$f" ] || { say FAIL "missing file"; return 1; }
  n=$(wc -c < "$f")
  [ "$n" -gt 500 ] && say PASS "nonempty ($n B)" || { say FAIL "$n B — an empty/truncated script passes bash -n too (GOTCHAS 12)"; return 1; }
  bash -n "$f" 2>/dev/null && say PASS "bash -n" || { say FAIL "bash -n: $(bash -n "$f" 2>&1 | head -1)"; return 1; }

  # Executable unit or source-only library? lib/*.sh are sourced, never run: lifecycle checks do not
  # apply. (A lib also documents these traps in comments — content checks below strip comments.)
  local unit=0
  head -1 "$f" | grep -q '^#!' && unit=1

  # GPU-exclusivity: a unit that boots/probes the GPU must register in the queue and hold the lock.
  if [ "$unit" = 1 ]; then
    if grep -vE '^[[:space:]]*#' "$f" | grep -q 'gpu-queue\.sh'; then
      grep -vE '^[[:space:]]*#' "$f" | grep -qE '(^|[;&|[:space:]])gpu_lock([;&|[:space:]]|$)' \
        && say PASS "queue+lock" || say FAIL "sources gpu-queue.sh but never calls gpu_lock"
    else
      say WARN "does not source gpu-queue.sh — ok only for non-GPU desk work"
    fi

    # GPU_QUEUE_NAME must match the filename — the marker IS the queue identity, and a stale copied
    # name makes the queue lie about which unit owns the GPU (r641 registered as r621).
    if grep -vE '^[[:space:]]*#' "$f" | grep -q 'GPU_QUEUE_NAME='; then
      grep -vE '^[[:space:]]*#' "$f" | grep -qE "GPU_QUEUE_NAME=['\"]?${base}['\"]?" \
        && say PASS "GPU_QUEUE_NAME=$base" \
        || say FAIL "GPU_QUEUE_NAME set but not '$base': $(grep -vE '^[[:space:]]*#' "$f" | grep -oE 'GPU_QUEUE_NAME=[^ ;]+' | head -1)"
    fi
  fi

  # A unit that mutates the served container must assert what came up — /v1/model, not docker state.
  if grep -qE 'docker (rm|stop|run)' "$f"; then
    grep -qE 'served_id|/v1/model|wait_served_id' "$f" \
      && say PASS "served-id assert present" \
      || say FAIL "touches the container but never checks /v1/model — a wrong serve looks like success"
  fi

  # docker inspect .Config on a possibly-stopped container is the GOTCHAS 14 trap. (.State.Status is
  # the sanctioned liveness field and does not match this pattern.)
  grep -vE '^[[:space:]]*#' "$f" | grep -qE 'docker inspect.*Config' \
    && say FAIL "docker inspect .Config used — reports a dead container's image, not what serves (GOTCHAS 14)" \
    || say PASS "no .Config liveness inspect"

  # EXTRA_ENV= as a full override must carry the daily's tuned set — the R614 trap. EXTRA_ENV_ADD is
  # the sanctioned way to add one flag; a bare short list is a boot-to-a-different-stack. Matching
  # real assignments only: value must start with EXL3_ (so `EXTRA_ENV=${EXTRA_ENV:-...}` defaulting in
  # the launcher itself is exempt), and #-comment lines are stripped.
  if grep -vE '^[[:space:]]*#' "$f" | grep -qE 'EXTRA_ENV=("|'"'"')?EXL3_'; then
    n2=$(grep -vE '^[[:space:]]*#' "$f" | grep -oE 'EXTRA_ENV="[^"]*"' | head -1 | grep -oc 'EXL3_')
    n2=${n2:-1}
    [ "$n2" -lt 15 ] \
      && say FAIL "EXTRA_ENV= override carries ~$n2 EXL3_ keys (<15): drops the daily's tuned set — use EXTRA_ENV_ADD (R614)" \
      || say PASS "EXTRA_ENV override carries the tuned set (~$n2 keys)"
  fi

  # Forced-length gates below 1024 tokens sit inside the start-of-generation transient that reads
  # ~30 % high (R638 correction, 2026-09-22). Same-shape A/B screening at 256 stays legal; quoting an
  # absolute rate from it does not.
  grep -qE -- '--tokens[ =]([0-9]{1,3})\b' "$f" \
    && say WARN "--tokens <1024 — inside the publish transient; legal for A/B screening, not for quoted rates" \
    || say PASS "no sub-transient token gate"

  # $HOME under set -u died under systemd (r399). If the script references HOME it must default it.
  if grep -q 'set -u' "$f" && grep -q '\$HOME' "$f" && ! grep -qE 'HOME=\$\{HOME[:-]' "$f"; then
    say WARN "set -u + \$HOME with no HOME default — dies under systemd-run env (r399)"
  fi

  # Filename discipline: one R number per experiment; retries edit in place. A second file sharing
  # this R number with a different suffix is the collision that hit r398.
  local rnum others
  rnum=$(echo "$base" | grep -oE '^r[0-9]+' || true)
  if [ -n "$rnum" ]; then
    others=$(ls "$(dirname "$f")"/${rnum}-*.sh 2>/dev/null | grep -v "/$base\.sh$" || true)
    [ -n "$others" ] && say WARN "other ${rnum}-*.sh files exist ($(echo "$others" | tr '\n' ' ')) — retry-in-place, not a new file, unless the design changed"
  fi
}

for f in "$@"; do fails=0; check_one "$f"; rc_total=$((rc_total+fails)); done
exit "$rc_total"
