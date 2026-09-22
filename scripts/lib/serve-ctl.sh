# lib/serve-ctl.sh — served-container helpers for Flash-Next experiment units (2026-09-22).
#
# WHY THIS FILE EXISTS. Every r-script re-implemented served_id()/boot()/finish() by copy, and the
# copies drifted in the expensive direction: r641 ran with GPU_QUEUE_NAME=r621-mtp-batched-confirm;
# r407 v1 waited on a `docker ps` name grep that cannot see a restart-loop (a restarting container is
# still listed) and burned 7 min per failed boot; a wrapper restored-or-not against
# `docker inspect .Config` on a STOPPED container, which reports what it was, not what answers
# (GOTCHAS 14). One correct implementation beats forty divergent ones.
#
# Source AFTER lib/gpu-queue.sh (finish_restore consults gpu_queue_others). Everything here is a
# function returning status — the caller decides whether a failure loses an arm or the unit.
#
#   . /srv/qwen5090/lib/gpu-queue.sh
#   . /srv/qwen5090/lib/serve-ctl.sh
#   SCTL_API=http://127.0.0.1:8022/v1 SCTL_NAME=flashnext   # defaults shown
#
# Functions: served_id, container_status, wait_served_id, wait_unserved, served_stop,
#            generation_state, vram_free, cards_loaded, assert_env_keys, finish_restore.

: "${SCTL_API:=http://127.0.0.1:8022/v1}"
: "${SCTL_NAME:=flashnext}"
: "${SCTL_LOG:=/dev/stderr}"

sctl_log(){ echo "$(date -Is) [sctl] $*" >> "$SCTL_LOG"; }

# The id the endpoint actually serves RIGHT NOW, or empty. This is the only restore/boot check that
# cannot lie: the API either answers with the model id or it does not.
served_id(){ curl -s -m 8 "$SCTL_API/model" 2>/dev/null \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null; }

# docker .State.Status — the field that CAN fail. `docker ps` name greps see a restart-looping
# container as present; .State.Status says restarting|exited|dead|created|running|gone.
container_status(){ sudo docker inspect -f '{{.State.Status}}' "$1" 2>/dev/null || echo gone; }

# Poll until /v1/model reports WANT. Fails fast (rc 1) when the container is observed
# restarting/exited/gone AFTER `grace` polls — a restart-loop is a failed boot in ~seconds, not after
# a 7-minute timeout (the r407f lesson). Default 240 polls * ~2 s + curl time ≈ 8-10 min ceiling.
wait_served_id(){ local want=$1 tries=${2:-240} grace=${3:-10} i st
  for i in $(seq "$tries"); do
    [ "$(served_id)" = "$want" ] && return 0
    st=$(container_status "$SCTL_NAME")
    [ "$st" = running ] || [ "$i" -le "$grace" ] || return 1
    sleep 2
  done
  return 1; }

# Poll until nothing answers on the API — the gate between stopping the old serve and booting the
# next, so a new boot never races a draining one.
wait_unserved(){ local tries=${1:-30} i
  for i in $(seq "$tries"); do
    [ -z "$(served_id)" ] && return 0
    sleep 2
  done
  return 1; }

served_stop(){ sudo docker stop -t 60 "$SCTL_NAME" >/dev/null 2>&1; sudo docker rm -f "$SCTL_NAME" >/dev/null 2>&1; }

# Free VRAM per card, MiB, space-separated ("1041 2531").
vram_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' '; }

# Every card must hold real residency: a layer-split boot that dropped a card serves at half the
# machine while looking healthy at the API. Echoes OK or EMPTY:<idx>:<freeMiB>; rc 1 on EMPTY.
# Threshold defaults to 24000 MiB free — an EMPTY 32 GB card shows ~32,000; the daily leaves ~1-2.5 GB.
cards_loaded(){ local i=0 f lim=${1:-24000}
  for f in $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits); do
    [ "$f" -gt "$lim" ] && { echo "EMPTY:$i:$f"; return 1; }
    i=$((i+1))
  done
  echo OK; }

# One greedy probe. A boot that serves but cannot generate loses the arm — that is a verdict about
# the boot, not a measurement of the arm. Prints GEN_SANE | GEN_GARBAGE | GEN_NONE.
# Content OR reasoning counts (a thinking model may spend the whole probe in the reasoning channel).
generation_state(){ local model=$1 out
  out=$(curl -s -m 120 "$SCTL_API/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Name the capital of France. One word.\"}],\"max_tokens\":200,\"temperature\":0}" 2>/dev/null)
  case "$(echo "$out" | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); ch=d.get("choices") or []
    m=(ch[0].get("message") or {}) if ch else {}
    print((m.get("content") or m.get("reasoning_content") or "<empty>").strip()[:200])
except Exception: print("<empty>")' 2>/dev/null)" in
    *Paris*|*PARIS*|*paris*) echo GEN_SANE ;;
    ""|"<empty>"*)          echo GEN_NONE ;;
    *)                      echo GEN_GARBAGE ;;
  esac; }

# Assert the RESOLVED env-keys line in a launcher boot log — count, plus must-have (-X = must NOT
# have) keys. The resolved line is what the container actually got; the caller's EXTRA_ENV is what it
# asked for, and R614 measured the difference at a fictional 74 % pool cut when EXTRA_ENV= silently
# dropped all 22 tuned keys. Usage:
#   assert_env_keys boot-ctl-1.log 22 EXL3_MOE_COOP_V2 -EXL3_MOE_DISTINCT_PROBE
# Prints a reason per violation; rc 1 on any. A boot log with no env-keys line is a violation (the
# launcher that emits it is the only sanctioned path).
assert_env_keys(){ local f=$1 want=$2 line n k rc=0; shift 2
  line=$(grep -aoE 'env keys \([0-9]+\): .*' "$f" 2>/dev/null | tail -1)
  [ -n "$line" ] || { echo "NO-ENV-LINE in $f"; return 1; }
  n=$(echo "$line" | grep -oE '\([0-9]+\)' | tr -d '()')
  [ "$n" = "$want" ] || { echo "COUNT $n != $want"; rc=1; }
  for k in "$@"; do
    case "$k" in
      -*) echo "$line" | grep -qw "${k#-}" && { echo "PRESENT ${k#-}"; rc=1; } ;;
      *)  echo "$line" | grep -qw "$k" || { echo "MISSING $k"; rc=1; } ;;
    esac
  done
  return $rc; }

# Restore the daily IFF this unit actually booted something AND no other live unit is queued — both
# directions of this check have been gotten wrong: restoring over a queued unit costs a down/up the
# chain did not need; NOT restoring because "we never served" leaves :8022 down. Callers set BOOTED=1
# once they mutate the served container. gpu_queue_others comes from lib/gpu-queue.sh; when it is not
# sourced the queue is unknowable and restoring is the availability-safe default.
finish_restore(){ local launcher=${1:-/srv/qwen5090/launch-flashnext.sh} i
  [ "${BOOTED:-0}" = 1 ] || { sctl_log "never booted; nothing to restore"; return 0; }
  if type gpu_queue_others >/dev/null 2>&1 && [ -n "$(gpu_queue_others)" ]; then
    sctl_log "GPU queue continues ($(gpu_queue_others)); leaving the GPUs to the next unit"
    return 0
  fi
  sctl_log "restoring the daily via $launcher"
  served_stop
  env -i HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    bash "$launcher" || sctl_log "RESTORE LAUNCHER FAILED rc=$?"
  for i in $(seq 240); do [ -n "$(served_id)" ] && break; sleep 2; done
  sctl_log "daily: $(served_id || echo '<no answer>')"; }
