#!/usr/bin/env bash
# R676 — slotfix-r1 gate + promotion on the Flash-Next daily (:8022).
# BUG (R586c, 2026-09-22 22:58-23:25 UTC): the recurrent-state pool exhausted — "Cannot create new state:
#   no available slots" (cache.py:358) 2,380x in 27 min, every request 503 while /v1/model answered.
#   Root cause by audit of the served source: all three alloc paths pop the slot handle BEFORE calling
#   recurrent_state_cls(); a throwing constructor (l.clear()/unstash() under VRAM pressure — five graph.cu
#   OOMs preceded the flood — or a restored stash's position assert) orphans the handle, and reap_failed_job
#   cannot recover a state that was never assigned to job.recurrent_state. ~1 throw per 3 min under 12-agent
#   churn drains 8 slots in ~25 min, matching the incident window. slotfix-r1 (patches/exllamav3/slotfix-r1):
#   constructor faults return the handle; release_state refuses double-release; reap_failed_job and cancel()
#   split their cleanups so one failure cannot strand the other; issued/reclaimed counters in the assert.
# GATE. A) fault injection: EXL3_SLOTFIX_TEST_FAULTS=6 + EXL3_SLOTFIX_TEST_FAULT_FILE arms the injector
#   AFTER boot — faults only fire while /tmp/slotfix-arm exists, so the launcher's warmup gen cannot
#   consume (and die on) an injected fault. Expect: exactly 6 "returned slot" warnings, client-visible
#   errors for the affected jobs, then recovery — and ZERO "no available slots". B) on the same boot: c1..c8 ramp, fn_churn 12 workers ~18 min with ~25% mid-stream
#   cancels + >2048-token forced gens (output_chunking => every one requeues through the stash/restore path)
#   + shared-prefix prompts; then a c8 bench proves all 8 slots still admit. C) fn_greedy fingerprint vs the
#   same-session stack-r1 reference: the patch touches no numerical path, so identity is required.
#   Fail-closed counts: "no available slots"=0, "released twice"=0, "returned slot"=6 exactly,
#   container RestartCount=0.
# Promotion (autonomous once the gates pass, user 2026-09-17): DAILY_IMG -> tabbyapi:slotfix-r1 in the live
#   launcher, live boot, generation sanity. Rollback: launch-flashnext.sh.pre-r676.
# GPU TIMEBOX ~50 min. Queue-chained.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-23-r676-slotfix-gate; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r676-slotfix.sh
GREEDY=/srv/qwen5090/probes/fn_greedy.py
CHURN=/srv/qwen5090/probes/fn_churn.py
IMG=tabbyapi:slotfix-r1
FAULTS=6
PROMOTED=0
log(){ echo "$(date -Is) [r676] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=r676-slotfix-gate
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
alive(){ [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
finish(){ sudo docker logs flashnext > "$R/docker-final.log" 2>&1
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R676 $1 ==="; }
rollback(){ log "GATE FAILED ($1): rolling back to .pre-r676"
  [ "$PROMOTED" = 1 ] && sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r676 "$LIVE"; PROMOTED=0; BOOTED=1
  finish "ROLLED-BACK ($1)"; exit 3; }
trap 'log "signal"; [ "$PROMOTED" = 1 ] && sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r676 "$LIVE"; BOOTED=1; finish ABORTED; exit 4' TERM INT HUP

for f in "$LIVE" "$CAND" "$GREEDY" "$CHURN" /srv/qwen5090/probes/fn_bench.py \
         /srv/qwen5090/patches/exllamav3/slotfix-r1/slotfix-r1.patch \
         /srv/qwen5090/patches/exllamav3/slotfix-r1/Dockerfile.box; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
grep -q '^DAILY_IMG=tabbyapi:stack-r1$' "$LIVE" || { log "ABORT: live launcher is not stack-r1"; exit 3; }
[ "$(diff "$LIVE" "$CAND" | grep '^>' | grep -vc '^> #')" = 1 ] || { log "ABORT: candidate must change exactly 1 non-comment line"; exit 3; }
diff "$LIVE" "$CAND" | grep -v '^[<>] #' | tee -a "$R/audit.log"

# Build: Python-only overlay on stack-r1, seconds.
( cd /srv/qwen5090/patches/exllamav3/slotfix-r1 && sudo docker build -f Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 \
  || { log "ABORT: build failed (see $R/build.log)"; exit 3; }
sudo docker run --rm --entrypoint python3 "$IMG" -c \
  'import exllamav3.cache.cache as c; assert c._SLOTFIX_BUILD == "r1"; print("slotfix-r1 landed")' >> "$R/audit.log" 2>&1 \
  || { log "ABORT: $IMG lacks slotfix-r1"; exit 3; }

gpu_lock
log "lock held; served at entry: $(served_id || echo none)"
BOOTED=1
boot(){ local L=$1 tag=$2; shift 2
  served_stop; wait_unserved 45 || return 1
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "$@" bash "$L" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { log "NO BOOT $tag: $(tail -3 "$R/boot-$tag.log" | tr '\n' ' ' | cut -c1-200)"; return 1; }; }
sane(){ generation_state "$MODEL" | grep -q GEN_SANE || { log "generation not sane"; return 1; }; }
ramp(){ local tag=$1 c
  for c in 1 2 3 4 5 6 7 8; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-ramp$c" --kind code --tokens 128 --conc $c \
      --runs 1 --out "$R/ramp.jsonl" > "$R/ramp-$tag-$c.log" 2>&1
    alive || { log "RAMP $tag: server not alive after c$c"; return 1; }
  done
  local oom; oom=$(sudo docker logs flashnext 2>&1 | grep -ac 'graph.cu')
  log "ramp $tag c1..c8 done; free $(vram_free); graph.cu lines $oom"; [ "$oom" = 0 ]; }

slotstats(){ # emits "assert=N returned=N twice=N" from the container log since boot
  local l; l=$(sudo docker logs flashnext 2>&1)
  echo "assert=$(echo "$l" | grep -ac 'no available slots') returned=$(echo "$l" | grep -ac 'returned slot') twice=$(echo "$l" | grep -ac 'released twice')"
}

# ---- reference: the unchanged daily (stack-r1), greedy fingerprint ----
[ "$(served_id)" = "$MODEL" ] && grep -qx "tabbyapi:stack-r1" <(sudo docker ps --format '{{.Image}}' -f name=flashnext) \
  || { boot "$LIVE" ref && sane || { finish ABORTED; exit 3; }; }
python3 "$GREEDY" --url http://127.0.0.1:8022 --tag ref --out "$R/greedy.jsonl" >> "$R/audit.log" 2>&1 \
  || { log "ABORT: greedy reference failed"; finish ABORTED; exit 3; }

# ---- candidate boot with injected constructor faults (armed post-boot via file gate) ----
boot "$CAND" cand "EXTRA_ENV_ADD=EXL3_SLOTFIX_TEST_FAULTS=$FAULTS EXL3_SLOTFIX_TEST_FAULT_FILE=/tmp/slotfix-arm" \
  || { finish NO-BOOT; exit 3; }
envline=$(grep -aoE 'env keys \([0-9]+\): .*' "$R/boot-cand.log" | tail -1)
log "candidate: image $(sudo docker ps --format '{{.Image}}' -f name=flashnext); ${envline%%:*}; $(slotstats); free $(vram_free)"
[[ "$envline" == *"EXL3_SLOTFIX_TEST_FAULTS=6"* ]] && [[ "$envline" == *"EXL3_SLOTFIX_TEST_FAULT_FILE=/tmp/slotfix-arm"* ]] \
  || { log "FAIL: fault env not passed"; finish FAILED; exit 3; }
sudo docker exec flashnext rm -f /tmp/slotfix-arm   # ensure clean state
sudo docker exec flashnext touch /tmp/slotfix-arm # arm the injector: next 6 state constructions raise

# Fault-proof: send 10 sequential short requests; exactly 6 fail on injected faults, then recovery.
python3 - <<'EOF' > "$R/fault-probe.log" 2>&1
import json, time, urllib.request
ok = err = 0
for i in range(10):
    body = {"model": "qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab", "prompt": "Say hi.",
            "max_tokens": 32, "temperature": 0}
    req = urllib.request.Request("http://127.0.0.1:8022/v1/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60).read(); ok += 1
    except Exception as e:
        err += 1; print(f"req{i}: {str(e)[:120]}")
    time.sleep(0.5)
print(f"ok={ok} err={err}")
EOF
sleep 2
ss=$(slotstats); log "fault probe: $(tail -1 $R/fault-probe.log); $ss"
[[ "$ss" == *"assert=0"* ]] && [[ "$ss" == *"returned=$FAULTS"* ]] && [[ "$ss" == *"twice=0"* ]] \
  || { log "FAIL: injected faults did not recover cleanly ($ss)"; finish FAILED; exit 3; }
grep -q "ok=[1-9]" "$R/fault-probe.log" || { log "FAIL: no request succeeded after injected faults"; finish FAILED; exit 3; }
sudo docker exec flashnext rm -f /tmp/slotfix-arm   # disarm; churn must see only real faults
sane || { log "FAIL: generation not sane after injected faults"; finish FAILED; exit 3; }
log "fault injection proven: $FAULTS constructor faults, pool self-healed, zero exhaustion"

# ---- identity + ramp + churn on the same candidate boot ----
python3 "$GREEDY" --url http://127.0.0.1:8022 --tag cand --out "$R/greedy.jsonl" >> "$R/audit.log" 2>&1
python3 "$GREEDY" --compare --ref ref --out "$R/greedy.jsonl" 2>&1 | tee -a "$R/audit.log" | grep -q "divergences=0" \
  || { log "FAIL: greedy differs from stack-r1 — exception-safety patch must be output-identical"; finish FAILED; exit 3; }
ramp cand || { finish FAILED; exit 3; }

log "churn: 12 workers, 18 min, ~25% mid-stream cancels, long gens (requeue/stash/restore) + shared prefix"
python3 "$CHURN" --url "$API" --model "$MODEL" --workers 12 --minutes 18 --cancel-rate 0.25 \
  --out "$R/churn.jsonl" > "$R/churn-summary.log" 2>&1
tail -15 "$R/churn-summary.log" | tee -a "$R/audit.log"
ss=$(slotstats); log "post-churn: $ss; restarts $(sudo docker inspect -f '{{.RestartCount}}' flashnext)"
[[ "$ss" == *"assert=0"* ]] && [[ "$ss" == *"twice=0"* ]] \
  || { log "FAIL: pool inconsistency under churn ($ss)"; finish FAILED; exit 3; }
# Any "returned slot" beyond the 6 injected means a REAL constructor fault fired — noteworthy but not fatal:
# the point of the fix is that the pool now survives it.
retn=${ss#*returned=}; retn=${retn%% *}
[ "$retn" -ge "$FAULTS" ] || { log "FAIL: returned-slot count regressed ($ss)"; finish FAILED; exit 3; }
[ "$retn" -gt "$FAULTS" ] && log "NOTE: $((retn-FAULTS)) non-injected constructor faults recovered during churn — the production trigger reproduced and the fix held"
alive || { log "FAIL: server not alive after churn"; finish FAILED; exit 3; }

# All 8 slots must still admit: a leaked pool shows up here as queued/failed streams.
python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag post-c8 --kind code --tokens 256 --conc 8 \
  --runs 1 --out "$R/post-c8.jsonl" > "$R/post-c8.log" 2>&1
ok8=$(python3 -c "import json;print(sum(1 for l in open('$R/post-c8.jsonl') if json.loads(l).get('ok')))")
[ "$ok8" = 8 ] || { log "FAIL: post-churn c8 admitted $ok8/8 streams ($ss)"; finish FAILED; exit 3; }
log "post-churn c8: 8/8 streams ok — pool intact"

# ---- promote ----
sudo cp -p "$LIVE" "$LIVE.pre-r676"
sudo sed -i 's/^DAILY_IMG=tabbyapi:stack-r1$/DAILY_IMG=tabbyapi:slotfix-r1/' "$LIVE"
PROMOTED=1
log "promoted: live launcher DAILY_IMG=tabbyapi:slotfix-r1; booting the daily"
boot "$LIVE" live && sane || rollback "live boot"
alive || rollback "post-promote alive"
log "PROMOTED slotfix-r1; served $(served_id) on $(sudo docker ps --format '{{.Image}}' -f name=flashnext); $(slotstats)"
finish DONE
