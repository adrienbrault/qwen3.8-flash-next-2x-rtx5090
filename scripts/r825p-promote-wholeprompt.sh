#!/usr/bin/env bash
# R825p (2026-10-01): operator-only promotion; no build/pull. Based on successful R823p.
# Candidate tabbyapi:r825c-hostprepare FROM pinned R825b 01c967e088a3 (R823c inherited).
# Operator fills prefill-throughput/r825c/IMAGE_ID.env after building; unset fails closed.
# EXACT launcher delta: header, image pin guard/check, DAILY_IMG, append WHOLE_PROMPT=1 and RESUMABLE=1 (46 keys).
# Full real container environment must equal archived daily env plus those two keys.
# Rollback .pre-r825 = 8b644c17e60049a8069fc901b6f091fa; queue-aware boot, signal-safe.
# Inherits R823p gates/order: FASTWARM; free >=1067/1635 MiB; c1<=9.95ms;
# R809 greedy/chat 6+6 exact; needles10/10 at ACTUAL105680/193464; T32 edit57344/HIT.
# BEFORE swap: measure all four scenarios on OLD daily, identical prompt artifact reused AFTER.
# Cold unique first pages: ~50k solo; decode-first + ~90k; /health + ~90k; +1s short (~200/16) + ~90k.
# AFTER swap: identical prompts, cached0, whole cold50 window from trace/source,
# engine Rich log <=0.95x old (conservative hundredth rounding); longest SSE gap <=1.5x old.
# SSE timestamps are delivery observations, grouped MTP tokens counted via real logprobs.
# A peer already active exercises the two-chunk cap; late arrivals to a solo window
# are measured at +1s (short TTFT <= old). /health GET every 200ms, socket timeout 3s;
# candidate: zero timeouts/errors, longest latency <=1s. Old health is evidence, NOT a gate.
# No synthetic gate values. See REVIEW-R825b.md for timer/uvloop and HTTP distinctions.
# INSTALL: upload this unit, launch-flashnext-tabby.sh.r825p.new, probes/{r825p_promote,
# test_r825p_promote,r823_reuse,r823_fixtures,fn_bench,fn_greedy,chat_greedy,fn_needle_oai}.py,
# probes/r825p-fixtures/ (ALL files), patches/exllamav3/r823-cachetrace/fixtures/,
# lib/{gpu-queue,serve-ctl,gateway-drain}.sh to matching /srv/qwen5090 paths.
# Also deploy prefill-throughput/r825c/IMAGE_ID.env with the operator-recorded build ID.
# Do not overwrite live launch-flashnext.sh during artifact installation.
# RUN: sudo systemd-run --unit=r825p-promote-wholeprompt --collect
#   -p RuntimeMaxSec=18000 -p TimeoutStopSec=600
#   /usr/bin/bash /srv/qwen5090/r825p-promote-wholeprompt.sh
# Manual rollback: cp launch-flashnext.sh.pre-r825 launch-flashnext.sh.new &&
#   mv launch-flashnext.sh.new launch-flashnext.sh && bash launch-flashnext.sh
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
UNIT=r825p-promote-wholeprompt
D=/srv/qwen5090
R=$(mktemp -d "$D/results/$(date +%F)-$UNIT-XXXXXX") || exit 3
LIVE=$D/launch-flashnext.sh
TPL=$D/launch-flashnext-tabby.sh.r825p.new
PRE=$D/launch-flashnext.sh.pre-r825
EXPECT_OLD=8b644c17e60049a8069fc901b6f091fa
EXPECT_TPL=262e9c31f714724409635fbac9df8ac2
IMG=tabbyapi:r825c-hostprepare
IMG_ID=$(sed -n 's/^R825C_IMAGE_ID=//p' "$D/prefill-throughput/r825c/IMAGE_ID.env" 2>/dev/null)
[[ "$IMG_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo 'ABORT: fill R825c IMAGE_ID.env after building' >&2; exit 3; }
OLD_IMG=tabbyapi:r823c-cachetail-inforward
OLD_ID=sha256:f5a3c35e2e47647f2200ff33ca822405036835f0729544fad11bad7c56c226b7
NKEYS=46 POOL=901120 REF_UP=1099/1667                              # REF_UP = R818 WARM boot VRAM free (user-accepted)
FN_MAX=9.95                                                        # ms/step, fn_bench c1 code median
EXP_PM="prefill-merge-r1 1 1"
EXP_RB="tokenize-offloop-r2 1 1 1 12000 0 1"
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
BASEURL=http://127.0.0.1:8022
API=$BASEURL/v1
REFD=$D/results/2026-09-29-r809p-promote-merge-2000 REF_TAG=R809 TAG=R825P
P=$D/probes
declare -A PMD5
PMD5[fn_bench.py]=b457fa447eee9919b28dfd088c96bc9c
PMD5[fn_greedy.py]=d24671eb40a9ced182d396f17648b47b
PMD5[chat_greedy.py]=1258992524a00843d7976be333315510
PMD5[fn_needle_oai.py]=027d57ac9b99e229028ac0988c7a1a1d
PMD5[r823_fixtures.py]=cf01a975bcb52268544adc3723dc75a7
PMD5[r823_reuse.py]=44527918d9554452d2c73157e0ba9f2b
PMD5[r825p_promote.py]=f71ef273c49ad710b7347b98b6d89650
PMD5[test_r825p_promote.py]=b6530c0cedeacc5c20c1deed354a854b
PMD5[r825p-fixtures/live-container-env.txt]=a365a02c038c9454f8a903c0a40d04ee
PMD5[r825p-fixtures/needles.jsonl]=59172cb2b38194b281eb08aa7bad1b26
PMD5[r825p-fixtures/provenance.json]=728db30ae92f7ed248f7262eac290f08
PMD5[r825p-fixtures/seed.sse]=7ebed6e193f57094ae420aa7fa97cee9
PMD5[r825p-fixtures/smoke-client.jsonl]=53c7c4df4f25c2ed1f9f07a5fae6ff39
PMD5[r825p-fixtures/smoke-container.log]=11b897f1aa268653d3c21789518a8828
CLEAN_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
QUIESCE="hermes hermes-webui owui-proxy"
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
note(){ echo "$(date -Is) [$UNIT] $*" >> "$R/audit.log"; }          # inside verify(): its stdout is the verdict
md5of(){ md5sum < "$1" | cut -c1-32; }
iid(){ sudo docker image inspect "$1" --format '{{.Id}}' 2>/dev/null; }
envline(){ sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$1" | head -1; }
dimg(){ grep -oE '^DAILY_IMG=\S+' "$1" | cut -d= -f2; }
running(){ [ "$(sudo docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }
cp "$0" "$R/" || exit 3

pair_ok(){
  python3 "$P/r825p_promote.py" pair "$LIVE" "$TPL" > "$R/launcher-pair.txt" 2>&1 \
    || { echo "launcher pair invalid (launcher-pair.txt)"; return; }
  diff "$LIVE" "$TPL" > "$R/launcher-diff.txt" || true
}
preflight(){ local x why
  [ "$(md5of "$LIVE")" = "$EXPECT_OLD" ] || { echo "live launcher md5 $(md5of "$LIVE") != $EXPECT_OLD"; return; }
  [ "$(md5of "$TPL")" = "$EXPECT_TPL" ] || { echo "template md5 $(md5of "$TPL") != $EXPECT_TPL"; return; }
  bash -n "$TPL" || { echo "template does not parse"; return; }
  why=$(pair_ok); [ -z "$why" ] || { echo "$why"; return; }
  [ "$(envline "$TPL" | wc -w | tr -dc 0-9)" = "$NKEYS" ] || { echo "template EXTRA_ENV is not $NKEYS keys"; return; }
  [ "$(dimg "$LIVE")" = "$OLD_IMG" ] && [ "$(dimg "$TPL")" = "$IMG" ] || { echo "DAILY_IMG live '$(dimg "$LIVE")' / template '$(dimg "$TPL")', want $IMG"; return; }
  [ "$(iid "$IMG")" = "$IMG_ID" ] || { echo "$IMG is $(iid "$IMG"), not the gated $IMG_ID"; return; }
  [ "$(iid "$OLD_IMG")" = "$OLD_ID" ] || { echo "rollback image differs from served R823c"; return; }
  for x in "${!PMD5[@]}"; do [ "$(md5of "$P/$x" 2>/dev/null)" = "${PMD5[$x]}" ] || { echo "$P/$x md5 != ${PMD5[$x]}"; return; }; done
  [ -s "$REFD/greedy.jsonl" ] && [ -s "$REFD/chat-greedy.jsonl" ] || { echo "missing $REFD/greedy.jsonl or chat-greedy.jsonl"; return; }
}
FIX=$D/patches/exllamav3/r823-cachetrace/fixtures
for f in "$D/lib/gpu-queue.sh" "$D/lib/serve-ctl.sh" "$D/lib/gateway-drain.sh" "$FIX/SHA256SUMS"; do
  [ -f "$f" ] || { log "ABORT: missing $f"; exit 3; }
done
why=$(preflight); [ -z "$why" ] || { log "ABORT before the lock: $why"; exit 3; }
# the R809 reference rows (6 fn_greedy pids + 6 chat_greedy --long pids); each run appends its rows to a copy
python3 - "$REFD/greedy.jsonl" "$REFD/chat-greedy.jsonl" "$REF_TAG" "$R/ref-fn.jsonl" "$R/ref-chat.jsonl" <<'PY' > "$R/ref.log" 2>&1 \
  || { log "ABORT before the lock: the $REF_TAG greedy reference is incomplete: $(tail -1 "$R/ref.log")"; exit 3; }
import json, sys
fn, chat, tag, ofn, ochat = sys.argv[1:6]
want = {ofn: (fn, ["short0", "short1", "short2", "short3", "short4", "long100k"]),
        ochat: (chat, ["code", "math", "prose", "tool", "multi", "long"])}
for out, (src, pids) in want.items():
    rows = [json.loads(l) for l in open(src) if l.strip()]
    rows = [r for r in rows if r.get("tag") == tag]
    got = sorted(r["pid"] for r in rows)
    if got != sorted(pids):
        sys.exit(f"{src}: tag {tag} pids {got}, want {sorted(pids)}")
    open(out, "w").write("".join(json.dumps(r) + "\n" for r in rows))
print(f"reference {tag}: 6 + 6 rows")
PY
# snapshot the probes: an edit while the unit is queued must not change what this run measures
mkdir -p "$R/probes"; for x in "${!PMD5[@]}"; do mkdir -p "$(dirname "$R/probes/$x")"; cp "$P/$x" "$R/probes/$x" || exit 3; done
for x in "${!PMD5[@]}"; do [ "$(md5of "$R/probes/$x")" = "${PMD5[$x]}" ] || { log "ABORT before the lock: the $x snapshot is not the pinned file"; exit 3; }; done
cp -R "$FIX" "$R/fixtures" || { log 'ABORT: fixture snapshot failed'; exit 3; }
python3 "$R/probes/r823_reuse.py" preflight --fixtures "$R/fixtures" > "$R/fixtures-preflight.txt" 2>&1 \
  || { log 'ABORT: immutable fixture check failed'; exit 3; }
cp "$TPL" "$R/template.sh" && cp "$LIVE" "$R/rollback-base.sh" || exit 3
sudo docker run --rm --network none --entrypoint cat "$IMG_ID" \
  /opt/venv/lib/python3.12/site-packages/exllamav3/cache/checkpoint_policy.py > "$R/image-policy.py" \
  || { log 'ABORT: CPU policy export failed'; exit 3; }
R825P_FIXTURES="$R/fixtures" R825P_POLICY="$R/image-policy.py" R825P_BASE="$R/rollback-base.sh" \
  R825P_TEMPLATE="$R/template.sh" R825P_UNIT="$R/$(basename "$0")" R825P_IMAGE_ID_FILE="$D/prefill-throughput/r825c/IMAGE_ID.env" \
  python3 -B "$R/probes/test_r825p_promote.py" > "$R/selftest.txt" 2>&1 \
  || { log 'ABORT: CPU self-tests failed'; exit 3; }
log "preflight OK: $EXPECT_OLD -> $EXPECT_TPL; image ${IMG_ID:7:12}, NVMe off, trace on; reference $REF_TAG"

export GPU_QUEUE_NAME=$UNIT
. "$D/lib/gpu-queue.sh"
. "$D/lib/serve-ctl.sh"
. "$D/lib/gateway-drain.sh"
SCTL_LOG="$R/audit.log"
trap 'rm -f "${GPU_QUEUE_MARK:-/nonexistent}"' EXIT
# A queued predecessor may defer the daily. If this unit is cancelled while
# queued and becomes the last owner, restore the archived old daily under the lock.
queued_signal(){
  trap '' TERM INT HUP
  local rc=4
  exec 9>"$D/gpu-exclusive.lock"
  if flock -n 9 && [ -z "$(served_id)" ] && [ -z "$(gpu_queue_others)" ]; then
    if [ "$(md5of "$LIVE")" = "$EXPECT_OLD" ] &&
       env -i HOME="$HOME" PATH="$CLEAN_PATH" timeout -k 30 600 bash "$R/rollback-base.sh" > "$R/boot-queue-signal.log" 2>&1 &&
       [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.Image}}' flashnext)" = "$OLD_ID" ]; then
      log 'queued signal: last owner restored old daily'
    else log 'queued signal: old daily restore failed'; rc=5; fi
  fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  exit "$rc"
}
trap queued_signal TERM INT HUP
gpu_lock
why=$(preflight); [ -z "$why" ] || { log "ABORT after lock (nothing mutated): $why"; exit 3; }
for x in "${!PMD5[@]}"; do
  [ "$(md5of "$R/probes/$x")" = "${PMD5[$x]}" ] || { log "ABORT: archived probe changed: $x"; exit 3; }
done
[ "$(md5of "$R/template.sh")" = "$EXPECT_TPL" ] || exit 3
STOPPED= MUTATED=0 COMMITTED=0 CLEANED=0 BASELINE_BOOTED=0 FAILURE='unexpected exit'
# R825p lifecycle begin: archive and rollback are mandatory even on a signal/unexpected exit.
rollback(){
  log "PROMOTION FAILED: $FAILURE -> rolling back to $PRE"
  sudo docker logs --timestamps flashnext > "$R/container-failed.log" 2>&1 || true
  if [ "$(md5of "$PRE")" != "$EXPECT_OLD" ]; then
    log 'ROLLBACK FAILED: backup identity changed'; return 1
  fi
  cp -p "$PRE" "$LIVE.new" && mv "$LIVE.new" "$LIVE" || { log 'ROLLBACK FAILED: install'; return 1; }
  echo "ROLLED BACK: $FAILURE" > "$R/decision.txt" || return 1
  # Restore files immediately; defer the old daily boot if a GPU unit is queued (OPERATIONS §12).
  served_stop || true
  if [ -n "$(gpu_queue_others)" ]; then
    log 'ROLLED BACK: launcher restored; daily boot deferred to next queued unit'
  elif boot rollback && [ "$(sudo docker inspect -f '{{.Image}}' flashnext)" = "$OLD_ID" ]; then
    log "ROLLED BACK: $EXPECT_OLD serving $MODEL, image $OLD_ID"
  else
    log 'ROLLBACK BOOT FAILED: see boot-rollback.log'
    echo "ROLLBACK BOOT FAILED: $FAILURE" > "$R/decision.txt"
    return 1
  fi
}
cleanup(){ local rc=$1 c
  [ "$CLEANED" = 0 ] || return "$rc"
  CLEANED=1
  trap '' TERM INT HUP
  if [ "$MUTATED" = 1 ] && [ "$COMMITTED" = 0 ]; then rollback || rc=5; fi
  if [ "$MUTATED" = 0 ] && [ "${BASELINE_BOOTED:-0}" = 1 ] &&
     [ -z "$(served_id)" ] && [ -z "$(gpu_queue_others)" ]; then
    # A failed baseline boot must not strand the daily when we are last in queue.
    boot rollback && [ "$(sudo docker inspect -f '{{.Image}}' flashnext)" = "$OLD_ID" ] || rc=5
  fi
  for c in $STOPPED; do
    sudo docker start "$c" >/dev/null 2>&1 || { log "client restart failed: $c"; rc=5; }
  done
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  log "restarted:${STOPPED:- none}; === $UNIT DONE (rc=$rc) ==="
  return "$rc"
}
# R825p lifecycle end
trap 'rc=$?; trap - EXIT; cleanup "$rc"; exit $?' EXIT
trap 'FAILURE=signal; exit 4' TERM INT HUP
gateway_drain || { FAILURE='gateway drain failed'; exit 3; }
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || { FAILURE='gateway idle timeout'; exit 3; }
for c in $QUIESCE; do
  if running "$c"; then
    STOPPED="$STOPPED $c"
    sudo docker stop -t 30 "$c" >/dev/null 2>&1 || { FAILURE="cannot quiesce $c"; exit 3; }
  fi
done
log "lock held; served at entry: $(served_id || echo none); clients stopped:${STOPPED:- none}"

boot(){
  served_stop || true
  wait_unserved 45 || return 1
  env -i HOME="$HOME" PATH="$CLEAN_PATH" bash "$LIVE" > "$R/boot-$1.log" 2>&1 || return 1
  wait_served_id "$MODEL" 240 10
}
frontend(){ local phase=$1 since
  since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  timeout 1200 python3 "$R/probes/r825p_promote.py" frontend-client "$API" "$R/frontend-prompts.json" "$phase" \
    "$R/frontend-$phase-client.json" > "$R/frontend-$phase-client.txt" 2>&1 || return 1
  sudo docker logs --timestamps --since "$since" flashnext > "$R/frontend-$phase-container.log" 2>&1 || return 1
  python3 "$R/probes/r825p_promote.py" frontend-check "$R/frontend-prompts.json" "$R/frontend-$phase-client.json" \
    "$R/frontend-$phase-container.log" "$phase" "$R/frontend-$phase.json" \
    > "$R/frontend-$phase-check.txt" 2>&1 || return 1
}
verify(){ local B=$R/boot-promote.log rb pm fw up f0 f1 r0 r1 n ms gf=$R/greedy-fn.jsonl gc=$R/chat-greedy.jsonl
  [ "$(served_id)" = "$MODEL" ] || { echo "not serving $MODEL"; return; }
  [ "$(sudo docker inspect -f '{{.Image}}' flashnext)" = "$IMG_ID" ] || { echo "container image is not $IMG_ID"; return; }
  assert_env_keys "$B" $NKEYS >/dev/null || { echo "env keys != $NKEYS: $(grep -aoE 'env keys \([0-9]+\)' "$B" | tail -1)"; return; }
  grep -aq "cache $POOL @" "$B" || { echo "pool is not $POOL: $(grep -aoE 'cache [0-9]+ @ [0-9,]+' "$B" | tail -1)"; return; }
  sudo docker inspect flashnext > "$R/inspect-promote.json" || { echo 'inspect failed'; return; }
  sudo docker exec flashnext cat /app/config.yml > "$R/config-promote.yml" || { echo 'config readback failed'; return; }
  cmp "$R/config-old.yml" "$R/config-promote.yml" >/dev/null || { echo 'served config differs from live daily'; return; }
  python3 "$R/probes/r825p_promote.py" env "$R/template.sh" "$R/inspect-promote.json" "$R/config-promote.yml" > "$R/env-readback.txt" 2>&1 \
    || { echo 'resolved whole-prompt selectors/trace/NVMe/config mismatch (env-readback.txt)'; return; }
  sudo docker exec -e CUDA_VISIBLE_DEVICES= flashnext python3 /opt/r825c/landing_r825c.py > "$R/landing-promote.txt" 2>&1 \
    || { echo 'served source landing failed'; return; }
  grep -q '^R825c landing PASS: inherited stack and host preparation delta imported and byte verified$' "$R/landing-promote.txt" \
    || { echo 'missing source landing PASS'; return; }
  sudo docker exec flashnext cat /opt/venv/lib/python3.12/site-packages/exllamav3/cache/checkpoint_policy.py > "$R/served-policy.py" \
    || { echo 'served policy export failed'; return; }
  cmp "$R/image-policy.py" "$R/served-policy.py" >/dev/null || { echo 'served policy differs from pinned image'; return; }
  sudo docker logs --timestamps flashnext > "$R/container-boot.log" 2>&1 || { echo 'boot log capture failed'; return; }
  grep -q 'R823 checkpoint policy: near=2048 pp=4096 chunk=2048 source=env' "$R/container-boot.log" &&
    grep -q 'R823b checkpoint tail: tokens=12288 pp=4096 outside=32768' "$R/container-boot.log" \
    || { echo 'D policy startup readback failed'; return; }
  rb=$(sudo docker exec -e CUDA_VISIBLE_DEVICES= -w /app flashnext python3 -c 'import sys; sys.path.insert(0, "/app"); import common.tokenize_offloop as T, exllamav3.tokenizer.tokenizer as E; print(T.REVISION, int(T.encode_once_enabled()), int(T.offloop_enabled()), int(E.tokenize_offloop_enabled()), T.offload_min_chars(), int(T.should_offload(12000)), int(T.should_offload(12001)))' 2>&1 | tail -1)
  [ "$rb" = "$EXP_RB" ] || { echo "readback '$rb' != '$EXP_RB'"; return; }
  pm=$(sudo docker exec -e CUDA_VISIBLE_DEVICES= flashnext python3 -c 'import exllamav3.cache.prefill_merge as p; print(p.REVISION, int(p.merge_enabled()), int(p.async_enabled()))' 2>&1 | tail -1)
  [ "$pm" = "$EXP_PM" ] || { echo "prefill-merge readback '$pm' != '$EXP_PM'"; return; }
  fw=$(grep -aoE 'FASTWARM (ok|FAILED).*' "$B" | tail -1 | cut -c1-240)
  [ "${fw:0:11}" = "FASTWARM ok" ] || { echo "launcher warm-up not ok: ${fw:-no FASTWARM line}"; return; }
  note "verify: $fw"
  up=$(grep -aoE 'VRAM free MiB [0-9]+/[0-9]+' "$B" | tail -1 | grep -oE '[0-9]+/[0-9]+'); f0=${up%/*}; f1=${up#*/}; r0=${REF_UP%/*}; r1=${REF_UP#*/}
  [ -n "$up" ] && [ "$f0" -ge $((r0 - 32)) ] && [ "$f1" -ge $((r1 - 32)) ] || { echo "boot VRAM free '$up' below $REF_UP - 32 MiB"; return; }
  # fn_bench c1 code (R813's call) BEFORE the greedy probes: fn_greedy's short prompts would switch a slow process
  timeout 900 python3 "$R/probes/fn_bench.py" --url "$API" --model "$MODEL" --tag "$TAG-c1-code" --kind code --distinct \
    --tokens 1024 --warmup-runs 1 --conc 1 --runs 3 --out "$R/records.jsonl" > "$R/bench-c1-code.log" 2>&1 \
    || { echo "fn_bench failed/timed out"; return; }
  n=$(grep -F "\"tag\": \"$TAG-c1-code\"" "$R/records.jsonl" 2>/dev/null | grep -cF '"ok": true')
  [ "${n:-0}" -ge 3 ] || { echo "fn_bench c1 code: ${n:-0} ok records, want 3: $(tail -2 "$R/bench-c1-code.log" | tr '\n' ' ' | cut -c1-160)"; return; }
  ms=$(python3 - "$R/records.jsonl" "$TAG-c1-code" <<'PY'
import json, statistics as st, sys
p, tag = sys.argv[1:3]
v = []
for ln in open(p):
    try:
        r = json.loads(ln)
    except ValueError:
        continue
    if r.get("tag") == tag and r.get("ok") and (r.get("client_frames") or 0) >= 2 and r.get("decode_window_s"):
        v.append(1000.0 * r["decode_window_s"] / (r["client_frames"] - 1))
print(f"{st.median(v):.3f} {len(v)} {' '.join(f'{x:.3f}' for x in v)}" if v else "none 0")
PY
)
  note "verify: fn_bench c1 code ms/step median + per-run: $ms"
  python3 -c 'import sys; m, c = sys.argv[1].split()[:2]; sys.exit(0 if m != "none" and int(c) >= 3 and float(m) <= float(sys.argv[2]) else 1)' "$ms" "$FN_MAX" \
    || { echo "fn_bench c1 code median ${ms%% *} ms/step (${ms#* } records) > $FN_MAX: not the fast state"; return; }
  # greedy identity vs R809 (R810 / R818's calls): fn_greedy then chat_greedy --long, each into a copy of the R809 rows
  cp "$R/ref-fn.jsonl" "$gf" || { echo "fn reference copy failed"; return; }
  timeout 3600 python3 "$R/probes/fn_greedy.py" --url "$BASEURL" --tag "$TAG" --out "$gf" > "$R/greedy-fn.log" 2>&1 \
    || { echo "fn_greedy failed/timed out"; return; }
  python3 "$R/probes/fn_greedy.py" --compare --out "$gf" --ref "$REF_TAG" > "$R/greedy-fn-compare.txt" 2>&1 \
    || { echo "fn_greedy vs $REF_TAG: $(grep -a "^GREEDY $TAG vs" "$R/greedy-fn-compare.txt" | tail -1 | cut -c1-120) (greedy-fn-compare.txt)"; return; }
  grep -q "^GREEDY $TAG vs $REF_TAG: 6 identical, IDENTICAL$" "$R/greedy-fn-compare.txt" \
    || { echo 'fn_greedy candidate incomplete'; return; }
  note "verify: $(grep -a "^GREEDY $TAG vs" "$R/greedy-fn-compare.txt" | tail -1)"
  cp "$R/ref-chat.jsonl" "$gc" || { echo "chat reference copy failed"; return; }
  timeout 3600 python3 "$R/probes/chat_greedy.py" --url "$API" --model "$MODEL" --tag "$TAG" --long --out "$gc" > "$R/chat-greedy.log" 2>&1 \
    || { echo "chat_greedy failed/timed out"; return; }
  python3 "$R/probes/chat_greedy.py" --compare --out "$gc" --ref "$REF_TAG" --cand "$TAG" --long > "$R/chat-greedy-compare.txt" 2>&1 \
    || { echo "chat_greedy --long vs $REF_TAG: $(grep -a '^CHAT-GREEDY .* vs ' "$R/chat-greedy-compare.txt" | tail -1 | cut -c1-120) (chat-greedy-compare.txt)"; return; }
  note "verify: $(grep -a '^CHAT-GREEDY .* vs ' "$R/chat-greedy-compare.txt" | tail -1)"
  # Retrieval at both daily long-context depths; this probe itself does not fail on MISS.
  timeout 3600 python3 "$R/probes/fn_needle_oai.py" --url "$API" --model "$MODEL" --tag "$TAG" \
    --ctx-tokens 131072 240000 --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needles.jsonl" > "$R/needles.log" 2>&1 \
    || { echo 'needle client failed/timed out'; return; }
  python3 "$R/probes/r825p_promote.py" needles "$R/needles.jsonl" "$TAG" > "$R/needles-check.txt" 2>&1 \
    || { echo 'needles incomplete/MISS/depth mismatch (needles-check.txt)'; return; }
  local since
  since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  timeout 900 python3 "$R/probes/r825p_promote.py" smoke-client "$R/fixtures" "$API" "$R/smoke-client.jsonl" > "$R/smoke-client.txt" 2>&1 \
    || { echo 'T32 smoke client/tokenizer failed'; return; }
  sudo docker logs --timestamps --since "$since" flashnext > "$R/smoke-container.log" 2>&1 \
    || { echo 'smoke log capture failed'; return; }
  python3 "$R/probes/r825p_promote.py" smoke-check "$R/fixtures" "$R/smoke-client.jsonl" "$R/smoke-container.log" \
    "$R/served-policy.py" "$R/smoke.json" > "$R/smoke-check.txt" 2>&1 \
    || { echo 'corrected T32 edited/HIT smoke failed (smoke-check.txt)'; return; }
  frontend candidate || { echo 'real frontend/cold/concurrency evidence failed (frontend-candidate-check.txt)'; return; }
  python3 "$R/probes/r825p_promote.py" frontend-compare "$R/frontend-old.json" "$R/frontend-candidate.json" \
    "$R/frontend-comparison.json" > "$R/frontend-comparison.txt" 2>&1 \
    || { echo 'frontend engine/decode-gap/health/late-arrival gate failed (frontend-comparison.txt)'; return; }
  sudo docker exec -e CUDA_VISIBLE_DEVICES= flashnext python3 /opt/r825c/landing_r825c.py > "$R/landing-end.txt" 2>&1 \
    || { echo 'post-traffic landing failed'; return; }
  [ "$(md5of "$LIVE")" = "$EXPECT_TPL" ] && [ "$(served_id)" = "$MODEL" ] &&
    [ "$(sudo docker inspect -f '{{.Id}}' flashnext)" = "$BOOT_CONTAINER" ] &&
    [ "$(sudo docker inspect -f '{{.Image}}' flashnext)" = "$IMG_ID" ] \
    || { echo 'launcher/container/model drift during gates'; return; }
  sudo docker logs --timestamps flashnext > "$R/container-promote.log" 2>&1 || { echo 'final log capture failed'; return; }
  echo "OK: served $MODEL, image ${IMG_ID:7:12}, $NKEYS keys, pool $POOL, readbacks '$rb' / '$pm', ${fw%% (*}, VRAM free $up (ref $REF_UP - 32), fn c1 code median ${ms%% *} ms/step (<= $FN_MAX), greedy + chat 6/6 identical to $REF_TAG, needles 10/10, corrected T32 edit policy-exact + immediate HIT >= N-256; frontend cold50 whole window <=0.95 x old daily, decode-first gap <=1.5 x old daily; health zero timeouts/errors and <=1s; late-arrival short TTFT <=old; trace ON, NVMe OFF"
}

# Measure the current served process BEFORE changing any launcher or stopping it.
# A baseline failure leaves the launcher untouched; candidate failures roll back.
# Boot a deferred daily only because this promotion requires OLD frontend evidence.
if [ -z "$(served_id)" ]; then
  BASELINE_BOOTED=1
  boot old || { FAILURE='deferred old daily baseline boot failed'; exit 3; }
fi
[ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.Image}}' flashnext)" = "$OLD_ID" ] \
  || { FAILURE='old daily identity failed'; exit 3; }
OLD_CONTAINER=$(sudo docker inspect -f '{{.Id}}' flashnext) || exit 3
sudo docker inspect flashnext > "$R/inspect-old.json" || exit 3
sudo docker exec flashnext cat /app/config.yml > "$R/config-old.yml" || exit 3
python3 "$R/probes/r825p_promote.py" old-env "$R/inspect-old.json" > "$R/env-old.txt" 2>&1 \
  || { FAILURE='old daily real env differs from reference'; exit 3; }
sudo docker exec -e CUDA_VISIBLE_DEVICES= flashnext python3 /opt/r823c-cachetail-inforward/landing_r823c.py \
  > "$R/landing-old.txt" 2>&1 || { FAILURE='old daily source landing failed'; exit 3; }
timeout 300 python3 "$R/probes/r825p_promote.py" prepare "$API" "$MODEL" "$R/frontend-prompts.json" \
  > "$R/frontend-prepare.txt" 2>&1 || { FAILURE='real-tokenizer frontend preparation failed'; exit 3; }
frontend old || { FAILURE='old daily frontend/concurrency baseline invalid'; exit 3; }
[ "$(sudo docker inspect -f '{{.Id}}' flashnext)" = "$OLD_CONTAINER" ] && \
  [ "$(md5of "$LIVE")" = "$EXPECT_OLD" ] || { FAILURE='baseline container/launcher drift'; exit 3; }

# Fail before changing the live file if any durable backup/archive cannot be written.
cp -p "$LIVE" "$PRE" && [ "$(md5of "$PRE")" = "$EXPECT_OLD" ] \
  || { FAILURE='backup identity/write failed'; exit 3; }
cp -p "$PRE" "$R/rollback.sh" || { FAILURE='backup archive failed'; exit 3; }
MUTATED=1
cp "$R/template.sh" "$LIVE.new" && mv "$LIVE.new" "$LIVE" \
  || { FAILURE='candidate installation failed'; exit 3; }
[ "$(md5of "$LIVE")" = "$EXPECT_TPL" ] || { FAILURE='installed template identity failed'; exit 3; }
log "installed $TPL as $LIVE; rollback $PRE = $EXPECT_OLD"
if boot promote; then
  BOOT_CONTAINER=$(sudo docker inspect -f '{{.Id}}' flashnext) || exit 3
  res=$(verify)
else
  res='promoted launcher did not come up (boot-promote.log)'
fi
case "$res" in
  OK:*) echo "PROMOTED $EXPECT_TPL" > "$R/decision.txt" || exit 3
        COMMITTED=1; log "PROMOTED: $res";;
  *) FAILURE=${res:-verify printed nothing}; exit 1;;
esac
