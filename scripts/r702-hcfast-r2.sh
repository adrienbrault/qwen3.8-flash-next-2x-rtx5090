#!/usr/bin/env bash
# R702 (2026-09-24): hcfast r2 gate on top of moefast m2 (HC-fast Opus agent r2; R697 review levers). OPERATIONS section 16 (stack
# track): a bitwise-identical flag is accepted on identity + no same-sign regression at c4d3/c8d1 over >= 5
# in-process pairs. Arms: OFF (served), HC1 = EXL3_HC_MIX_V3=1 DOTS_B=2 UP_B=8 (hcfast r1 at the R699 candidate
# tiling), HC2 = EXL3_HC_MIX_V3=2 + the knob tables P0 recommends (hcfast r2). Image tabbyapi:hcfast-r2 carries
# both (r1's kernels compile to the same SASS in it: r1-sass-identity.txt). Nothing here ran on a GPU yet.
#   0 build tabbyapi:hcfast-r2 on slotfix-r1, INSIDE the lock, before any measurement (OPERATIONS section 15)
#   1 parity (one card, bitwise, r1 + r2 + PDL configurations, graph capture)   -> any FAIL stops the round
#   2 P0 microbench (sweep + interleaved confirmation, 24 DRAM-cold sites)       -> CONTROL FAIL stops the round;
#     its RECOMMEND line is the HC2 arm's environment
#   3 P1 in-process harness, 4k, untraced: 5 rounds x (OFF, HC1, HC2), order rotated per round, at c4d3, c8d1,
#     c1d3; sequence hashes of HC1 and HC2 must equal the round's OFF
#   4 fn_greedy on the served launcher: ref (daily) vs HC2 ON vs image OFF -> divergences = 0 both
# DECISION (pre-registered, section 16), per flag set F in {HC1, HC2}:
#   identity: hashes identical in every round and shape (and, for HC2, greedy 0 divergences on and off)
#   reject:   all 5 F-OFF deltas > 0 at c4d3 or at c8d1 (same-sign regression)
#   accept:   no rejection and at least one shape with all 5 F-OFF deltas < 0
#   HC2 replaces HC1 in STACK.md if HC2 is accepted and HC2-HC1 is not a same-sign regression at any shape.
# STACK (section 16 "stack small wins"): STACK_FLAGS (default empty) holds the flags already accepted below hcfast in
# STACK.md (e.g. EXL3_MOE_COOP_V3=2 if R701 accepts moefast m2). It is prepended to EVERY arm (OFF, HC1, HC2, greedy on
# and off), so P1 measures HC on the stack, not HC vs the bare daily. A non-empty STACK_FLAGS that names
# EXL3_MOE_COOP_V3 needs an image built on tabbyapi:moefast-r1: set HC_BASE=tabbyapi:moefast-r1 (IMG defaults to
# tabbyapi:hcfast-r2-moe then). The operator fills STACK_FLAGS from STACK.md at queue time.
# The unit measures and reports; it does not promote. GPU ~75 min. Queue-chained; the daily is restored at the end.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
TAG=${TAG:-r702-hcfast-r2}
R=${R:-/srv/qwen5090/results/$(date +%F)-$TAG}; mkdir -p "$R"
HCF=${HCF:-/srv/qwen5090/patches/exllamav3/hcfast-r2}
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
HARNESS=/srv/qwen5090/probes/r465/profile_decode_events.py
METER=/srv/qwen5090/probes/r465/events_meter.py
GREEDY=/srv/qwen5090/probes/fn_greedy.py
STACK_FLAGS=${STACK_FLAGS:-}
HC_BASE=${HC_BASE:-}
ROUNDS=${ROUNDS:-5}
HC1_FLAGS="EXL3_HC_MIX_V3=1 EXL3_HC_MIX_V3_DOTS_B=2 EXL3_HC_MIX_V3_UP_B=8"
log(){ echo "$(date -Is) [$TAG] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$TAG
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ sudo docker rm -f hcfast-p hcfast-p1 >/dev/null 2>&1 || true
  sudo docker logs flashnext > "$R/container-final.log" 2>&1 || true
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== $TAG $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$HARNESS" "$METER" "$GREEDY" "$HCF/Dockerfile.box" "$HCF/hcfast-r2.patch"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
BASE=$(grep -oE '^DAILY_IMG=\S+' "$LIVE" | cut -d= -f2)
# R702: the daily became tabbyapi:stack-r2 (R701 = slotfix-r1 + hcfast r1 + moefast r1) before this unit ran. Queued with
# HC_BASE=tabbyapi:moefast-r1 STACK_FLAGS="EXL3_MOE_COOP_V3=2": the live EXTRA_ENV already carries HC1 (V3=1 DOTS_B=2 UP_B=8),
# so OFF and HC1 arms both run HC1 and the decision that matters is HC2 vs OFF (= HC2 vs the served HC1). Later -e wins.
case "$BASE" in tabbyapi:slotfix-r1|tabbyapi:stack-r2) ;; *) log "ABORT: daily image is $BASE"; exit 3 ;; esac
HC_BASE=${HC_BASE:-$BASE}
case "$HC_BASE" in
  tabbyapi:slotfix-r1) IMG=${IMG_OVERRIDE:-tabbyapi:hcfast-r2} ;;
  tabbyapi:moefast-r1) IMG=${IMG_OVERRIDE:-tabbyapi:hcfast-r2-moe} ;;
  *) log "ABORT: HC_BASE=$HC_BASE (slotfix-r1 or moefast-r1 only)"; exit 3 ;;
esac
case " $STACK_FLAGS " in *" EXL3_MOE_COOP_V3="*) [ "$HC_BASE" = tabbyapi:moefast-r1 ] \
  || { log "ABORT: STACK_FLAGS enables moefast but HC_BASE=$HC_BASE"; exit 3; } ;; esac
[ "$ROUNDS" -ge 5 ] || { log "ABORT: ROUNDS=$ROUNDS < 5 (section 16)"; exit 3; }

gpu_lock
log "lock held; served at entry: $(served_id || echo none); IMG $IMG on $HC_BASE; STACK_FLAGS [${STACK_FLAGS}]"

# ---- 0. build (in-lock, before any measurement) + landing ----
if ! sudo docker image inspect "$IMG" >/dev/null 2>&1; then
  log "building $IMG on $HC_BASE"
  ( cd "$HCF" && sudo docker build -f Dockerfile.box --build-arg BASE="$HC_BASE" --build-arg MAX_JOBS=4 -t "$IMG" . ) > "$R/build.log" 2>&1 \
    || { log "BUILD FAILED"; tail -30 "$R/build.log" | sed 's/^/  /' | tee -a "$R/audit.log"; finish BUILD-FAILED; exit 3; }
fi
grep -aE "hcfast r2 landed|selects r[12]|FLOW PASS|hcfast r2: patch applied" "$R/build.log" 2>/dev/null | sed 's/^/  [build] /' | tee -a "$R/audit.log"
sudo docker run --rm --entrypoint python3 "$IMG" -c '
import torch, exllamav3_ext as e, exllamav3.modules.hyperconnections as h
assert e.hc_mix_v3_revision == 2 and h._HC_MIX_V3_BUILD == "r2"
print("hcfast", h._HC_MIX_V3_BUILD, "ext rev", e.hc_mix_v3_revision)' > "$R/landed.txt" 2>&1 \
  || { log "ABORT: landing check failed: $(tail -2 "$R/landed.txt")"; finish ABORTED; exit 3; }
log "image $IMG: $(cat "$R/landed.txt")"

served_stop; wait_unserved 45
# ---- 1. parity ----
sudo docker run --rm --name hcfast-p --gpus '"device=0"' -v /srv/qwen5090/models:/models:ro -v "$R":/results \
  --entrypoint python3 "$IMG" /opt/hcfast-r2/test_hcfast_parity.py \
  --model /models/$MODEL --json /results/parity.json > "$R/parity.log" 2>&1
rc=$?
log "parity rc=$rc: $(grep -aE 'PARITY (PASS|FAILED)' "$R/parity.log" | tail -1)"
grep -aE "^\s+[0-9]+/[0-9]+ identical|AUTO-CAPTURE" "$R/parity.log" | sed 's/^/  [parity] /' | tee -a "$R/audit.log"
grep -aq "PARITY PASS" "$R/parity.log" && [ $rc = 0 ] || { tail -20 "$R/parity.log" | sed 's/^/  /' | tee -a "$R/audit.log"; finish PARITY-FAIL; exit 5; }
PDL_STATE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"].get("pdl", "missing"))' "$R/parity.json" 2>/dev/null)
log "parity PDL: $PDL_STATE"
NOPDL=""; [ "$PDL_STATE" = ok ] || NOPDL="--no-pdl"

# ---- 2. P0 microbench ----
sudo docker run --rm --name hcfast-p --gpus '"device=0"' -v /srv/qwen5090/models:/models:ro -v "$R":/results \
  --entrypoint python3 "$IMG" /opt/hcfast-r2/bench_hcfast_p0.py \
  --model /models/$MODEL --rows 1 4 8 16 --reps 504 $NOPDL --json /results/p0.json > "$R/p0.log" 2>&1
log "P0 rc=$?"
grep -aE "^R=|^ +(A|B|C) |CONTROL|RECOMMEND|PDL " "$R/p0.log" | sed 's/^/  [p0] /' | tee -a "$R/audit.log"
if grep -aq "CONTROL FAIL" "$R/p0.log"; then finish P0-CONTROL-FAIL; exit 6; fi
HC2_FLAGS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommend"])' "$R/p0.json" 2>/dev/null)
case "$HC2_FLAGS" in EXL3_HC_MIX_V3=2\ *) ;; *) log "ABORT: no RECOMMEND line in p0.json"; finish P0-NO-RECOMMEND; exit 6 ;; esac
log "arms: HC1 = $HC1_FLAGS ; HC2 = $HC2_FLAGS"

# ---- 3. P1 in-process, ROUNDS x (OFF, HC1, HC2) ----
EXTRA=$(sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-([^}]*)\}.*/\1/p' "$LIVE"); ENVS=""; for kv in $EXTRA; do ENVS="$ENVS -e $kv"; done
p1(){ local tag=$1 flags=$2 b=$3 d=$4 fe=""
  for kv in $flags; do fe="$fe -e $kv"; done
  sudo docker run --rm --name hcfast-p1 --gpus all --ipc=host --shm-size=16g \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v "$HARNESS":/probe/profile_decode_events.py:ro \
    -v "$METER":/probe/events_meter.py:ro -v "$R":/results $ENVS $fe \
    --entrypoint python3 "$IMG" /probe/profile_decode_events.py --model /models/$MODEL \
    --out /results/p1-$tag --cache-quant 8,8 --gpu-split 30,30 --max-chunk-size 2048 --tokens 64 \
    --warmup-steps 16 --settle-steps 8 --capture-steps 32 --capture-mode events \
    --contexts 4096 --batch $b --draft $d > "$R/p1-$tag.log" 2>&1
  log "  P1 $tag rc=$?: $(grep -ahE 'ms/iterate' "$R"/p1-$tag/ctx4096_b${b}_d${d}/kernels.txt 2>/dev/null | head -1)"; }
for shape in "c4d3 4 3" "c8d1 8 1" "c1d3 1 3"; do set -- $shape
  for r in $(seq 1 "$ROUNDS"); do
    case $(( r % 3 )) in 1) order="OFF HC1 HC2" ;; 2) order="HC1 HC2 OFF" ;; 0) order="HC2 OFF HC1" ;; esac
    for a in $order; do
      case $a in OFF) p1 $1-$a$r "$STACK_FLAGS" $2 $3 ;; HC1) p1 $1-$a$r "$STACK_FLAGS $HC1_FLAGS" $2 $3 ;; HC2) p1 $1-$a$r "$STACK_FLAGS $HC2_FLAGS" $2 $3 ;; esac
    done
    for a in HC1 HC2; do
      cmp -s "$R/p1-$1-OFF$r/ctx4096_b$2_d$3/sequence-hashes.json" "$R/p1-$1-$a$r/ctx4096_b$2_d$3/sequence-hashes.json" \
        && log "  $1 round $r $a: hashes IDENTICAL" || log "  $1 round $r $a: hashes DIFFER -> FAIL"
    done
  done
done
python3 - "$R" "$ROUNDS" <<'PY' 2>&1 | tee -a "$R/audit.log"
import re, sys, statistics as st
R, N = sys.argv[1], int(sys.argv[2])
def ms(tag, b, d):
    try:
        t = open(f"{R}/p1-{tag}/ctx4096_b{b}_d{d}/kernels.txt").read()
        return float(re.search(r"([0-9.]+) ms/iterate", t).group(1))
    except Exception:
        return None
def sign(ds):
    return "GAIN" if ds and all(v < 0 for v in ds) else "REGRESSION" if ds and all(v > 0 for v in ds) else "mixed"
verdict = {}
for sh, b, d in (("c4d3", 4, 3), ("c8d1", 8, 1), ("c1d3", 1, 3)):
    off = [ms(f"{sh}-OFF{r}", b, d) for r in range(1, N + 1)]
    offm = st.mean([v for v in off if v]) if any(off) else float("nan")
    for x, y in (("HC1", "OFF"), ("HC2", "OFF"), ("HC2", "HC1")):
        ds = []
        for r in range(1, N + 1):
            a, o = ms(f"{sh}-{x}{r}", b, d), ms(f"{sh}-{y}{r}", b, d)
            if a and o:
                ds.append(a - o)
        if not ds:
            print(f"P1 {sh} {x}-{y}: no data"); continue
        s = sign(ds); verdict[(sh, x, y)] = s
        print(f"P1 {sh} {x}-{y} ms: " + " ".join(f"{v:+.2f}" for v in ds)
              + f"  mean {st.mean(ds):+.3f} ({100 * st.mean(ds) / offm:+.2f} %) {s} (n={len(ds)})")
for F in ("HC1", "HC2"):
    rej = [sh for sh in ("c4d3", "c8d1") if verdict.get((sh, F, "OFF")) == "REGRESSION"]
    gain = [sh for sh in ("c4d3", "c8d1", "c1d3") if verdict.get((sh, F, "OFF")) == "GAIN"]
    print(f"DECISION {F}: " + ("REJECT (same-sign regression at " + ", ".join(rej) + ")" if rej
          else ("ACCEPT (gains at " + ", ".join(gain) + "; identity per the hash lines above)") if gain
          else "FLAT (no same-sign regression, no same-sign gain)"))
reg = [sh for sh in ("c4d3", "c8d1", "c1d3") if verdict.get((sh, "HC2", "HC1")) == "REGRESSION"]
print("HC2 vs HC1: " + ("HC2 regresses vs HC1 at " + ", ".join(reg) if reg else "no same-sign regression vs HC1"))
PY

# ---- 4. greedy identity on the served launcher ----
G=$R/greedy.jsonl
boot(){ served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "$@" bash "$LIVE" > "$R/boot-$(date +%s).log" 2>&1 \
    && wait_served_id "$MODEL" 200 8; }
boot || { log "daily NO BOOT"; finish NO-BOOT; exit 3; }
python3 "$GREEDY" --url http://127.0.0.1:8022 --tag ref --out "$G" > "$R/greedy-ref.log" 2>&1
for arm in on off; do
  if [ $arm = on ]; then boot IMG=$IMG EXTRA_ENV_ADD="$STACK_FLAGS $HC2_FLAGS"; else boot IMG=$IMG EXTRA_ENV_ADD="$STACK_FLAGS"; fi || { log "$arm NO BOOT"; continue; }
  log "  greedy $arm: free $(vram_free); $(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'); r2 flag in env: $(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c '^EXL3_HC_MIX_V3=2$') $(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^EXL3_(HC_MIX_V3_(DOTS_B|DOTS_J|DOTS_PF|UP_B|UP_Q|PDL)|MOE_COOP_V3)=' | tr '\n' ' '); OOM $(sudo docker logs flashnext 2>&1 | grep -acE 'OutOfMemoryError|out of memory|graph.cu')"
  python3 "$GREEDY" --url http://127.0.0.1:8022 --tag $arm --out "$G" > "$R/greedy-$arm.log" 2>&1
  python3 "$GREEDY" --compare --ref ref --out "$G" 2>&1 | grep -aE "$arm vs ref|GREEDY-SUMMARY" | sed "s/^/  [greedy $arm] /" | tee -a "$R/audit.log"
  sudo docker logs flashnext > "$R/container-greedy-$arm.log" 2>&1
done
finish DONE
