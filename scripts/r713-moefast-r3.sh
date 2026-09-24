#!/usr/bin/env bash
# moefast r3 gate (2026-09-24; modelled on flan/r703-moefast-r2.sh with the R703b review's gate fixes). The operator
# assigns the R number: install as /srv/qwen5090/rNNN-moefast-r3.sh and queue it (UNIT = the file name).
# Candidate: r2's routed-expert kernels (EXL3_MOE_COOP_V3=3, R703b: bitwise, faster in situ, flat in P1 because the
# side-stream shared expert lost its overlap) + r3's overlap levers, all default off, bitwise by construction:
#   EXL3_SHARED_EXPERT_EARLY=1  fork the shared expert before the router (BlockSparseMLP.forward -> bc.start_shared)
#   EXL3_SHARED_EXPERT_PRIO=1   side stream at the device's greatest priority
#   EXL3_MOE_COOP_V3_HEAD=<ns>  diagnostic only (P0 arms m3H/m3H2), never in R3
# Image: tabbyapi:stack-moefast-r3 = the daily image (DAILY_IMG, tabbyapi:stack-r2) + moefast-r3.patch (cumulative
# over stack-r2). Arms: OFF (V2) / M2 (the served stack-r2 env: moefast r1 mode 2, late fork) / R3. Question: R3 - M2.
# Steps:
#   0 build in-lock BEFORE any measurement (§15) when missing or built from another patch/base (labels); the install
#     fails unless every stack-r2 function keeps its SASS and exactly 36 r2 + 1 head kernels are added; landing check
#     imports torch BEFORE exllamav3_ext
#   1 parity, one card: r2 suite at rows 1-16 (incl. 14 = c7d1, 15 = c5d2), side-stream layer calls (serial / late /
#     early / early+prio / head vs serial V2), join stress, dispatch -> any FAIL stops the unit
#   2 P0 with the served side stream + router (--extra): instrument acceptance (m3 - m2 A->B gap >= 5 us at r4/D28 =
#     R703b reproduced) and the pre-registered PICK -> R3_ENV. rc != 0 stops the unit. If the instrument does not
#     reproduce or the pick fails, R3 falls back to the pre-registered default (R3_DEFAULT) and the audit says so.
#     An operator override R3_ENV=... (env) wins over both and is logged.
#   2a P0 L2 probe (r2's open L2-policy question; the R703 TypeError is fixed). rc != 0 stops the unit.
#   3 nsys c1d3 and c4d3, arms OFF/M2/R3 (+ M2E = mode 2 with R3's overlap flags, c1d3 only) -> moe_timeline.py with
#     --gap-target 4 (r3 target: A->B gap ~2 us). Any nsys/timeline error stops the unit; GAP HIGH is reported.
#   4 P1 in-process harness, 4k, untraced: ROUNDS (default 6) rounds x (OFF, M2, R3) at c4d3, c8d1, c1d3 (+ c2d3 if
#     C2D3=1, report-only), each arm first in exactly ROUNDS/3 rounds, one DISCARDED warm-up run at the start of every
#     round; sequence hashes of M2 and R3 == the round's OFF in the d AND the d0 cell
#   5 fn_greedy on the served launcher: ref (daily) vs $IMG with EXTRA_ENV = served env minus arm keys + R3 (on) vs
#     $IMG plain (off = the served M2 env on the new image); the effective container env is asserted with docker exec
#     (R3 keys present, no duplicate keys); divergences are GATED (0 required)
# DECISION (§16 incl. the 2026-09-24 gate template, pre-registered; deltas R3 - M2 paired by round, MEDIANS read):
#   identity: parity PASS; hashes identical in every round, shape and cell (d and d0); greedy on and off 0 divergences
#   pairs:    >= 5 valid R3/M2 pairs at every shape (c1d3 included)
#   no regression: c4d3 and c8d1 not all-positive (same-sign regression = REJECT)
#   c1 bound: an all-positive c1d3 is tolerated only if it is <= 2 % of M2 by BOTH the in-process mean (§16 wording)
#             and the median, AND c4d3 and/or c8d1 is an all-negative GAIN whose summed ms (mean and median) exceeds
#             the c1d3 loss
#   gain clause: at least one of c1d3/c4d3/c8d1 all-negative (6/6); otherwise "FLAT: not a stack entry"
#   d0 cells (ctx4096_b{1,4,8}_d0) are read and reported (and hash-checked), not gated.
# This unit does not promote. GPU ~2 h (estimate: build ~15 min if missing, parity ~8, P0 ~8, L2 ~2, nsys ~8, P1 72
# harness runs ~75, three boots + greedy ~12). Queue-chained; the daily is restored at the end.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
UNIT=${UNIT:-$(basename "$0" .sh)}
R=${R:-/srv/qwen5090/results/$(date +%F)-$UNIT}; mkdir -p "$R"
MF=${MF:-/srv/qwen5090/patches/exllamav3/moefast-r3}
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
HARNESS=/srv/qwen5090/probes/r465/profile_decode_events.py
METER=/srv/qwen5090/probes/r465/events_meter.py
GREEDY=/srv/qwen5090/probes/fn_greedy.py
NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
IMG=${IMG_R3:-tabbyapi:stack-moefast-r3}
ROUNDS=${ROUNDS:-6}
C2D3=${C2D3:-0}
R3_DEFAULT="EXL3_MOE_COOP_V3=3 EXL3_SHARED_EXPERT_EARLY=1"
R3_OVERRIDE=${R3_ENV:-}
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$UNIT
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ sudo docker rm -f moefast-p moefast-p1 moefast-n >/dev/null 2>&1 || true
  sudo docker logs flashnext > "$R/container-final.log" 2>&1 || true
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== moefast r3 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$HARNESS" "$METER" "$GREEDY" "$MF/Dockerfile.box" "$MF/moefast-r3.patch" "$MF/install-moefast.sh" \
         "$MF/moe_timeline.py" "$MF/bench_moefast_p0.py" "$MF/moefast_r3_side.py" "$MF/test_moefast_parity.py" \
         "$MF/sass_hashes.py" "$MF/sass_identity.py" "$MF/d0/d0_microbench.py"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
[ $((ROUNDS % 3)) = 0 ] && [ "$ROUNDS" -ge 6 ] || { log "ABORT: ROUNDS must be a multiple of 3 and >= 6 (balanced rotation, >= 5 pairs)"; exit 3; }
BASE=$(grep -m1 -oE '^DAILY_IMG=[^ ]+' "$LIVE" | cut -d= -f2 | tr -d "'\"")
EXTRA=$(sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-([^}]*)\}.*/\1/p' "$LIVE")
[ -n "$BASE" ] && [ -n "$EXTRA" ] || { log "ABORT: could not read DAILY_IMG / EXTRA_ENV from $LIVE"; exit 3; }
for kv in EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1 EXL3_MOE_COOP_V3=2; do
  echo "$EXTRA" | tr ' ' '\n' | grep -qx "$kv" || { log "ABORT: live env lacks $kv (M2 must be the served config)"; exit 3; }; done
# the served env minus every key an arm sets (no reliance on docker's last-wins; R703b review note 10). ENVS feeds
# docker run -e; ENVSTR is the same set as a string for the launcher's full EXTRA_ENV override (greedy "on")
ARMKEYS='^(EXL3_MOE_COOP_V3|EXL3_MOE_COOP_V3_MAP|EXL3_MOE_COOP_V3_HEAD|EXL3_SHARED_EXPERT_EARLY|EXL3_SHARED_EXPERT_PRIO)='
ENVS=""; ENVSTR=""; for kv in $EXTRA; do echo "$kv" | grep -qE "$ARMKEYS" || { ENVS="$ENVS -e $kv"; ENVSTR="$ENVSTR $kv"; }; done
PSHA=$(sha256sum "$MF/moefast-r3.patch" | cut -c1-64)
sudo docker image inspect "$BASE" >/dev/null 2>&1 || { log "ABORT: base image $BASE missing"; exit 3; }

gpu_lock
log "lock held; served at entry: $(served_id || echo none); BASE $BASE; IMG $IMG; patch sha $PSHA; rounds $ROUNDS; c2d3 $C2D3"

# ---- 0. build + landing ----
lbl(){ sudo docker image inspect "$IMG" --format "{{ index .Config.Labels \"$1\" }}" 2>/dev/null; }
if ! sudo docker image inspect "$IMG" >/dev/null 2>&1 || [ "$(lbl local.moefast.patch_sha256)" != "$PSHA" ] || [ "$(lbl local.moefast.base)" != "$BASE" ]; then
  log "building $IMG on $BASE"
  ( cd "$MF" && sudo docker build -f Dockerfile.box --build-arg BASE="$BASE" --build-arg MAX_JOBS=4 --build-arg PATCH_SHA="$PSHA" -t "$IMG" . ) > "$R/build.log" 2>&1 \
    || { log "BUILD FAILED"; tail -30 "$R/build.log" | sed 's/^/  /' | tee -a "$R/audit.log"; finish BUILD-FAILED; exit 3; }
  grep -aE "moefast-r3: (base functions|served SASS)|moefast r3 landed|CPU TESTS" "$R/build.log" | sed 's/^/  [build] /' | tee -a "$R/audit.log"
fi
sudo docker run --rm --entrypoint python3 "$IMG" -c '
import torch, exllamav3_ext as e
assert e.moe_coop_v3_revision == 3, e.moe_coop_v3_revision
assert hasattr(e, "exl3_moe_coop_ev") and hasattr(e.BC_BlockSparseMLP, "start_shared")
import exllamav3.modules.block_sparse_mlp as b, exllamav3.modules.hyperconnections as h
assert hasattr(b.BlockSparseMLP, "_shared_early_ok")
print("moefast", e.moe_coop_v3_revision, "hcfast", getattr(h, "_HC_MIX_V3_BUILD", None))' > "$R/landed.txt" 2>&1 \
  || { log "ABORT: landing check failed: $(tail -2 "$R/landed.txt")"; finish ABORTED; exit 3; }
log "image $IMG: $(tail -1 "$R/landed.txt")"

served_stop; wait_unserved 45
one(){ sudo docker run --rm --name moefast-p --gpus '"device=0"' --ipc=host -v /srv/qwen5090/models:/models:ro -v "$R":/results \
  -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
  $ENVS -e CUDA_VISIBLE_DEVICES=0 --entrypoint python3 "$IMG" "$@"; }

# ---- 1. parity ----
one /opt/moefast-r3/test_moefast_parity.py --model /models/$MODEL --json /results/parity.json > "$R/parity.log" 2>&1
rc=$?
log "parity rc=$rc: $(grep -aE 'PARITY (PASS|FAIL)|comparisons equal|side .*equal|join stress' "$R/parity.log" | tail -8 | tr '\n' ' ')"
grep -aq "PARITY PASS" "$R/parity.log" && [ $rc = 0 ] || { tail -20 "$R/parity.log" | sed 's/^/  /' | tee -a "$R/audit.log"; finish PARITY-FAIL; exit 5; }

# ---- 2. P0 with the served side stream ----
one /opt/moefast-r3/bench_moefast_p0.py --model /models/$MODEL --out /results/p0 --extra > "$R/p0.log" 2>&1
rc=$?
log "P0 rc=$rc"
grep -aE "^\[p0\] (K[0-9]|  |control|r[0-9]+ D|INSTRUMENT|PICK|R3_ENV|side|WARNING)|^\[side\]" "$R/p0.log" | sed 's/^/  /' | tee -a "$R/audit.log"
[ $rc = 0 ] || { log "DIAGNOSTIC ERROR: P0 rc=$rc"; tail -20 "$R/p0.log" | sed 's/^/  /' | tee -a "$R/audit.log"; finish DIAG-FAIL; exit 6; }
PICKED=$(sed -nE 's/^\[p0\] R3_ENV=(.*)$/\1/p' "$R/p0.log" | tail -1)
if [ -n "$R3_OVERRIDE" ]; then R3="$R3_OVERRIDE"; log "R3 = operator override: $R3"
elif grep -aq "^\[p0\] INSTRUMENT OK" "$R/p0.log" && [ -n "$PICKED" ]; then R3="$PICKED"; log "R3 = P0 pick: $R3"
else R3="$R3_DEFAULT"; log "R3 = PRE-REGISTERED DEFAULT ($R3): P0 instrument $(grep -aoE 'INSTRUMENT (OK|NOT REPRODUCED)' "$R/p0.log" | tail -1), pick '${PICKED:-none}' -- READ P0 BEFORE TRUSTING THIS ROUND"; fi
echo "$R3" | tr ' ' '\n' | grep -qE '^EXL3_SHARED_EXPERT_EARLY=1$' || log "NOTE: R3 has no early fork"
M2E="EXL3_MOE_COOP_V3=2 $(echo "$R3" | tr ' ' '\n' | grep -E '^EXL3_SHARED_EXPERT_' | tr '\n' ' ')"
# ---- 2a. L2 probe (legacy calls, no side stream) ----
one /opt/moefast-r3/bench_moefast_p0.py --model /models/$MODEL --out /results/p0 --l2probe > "$R/p0-l2probe.log" 2>&1
rc=$?
log "L2 probe rc=$rc"
grep -aE "^\[p0\] l2probe" "$R/p0-l2probe.log" | sed 's/^/  /' | tee -a "$R/audit.log"
[ $rc = 0 ] || { log "DIAGNOSTIC ERROR: L2 probe rc=$rc"; tail -10 "$R/p0-l2probe.log" | sed 's/^/  /' | tee -a "$R/audit.log"; finish DIAG-FAIL; exit 6; }

# ---- arms ----
declare -A ARMF=( [OFF]="EXL3_MOE_COOP_V3=0" [M2]="EXL3_MOE_COOP_V3=2" [R3]="$R3" [M2E]="$M2E" )
flags_of(){ local fe="" kv; for kv in ${ARMF[$1]}; do fe="$fe -e $kv"; done; echo "$fe"; }
log "arms: OFF='${ARMF[OFF]}' M2='${ARMF[M2]}' R3='${ARMF[R3]}' (nsys only: M2E='${ARMF[M2E]}'); common = served env minus arm keys"
run_harness(){ local name=$1 arm=$2; shift 2
  sudo docker run --rm --name "$name" --gpus all --ipc=host --shm-size=16g \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v "$HARNESS":/probe/profile_decode_events.py:ro \
    -v "$METER":/probe/events_meter.py:ro -v "$R":/results $ENVS $(flags_of "$arm") \
    --entrypoint "$@"; }

# ---- 3. nsys timelines, d3 shape only (--worker) ----
if sudo docker run --rm --entrypoint test "$IMG" -x "$NSYS"; then
  for shape in "c1d3 1 3" "c4d3 4 3"; do set -- $shape
    arms="OFF M2 R3"; [ "$1" = c1d3 ] && arms="OFF M2 R3 M2E"
    tl=""
    for a in $arms; do
      run_harness moefast-n $a "$NSYS" "$IMG" profile -t cuda,nvtx --cuda-graph-trace=node -o /results/nsys-$1-$a \
        --force-overwrite true --export sqlite \
        python3 /probe/profile_decode_events.py --model /models/$MODEL --out /results/nsys-run-$1-$a --worker \
        --cache-quant 8,8 --gpu-split 30,30 --max-chunk-size 2048 --tokens 64 \
        --warmup-steps 16 --settle-steps 8 --capture-steps 32 --capture-mode wall \
        --contexts 4096 --batch $2 --draft $3 > "$R/nsys-$1-$a.log" 2>&1
      rc=$?
      log "  nsys $1 $a rc=$rc: $(ls -la "$R"/nsys-$1-$a.sqlite 2>/dev/null | awk '{print $5}') bytes"
      [ $rc = 0 ] && [ -s "$R/nsys-$1-$a.sqlite" ] || { log "DIAGNOSTIC ERROR: nsys $1 $a"; finish DIAG-FAIL; exit 6; }
      tl="$tl $a=$R/nsys-$1-$a.sqlite"
    done
    python3 "$MF/moe_timeline.py" --layers-per-step 48 --gap-target 4 $tl > "$R/timeline-$1.txt" 2>&1
    rc=$?
    log "  timeline $1 (rc=$rc):"; sed "s/^/    [$1] /" "$R/timeline-$1.txt" | head -56 | tee -a "$R/audit.log"
    [ $rc = 0 ] || { log "DIAGNOSTIC ERROR: moe_timeline $1"; finish DIAG-FAIL; exit 6; }
  done
else
  log "DIAGNOSTIC ERROR: nsys not in $IMG"; finish DIAG-FAIL; exit 6
fi

# ---- 4. P1 in-process, balanced rotation + discarded warm-up per round ----
p1(){ local tag=$1 arm=$2 b=$3 d=$4
  run_harness moefast-p1 $arm python3 "$IMG" /probe/profile_decode_events.py --model /models/$MODEL \
    --out /results/p1-$tag --cache-quant 8,8 --gpu-split 30,30 --max-chunk-size 2048 --tokens 64 \
    --warmup-steps 16 --settle-steps 8 --capture-steps 32 --capture-mode events \
    --contexts 4096 --batch $b --draft $d > "$R/p1-$tag.log" 2>&1
  local rc=$?
  [ $rc = 0 ] || log "  DIAGNOSTIC ERROR: P1 $tag rc=$rc (its pairs/hashes count as MISSING below)"
  log "  P1 $tag rc=$rc: $(grep -ahE 'ms/iterate' "$R"/p1-$tag/ctx4096_b${b}_d${d}/kernels.txt 2>/dev/null | head -1) | d0 $(grep -ahE 'ms/iterate' "$R"/p1-$tag/ctx4096_b${b}_d0/kernels.txt 2>/dev/null | head -1)"; }
ARMS="OFF M2 R3"
SHAPES="c4d3:4:3 c8d1:8:1 c1d3:1:3"; [ "$C2D3" = 1 ] && SHAPES="$SHAPES c2d3:2:3"
for s in $SHAPES; do IFS=: read -r sh b d <<< "$s"
  for r in $(seq 1 "$ROUNDS"); do
    k=$(( (r - 1) % 3 )); order=$(echo $ARMS $ARMS | cut -d' ' -f$(( k + 1 ))-$(( k + 3 )))
    first=${order%% *}
    p1 $sh-W$r $first $b $d          # warm-up, discarded (the review's first-in-round slow runs)
    for a in $order; do p1 $sh-$a$r $a $b $d; done
    for a in M2 R3; do for dd in $d 0; do
      h0="$R/p1-$sh-OFF$r/ctx4096_b${b}_d$dd/sequence-hashes.json"; h1="$R/p1-$sh-$a$r/ctx4096_b${b}_d$dd/sequence-hashes.json"
      if [ ! -s "$h0" ] || [ ! -s "$h1" ]; then log "  $sh round $r $a d$dd: hashes MISSING (run failed) -> FAIL"
      elif cmp -s "$h0" "$h1"; then log "  $sh round $r $a d$dd: hashes IDENTICAL"
      else log "  $sh round $r $a d$dd: hashes DIFFER -> FAIL"; fi
    done; done
  done
done
python3 - "$R" "$ROUNDS" "$SHAPES" > "$R/p1-summary.txt" 2>&1 <<'PY'
import filecmp, os, re, statistics as st, sys
R, N, SHAPES = sys.argv[1], int(sys.argv[2]), [s.split(":") for s in sys.argv[3].split()]
def ms(tag, b, d):
    try:
        t = open(f"{R}/p1-{tag}/ctx4096_b{b}_d{d}/kernels.txt").read()
        return float(re.search(r"([0-9.]+) ms/iterate", t).group(1))
    except Exception:
        return None
def same(sh, a, r, b, d):
    h0, h1 = (f"{R}/p1-{sh}-{x}{r}/ctx4096_b{b}_d{d}/sequence-hashes.json" for x in ("OFF", a))
    if not (os.path.exists(h0) and os.path.exists(h1)):
        return "missing"
    return "same" if filecmp.cmp(h0, h1, shallow=False) else "differ"
res = {}
ident = {"differ": 0, "missing": 0}
for sh, b, d in SHAPES:
    for cell, dd in (("d", d), ("d0", "0")):
        for x, y in (("R3", "M2"), ("M2", "OFF"), ("R3", "OFF")):
            ds = []
            for r in range(1, N + 1):
                vx, vy = ms(f"{sh}-{x}{r}", b, dd), ms(f"{sh}-{y}{r}", b, dd)
                if vx and vy:
                    ds.append(vx - vy)
            ref = [v for v in (ms(f"{sh}-{y}{r}", b, dd) for r in range(1, N + 1)) if v]
            if not ds or not ref:
                print(f"P1 {sh} {cell} {x}-{y}: NO DATA"); continue
            sign = "GAIN" if all(v < 0 for v in ds) else "REGRESSION" if all(v > 0 for v in ds) else "mixed"
            med, base = st.median(ds), st.median(ref)
            print(f"P1 {sh} {cell} {x}-{y} ms: " + " ".join(f"{v:+.3f}" for v in ds) +
                  f"  median {med:+.3f} ({100 * med / base:+.2f} % of {y} median {base:.3f}); mean {st.mean(ds):+.3f}; "
                  f"{sign} [{sum(v < 0 for v in ds)}/{len(ds)} negative]")
            if x == "R3" and y == "M2":
                mbase = st.mean(ref)
                res[(sh, cell)] = dict(n=len(ds), sign=sign, med=med, pct=100 * med / base,
                                       mean=st.mean(ds), pct_mean=100 * st.mean(ds) / mbase)
        for a in ("M2", "R3"):
            v = [same(sh, a, r, b, dd) for r in range(1, N + 1)]
            print(f"P1 {sh} {cell} {a} hashes vs OFF: {v.count('same')}/{N} identical, "
                  f"{v.count('differ')} DIFFER, {v.count('missing')} missing")
            for k_ in ident:
                ident[k_] += v.count(k_)
gated = [s for s, _, _ in SHAPES if s in ("c1d3", "c4d3", "c8d1")]
why = []
if ident["differ"]:
    why.append(f"{ident['differ']} hash comparisons DIFFER (numerics)")
if ident["missing"]:
    why.append(f"{ident['missing']} hash comparisons missing (run failures, see DIAGNOSTIC ERROR lines)")
for s in gated:
    if res.get((s, "d"), {}).get("n", 0) < 5:
        why.append(f"{s}: fewer than 5 pairs")
for s in ("c4d3", "c8d1"):
    if res.get((s, "d"), {}).get("sign") == "REGRESSION":
        why.append(f"{s}: same-sign regression (median {res[(s, 'd')]['med']:+.3f} ms)")
gains = [s for s in gated if res.get((s, "d"), {}).get("sign") == "GAIN"]
c1 = res.get(("c1d3", "d"), {})
if c1.get("sign") == "REGRESSION":
    # OPERATIONS §16 c1 exception: <= 2 % of the in-process MEAN, and the c4d3/c8d1 gain larger in ms; the brief
    # also reads MEDIANS, so both statistics must pass
    hg = [res[(s, "d")] for s in ("c4d3", "c8d1") if res.get((s, "d"), {}).get("sign") == "GAIN"]
    g_med, g_mean = -sum(x["med"] for x in hg), -sum(x["mean"] for x in hg)
    if not (c1["pct"] <= 2.0 and c1["pct_mean"] <= 2.0 and g_med > c1["med"] and g_mean > c1["mean"]):
        why.append(f"c1d3: same-sign regression median {c1['pct']:+.2f} % ({c1['med']:+.3f} ms), mean {c1['pct_mean']:+.2f} % "
                   f"({c1['mean']:+.3f} ms) not within 2 % and covered by c4/c8 gains (median {g_med:.3f}, mean {g_mean:.3f} ms)")
if why:
    print("P1 VERDICT: REJECT: " + "; ".join(why))
elif not gains:
    print("P1 VERDICT: FLAT: identity and no regression, but no shape gains (6/6) -- not a stack entry")
else:
    print("P1 VERDICT: PASS: identity, no same-sign regression, gain at " + ", ".join(gains) +
          (f"; c1d3 cost median {c1['pct']:+.2f} % / mean {c1['pct_mean']:+.2f} % (record in STACK.md)" if c1.get("sign") == "REGRESSION" else ""))
PY
tee -a "$R/audit.log" < "$R/p1-summary.txt"

# ---- 5. greedy identity on the served launcher (gated) ----
G=$R/greedy.jsonl
boot(){ served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "$@" bash "$LIVE" > "$R/boot-$(date +%s).log" 2>&1 \
    && wait_served_id "$MODEL" 200 8; }
effenv(){ sudo docker exec flashnext env | grep -E '^EXL3_(MOE_COOP_V3|MOE_COOP_V3_MAP|MOE_COOP_V3_HEAD|SHARED_EXPERT_[A-Z]+)=' | sort | tr '\n' ' '; }
GOK=1
boot || { log "daily NO BOOT"; finish NO-BOOT; exit 3; }
python3 "$GREEDY" --url http://127.0.0.1:8022 --tag ref --out "$G" > "$R/greedy-ref.log" 2>&1
for a in on off; do
  if [ $a = on ]; then boot IMG=$IMG EXTRA_ENV="$ENVSTR $R3"; else boot IMG=$IMG; fi || { log "greedy $a NO BOOT -> FAIL"; GOK=0; continue; }
  env_now=$(effenv)
  dup=$(echo "$env_now" | tr ' ' '\n' | grep . | cut -d= -f1 | sort | uniq -d | tr '\n' ' ')
  [ -z "$dup" ] || { log "  greedy $a: duplicate keys in the container env ($dup) -> FAIL"; GOK=0; }
  log "  greedy $a: $(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'); effective env: $env_now"
  if [ $a = on ]; then for kv in $R3; do echo " $env_now" | grep -qF " $kv " || { log "  greedy on: effective env lacks $kv -> FAIL"; GOK=0; }; done
  else echo " $env_now" | grep -qF " EXL3_MOE_COOP_V3=2 " && ! echo "$env_now" | grep -qF "EXL3_SHARED_EXPERT_EARLY=1" \
         || { log "  greedy off: effective env is not the served M2 -> FAIL"; GOK=0; }; fi
  python3 "$GREEDY" --url http://127.0.0.1:8022 --tag $a --out "$G" > "$R/greedy-$a.log" 2>&1
  python3 "$GREEDY" --compare --ref ref --out "$G" > "$R/greedy-compare-$a.txt" 2>&1
  grep -aE "GREEDY $a vs ref|GREEDY-SUMMARY|DIVERGE $a|MISSING $a" "$R/greedy-compare-$a.txt" | sed "s/^/  [greedy $a] /" | tee -a "$R/audit.log"
  grep -aqE "^GREEDY $a vs ref: [0-9]+ identical, IDENTICAL" "$R/greedy-compare-$a.txt" || { log "  greedy $a: divergences or no result -> FAIL"; GOK=0; }
  sudo docker logs flashnext > "$R/container-greedy-$a.log" 2>&1
  log "  greedy $a: OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-greedy-$a.log"); tracebacks $(grep -ac Traceback "$R/container-greedy-$a.log"); free $(vram_free 2>/dev/null || echo n/a)"
done

# ---- DECISION ----
V=$(grep -aE '^P1 VERDICT' "$R/p1-summary.txt" | tail -1)
if [ "$GOK" != 1 ]; then log "DECISION: REJECT (greedy identity failed) | $V | R3='$R3'"
else case "$V" in
  *"VERDICT: PASS"*) log "DECISION: ACCEPT into the stack ledger | $V | greedy on/off 0 divergences | R3='$R3'";;
  *"VERDICT: FLAT"*) log "DECISION: NOT AN ENTRY (flat) | $V | greedy 0 divergences | R3='$R3'";;
  *) log "DECISION: REJECT | $V | R3='$R3'";;
esac; fi
finish DONE
