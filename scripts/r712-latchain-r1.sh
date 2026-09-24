#!/usr/bin/env bash
# R712 (queued 2026-09-24 by the operator from patches/exllamav3/latchain-r1/rXXX-latchain-r1.sh; RN default only): latchain r1 stack gate.
# Latency-chain levers (latchain-r1 Opus agent, 2026-09-24), each its own flag, all default off, built on the served
# stack-r2 (BASE=tabbyapi:stack-r2 + the live launcher's 27-key EXTRA_ENV in every arm, so OFF = the daily):
#   RR  EXL3_LC_GDN_RR=1      register-resident GDN recurrent kernel
#   QF  EXL3_LC_QSA_FORK=1    QSA indexer chain as a parallel CUDA-graph branch
#   GF  EXL3_LC_GDN_FORK=1    GDN b/a GEMV as a parallel CUDA-graph branch
#   QT  EXL3_LC_QSA_SPLIT_STAGES / _COMBINE_STAGES / _DIV16 = the P0-chosen QSA compile options (skipped if none)
#   ALL every flag above together (the composition check; not itself a stack entry)
# Bars: OPERATIONS §16 incl. the R705 gate template, pre-registered here:
#   identity   parity PASS (kernel-level, bitwise); model parity not FAIL (logits hashes, when the A/A control is
#              deterministic); P1 sequence hashes of the arm == the round's OFF in EVERY round, shape and cell
#              (drafting and d0); fn_greedy divergences = 0 for on-vs-ref AND off-vs-ref (gated, not logged)
#   speed      per arm vs OFF at c1d3, c4d3, c8d1 (drafting cells ctx4096_b{1,4,8}_d{3,3,1}), >= 5 interleaved pairs
#              per shape: all pairs < 0 = GAIN, all > 0 = REGRESSION, mixed = flat.
#              REJECT on a same-sign regression at c4d3 or c8d1.
#              c1d3 same-sign regression allowed only if its mean <= 2 % AND c4d3 and/or c8d1 GAIN AND the larger
#              GAIN-shape mean |delta ms| exceeds the c1d3 mean loss in ms.
#              GAIN CLAUSE: at least one shape GAIN, else the arm is flat and is NOT a stack entry.
#   d0 cells   ctx4096_b{1,4,8}_d0 (5-20x quieter) are read and reported per arm next to the drafting cells
#   rotation   each round rotates the arm order by one, ROUNDS = a multiple of the arm count (>= 5), so every arm
#              runs first equally often
#   headroom   cuda:0 used MiB after the ON greedy boot <= the ref boot's + 32 MiB (logged next to the decision)
# Steps:
#   0 build $IMG = $BASE + latchain-r1.patch in the lock, before any measurement (§15); landing check imports torch first
#   1 parity, one card, bitwise (GDN rr kernel incl. graph replay; QSA compile variants) -> FAIL stops
#   2 P0 (diagnostic): GDN served vs rr per shape; QSA variants per shape -> the QT choice
#   3 model parity, both cards: OFF, OFF (A/A), ALL at b1d3, b4d3, b8d1; logits hashes -> FAIL stops (if A/A holds)
#   4 nsys c1d3 and c8d1, OFF vs ALL, harness --worker (for off-box chain_kernels.py / segment analysis)
#   5 P1 untraced harness, 4k, rotated rounds at c4d3, c8d1, c1d3; sequence hashes vs OFF per round (both cells)
#   6 P1 summary + per-arm speed verdict
#   7 fn_greedy on the served launcher: ref (daily) vs $IMG ON (the speed-accepted flags; ALL if none) vs $IMG OFF
#   8 FINAL decision per arm
# GPU ~3 h at the default 6 arms x 6 rounds x 3 shapes (estimate: r703 ran 50 P1 runs + extras in ~70 min).
# Queue-chained; the daily is restored at the end. This unit does not promote.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
RN=${RN:-r712}
R=${R:-/srv/qwen5090/results/$(date +%F)-$RN-latchain-r1}; mkdir -p "$R"
LC=${LC:-/srv/qwen5090/patches/exllamav3/latchain-r1}
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
HARNESS=/srv/qwen5090/probes/r465/profile_decode_events.py
METER=/srv/qwen5090/probes/r465/events_meter.py
GREEDY=/srv/qwen5090/probes/fn_greedy.py
NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
BASE=${BASE:-tabbyapi:stack-r2}
IMG=${IMG:-tabbyapi:stack-latchain-r1}
log(){ echo "$(date -Is) [latchain-r1] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$RN-latchain-r1
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ sudo docker rm -f latchain-p latchain-p1 latchain-m latchain-n >/dev/null 2>&1 || true
  sudo docker logs flashnext > "$R/container-final.log" 2>&1 || true
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== latchain r1 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$HARNESS" "$METER" "$GREEDY" "$LC/Dockerfile.box" "$LC/latchain-r1.patch" "$LC/lc_model_parity.py"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
sudo docker image inspect "$BASE" >/dev/null 2>&1 || { log "ABORT: base image $BASE missing"; exit 3; }
grep -qx "DAILY_IMG=$BASE" "$LIVE" || log "WARNING: $BASE is not the launcher's DAILY_IMG ($(grep -m1 '^DAILY_IMG=' "$LIVE"))"
EXTRA=$(sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-([^}]*)\}.*/\1/p' "$LIVE"); ENVS=""; for kv in $EXTRA; do ENVS="$ENVS -e $kv"; done
echo "$EXTRA" | tr ' ' '\n' | grep -qx 'EXL3_GDN_STATE_BF16=1' || { log "ABORT: live env lacks EXL3_GDN_STATE_BF16=1"; exit 3; }
echo "$EXTRA" | tr ' ' '\n' | grep -q '^EXL3_LC_' && { log "ABORT: live env already sets an EXL3_LC_ key"; exit 3; }

# ---- arms: every LC key explicit in every arm (0 unless the arm sets it) ----
LC_KEYS="EXL3_LC_GDN_RR EXL3_LC_QSA_FORK EXL3_LC_GDN_FORK EXL3_LC_QSA_SPLIT_STAGES EXL3_LC_QSA_COMBINE_STAGES EXL3_LC_QSA_DIV16"
declare -A ARMF=([OFF]="" [RR]="EXL3_LC_GDN_RR=1" [QF]="EXL3_LC_QSA_FORK=1" [GF]="EXL3_LC_GDN_FORK=1" [QT]="")
arm_set(){ if [ "$1" = ALL ]; then echo "${ARMF[RR]} ${ARMF[QF]} ${ARMF[GF]} ${ARMF[QT]}"; else echo "${ARMF[$1]}"; fi; }
lc_env(){   # "K=V ..." for every LC key; args: arm names whose settings are unioned
  local k kv a; declare -A v=(); for k in $LC_KEYS; do v[$k]=0; done
  for a in "$@"; do for kv in $(arm_set "$a"); do v[${kv%%=*}]=${kv#*=}; done; done
  for k in $LC_KEYS; do printf '%s=%s ' "$k" "${v[$k]}"; done; }
flags_of(){ local kv fe=""; for kv in $(lc_env "$@"); do fe="$fe -e $kv"; done; echo "$fe"; }

gpu_lock
log "lock held; served at entry: $(served_id || echo none); BASE $BASE; IMG $IMG"

# ---- 0. build + landing ----
if ! sudo docker image inspect "$IMG" >/dev/null 2>&1; then
  log "building $IMG on $BASE"
  ( cd "$LC" && sudo docker build -f Dockerfile.box --build-arg BASE="$BASE" --build-arg MAX_JOBS=4 -t "$IMG" . ) > "$R/build.log" 2>&1 \
    || { log "BUILD FAILED"; tail -30 "$R/build.log" | sed 's/^/  /' | tee -a "$R/audit.log"; finish BUILD-FAILED; exit 3; }
fi
grep -aE "latchain r1 landed|CPU PASS|CPU FAIL" "$R/build.log" 2>/dev/null | sed 's/^/  [build] /' | tee -a "$R/audit.log"
sudo docker run --rm --entrypoint python3 "$IMG" -c '
import torch, exllamav3_ext as e
assert e.latchain_revision == 1 and e.moe_coop_v3_revision == 1
assert hasattr(e.TritonKernel, "launch_py")
import exllamav3.modules.attention_fn.bc_attn as b
assert b.LC_BUILD == "latchain-r1" and b._LC_QSA_SPLIT_STAGES == 0 and b._LC_QSA_COMBINE_STAGES == 0 and not b._LC_QSA_DIV16
print("latchain", e.latchain_revision, "moefast", e.moe_coop_v3_revision, b.LC_BUILD)' > "$R/landed.txt" 2>&1 \
  || { log "ABORT: landing check failed: $(tail -2 "$R/landed.txt")"; finish ABORTED; exit 3; }
log "image $IMG: $(cat "$R/landed.txt")"

served_stop; wait_unserved 45
one(){ sudo docker run --rm --name latchain-p --gpus '"device=0"' --ipc=host -v /srv/qwen5090/models:/models:ro -v "$R":/results \
  -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache \
  $ENVS $(flags_of OFF) -e CUDA_VISIBLE_DEVICES=0 --entrypoint python3 "$IMG" "$@"; }

# ---- 1. parity ----
one /opt/latchain-r1/test_latchain_parity.py --json /results/parity.json > "$R/parity.log" 2>&1
rc=$?
log "parity rc=$rc: $(grep -aE 'PARITY (PASS|FAIL)|comparisons equal' "$R/parity.log" | tail -2 | tr '\n' ' ')"
grep -aq "PARITY PASS" "$R/parity.log" && [ $rc = 0 ] || { grep -aE "MISMATCH|Error|error" "$R/parity.log" | head -20 | sed 's/^/  /' | tee -a "$R/audit.log"; finish PARITY-FAIL; exit 5; }

# ---- 2. P0 (diagnostic) + QT choice ----
one /opt/latchain-r1/bench_latchain_p0.py --out /results/p0 > "$R/p0.log" 2>&1
log "P0 rc=$?"
grep -aE "^\[p0\] (GDN|QSA c[148]d[013] best|QT)" "$R/p0.log" | sed 's/^/  /' | tee -a "$R/audit.log"
QT=$(sed -nE 's/^\[p0\] QT choice (EXL3_LC_QSA_SPLIT_STAGES=[0-9] EXL3_LC_QSA_COMBINE_STAGES=[0-9] EXL3_LC_QSA_DIV16=[01])$/\1/p' "$R/p0.log" | tail -1)
ARMS=${ARMS:-}
if [ -z "$ARMS" ]; then
  if [ -n "$QT" ]; then ARMF[QT]="$QT"; ARMS="OFF RR QF GF QT ALL"; else ARMS="OFF RR QF GF ALL"; fi
elif [ -n "$QT" ]; then ARMF[QT]="$QT"; fi
case " $ARMS " in *" QT "*) [ -n "${ARMF[QT]}" ] || { log "ARMS names QT but P0 chose none: dropping QT"; ARMS=$(echo " $ARMS " | sed 's/ QT / /'); } ;; esac
set -- $ARMS; NARM=$#
k=$NARM; while [ $k -lt 5 ]; do k=$((k + NARM)); done
ROUNDS=${ROUNDS:-$k}
[ $((ROUNDS % NARM)) = 0 ] && [ "$ROUNDS" -ge 5 ] || log "WARNING: ROUNDS=$ROUNDS with $NARM arms: first-position rotation not balanced / < 5 pairs"
log "arms: $ARMS (QT: '${ARMF[QT]:-none}'); rounds $ROUNDS; ALL = $(lc_env ALL)"

# ---- 3. model parity (logits hashes), both cards ----
mp(){ local tag=$1 arm=$2 b=$3 d=$4
  sudo docker run --rm --name latchain-m --gpus all --ipc=host --shm-size=16g \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v "$R":/results $ENVS $(flags_of $arm) \
    --entrypoint python3 "$IMG" /opt/latchain-r1/lc_model_parity.py --model /models/$MODEL \
    --batch $b --draft $d --out /results/mp-$tag.json > "$R/mp-$tag.log" 2>&1
  log "  model parity $tag rc=$?: $(grep -aE '^lc_model_parity' "$R/mp-$tag.log" | tail -1 | cut -c1-160)"; }
MPFAIL=0; MPAA=1
for shape in "b1d3 1 3" "b4d3 4 3" "b8d1 8 1"; do set -- $shape
  mp $1-OFF OFF $2 $3; mp $1-ALL ALL $2 $3; mp $1-OFF2 OFF $2 $3
  python3 "$LC/lc_model_parity.py" --compare "$R/mp-$1-OFF.json" "$R/mp-$1-OFF2.json" "$R/mp-$1-ALL.json" > "$R/mp-$1-compare.txt" 2>&1
  sed "s/^/  [mp $1] /" "$R/mp-$1-compare.txt" | tee -a "$R/audit.log"
  grep -q "aa=IDENTICAL" "$R/mp-$1-compare.txt" || MPAA=0
  grep -q "aa=IDENTICAL arms=DIFFERENT" "$R/mp-$1-compare.txt" && MPFAIL=1
done
if [ $MPFAIL = 1 ]; then log "MODEL PARITY FAIL: ALL differs from OFF where OFF/OFF is deterministic"; finish MODEL-PARITY-FAIL; exit 5; fi
[ $MPAA = 1 ] && log "model parity PASS (logits-level, all three shapes)" \
  || log "model parity: an OFF/OFF A/A differs -> logits-level identity unusable on this build; token-level gates only"
echo "$MPFAIL $MPAA" > "$R/model-parity.status"

# ---- 4. nsys timelines (for off-box segment analysis) ----
run_harness(){ local name=$1 arm=$2; shift 2   # remaining args: entrypoint + argv
  sudo docker run --rm --name "$name" --gpus all --ipc=host --shm-size=16g \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v "$HARNESS":/probe/profile_decode_events.py:ro \
    -v "$METER":/probe/events_meter.py:ro -v "$R":/results $ENVS $(flags_of $arm) \
    --entrypoint "$@"; }
if sudo docker run --rm --entrypoint test "$IMG" -x "$NSYS"; then
  for shape in "c1d3 1 3" "c8d1 8 1"; do set -- $shape
    for a in OFF ALL; do
      run_harness latchain-n $a "$NSYS" "$IMG" profile -t cuda,nvtx --cuda-graph-trace=node -o /results/nsys-$1-$a \
        --force-overwrite true --export sqlite \
        python3 /probe/profile_decode_events.py --model /models/$MODEL --out /results/nsys-run-$1-$a --worker \
        --cache-quant 8,8 --gpu-split 30,30 --max-chunk-size 2048 --tokens 64 \
        --warmup-steps 16 --settle-steps 8 --capture-steps 32 --capture-mode wall \
        --contexts 4096 --batch $2 --draft $3 > "$R/nsys-$1-$a.log" 2>&1
      log "  nsys $1 $a rc=$?: $(ls -la "$R"/nsys-$1-$a.sqlite 2>/dev/null | awk '{print $5}') bytes"
    done
  done
else
  log "nsys not in $IMG: timeline step skipped"
fi

# ---- 5. P1 in-process, rotated rounds ----
cell(){ grep -ahE 'ms/iterate' "$1/kernels.txt" 2>/dev/null | head -1; }
p1(){ local tag=$1 arm=$2 b=$3 d=$4
  run_harness latchain-p1 $arm python3 "$IMG" /probe/profile_decode_events.py --model /models/$MODEL \
    --out /results/p1-$tag --cache-quant 8,8 --gpu-split 30,30 --max-chunk-size 2048 --tokens 64 \
    --warmup-steps 16 --settle-steps 8 --capture-steps 32 --capture-mode events \
    --contexts 4096 --batch $b --draft $d > "$R/p1-$tag.log" 2>&1
  log "  P1 $tag rc=$?: d$d $(cell "$R/p1-$tag/ctx4096_b${b}_d${d}") | d0 $(cell "$R/p1-$tag/ctx4096_b${b}_d0")"; }
rounds(){ local sh=$1 b=$2 d=$3 r i a dd order; local -a arr=($ARMS); local n=${#arr[@]}
  for r in $(seq 1 $ROUNDS); do
    order=""; for i in $(seq 0 $((n - 1))); do order="$order ${arr[$(( (r - 1 + i) % n ))]}"; done
    log "  $sh round $r order:$order"
    for a in $order; do p1 $sh-$a$r $a $b $d; done
    for a in "${arr[@]}"; do [ $a = OFF ] && continue
      for dd in $d 0; do
        f0="$R/p1-$sh-OFF$r/ctx4096_b${b}_d${dd}/sequence-hashes.json"; f1="$R/p1-$sh-$a$r/ctx4096_b${b}_d${dd}/sequence-hashes.json"
        if [ -s "$f0" ] && cmp -s "$f0" "$f1"; then log "  $sh round $r $a d$dd: hashes IDENTICAL"
        else log "  $sh round $r $a d$dd: hashes DIFFER -> FAIL"; fi
      done
    done
  done; }
rounds c4d3 4 3
rounds c8d1 8 1
rounds c1d3 1 3

# ---- 6. P1 summary + speed verdicts ----
SUMMARY=$(cat <<'EOF'
import json, re, sys, statistics as st
R, rounds, arms = sys.argv[1], int(sys.argv[2]), sys.argv[3].split()
SH = {"c1d3": (1, 3), "c4d3": (4, 3), "c8d1": (8, 1)}
audit = open(f"{R}/audit.log").read()
def ms(tag, b, d):
    try:
        t = open(f"{R}/p1-{tag}/ctx4096_b{b}_d{d}/kernels.txt").read()
        return float(re.search(r"([0-9.]+) ms/iterate", t).group(1))
    except Exception:
        return None
def pairs(arm, sh, b, d):
    ds, base = [], []
    for r in range(1, rounds + 1):
        x, y = ms(f"{sh}-{arm}{r}", b, d), ms(f"{sh}-OFF{r}", b, d)
        if x is not None and y is not None:
            ds.append(x - y); base.append(y)
    return ds, base
def sign(ds):
    return "GAIN" if ds and all(v < 0 for v in ds) else "REGRESSION" if ds and all(v > 0 for v in ds) else "flat"
out = {}
for arm in arms:
    if arm == "OFF":
        continue
    v = {"shapes": {}, "d0": {}, "reasons": []}
    for sh, (b, d) in SH.items():
        ds, base = pairs(arm, sh, b, d)
        s = sign(ds)
        mean = st.mean(ds) if ds else None
        pct = 100 * mean / st.mean(base) if ds else None
        v["shapes"][sh] = {"n": len(ds), "deltas_ms": ds, "mean_ms": mean, "mean_pct": pct, "sign": s}
        print(f"P1 {sh} {arm}-OFF ms: " + " ".join(f"{x:+.2f}" for x in ds)
              + (f"  mean {mean:+.3f} ({pct:+.2f} %) {s}  [n={len(ds)}]" if ds else "  NO DATA"))
        d0s, b0 = pairs(arm, sh, b, 0)
        s0 = sign(d0s)
        v["d0"][f"b{b}"] = {"n": len(d0s), "deltas_ms": d0s, "mean_pct": 100 * st.mean(d0s) / st.mean(b0) if d0s else None, "sign": s0}
        print(f"P1 d0 ctx4096_b{b}_d0 {arm}-OFF ms: " + " ".join(f"{x:+.3f}" for x in d0s)
              + (f"  mean {st.mean(d0s):+.4f} ({100 * st.mean(d0s) / st.mean(b0):+.2f} %) {s0}" if d0s else "  NO DATA"))
    S = v["shapes"]
    hashes_bad = len(re.findall(rf" round \d+ {arm} d\d: hashes DIFFER", audit))
    hashes_ok = len(re.findall(rf" round \d+ {arm} d\d: hashes IDENTICAL", audit))
    v["hashes_identical"], v["hashes_differ"] = hashes_ok, hashes_bad
    if hashes_bad or hashes_ok < 2 * 3 * rounds:
        v["reasons"].append(f"hashes {hashes_ok} identical / {hashes_bad} differ (need {2 * 3 * rounds} identical)")
    if any(S[s]["n"] < 5 for s in SH):
        v["reasons"].append("fewer than 5 pairs at some shape: " + ", ".join(f"{s} n={S[s]['n']}" for s in SH))
    for s in ("c4d3", "c8d1"):
        if S[s]["sign"] == "REGRESSION":
            v["reasons"].append(f"same-sign regression at {s} ({S[s]['mean_pct']:+.2f} %)")
    gains = [s for s in SH if S[s]["sign"] == "GAIN"]
    if S["c1d3"]["sign"] == "REGRESSION":
        big = [s for s in ("c4d3", "c8d1") if S[s]["sign"] == "GAIN"]
        best_ms = max((abs(S[s]["mean_ms"]) for s in big), default=0.0)
        if not (S["c1d3"]["mean_pct"] <= 2.0 and big and best_ms > S["c1d3"]["mean_ms"]):
            v["reasons"].append(f"c1d3 same-sign regression {S['c1d3']['mean_pct']:+.2f} % / {S['c1d3']['mean_ms']:+.3f} ms "
                                f"not covered (bound 2 %, c4/c8 GAIN larger in ms: {big or 'none'} {best_ms:.3f} ms)")
        else:
            v["c1_cost"] = f"c1d3 {S['c1d3']['mean_pct']:+.2f} % (record in STACK.md)"
    if not gains:
        v["reasons"].append("gain clause: no shape GAIN (all flat) -> not a stack entry")
    v["speed_ok"] = not v["reasons"]
    out[arm] = v
    print(f"P1-VERDICT {arm}: {'SPEED+HASHES OK' if v['speed_ok'] else 'NO'}"
          + (f" ({'; '.join(v['reasons'])})" if v["reasons"] else "") + (f" [{v['c1_cost']}]" if "c1_cost" in v else ""))
json.dump(out, open(f"{R}/p1-verdicts.json", "w"), indent=1)
EOF
)
python3 -c "$SUMMARY" "$R" "$ROUNDS" "$ARMS" 2>&1 | tee "$R/p1-summary.txt" | tee -a "$R/audit.log"

# ---- 7. greedy identity on the served launcher (the union of the speed-accepted flag arms; ALL if none) ----
ACC=""; for a in $ARMS; do case $a in OFF|ALL) ;; *) grep -q "^P1-VERDICT $a: SPEED+HASHES OK" "$R/p1-summary.txt" && ACC="$ACC $a" ;; esac; done
if [ -n "$ACC" ]; then ONENV=$(lc_env $ACC); else ONENV=$(lc_env ALL); fi
OFFENV=$(lc_env OFF)
log "greedy ON config (accepted arms:${ACC:- none -> ALL}): $ONENV"
G=$R/greedy.jsonl
boot(){ served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "$@" bash "$LIVE" > "$R/boot-$(date +%s).log" 2>&1 \
    && wait_served_id "$MODEL" 200 8; }
used0(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' '; }
boot || { log "daily NO BOOT"; finish NO-BOOT; exit 3; }
python3 "$GREEDY" --url http://127.0.0.1:8022 --tag ref --out "$G" > "$R/greedy-ref.log" 2>&1
U_REF=$(used0); log "  ref boot: cuda:0 used ${U_REF} MiB"
U_ON=""
for a in on off; do
  if [ $a = on ]; then boot IMG=$IMG EXTRA_ENV_ADD="$ONENV"; else boot IMG=$IMG EXTRA_ENV_ADD="$OFFENV"; fi \
    || { log "$a NO BOOT"; continue; }
  log "  greedy $a: $(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'); LC env: $(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^EXL3_LC_' | tr '\n' ' ')"
  python3 "$GREEDY" --url http://127.0.0.1:8022 --tag $a --out "$G" > "$R/greedy-$a.log" 2>&1
  [ $a = on ] && { U_ON=$(used0); log "  on boot: cuda:0 used ${U_ON} MiB (ref ${U_REF})"; }
  sudo docker logs flashnext > "$R/container-greedy-$a.log" 2>&1
done
python3 "$GREEDY" --compare --ref ref --out "$G" > "$R/greedy-compare.txt" 2>&1
grep -aE "GREEDY (on|off) vs ref|GREEDY-SUMMARY|DIVERGE|MISSING" "$R/greedy-compare.txt" | sed 's/^/  [greedy] /' | tee -a "$R/audit.log"

# ---- 8. FINAL ----
FINAL=$(cat <<'EOF'
import json, re, sys
R, acc, u_ref, u_on, mp = sys.argv[1], sys.argv[2].split(), sys.argv[3], sys.argv[4], sys.argv[5].split()
v = json.load(open(f"{R}/p1-verdicts.json"))
g = open(f"{R}/greedy-compare.txt").read()
def div(tag):
    m = re.search(rf"^GREEDY {tag} vs ref: (\d+) identical, (IDENTICAL|(\d+) DIVERGENT)", g, re.M)
    return None if not m else (0 if m.group(2) == "IDENTICAL" else int(m.group(3)))
don, doff = div("on"), div("off")
greedy_ok = don == 0 and doff == 0
head_ok = u_ref.isdigit() and u_on.isdigit() and int(u_on) <= int(u_ref) + 32
parity_ok = "PARITY PASS" in open(f"{R}/parity.log", errors="replace").read()
mp_fail = mp[0] == "1"
print(f"FINAL identity: parity {'PASS' if parity_ok else 'FAIL'}; model parity {'FAIL' if mp_fail else 'ok' if mp[1] == '1' else 'A/A nondeterministic (token-level only)'}; "
      f"greedy on {don} / off {doff} divergences; headroom cuda:0 used on {u_on or '?'} vs ref {u_ref or '?'} MiB "
      f"({'ok' if head_ok else 'CHECK'}: bar ref + 32)")
entries = []
for arm, x in v.items():
    if arm == "ALL":
        print(f"FINAL ALL (composition, not an entry): " + ", ".join(
            f"{s} {x['shapes'][s]['mean_pct']:+.2f} % {x['shapes'][s]['sign']}" for s in x["shapes"] if x["shapes"][s]["n"]))
        continue
    reasons = list(x["reasons"])
    if not parity_ok: reasons.append("parity FAIL")
    if mp_fail: reasons.append("model parity FAIL")
    if arm in acc and not greedy_ok: reasons.append(f"greedy divergences on={don} off={doff}")
    if arm in acc and not head_ok: reasons.append("headroom above ref + 32 MiB (check the ON boot)")
    ok = not reasons and arm in acc
    line = ", ".join(f"{s} {x['shapes'][s]['mean_pct']:+.2f} % {x['shapes'][s]['sign']}" for s in x["shapes"] if x["shapes"][s]["n"])
    print(f"FINAL {arm}: {'ACCEPT (stack entry)' if ok else 'REJECT'} [{line}]" + (f" ({'; '.join(reasons)})" if reasons else "")
          + (f" [{x['c1_cost']}]" if "c1_cost" in x else ""))
    if ok: entries.append(arm)
print("FINAL stack entries: " + (" ".join(entries) if entries else "none"))
EOF
)
python3 -c "$FINAL" "$R" "$ACC" "${U_REF:-}" "${U_ON:-}" "$(cat "$R/model-parity.status")" 2>&1 | tee "$R/final.txt" | tee -a "$R/audit.log"
finish DONE
