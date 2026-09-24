#!/usr/bin/env bash
# R716b (2026-09-24): R716 re-queued after its BUILD-FAILED: landing.py asserted no EXL3_* env, but tabbyapi:stack-r2 bakes 6 EXL3_* keys into its image ENV; landing.py now re-execs itself with them stripped.
# stack-r3 batch gate (2026-09-24; OPERATIONS §16 incl. the R705 / R709c / R712 / R710b review rules, pre-registered).
# The operator assigns the R number: install out-stack-r3/ as /srv/qwen5090/patches/exllamav3/stack-r3 and this file as
# /srv/qwen5090/rNNN-stack-r3.sh, then queue rNNN-stack-r3 (UNIT = the file name).
#
# Image tabbyapi:stack-r3 = tabbyapi:stack-r2 (the daily) + ONE patch series (series/SERIES), ONE rebuild:
#   core  hcfast r1 -> r2, latchain-r1b (latchain r1 re-anchored on bindings.cpp; code identical)
#   mf3   moefast-r3 (R713)                       INCLUDE contains mf3
#   dg1   densegemm-r1 (R710b gemv twin)          INCLUDE contains dg1   } mutually exclusive; dg2 supersedes dg1
#   dg2   densegemm-r2 (R714, DGV2_NVCC_DEFS)     INCLUDE contains dg2   } if R714 passes
#   auto  dense-lcguard with dg1/dg2: latchain's QSA side-branch dense calls get their own V2 scratch (upper half of
#         the gemv hctr counters, separate gemm slots), so EXL3_LC_QSA_FORK=1 and EXL3_DENSE_V2 can both be on
# Components of the union (families of keys; the served stack-r2 env is kept except the families a config sets):
#   HC2  EXL3_HC_MIX_V3=2 + R702 P0 RECOMMEND tables (replaces the served HC1 DOTS_B=2 UP_B=8)      fixed (R702)
#   RR   EXL3_LC_GDN_RR=1                                                                           fixed (R712)
#   QT   EXL3_LC_QSA_SPLIT_STAGES=2 _COMBINE_STAGES=1 _DIV16=1                                      carried (R712)
#   QF   EXL3_LC_QSA_FORK=1                                                                         carried (R712)
#   DG   EXL3_DENSE_V2=3 (dg1) | =1 (dg2: every r2 twin; = GM + GV)                                 carried (R710b / R714)
#   MF3  EXL3_MOE_COOP_V3=3 EXL3_MOE_COOP_V3_MAP=2-4:2 EXL3_SHARED_EXPERT_EARLY=1 (or MF3_ENV = R713's R3 line)                optional (R713)
#   GF is never carried (R712: marginal ~0) and aborts the unit if named.
# UNION (pre-registered default) = HC2 RR QT QF DG [MF3]. UNION_PRESET=noqf is the pre-registered fallback without QF
# (use it if the image cannot carry the guard). UNION="K=V ..." overrides both and is logged.
#
# Steps:
#   0 build $IMG in-lock BEFORE any measurement (§15) when missing or its labels differ (base, INCLUDE, series sha,
#     defines); the install checks served-SASS identity, the STACK bound, markers, and the CPU tests. Landing check
#     (tools/landing.py: torch first, every marker, flags default off, HC2 tables parse); guard marker when QF + DG.
#   1 kernel parity, one card, for every included patch: hcfast r2, latchain, [moefast r3], [densegemm (mode of DG)]
#   2 model parity (logits hashes), both cards: OFF, U, OFF2 at b1d3, b4d3, b8d1. A/A identical and U different = FAIL
#   3 P1 in-process harness, 4k, untraced: arms OFF U Uno<X>... OFF2 (Uno<X> = the union without optional X; padded to
#     6 arms with UnoRR / UnoHC2, information only), rounds = the smallest multiple of the arm count >= 6, order rotated so every arm runs first equally often, at c4d3, c8d1,
#     c1d3 + the draft-0 cells; sequence hashes == the round's OFF in the d and d0 cells. tools/p1_stack.py reads raw,
#     per-iterate median and stall-excluded paired means (iterates > median + 8 ms dropped from both runs; count shown)
#   4 served: boots A1 B1 A2 B2 A3 B3 C, NVME_TIER= on every boot (§16 R709c: the launcher enables the tier for
#     IMG = DAILY_IMG). A = the daily; B = $IMG + full EXTRA_ENV override (served env minus the union's families + the
#     union); C = $IMG with the served env (flags off). Per boot: UP-line free VRAM, container env (EXL3_*), greedy_streams
#     c2/c4/c8 (distinct salted prompts, min_tokens, full texts), fn_greedy (incl. the long prompt), ramp c1..c8
#     (fn_bench --distinct), canonical fn_gate RUNS=3 (A/B boots only), container log.
#     B1 identity check right after its probes: a divergence triggers one more daily boot (A/A); if the A/A is identical,
#     one greedy boot per optional component without it (LOO) names the culprit and the unit stops before the ABAB.
# DECISION (pre-registered; this unit does not promote):
#   identity  every parity PASS; model parity not FAIL; P1 hashes identical everywhere (all arms, d and d0); fn_greedy
#             B and C vs A1: 0 divergences (if an A/A boot A2 / A3 diverges from A1 the served identity is VOID: re-run,
#             not a reject); greedy_streams B and C vs A1:
#             strict when the A/A boots are identical at that concurrency, BOUNDED (never worse than the A/A) otherwise
#   P1        P1-VERDICT UNION PASS (no same-sign regression at c4d3/c8d1 on excl or med; c1 bound; >= one GAIN on excl)
#   served    tools/served_stack.py: headroom UP-line free >= min(A) - 32 MiB on both cards for every B and C boot;
#             0 OOM / TORCH_CHECK / traceback; ramp and leg B all ok; union keys present, no duplicate keys; guard line
#             in every B container log when QF and DG are both on AND the guard engaged in-process (else a caveat); fn_gate 3 pairs: no cell mean ON/OFF < 0.99 except
#             leg-A c1 >= 0.98, aggregate > 1.000
#   components  HC2 and RR enter with the union. Optional X enters iff the union passes AND its P1 marginal (U - Uno<X>)
#             is KEEP. If P1 marks exactly one X DROP, the served boots run the union without X (UNION-X, measured in P1
#             as Uno<X>); two or more DROPs stop the unit before the served steps with the suggested UNION.
# GPU ~3.5-4.5 h: build 15-25 min CPU in-lock if missing (the daily keeps serving); kernel parity ~20-30; model parity
# ~15; P1 108 runs ~50-55 (147 with mf3 ~70); served 7 boots x ~17-20 min ~2 h (boot 3, greedy_streams 2-3, fn_greedy
# incl. the 100k prompt ~4, ramp ~3, fn_gate RUNS=3 ~7-8); +1 A/A boot and up to 5 LOO boots only on a B1 divergence.
# Queue-chained; the daily is restored at the end.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
UNIT=${UNIT:-$(basename "$0" .sh)}
R=${R:-/srv/qwen5090/results/$(date +%F)-$UNIT}; mkdir -p "$R"
S3=${S3:-/srv/qwen5090/patches/exllamav3/stack-r3}
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
HARNESS=/srv/qwen5090/probes/r465/profile_decode_events.py
METER=/srv/qwen5090/probes/r465/events_meter.py
GREEDY=/srv/qwen5090/probes/fn_greedy.py
GATE=/srv/qwen5090/probes/fn_gate.sh
BENCH=/srv/qwen5090/probes/fn_bench.py
GSTREAMS=$S3/tools/greedy_streams.py
P1S=$S3/tools/p1_stack.py
SERVED=$S3/tools/served_stack.py
R702P0=${R702P0:-/srv/qwen5090/results/2026-09-24-r702-hcfast-r2/p0.json}
IMG=${IMG_S3:-tabbyapi:stack-r3}
INCLUDE=$(echo "${INCLUDE:-mf3 dg2}" | tr ',' ' ' | xargs)
DGV2_NVCC_DEFS=${DGV2_NVCC_DEFS:-}
has(){ case " $INCLUDE " in *" $1 "*) return 0 ;; esac; return 1; }
HC2_ENV=${HC2_ENV:-EXL3_HC_MIX_V3=2 EXL3_HC_MIX_V3_DOTS_B=1:1,4:2,32:4 EXL3_HC_MIX_V3_UP_B=1:1,8:4,32:8 EXL3_HC_MIX_V3_DOTS_J=1:4,32:8 EXL3_HC_MIX_V3_DOTS_PF=1:1,32:0 EXL3_HC_MIX_V3_UP_Q=1:4,8:2,32:4 EXL3_HC_MIX_V3_PDL=0}
RR_ENV=${RR_ENV:-EXL3_LC_GDN_RR=1}
QT_ENV=${QT_ENV:-EXL3_LC_QSA_SPLIT_STAGES=2 EXL3_LC_QSA_COMBINE_STAGES=1 EXL3_LC_QSA_DIV16=1}
QF_ENV=${QF_ENV:-EXL3_LC_QSA_FORK=1}
if [ -z "${DG_ENV+x}" ]; then if has dg2; then DG_ENV=EXL3_DENSE_V2=1; elif has dg1; then DG_ENV=EXL3_DENSE_V2=3; else DG_ENV=; fi; fi
if [ -z "${MF3_ENV+x}" ]; then if has mf3; then MF3_ENV="EXL3_MOE_COOP_V3=3 EXL3_MOE_COOP_V3_MAP=2-4:2 EXL3_SHARED_EXPERT_EARLY=1"; else MF3_ENV=; fi; fi
UNION_PRESET=${UNION_PRESET:-default}
P1_HARNESS_ARGS=${P1_HARNESS_ARGS:-}   # e.g. "--gc-mode freeze" if R715 shows it removes the ~34 ms iterate stall (needs that harness)
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$UNIT
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ sudo docker rm -f s3-p s3-m s3-p1 >/dev/null 2>&1 || true
  sudo docker logs flashnext > "$R/container-final.log" 2>&1 || true
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== stack-r3 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$HARNESS" "$METER" "$GREEDY" "$GATE" "$BENCH" "$GSTREAMS" "$P1S" "$SERVED" "$S3/Dockerfile.box" \
         "$S3/install-stack.sh" "$S3/series/SERIES" "$S3/series/apply-series.sh" "$S3/tools/landing.py"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done

# ---- configuration: families, union, arms ----
fam(){ case "${1%%=*}" in
  EXL3_HC_MIX_V3|EXL3_HC_MIX_V3_*) echo HC2 ;;
  EXL3_LC_GDN_RR) echo RR ;;
  EXL3_LC_QSA_SPLIT_STAGES|EXL3_LC_QSA_COMBINE_STAGES|EXL3_LC_QSA_DIV16) echo QT ;;
  EXL3_LC_QSA_FORK) echo QF ;;
  EXL3_LC_GDN_FORK) echo GF ;;
  EXL3_DENSE_V2|EXL3_DENSE_ROWS32) echo DG ;;
  EXL3_MOE_COOP_V3|EXL3_MOE_COOP_V3_MAP|EXL3_MOE_COOP_V3_HEAD|EXL3_MOE_COOP_V3_L2EF|EXL3_SHARED_EXPERT_EARLY|EXL3_SHARED_EXPERT_PRIO) echo MF3 ;;
  *) echo - ;; esac; }
comp_of(){ local X=$1 kv; for kv in $2; do [ "$(fam "$kv")" = "$X" ] && printf '%s ' "$kv"; done; }
minus(){ local X=$1 kv; for kv in $2; do [ "$(fam "$kv")" = "$X" ] || printf '%s ' "$kv"; done; }
# the served env with every family the flags set removed, then the flags (no reliance on docker's last-wins)
arm_env(){ local flags=$1 kv f fams=" " out=""
  for kv in $flags; do fams="$fams$(fam "$kv") "; done
  for kv in $EXTRA; do f=$(fam "$kv"); if [ "$f" != - ]; then case "$fams" in *" $f "*) continue ;; esac; fi; out="$out $kv"; done
  echo $out $flags; }
dflags(){ local kv fe=""; for kv in $(arm_env "$1"); do fe="$fe -e $kv"; done; echo "$fe"; }

BASE=$(grep -m1 -oE '^DAILY_IMG=[^ ]+' "$LIVE" | cut -d= -f2 | tr -d "'\"")
EXTRA=$(sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-([^}]*)\}.*/\1/p' "$LIVE")
[ -n "$BASE" ] && [ -n "$EXTRA" ] || { log "ABORT: could not read DAILY_IMG / EXTRA_ENV from $LIVE"; exit 3; }
[ "$BASE" = tabbyapi:stack-r2 ] || { log "ABORT: DAILY_IMG is $BASE; the series was made against tabbyapi:stack-r2"; exit 3; }
for kv in EXL3_GDN_STATE_BF16=1 EXL3_HC_MIX_V3=1 EXL3_MOE_COOP_V3=2; do
  echo "$EXTRA" | tr ' ' '\n' | grep -qx "$kv" || { log "ABORT: live env lacks $kv (not the stack-r2 daily)"; exit 3; }; done
echo "$EXTRA" | tr ' ' '\n' | grep -qE '^EXL3_(LC_|DENSE_)' && { log "ABORT: live env already sets an EXL3_LC_ / EXL3_DENSE_ key"; exit 3; }
if has dg1 && has dg2; then log "ABORT: INCLUDE names dg1 and dg2 (mutually exclusive)"; exit 3; fi
has dg2 || [ -z "$DGV2_NVCC_DEFS" ] || { log "ABORT: DGV2_NVCC_DEFS without dg2"; exit 3; }

if [ -n "${UNION:-}" ]; then USRC="override"
else
  UNION=$(echo $HC2_ENV $RR_ENV $QT_ENV $QF_ENV $DG_ENV $MF3_ENV)
  case $UNION_PRESET in default) USRC="pre-registered default" ;; noqf) UNION=$(echo $(minus QF "$UNION")); USRC="pre-registered fallback (no QF)" ;;
    *) log "ABORT: UNION_PRESET=$UNION_PRESET (default | noqf)"; exit 3 ;; esac
fi
UNION=$(echo $UNION)
for kv in $UNION; do
  case "$(fam "$kv")" in -) log "ABORT: UNION key $kv belongs to no stack-r3 component"; exit 3 ;; GF) log "ABORT: GF ($kv) is not carried (R712)"; exit 3 ;; esac
done
dupk=$(for kv in $UNION; do echo "${kv%%=*}"; done | sort | uniq -d | xargs)
[ -z "$dupk" ] || { log "ABORT: UNION sets $dupk twice"; exit 3; }
val(){ local kv; for kv in $UNION; do [ "${kv%%=*}" = "$1" ] && { echo "${kv#*=}"; return; }; done; echo 0; }
PRESENT=""; for X in HC2 RR QT QF DG MF3; do [ -n "$(comp_of $X "$UNION")" ] && PRESENT="$PRESENT $X"; done
OPTS=""; for X in QT QF DG MF3; do case " $PRESENT " in *" $X "*) OPTS="$OPTS $X" ;; esac; done
PRESENT=$(echo $PRESENT); OPTS=$(echo $OPTS)
for X in HC2 RR; do case " $PRESENT " in *" $X "*) ;; *) log "WARNING: fixed component $X is not in UNION ($USRC)" ;; esac; done
DGV=$(val EXL3_DENSE_V2); R32=$(val EXL3_DENSE_ROWS32); QFV=$(val EXL3_LC_QSA_FORK)
[ "$R32" = 0 ] || { log "ABORT: EXL3_DENSE_ROWS32 must be off (no served decode shape reaches 17 rows)"; exit 3; }
if [ "$DGV" != 0 ]; then
  has dg1 || has dg2 || { log "ABORT: UNION sets EXL3_DENSE_V2=$DGV but INCLUDE='$INCLUDE' has no densegemm"; exit 3; }
  if has dg1 && [ "$DGV" != 3 ]; then log "ABORT: densegemm-r1 is mode 3 only (its gemm/mgemm twins spill, R710)"; exit 3; fi
  case $DGV in 1|2|3) ;; *) log "ABORT: EXL3_DENSE_V2=$DGV"; exit 3 ;; esac
fi
case " $PRESENT " in *" MF3 "*) has mf3 || { log "ABORT: UNION has moefast-r3 keys but INCLUDE='$INCLUDE' lacks mf3"; exit 3; } ;; esac
NEED_GUARD=0; [ "$QFV" = 1 ] && [ "$DGV" != 0 ] && NEED_GUARD=1
if [ -s "$R702P0" ]; then
  REC=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommend"])' "$R702P0" 2>/dev/null)
  if [ "$(echo $REC | tr ' ' '\n' | sort | xargs)" = "$(comp_of HC2 "$UNION" | tr ' ' '\n' | sed '/^$/d' | sort | xargs)" ]; then
    log "HC2 = R702 P0 RECOMMEND ($R702P0)"
  else log "ABORT: HC2 in UNION ($(comp_of HC2 "$UNION")) != R702 RECOMMEND ($REC)"; exit 3; fi
else log "WARNING: $R702P0 not found; HC2 = the tables FINDINGS R702 records (not cross-checked)"; fi
SSHA=$(sha256sum "$S3/series/SERIES" | cut -c1-64)

# P1 arms: OFF, U, one Uno<X> per optional component, OFF2 (A/A); rounds = smallest multiple of the arm count >= 6
declare -A ARMF=([OFF]="" [OFF2]="" [U]="$UNION")
ARMS="OFF U"; for X in $OPTS; do ARMF[Uno$X]=$(echo $(minus $X "$UNION")); ARMS="$ARMS Uno$X"; done
# fewer than 6 arms (e.g. UNION_PRESET=noqf): pad with the fixed components' leave-one-out arms (information only: their
# marginal is reported, their decision is the union's), so 6 rounds still rotate fairly instead of 10
for X in RR HC2; do set -- $ARMS OFF2; [ $# -ge 6 ] && break
  case " $PRESENT " in *" $X "*) ARMF[Uno$X]=$(echo $(minus $X "$UNION")); ARMS="$ARMS Uno$X" ;; esac; done
ARMS="$ARMS OFF2"
set -- $ARMS; NARM=$#
k=$NARM; while [ $k -lt 6 ]; do k=$((k + NARM)); done
ROUNDS=${ROUNDS:-$k}
[ $((ROUNDS % NARM)) = 0 ] && [ "$ROUNDS" -ge 6 ] || { log "ABORT: ROUNDS=$ROUNDS with $NARM arms (multiple of the arm count, >= 6)"; exit 3; }
sudo docker image inspect "$BASE" >/dev/null 2>&1 || { log "ABORT: base image $BASE missing"; exit 3; }

gpu_lock
log "lock held; served at entry: $(served_id || echo none); BASE $BASE; IMG $IMG; INCLUDE '$INCLUDE'; defines '$DGV2_NVCC_DEFS'; series sha $SSHA"
log "UNION ($USRC): $UNION"
[ -n "$P1_HARNESS_ARGS" ] && log "P1 harness extra args: $P1_HARNESS_ARGS"
log "components: $PRESENT (optional: ${OPTS:-none}); guard required: $NEED_GUARD; P1 arms: $ARMS x $ROUNDS rounds"
for a in $ARMS; do log "  arm $a flags: ${ARMF[$a]:-<served env>}"; done
log "  served B env: $(arm_env "$UNION")"

# ---- 0. build + landing ----
lbl(){ sudo docker image inspect "$IMG" --format "{{ index .Config.Labels \"$1\" }}" 2>/dev/null; }
if ! sudo docker image inspect "$IMG" >/dev/null 2>&1 || [ "$(lbl local.stack.series_sha256)" != "$SSHA" ] \
   || [ "$(lbl local.stack.base)" != "$BASE" ] || [ "$(lbl local.stack.include)" != "$INCLUDE" ] \
   || [ "$(lbl local.stack.dgv2_defs)" != "$DGV2_NVCC_DEFS" ]; then
  log "building $IMG on $BASE (INCLUDE '$INCLUDE')"
  ( cd "$S3" && sudo docker build -f Dockerfile.box --build-arg BASE="$BASE" --build-arg INCLUDE="$INCLUDE" \
      --build-arg DGV2_NVCC_DEFS="$DGV2_NVCC_DEFS" --build-arg SERIES_SHA="$SSHA" --build-arg MAX_JOBS=4 -t "$IMG" . ) > "$R/build.log" 2>&1 \
    || { log "BUILD FAILED"; tail -40 "$R/build.log" | sed 's/^/  /' | tee -a "$R/audit.log"; finish BUILD-FAILED; exit 3; }
fi
grep -aE "stack-r3: |SASS-IDENTITY|STACK-BOUND|stack-r3 landed|CPU PASS|CPU FAIL" "$R/build.log" 2>/dev/null | tail -30 | sed 's/^/  [build] /' | tee -a "$R/audit.log"
sudo docker run --rm --entrypoint python3 "$IMG" /opt/stack-r3/tools/landing.py --include "$INCLUDE" > "$R/landed.txt" 2>&1 \
  || { log "ABORT: landing check failed: $(tail -3 "$R/landed.txt" | tr '\n' ' ')"; finish ABORTED; exit 3; }
log "image $IMG: $(tail -1 "$R/landed.txt")"
if [ $NEED_GUARD = 1 ]; then
  G=$(sudo docker run --rm --entrypoint python3 "$IMG" -c 'import torch, exllamav3_ext as e; print(getattr(e, "dense_v2_lc_guard", 0))' 2>/dev/null | tail -1)
  [ "$G" = 1 ] || { log "ABORT: QF with EXL3_DENSE_V2=$DGV needs the lcguard in the image (dense_v2_lc_guard=$G); use UNION_PRESET=noqf"; finish ABORTED; exit 3; }
  log "lcguard present (dense_v2_lc_guard=1): QF and EXL3_DENSE_V2=$DGV co-exist"
fi

served_stop; wait_unserved 45
# ---- 1. kernel parity, one card ----
one(){ local name=$1 flags=$2; shift 2
  sudo docker run --rm --name s3-p --gpus '"device=0"' --ipc=host -v /srv/qwen5090/models:/models:ro -v "$R":/results \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    $flags -e CUDA_VISIBLE_DEVICES=0 --entrypoint python3 "$IMG" "$@" > "$R/parity-$name.log" 2>&1
  local rc=$?
  log "parity $name rc=$rc: $(grep -aE 'PARITY (PASS|FAIL)' "$R/parity-$name.log" | tail -1)"
  grep -aq "PARITY PASS" "$R/parity-$name.log" && [ $rc = 0 ] \
    || { grep -aE "MISMATCH|FAIL|Error|error" "$R/parity-$name.log" | head -20 | sed 's/^/  /' | tee -a "$R/audit.log"; finish "PARITY-FAIL ($name)"; exit 5; }; }
T=/opt/stack-r3/tests
one hcfast "" $T/hcfast/test_hcfast_parity.py --model /models/$MODEL --json /results/parity-hcfast.json
one latchain "$(dflags "")" $T/latchain/test_latchain_parity.py --json /results/parity-latchain.json
has mf3 && one moefast "$(dflags "")" $T/moefast/test_moefast_parity.py --model /models/$MODEL --json /results/parity-moefast.json
DGT=""; has dg1 && DGT=densegemm-r1; has dg2 && DGT=densegemm-r2
[ -n "$DGT" ] && [ "$DGV" = 0 ] && log "parity densegemm: SKIPPED ($DGT in the image, EXL3_DENSE_V2 off in UNION)"
[ -n "$DGT" ] && [ "$DGV" != 0 ] && one densegemm "$(dflags "")" $T/$DGT/test_densegemm_parity.py --model /models/$MODEL --mode "$DGV" --json /results/parity-densegemm.json

# ---- 2. model parity (logits hashes), both cards ----
mp(){ local tag=$1 arm=$2 b=$3 d=$4
  sudo docker run --rm --name s3-m --gpus all --ipc=host --shm-size=16g \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v "$R":/results $(dflags "${ARMF[$arm]}") \
    --entrypoint python3 "$IMG" $T/latchain/lc_model_parity.py --model /models/$MODEL \
    --batch $b --draft $d --out /results/mp-$tag.json > "$R/mp-$tag.log" 2>&1
  log "  model parity $tag rc=$?: $(grep -aE '^lc_model_parity' "$R/mp-$tag.log" | tail -1 | cut -c1-160)"; }
MPFAIL=0; MPAA=1; MPGUARD=""
for shape in "b1d3 1 3" "b4d3 4 3" "b8d1 8 1"; do set -- $shape
  mp $1-OFF OFF $2 $3; mp $1-U U $2 $3; mp $1-OFF2 OFF2 $2 $3
  python3 "$S3/tests/latchain/lc_model_parity.py" --compare "$R/mp-$1-OFF.json" "$R/mp-$1-OFF2.json" "$R/mp-$1-U.json" \
    > "$R/mp-$1-compare.txt" 2>&1
  sed "s/^/  [mp $1] /" "$R/mp-$1-compare.txt" | tee -a "$R/audit.log"
  grep -q "aa=IDENTICAL" "$R/mp-$1-compare.txt" || MPAA=0
  grep -q "aa=IDENTICAL arms=DIFFERENT" "$R/mp-$1-compare.txt" && MPFAIL=1
  grep -q "aa=" "$R/mp-$1-compare.txt" || MPFAIL=1
  MPGUARD="$MPGUARD $1:$(grep -ac 'densegemm lcguard' "$R/mp-$1-U.log")"
done
[ $NEED_GUARD = 1 ] && log "model parity U: lcguard lines per shape:$MPGUARD (0 = no side-branch dense call reached the guard in-process)"
if [ $MPFAIL = 1 ]; then log "MODEL PARITY FAIL: U differs from OFF where OFF/OFF2 is identical (or no summary)"; finish MODEL-PARITY-FAIL; exit 5; fi
[ $MPAA = 1 ] && log "model parity PASS (logits-level, all three shapes)" \
  || log "model parity: an OFF/OFF2 A/A differs -> logits-level identity unusable on this build; token-level gates only"

# ---- 3. P1 in-process, rotated rounds ----
run_harness(){ local name=$1 arm=$2; shift 2
  sudo docker run --rm --name "$name" --gpus all --ipc=host --shm-size=16g \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v "$HARNESS":/probe/profile_decode_events.py:ro \
    -v "$METER":/probe/events_meter.py:ro -v "$R":/results $(dflags "${ARMF[$arm]}") --entrypoint "$@"; }
cell(){ grep -ahE 'ms/iterate' "$1/kernels.txt" 2>/dev/null | head -1; }
p1(){ local tag=$1 arm=$2 b=$3 d=$4
  run_harness s3-p1 $arm python3 "$IMG" /probe/profile_decode_events.py --model /models/$MODEL \
    --out /results/p1-$tag --cache-quant 8,8 --gpu-split 30,30 --max-chunk-size 2048 --tokens 64 \
    --warmup-steps 16 --settle-steps 8 --capture-steps 32 --capture-mode events \
    --contexts 4096 --batch $b --draft $d $P1_HARNESS_ARGS > "$R/p1-$tag.log" 2>&1
  log "  P1 $tag rc=$?: d$d $(cell "$R/p1-$tag/ctx4096_b${b}_d${d}") | d0 $(cell "$R/p1-$tag/ctx4096_b${b}_d0")"; }
rounds(){ local sh=$1 b=$2 d=$3 r i a dd order f0 f1; local -a arr=($ARMS); local n=${#arr[@]}
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
python3 "$P1S" --results "$R" --arms "$ARMS" --rounds "$ROUNDS" --json "$R/p1-verdicts.json" > "$R/p1-summary.txt" 2>&1
tee -a "$R/audit.log" < "$R/p1-summary.txt"
P1U=$(sed -nE 's/^P1-VERDICT UNION: (PASS|FAIL).*/\1/p' "$R/p1-summary.txt")
DROPS=""; for X in $OPTS; do grep -q "^P1-VERDICT $X (U - Uno$X): DROP" "$R/p1-summary.txt" && DROPS="$DROPS $X"; done; DROPS=$(echo $DROPS)
SUNION=$UNION; SARM=U
if [ -n "$DROPS" ]; then
  set -- $DROPS
  if [ $# -ge 2 ]; then
    SUG=$UNION; for X in $DROPS; do SUG=$(minus $X "$SUG"); done
    log "DECISION: STOP before the served steps: P1 marks $DROPS DROP; re-queue with UNION=\"$(echo $SUG)\""; finish P1-DROPS; exit 6
  fi
  SUNION=$(echo $(minus $1 "$UNION")); SARM=Uno$1
  P1U=$(sed -nE "s/^P1-VERDICT UNION-$1: (PASS|FAIL).*/\1/p" "$R/p1-summary.txt")
  log "P1 marks $1 DROP: the served steps run UNION-$1 = $SUNION (P1 arm Uno$1: $P1U)"
fi
[ "$P1U" = PASS ] || { log "DECISION: STOP before the served steps: P1 $SARM vs OFF $P1U ($(grep -E "^P1-VERDICT UNION(-[A-Z0-9]+)?: " "$R/p1-summary.txt" | tr '\n' ' '))"; finish P1-FAIL; exit 6; }
SQF=$(for kv in $SUNION; do [ "${kv%%=*}" = EXL3_LC_QSA_FORK ] && echo "${kv#*=}"; done)
SDG=$(for kv in $SUNION; do [ "${kv%%=*}" = EXL3_DENSE_V2 ] && echo "${kv#*=}"; done)
EXPECT_GUARD=""
if [ "${SQF:-0}" = 1 ] && [ "${SDG:-0}" != 0 ]; then
  GIN=$(cat "$R"/mp-*-U.log "$R"/p1-c[148]d[13]-$SARM[0-9]*.log 2>/dev/null | grep -ac 'densegemm lcguard')
  if [ "${GIN:-0}" -gt 0 ]; then EXPECT_GUARD=--expect-lcguard; log "lcguard engaged in-process ($GIN lines in the $SARM runs): every served B boot must show it too"
  else log "CAVEAT: lcguard never engaged in-process (0 lines in the $SARM runs): no side-branch dense call on this kernel inventory; QF + DG co-existence rests on the CPU model (test_dense_lcguard_cpu.py)"; fi
fi

# ---- 4. served boots ----
SENV=$(arm_env "$SUNION")
NONCE=$(( $(date +%s) % 100000 ))
GS=$R/gstreams.jsonl; G=$R/greedy.jsonl
boot(){ local tag=$1; shift
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { log "[$tag] NO BOOT"; sudo docker logs flashnext > "$R/container-$tag.log" 2>&1; return 1; }
  sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$R/env-$tag.txt"
  log "[$tag] booted $(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'); $(wc -l < "$R/env-$tag.txt") EXL3 keys; tier $(grep -c '^EXL3_NVME_TIER=' "$R/env-$tag.txt"); $(grep -aoE 'UP on .*' "$R/boot-$tag.log" | tail -1)"
  grep -q '^EXL3_NVME_TIER=' "$R/env-$tag.txt" && { log "[$tag] ABORT: the NVMe tier is on (NVME_TIER= did not reach the launcher): arms would not be comparable (R709c)"; return 1; }
  return 0; }
probes(){ local tag=$1 gt=$1; [ "$tag" = A1 ] && gt=ref
  python3 "$GSTREAMS" --url http://127.0.0.1:8022 --tag "$tag" --conc 2 4 8 --rounds 2 --tokens 384 --out "$GS" > "$R/gstreams-$tag.log" 2>&1
  python3 "$GREEDY" --url http://127.0.0.1:8022 --tag "$gt" --out "$G" > "$R/greedy-$tag.log" 2>&1
  python3 "$BENCH" --url http://127.0.0.1:8022/v1 --model "$MODEL" --tag ramp --kind prose --ctx 4000 --tokens 256 \
    --conc 1 2 3 4 5 6 7 8 --runs 1 --warmup-runs 0 --unique --distinct --salt $(( (NONCE + 4099) % 100000 )) \
    --out "$R/ramp-$tag.jsonl" > "$R/ramp-$tag.log" 2>&1
  log "[$tag] greedy_streams $(tail -1 "$R/gstreams-$tag.log"); ramp $(grep -c '"ok": true' "$R/ramp-$tag.jsonl" 2>/dev/null)/$(wc -l < "$R/ramp-$tag.jsonl" 2>/dev/null) ok"; }
gate(){ local tag=$1 pair=$2
  RUNS=3 bash "$GATE" "$R/gate-$tag" http://127.0.0.1:8022/v1 "$MODEL" $(( (NONCE + pair * 7919) % 100000 )) > "$R/gate-$tag.out" 2>&1
  log "[$tag] gate done; rows A $(wc -l < "$R/gate-$tag/bench-A.jsonl" 2>/dev/null) B $(grep -c '"ok": true' "$R/gate-$tag/bench-B.jsonl" 2>/dev/null)/$(wc -l < "$R/gate-$tag/bench-B.jsonl" 2>/dev/null)"; }
clog(){ sudo docker logs flashnext > "$R/container-$1.log" 2>&1
  log "[$1] container: OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$1.log"), TORCH_CHECK $(grep -acE 'TORCH_CHECK|c10::Error' "$R/container-$1.log"), tracebacks $(grep -ac Traceback "$R/container-$1.log"), lcguard $(grep -ac 'densegemm lcguard' "$R/container-$1.log"); free now $(vram_free)"; }
gcmp(){ python3 "$GREEDY" --compare --ref ref --out "$G" > "$R/greedy-compare.txt" 2>&1
  python3 "$GSTREAMS" --compare --out "$GS" --ref A1 --aa $1 --arms $2 > "$R/gstreams-compare.txt" 2>&1; }
bdiv(){ local tag=$1 n; n=$(sed -nE "s/^GREEDY $tag vs ref: [0-9]+ identical, ([0-9]+) DIVERGENT.*/\1/p" "$R/greedy-compare.txt")
  grep -q "^GREEDY $tag vs ref: " "$R/greedy-compare.txt" || n=missing
  echo "${n:-0}"; }
sdiv(){ grep -E "^GSTREAM $1 vs A1 c[0-9]+: " "$R/gstreams-compare.txt" | grep -c -- "-> FAIL"; }

boot A1 || { finish NO-BOOT; exit 3; }; probes A1; gate A1 1; clog A1
boot B1 IMG="$IMG" EXTRA_ENV="$SENV" || { finish "NO-BOOT (B1)"; exit 3; }; probes B1; clog B1
gcmp "" B1
B1G=$(bdiv B1); B1S=$(sdiv B1)
log "[B1] identity vs A1: fn_greedy divergences $B1G; greedy_streams failing levels $B1S"
AAX=""
if [ "$B1G" != 0 ] || [ "$B1S" != 0 ]; then
  log "[B1] DIVERGES: one more daily boot (A/A) before any conclusion"
  boot A1b || { finish NO-BOOT; exit 3; }; probes A1b; clog A1b; AAX=A1b
  gcmp A1b B1
  AAG=$(bdiv A1b)
  log "[A1b] A/A vs A1: fn_greedy divergences $AAG; greedy_streams: $(grep -E '^GSTREAM A/A A1b' "$R/gstreams-compare.txt" | tr '\n' ' ')"
  if [ "$AAG" = 0 ] && ! grep -E "^GSTREAM A/A A1b vs A1" "$R/gstreams-compare.txt" | grep -qv " div 0 "; then
    log "[B1] the A/A is identical, so B1's divergence is real: one greedy boot per optional component without it"
    for X in $(for kv in $SUNION; do fam "$kv"; done | sort -u); do
      case " $OPTS " in *" $X "*) ;; *) continue ;; esac
      boot L$X IMG="$IMG" EXTRA_ENV="$(arm_env "$(minus $X "$SUNION")")" || continue
      probes L$X; clog L$X
    done
    boot LOFF IMG="$IMG" && { probes LOFF; clog LOFF; }
    python3 "$GREEDY" --compare --ref ref --out "$G" > "$R/greedy-compare.txt" 2>&1
    python3 "$GSTREAMS" --compare --out "$GS" --ref A1 --aa A1b --arms B1 $(ls "$R"/boot-L*.log 2>/dev/null | sed -E 's/.*boot-(L[A-Z0-9]+)\.log/\1/') > "$R/gstreams-compare.txt" 2>&1
    grep -aE "^GREEDY |^GSTREAM " "$R/greedy-compare.txt" "$R/gstreams-compare.txt" | sed 's/^/  [loo] /' | tee -a "$R/audit.log"
    log "DECISION: REJECT (served identity): the union diverges from the daily; LOO boots L<X> = union without X (identical => X is the culprit), LOFF = image flags off"
    finish IDENTITY-FAIL; exit 5
  fi
  log "[B1] the A/A itself diverges (instrument nondeterministic at some level): continue; greedy_streams is judged BOUNDED there"
  boot B1 IMG="$IMG" EXTRA_ENV="$SENV" || { finish "NO-BOOT (B1 again)"; exit 3; }   # A1b replaced B1's container
fi
gate B1 1; clog B1
for p in 2 3; do
  boot A$p || { finish NO-BOOT; exit 3; }; probes A$p; gate A$p $p; clog A$p
  boot B$p IMG="$IMG" EXTRA_ENV="$SENV" || { finish "NO-BOOT (B$p)"; exit 3; }; probes B$p; gate B$p $p; clog B$p
done
boot C IMG="$IMG" && { probes C; clog C; }

# ---- 5. analysis + DECISION ----
AA="A2 A3 $AAX"
gcmp "$AA" "B1 B2 B3 C"
grep -aE "^GREEDY |^  (DIVERGE|MISSING)|GREEDY-SUMMARY" "$R/greedy-compare.txt" | sed 's/^/  [greedy] /' | tee -a "$R/audit.log"
grep -aE "^GSTREAM" "$R/gstreams-compare.txt" | sed 's/^/  [gstreams] /' | tee -a "$R/audit.log"
python3 "$SERVED" --results "$R" --a "A1 A2 A3" --b "B1 B2 B3" --extra C --union "$SUNION" $EXPECT_GUARD --json "$R/served.json" > "$R/served.txt" 2>&1
tee -a "$R/audit.log" < "$R/served.txt"
GBAD=0; GVOID=0
for t in A2 A3 $AAX; do [ "$(bdiv $t)" = 0 ] || { GVOID=1; log "fn_greedy A/A $t vs ref: $(bdiv $t) divergent (the daily disagrees with itself)"; }; done
for t in B1 B2 B3 C; do [ "$(bdiv $t)" = 0 ] || { GBAD=1; log "fn_greedy $t vs ref: $(bdiv $t) divergent"; }; done
GSV=$(tail -1 "$R/gstreams-compare.txt")
SV=$(tail -1 "$R/served.txt")
BOUNDED=$(grep -c "BOUNDED" "$R/gstreams-compare.txt"); BOUNDED=${BOUNDED:-0}
why=""
[ $GBAD = 0 ] || [ $GVOID = 1 ] || why="$why fn_greedy divergences;"
case "$GSV" in "GSTREAM-SUMMARY PASS"*) ;; *) why="$why greedy_streams: $GSV;" ;; esac
case "$SV" in "SERVED-VERDICT PASS") ;; *) why="$why ${SV#SERVED-VERDICT };" ;; esac
MPNOTE=$([ $MPAA = 1 ] && echo "model parity identical x3" || echo "model parity A/A nondeterministic (token-level only)")
AAP1=$(sed -nE 's/^P1-VERDICT A\/A \(OFF2 - OFF\): //p' "$R/p1-summary.txt")
if [ $GVOID = 1 ]; then
  log "DECISION UNION: VOID (served identity instrument: fn_greedy A/A boots diverge from A1; B/C divergences $GBAD) | other bars:${why:- all pass} | re-run the served steps"
  why="${why} served identity VOID;"
elif [ -z "$why" ]; then
  log "DECISION UNION: PASS ($SARM = $SUNION) | $MPNOTE | P1 $SARM vs OFF PASS | P1 $AAP1 | served $(grep -E '^AGGREGATE' "$R/served.txt")$([ "$BOUNDED" -gt 0 ] && echo " | greedy_streams BOUNDED at $BOUNDED arm-levels (A/A nondeterministic there): read gstreams-compare.txt")"
else
  log "DECISION UNION: REJECT ($SARM):$why"
fi
for X in RR HC2; do grep -q "^P1-VERDICT $X (U - Uno$X)" "$R/p1-summary.txt" && log "info: fixed component $X marginal in P1: $(sed -nE "s/^P1-VERDICT $X \(U - Uno$X\): //p" "$R/p1-summary.txt")"; done
for X in HC2 RR; do case " $PRESENT " in *" $X "*)
  [ -z "$why" ] && log "DECISION $X: ENTERS with the union (fixed component)" || log "DECISION $X: HOLD (union rejected)" ;; esac; done
for X in $OPTS; do
  v=$(sed -nE "s/^P1-VERDICT $X \(U - Uno$X\): //p" "$R/p1-summary.txt")
  case " $DROPS " in *" $X "*) log "DECISION $X: DROP (P1 marginal: $v)"; continue ;; esac
  [ -z "$why" ] && log "DECISION $X: ENTERS (P1 marginal: $v; served union PASS)" || log "DECISION $X: HOLD (P1 marginal: $v; union rejected)"
done
if [ -z "$why" ]; then
  log "RECOMMENDED daily: IMG=$IMG (INCLUDE '$INCLUDE'${DGV2_NVCC_DEFS:+, defines '$DGV2_NVCC_DEFS'}) EXTRA_ENV=\"$SENV\""
  log "  apply via launch-flashnext-stack-r3.sh.diff (update its EXTRA_ENV to the line above if it differs); rollback: /srv/qwen5090/launch-flashnext.sh.pre-stack-r3. Not promoted by this unit."
fi
finish DONE
