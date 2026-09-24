#!/usr/bin/env bash
# R701 (2026-09-24) — first STACK gate under OPERATIONS §16 (stack small bitwise-identical wins; ledger STACK.md).
# Stack = hcfast r1 with DOTS_B=2 (R699 candidate: c1d3 -4.6 %, c8d1 ~ -2.6 %; c4d3 VOID) [+ moefast r1 if R700b passes, MOE_MODE].
# Image: tabbyapi:stack-r2 = tabbyapi:hcfast-r1 + moefast-r1 (patches apply in sequence at --fuzz=0, 0 offsets, checked
# 2026-09-24 on the pristine slotfix-r1 tree); with MOE_MODE empty the stack is hcfast-r1 alone.
# Parts:
#   0 build (in-lock, BEFORE any measurement; OPERATIONS §15) + both landing markers
#   1 P1 in-process harness, 4k, untraced: 5 rounds x (OFF, HC, STACK), order rotated per round, at c4d3, c8d1, c1d3;
#     sequence hashes of HC and STACK must equal the round's OFF. Gives the ledger's per-flag deltas (HC clean c4d3).
#   2 served ABAB x3 (OFF = the daily launcher; ON = IMG=stack + EXTRA_ENV_ADD flags): free VRAM, fn_greedy vs the first OFF,
#     canonical fn_gate RUNS=3, OOM count, free after.
# DECISION (pre-registered, §16): promote the stack if
#   - identity: P1 hashes identical in every round and shape; greedy ON vs ref 0 divergences in all 3 ON boots;
#   - no regression: P1 no shape with all 5 STACK-OFF deltas > 0; served no leg-A/B cell with 3-pair mean ON/OFF < 0.99;
#   - aggregate gain: mean over leg-A cells and leg B of the 3-pair ON/OFF > 1.000;
#   - safety: 0 OOM lines, leg B 8/8 x3, free cuda:0 after the gate >= the OFF boots' minimum - 32 MiB.
# Otherwise hold, report per-cell numbers, and put each flag's P1 deltas in STACK.md. This unit does not promote.
# GPU ~70 min. Queue-chained; the daily is restored at the end.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
R=${R:-/srv/qwen5090/results/2026-09-24-r701-stack-gate}; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
HARNESS=/srv/qwen5090/probes/r465/profile_decode_events.py
METER=/srv/qwen5090/probes/r465/events_meter.py
GREEDY=/srv/qwen5090/probes/fn_greedy.py
GATE=/srv/qwen5090/probes/fn_gate.sh
MF=/srv/qwen5090/patches/exllamav3/moefast-r1
MOE_MODE=${MOE_MODE:-}                      # 1 or 2 from R700b; empty = HC only
HC_FLAGS="EXL3_HC_MIX_V3=1 EXL3_HC_MIX_V3_DOTS_B=2 EXL3_HC_MIX_V3_UP_B=8"
if [ -n "$MOE_MODE" ]; then IMG=tabbyapi:stack-r2; STACK_FLAGS="$HC_FLAGS EXL3_MOE_COOP_V3=$MOE_MODE"
else IMG=tabbyapi:hcfast-r1; STACK_FLAGS="$HC_FLAGS"; fi
log(){ echo "$(date -Is) [r701] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=r701-stack-gate
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ sudo docker rm -f r701-p1 >/dev/null 2>&1 || true
  sudo docker logs flashnext > "$R/container-final.log" 2>&1 || true
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== R701 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$HARNESS" "$METER" "$GREEDY" "$GATE"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
BASE=$(grep -oE '^DAILY_IMG=\S+' "$LIVE" | cut -d= -f2)
[ "$BASE" = tabbyapi:slotfix-r1 ] || { log "ABORT: daily image is $BASE, the stack was built against slotfix-r1"; exit 3; }
sudo docker image inspect tabbyapi:hcfast-r1 >/dev/null 2>&1 || { log "ABORT: tabbyapi:hcfast-r1 missing (R698)"; exit 3; }

gpu_lock
log "lock held; served at entry: $(served_id || echo none); IMG $IMG; STACK_FLAGS $STACK_FLAGS"

# ---- 0. build + landing ----
if [ -n "$MOE_MODE" ] && ! sudo docker image inspect "$IMG" >/dev/null 2>&1; then
  log "building $IMG = tabbyapi:hcfast-r1 + moefast-r1"
  ( cd "$MF" && sudo docker build -f Dockerfile.box --build-arg BASE=tabbyapi:hcfast-r1 --build-arg MAX_JOBS=4 -t "$IMG" . ) > "$R/build.log" 2>&1 \
    || { log "BUILD FAILED"; tail -30 "$R/build.log" | sed 's/^/  /' | tee -a "$R/audit.log"; finish BUILD-FAILED; exit 3; }
fi
sudo docker run --rm --entrypoint python3 "$IMG" -c '
import torch, exllamav3_ext as e, exllamav3.modules.hyperconnections as h
assert h._HC_MIX_V3_BUILD == "r1"
print("hcfast", h._HC_MIX_V3_BUILD, "moefast", getattr(e, "moe_coop_v3_revision", None))' > "$R/landed.txt" 2>&1 \
  || { log "ABORT: landing check failed: $(tail -2 "$R/landed.txt")"; finish ABORTED; exit 3; }
[ -z "$MOE_MODE" ] || grep -q "moefast 1" "$R/landed.txt" || { log "ABORT: moefast marker missing: $(cat "$R/landed.txt")"; finish ABORTED; exit 3; }
log "image $IMG: $(cat "$R/landed.txt")"

served_stop; wait_unserved 45
# ---- 1. P1 in-process, 5 rounds x (OFF, HC, STACK) ----
EXTRA=$(sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-([^}]*)\}.*/\1/p' "$LIVE"); ENVS=""; for kv in $EXTRA; do ENVS="$ENVS -e $kv"; done
p1(){ local tag=$1 flags=$2 b=$3 d=$4 fe=""
  for kv in $flags; do fe="$fe -e $kv"; done
  sudo docker run --rm --name r701-p1 --gpus all --ipc=host --shm-size=16g \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v "$HARNESS":/probe/profile_decode_events.py:ro \
    -v "$METER":/probe/events_meter.py:ro -v "$R":/results $ENVS $fe \
    --entrypoint python3 "$IMG" /probe/profile_decode_events.py --model /models/$MODEL \
    --out /results/p1-$tag --cache-quant 8,8 --gpu-split 30,30 --max-chunk-size 2048 --tokens 64 \
    --warmup-steps 16 --settle-steps 8 --capture-steps 32 --capture-mode events \
    --contexts 4096 --batch $b --draft $d > "$R/p1-$tag.log" 2>&1
  log "  P1 $tag rc=$?: $(grep -ahE 'ms/iterate' "$R"/p1-$tag/ctx4096_b${b}_d${d}/kernels.txt 2>/dev/null | head -1)"; }
for shape in "c4d3 4 3" "c8d1 8 1" "c1d3 1 3"; do set -- $shape
  for r in 1 2 3 4 5; do
    case $(( r % 3 )) in
      1) order="OFF HC ST" ;; 2) order="HC ST OFF" ;; 0) order="ST OFF HC" ;;
    esac
    [ -n "$MOE_MODE" ] || order=$(echo $order | sed 's/ST//')   # HC-only stack: ST would duplicate HC
    for a in $order; do
      case $a in OFF) p1 $1-$a$r "" $2 $3 ;; HC) p1 $1-$a$r "$HC_FLAGS" $2 $3 ;; ST) p1 $1-$a$r "$STACK_FLAGS" $2 $3 ;; esac
    done
    for a in HC ${MOE_MODE:+ST}; do
      cmp -s "$R/p1-$1-OFF$r/ctx4096_b$2_d$3/sequence-hashes.json" "$R/p1-$1-$a$r/ctx4096_b$2_d$3/sequence-hashes.json" \
        && log "  $1 round $r $a: hashes IDENTICAL" || log "  $1 round $r $a: hashes DIFFER -> FAIL"
    done
  done
done
python3 - "$R" <<'EOF' 2>&1 | tee -a "$R/audit.log"
import re, sys, glob, statistics as st
R = sys.argv[1]
def ms(tag, b, d):
    try:
        t = open(f"{R}/p1-{tag}/ctx4096_b{b}_d{d}/kernels.txt").read()
        return float(re.search(r"([0-9.]+) ms/iterate", t).group(1))
    except Exception: return None
for sh, b, d in (("c4d3", 4, 3), ("c8d1", 8, 1), ("c1d3", 1, 3)):
    for a in ("HC", "ST"):
        ds = []
        for r in range(1, 6):
            o, x = ms(f"{sh}-OFF{r}", b, d), ms(f"{sh}-{a}{r}", b, d)
            if o and x: ds.append(x - o)
        if not ds: print(f"P1 {sh} {a}: no data"); continue
        off = st.mean([v for v in (ms(f"{sh}-OFF{r}", b, d) for r in range(1, 6)) if v])
        sign = "GAIN" if all(v < 0 for v in ds) else "REGRESSION" if all(v > 0 for v in ds) else "mixed"
        print(f"P1 {sh} {a}-OFF ms: " + " ".join(f"{v:+.2f}" for v in ds) + f"  mean {st.mean(ds):+.3f} ({100*st.mean(ds)/off:+.2f} %) {sign}")
EOF

# ---- 2. served ABAB x3 ----
NONCE=$(( $(date +%s) % 100000 ))
oom(){ sudo docker logs flashnext 2>&1 | grep -acE 'OutOfMemoryError|out of memory|graph.cu'; }
arm(){ local tag=$1 pair=$2; shift 2
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { log "[$tag] NO BOOT"; sudo docker logs flashnext > "$R/container-$tag.log" 2>&1; return 1; }
  log "[$tag] booted $(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'); stack flags in env: $(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -cE '^EXL3_(HC_MIX_V3|MOE_COOP_V3)='); free $(vram_free)"
  local gt=$tag; [ "$tag" = A1 ] && gt=ref
  python3 "$GREEDY" --url http://127.0.0.1:8022 --tag "$gt" --out "$R/greedy.jsonl" > "$R/greedy-$tag.log" 2>&1
  [ "$gt" = ref ] || python3 "$GREEDY" --compare --ref ref --out "$R/greedy.jsonl" 2>&1 | grep -aE "^GREEDY $gt vs ref|MISSING $gt " | sed "s/^/  [$tag] /" | tee -a "$R/audit.log"
  RUNS=3 bash "$GATE" "$R/gate-$tag" http://127.0.0.1:8022/v1 "$MODEL" $(( (NONCE + pair * 7919) % 100000 )) > "$R/gate-$tag.out" 2>&1
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  log "[$tag] gate done; rows A $(wc -l < "$R/gate-$tag/bench-A.jsonl" 2>/dev/null) B $(grep -c '"ok": true' "$R/gate-$tag/bench-B.jsonl" 2>/dev/null)/$(wc -l < "$R/gate-$tag/bench-B.jsonl" 2>/dev/null); OOM $(oom); tracebacks $(grep -ac Traceback "$R/container-$tag.log"); free after $(vram_free)"; }
for p in 1 2 3; do
  arm A$p $p
  arm B$p $p IMG=$IMG EXTRA_ENV_ADD="$STACK_FLAGS"
done
python3 - "$R" <<'EOF' 2>&1 | tee -a "$R/audit.log"
import json, sys, statistics as st, os
R = sys.argv[1]
def cells(tag, leg):
    p = f"{R}/gate-{tag}/bench-{leg}.jsonl"; out = {}
    if not os.path.exists(p): return out
    for ln in open(p):
        try: d = json.loads(ln)
        except Exception: continue
        if not d.get("ok"): continue
        c = d.get("conc"); v = d.get("decode_tps") or d.get("per_stream_decode_tps")
        if c is None or v is None: continue
        out.setdefault(c, []).append(v)
    return {c: st.median(v) for c, v in out.items()}
means = []
for leg in ("A", "B"):
    per = {}
    for p in (1, 2, 3):
        a, b = cells(f"A{p}", leg), cells(f"B{p}", leg)
        for c in sorted(set(a) | set(b)):
            r = (b.get(c, 0) / a[c]) if a.get(c) else float("nan")
            per.setdefault(c, []).append(r)
            print(f"RATIO leg {leg} pair {p} c{c}: OFF {a.get(c, float('nan')):.1f} ON {b.get(c, float('nan')):.1f} ON/OFF {r:.3f}")
    for c, rs in sorted(per.items()):
        m = st.mean(rs); means.append(m)
        print(f"MEAN leg {leg} c{c}: {m:.3f}" + ("  < 0.99 REGRESSION" if m < 0.99 else ""))
if means: print(f"AGGREGATE mean ON/OFF over cells: {st.mean(means):.4f}")
EOF
finish DONE
