#!/usr/bin/env bash
# R717b (2026-09-24): rows32-r4 follow-up to R717's REJECT (review: opus-decode/review-r717/out/REVIEW.md, verdict
# "PROMOTABLE AFTER" the user's headroom decision, the c7d2 cells and the decide-script fixes). No promotion here.
#   1 c7 P1 cells at --gpu-split 28,30 (R717's four c7-R32 runs OOMed in the harness at 30,30, exl3_gemv_int8.cu:110,
#     a 16 MiB raw cudaMalloc; same class as R716c's b2d3): SRV (A env, d1) x4, R32 (B env, d2) x4, ON1 (B env, d1) x2;
#     identity = R32 rounds identical, ON1 == SRV; timing diagnostic
#   2 where cuda:0's extra 48 MiB goes (UP line only, no load): X1 = B image + B env + policy A,
#     X2 = B image + A env + policy A, X0 = the daily (A image + A env + policy A)
#   3 the trade the user will decide on: P = B image + B env + policy B at CACHE=983040 (R683: the lowest rung before
#     the loader moves layer 24 onto cuda:0) vs Q = the daily at 999,424, same sequence on both: fn_greedy (incl.
#     ~100k), ramp c1..c8, canonical fn_gate RUNS=1 (leg A c1/c4/c8 + leg B 26k c4). Reports UP-line and post-sequence
#     free per card, OOM / TORCH_CHECK, greedy vs R717's ref.
# Envs are R717's exactly (read from its audit.log). NVME_TIER= on every boot. GPU ~35 min. The daily is restored.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
UNIT=${UNIT:-$(basename "$0" .sh)}
R=${R:-/srv/qwen5090/results/$(date +%F)-$UNIT}; mkdir -p "$R"
R717=${R717:-/srv/qwen5090/results/2026-09-24-r717-rows32-r4}
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
HARNESS=/srv/qwen5090/probes/r465/profile_decode_events.py
METER=/srv/qwen5090/probes/r465/events_meter.py
GREEDY=/srv/qwen5090/probes/fn_greedy.py
GATE=/srv/qwen5090/probes/fn_gate.sh
BENCH=/srv/qwen5090/probes/fn_bench.py
BASE=tabbyapi:stack-r3
IMG=tabbyapi:stack-r3-rows32
POLICY_A='[[4, 3], [5, 2], [8, 1]]'
POLICY_B='[[4, 3], [8, 2]]'
POOL_P=${POOL_P:-983040}
NONCE=$(( $(date +%s) % 100000 ))
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$UNIT
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ sudo docker rm -f r32b-p1 >/dev/null 2>&1 || true
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== r717b $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$R717/audit.log" "$HARNESS" "$METER" "$GREEDY" "$GATE" "$BENCH" "$R717/greedy.jsonl"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
AENV=$(sed -nE 's/.*\]   A env \([0-9]+ keys\): //p' "$R717/audit.log" | head -1)
BENV=$(sed -nE 's/.*\]   B env \([0-9]+ keys\): //p' "$R717/audit.log" | head -1)
[ -n "$AENV" ] && [ -n "$BENV" ] || { log "ABORT: could not read R717's A/B envs"; exit 3; }
log "A env ($(echo $AENV | wc -w) keys); B env ($(echo $BENV | wc -w) keys); B - A: $(comm -13 <(echo $AENV | tr ' ' '\n' | sort) <(echo $BENV | tr ' ' '\n' | sort) | xargs)"
for i in "$BASE" "$IMG"; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: image $i missing"; exit 3; }; done
dflags(){ local kv fe=""; for kv in $1; do fe="$fe -e $kv"; done; echo "$fe"; }
vram_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | xargs; }

gpu_lock
log "lock held; served at entry: $(served_id || echo none)"
served_stop; wait_unserved 45
for i in $(seq 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${u:-99999}" -lt 1024 ] && break; sleep 2; done

# ---- 1. c7 P1 cells at --gpu-split 28,30 ----
p1(){ local tag=$1 envs=$2 b=$3 d=$4
  sudo docker run --rm --name r32b-p1 --gpus all --ipc=host --shm-size=16g \
    -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v "$HARNESS":/probe/profile_decode_events.py:ro \
    -v "$METER":/probe/events_meter.py:ro -v "$R":/results $(dflags "$envs") \
    --entrypoint python3 "$IMG" /probe/profile_decode_events.py --model /models/$MODEL \
    --out /results/p1-$tag --cache-quant 8,8 --gpu-split 28,30 --max-chunk-size 2048 --tokens 64 \
    --warmup-steps 16 --settle-steps 8 --capture-steps 32 --capture-mode events \
    --contexts 4096 --batch $b --draft $d > "$R/p1-$tag.log" 2>&1
  local rc=$?
  log "  P1 $tag rc=$rc: d$d $(grep -ahoE '[0-9.]+ ms/iterate' "$R"/p1-$tag/ctx4096_b${b}_d${d}/kernels.txt 2>/dev/null | head -1)$(grep -aq 'out of memory' "$R/p1-$tag.log" && echo ' OOM')"; }
same(){ local f1="$R/p1-$1/ctx4096_b$3_d$4/sequence-hashes.json" f2="$R/p1-$2/ctx4096_b$3_d$4/sequence-hashes.json"
  if [ ! -s "$f1" ] || [ ! -s "$f2" ]; then echo MISSING; elif cmp -s "$f1" "$f2"; then echo IDENTICAL; else echo DIFFER; fi; }
C7BAD=0
for r in 1 2 3 4; do
  if [ $(( r % 2 )) = 1 ]; then order="SRV R32"; else order="R32 SRV"; fi
  for a in $order; do if [ $a = R32 ]; then p1 c7-R32$r "$BENV" 7 2; else p1 c7-SRV$r "$AENV" 7 1; fi; done
  if [ $r = 1 ] || [ $r = 3 ]; then p1 c7-ON1$r "$BENV" 7 1; v=$(same c7-SRV$r c7-ON1$r 7 1); log "  c7d1 round $r ON1 vs SRV: $v"; [ "$v" = IDENTICAL ] || C7BAD=1; fi
  if [ $r -gt 1 ]; then v=$(same c7-R321 c7-R32$r 7 2); log "  c7d2 R32 round $r vs round 1: $v"; [ "$v" = IDENTICAL ] || C7BAD=1; fi
done
python3 - "$R" <<'EOF' | tee -a "$R/audit.log"
import re, sys, glob, statistics as st
R = sys.argv[1]
def ms(tag, d):
    try: return float(re.search(r"([0-9.]+) ms/iterate", open(f"{R}/p1-{tag}/ctx4096_b7_d{d}/kernels.txt").read()).group(1))
    except Exception: return None
s = [ms(f"c7-SRV{r}", 1) for r in range(1, 5)]; b = [ms(f"c7-R32{r}", 2) for r in range(1, 5)]
if all(s) and all(b):
    print(f"  c7 step ms: SRV d1 {[round(x, 2) for x in s]} median {st.median(s):.2f}; R32 d2 {[round(x, 2) for x in b]} median {st.median(b):.2f}; d2/d1 {st.median(b) / st.median(s):.3f} (diagnostic)")
else:
    print(f"  c7 step ms: incomplete SRV {s} R32 {b}")
EOF
log "C7 IDENTITY: $([ $C7BAD = 0 ] && echo PASS || echo FAIL)"

# ---- 2. where the 48 MiB goes: UP-line boots ----
boot(){ local tag=$1; shift
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { log "[$tag] NO BOOT"; sudo docker logs flashnext > "$R/container-$tag.log" 2>&1; return 1; }
  sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$R/env-$tag.txt"
  grep -q '^EXL3_NVME_TIER=' "$R/env-$tag.txt" && { log "[$tag] ABORT: NVMe tier on"; return 1; }
  log "[$tag] booted $(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'); $(wc -l < "$R/env-$tag.txt") EXL3 keys; $(grep -aoE "cache [0-9]+" "$R/boot-$tag.log" | tail -1); $(grep -aoE "policy '[^']*'" "$R/boot-$tag.log" | tail -1); $(grep -aoE 'VRAM free MiB [0-9/]+' "$R/boot-$tag.log" | tail -1)"; }
boot X0 IMG="$BASE" "EXTRA_ENV=$AENV" "DRAFT_POLICY=$POLICY_A" || { finish NO-BOOT; exit 3; }
boot X2 IMG="$IMG" "EXTRA_ENV=$AENV" "DRAFT_POLICY=$POLICY_A" || { finish NO-BOOT; exit 3; }
boot X1 IMG="$IMG" "EXTRA_ENV=$BENV" "DRAFT_POLICY=$POLICY_A" || { finish NO-BOOT; exit 3; }
boot X3 IMG="$IMG" "EXTRA_ENV=$BENV" "DRAFT_POLICY=$POLICY_B" || { finish NO-BOOT; exit 3; }

# ---- 3. the trade: P (rows32 at POOL_P) vs Q (daily at 999,424), same sequence ----
G=$R/greedy.jsonl; cp "$R717/greedy.jsonl" "$G"
seq_run(){ local tag=$1 pair=$2
  python3 "$GREEDY" --url http://127.0.0.1:8022 --tag "$tag" --out "$G" > "$R/greedy-$tag.log" 2>&1
  python3 "$BENCH" --url http://127.0.0.1:8022/v1 --model "$MODEL" --tag ramp --kind prose --ctx 4000 --tokens 256 \
    --conc 1 2 3 4 5 6 7 8 --runs 1 --warmup-runs 0 --unique --distinct --salt $(( (NONCE + 4099) % 100000 )) \
    --out "$R/ramp-$tag.jsonl" > "$R/ramp-$tag.log" 2>&1
  RUNS=1 bash "$GATE" "$R/gate-$tag" http://127.0.0.1:8022/v1 "$MODEL" $(( (NONCE + pair * 7919) % 100000 )) > "$R/gate-$tag.out" 2>&1
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  log "[$tag] ramp $(grep -c '"ok": true' "$R/ramp-$tag.jsonl" 2>/dev/null)/$(wc -l < "$R/ramp-$tag.jsonl" 2>/dev/null) ok; gate leg B $(grep -c '"ok": true' "$R/gate-$tag/bench-B.jsonl" 2>/dev/null)/$(wc -l < "$R/gate-$tag/bench-B.jsonl" 2>/dev/null) ok; OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$tag.log"); TORCH_CHECK $(grep -acE 'TORCH_CHECK|c10::Error' "$R/container-$tag.log"); tracebacks $(grep -ac Traceback "$R/container-$tag.log"); free after the sequence $(vram_free) MiB"; }
boot Q IMG="$BASE" "EXTRA_ENV=$AENV" "DRAFT_POLICY=$POLICY_A" && seq_run Q 1
boot P IMG="$IMG" "EXTRA_ENV=$BENV" "DRAFT_POLICY=$POLICY_B" CACHE=$POOL_P && seq_run P 1
python3 "$GREEDY" --compare --ref ref --out "$G" > "$R/greedy-compare.txt" 2>&1
grep -aE "^GREEDY (P|Q) |GREEDY-SUMMARY" "$R/greedy-compare.txt" | sed 's/^/  [greedy] /' | tee -a "$R/audit.log"
tail -4 "$R/gate-Q.out" 2>/dev/null | sed 's/^/  [gate Q] /' | tee -a "$R/audit.log"
tail -4 "$R/gate-P.out" 2>/dev/null | sed 's/^/  [gate P] /' | tee -a "$R/audit.log"
finish DONE
