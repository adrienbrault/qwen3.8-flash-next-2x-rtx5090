#!/usr/bin/env bash
# R723 (2026-09-25): does the daily fit WITHOUT the draft-KV window at the served pool (983,040)?
# R721 (reviewed, CONFIRMED WITH CAVEATS): EXL3_MTP_KV_WINDOW loses draft acceptance on prompts revived from the prompt
# cache in a later request group (revived streams 0.655 vs 0.868 without the window; fresh 0.911 in both), worth +12-13 %
# c8 decode aggregate on that sequence. R721 ran window-off at 950,272 (the first rung of its ladder) and booted there
# with 1305 / 1813 MiB free vs the daily's 1125 / 2513; its review noted 983,040 was never tried. If it fits, the revive
# loss is gone with no patch and no pool cost. R636/R637 also measured the window LOSING acceptance above 16k tokens of
# context, so window-off may help long prompts as well.
#   D = the daily as served (window on, 983,040). N0 = window off at 983,040. N1 = window off at 999,424 (diagnostic).
#   Each passes R717b's trade sequence: fn_greedy (incl. ~100k), ramp c1..c8 (prose 4k, 256 tokens), canonical fn_gate
#   RUNS=1 (leg A c1/c4/c8 + leg B 26k c4). UP-line and post-sequence free per card, OOM / TORCH_CHECK / tracebacks.
# DECISION (pre-registered, N0 vs D):
#   FITS      N0 boots; UP-line cuda:0 >= D's - 32 MiB and cuda:1 >= 835; post-sequence free >= D's - 32 on each card;
#             0 OOM / TORCH_CHECK / tracebacks; leg B 4/4; greedy runs complete -> next: promotion gates for window-off
#   TIGHT     boots and completes, but a headroom bar misses -> ask the user (pool or headroom trade)
#   NO-FIT    N0 does not boot or errors
# Greedy vs R717's reference is reported: the window moves a layer (R579), so a different c1 fingerprint is expected
# (layout-dependent text) and is evidence, not a verdict. GPU ~35 min. Daily restored.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
UNIT=${UNIT:-$(basename "$0" .sh)}
R=${R:-/srv/qwen5090/results/$(date +%F)-$UNIT}; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
GREEDY=/srv/qwen5090/probes/fn_greedy.py
GATE=/srv/qwen5090/probes/fn_gate.sh
BENCH=/srv/qwen5090/probes/fn_bench.py
R717=${R717:-/srv/qwen5090/results/2026-09-24-r717-rows32-r4}
ORDER=${ORDER:-"D N0 N1"}
NONCE=$(( $(date +%s) % 100000 ))
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$UNIT
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== r723 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$GREEDY" "$GATE" "$BENCH" "$R717/greedy.jsonl"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
WENV=$(sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$/\1/p' "$LIVE" | head -1)
echo "$WENV" | grep -q 'EXL3_MTP_KV_WINDOW=' || { log "ABORT: served env has no EXL3_MTP_KV_WINDOW"; exit 3; }
NENV=$(echo "$WENV" | tr ' ' '\n' | grep -v '^EXL3_MTP_KV_WINDOW=' | xargs)
vram_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | xargs; }
log "W env $(echo $WENV | wc -w) keys; N env $(echo $NENV | wc -w) keys; order $ORDER; nonce $NONCE"

gpu_lock
log "lock held; served at entry: $(served_id || echo none)"
G=$R/greedy.jsonl; cp "$R717/greedy.jsonl" "$G"
boot(){ local tag=$1; shift
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { log "[$tag] NO BOOT: $(grep -aE 'Insufficient VRAM|out of memory|Error' "$R/boot-$tag.log" | tail -1 | cut -c1-160)"; sudo docker logs flashnext > "$R/container-$tag.log" 2>&1; return 1; }
  sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$R/env-$tag.txt"
  grep -q '^EXL3_NVME_TIER=' "$R/env-$tag.txt" && { log "[$tag] ABORT: NVMe tier on"; return 1; }
  log "[$tag] UP: window $(grep -c '^EXL3_MTP_KV_WINDOW=' "$R/env-$tag.txt"); $(grep -aoE "cache [0-9]+" "$R/boot-$tag.log" | tail -1); $(grep -aoE "policy '[^']*'" "$R/boot-$tag.log" | tail -1); $(grep -aoE 'VRAM free MiB [0-9/]+' "$R/boot-$tag.log" | tail -1)"; }
seq_run(){ local tag=$1
  python3 "$GREEDY" --url http://127.0.0.1:8022 --tag "$tag" --out "$G" > "$R/greedy-$tag.log" 2>&1
  python3 "$BENCH" --url http://127.0.0.1:8022/v1 --model "$MODEL" --tag ramp --kind prose --ctx 4000 --tokens 256 \
    --conc 1 2 3 4 5 6 7 8 --runs 1 --warmup-runs 0 --unique --distinct --salt $(( (NONCE + 4099) % 100000 )) \
    --out "$R/ramp-$tag.jsonl" > "$R/ramp-$tag.log" 2>&1
  RUNS=1 bash "$GATE" "$R/gate-$tag" http://127.0.0.1:8022/v1 "$MODEL" $(( (NONCE + 7919) % 100000 )) > "$R/gate-$tag.out" 2>&1
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  log "[$tag] ramp $(grep -c '"ok": true' "$R/ramp-$tag.jsonl" 2>/dev/null)/$(wc -l < "$R/ramp-$tag.jsonl" 2>/dev/null) ok; leg B $(grep -c '"ok": true' "$R/gate-$tag/bench-B.jsonl" 2>/dev/null)/$(wc -l < "$R/gate-$tag/bench-B.jsonl" 2>/dev/null) ok; OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$tag.log"); TORCH_CHECK $(grep -acE 'TORCH_CHECK|c10::Error' "$R/container-$tag.log"); tracebacks $(grep -ac Traceback "$R/container-$tag.log"); FREE-AFTER $(vram_free) MiB"
  tail -4 "$R/gate-$tag.out" 2>/dev/null | sed "s/^/  [gate $tag] /" | tee -a "$R/audit.log"; }
for t in $ORDER; do
  case $t in
    D)  boot D && seq_run D || { finish NO-BOOT; exit 3; } ;;
    N0) boot N0 "EXTRA_ENV=$NENV" CACHE=983040 && seq_run N0 ;;
    N1) boot N1 "EXTRA_ENV=$NENV" CACHE=999424 && seq_run N1 ;;
  esac
done
python3 "$GREEDY" --compare --ref ref --out "$G" > "$R/greedy-compare.txt" 2>&1
grep -aE "^GREEDY (D|N0|N1) |GREEDY-SUMMARY" "$R/greedy-compare.txt" | sed 's/^/  [greedy] /' | tee -a "$R/audit.log"
python3 - "$R/audit.log" <<'EOF' | tee -a "$R/audit.log"
import re, sys
t = open(sys.argv[1]).read()
def up(tag):
    m = re.search(rf"\[{tag}\] UP: .*?VRAM free MiB (\d+)/(\d+)", t); return tuple(map(int, m.groups())) if m else None
def after(tag):
    m = re.search(rf"\[{tag}\] ramp .*?leg B (\d+)/(\d+) ok; OOM (\d+); TORCH_CHECK (\d+); tracebacks (\d+); FREE-AFTER (\d+) (\d+)", t)
    return tuple(map(int, m.groups())) if m else None
d, n, da, na = up("D"), up("N0"), after("D"), after("N0")
print(f"HEADROOM D up {d} after {da and da[5:]} | N0 up {n} after {na and na[5:]} | N1 up {up('N1')} after {(after('N1') or (None,)*7)[5:]}")
if not n or not na: print("DECISION: NO-FIT (N0 did not boot or did not finish)"); sys.exit()
if na[2] or na[3] or na[4] or na[0] != na[1] or na[1] == 0: print(f"DECISION: NO-FIT (errors or leg B {na[0]}/{na[1]})"); sys.exit()
ok = d and da and n[0] >= d[0] - 32 and n[1] >= 835 and na[5] >= da[5] - 32 and na[6] >= da[6] - 32
print("DECISION: FITS" if ok else "DECISION: TIGHT (a headroom bar misses; the user decides)")
EOF
finish DONE
