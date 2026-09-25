#!/usr/bin/env bash
# R726 (2026-09-25): the memory OC is OFF, and has been since at least 2026-09-19. Restore it, measure what it is worth
# on the Flash-Next daily, and probe higher offsets.
# gpu-tune.service applied +4500 on both cards at boot 2026-09-02 20:30 (legacy nvmlDeviceSetMemClkVfOffset, readback
# OK). On 2026-09-25 both cards read offset 0 and 13,801 MHz under load (R137 measured 16,051 MHz at +4500). Every
# Flash-Next log from R520b (2026-09-19) on shows 13,801 MHz, so every Flash-Next number, published ones included, is
# stock memory clock. No reboot happened since 09-02; the reset's cause is not established (candidates: a driver-side
# reset after the Xid 13/31 faults, e.g. 2026-09-24 01:33; nothing in the repo writes the offset except gpu-tune.sh and
# R137's revert path).
# R137 (27B vLLM TP2, 2026-08-31): +4500 -> 1,641 / 1,632 GB/s vs 1,422 / 1,421 stock (+15 %), decode +4 % (TP2 decode
# co-bound by allreduce latency). The Flash-Next daily splits layers across the cards (no TP), so more of its step may
# be bandwidth-bound.
# Part 1, engine DOWN: per-card DRAM bandwidth (probes/membw.py, 1 GiB d2d copy, host-side memclk sampling) at
#   offsets LADDER (default 0 4500 5000 5500 6000), both cards set together. Stop climbing at the first rung where
#   (a) any new NVRM Xid appears in the kernel log since unit start, (b) the copy is not exact, or (c) either card's
#   bandwidth fails to rise >= 1.0 % over the previous rung (GDDR7 EDC retries eat the clock: the knee).
#   BEST = the highest rung that passed all three.
# Part 2, one boot of the daily (NVME_TIER=), offsets switched LIVE (same boot, paired, no boot noise):
#   O0 -> O45 -> O0b -> O45b -> OB (BEST, if above 4500) ; per state: fn_greedy (c1 canonical prompts: a corruption
#   canary, compared to O0) and fn_bench --distinct code+prose c1/c4/c8, 512 tokens, 1 warm-up + 2 runs.
# DECISION (pre-registered):
#   decode gain at +4500 = mean over (O0,O45) (O0b,O45b) of per-stream decode ratios, per (kind, conc).
#   KEEP-4500   greedy O45/O45b identical to O0, 0 new Xid, gain >= 1.00 everywhere -> +4500 stays (the documented daily
#               state since R137; gpu-tune.service) and the unit leaves it applied.
#   REVERT      any greedy divergence at +4500 beyond O0b's own, or a new Xid -> offsets back to 0, report.
#   HIGHER      reported only: OB's gain over O45 and its greedy result. Serving above +4500 is the user's call.
# The unit leaves +4500 applied on KEEP-4500, 0 otherwise. GPU ~30 min. Daily restored.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
UNIT=${UNIT:-$(basename "$0" .sh)}
R=${R:-/srv/qwen5090/results/$(date +%F)-$UNIT}; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
API=http://127.0.0.1:8022/v1
BENCH=/srv/qwen5090/probes/fn_bench.py
GREEDY=/srv/qwen5090/probes/fn_greedy.py
MEMBW=/srv/qwen5090/probes/membw.py
LADDER=${LADDER:-"0 4500 5000 5500 6000"}
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$UNIT
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
KEEP=0
setoff(){ sudo python3 - "$1" <<'PY'
import sys, pynvml as N
N.nvmlInit(); t = int(sys.argv[1]); out = []
for i in range(N.nvmlDeviceGetCount()):
    h = N.nvmlDeviceGetHandleByIndex(i); N.nvmlDeviceSetMemClkVfOffset(h, t); out.append(N.nvmlDeviceGetMemClkVfOffset(h))
print(" ".join(map(str, out))); sys.exit(0 if all(o == t for o in out) else 1)
PY
}
finish(){ if [ $KEEP = 1 ]; then log "leaving +4500: readback $(setoff 4500)"; else log "reverting offsets to 0: readback $(setoff 0)"; fi
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== r726 $1 ==="; }
trap 'log "signal"; KEEP=0; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$BENCH" "$GREEDY" "$MEMBW"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
IMG=$(grep -m1 -oE '^DAILY_IMG=[^ ]+' "$LIVE" | cut -d= -f2 | tr -d "'\"")
xids(){ sudo journalctl -k --since "$T0" --no-pager 2>/dev/null | grep -c 'NVRM: Xid'; }

gpu_lock
T0=$(date '+%F %T')
log "lock held; served at entry: $(served_id || echo none); offsets at entry: $(sudo python3 -c 'import pynvml as N;N.nvmlInit();print([N.nvmlDeviceGetMemClkVfOffset(N.nvmlDeviceGetHandleByIndex(i)) for i in range(N.nvmlDeviceGetCount())])'); image $IMG; ladder $LADDER"
served_stop; wait_unserved 45
for i in $(seq 60); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "${used:-99999}" -lt 1024 ] && break; sleep 2; done
[ "${used:-99999}" -lt 1024 ] || { log "ABORT: GPU memory not released"; finish ABORTED; exit 3; }

# ---- part 1: bandwidth ladder, engine down ----
bw(){ local off=$1 d out
  for d in 0 1; do
    ( for k in $(seq 40); do nvidia-smi -i $d --query-gpu=clocks.mem --format=csv,noheader,nounits; sleep 0.1; done ) > "$R/memclk-$off-$d.txt" &
    out=$(sudo docker run --rm --gpus device=$d --entrypoint python3 -v "$MEMBW":/membw.py:ro "$IMG" /membw.py --mib 1024 --iters 300 2>&1 | grep -a MEMBW)
    wait
    echo "$d $out host_memclk_max=$(sort -n "$R/memclk-$off-$d.txt" | tail -1)"
  done; }
PREV0=0; PREV1=0; BEST=0
for off in $LADDER; do
  setoff $off > "$R/setoff-$off.txt" || { log "rung +$off: offset did not stick ($(cat "$R/setoff-$off.txt"))"; break; }
  sleep 2
  res=$(bw $off); echo "$res" > "$R/bw-$off.txt"
  g0=$(echo "$res" | awk '$1==0' | grep -oE 'gbps=[0-9.]+' | cut -d= -f2); g1=$(echo "$res" | awk '$1==1' | grep -oE 'gbps=[0-9.]+' | cut -d= -f2)
  eq=$(echo "$res" | grep -c 'copy_equal=True'); nx=$(xids)
  log "rung +$off: cuda:0 ${g0:-?} GB/s, cuda:1 ${g1:-?} GB/s, exact copies $eq/2, new Xid $nx | $(echo "$res" | grep -oE 'host_memclk_max=[0-9]+' | xargs)"
  [ "$nx" -gt 0 ] && { log "STOP: new Xid at +$off"; break; }
  [ "$eq" = 2 ] && [ -n "$g0" ] && [ -n "$g1" ] || { log "STOP: inexact copy or no measurement at +$off"; break; }
  if [ "$off" != 0 ]; then
    up=$(python3 -c "print(int($g0 >= $PREV0 * 1.01 and $g1 >= $PREV1 * 1.01))")
    [ "$up" = 1 ] || { log "STOP: bandwidth stopped rising at +$off (knee)"; break; }
  fi
  PREV0=$g0; PREV1=$g1; BEST=$off
done
log "LADDER BEST +$BEST"
[ "$(xids)" -gt 0 ] && { log "new Xid during the ladder: reverting and restoring the daily"; finish XID; exit 3; }

# ---- part 2: decode A/B on one boot, offsets switched live ----
setoff 0 >/dev/null
env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= bash "$LIVE" > "$R/boot.log" 2>&1 \
  && wait_served_id "$MODEL" 200 8 || { log "NO BOOT"; finish NO-BOOT; exit 3; }
log "booted: $(grep -aoE "cache [0-9]+" "$R/boot.log" | tail -1); $(grep -aoE 'VRAM free MiB [0-9/]+' "$R/boot.log" | tail -1)"
G=$R/greedy.jsonl; REC=$R/records.jsonl
state(){ local tag=$1 off=$2
  setoff $off > /dev/null || { log "[$tag] offset +$off did not stick"; return 1; }; sleep 2
  python3 "$GREEDY" --url http://127.0.0.1:8022 --tag "$tag" --out "$G" > "$R/greedy-$tag.log" 2>&1
  for kind in code prose; do
    python3 "$BENCH" --url "$API" --model "$MODEL" --tag "$tag-$kind" --kind $kind --distinct --tokens 512 \
      --conc 1 4 8 --warmup-runs 1 --runs 2 --out "$REC" > "$R/bench-$tag-$kind.log" 2>&1
  done
  log "[$tag] offset +$off done; memclk now $(nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits | xargs); new Xid $(xids); engine $(served_id || echo DOWN)"; }
state O0 0; state O45 4500; state O0b 0; state O45b 4500
[ "$BEST" -gt 4500 ] && state OB "$BEST"
sudo docker logs flashnext > "$R/container.log" 2>&1
python3 "$GREEDY" --compare --ref O0 --out "$G" > "$R/greedy-compare.txt" 2>&1
grep -aE "^GREEDY |GREEDY-SUMMARY" "$R/greedy-compare.txt" | sed 's/^/  [greedy] /' | tee -a "$R/audit.log"
python3 - "$REC" "$R/greedy-compare.txt" "$(xids)" "$BEST" <<'EOF' | tee "$R/summary.txt" | tee -a "$R/audit.log"
import json, sys, re, statistics as st
rec, gc, nx, best = sys.argv[1], open(sys.argv[2]).read(), int(sys.argv[3]), int(sys.argv[4])
d = {}
for ln in open(rec):
    try: r = json.loads(ln)
    except Exception: continue
    if r.get("ok") and r.get("decode_tps"): d.setdefault((r["tag"], r.get("conc")), []).append(r["decode_tps"])
med = lambda t, k, c: st.median(d[(f"{t}-{k}", c)]) if d.get((f"{t}-{k}", c)) else None
worst = 9.0
for k in ("code", "prose"):
    for c in (1, 4, 8):
        a = [med(t, k, c) for t in ("O0", "O45", "O0b", "O45b")]
        if None in a: print(f"{k} c{c}: MISSING {a}"); worst = 0; continue
        g = (a[1] / a[0] + a[3] / a[2]) / 2; worst = min(worst, g)
        ob = med("OB", k, c)
        print(f"{k} c{c}: per-stream O0 {a[0]:.1f} O45 {a[1]:.1f} O0b {a[2]:.1f} O45b {a[3]:.1f} -> +4500 gain {g:.3f}"
              + (f" | OB(+{best}) {ob:.1f} vs O45b {ob / a[3]:.3f}" if ob else ""))
div = {m.group(1): (0 if m.group(2) == "IDENTICAL" else int(m.group(2).split()[0])) for m in re.finditer(r"^GREEDY (\S+) vs \S+: \d+ identical, (IDENTICAL|\d+ DIVERGENT)", gc, re.M)}
print(f"greedy divergences vs O0: {div or 'see greedy-compare.txt'}; new Xid {nx}")
bad45 = any(v > div.get("O0b", 0) for t, v in div.items() if t.startswith("O45"))
if nx > 0 or bad45: print("DECISION: REVERT")
elif worst >= 1.00: print(f"DECISION: KEEP-4500 (worst +4500 gain {worst:.3f})")
else: print(f"DECISION: KEEP-4500-NO-GAIN (worst {worst:.3f}; kept as the documented state, gain not shown)")
EOF
grep -q '^DECISION: KEEP' "$R/summary.txt" && KEEP=1
finish DONE
