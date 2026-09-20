#!/usr/bin/env bash
# R590 — where does the NVMe tier's decode tax come from, and does the tier ever pay it back?
#
# R589 measured the tax directly at one stream: tier ON 211.5 t/s against tier OFF 287.5, a 26.4 % decode tax,
# and no prefill benefit at all — the same ~45k prompt resubmitted on a fresh boot came back at 4.75 s TTFT with
# the tier against 4.47 s without it. Taken at face value that says switch the tier off. It should not be taken
# at face value yet, for one reason: the tier is AT ITS CAP. The daily's namespace holds 63.93 GiB against a
# 64.0 GiB limit, 13,995 pages, and the boot line reports 741 stranded pages pruned. A cache running permanently
# at its eviction threshold is doing work on every admission, and R589's own probe prompt was very likely pruned
# between the two boots, which would explain a benefit of exactly zero.
#
# So the tax has two candidate sources and they have opposite fixes:
#   (a) pruning at the cap — fix by raising the cap, changing the admission threshold, or pruning off the hot path
#   (b) inherent to having the tier attached at all — fix only by switching it off
# Three arms separate them. Same pool, same victim, same boot path, one boot each:
#   A  tier OFF                               (control; R589 read 287.5 here)
#   B  tier ON, the daily's namespace         (at cap, 13,995 pages — the state production is actually in)
#   C  tier ON, a FRESH EMPTY namespace       (same code path, same cap, nothing to prune)
# If C lands on A, the tax is (a) and the tier is worth keeping with a different cap policy. If C lands on B, the
# tax is (b) and the tier costs a quarter of decode for a prefix cache whose benefit has never been measured
# above zero. Either answer is actionable; the current state, a 26 % tax of unknown origin on the daily, is not.
#
# CONCURRENCY. R589 measured one stream. Production runs ~4.2 concurrent streams (R586), and a tier that reads
# and writes pages is likely to cost more, not less, when four jobs do it at once. Every arm is measured at c1
# and c4 so the number that gets quoted is the one that matches the seat.
#
# The tier is the user's design decision (2026-09-19: a persistent NVMe prefix tier, explicitly not a host-RAM
# tier). This round measures it; it does not switch anything off. Nothing is promoted and no launcher is edited.
# GPU TIMEBOX ~20 min. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-20-r590-tier-tax-source; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
FRESH=/srv/qwen5090/fast/exl3-nvme-r590   # a scratch tier directory; removed at the end
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r590] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|max_batch_size|chunk_size)' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
# Read the tier off the container, never off the arm label (R588 mislabelled three arms from a constant).
tierstate(){ if sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep -q '^EXL3_NVME_TIER='; then echo ON; else echo OFF; fi; }
# The engine prints one tier line at boot with the page and checkpoint counts. Keep it verbatim per arm.
tierline(){ sudo docker logs flashnext 2>&1 | grep -ai "nvme tier" | tail -1 | cut -c1-260; }
finish(){
  sudo rm -rf "$FRESH" 2>/dev/null
  if [ "$BOOTED" = 1 ]; then
    sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done
      log "daily: $(served_id) $(cfgline) tier $(tierstate)"; log "daily tier line: $(tierline)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R590 $1 ==="; }
trap 'log signal; finish ABORTED; exit 4' TERM INT HUP
[ -e "$LIVE" ] || { log "ABORT: missing $LIVE"; exit 3; }

export GPU_QUEUE_NAME=r590-tier-tax-source
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline) tier $(tierstate)"
log "daily tier at entry: $(tierline)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1

up(){ local L=$1 tag=$2 i st lp; shift 2
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" "$@" bash "$L" > "$R/boot-$tag.log" 2>&1 ) & lp=$!
  for i in $(seq 200); do
    [ "$(served_id)" = "$NEWM" ] && { wait $lp; return 0; }
    st=$(cstatus); case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 40 ] && { sleep 3; continue; }
      log "NO BOOT $tag ($st): $(sudo docker logs --tail 80 flashnext 2>&1 | grep -aE 'RuntimeError|Error|memory' | tail -1 | cut -c1-160)"
      kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1;; esac
    [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 0)" -ge 1 ] && {
      log "NO BOOT $tag (restart loop)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1; }
    sleep 3; done
  log "NO BOOT $tag (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }

SALT=$(( $(date +%s) % 100000 ))
VCTX=13000   # ~9.8k counted prompt tokens -- the victim depth R583/R585/R588/R589 all used.

# arm NAME WANT_TIER [ENV...]
arm(){ local name=$1 want=$2; shift 2
  log "=== arm $name: booting (want tier $want) ==="
  up "$LIVE" "$name" "$@" || { log "arm $name DID NOT BOOT — skipping"; return 0; }
  local ts; ts=$(tierstate)
  log "UP $name: $(cfgline); VRAM free $(vram); tier reads $ts (wanted $want)"
  [ "$ts" = "$want" ] || { log "ABORT: arm $name booted with tier $ts, not $want"; finish ABORTED; exit 3; }
  log "arm $name tier line: $(tierline)"
  # Warm both batch shapes before either is recorded; an unwarmed c4 reads low and would be charged to the tier.
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "warm-$name" --kind prose --tokens 256 \
    --conc 1 4 --runs 1 --ctx $VCTX --unique --salt $(( SALT + 11 )) --out "$R/warm.jsonl" > "$R/warm-$name.log" 2>&1
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "v-$name" \
    --kind prose --tokens 3000 --conc 1 4 --runs 2 --warmup-runs 0 --ctx $VCTX --unique \
    --salt $(( SALT + RANDOM )) --out "$R/records.jsonl" 2>&1 \
    | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$name]/" | cut -c1-220 | tee -a "$R/audit.log"
  alive || { log "FAIL: server not alive after $name"; finish ABORTED; exit 3; }
  log "arm $name tier line after traffic: $(tierline)"; }

sudo rm -rf "$FRESH" 2>/dev/null; sudo mkdir -p "$FRESH"
arm OFF   OFF NVME_TIER=
arm FULL  ON
arm FRESH ON  NVME_TIER="$FRESH" NVME_TIER_GB=64

python3 - "$R/records.jsonl" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("ok") and r.get("decode_tps"): d[(r["tag"],r.get("conc"))].append(r["decode_tps"])
def m(t,c): return st.mean(d[(t,c)]) if d.get((t,c)) else None
names={"OFF":"tier off (control)","FULL":"tier on, daily namespace at its 64 GiB cap","FRESH":"tier on, empty namespace"}
print("victim: greedy, 3,000 forced tokens at ~9.8k context, 2 runs, cache_size 999424\n")
print(f"{'arm':>6}  {'c1 t/s':>9} {'vs off':>8}   {'c4 t/s':>9} {'vs off':>8}   configuration")
b1,b4=m("v-OFF",1),m("v-OFF",4)
for a in ("OFF","FULL","FRESH"):
    x1,x4=m(f"v-{a}",1),m(f"v-{a}",4)
    if x1 is None and x4 is None: print(f"{a:>6}  {'missing':>9}"); continue
    r1=f"{x1/b1:.2f}x" if (x1 and b1) else "-"
    r4=f"{x4/b4:.2f}x" if (x4 and b4) else "-"
    print(f"{a:>6}  {x1 or 0:>9.1f} {r1:>8}   {x4 or 0:>9.1f} {r4:>8}   {names[a]}")
f1,u1=m("v-FRESH",1),m("v-FULL",1)
print()
if b1 and u1: print(f"Tax of the tier as production runs it: {100*(1-u1/b1):.1f} % at c1.")
if b1 and f1: print(f"Tax of the tier with nothing to prune:  {100*(1-f1/b1):.1f} % at c1.")
if b1 and u1 and f1:
    if abs(f1-b1) < abs(f1-u1):
        print("\nFRESH sits with OFF, so the tax is the CAP, not the tier: the daily's namespace runs permanently at")
        print("its eviction threshold and pays for it on the decode path. The fix is cap and admission policy, and")
        print("the tier itself is worth keeping.")
    else:
        print("\nFRESH sits with FULL, so the tax is INHERENT to having the tier attached, not the pruning. Keeping it")
        print("then costs about a quarter of decode for a prefix cache whose benefit R589 could not measure above 0.")
print("\nReference points at this victim shape, c1: R583 off 283, R588 off 275.4, R589 off 287.5; R585 on 236.4, R589 on 211.5.")
PY
log "VRAM free at the end: $(vram)"
finish DONE
