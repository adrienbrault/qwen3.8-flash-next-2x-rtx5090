#!/usr/bin/env bash
# R592 — the NVMe tier's cost at the concurrency the box actually serves, measured well enough to act on.
#
# THE RECOMMENDATION TURNS ON THIS NUMBER AND IT IS CURRENTLY UNRESOLVED. R589 and R590 measured a 20–29 %
# decode tax at ONE stream, and R586 showed the tier returns essentially nothing in production (1,134 GiB
# written, 3 disk hits in 10,156 lookups over 7 hours). On that basis the tier looks like pure cost. But R590's
# c4 column disagrees with its own c1 column: the daily namespace read 0.71x at c1 and 0.97x at c4, which would
# mean the tax nearly vanishes at the concurrency production runs (R586: 4.21 mean streams). R590 ran two rounds
# per point and its c4 spread within a single arm was 157.4 against 170.3, so that column cannot carry a
# decision. If the tax really is ~3 % at c4 the tier is close to free where it matters and should stay; if it is
# ~25 % at c4 as well, it costs a quarter of production decode for three cache hits a day.
#
# So: two arms, four concurrencies, four runs each, warmed per shape, alternating so drift cannot land on one arm.
#   ON   tier on, the daily namespace exactly as production has it
#   OFF  tier off (NVME_TIER=)
#   c1, c2, c4, c8 -- c4 brackets production's 4.21 mean streams, c8 is the slot ceiling
# Arms are run ON, OFF, ON2, OFF2 on four boots. The repeat pair is the point: it gives a within-round estimate
# of boot-to-boot noise, which is the thing R590 did not have and the reason its c4 column is unreadable.
#
# EXTRA_ENV is read from the launcher and extended, never replaced -- replacing it drops the 22 served flags and
# the candidate will not boot (R591 try 1).
# Measurement only: nothing is promoted, no launcher is edited. GPU TIMEBOX ~30 min. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-20-r592-tier-concurrency; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r592] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|max_batch_size|chunk_size)' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
tierstate(){ if sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep -q '^EXL3_NVME_TIER='; then echo ON; else echo OFF; fi; }
tierline(){ sudo docker logs flashnext 2>&1 | grep -ai "nvme tier" | tail -1 | cut -c1-300; }
gstop(){ sudo docker stop -t 60 flashnext >/dev/null 2>&1; sudo docker rm -f flashnext >/dev/null 2>&1; }
finish(){
  if [ "$BOOTED" = 1 ]; then
    sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; gstop
    else log "restoring the daily"; gstop
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done
      log "daily: $(served_id) $(cfgline) tier $(tierstate)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R592 $1 ==="; }
trap 'log signal; finish ABORTED; exit 4' TERM INT HUP
[ -e "$LIVE" ] || { log "ABORT: missing $LIVE"; exit 3; }

export GPU_QUEUE_NAME=r592-tier-concurrency
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline) tier $(tierstate)"
BOOTED=1

up(){ local L=$1 tag=$2 i st lp; shift 2
  gstop; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
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
VCTX=13000
CONCS="1 2 4 8"

# arm TAG WANT [ENV...]
arm(){ local tag=$1 want=$2; shift 2
  log "=== $tag: booting (want tier $want) ==="
  up "$LIVE" "$tag" "$@" || { log "$tag DID NOT BOOT — skipping"; return 0; }
  local ts; ts=$(tierstate)
  [ "$ts" = "$want" ] || { log "ABORT: $tag booted with tier $ts, not $want"; finish ABORTED; exit 3; }
  log "UP $tag: $(cfgline); VRAM free $(vram); tier $ts"
  [ "$want" = ON ] && log "$tag tier line: $(tierline)"
  # Warm every shape this arm will be scored on, on this boot. An unwarmed c8 reads low and the deficit would be
  # charged to whichever arm happened to meet it first.
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "warm-$tag" --kind prose \
    --tokens 256 --conc $CONCS --runs 1 --ctx $VCTX --unique --salt $(( SALT + 11 )) \
    --out "$R/warm.jsonl" > "$R/warm-$tag.log" 2>&1
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$tag" \
    --kind prose --tokens 3000 --conc $CONCS --runs 4 --warmup-runs 0 --ctx $VCTX --unique \
    --salt $(( SALT + RANDOM )) --out "$R/records.jsonl" 2>&1 \
    | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-220 | tee -a "$R/audit.log"
  alive || { log "FAIL: server not alive after $tag"; finish ABORTED; exit 3; }
  [ "$want" = ON ] && log "$tag tier after traffic: $(tierline)"
  return 0; }

# Alternated, and each arm run twice: the ON/ON2 and OFF/OFF2 gaps are this round's noise floor.
arm ON   ON
arm OFF  OFF NVME_TIER=
arm ON2  ON
arm OFF2 OFF NVME_TIER=

python3 - "$R/records.jsonl" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("ok") and r.get("decode_tps"): d[(r["tag"],r.get("conc"))].append(r["decode_tps"])
CONCS=[1,2,4,8]
def g(tag,c): return d.get((tag,c)) or []
def mean(v): return st.mean(v) if v else None
print("victim: greedy, 3,000 forced tokens at ~9.8k context, 4 runs per point, cache_size 999424")
print("tier ON uses the daily namespace exactly as production has it.\n")
print(f"{'conc':>5}  {'ON':>8} {'ON2':>8} {'OFF':>8} {'OFF2':>8}   {'tax':>7}  {'noise':>7}")
rows=[]
for c in CONCS:
    on,on2,off,off2=mean(g("ON",c)),mean(g("ON2",c)),mean(g("OFF",c)),mean(g("OFF2",c))
    vals=[x for x in (on,on2,off,off2) if x]
    if len(vals)<4: print(f"{c:>5}  incomplete"); continue
    onm,offm=(on+on2)/2,(off+off2)/2
    tax=100*(1-onm/offm)
    # The noise floor is the bigger of the two same-configuration gaps, as a percentage of the OFF mean.
    noise=100*max(abs(on-on2),abs(off-off2))/offm
    rows.append((c,tax,noise))
    print(f"{c:>5}  {on:>8.1f} {on2:>8.1f} {off:>8.1f} {off2:>8.1f}   {tax:>6.1f}% {noise:>6.1f}%")
print("\nTax is (1 - mean ON / mean OFF). Noise is the larger same-configuration repeat gap, on the same scale:")
print("a tax smaller than its noise column is not a measurement, it is a coin flip.")
print("\nReference: R586 puts production at 4.21 mean concurrent streams, so the c4 row is the one that decides this.")
for c,tax,noise in rows:
    if c!=4: continue
    if tax > 3*noise and tax > 10:
        print(f"\nc4 tax {tax:.1f}% against a {noise:.1f}% noise floor. The tier costs real throughput at the")
        print("concurrency production runs, and R586 measured what it returns for that: 3 disk hits in 10,156")
        print("lookups over 7 hours, for 1,134 GiB written. Recommend switching it off -- a user decision, since")
        print("the persistent NVMe tier is a deliberate design choice and turning it off is not a promotion.")
    elif tax < noise:
        print(f"\nc4 tax {tax:.1f}% is inside the {noise:.1f}% noise floor. The tier is close to free where it")
        print("matters, whatever it costs at one stream, and the case for removing it rests on NVMe wear alone.")
    else:
        print(f"\nc4 tax {tax:.1f}% against a {noise:.1f}% noise floor -- real but not decisive on its own.")
PY
log "VRAM free at the end: $(vram)"
finish DONE
