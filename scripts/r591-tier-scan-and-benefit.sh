#!/usr/bin/env bash
# R591 — the two things R589 and R590 between them still cannot answer about the NVMe prefix tier.
#
# (1) IS THE TAX THE BOOT-TIME OPEN SCAN? The overlay's own round-3b risk list says the open scan "reads every
#     stored checkpoint once per boot, in the background", competing with early restores, and that
#     EXL3_NVME_TIER_SCAN=0 turns it off once the checkpoints have been shown sound. The daily namespace holds
#     253 checkpoints. R590's FRESH arm has an empty namespace, so it has neither a scan NOR anything to prune,
#     and cannot tell those two apart. This round boots the DAILY namespace with the scan off. Against R590's
#     FULL arm — same namespace, same cap, scan on — the difference is the scan and nothing else.
#
# (2) DOES THE TIER EVER PAY? R589 submitted a ~45k prompt, restarted, resubmitted, and measured no benefit at
#     all (4.75 s against a 4.56 s cold TTFT). That test was run with `docker rm -f`, which is SIGKILL with no
#     grace period. The tier is write-ahead: pages become durable at a checkpoint, and the boot line reports
#     `741 stranded pruned`, which is what pages written after the last checkpoint look like at the next boot.
#     A prompt submitted seconds before a hard kill is the page most likely to be stranded and then dropped, so
#     R589 may have measured its own teardown. Here the restart is `docker stop -t 120` — SIGTERM and two
#     minutes of grace — and the boot line's page/checkpoint counts are recorded on both sides, so the round can
#     say whether the prompt was actually durable before it asks whether it was fast.
#
# Arms, one boot each, daily namespace throughout, victim = greedy 3,000 tokens at ~9.8k ctx at c1 and c4:
#   SCANOFF   tier ON, EXL3_NVME_TIER_SCAN=0      -> compare against R590's FULL arm (same namespace, scan on)
#   then, on that same boot, the benefit test with a graceful stop:
#     submit P (~45k) -> log the tier counters -> docker stop -t 120 -> boot -> log counters -> resubmit P
#   BASE      tier OFF, same victim               -> this round's own control, so nothing is compared across boots
#
# TRY 2 (2026-09-20): try 1 aborted because EXTRA_ENV REPLACES the launcher's default rather than adding to it.
# The default is 24 flags (EXL3_MTP_KV_WINDOW=16384, EXL3_SHARED_EXPERT_OVERLAP=1, the whole served set), and
# passing EXTRA_ENV="EXL3_NVME_TIER_SCAN=0" dropped every one of them, so the candidate went into a restart
# loop. The launcher default is now read out of the file and the new flag appended to it, the way R577 does it.
# Measurement only: nothing is promoted, no launcher is edited, the daily namespace is read and written exactly
# as production writes it. GPU TIMEBOX ~20 min. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-20-r591-tier-scan-and-benefit; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r591] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|max_batch_size|chunk_size)' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
tierstate(){ if sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep -q '^EXL3_NVME_TIER='; then echo ON; else echo OFF; fi; }
tierline(){ sudo docker logs flashnext 2>&1 | grep -ai "nvme tier" | tail -1 | cut -c1-300; }
# The whole point of the benefit half: stop the container the way a service is stopped, not the way a test is.
gstop(){ log "graceful stop (SIGTERM, up to 120 s)"; sudo docker stop -t 120 flashnext >/dev/null 2>&1; sudo docker rm -f flashnext >/dev/null 2>&1; }
finish(){
  if [ "$BOOTED" = 1 ]; then
    sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; gstop
    else log "restoring the daily"; gstop
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done
      log "daily: $(served_id) $(cfgline) tier $(tierstate)"; log "daily tier line: $(tierline)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R591 $1 ==="; }
trap 'log signal; finish ABORTED; exit 4' TERM INT HUP
[ -e "$LIVE" ] || { log "ABORT: missing $LIVE"; exit 3; }

export GPU_QUEUE_NAME=r591-tier-scan-and-benefit
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline) tier $(tierstate)"
log "tier at entry: $(tierline)"
BOOTED=1

# up LAUNCHER TAG [ENV...] -- unlike the earlier rounds this stops gracefully first, so a boot in this script
# never strands the pages the previous boot wrote.
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
# EXTRA_ENV replaces the launcher default, it does not extend it, so the served flag set has to be read out of
# the launcher and the new flag appended. Dropping it is what aborted try 1.
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE" | tail -1)
[ -n "$LENV" ] || { log "ABORT: could not read the launcher EXTRA_ENV default"; finish ABORTED; exit 3; }
log "launcher EXTRA_ENV default has $(echo $LENV | wc -w) flags"
SCANENV="$LENV EXL3_NVME_TIER_SCAN=0"
PSALT=$(( SALT + 60000 ))   # the one prompt used for the benefit test, identical across the restart

victim(){ local t=$1
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "v-$t" \
    --kind prose --tokens 3000 --conc 1 4 --runs 2 --warmup-runs 0 --ctx $VCTX --unique \
    --salt $(( SALT + RANDOM )) --out "$R/records.jsonl" 2>&1 \
    | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$t]/" | cut -c1-220 | tee -a "$R/audit.log"
  alive || { log "FAIL: server not alive after $t"; finish ABORTED; exit 3; }; }

bigprompt(){ local t=$1
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "P-$t" \
    --kind prose --tokens 16 --conc 1 --runs 1 --warmup-runs 0 --ctx 60000 --unique \
    --salt "$PSALT" --out "$R/records.jsonl" 2>&1 \
    | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[P-$t]/" | cut -c1-220 | tee -a "$R/audit.log"; }

warm(){ python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "warm-$1" --kind prose \
    --tokens 256 --conc 1 4 --runs 1 --ctx $VCTX --unique --salt $(( SALT + 11 )) --out "$R/warm.jsonl" \
    > "$R/warm-$1.log" 2>&1; }

log "=== arm BASE: tier OFF (this round's own control) ==="
up "$LIVE" BASE NVME_TIER= || { log "BASE did not boot"; finish ABORTED; exit 3; }
[ "$(tierstate)" = OFF ] || { log "ABORT: BASE booted with tier $(tierstate)"; finish ABORTED; exit 3; }
log "UP BASE: VRAM free $(vram); tier $(tierstate)"
warm BASE; victim BASE

log "=== arm SCANOFF: tier ON, daily namespace, EXL3_NVME_TIER_SCAN=0 ==="
up "$LIVE" SCANOFF EXTRA_ENV="$SCANENV" || { log "SCANOFF did not boot"; finish ABORTED; exit 3; }
[ "$(tierstate)" = ON ] || { log "ABORT: SCANOFF booted with tier $(tierstate)"; finish ABORTED; exit 3; }
log "UP SCANOFF: VRAM free $(vram); tier $(tierstate)"
log "SCANOFF tier line: $(tierline)"
sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep -i "NVME_TIER_SCAN" | while read -r l; do log "  env $l"; done
warm SCANOFF; victim SCANOFF

log "=== benefit test on the SCANOFF boot: submit P, stop gracefully, boot, resubmit P ==="
log "tier before P: $(tierline)"
bigprompt cold
log "tier after P:  $(tierline)"
up "$LIVE" AFTER EXTRA_ENV="$SCANENV" || { log "did not come back for the warm half"; finish ABORTED; exit 3; }
log "tier at the next boot: $(tierline)"
bigprompt warm

python3 - "$R/records.jsonl" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
d=collections.defaultdict(list); tt=collections.defaultdict(list); pt=collections.defaultdict(list)
for l in open(sys.argv[1]):
    r=json.loads(l)
    if not r.get("ok"): continue
    if r.get("decode_tps"): d[(r["tag"],r.get("conc"))].append(r["decode_tps"])
    if r.get("ttft_s") is not None: tt[r["tag"]].append(r["ttft_s"])
    if r.get("prompt_tokens"): pt[r["tag"]].append(r["prompt_tokens"])
def m(t,c): return st.mean(d[(t,c)]) if d.get((t,c)) else None
def mt(t): return st.mean(tt[t]) if tt.get(t) else None
print("victim: greedy, 3,000 forced tokens at ~9.8k context, 2 runs, cache_size 999424, daily namespace\n")
print(f"{'arm':>8}  {'c1 t/s':>9} {'vs off':>8}   {'c4 t/s':>9} {'vs off':>8}")
b1,b4=m("v-BASE",1),m("v-BASE",4)
for a in ("BASE","SCANOFF"):
    x1,x4=m(f"v-{a}",1),m(f"v-{a}",4)
    if x1 is None and x4 is None: print(f"{a:>8}  {'missing':>9}"); continue
    r1=f"{x1/b1:.2f}x" if (x1 and b1) else "-"
    r4=f"{x4/b4:.2f}x" if (x4 and b4) else "-"
    print(f"{a:>8}  {x1 or 0:>9.1f} {r1:>8}   {x4 or 0:>9.1f} {r4:>8}")
s1=m("v-SCANOFF",1)
print()
if b1 and s1:
    print(f"Tier tax with the open scan OFF: {100*(1-s1/b1):.1f} % at c1.")
    print("R589 measured 26.4 % with the scan on, and R590's FULL arm is the paired same-namespace number.")
    print("If this is near zero the scan was the tax and EXL3_NVME_TIER_SCAN=0 is the fix. If it is still ~26 %")
    print("the scan is not the tax and the cap or the tier itself is.")
c,w=mt("P-cold"),mt("P-warm")
n=st.mean(pt["P-cold"]) if pt.get("P-cold") else None
print(f"\nThe same ~{n:,.0f}-token prompt across a GRACEFUL restart (R589 used a hard kill):" if n else "\nBenefit test:")
if c is not None and w is not None:
    print(f"  cold {c:.2f} s TTFT -> after restart {w:.2f} s   ({c-w:+.2f} s)")
    if w < 0.6*c: print("  The tier served the prefix from disk. R589's zero was its own teardown.")
    else:         print("  Still no benefit with a graceful stop, so the hard kill was not the explanation.")
else:
    print("  one half missing")
print("\nRead the tier counter lines in audit.log around the restart: pages and checkpoints before the stop and")
print("at the next boot say whether the prompt was durable at all, which is prior to whether it was fast.")
PY
log "VRAM free at the end: $(vram)"
finish DONE
