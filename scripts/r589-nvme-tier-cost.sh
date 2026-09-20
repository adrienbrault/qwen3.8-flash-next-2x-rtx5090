#!/usr/bin/env bash
# R589 — backlog S19: what does the NVMe prefix tier cost decode, and what does it buy prefill?
#
# WHY THIS IS NOW A DECISION AND NOT A CURIOSITY. Three rounds measured the same victim shape (greedy, 3,000
# forced tokens at ~9.8k context, one stream) and split cleanly by whether the tier was mounted:
#   R583  tier OFF   283 t/s
#   R585  tier ON    236.4
#   R588  tier OFF   275.4    (all three chunk arms 269-275, and the tier was off for an unintended reason)
# R588's arms were supposed to run with the tier on. They did not: R587 promoted the daily image to
# `tabbyapi:mtpwin-r2-metrics1` while the launcher still enabled the tier only when IMG was literally
# `tabbyapi:mtpwin-r2`, so every boot from that launcher — the daily included — served with no prefix tier and
# nothing in the promotion gates looks at the tier. Fixed 2026-09-20 09:10 UTC (the condition now tests against
# DAILY_IMG), and the accident is what makes the comparison above possible: two independent tier-OFF rounds at
# 275-283 against one tier-ON round at 236. That is a ~14 % decode tax, large enough to be worth confirming
# properly rather than inferring across rounds that differ in warm-up.
#
# BOTH SIDES OF THE TRADE, because the cost alone does not decide it. The tier exists to serve prefixes from disk
# after a restart, and production prompts are 97 % cached (R586). A round that measures only the decode tax would
# recommend turning off a cache without ever measuring what the cache does.
#   Arm ON   boot with the tier (launcher default) -> warm -> victim x3            -> submit P -> restart -> submit P
#   Arm OFF  boot with NVME_TIER= (explicit empty) -> warm -> victim x3            -> submit P -> restart -> submit P
# P is one ~45k-token prompt with a per-arm salt, so the second submission in each arm is the same bytes as the
# first but on a FRESH BOOT with an empty VRAM cache. With the tier on, that second TTFT should be served from
# disk; with it off it must be a full prefill. The gap between the two arms' second TTFT is what the tier buys.
# Each arm gets its own salt so arm OFF cannot be served pages arm ON wrote.
#
# Measurement only: nothing is promoted, no launcher is edited. GPU TIMEBOX ~20 min. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-20-r589-nvme-tier-cost; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r589] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|max_batch_size|chunk_size)' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
# Read the tier straight off the container rather than trusting the arm label -- believing a label is exactly how
# R588 recorded "NVMe tier on" for three arms that had none.
tierstate(){ if sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep -q '^EXL3_NVME_TIER='; then echo ON; else echo OFF; fi; }
finish(){
  if [ "$BOOTED" = 1 ]; then
    sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done
      log "daily: $(served_id) $(cfgline) tier $(tierstate)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R589 $1 ==="; }
trap 'log signal; finish ABORTED; exit 4' TERM INT HUP
[ -e "$LIVE" ] || { log "ABORT: missing $LIVE"; exit 3; }

export GPU_QUEUE_NAME=r589-nvme-tier-cost
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline) tier $(tierstate)"
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
VCTX=13000   # ~9.8k counted prompt tokens -- R583/R585/R588's victim depth, kept identical so all four compare.

victim(){ local t=$1
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "victim-$t" \
    --kind prose --tokens 3000 --conc 1 --runs 3 --warmup-runs 0 --ctx $VCTX --unique \
    --salt $(( SALT + RANDOM )) --out "$R/records.jsonl" 2>&1 \
    | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$t]/" | cut -c1-220 | tee -a "$R/audit.log"
  alive || { log "FAIL: server not alive after $t"; return 1; }; }

# bigprompt TAG SALT -- one ~45k-token prompt, 16 tokens out. The same salt twice is the same bytes twice, so the
# second call on a fresh boot measures whatever survived the restart.
bigprompt(){ local t=$1 s=$2
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "P-$t" \
    --kind prose --tokens 16 --conc 1 --runs 1 --warmup-runs 0 --ctx 60000 --unique \
    --salt "$s" --out "$R/records.jsonl" 2>&1 \
    | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[P-$t]/" | cut -c1-220 | tee -a "$R/audit.log"; }

# arm NAME [ENV...] -- boot, confirm the tier state matches the label, victim x3, P, restart, P again.
arm(){ local name=$1 want=$2; shift 2
  log "=== arm $name: booting (want tier $want) ==="
  up "$LIVE" "$name" "$@" || { log "arm $name DID NOT BOOT — skipping"; return 0; }
  local ts; ts=$(tierstate)
  log "UP $name: $(cfgline); VRAM free $(vram); tier reads $ts (wanted $want)"
  [ "$ts" = "$want" ] || { log "ABORT: arm $name booted with tier $ts, not $want — the arm would be mislabelled"; finish ABORTED; exit 3; }
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "warm-$name" --kind prose --tokens 256 \
    --conc 1 --runs 1 --ctx $VCTX --unique --salt $(( SALT + 11 )) --out "$R/warm.jsonl" > "$R/warm-$name.log" 2>&1
  log "arm $name: victim x3"
  victim "$name" || { finish ABORTED; exit 3; }
  local ps=$(( SALT + 50000 + ${#name} * 977 ))
  log "arm $name: P cold (first submission of this prompt)"
  bigprompt "${name}-cold" "$ps"
  log "arm $name: restarting, same tier setting, then resubmitting the same prompt"
  up "$LIVE" "$name-again" "$@" || { log "arm $name did not come back — skipping its warm half"; return 0; }
  log "UP $name again: tier reads $(tierstate)"
  bigprompt "${name}-warm" "$ps"; }

arm ON  ON
arm OFF OFF NVME_TIER=

python3 - "$R/records.jsonl" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
dec=collections.defaultdict(list); tt=collections.defaultdict(list); pt=collections.defaultdict(list)
for l in open(sys.argv[1]):
    r=json.loads(l)
    if not r.get("ok"): continue
    if r.get("decode_tps"): dec[r["tag"]].append(r["decode_tps"])
    if r.get("ttft_s") is not None: tt[r["tag"]].append(r["ttft_s"])
    if r.get("prompt_tokens"): pt[r["tag"]].append(r["prompt_tokens"])
def m(d,t): return st.mean(d[t]) if d.get(t) else None
print("victim: greedy, 3,000 forced tokens at ~9.8k context, 1 stream, 3 runs, cache_size 999424\n")
print(f"{'tier':>5}  {'decode t/s':>11} {'vs tier OFF':>12}")
off=m(dec,"victim-OFF")
for a in ("ON","OFF"):
    v=m(dec,f"victim-{a}")
    if v is None: print(f"{a:>5}  {'missing':>11}"); continue
    print(f"{a:>5}  {v:>11.1f} {(v/off if off else 0):>11.2f}x")
print("\nThe same ~45k-token prompt, submitted once and then again on a fresh boot (TTFT seconds):")
print(f"{'tier':>5}  {'cold':>8} {'after restart':>14} {'prompt tokens':>14}")
for a in ("ON","OFF"):
    c,w=m(tt,f"P-{a}-cold"),m(tt,f"P-{a}-warm")
    n=m(pt,f"P-{a}-cold")
    cs=f"{c:.2f}" if c is not None else "-"
    ws=f"{w:.2f}" if w is not None else "-"
    ns=f"{n:,.0f}" if n else "-"
    print(f"{a:>5}  {cs:>8} {ws:>14} {ns:>14}")
on_w,off_w=m(tt,"P-ON-warm"),m(tt,"P-OFF-warm")
on_d,off_d=m(dec,"victim-ON"),m(dec,"victim-OFF")
print()
if on_d and off_d:
    print(f"Decode tax of the tier: {100*(1-on_d/off_d):.1f} % ({on_d:.1f} against {off_d:.1f} t/s).")
if on_w and off_w:
    print(f"Prefill saved after a restart: {off_w-on_w:.2f} s on a ~45k-token prompt "
          f"({on_w:.2f} against {off_w:.2f} s TTFT).")
if on_d and off_d and on_w and off_w:
    lost=(off_d-on_d)
    print(f"\nRead the trade per long generation: the tier costs {lost:.1f} t/s of decode, so a 3,000-token reply")
    print(f"pays {3000/on_d-3000/off_d:.1f} s more, against {off_w-on_w:.2f} s saved on a cold ~45k prompt. Production")
    print("prompts were 97 % cached (R586), but most of that is the in-VRAM cache, which both arms have.")
print("\nReference points at this victim shape: R583 tier OFF 283, R585 tier ON 236.4, R588 tier OFF 275.4.")
PY
log "VRAM free at the end: $(vram)"
finish DONE
