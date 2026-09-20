#!/usr/bin/env bash
# R588 — backlog S20: does chunk_size change the PRICE of prefill interference?
#
# R585 found what the production gap actually is: a long generation loses 35 % of its decode rate when fresh ~45k
# prompts keep arriving beside it (236.4 t/s alone -> 154.5 under arrivals every 8 s), and B-vs-C proved it is the
# prefill SIZE that does it, not the presence of other traffic. chunk_size is the one knob that directly sets how
# that cost is paid out: at the served 2048, a 45k prompt is 22 forward passes the decoder does not get. At 1024 it
# is 44 shorter ones, at 512 it is 88 shorter still. Total prefill work is roughly constant, so the question is not
# how much work there is but how it is GRANULATED against the decode stream -- and whether finer granulation lets
# decode steps slot in sooner, or just adds per-chunk overhead and makes both sides worse.
#
# This is the cheapest of the three S18-S20 ideas and the only one that needs no new code: CHUNK is already an env
# knob on the launcher (launch-flashnext.sh:175, allowed 256|512|768|1024|1536|2048|4096).
#
# WHAT IS HELD FIXED. CACHE=999424 is pinned on every arm. The launcher's own comment (lines 171-173) says the
# loader's budget check runs every module on a dummy chunk of chunk_size tokens and keeps headroom for the largest
# transient it measured, so a smaller chunk BUYS POOL. Letting the pool float would confound chunk against KV size
# and the round would measure nothing. Pinning it means the extra headroom shows up as free VRAM instead, which is
# logged per arm -- that number is the input to the "+20 % KV is worth a small prefill dip" rule, for free.
# R574 closed chunk 4096 on boot grounds at the old 966,656 pool; the pool is larger now, so 4096 is not retried.
# The question here is the interference profile below the served value, which R574 never looked at.
#
# Arms, one boot each, NVMe tier left ON (production runs with it on, and R585 measured with it on):
#   chunk 512 / 1024 / 2048      x     A victim alone (control)
#                                      B victim + fresh ~45k-token prompts every 8 s  (= R585 arm B, the one that
#                                                                                       reproduced production)
# Victim is R585's verbatim: greedy, 3,000 forced tokens at ~9.8k context, 1 stream, 2 runs.
# What decides it: the B/A ratio per chunk. 2048 is the baseline at R585's 0.65x. A chunk that lifts that ratio
# without costing the victim its alone-rate is a free win on the daily; one that lifts B by dropping A is not.
# The interferer's own ttft_s is recorded for both halves of the trade: finer chunks should cost the ARRIVING
# request latency even where they buy the RUNNING one throughput, and that trade is the actual decision.
#
# Measurement only: nothing is promoted, no launcher is edited. GPU TIMEBOX ~25 min. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-20-r588-chunk-vs-interference; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
CACHE_PIN=999424
CHUNKS="512 1024 2048"
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0; NOISE_PIDS=()
log(){ echo "$(date -Is) [r588] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|max_batch_size|chunk_size)' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
# Every arm must leave the box quiet, or the next arm measures this one's leftovers (R585's rule, kept).
stop_noise(){ local p; for p in "${NOISE_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
  for p in "${NOISE_PIDS[@]:-}"; do [ -n "$p" ] && wait "$p" 2>/dev/null; done; NOISE_PIDS=()
  pkill -f "fn_bench.py.*r588-noise" 2>/dev/null
  # Wait for the server to go idle rather than trusting the kill: an in-flight 45k prefill outlives its client.
  local i; for i in $(seq 60); do
    [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd+ | bc)" -lt 10 ] && break; sleep 2; done; }
finish(){
  stop_noise
  if [ "$BOOTED" = 1 ]; then
    sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done; log "daily: $(served_id) $(cfgline)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R588 $1 ==="; }
trap 'log signal; finish ABORTED; exit 4' TERM INT HUP
[ -e "$LIVE" ] || { log "ABORT: missing $LIVE"; exit 3; }

export GPU_QUEUE_NAME=r588-chunk-vs-interference
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"
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
VCTX=13000   # ~9.8k counted prompt tokens -- R585's victim depth, kept identical so the two rounds compare.

# noise_loop CHUNK -- fresh ~45k-token arrivals until killed. Tagged r588-noise-<chunk> so stop_noise finds them
# and the analysis can read the interferer's own ttft_s per arm.
noise_loop(){ local ch=$1 k=0
  while :; do
    k=$((k+1))
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "r588-noise-$ch" \
      --kind prose --tokens 64 --conc 1 --runs 1 --ctx 60000 --unique \
      --salt $(( SALT + 40000 + k*613 + RANDOM )) --out "$R/noise.jsonl" > /dev/null 2>&1
    sleep 8
  done; }

# victim TAG -- one 3,000-token generation, measured. Two runs so a single unlucky round cannot carry the arm.
victim(){ local t=$1
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "victim-$t" \
    --kind prose --tokens 3000 --conc 1 --runs 2 --warmup-runs 0 --ctx $VCTX --unique \
    --salt $(( SALT + RANDOM )) --out "$R/records.jsonl" 2>&1 \
    | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$t]/" | cut -c1-220 | tee -a "$R/audit.log"
  alive || { log "FAIL: server not alive after $t"; return 1; }; }

for CH in $CHUNKS; do
  log "=== chunk $CH: booting (CACHE pinned at $CACHE_PIN) ==="
  if ! up "$LIVE" "chunk$CH" CHUNK=$CH CACHE=$CACHE_PIN; then
    log "chunk $CH DID NOT BOOT — skipping this arm, continuing the round"
    echo "chunk $CH: no boot" >> "$R/vram.txt"; continue; fi
  V=$(vram); log "UP chunk $CH: $(cfgline); VRAM free $V"
  echo "chunk $CH: $V" >> "$R/vram.txt"
  # Warm the batch shapes once per boot; an unwarmed round reads low and would be charged to the chunk size.
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "warm$CH" --kind prose --tokens 256 \
    --conc 1 --runs 1 --ctx $VCTX --unique --salt $(( SALT + CH )) --out "$R/warm.jsonl" > "$R/warm-$CH.log" 2>&1
  log "chunk $CH arm A: victim alone"
  victim "${CH}A" || { finish ABORTED; exit 3; }
  log "chunk $CH arm B: victim + ~45k-token arrivals every 8 s"
  noise_loop "$CH" & NOISE_PIDS+=($!)
  sleep 10                      # let one big prefill already be in flight when the victim starts
  victim "${CH}B" || { finish ABORTED; exit 3; }
  stop_noise
done

python3 - "$R/records.jsonl" "$R/noise.jsonl" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
rows=collections.defaultdict(list); ttft=collections.defaultdict(list)
for path in sys.argv[1:]:
    try: f=open(path)
    except OSError: continue
    for l in f:
        r=json.loads(l)
        if not r.get("ok"): continue
        if r.get("decode_tps"): rows[r["tag"]].append(r["decode_tps"])
        if r.get("ttft_s"): ttft[r["tag"]].append(r["ttft_s"])
def m(t): return st.mean(rows[t]) if rows.get(t) else None
CH=[512,1024,2048]
print("victim: greedy, 3,000 forced tokens at ~9.8k context, 1 stream, NVMe tier on, cache_size pinned 999424")
print("interference: fresh ~45k-token prompts, 64 tokens out, one every 8 s (= R585 arm B)\n")
print(f"{'chunk':>6}  {'alone':>8} {'under noise':>12} {'B/A':>7}   {'noise TTFT s':>13}")
base=None
for c in CH:
    a,b=m(f"victim-{c}A"),m(f"victim-{c}B")
    if a is None and b is None: print(f"{c:>6}  {'(no boot)':>8}"); continue
    t=ttft.get(f"r588-noise-{c}") or []
    tt=f"{st.median(t):.1f}" if t else "-"
    ratio=f"{b/a:.2f}x" if (a and b) else "-"
    print(f"{c:>6}  {a if a else 0:>8.1f} {b if b else 0:>12.1f} {ratio:>7}   {tt:>13}")
    if c==2048: base=(a,b)
print("\nR585 at the served chunk 2048 read 236.4 alone and 154.5 under this exact noise (0.65x).")
if base and base[0] and base[1]:
    print(f"This round's 2048 arm: {base[0]:.1f} alone, {base[1]:.1f} under noise ({base[1]/base[0]:.2f}x) — the")
    print("reproduction check. A 2048 arm far from 0.65x means something else moved and the sweep is not readable.")
for c in CH:
    a,b=m(f"victim-{c}A"),m(f"victim-{c}B")
    if not (a and b and base and base[0] and base[1]) or c==2048: continue
    if b > base[1]*1.05 and a > base[0]*0.95:
        print(f"chunk {c} BEATS the served 2048 under interference: {b:.1f} vs {base[1]:.1f} t/s, "
              f"alone-rate intact ({a:.1f} vs {base[0]:.1f}). Candidate for the daily; check the TTFT column first.")
PY
# Printed from bash, not from the heredoc: inside `python3 - ...` argv[0] is "-", so the script cannot find its
# own results dir to read vram.txt from.
log "boot VRAM free per arm (a smaller chunk buys loader headroom; the pool was pinned so it lands here instead):"
cat "$R/vram.txt" 2>/dev/null | while read -r l; do log "  vram $l"; done
log "VRAM free at the end: $(vram)"
finish DONE
