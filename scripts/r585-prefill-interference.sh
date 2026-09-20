#!/usr/bin/env bash
# R585 — the production gap is probably ARRIVAL SHAPE, not length and not sampling alone.
# The numbers that have to be reconciled (FINDINGS, PRODUCTION OBSERVATION 2026-09-20):
#   * README headline, greedy, short output, synchronized streams ......... 213 t/s per stream
#   * R583, greedy, 3,000 tokens, ~10k ctx, ONE stream, idle box .......... 283
#   * R583, greedy, 3,000 tokens, ~10k ctx, THREE synchronized streams .... 181 per stream
#   * fn_bench, greedy, 3,000 tokens, ~10k ctx, run ALONGSIDE the user's
#     real 3-agent traffic ................................................ 88.9
#   * the user's own requests, same session ............................... 65.6
# The 88.9 measurement was greedy -- fn_bench hardcoded temperature 0 until 2026-09-20 -- so sampling cannot be
# what separates 181 from 89. The only difference between those two rows is what the OTHER streams were doing.
# fn_bench at conc 3 starts three streams together: they prefill together, then decode together in a steady batch.
# Real agents arrive staggered, so while one request is 100 seconds into a 3,000-token generation another submits a
# fresh 30-60k prompt, and chunked prefill (chunk_size 2048) interleaves with the long request's decode steps. Each
# interleaved chunk is a forward pass the decoder does not get. The two slowest production requests ran 150-220 s,
# long enough to absorb many arrivals.
# Four arms on ONE boot of the served daily, victim = greedy, 3,000 forced tokens at ~10k context, 1 stream:
#   A  victim alone (control)
#   B  victim + a storm of fresh ~45k-token prompts, 64 tokens out, one every ~8 s  -> big prefills interleaved
#   C  victim + the same arrival RATE at ~1k-token prompts                          -> isolates prefill SIZE from
#                                                                                      the mere presence of traffic
#   D  victim + two more 3,000-token generations started 20 s apart                 -> staggered decode, no prefill
# NVMe TIER IS LEFT ON: production runs with it on and R583/R584 both turned it off. The interferer prompts are
# --unique with a per-request salt so no arm is served another's cached pages.
# Measurement only: nothing is promoted, no launcher is edited. GPU TIMEBOX ~25 min. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-20-r585-prefill-interference; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0; NOISE_PIDS=()
log(){ echo "$(date -Is) [r585] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|max_batch_size|chunk_size)' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
# Every arm must leave the box quiet, or the next arm measures this one's leftovers.
stop_noise(){ local p; for p in "${NOISE_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
  for p in "${NOISE_PIDS[@]:-}"; do [ -n "$p" ] && wait "$p" 2>/dev/null; done; NOISE_PIDS=()
  pkill -f "fn_bench.py.*r585-noise" 2>/dev/null
  # Wait for the server to go idle again rather than trusting the kill: an in-flight 45k prefill outlives its client.
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
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R585 $1 ==="; }
trap 'log signal; finish ABORTED; exit 4' TERM INT HUP
[ -e "$LIVE" ] || { log "ABORT: missing $LIVE"; exit 3; }

export GPU_QUEUE_NAME=r585-prefill-interference
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
VCTX=13000   # ~9.8k counted prompt tokens for the victim, the depth the slowest production requests sat at.

# noise_loop CTX_BUDGET TOKENS PERIOD -- fresh arrivals until killed. Tagged r585-noise so stop_noise can find them.
noise_loop(){ local ctx=$1 tok=$2 period=$3 k=0
  while :; do
    k=$((k+1))
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "r585-noise-ctx$ctx" \
      --kind prose --tokens $tok --conc 1 --runs 1 --ctx $ctx --unique \
      --salt $(( SALT + 40000 + k*613 + RANDOM )) --out "$R/noise.jsonl" > /dev/null 2>&1
    sleep "$period"
  done; }

# victim ARM -- one 3,000-token generation, measured. Two runs so a single unlucky round cannot carry the arm.
victim(){ local arm=$1
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "victim-$arm" \
    --kind prose --tokens 3000 --conc 1 --runs 2 --warmup-runs 0 --ctx $VCTX --unique \
    --salt $(( SALT + RANDOM )) --out "$R/records.jsonl" 2>&1 \
    | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$arm]/" | cut -c1-220 | tee -a "$R/audit.log"
  alive || { log "FAIL: server not alive after arm $arm"; return 1; }; }

up "$LIVE" served || { log "FAIL: did not boot"; finish ABORTED; exit 3; }
log "UP: $(cfgline); VRAM free $(vram); NVMe tier left at the launcher default (ON in production)"
# Warm the batch shapes once before any arm; an unwarmed round reads low and would be charged to the noise.
python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag warm --kind prose --tokens 256 \
  --conc 1 3 --runs 1 --ctx $VCTX --unique --salt $SALT --out "$R/warm.jsonl" > "$R/warm.log" 2>&1
log "warmed"

log "=== arm A: victim alone ==="
victim A || { finish ABORTED; exit 3; }

log "=== arm B: victim + ~45k-token prompt arrivals every 8 s ==="
noise_loop 60000 64 8 & NOISE_PIDS+=($!)
sleep 10                      # let one big prefill already be in flight when the victim starts
victim B || { finish ABORTED; exit 3; }
stop_noise

log "=== arm C: victim + ~750-token prompt arrivals every 8 s ==="
noise_loop 1000 64 8 & NOISE_PIDS+=($!)
sleep 10
victim C || { finish ABORTED; exit 3; }
stop_noise

log "=== arm D: victim + two staggered 3,000-token generations ==="
( sleep 0;  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag r585-noise-long1 \
    --kind prose --tokens 3000 --conc 1 --runs 3 --ctx $VCTX --unique --salt $(( SALT + 7001 )) \
    --out "$R/noise.jsonl" >/dev/null 2>&1 ) & NOISE_PIDS+=($!)
( sleep 20; python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag r585-noise-long2 \
    --kind prose --tokens 3000 --conc 1 --runs 3 --ctx $VCTX --unique --salt $(( SALT + 7002 )) \
    --out "$R/noise.jsonl" >/dev/null 2>&1 ) & NOISE_PIDS+=($!)
sleep 30
victim D || { finish ABORTED; exit 3; }
stop_noise

python3 - "$R/records.jsonl" "$R/noise.jsonl" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
rows=collections.defaultdict(list)
for path in sys.argv[1:]:
    try: f=open(path)
    except OSError: continue
    for l in f:
        r=json.loads(l)
        if not r.get("ok") or not r.get("decode_tps"): continue
        rows[r["tag"]].append(r["decode_tps"])
def m(t): return st.mean(rows[t]) if rows.get(t) else None
base=m("victim-A")
names={"A":"alone","B":"+ ~45k prompt arrivals /8 s","C":"+ ~750-token arrivals /8 s","D":"+ 2 staggered long generations"}
print("victim: greedy, 3,000 forced tokens at ~9.8k context, 1 stream, NVMe tier on\n")
print(f"{'arm':>4}  {'decode t/s':>10} {'vs alone':>9}   background")
for a in "ABCD":
    v=m(f"victim-{a}")
    if v is None: print(f"{a:>4}  {'missing':>10}"); continue
    print(f"{a:>4}  {v:>10.1f} {(v/base if base else 0):>8.2f}x   {names[a]}")
n=[t for t in rows if t.startswith("r585-noise")]
if n:
    print("\nbackground requests, for the record:")
    for t in sorted(n): print(f"  {t:<28} n={len(rows[t]):>3}  mean {st.mean(rows[t]):.1f} t/s")
print("\nReference points: R583 measured 283 t/s alone and 181 at three SYNCHRONIZED streams, both greedy at this")
print("shape. Production delivered 65.6 per stream, and a greedy probe run alongside that traffic read 88.9.")
if base:
    for a in "BCD":
        v=m(f"victim-{a}")
        if v and v < 0.6*base:
            print(f"Arm {a} reproduces the production regime: {v:.1f} t/s against {base:.1f} alone.")
PY
log "VRAM free at the end: $(vram)"
finish DONE
