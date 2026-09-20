#!/usr/bin/env bash
# R584 — backlog S17: which part of the REGIME explains the production gap, once length is ruled out?
# R583 measured the length ladder on the served daily and found it flat: at ~9.8k context, greedy, one stream,
# 3,000 forced tokens ran at 283 t/s against 277 at 512. Generation length alone does not slow this server down,
# so R579's draft-cache window is not the whole story and probably not the story at all.
# What is left between the published 213 t/s and the 65.6 t/s the user actually gets (FINDINGS, PRODUCTION
# OBSERVATION 2026-09-20) is the REGIME, and the largest untested difference is sampling:
#   * every number this repo has ever published was measured at temperature 0. fn_bench hardcoded it until today.
#   * the desktop harness samples at 0.6. Under argmax an MTP draft token is accepted whenever it matches the
#     argmax of the verify pass; under sampling it has to survive the sampler, so acceptance -- and with it the
#     whole speedup MTP buys -- can fall away without a single line of server config changing.
# Second difference: content. The suite's filler is English prose or generic code; the user generates code, diffs
# and tool-call JSON, and a draft model's hit rate is content-dependent.
# One boot of the served daily, NVMe tier off, everything at ~10k context and 3,000 forced tokens:
#   temp {0.0, 0.6} x kind {prose, code} x streams {1, 3}, 2 runs each, plus one 512-token shape at 0.6 to show
#   whether sampling and length interact rather than just adding.
# Measurement only: nothing is promoted, no launcher is edited. GPU TIMEBOX ~25 min. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-20-r584-sampling-regime; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r584] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|max_batch_size|chunk_size)' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
finish(){
  if [ "$BOOTED" = 1 ]; then
    sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done; log "daily: $(served_id) $(cfgline)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R584 $1 ==="; }
trap 'log signal; finish ABORTED; exit 4' TERM INT HUP
[ -e "$LIVE" ] || { log "ABORT: missing $LIVE"; exit 3; }
# The --temp flag is what this round is about; an older probe would silently measure greedy eight times.
python3 /srv/qwen5090/probes/fn_bench.py --help 2>&1 | grep -q -- "--temp" || { log "ABORT: fn_bench has no --temp"; exit 3; }

export GPU_QUEUE_NAME=r584-sampling-regime
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
CTX=13000   # ~9.8k counted prompt tokens, the depth the two slowest production requests sat at.
# run TEMP KIND TOKENS CONC. The tag encodes every factor so the analysis can pivot on any of them.
run(){ local temp=$1 kind=$2 tok=$3 conc=$4 t=${1/./p}
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "t$t-$kind-tok$tok-c$conc" \
    --kind "$kind" --tokens $tok --conc $conc --runs 2 --warmup-runs 1 --ctx $CTX --unique --temp "$temp" \
    --salt $(( SALT + tok + conc*7919 + RANDOM )) --out "$R/records.jsonl" 2>&1 \
    | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[t$temp $kind tok$tok c$conc]/" | cut -c1-220 | tee -a "$R/audit.log"
  alive || { log "FAIL: server not alive after t$temp $kind tok$tok c$conc"; return 1; }; }

up "$LIVE" served NVME_TIER= || { log "FAIL: did not boot"; finish ABORTED; exit 3; }
log "UP: $(cfgline); VRAM free $(vram)"

for temp in 0.0 0.6; do
  for kind in prose code; do
    for conc in 1 3; do
      run "$temp" "$kind" 3000 "$conc" || { finish ABORTED; exit 3; }
    done
  done
done
# Does sampling cost more on a long generation than a short one, or is the penalty flat?
run 0.6 prose 512 1 || { finish ABORTED; exit 3; }
run 0.0 prose 512 1 || { finish ABORTED; exit 3; }

python3 - "$R/records.jsonl" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
rows=collections.defaultdict(list)
for l in open(sys.argv[1]):
    r=json.loads(l)
    if not r.get("ok") or not r.get("decode_tps"): continue
    rows[r["tag"]].append(r["decode_tps"])
def m(tag): return st.mean(rows[tag]) if rows.get(tag) else None
print("per-request decode t/s at ~9.8k context, 3,000 forced tokens, NVMe tier off\n")
print(f"{'kind':>6} {'streams':>8} {'greedy':>9} {'temp 0.6':>10} {'sampled/greedy':>16}")
for kind in ("prose","code"):
    for conc in (1,3):
        g=m(f"t0p0-{kind}-tok3000-c{conc}"); s=m(f"t0p6-{kind}-tok3000-c{conc}")
        if g and s: print(f"{kind:>6} {conc:>8} {g:>9.1f} {s:>10.1f} {s/g:>15.2f}x")
        else:       print(f"{kind:>6} {conc:>8} {'missing':>9} {'missing':>10}")
print()
g5=m("t0p0-prose-tok512-c1"); s5=m("t0p6-prose-tok512-c1")
g3=m("t0p0-prose-tok3000-c1"); s3=m("t0p6-prose-tok3000-c1")
if g5 and s5: print(f"512 tokens, prose, 1 stream: greedy {g5:.1f} -> sampled {s5:.1f} ({s5/g5:.2f}x)")
if g3 and s3: print(f"3,000 tokens, prose, 1 stream: greedy {g3:.1f} -> sampled {s3:.1f} ({s3/g3:.2f}x)")
if g5 and s5 and g3 and s3:
    print(f"\nsampling penalty at 3,000 tokens vs at 512: {(s3/g3)/(s5/g5):.2f}x "
          f"(1.00 = sampling costs the same whatever the length, below 1 = the two compound)")
# The published headline is greedy prose at one stream. Say plainly how far the sampled number sits from it.
if s3: print(f"\nThe user's regime (sampled, 3,000 tokens, ~10k ctx, 1 stream) measures {s3:.1f} t/s. "
             f"Production delivered 65.6 t/s per stream; the README headline is 213.")
PY
log "VRAM free at the end: $(vram)"
finish DONE
