#!/usr/bin/env bash
# R583 — backlog S16: does decode rate fall with GENERATION LENGTH, and is R579's draft-cache window the cause?
# The user's own DSH session (2026-09-20 00:05, FINDINGS "PRODUCTION OBSERVATION") delivered 65.6 t/s per stream and
# 134.4 t/s aggregate where the published table says 213 / 627. 19 requests under 50 T/s ate 59 % of all decode
# seconds for 21 % of the tokens; the slowest were 2,700-4,000-token generations at 13-20 T/s, and the two worst sat
# at only ~10.7k context -- so it tracks generated length, not context depth. Reproduced from a fast local client at
# 88.9 T/s for 3,000 tokens at ~10k context, which rules out client backpressure.
# NOTHING IN THIS REPO GENERATES MORE THAN 1,024 TOKENS, so every gate R579 passed was blind to this.
#   A = the served daily: window ON (EXL3_MTP_KV_WINDOW=16384), pool 999,424.
#   B = the previous daily: launch-flashnext.sh.pre-r579, window OFF, pool 966,656.
#   The two differ in pool as well as window because the window is what bought the pool step; that is the honest
#   promotion comparison. If B is fast where A is slow, R579 is the cause and the rollback is one launcher line.
# Per arm, one boot, NVMe tier off, greedy (the user's traffic is sampled at T=0.6 -- noted, not reproduced here):
#   LENGTH LADDER at 1 stream, ctx ~10k and ~50k: 512, 1024, 2048, 3000 forced tokens, 2 runs each.
#     The ladder is the whole point: a flat curve says length is innocent, a falling one reproduces the complaint.
#   CONCURRENT: 3 streams x 3,000 tokens at ctx ~10k, 2 runs.
# Measurement only: nothing is promoted, no launcher is edited. GPU TIMEBOX ~45 min. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-20-r583-long-generation; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
PREV=/srv/qwen5090/launch-flashnext.sh.pre-r579
CFG=/srv/qwen5090/flashnext-config.yml
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r583] $*" | tee -a "$R/audit.log"; }
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
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R583 $1 ==="; }
trap 'log signal; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$PREV" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done

export GPU_QUEUE_NAME=r583-long-generation
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
# run ARM CTX_BUDGET TOKENS CONC -- one shape into records.jsonl. --unique + a fresh salt per call keeps every
# request's filler distinct, so no arm can be handed another's cached pages.
run(){ local arm=$1 ctx=$2 tok=$3 conc=$4
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$arm-ctx$ctx-t$tok-c$conc" \
    --kind prose --tokens $tok --conc $conc --runs 2 --warmup-runs 1 --ctx $ctx --unique \
    --salt $(( SALT + ctx/1000 + tok + conc*7919 + RANDOM )) --out "$R/records.jsonl" 2>&1 \
    | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$arm ctx$ctx t$tok c$conc]/" | cut -c1-220 | tee -a "$R/audit.log"
  alive || { log "FAIL: server not alive after $arm ctx$ctx t$tok c$conc"; return 1; }; }

for arm in A B; do
  case $arm in
    A) L="$LIVE";  WIN="window ON (served)";;
    B) L="$PREV";  WIN="window OFF (.pre-r579)";;
  esac
  log "=== arm $arm: $WIN ==="
  up "$L" "$arm" NVME_TIER= || { log "FAIL: arm $arm did not boot"; finish ABORTED; exit 3; }
  envwin=$(sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' flashnext 2>/dev/null | grep -c EXL3_MTP_KV_WINDOW)
  log "UP $arm: $(cfgline); MTP_KV_WINDOW env present: $envwin; VRAM free $(vram)"
  # 13,000 and 66,000 filler budget land near 10k and 50k prompt tokens (fn_bench's --ctx is ~0.75 tokens per unit).
  for ctx in 13000 66000; do
    for tok in 512 1024 2048 3000; do
      run "$arm" $ctx $tok 1 || { finish ABORTED; exit 3; }
    done
  done
  run "$arm" 13000 3000 3 || { finish ABORTED; exit 3; }
done

python3 - "$R/records.jsonl" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
rows=collections.defaultdict(list); ptok=collections.defaultdict(list)
for l in open(sys.argv[1]):
    r=json.loads(l)
    if not r.get("ok") or not r.get("decode_tps"): continue
    rows[r["tag"]].append(r["decode_tps"]); ptok[r["tag"]].append(r.get("prompt_tokens") or 0)
def get(tag): return (st.mean(rows[tag]), st.mean(ptok[tag]), len(rows[tag])) if rows.get(tag) else None
print("per-request decode t/s, greedy, tier off\n")
for ctx in (13000, 66000):
    print(f"-- filler budget {ctx} --")
    print(f"{'tokens':>8} {'A window ON':>13} {'B window OFF':>14} {'B/A':>8}   prompt tokens")
    for tok in (512, 1024, 2048, 3000):
        a=get(f"A-ctx{ctx}-t{tok}-c1"); b=get(f"B-ctx{ctx}-t{tok}-c1")
        if a and b:
            print(f"{tok:>8} {a[0]:>13.1f} {b[0]:>14.1f} {b[0]/a[0]:>7.2f}x   {a[1]:,.0f}")
        else:
            print(f"{tok:>8} {'missing':>13} {'missing':>14}")
    print()
a=get("A-ctx13000-t3000-c3"); b=get("B-ctx13000-t3000-c3")
if a and b: print(f"3 streams x 3,000 tokens at ~10k ctx: A {a[0]:.1f} vs B {b[0]:.1f} per stream ({b[0]/a[0]:.2f}x)")
# The verdict this round exists to reach.
for ctx in (13000, 66000):
    s=get(f"A-ctx{ctx}-t512-c1"); l=get(f"A-ctx{ctx}-t3000-c1")
    if s and l: print(f"A ctx{ctx}: 3,000 tokens runs {l[0]/s[0]:.2f}x the 512-token rate (1.00 = length is innocent)")
PY
log "VRAM free at the end: $(vram)"
finish DONE
