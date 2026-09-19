#!/usr/bin/env bash
# R580 — the full decode curve of the served daily, 1 to 8 concurrent streams, code and prose, on ONE boot.
# Why: the published decode rows are stitched from several rounds (c1 from R575's mp_decode, c4-c8 from R570/R571's
# fn_bench A arms) and there is no measurement at 2 or 3 streams at all, so the README's figures cannot start at 1.
# This round reads every point with one instrument on one boot of the live launcher: fn_bench, greedy, 1,024 tokens,
# 1 warm-up round + 3 recorded rounds per shape, tier off (as R570/R571 measured, so the numbers stay comparable).
# Measurement only: nothing is promoted, no launcher is edited, the live launcher is booted unmodified.
# It also reads cold prefill at true prompt lengths of 30k, 60k, 120k, 200k and 240k tokens: the published prefill
# figure stops at 90,008 tokens because fn_bench's --ctx 120000 produces that many, and nothing deeper has ever been
# measured cold here. The words-to-tokens ratio is calibrated on this boot from the server's usage counter.
#   Records: $R/records.jsonl and $R/prefill.jsonl (raw), $R/curve.tsv and $R/prefill.tsv (summaries), for bench/plot.py.
# TRY 2 (2026-09-20): try 1 aborted 6 s in. pfstat read the tag from $2 but is called with one argument, so it
# matched no record and the calibration looked empty; the calibration prefill itself was fine (budget 60,000 ->
# 44,976 prompt tokens). The served pool is 999,424 since R579, not 966,656.
# GPU TIMEBOX ~50 min. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-20-r580-decode-curve-try2; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r580] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
finish(){
  if [ "$BOOTED" = 1 ]; then
    sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done; log "daily: $(served_id) $(cfgline)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R580 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done

export GPU_QUEUE_NAME=r580-decode-curve
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$NEWM" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
# up LAUNCHER TAG [ENV...] -> 0 up, 1 no boot
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
      log "NO BOOT $tag (restart loop): $(sudo docker logs --tail 80 flashnext 2>&1 | grep -aE 'RuntimeError|Error|memory' | tail -1 | cut -c1-160)"
      kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1; }
    sleep 3; done
  log "NO BOOT $tag (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }

log "=== one boot of the live launcher, tier off ==="
up "$LIVE" S NVME_TIER= || { log "FAIL: no boot"; finish ABORTED; exit 3; }
pl=$(sudo grep -E 'draft_num_tokens_by_batch' $CFG | awk '{$1=$1; print}')
log "UP: $(cfgline); $pl; VRAM free $(vram); c1 fingerprint $(greedy S)"

# Cold prefill at TRUE prompt lengths, to the top of the window. fn_bench's --ctx is a filler budget, not a token
# count: the salted passage is ctx/1.6 words, which lands near 0.75 x ctx tokens, which is why the published "120k"
# point is really 90,008 tokens and nothing deeper has ever been measured cold here. So calibrate the ratio once on
# this boot from the server's own usage counter, then ask for the budget that lands on each target, and report the
# prompt tokens the server counted, never the budget. Three salted runs per depth, tier off, so all are cold.
SALT=$(( $(date +%s) % 100000 ))
pf(){ # pf TAG BUDGET SALT_K -> one salted cold prefill into prefill.jsonl
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$1" --kind prose --tokens 64 \
    --conc 1 --runs 1 --ctx "$2" --unique --salt $(( SALT + $2/1000 + $3*997 + RANDOM )) \
    --out "$R/prefill.jsonl" > "$R/prefill-$1-$3.log" 2>&1; }
pfstat(){ python3 -c "
import json,sys,statistics as st
rows=[json.loads(l) for l in open(sys.argv[1])]
rows=[r for r in rows if r.get('tag')==sys.argv[2] and r.get('ttft_s') and r.get('prompt_tokens')]
if not rows: print('none'); raise SystemExit
print(f\"{st.mean(r['prompt_tokens'] for r in rows):.0f}\t{st.mean(r['prompt_tokens']/r['ttft_s'] for r in rows):.0f}\t{st.mean(r['ttft_s'] for r in rows):.2f}\t{len(rows)}\")
" "$R/prefill.jsonl" "$1" 2>/dev/null || echo none; }

pf cal 60000 0
CAL=$(pfstat cal | cut -f1)
case "$CAL" in ''|none|0) log "FAIL: prefill calibration produced no usable record"; finish ABORTED; exit 3;; esac
log "prefill calibration: budget 60,000 -> $CAL prompt tokens ($(python3 -c "print(f'{$CAL/60000:.4f}')") tokens per budget unit)"

: > "$R/prefill.tsv"; printf "target\tbudget\tprompt_tokens\ttps\tttft_s\tn\n" >> "$R/prefill.tsv"
for T in 30000 60000 120000 200000 240000; do
  B=$(python3 -c "print(round($T * 60000 / $CAL))")
  for k in 1 2 3; do pf "pf-$T" "$B" $k; done
  alive || { log "FAIL: server not alive after the $T-token prefills"; finish ABORTED; exit 3; }
  ST=$(pfstat "pf-$T")
  case "$ST" in none) log "prefill $T: no usable record (budget $B; the server may have refused the length)";;
    *) log "prefill target $T: budget $B -> $(echo "$ST" | awk -F'\t' '{printf "%s prompt tokens, %s t/s, TTFT %s s (n %s)", $1, $2, $3, $4}')"
       printf "%s\t%s\t%s\n" "$T" "$B" "$ST" >> "$R/prefill.tsv";; esac
done

# The curve: every concurrency from 1 to 8, both kinds, on this one boot. A shape that fails leaves its rows out of
# curve.tsv rather than stopping the round, so one bad shape cannot cost the whole measurement.
for conc in 1 2 3 4 5 6 7 8; do
  for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "S-c$conc-$kind" --kind $kind \
      --tokens 1024 --warmup-runs 1 --conc $conc --runs 3 --out "$R/records.jsonl" 2>&1 \
      | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[S c$conc $kind]/" | cut -c1-240 | tee -a "$R/audit.log"
    alive || { log "FAIL: server not alive after c$conc $kind"; finish ABORTED; exit 3; }
  done
done

python3 - "$R/records.jsonl" "$R/curve.tsv" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
tok=collections.defaultdict(int); wall={}; per=collections.defaultdict(list)
for l in open(sys.argv[1]):
    r=json.loads(l)
    if not r.get("ok"): continue
    tok[(r["tag"],r["run"])]+=r["completion_tokens"] or 0
    wall[(r["tag"],r["run"])]=r["round_wall_s"]
    per[r["tag"]].append(r["wall_tps"])
agg=collections.defaultdict(list)
for (t,run),v in tok.items(): agg[t].append(v/wall[(t,run)])
out=open(sys.argv[2],"w"); out.write("conc\tkind\taggregate_tps\tper_stream_tps\trounds\n")
for conc in range(1,9):
    for kind in ("code","prose"):
        t=f"S-c{conc}-{kind}"
        if t not in agg: print(f"c{conc} {kind}: MISSING"); continue
        a=st.mean(agg[t]); p=st.mean(per[t])
        print(f"c{conc} {kind}: aggregate {a:.1f} t/s, per stream {p:.1f} t/s (n {len(agg[t])})")
        out.write(f"{conc}\t{kind}\t{a:.1f}\t{p:.1f}\t{len(agg[t])}\n")
out.close()
PY
log "VRAM free at the end: $(vram)"
finish DONE
