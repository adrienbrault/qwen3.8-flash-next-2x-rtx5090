#!/usr/bin/env bash
# R520b — can a <1 % decode difference be resolved? EXL3_INT8_GEMV=0 vs the default, redone as a precision A/B.
# R520 read "+0.5 to +1.0 %" for OFF, but every ON run 0 of the FIRST boot read 1-4 % low (ON always booted first); dropping
# each boot's run 0 flips the sign (-0.6 / -0.3 / +0.1 / -0.3 %). Design here, so order, warm-up and boot-to-boot variance
# cannot fake a 0.5 % effect:
#   - 8 boots in counterbalanced order A B B A B A A B (A = daily, B = + EXL3_INT8_GEMV=0): linear drift cancels in each block
#   - per boot: c1 greedy fingerprint (both arms must be canonical), then fn_bench code 2,048 tokens with one full-length
#     unrecorded warm-up round per shape (--warmup-runs 1), c1 x 3 runs, c4 x 2 runs
#   - GPU clocks / temperature / power logged before and after each boot's runs; the TabbyAPI log of each boot is kept for
#     draft acceptance (if the int8 path only runs in the non-graphed draft chain, acceptance is where a change would show)
#   - analysis: boot means per arm, difference with a Welch 95 % CI over the 4 boots per arm and the 4 within-block pairs,
#     the run-level SD, and the smallest effect this design resolves
# GPU TIMEBOX ~17 min (8 boots x ~2 min). RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=${HOME:?} PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r520b-int8gemv-precise; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r520b] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
gpu(){ nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit --format=csv,noheader | tr '\n' ';'; }
finish(){ if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R520b $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
python3 /srv/qwen5090/probes/fn_bench.py --help | grep -q -- --warmup-runs || { log "ABORT: fn_bench has no --warmup-runs"; exit 3; }
ENVS=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE"); [ -n "$ENVS" ] || { log "ABORT: no daily EXTRA_ENV"; exit 3; }
case " $ENVS " in *" EXL3_INT8_GEMV="*) log "ABORT: the daily already sets EXL3_INT8_GEMV"; exit 3;; esac
export GPU_QUEUE_NAME=r520b-int8gemv-precise
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 1080 ))
log "lock held; timebox ends $(date -Is -d @$END)"
BOOTED=1
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
n=0
for arm in A B B A B A A B; do
  n=$((n+1)); tag="$arm$n"; extra=""; [ "$arm" = B ] && extra="EXL3_INT8_GEMV=0"
  [ $(( END - $(date +%s) )) -lt 130 ] && { log "SKIP $tag and later boots: timebox"; break; }
  sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
  "${CLEAN_ENV[@]}" EXTRA_ENV="$ENVS $extra" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 || { log "NO BOOT $tag"; finish ABORTED; exit 3; }
  for i in $(seq 120); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
  [ "$(served_id)" = "$MODEL" ] || { log "NO BOOT $tag"; finish ABORTED; exit 3; }
  has=$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c '^EXL3_INT8_GEMV=0$')
  want=0; [ "$arm" = B ] && want=1
  [ "$has" = "$want" ] || { log "ABORT $tag: container env mismatch ($has, want $want)"; finish ABORTED; exit 3; }
  since=$(date -Is)
  fp=$(greedy $tag); log "UP $tag (${extra:-default}) c1 fingerprint $fp | gpu before: $(gpu)"
  [ "$fp" = ae890c45d1000582 ] || log "WARNING $tag: fingerprint not canonical"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-c1" --kind code --tokens 2048 --warmup-runs 1 \
    --conc 1 --runs 3 --out "$R/records.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-200 | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-c4" --kind code --tokens 2048 --warmup-runs 1 \
    --conc 4 --runs 2 --out "$R/records.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-200 | tee -a "$R/audit.log"
  log "[$tag] gpu after: $(gpu)"
  sudo docker logs --since "$since" flashnext > "$R/tabby-$tag.log" 2>&1
done
python3 - "$R" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,re,math,statistics as st,collections,pathlib
R=pathlib.Path(sys.argv[1]); rounds=collections.defaultdict(list)
for l in open(R/"records.jsonl"):
    r=json.loads(l); boot,shape=r["tag"].rsplit("-",1); rounds[(boot,shape,r["run"])].append(r)
runs=collections.defaultdict(list)
for (boot,shape,run),rs in rounds.items(): runs[(boot,shape)].append(sum(x["completion_tokens"] for x in rs)/max(x["round_wall_s"] for x in rs))
boots=sorted({b for b,_ in runs}, key=lambda b:int(b[1:]))
T={2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306}
for shape in ("c1","c4"):
    bm={b:st.mean(runs[(b,shape)]) for b in boots if (b,shape) in runs}
    A=[v for b,v in bm.items() if b[0]=="A"]; B=[v for b,v in bm.items() if b[0]=="B"]
    within=[st.stdev(runs[(b,shape)])/st.mean(runs[(b,shape)])*100 for b in bm if len(runs[(b,shape)])>1]
    print(f"{shape}: boot means " + " ".join(f"{b}={bm[b]:.1f}" for b in bm))
    if len(A)>1 and len(B)>1:
        d=st.mean(B)-st.mean(A); se=math.sqrt(st.variance(A)/len(A)+st.variance(B)/len(B)); df=min(len(A),len(B))-1
        ci=T.get(df,2.0)*se
        print(f"{shape}: A {st.mean(A):.1f} (boot sd {st.stdev(A)/st.mean(A)*100:.2f} %) B {st.mean(B):.1f} (boot sd {st.stdev(B)/st.mean(B)*100:.2f} %)"
              f" -> B-A {d/st.mean(A)*100:+.2f} %, 95 % CI [{(d-ci)/st.mean(A)*100:+.2f}, {(d+ci)/st.mean(A)*100:+.2f}] %; within-boot run sd {st.mean(within):.2f} %")
    pairs=[]
    order=[b for b in boots if b in bm]
    for i in range(0,len(order)-3,4):
        blk=order[i:i+4]; a=[bm[b] for b in blk if b[0]=="A"]; bb=[bm[b] for b in blk if b[0]=="B"]
        if a and bb: pairs.append((st.mean(bb)/st.mean(a)-1)*100)
    if pairs: print(f"{shape}: per-block B-A " + " ".join(f"{p:+.2f} %" for p in pairs))
acc=collections.defaultdict(lambda:[0,0])
for f in R.glob("tabby-*.log"):
    txt=re.sub(r"\s+"," ",f.read_text(errors="replace"))
    for a,b in re.findall(r"draft (\d+)/(\d+) accepted",txt): acc[f.stem.split("-")[1][0]][0]+=int(a); acc[f.stem.split("-")[1][0]][1]+=int(b)
for k,(a,b) in sorted(acc.items()): print(f"draft acceptance arm {k}: {a}/{b} = {a/b*100:.2f} %" if b else f"arm {k}: no draft lines")
PY
finish DONE
