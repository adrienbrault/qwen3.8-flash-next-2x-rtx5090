#!/usr/bin/env bash
# R533 — deterministic E3 prefill (e3-det r1), precise A/B on the SERVED chain. R531 passed every gate except G5 60k (0.968x
# from 2 unpaired runs whose atomic second runs were the fast outliers; 120k 0.990x). Here: one image, e3-det r1 built on
# the live image (the served overlays touch none of its files), 8 boots A B B A B A A B; A = daily env (atomic E3), B = daily
# env + EXL3_MOE_PREFILL_E3_DET=1. Per boot: c1 fingerprint, salted cold prefill 60k/120k x 2 (salt per boot), fn_bench
# code c1 x 2 (warm-up 1). Decision: B/A prefill mean at both sizes with a 95 % CI; decode c1 within noise.
# GPU TIMEBOX 25 min. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r533-e3-det-precise; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/e3-det-r1
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SALT=$(( $(date +%s) % 100000 ))
C1=e7fb377c987d685c
BOOTED=0
log(){ echo "$(date -Is) [r533] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){ if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R533 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$SRC/Dockerfile.box" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
ENVS=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE"); [ -n "$ENVS" ] || { log "ABORT: no daily EXTRA_ENV"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")
case " $ENVS " in *" EXL3_MOE_PREFILL_E3=1 "*) ;; *) log "ABORT: live env has no E3"; exit 3;; esac
IMG="$LIMG-e3det"
log "building $IMG on $LIMG (before the lock); env: $ENVS"
(cd "$SRC" && sudo docker build --build-arg BASE="$LIMG" -f Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "ABORT: build: $(grep -aiE 'error' "$R/build.log" | head -3 | tr '\n' ' ' | cut -c1-300)"; exit 3; }
grep -a "MOE_PREFILL_E3_DET default" "$R/build.log" | tail -1 | cut -c1-160 | tee -a "$R/audit.log"
export GPU_QUEUE_NAME=r533-e3-det-precise
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 1500 ))
log "lock held; timebox ends $(date -Is -d @$END)"
BOOTED=1
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
rc=0; n=0
for arm in A B B A B A A B; do
  n=$((n+1)); tag="$arm$n"; extra=""; [ "$arm" = B ] && extra="EXL3_MOE_PREFILL_E3_DET=1"
  [ $(( END - $(date +%s) )) -lt 150 ] && { log "SKIP $tag and later boots: timebox"; break; }
  sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
  "${CLEAN_ENV[@]}" IMG="$IMG" EXTRA_ENV="$ENVS $extra" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 || { log "NO BOOT $tag"; rc=1; break; }
  for i in $(seq 150); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
  [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.Config.Image}}' flashnext)" = "$IMG" ] || { log "NO BOOT $tag"; rc=1; break; }
  has=$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c '^EXL3_MOE_PREFILL_E3_DET=1$')
  want=0; [ "$arm" = B ] && want=1; [ "$has" = "$want" ] || { log "ABORT $tag: DET env $has, want $want"; rc=1; break; }
  fp=$(greedy $tag); log "UP $tag (DET $has) c1 $fp; VRAM free $(vram)"; [ "$fp" = "$C1" ] || { log "FAIL $tag: c1 fingerprint not canonical"; rc=1; }
  for c in 60000 120000; do for k in 1 2; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-pf$c-$k" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c \
      --unique --salt $(( SALT + n*1000 + c/1000 + k*7 )) --out "$R/prefill.jsonl" 2>&1 | grep -E "FAILED|Traceback" | tee -a "$R/audit.log"
    [ $c = 120000 ] && [ $k = 1 ] && log "[$tag] VRAM free during/after 120k: $(vram)"; done; done
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-c1" --kind code --tokens 2048 --warmup-runs 1 \
    --conc 1 --runs 2 --out "$R/records.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-200 | tee -a "$R/audit.log"
done
python3 - "$R" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,math,statistics as st,collections,pathlib
R=pathlib.Path(sys.argv[1]); T={1:12.71,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365}
def ab(bm,label):
    A=[v for b,v in bm.items() if b[0]=="A"]; B=[v for b,v in bm.items() if b[0]=="B"]
    print(f"{label}: boot means " + " ".join(f"{b}={v:.0f}" for b,v in bm.items()))
    if len(A)>1 and len(B)>1:
        d=st.mean(B)-st.mean(A); se=math.sqrt(st.variance(A)/len(A)+st.variance(B)/len(B)); ci=T.get(min(len(A),len(B))-1,2.0)*se
        print(f"{label}: atomic (A) {st.mean(A):.1f}, DET (B) {st.mean(B):.1f} -> B/A {st.mean(B)/st.mean(A):.4f}, B-A 95 % CI [{(d-ci)/st.mean(A)*100:+.2f}, {(d+ci)/st.mean(A)*100:+.2f}] % (n {len(A)}+{len(B)} boots)")
pf=collections.defaultdict(list)
for l in open(R/"prefill.jsonl"):
    r=json.loads(l)
    if r.get("ttft_s") and r.get("prompt_tokens"): pf[(r["tag"].split("-pf")[0], r["ctx_requested"])].append(r["prompt_tokens"]/r["ttft_s"])
for c in (60000,120000):
    bm={b:st.mean(v) for (b,cc),v in sorted(pf.items(), key=lambda x:int(x[0][0][1:])) if cc==c}
    ab(bm, f"prefill {c}")
rounds=collections.defaultdict(list)
for l in open(R/"records.jsonl"):
    r=json.loads(l); boot,shape=r["tag"].rsplit("-",1); rounds[(boot,r["run"])].append(r)
runs=collections.defaultdict(list)
for (boot,run),rs in rounds.items(): runs[boot].append(sum(x["completion_tokens"] for x in rs)/max(x["round_wall_s"] for x in rs))
ab({b:st.mean(v) for b,v in sorted(runs.items(), key=lambda x:int(x[0][1:]))}, "decode code c1")
PY
[ $rc = 0 ] && finish DONE || finish FAILED
