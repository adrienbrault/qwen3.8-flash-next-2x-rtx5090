#!/usr/bin/env bash
# R518 — 6 decode slots on the Flash-Next daily (user 2026-09-19: "6 slots sounds interesting"; promote if the test confirms it,
# then leave the slot count alone). R480 (3.05): slot count changes no numerics and c1/c4 are equal at 4 vs 8 slots; each slot
# holds ~0.42 GiB of fp32 GDN state, so 6 slots cost ~0.85 GiB = ~56k pool tokens. The 2.50 pool (786,432) is no longer the
# binding limit (8 SWE-bench agents at p90 context = ~500k). Draft policy [[4,3],[8,1]] unchanged: c5-c6 draft at depth 1.
#   LADDER  6 slots at 786,432 / 753,664 / 720,896 / 688,128: first that boots AND survives a c6 stress pass (2,048 tok code+prose).
#   ARMS    S4 (daily) / S6 / S4b / S6b. Fingerprints on every arm (canonical c1 ae890c45d1000582 / 30k 4a255910dee2d9c5);
#           fn_bench code+prose c1/c4/c6 2,048 x 2 on S4 and S6; agent replay on all four arms (probes/agent_replay.py: 16
#           recorded Flash-Next SWE-bench conversations x first 24 calls, 8 concurrent agents, 2 s tool gap) with the TabbyAPI
#           per-request cache log (cached / new prompt tokens) parsed per replay window.
#   DECIDE  S6 promotes if: fingerprints canonical; c1 and c4 (code, prose) within -2 % of S4; and (c6 aggregate >= 1.05 x S4's
#           c6, or agent-replay generated tok/s >= 1.03 x S4's, both arms' mean of two runs).
#   GATES   G0 sampler fallbacks = live; G2 agentic-edit 4/4 modes 6/6; G3 needles 131k/240k 5/5; G4 tool-eval 69 x 4 >= 82.
#   PROMOTE launch-flashnext-r518.sh = launch-flashnext-r517.sh with MAXBS default 6 and CACHE default = the ladder pool.
# RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r518-slots6; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
BASE=/srv/qwen5090/launch-flashnext-r517.sh
NEW=/srv/qwen5090/launch-flashnext-r518.sh
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
TRAJS="/srv/qwen5090/results/2026-09-16-r359-swebench-10/out/*/*.traj.json /srv/qwen5090/results/2026-09-16-r360-swebench-strat/out/*/*.traj.json /srv/qwen5090/results/2026-09-16-r361-swebench-failed/out/*/*.traj.json /srv/qwen5090/results/2026-09-16-r369-swebench-30/out/*/*.traj.json"
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0; PROMOTED=0
log(){ echo "$(date -Is) [r518] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
finish(){
  if [ "$BOOTED" = 1 ] && [ "$PROMOTED" = 0 ]; then log "restoring the unchanged daily"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done
    log "daily: $(served_id) $(cfgline) VRAM free $(vram)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R518 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$BASE" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/agent_replay.py /srv/qwen5090/probes/agentic-edit.py \
         /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/tooleval_summary.py; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
cmp -s "$LIVE" "$BASE" || { log "ABORT: the live launcher is not launch-flashnext-r517.sh"; exit 3; }
grep -q '^MAXBS=${MAXBS:-4}$' "$BASE" && grep -q '^CACHE=${CACHE:-786432}' "$BASE" || { log "ABORT: r517 launcher lacks the MAXBS/CACHE default lines"; exit 3; }
export GPU_QUEUE_NAME=r518-slots6
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"

fallback(){ python3 - "$API" "$MODEL" "$R/fallback-$1.json" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,hashlib,urllib.request
api,model,out=sys.argv[1:4]; res={}
cases={"presence":{"temperature":0,"presence_penalty":0.5,"max_tokens":200,"min_tokens":200},
       "rep":{"temperature":0,"repetition_penalty":1.1,"max_tokens":200,"min_tokens":200},
       "stop":{"temperature":0,"max_tokens":400,"stop":["\n\n\n","def test_"]}}
for name,extra in cases.items():
    body={"model":model,"messages":[{"role":"user","content":"Write a Python function that merges overlapping intervals, then tests for it."}],**extra}
    d=json.loads(urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600).read())
    m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or "")
    res[name]=hashlib.sha256(t.encode()).hexdigest()[:16]+f" len={len(t)} fin={d['choices'][0].get('finish_reason')}"
json.dump(res,open(out,"w"),sort_keys=True); print("fallback", json.dumps(res,sort_keys=True))
PY
}
[ "$(served_id)" = "$MODEL" ] || { log "ABORT: live daily is '$(served_id)'"; finish ABORTED; exit 3; }
fallback live >/dev/null
[ -s "$R/fallback-live.json" ] || { log "ABORT: no G0 reference"; finish ABORTED; exit 3; }
log "G0 reference: $(cat "$R/fallback-live.json")"

BOOTED=1
# boot TAG MAXBS CACHE -> 0 serving that config, 1 no boot, 2 mismatch
boot(){ local tag=$1 mbs=$2 cache=$3 i st lp got
  log "boot $tag: slots $mbs cache $cache"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" MAXBS=$mbs CACHE=$cache bash "$BASE" >> "$R/boot-$tag.log" 2>&1 ) & lp=$!
  for i in $(seq 360); do
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp; got=$(cfgline)
      case "$got" in *"cache_size: $cache cache_mode: 8,8 max_batch_size: $mbs gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;; *) log "ABORT: config '$got'"; return 2;; esac
      [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = tabbyapi:stack-r4-e3r2 ] || { log "ABORT: container image"; return 2; }
      BOOT_TS=$(date -Is); log "UP $tag: VRAM free MiB $(vram)"; return 0; fi
    st=$(cstatus); case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 60 ] && { sleep 3; continue; }
      log "NO BOOT $tag ($st): $(sudo docker logs --tail 60 flashnext 2>&1 | grep -aE 'RuntimeError|Error|VRAM' | tail -1 | cut -c1-220)"
      sudo docker logs flashnext > "$R/boot-$tag.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1;; esac
    [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 0)" -ge 1 ] && {
      log "NO BOOT $tag (restart loop): $(sudo docker logs --tail 60 flashnext 2>&1 | grep -aE 'RuntimeError|Error|VRAM' | tail -1 | cut -c1-220)"
      sudo docker logs flashnext > "$R/boot-$tag.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1; }
    sleep 3; done
  log "NO BOOT $tag (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
greedy30k(){ python3 - "$API" "$MODEL" "$R/greedy30k-$1.json" <<'PY' 2>/dev/null || echo none
import json,sys,hashlib,urllib.request,random
api,model,out=sys.argv[1:4]
rng=random.Random(4242); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
body=" ".join(rng.choice(words) for _ in range(23000))
msg=body+"\n\nSummarize the passage above in one sentence, then write a haiku about it."
req=json.dumps({"model":model,"temperature":0,"max_tokens":128,"min_tokens":128,"messages":[{"role":"user","content":msg}]}).encode()
d=json.loads(urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=req,headers={"Content-Type":"application/json"}),timeout=900).read())
open(out,"w").write(json.dumps(d)); m=d["choices"][0]["message"]
print(hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16])
PY
}
fp(){ local a b; a=$(greedy "$1"); b=$(greedy30k "$1"); log "[$1] fingerprints c1 $a / 30k $b (canonical ae890c45d1000582 / 4a255910dee2d9c5)"
  [ "$a" = ae890c45d1000582 ] && [ "$b" = 4a255910dee2d9c5 ]; }
speed(){ local t=$1 kind
  for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$t-$kind" --kind $kind --tokens 2048 --conc 1 4 6 --runs 2 \
      --out "$R/records-$t.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$t $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done; }
stress(){ local kind; for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "stress-$1-$kind" --kind $kind --tokens 2048 --conc 6 --runs 1 \
      --out "$R/stress-$1.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[stress $1 $kind]/" | cut -c1-200 | tee -a "$R/audit.log"; done
  alive; }
replay(){ local t=$1 since
  since=$(date -Is)
  timeout 2400 python3 /srv/qwen5090/probes/agent_replay.py --url "$API" --model "$MODEL" --trajs $TRAJS --agents 8 --convs 16 --calls 24 \
    --tool-gap 2 --tag "$t" --out "$R/replay.jsonl" 2>&1 | tail -1 | sed "s/^/[replay $t] /" | tee -a "$R/audit.log"
  sudo docker logs --since "$since" flashnext > "$R/replay-$t.tabby.log" 2>&1
  python3 - "$R/replay-$t.tabby.log" "$t" <<'PY' 2>&1 | tee -a "$R/audit.log"
import re,sys
rx=re.compile(r"prompt ([\d,]+) tokens, (none|[\d,]+) cached, ([\d,]+) new in ([\d.]+) s")
n=tot=cached=new=0; secs=0.0; full=0
for l in open(sys.argv[1], errors="replace"):
    m=rx.search(l)
    if not m: continue
    p=int(m[1].replace(",","")); c=0 if m[2]=="none" else int(m[2].replace(",","")); w=int(m[3].replace(",",""))
    n+=1; tot+=p; cached+=c; new+=w; secs+=float(m[4]); full+= (c==0 and p>4096)
print(f"[cache {sys.argv[2]}] requests {n}: prompt {tot:,} tokens, cached {cached:,} ({100*cached/max(tot,1):.1f} %), new {new:,} "
      f"(mean {new/max(n,1):,.0f}/request), prefill time {secs:.0f} s, cold >4k-token prompts {full}")
PY
}

log "=== ladder: largest 6-slot pool that boots and survives c6 stress ==="
BIG=0
for c in 786432 753664 720896 688128; do boot "L6-$c" 6 $c; rc=$?; [ $rc = 2 ] && { finish ABORTED; exit 3; }
  [ $rc = 0 ] || continue
  if stress "L6-$c"; then BIG=$c; log "6 slots @ $c boot and survive c6 stress (VRAM free MiB $(vram))"; break
  else log "6 slots @ $c boot but do NOT survive c6 stress: $(sudo docker logs --tail 80 flashnext 2>&1 | grep -aE 'out of memory|Error' | tail -1 | cut -c1-200)"
    sudo docker logs flashnext > "$R/stress-$c.docker.log" 2>&1; fi; done
[ "$BIG" = 0 ] && { log "no 6-slot pool down to 688,128 survives; stopping"; finish DONE; exit 0; }
log "6-slot pool: $BIG ($(( (BIG - 786432) * 100 / 786432 )) % vs 786,432)"

FPOK6=1
for arm in "S4|4|786432" "S6|6|$BIG" "S4b|4|786432" "S6b|6|$BIG"; do
  IFS='|' read -r tag mbs cache <<< "$arm"
  boot "$tag" "$mbs" "$cache" || { finish ABORTED; exit 3; }
  if ! fp "$tag"; then
    [ "$mbs" = 4 ] && { log "[$tag] 4-slot control not canonical"; finish ABORTED; exit 3; }
    FPOK6=0; log "[$tag] 6-slot fingerprints NOT canonical: no promotion"; fi
  case $tag in S4|S6) speed "$tag";; esac
  replay "$tag"
  alive || { log "[$tag] server not alive after the arm"; finish ABORTED; exit 3; }
done

log "=== decision ==="
python3 - "$R" "$FPOK6" <<'PY' 2>&1 | tee "$R/decision.txt" | tee -a "$R/audit.log"
import json,sys,statistics as st,pathlib
R=pathlib.Path(sys.argv[1]); fpok=sys.argv[2]=="1"
def agg(tag):
    out={}
    for l in open(R/f"records-{tag}.jsonl"):
        r=json.loads(l); kind=r.get("tag","").rsplit("-",1)[-1]; out.setdefault((kind,r["conc"]),[]).append(r["aggregate_tps"])
    return {k:st.mean(v) for k,v in out.items()}
a,b=agg("S4"),agg("S6"); ok=fpok; lines=[]
for k in sorted(a):
    d=(b.get(k,0)/a[k]-1)*100; lines.append(f"{k[0]} c{k[1]}: S4 {a[k]:.1f} / S6 {b.get(k,0):.1f} ({d:+.1f} %)")
    if k[1] in (1,4) and d < -2.0: ok=False
rp={}
for l in open(R/"replay.jsonl"):
    r=json.loads(l)
    if r.get("summary"): rp.setdefault(r["tag"].rstrip("b"),[]).append(r)
g4=st.mean(x["gen_tps"] for x in rp["S4"]); g6=st.mean(x["gen_tps"] for x in rp["S6"])
w4=st.mean(x["wall_s"] for x in rp["S4"]); w6=st.mean(x["wall_s"] for x in rp["S6"])
c6=min(b[("code",6)]/a[("code",6)], b[("prose",6)]/a[("prose",6)])
lines.append(f"agent replay: S4 {g4:.1f} gen tok/s, wall {w4:.0f} s / S6 {g6:.1f} gen tok/s, wall {w6:.0f} s ({(g6/g4-1)*100:+.1f} %)")
errs=sum(x["errors"] for v in rp.values() for x in v)
if errs: lines.append(f"agent replay had {errs} failed calls: replay comparison void"); g6=0
win = c6 >= 1.05 or g6 >= 1.03*g4
lines.append(f"c6 ratio (min of code/prose) {c6:.3f}; fingerprints {'canonical' if fpok else 'NOT canonical'}")
print("\n".join(lines)); print("DECISION", "PROMOTE-CANDIDATE" if ok and win else "KEEP-4")
PY
grep -q "DECISION PROMOTE-CANDIDATE" "$R/decision.txt" || { log "S6 does not qualify; daily stays at 4 slots"; finish DONE; exit 0; }

log "=== gates on the 6-slot candidate (S6b is serving) ==="
sed -e 's/^MAXBS=${MAXBS:-4}$/MAXBS=${MAXBS:-6}   # R518: 6 slots/' -e "s/^CACHE=\${CACHE:-786432}/CACHE=\${CACHE:-$BIG}   # R518 (6 slots); R511 4 slots: 786432 --/" "$BASE" > "$NEW.new" \
  && chmod 755 "$NEW.new" && mv "$NEW.new" "$NEW"
diff "$BASE" "$NEW" | tee -a "$R/audit.log"
[ "$(diff "$BASE" "$NEW" | grep -c '^>')" = 2 ] || { log "ABORT: r518 launcher did not get exactly two changed lines"; finish ABORTED; exit 3; }
sudo docker rm -f flashnext >/dev/null 2>&1
"${CLEAN_ENV[@]}" bash "$NEW" > "$R/boot-G.log" 2>&1 || { log "G1 FAIL: r518 launcher exit"; finish ABORTED; exit 3; }
for i in $(seq 120); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
case "$(cfgline)" in *"cache_size: $BIG cache_mode: 8,8 max_batch_size: 6 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;;
  *) log "G1 FAIL: config '$(cfgline)'"; finish ABORTED; exit 3;; esac
fp G || { log "G1 FAIL: fingerprints"; finish ABORTED; exit 3; }
fallback cand >/dev/null
cmp -s "$R/fallback-live.json" "$R/fallback-cand.json" && log "G0 PASS" || { log "G0 FAIL: $(cat "$R/fallback-cand.json")"; finish ABORTED; exit 3; }
python3 /srv/qwen5090/probes/agentic-edit.py --url "$API" --model "$MODEL" --tag S6 --conc 1 4 --modes greedy sampled --out "$R/agentic-edit.jsonl" 2>&1 | tee "$R/agentic-edit.log" | grep -a "agentic-edit" | tee -a "$R/audit.log"
[ "$(grep -ac "6/6 ok" "$R/agentic-edit.log")" = 4 ] && alive && log "G2 PASS" || { log "G2 FAIL"; finish ABORTED; exit 3; }
python3 /srv/qwen5090/probes/fn_needle_oai.py --url "$API" --model "$MODEL" --tag needle --ctx-tokens 131072 240000 \
  --fracs 0.08 0.3 0.55 0.8 0.96 --out "$R/needle.jsonl" 2>&1 | grep -E "retrieved|FAIL|Error" | tee "$R/needle.log" | tee -a "$R/audit.log"
[ "$(grep -c '5/5 retrieved' "$R/needle.log")" = 2 ] && log "G3 PASS" || { log "G3 FAIL"; finish ABORTED; exit 3; }
( cd "$HOME" && timeout 10800 tool-eval-bench --base-url "http://127.0.0.1:8022" --model "$MODEL" --temperature 0.6 --top-p 0.95 --top-k 20 \
    --trials 4 --parallel 8 --json-file "$R/tooleval.json" > "$R/tooleval.log" 2>&1 )
python3 /srv/qwen5090/probes/tooleval_summary.py "$R/tooleval.json" S6 2>&1 | tee -a "$R/audit.log"
mean=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trial_statistics"]["final_score_mean"])' "$R/tooleval.json" 2>/dev/null)
[ -n "$mean" ] && python3 -c "import sys; sys.exit(0 if float('$mean') >= 82.0 else 1)" && log "G4 PASS ($mean)" || { log "G4 FAIL (${mean:-unparsed})"; finish ABORTED; exit 3; }
alive || { log "server not alive after the gates"; finish ABORTED; exit 3; }

log "=== PROMOTE: launch-flashnext.sh := launch-flashnext-r518.sh ==="
sudo cp -p "$LIVE" /srv/qwen5090/launch-flashnext.sh.pre-r518
cp "$NEW" "$LIVE.new" && chmod 755 "$LIVE.new" && mv "$LIVE.new" "$LIVE"
cmp -s "$NEW" "$LIVE" || { log "PROMOTE FAIL"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r518 "$LIVE"; finish ABORTED; exit 3; }
PROMOTED=1
sudo docker rm -f flashnext >/dev/null 2>&1
"${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || { log "LIVE LAUNCH FAILED — rolling back"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r518 "$LIVE"; PROMOTED=0; finish ABORTED; exit 3; }
for i in $(seq 120); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
log "daily now: $(served_id) $(cfgline); VRAM free MiB $(vram)"
[ "$(served_id)" = "$MODEL" ] || { log "daily did not come up — rolling back"; sudo cp -p /srv/qwen5090/launch-flashnext.sh.pre-r518 "$LIVE"; PROMOTED=0; finish ABORTED; exit 3; }
finish PROMOTED
