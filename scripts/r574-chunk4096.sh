#!/usr/bin/env bash
# R574 — backlog S3(b): prefill chunk 4096 on the served daily (R565 launcher + image, tier off, 8 slots @ 966,656). R553 phase C
# never measured it with salted prompts; E3-DET prefill scratch grows with the chunk (~400 MiB at 4096 vs cuda:1's floor), so
# headroom and a loaded survival pass decide as much as the prefill rate.
#   S  CHUNK 2048 (served)   C4 CHUNK 4096   (A B, one boot each; then S again only if C4 looks like a win: S2 control)
#   per arm: boot free per card, c1 fingerprint, 3 salted cold prefills at 30k / 60k / 120k (fn_bench --ctx), then survival:
#   one 120k-class cold prefill while a c4 code round decodes (the loaded case), server alive after.
# Rule (fixed now): C4 is a promotion candidate iff boot free >= S - 32 MiB on both cards, survival holds, c1 fingerprint = S's,
# and mean prefill >= 1.05 x S at 60k and at 120k. Screen only (a promotion unit follows with S2 + gates). GPU TIMEBOX 20 min.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r574-chunk4096; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r574] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){
  if [ "$BOOTED" = 1 ]; then sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" = "$NEWM" ] && break; sleep 2; done; log "daily: $(served_id)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R574 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
grep -q '^CHUNK=${CHUNK:-2048}$' "$LIVE" || { log "ABORT: live launcher chunk is not 2048"; exit 3; }
export GPU_QUEUE_NAME=r574-chunk4096
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id)"
BOOTED=1
SALT=$(( $(date +%s) % 100000 ))
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$NEWM" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
up(){ local tag=$1 i st lp; shift
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 ) & lp=$!
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
pf(){ local tag=$1 c k
  for k in 1 2 3; do for c in 30000 60000 120000; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "pf-$tag-$c" --kind prose --tokens 64 --conc 1 --runs 1 --ctx $c --unique \
      --salt $(( SALT + c/1000 + k*997 + ${#tag}*13 + RANDOM )) --out "$R/prefill.jsonl" > /dev/null 2>&1; done; done; }
loaded(){ local tag=$1 bp
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "load-$tag" --kind code --tokens 1024 --conc 4 --runs 1 --out "$R/loaded.jsonl" > "$R/loaded-$tag.log" 2>&1 & bp=$!
  sleep 3
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "loadpf-$tag" --kind prose --tokens 64 --conc 1 --runs 1 --ctx 120000 --unique \
    --salt $(( SALT + 777 + RANDOM )) --out "$R/loaded.jsonl" >> "$R/loaded-$tag.log" 2>&1
  wait $bp
  grep -E "^  c=|FAILED|Traceback" "$R/loaded-$tag.log" | cut -c1-200 | sed "s/^/[$tag loaded] /" | tee -a "$R/audit.log"
  alive; }
: > "$R/arms.tsv"
for arm in S C4; do
  ch=2048; [ $arm = C4 ] && ch=4096
  up $arm NVME_TIER= CHUNK=$ch || { log "NO BOOT $arm"; continue; }
  f=$(vram); fp=$(greedy $arm)
  log "UP $arm: chunk $(sudo grep -E '^  chunk_size:' /srv/qwen5090/flashnext-config.yml | awk '{print $2}'); free at boot $f; c1 $fp"
  pf $arm
  loaded $arm && ok=alive || ok=DEAD
  log "$arm loaded survival: $ok"
  printf "%s\t%s\t%s\t%s\n" $arm "$f" "$fp" $ok >> "$R/arms.tsv"
  sudo docker logs flashnext > "$R/docker-$arm.log" 2>&1
done
python3 - "$R/prefill.jsonl" "$R/arms.tsv" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
d=collections.defaultdict(list)
for r in map(json.loads,open(sys.argv[1])):
    if r.get("ttft_s") and r.get("prompt_tokens"): d[r["tag"]].append(r["prompt_tokens"]/r["ttft_s"])
m={}
for t in sorted(d): m[t]=st.mean(d[t]); print(f"{t}: {', '.join(f'{x:.0f}' for x in d[t])} mean {m[t]:.0f}")
arms={l.split("\t")[0]:l.rstrip("\n").split("\t") for l in open(sys.argv[2])}
why=[]
if "S" not in arms or "C4" not in arms: why.append("arm missing")
else:
    s,c=arms["S"],arms["C4"]
    sf=[int(x) for x in s[1].strip("/").split("/")]; cf=[int(x) for x in c[1].strip("/").split("/")]
    for i in (0,1):
        if cf[i] < sf[i]-32: why.append(f"headroom cuda:{i} {cf[i]} < {sf[i]}-32")
    if c[2]!=s[2]: why.append(f"c1 {c[2]} != {s[2]}")
    if c[3]!="alive": why.append("loaded survival")
    for ctx in ("30000","60000","120000"):
        a,b=m.get(f"pf-S-{ctx}"),m.get(f"pf-C4-{ctx}")
        if a and b:
            print(f"{ctx}: C4/S {b/a:.3f}")
            if ctx!="30000" and b < 1.05*a: why.append(f"{ctx} {b/a:.3f} < 1.05")
print("C4 CANDIDATE" if not why else "C4 NO: "+"; ".join(why))
PY
finish DONE
