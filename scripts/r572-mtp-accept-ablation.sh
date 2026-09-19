#!/usr/bin/env bash
# R572 — why does exllamav3's MTP accept fewer drafts than the vLLM route? (user: "Why does exllamav3 mtp suck so much? ... with
# vllm it worked much better"). Same fn_bench code prompt, greedy, 2,048 tokens, c1: vLLM route d-mtp3 2.26 accepted drafts per
# verification (R476, 3.05 bpw, BF16 KV; R472b 2.24); exllamav3 daily 1.566 code / 1.547 prose (R564 B arm, 2.50 bpw, 8-bit KV,
# per position 0.738 / 0.504 / 0.326). Ablation on the R564 stats image (EXL3_DRAFT_TOPK_STATS: per-request rounds + per-position
# accepts), tier off, fn_bench code + prose 2,048 c1 greedy, one request each (deterministic):
#   S    served flags, served pool (control: must reproduce R564's 798 / 804 rounds; c1 fingerprint canonical)
#   D1   DRAFT_POLICY '[[8, 1]]'           -> p0 without recursive drafting (state / position bug check: p0 should equal S's)
#   H    minus EXL3_MTP_HEAD_N              -> full draft vocabulary (pruned head cost)
#   M    minus EXL3_HC_MIX_V2_INT8          -> int8 mixer weights off (target numerics)
#   DC   draft_cache_mode FP16 (launcher copy)                                    -> draft KV precision
#   K16  cache_mode FP16 + draft FP16 at pool 393,216 (launcher copy, diagnostic) -> target KV precision (BF16-class KV like vLLM)
#   V    EXTRA_ENV empty (every served overlay flag off) at pool 655,360        -> engine overlays as a whole
# Reading (fixed now): an arm moves acceptance if accepted drafts per round differs from S by >= 0.10 on code or prose. If no arm
# moves it and D1's p0 equals S's p0, the gap to vLLM is the 2.50 checkpoint itself (target + draft head), not the engine.
# Measurement only; K16 / DC are diagnostics above the 8-bit KV floor, never served. GPU TIMEBOX 20 min.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r572-mtp-accept-ablation; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
C1=f4add302e176d78e
SDIR_H=/srv/qwen5090/.exl3cache/draft-topk-r572; SDIR_C=/exl3-cache/draft-topk-r572
LIMG=tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16; NIMG="$LIMG-topkstats"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r572] $*" | tee -a "$R/audit.log"; }
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
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R572 $1 ==="; }
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$NEWM" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
up(){ local tag=$1 i st lp; shift
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" "$@" bash "${L:-$LIVE}" > "$R/boot-$tag.log" 2>&1 ) & lp=$!
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
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
sudo docker image inspect "$NIMG" >/dev/null 2>&1 || { log "ABORT: $NIMG missing (built by R564)"; exit 3; }
LENV=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE" | tr ' ' '\n' | grep -v '^EXL3_NGRAM_PREFETCH2=' | tr '\n' ' ' | sed 's/ $//')
[ -n "$LENV" ] || { log "ABORT: cannot parse EXTRA_ENV"; exit 3; }
without(){ echo "$LENV" | tr ' ' '\n' | grep -v "^$1=" | tr '\n' ' ' | sed 's/ $//'; }
STATS="EXL3_DRAFT_TOPK_STATS=1 EXL3_DRAFT_TOPK_STATS_DIR=$SDIR_C"
# launcher copies for the two KV diagnostics (the live launcher is never edited)
LDC=$R/launch-dcfp16.sh; LK16=$R/launch-k16.sh
sed 's/^  draft_cache_mode: Q8$/  draft_cache_mode: FP16/' "$LIVE" > "$LDC"
sed -e 's/^  draft_cache_mode: Q8$/  draft_cache_mode: FP16/' -e 's/^case "\$CACHE_MODE" in \[2-8\],\[2-8\]) ;; \*)/case "$CACHE_MODE" in FP16|[2-8],[2-8]) ;; *)/' "$LIVE" > "$LK16"
[ "$(diff "$LIVE" "$LDC" | grep -c '^>')" = 1 ] && [ "$(diff "$LIVE" "$LK16" | grep -c '^>')" = 2 ] || { log "ABORT: launcher copies ($(diff "$LIVE" "$LK16" | grep -c '^>') lines)"; diff "$LIVE" "$LK16" | tee -a "$R/audit.log"; exit 3; }
sudo rm -rf "$SDIR_H"; sudo mkdir -p "$SDIR_H"; sudo chmod 777 "$SDIR_H"
export GPU_QUEUE_NAME=r572-mtp-accept-ablation
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id)"
BOOTED=1
END=$(( $(date +%s) + 1200 ))
: > "$R/arms.tsv"
arm(){ local tag=$1 k n0 n1 a; shift
  [ $(( END - $(date +%s) )) -lt 120 ] && { log "timebox: skipping $tag"; return 1; }
  up "$tag" NVME_TIER= IMG="$NIMG" "$@" || return 1
  a=$(greedy "$tag")
  log "UP $tag: $(sudo grep -E '^  (cache_size|cache_mode|draft_cache_mode|draft_num_tokens_by_batch):' /srv/qwen5090/flashnext-config.yml | awk '{$1=$1; print}' | tr '\n' ' '); stats env $(sudo docker exec flashnext env | grep -c '^EXL3_DRAFT_TOPK_STATS=1'); free $(vram); c1 $a"
  [ "$tag" = S ] && [ "$a" != "$C1" ] && log "S: c1 $a != canonical $C1"
  n0=$(sudo cat "$SDIR_H/draft_topk.jsonl" 2>/dev/null | wc -l)
  for k in code prose; do python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$tag-$k" --kind $k --tokens 2048 --warmup-runs 0 --conc 1 --runs 1 \
    --out "$R/c1.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | cut -c1-200 | sed "s/^/[$tag $k] /" | tee -a "$R/audit.log"; done
  sleep 2; n1=$(sudo cat "$SDIR_H/draft_topk.jsonl" 2>/dev/null | wc -l)
  sudo sed -n "$((n0+1)),${n1}p" "$SDIR_H/draft_topk.jsonl" > "$R/stats-$tag.jsonl"
  printf "%s\t%s\n" "$tag" "$a" >> "$R/arms.tsv"
  sudo docker logs flashnext > "$R/docker-$tag.log" 2>&1
  alive || log "$tag: NOT ALIVE"; }
arm S   EXTRA_ENV="$LENV $STATS"
arm D1  EXTRA_ENV="$LENV $STATS" DRAFT_POLICY='[[8, 1]]'
arm H   EXTRA_ENV="$(without EXL3_MTP_HEAD_N) $STATS"
arm M   EXTRA_ENV="$(without EXL3_HC_MIX_V2_INT8) $STATS"
L=$LDC  arm DC  EXTRA_ENV="$LENV $STATS"
L=$LK16 arm K16 EXTRA_ENV="$LENV $STATS" CACHE_MODE=FP16 CACHE=393216
arm V   EXTRA_ENV="$STATS" CACHE=655360
python3 - "$R" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,os
R=sys.argv[1]; res={}
for l in open(f"{R}/arms.tsv"):
    tag,fp=l.split()
    reqs=[json.loads(x) for x in open(f"{R}/stats-{tag}.jsonl") if '"rec":"request"' in x.replace(" ","")]
    reqs=[r for r in reqs if r.get("tokens")==2048]
    out=[]
    for kind,r in zip(("code","prose"),reqs):
        n=r["rounds"]; pos=[r["pos"][k][0]/n for k in sorted(r["pos"])]
        out.append((kind,r["tokens"]/n-1,pos))
        print(f"{tag:4s} {kind:5s} c1 {fp}: rounds {n}, accepted drafts/round {r['tokens']/n-1:.3f}, per position {[round(p,3) for p in pos]}")
    if len(reqs)!=2: print(f"{tag}: {len(reqs)} 2,048-token requests recorded (want 2)")
    res[tag]=out
S={k:a for k,a,_ in res.get("S",[])}
for tag,out in res.items():
    if tag=="S": continue
    for k,a,pos in out:
        if k in S: print(f"{tag} vs S {k}: {a-S[k]:+.3f} accepted drafts/round{'  <- MOVES' if abs(a-S[k])>=0.10 and tag!='D1' else ''}")
if "D1" in res and "S" in res:
    for (k,a,p),(k2,a2,p2) in zip(res["D1"],res["S"]): print(f"D1 p0 {p[0]:.3f} vs S p0 {p2[0]:.3f} ({k})")
PY
finish DONE
