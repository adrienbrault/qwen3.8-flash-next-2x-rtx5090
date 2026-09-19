#!/usr/bin/env bash
# R428 — mixer V2 r2 on top of the promoted stack (coopwide + host-gap): HC8 = V2 at >= 8 rows (codex default), HC1 = V2 for
# every row count (R426 unit test: 1.35x at R=1 .. 2.25x at R=32, bit-exact). Gates: c1 greedy identical on BOTH arms (HC1 exercises
# V2 at c1, so identity there proves V2 bit-exact in serving), ladder x2. Arms BASE1 / HC8 / HC1 / BASE2; envs joined with "+".
set -uo pipefail
export HOME=${HOME:?}
R=/srv/qwen5090/results/2026-09-17-r428-hcmix2-stack-ab; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-3.05bpw
API=http://127.0.0.1:8022/v1
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
ARMS=${ARMS:-"BASE1=tabbyapi:qsa-cid-pr337-bszn16-coopwide-hostgap|EXL3_HOST_GAP_REWIND=1 HC8=tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap|EXL3_HOST_GAP_REWIND=1+EXL3_HC_MIX_V2=1 HC1=tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap|EXL3_HOST_GAP_REWIND=1+EXL3_HC_MIX_V2=1+EXL3_HC_MIX_V2_MIN_R=1 BASE2=tabbyapi:qsa-cid-pr337-bszn16-coopwide-hostgap|EXL3_HOST_GAP_REWIND=1"}
# arm syntax: NAME=IMAGE or NAME=IMAGE|KEY=VAL (one extra container env)
BOOTED=0
log(){ echo "$(date -Is) [r428] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
wait_id(){ local want=$1 i st; for i in $(seq 240); do [ "$(served_id)" = "$want" ] && return 0
  st=$(cstatus); case "$st" in Restarting*|Exited*) log "container $st"; sudo docker logs --tail 12 flashnext 2>&1 | grep -aE "Error|error|Traceback|CUDA" | tail -4 | cut -c1-240 | tee -a "$R/audit.log"; return 1;; esac
  sleep 2; done; return 1; }
boot(){ # $1 = arm name, $2 = image[|KEY=VAL]
  local img=${2%%|*} env=""; [ "$2" != "${2%%|*}" ] && env=${2#*|}; env=${env//+/ }
  BOOTED=1; log "booting arm $1 IMG=$img EXTRA_ENV='$env' (other launcher defaults = promoted config)"
  IMG="$img" EXTRA_ENV="$env" bash "$LIVE" >> "$R/audit.log" 2>&1 || { log "BOOT FAILED"; return 1; }
  wait_id "$MODEL" || { log "BOOT UNVERIFIED"; return 1; }
  [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = "$img" ] || { log "ABORT: container image is not $img"; return 1; }
  for kv in $env; do [ "$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c "^$kv$")" = 1 ] || { log "ABORT: container env lacks $kv"; return 1; }; done
  log "up: $1 on $img $env"; }
ladder(){ # decode unchanged gate
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1" --kind code --tokens 2048 --conc 1 4 8 --runs 2 \
    --out "$R/records-$1.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$1]/" | tee -a "$R/audit.log"; }
ttft(){ # the prefill measurement: unique 30k and 120k fillers, 64 forced tokens, 3 runs; plus 30k at c4 (mixed drain policy)
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "ttft-$1" --kind prose --tokens 64 --conc 1 --runs 3 --ctx 30000 120000 --unique \
    --out "$R/ttft-$1.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[ttft-$1]/" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "mix-$1" --kind prose --tokens 512 --conc 4 --runs 1 --ctx 30000 --unique --distinct \
    --out "$R/mix-$1.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[mix-$1]/" | tee -a "$R/audit.log"
  log "layout lines: $(sudo docker logs flashnext 2>&1 | grep -ac 'LS prefill pipeline')"; sudo docker logs flashnext 2>&1 | grep -a 'LS prefill pipeline' | head -1 | cut -c1-300 | tee -a "$R/audit.log"; }
greedy30k(){ python3 - "$API" "$MODEL" "$R/greedy30k-$1.json" "$1" <<'PY2' | tee -a "$R/audit.log"
import json,sys,hashlib,urllib.request,random
api,model,out,tag=sys.argv[1:5]
rng=random.Random(4242); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
body=" ".join(rng.choice(words) for _ in range(23000))  # ~30k tokens, deterministic
msg=body+"\n\nSummarize the passage above in one sentence, then write a haiku about it."
req=json.dumps({"model":model,"temperature":0,"max_tokens":128,"min_tokens":128,"messages":[{"role":"user","content":msg}]}).encode()
r=urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=req,headers={"Content-Type":"application/json"}),timeout=900)
d=json.loads(r.read()); open(out,"w").write(json.dumps(d)); m=d["choices"][0]["message"]
t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or "")
print(tag,"30k greedy sha",hashlib.sha256(t.encode()).hexdigest()[:16],"usage",d.get("usage"))
PY2
}
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$MODEL" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(sys.argv[2], "c1 greedy sha", hashlib.sha256(t.encode()).hexdigest()[:16], "usage", d.get("usage"))' "$R/greedy-$1.json" "$1" | tee -a "$R/audit.log"; }
finish(){ if [ "$BOOTED" = 1 ]; then log "restoring the daily (promoted config)"; env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    wait_id "$MODEL" && log "RESTORED: serving $MODEL on $(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" || log "RESTORE UNVERIFIED: $(served_id)"; fi; log "=== R428 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
for arm in $ARMS; do im=${arm#*=}; im=${im%%|*}; sudo docker image inspect "$im" >/dev/null 2>&1 || { log "ABORT: image $im missing"; exit 3; }; done
# swebench-rearchive holds the GPU lock per image (since 2026-09-17 03:35), so it cannot stream through the page cache while this unit holds the lock.
GPU_QUEUE_NAME=r428-hcmix2-stack-ab
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id)"
for arm in $ARMS; do name=${arm%%=*}; env=${arm#*=}
  boot "$name" "$env" || { log "arm $name skipped (boot failed)"; continue; }
  greedy "$name"; ladder "$name"
done
python3 - "$R" <<'PY' | tee -a "$R/audit.log"
import json,sys,pathlib,hashlib,glob
R=pathlib.Path(sys.argv[1])
def sha(p):
    d=json.load(open(p)); m=d["choices"][0]["message"]; return hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16]
shas={p.stem.split("-",1)[1]:sha(p) for p in sorted(R.glob("greedy-*.json"))}
print("fingerprints:", shas)
print("30k fingerprints:", {p.stem.split("-",1)[1]:sha(p) for p in sorted(R.glob("greedy30k-*.json"))})
for p in sorted(R.glob("ttft-*.jsonl"))+sorted(R.glob("mix-*.jsonl")):
    rows=[json.loads(l) for l in open(p) if l.strip() and json.loads(l).get("ok")]
    by={}
    for r in rows: by.setdefault((r.get("ctx_requested"),r.get("conc")),[]).append(r.get("ttft_s"))
    print(p.stem, {f"ctx{k[0]}_c{k[1]}": {"ttft_med": sorted(v)[len(v)//2], "n": len(v)} for k,v in sorted(by.items())})
for p in sorted(R.glob("records-*.jsonl")):
    rows=[json.loads(l) for l in open(p) if l.strip()]
    out={}
    for c in sorted({r.get("conc") for r in rows if r.get("ok")}):
        g=[r for r in rows if r.get("ok") and r.get("conc")==c]
        runs=sorted({r.get("run") for r in g})
        aggs=[]
        for run in runs:
            gr=[r for r in g if r.get("run")==run]; wall=gr[0].get("round_wall_s") or 0
            aggs.append(round(sum(r["completion_tokens"] for r in gr)/wall,1) if wall else None)
        dec=sorted(r["decode_tps"] for r in g if r.get("decode_tps"))
        out[f"c{c}"]={"agg_per_run":aggs,"decode_med":dec[len(dec)//2] if dec else None}
    print(p.stem, json.dumps(out))
PY
finish DONE
