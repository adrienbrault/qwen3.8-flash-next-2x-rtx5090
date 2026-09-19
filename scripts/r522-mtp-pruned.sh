#!/usr/bin/env bash
# R522 — MTP device draft chain on a pruned embedding mirror (Opus round mtp-device-draft-pruned r1, survey P2b;
# patches/exllamav3/mtp-device-draft-pruned/r1). Image tabbyapi:mtp-pruned-r1 = stack-r4-e3r2 + pure-Python overlay (flag
# default off). ON = daily EXTRA_ENV + EXL3_MTP_DEVICE_DRAFT=1 EXL3_EMBED_GPU=1 EXL3_EMBED_GPU_PRUNED=1: a 320 MiB mirror of
# the 65,536 draft-head rows on cuda:1 instead of r4's 1.27 GB full mirror, so the draft chain stays on the device.
# Spec box-ab-spec.md: G0 standalone identity tests (unit + chain identity off/pruned/full) → ON arm → OFF arm, each: boot
# log line, VRAM, fingerprints (canonical c1 ae890c45 / 30k 4a255910), fn_bench code/prose c1/c4 ×2 (+c6 when the live
# launcher runs 6 slots), c4 30k stress + VRAM floor (hold if cuda:1 < 600 MiB). Promote only if byte-identical and c1 up
# beyond run spread in code and prose, c4 not worse (decided by hand from the summary). GPU TIMEBOX 15 min.
# RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=${HOME:?}
R=/srv/qwen5090/results/2026-09-19-r522-mtp-pruned; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/mtp-pruned-r1
IMG=tabbyapi:mtp-pruned-r1
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
ONENV="EXL3_MTP_DEVICE_DRAFT=1 EXL3_EMBED_GPU=1 EXL3_EMBED_GPU_PRUNED=1"
BOOTED=0
log(){ echo "$(date -Is) [r522] $*" | tee -a "$R/audit.log"; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
finish(){ sudo docker rm -f r522-probe >/dev/null 2>&1 || true
  if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"
    sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R522 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$SRC/Dockerfile.box" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
ENVS=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE")
[ -n "$ENVS" ] || { log "ABORT: cannot read the daily EXTRA_ENV"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE"); [ "$LIMG" = tabbyapi:stack-r4-e3r2 ] || { log "ABORT: live image is '$LIMG', overlay base is stack-r4-e3r2"; exit 3; }
MAXBS=$(sed -n 's/^MAXBS=\${MAXBS:-\([0-9]*\)}.*/\1/p' "$LIVE"); CONCS="1 4"; [ "${MAXBS:-4}" -ge 6 ] && CONCS="1 4 6"
log "building $IMG (CPU only, before the lock); live slots $MAXBS, env: $ENVS"
(cd "$SRC" && sudo docker build -t "$IMG" -f Dockerfile.box . ) > "$R/build.log" 2>&1 || { log "ABORT: build failed: $(tail -3 "$R/build.log" | tr '\n' ' ' | cut -c1-240)"; exit 3; }
export GPU_QUEUE_NAME=r522-mtp-pruned
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 900 ))
log "lock held; timebox ends $(date -Is -d @$END)"
BOOTED=1
sudo docker stop -t 30 flashnext >/dev/null 2>&1; sudo docker rm -f flashnext >/dev/null 2>&1; sleep 3
DENV=(); for kv in $ENVS; do DENV+=(-e "$kv"); done
log "G0: standalone identity tests"
timeout 420 sudo docker run --rm --name r522-probe --gpus all --ipc=host --shm-size=16g "${DENV[@]}" \
  -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
  -v /srv/qwen5090/models:/models:ro -v "$R":/out --entrypoint sh "$IMG" -c \
  "python3 /opt/mtp-pruned-r1/tests/gpu_unit_pruned_embed.py --out /out/gpu-unit.json > /out/gpu-unit.log 2>&1; echo unit rc=\$?; python3 /opt/mtp-pruned-r1/tests/gpu_chain_identity.py --out /out > /out/chain.log 2>&1; echo chain rc=\$?" 2>&1 | tee -a "$R/audit.log"
U=$(tail -1 "$R/gpu-unit.log" 2>/dev/null); C=$(tail -1 "$R/chain.log" 2>/dev/null)
log "G0 unit: $U | chain: $C"
case "$U" in *"ALL PASS"*) ;; *) log "G0 FAIL (unit)"; finish G0-FAIL; exit 1;; esac
case "$C" in *PASS*) ;; *) log "G0 FAIL (chain)"; finish G0-FAIL; exit 1;; esac
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("chain-summary", json.dumps({k:d.get(k) for k in ("compare","pruned_host_fallback","chain","tok_s","vram_free_mib_before_after")})[:900])' "$R/chain-summary.json" 2>&1 | tee -a "$R/audit.log"
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
stress30k(){ python3 - "$API" "$MODEL" "$1" <<'PY'
import json,sys,random,time,urllib.request,concurrent.futures as cf
api,model,tag=sys.argv[1:4]; nonce=int(time.time()*1000)%100000
words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
def one(i):
    rng=random.Random(nonce*10+i); body=" ".join(rng.choice(words) for _ in range(23000))
    req=json.dumps({"model":model,"temperature":0,"max_tokens":512,"min_tokens":512,"messages":[{"role":"user","content":f"[{nonce}-{i}] "+body+"\n\nSummarize."}]}).encode()
    d=json.loads(urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=req,headers={"Content-Type":"application/json"}),timeout=900).read())
    return d["usage"]["completion_tokens"] if d.get("usage") else len((d["choices"][0]["message"].get("content") or ""))
with cf.ThreadPoolExecutor(4) as ex: r=list(ex.map(one,range(4)))
print(f"[stress {tag}] 4x30k ok {r}")
PY
}
arm(){ local tag=$1 extra=$2 i
  [ $(( END - $(date +%s) )) -lt 240 ] && { log "SKIP $tag: timebox"; return 0; }
  sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
  "${CLEAN_ENV[@]}" IMG="$IMG" EXTRA_ENV="$ENVS $extra" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 || { log "NO BOOT $tag"; return 1; }
  for i in $(seq 120); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
  [ "$(served_id)" = "$MODEL" ] || { log "NO BOOT $tag"; return 1; }
  [ "$(sudo docker inspect -f '{{.Config.Image}}' flashnext)" = "$IMG" ] || { log "ABORT $tag: container image"; return 1; }
  sudo docker logs flashnext > "$R/docker-$tag.log" 2>&1
  local line; line=$(grep -a "EXL3_EMBED_GPU_PRUNED" "$R/docker-$tag.log" | head -2 | cut -c1-200)
  log "UP $tag: VRAM free MiB $(vram) | mirror log: ${line:-none}"
  if [ -n "$extra" ]; then case "$line" in *"embedding mirror"*) ;; *) log "G1 FAIL $tag: mirror not active"; return 1;; esac
  else [ -z "$line" ] || { log "G1 FAIL $tag: OFF arm logged the mirror"; return 1; }; fi
  local a b; a=$(greedy "$tag"); b=$(greedy30k "$tag"); log "[$tag] fingerprints c1 $a / 30k $b (canonical ae890c45d1000582 / 4a255910dee2d9c5)"
  [ "$a" = ae890c45d1000582 ] && [ "$b" = 4a255910dee2d9c5 ] || { log "G2 FAIL $tag"; return 1; }
  for kind in code prose; do
    timeout $(( END - $(date +%s) > 30 ? END - $(date +%s) : 30 )) python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-$kind" --kind $kind --tokens 2048 \
      --conc $CONCS --runs 2 --out "$R/records.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$tag $kind]/" | cut -c1-220 | tee -a "$R/audit.log"; done
  stress30k "$tag" 2>&1 | tail -2 | tee -a "$R/audit.log"
  log "[$tag] after stress VRAM free MiB $(vram) | errors: $(sudo docker logs flashnext 2>&1 | grep -acE 'out of memory|CUDA error|Traceback')"; }
arm ON "$ONENV"; on=$?
arm OFF ""; off=$?
python3 - "$R/records.jsonl" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,statistics as st,collections
d=collections.defaultdict(list); rounds=collections.defaultdict(list)
for l in open(sys.argv[1]):
    r=json.loads(l); arm,kind=r["tag"].rsplit("-",1); rounds[(arm,kind,r["conc"],r["run"])].append(r)
for (arm,kind,c,run),rs in rounds.items(): d[(arm,kind,c)].append(sum(x["completion_tokens"] for x in rs)/max(x["round_wall_s"] for x in rs))
for kind in ("code","prose"):
    for c in (1,4,6):
        a,b=d.get(("OFF",kind,c)),d.get(("ON",kind,c))
        if a and b: print(f"{kind} c{c}: OFF {st.mean(a):.1f} {[round(x,1) for x in a]} / ON {st.mean(b):.1f} {[round(x,1) for x in b]} -> {(st.mean(b)/st.mean(a)-1)*100:+.1f} %")
PY
log "arms: ON rc=$on OFF rc=$off"
finish DONE
