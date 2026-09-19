#!/usr/bin/env bash
# R526 — NVMe prefix tier round 3 (Opus round, patches/exllamav3/nvme-tier/r3; user 2026-09-19: no host-RAM tier, a
# persistent NVMe tier on /srv/qwen5090/fast instead). Image tabbyapi:nvme-tier-r3 = daily image + Python overlay, opt-in
# EXL3_NVME_TIER=<dir>. Spec box-ab-spec.md, plus a tier-OFF boot of the same image first so the canonical fingerprints and
# the decode baseline come from this session (the daily's EXTRA_ENV may have moved with R525):
#   OFF boot: c1 + 30k fingerprints, fn_bench code c1/c4 (warm-up 1, 2 runs)
#   boot 1 (tier ON, fresh dir a): fingerprints must equal OFF; fill 30k+120k cold/warm; wait drained
#   boot 2 (docker rm -f + relaunch, dir a): same page/ckpt counts, 0 torn; verify = restored output hash == warm hash;
#     fn_bench code c1/c4 (idle tier) vs OFF; decode during drain of a fresh 120k chain
#   boot 3 (fresh dir b, cap 2 GiB): churn 5 x 40k, du <= cap after every drain
# GPU TIMEBOX 20 min. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r526-nvme-tier; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/nvme-tier-r3
IMG=tabbyapi:nvme-tier-r3
T=$SRC/tests/gpu_nvme_ab.py
DA=/srv/qwen5090/fast/exl3-nvme-r3a; DB=/srv/qwen5090/fast/exl3-nvme-r3b
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r526] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
finish(){ if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  sudo du -sh "$DA" "$DB" 2>/dev/null | tr '\n' ' ' | sed 's/^/tier dirs kept: /' | tee -a "$R/audit.log"; echo
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R526 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$SRC/Dockerfile.box" "$SRC/launcher-nvme-tier.patch" "$T" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
ENVS=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE"); [ -n "$ENVS" ] || { log "ABORT: no daily EXTRA_ENV"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE"); [ "$LIMG" = tabbyapi:stack-r4-e3r2 ] || { log "ABORT: live image '$LIMG' is not the overlay base"; exit 3; }
cp "$LIVE" "$R/launch-nvme.sh" && patch --fuzz=0 "$R/launch-nvme.sh" < "$SRC/launcher-nvme-tier.patch" > "$R/launcher-patch.log" 2>&1 || { log "ABORT: launcher patch: $(tr '\n' ' ' < "$R/launcher-patch.log" | cut -c1-200)"; exit 3; }
read -r size avail < <(df -B1 --output=size,avail /srv/qwen5090/fast | tail -1)
[ $(( avail - size * 15 / 100 )) -gt $(( 6 * 1024 * 1024 * 1024 )) ] || { log "ABORT: /srv/qwen5090/fast has $((avail>>30)) GiB free of $((size>>30)), need > 15 % + 6 GiB"; exit 3; }
log "building $IMG (before the lock); env: $ENVS"
(cd "$SRC" && sudo docker build -f Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "ABORT: build failed: $(tail -3 "$R/build.log" | tr '\n' ' ' | cut -c1-240)"; exit 3; }
grep -a "nvme-tier smoke OK" "$R/build.log" | cut -c1-200 | tee -a "$R/audit.log"
export GPU_QUEUE_NAME=r526-nvme-tier
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 1200 ))
log "lock held; timebox ends $(date -Is -d @$END)"
BOOTED=1
sudo rm -rf "$DA" "$DB"
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
boot(){ local tag=$1 dir=$2 gb=$3 i
  sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
  "${CLEAN_ENV[@]}" IMG="$IMG" NVME_TIER="$dir" NVME_TIER_GB="$gb" EXTRA_ENV="$ENVS" bash "$R/launch-nvme.sh" > "$R/boot-$tag.log" 2>&1 || { log "NO BOOT $tag"; return 1; }
  for i in $(seq 150); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
  [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.Config.Image}}' flashnext)" = "$IMG" ] || { log "NO BOOT $tag"; return 1; }
  local line; line=$(sudo docker logs flashnext 2>&1 | grep -a -- ' -- nvme tier' | tail -1 | cut -c1-260)
  log "UP $tag (tier '${dir:-off}' cap ${gb:-default}): ${line:-no nvme tier line}"
  if [ -n "$dir" ]; then case "$line" in *"nvme tier: ON"*) ;; *) log "FAIL $tag: tier not ON"; return 1;; esac
  else [ -z "$line" ] || case "$line" in *"nvme tier: ON"*) log "FAIL $tag: tier ON without NVME_TIER"; return 1;; esac; fi; }
bench(){ for c in 1 4; do python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-c$c" --kind code --tokens 2048 --warmup-runs 1 \
  --conc $c --runs 2 --out "$R/records.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$1]/" | cut -c1-200 | tee -a "$R/audit.log"; done; }
step(){ log "--- $*"; }
rc=0
# OFF baseline
boot OFF "" "" || { finish ABORTED; exit 3; }
F1=$(greedy OFF); F30=$(greedy30k OFF); log "[OFF] fingerprints c1 $F1 / 30k $F30"
bench OFF
# boot 1: tier ON, fresh dir
boot ON1 "$DA" "" || { finish FAILED; exit 1; }
a=$(greedy ON1); b=$(greedy30k ON1); log "[ON1] fingerprints c1 $a / 30k $b (OFF $F1 / $F30)"
[ "$a" = "$F1" ] && [ "$b" = "$F30" ] || { log "G2 FAIL: tier ON changes cold output"; rc=1; }
step fill; timeout 600 python3 "$T" fill --out "$R" --sizes 30000,120000 2>&1 | tail -8 | cut -c1-220 | tee -a "$R/audit.log"
step wait-drained; timeout 300 python3 "$T" wait-drained --out "$R" --container flashnext 2>&1 | tail -4 | cut -c1-260 | tee -a "$R/audit.log"
sudo docker logs flashnext > "$R/docker-ON1.log" 2>&1
# boot 2: crash restart
step crash restart
boot ON2 "$DA" "" || { finish FAILED; exit 1; }
step verify; timeout 600 python3 "$T" verify --out "$R" 2>&1 | tail -10 | cut -c1-240 | tee -a "$R/audit.log"
grep -aq "VERIFY PASS" "$R"/*.log "$R"/*.json 2>/dev/null || grep -aq "VERIFY PASS" "$R/audit.log" || rc=1
bench ON2
if [ $(( END - $(date +%s) )) -ge 330 ]; then step decode during drain; timeout 420 python3 "$T" decode --out "$R" --ntok 120000 --max-tokens 1024 2>&1 | tail -8 | cut -c1-220 | tee -a "$R/audit.log"
else log "SKIP decode-during-drain: timebox"; fi
sudo docker logs flashnext > "$R/docker-ON2.log" 2>&1
# boot 3: cap churn
if [ $(( END - $(date +%s) )) -ge 240 ]; then
  boot ON3 "$DB" 2 || { finish FAILED; exit 1; }
  step churn; timeout 480 python3 "$T" churn --out "$R" --dir "$DB" --cap-gb 2 --n 5 --ntok 40000 2>&1 | tail -8 | cut -c1-220 | tee -a "$R/audit.log"
  grep -aq "CHURN.*PASS" "$R/audit.log" || { rc=1; sudo docker rm -f flashnext >/dev/null 2>&1; }
  sudo docker logs flashnext > "$R/docker-ON3.log" 2>&1
else log "SKIP churn: timebox"; fi
python3 - "$R" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,collections,statistics as st,pathlib
R=pathlib.Path(sys.argv[1]); rounds=collections.defaultdict(list)
for l in open(R/"records.jsonl"):
    r=json.loads(l); boot,shape=r["tag"].rsplit("-",1); rounds[(boot,shape,r["run"])].append(r)
runs=collections.defaultdict(list)
for (b,s,run),rs in rounds.items(): runs[(b,s)].append(sum(x["completion_tokens"] for x in rs)/max(x["round_wall_s"] for x in rs))
for s in ("c1","c4"):
    o=runs.get(("OFF",s)); n=runs.get(("ON2",s))
    if o and n: print(f"code {s}: OFF {st.mean(o):.1f} / tier ON (idle, restored) {st.mean(n):.1f} -> {(st.mean(n)/st.mean(o)-1)*100:+.1f} % (one boot each)")
PY
grep -ah -- ' -- nvme tier' "$R"/docker-ON*.log | cut -c1-240 | tail -8 | tee -a "$R/audit.log"
[ $rc = 0 ] && finish DONE || finish FAILED
