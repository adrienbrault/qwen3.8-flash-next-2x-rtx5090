#!/usr/bin/env bash
# R532 — NVMe prefix tier round 4 on the SERVED chain (patches/exllamav3/nvme-tier/r4, Opus round): image
# tabbyapi:nvme-tier-r4 = live image (mtp-pruned-r1-tc1-plefix) + tier overlay (variant stack-r4-e3r2+mtp-pruned-r1: the
# pruned-draft generator.py plus the tier's five hunks). Opt-in EXL3_NVME_TIER=<dir>. Spec box-ab-spec.md §R4.
# Differences from R526: the daily EXTRA_ENV verbatim (device draft chain ON), BASE = live image, no plefix layering (the base
# has it; install.py refuses a base without it), fresh dirs r4a/r4b, launcher patch at fuzz 0, and step 11: a counterbalanced
# idle-tier decode A/B, 8 boots A B B A B A A B (A = the daily as served, B = r4 image with the tier ON on the drained dir a).
#   OFF boot (r4 image, no tier): fingerprints must equal the daily canonical (tier off = pruned generator on hardware)
#   boot 1 ON fresh dir a: fingerprints = OFF; fill 30k+120k cold/warm; wait drained
#   boot 2 crash restart: verify restored = warm (and cold); idle bench; decode during drain of a fresh 120k chain
#   boot 3 fresh dir b, cap 2 GiB: churn
#   step 11: 8-boot A/B idle decode, fn_bench code c1 x3 / c4 x2, warm-up 1
# GPU TIMEBOX 45 min. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r532-nvme-tier-r4; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
SRC=/srv/qwen5090/overlay-src/nvme-tier-r4
IMG=tabbyapi:nvme-tier-r4
T=$SRC/tests/gpu_nvme_ab.py
DA=/srv/qwen5090/fast/exl3-nvme-r4a; DB=/srv/qwen5090/fast/exl3-nvme-r4b
CANON1=e7fb377c987d685c; CANON30=4a255910dee2d9c5
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r532] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
finish(){ if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  sudo du -sh "$DA" "$DB" 2>/dev/null | tr '\n' ' ' | sed 's/^/tier dirs kept: /' | tee -a "$R/audit.log"; echo
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R532 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$SRC/Dockerfile.box" "$SRC/launcher-nvme-tier.patch" "$T" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
ENVS=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE"); [ -n "$ENVS" ] || { log "ABORT: no daily EXTRA_ENV"; exit 3; }
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")
# The tier is built ON the live image (variant picked from installed file hashes); only the plefix served chain qualifies.
case "$LIMG" in tabbyapi:mtp-pruned-r1-*plefix*) ;; *) log "ABORT: live image '$LIMG' is not the plefix served chain"; exit 3;; esac
cp "$LIVE" "$R/launch-nvme.sh" && patch --fuzz=0 "$R/launch-nvme.sh" < "$SRC/launcher-nvme-tier.patch" > "$R/launcher-patch.log" 2>&1 || { log "ABORT: launcher patch: $(tr '\n' ' ' < "$R/launcher-patch.log" | cut -c1-200)"; exit 3; }
read -r size avail < <(df -B1 --output=size,avail /srv/qwen5090/fast | tail -1)
[ $(( avail - size * 15 / 100 )) -gt $(( 6 * 1024 * 1024 * 1024 )) ] || { log "ABORT: /srv/qwen5090/fast has $((avail>>30)) GiB free of $((size>>30)), need > 15 % + 6 GiB"; exit 3; }
log "building $IMG on $LIMG (before the lock); env: $ENVS"
(cd "$SRC" && sudo docker build --build-arg BASE="$LIMG" -f Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "ABORT: build failed: $(tail -3 "$R/build.log" | tr '\n' ' ' | cut -c1-240)"; exit 3; }
grep -a "nvme-tier smoke OK\|files installed\|ple-ckpt-clone" "$R/build.log" | cut -c1-200 | tee -a "$R/audit.log"
grep -aq "stack-r4-e3r2+mtp-pruned-r1: 5 files installed" "$R/build.log" || { log "ABORT: wrong tier variant"; exit 3; }
gsha=$(sudo docker run --rm --entrypoint sha256sum "$IMG" /opt/venv/lib/python3.12/site-packages/exllamav3/generator/generator.py | cut -c1-8)
[ "$gsha" = b03ce7fe ] || { log "ABORT: installed generator.py $gsha, want b03ce7fe"; exit 3; }
export GPU_QUEUE_NAME=r532-nvme-tier-r4
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 2700 ))
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
  # the ON line, not the last tier line: r3b logs "open scan: N/N checkpoints intact" after it on a restart (R526 try 3)
  local line; line=$(sudo docker logs flashnext 2>&1 | grep -a -- ' -- nvme tier: ON' | tail -1 | cut -c1-260)
  [ -n "$line" ] || line=$(sudo docker logs flashnext 2>&1 | grep -a -- ' -- nvme tier' | tail -1 | cut -c1-260)
  log "UP $tag (tier '${dir:-off}' cap ${gb:-default}): ${line:-no nvme tier line}"
  sudo docker logs flashnext 2>&1 | grep -aq "EXL3_EMBED_GPU_PRUNED=1 inactive" && { log "FAIL $tag: pruned mirror inactive"; return 1; }
  [ "$(sudo docker logs flashnext 2>&1 | grep -ac 'embedding mirror')" -gt 0 ] || { log "FAIL $tag: no embedding mirror line (device draft chain off?)"; return 1; }
  if [ -n "$dir" ]; then case "$line" in *"nvme tier: ON"*) ;; *) log "FAIL $tag: tier not ON"; return 1;; esac
  else [ -z "$line" ] || case "$line" in *"nvme tier: ON"*) log "FAIL $tag: tier ON without NVME_TIER"; return 1;; esac; fi; }
bench(){ for c in 1 4; do python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-c$c" --kind code --tokens 2048 --warmup-runs 1 \
  --conc $c --runs 2 --out "$R/records.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$1]/" | cut -c1-200 | tee -a "$R/audit.log"; done; }
step(){ log "--- $*"; }
rc=0
# OFF baseline
boot OFF "" "" || { finish ABORTED; exit 3; }
F1=$(greedy OFF); F30=$(greedy30k OFF); log "[OFF] fingerprints c1 $F1 / 30k $F30"
[ "$F1" = "$CANON1" ] && [ "$F30" = "$CANON30" ] || { log "G0 FAIL: tier-off r4 image fingerprints $F1 / $F30, daily $CANON1 / $CANON30"; finish FAILED; exit 1; }
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
sudo docker logs flashnext 2>&1 | grep -a "open scan" | tail -1 | cut -c1-240 | tee -a "$R/audit.log"
step verify; timeout 600 python3 "$T" verify --out "$R" 2>&1 | tail -10 | cut -c1-240 | tee -a "$R/audit.log"
sudo docker logs flashnext 2>&1 | grep -aE "nvme tier: lookup|quarantin|restore aborted" | tail -6 | cut -c1-240 | tee -a "$R/audit.log"
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
grep -ah -- ' -- nvme tier' "$R"/docker-ON*.log | cut -c1-240 | tail -8 | tee -a "$R/audit.log"
# step 11: counterbalanced idle-tier decode, A = the daily as served (live launcher, live image), B = r4 + tier ON (dir a)
step "idle decode A/B, 8 boots"
n=0
for arm in A B B A B A A B; do
  n=$((n+1)); tag="$arm$n"
  [ $(( END - $(date +%s) )) -lt 150 ] && { log "SKIP $tag and later boots: timebox"; break; }
  sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
  if [ "$arm" = A ]; then "${CLEAN_ENV[@]}" bash "$LIVE" > "$R/boot-$tag.log" 2>&1; want="$LIMG"
  else "${CLEAN_ENV[@]}" IMG="$IMG" NVME_TIER="$DA" EXTRA_ENV="$ENVS" bash "$R/launch-nvme.sh" > "$R/boot-$tag.log" 2>&1; want="$IMG"; fi
  for i in $(seq 150); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
  [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.Config.Image}}' flashnext)" = "$want" ] || { log "NO BOOT $tag"; rc=1; break; }
  t=$(sudo docker logs flashnext 2>&1 | grep -ac -- ' -- nvme tier: ON'); { [ "$arm" = B ] && [ "$t" -gt 0 ]; } || { [ "$arm" = A ] && [ "$t" = 0 ]; } || { log "ABORT $tag: tier line count $t"; rc=1; break; }
  fp=$(greedy $tag); log "UP $tag c1 fingerprint $fp"; [ "$fp" = "$CANON1" ] || log "WARNING $tag: fingerprint not canonical"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-c1" --kind code --tokens 2048 --warmup-runs 1 \
    --conc 1 --runs 3 --out "$R/ab.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-200 | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-c4" --kind code --tokens 2048 --warmup-runs 1 \
    --conc 4 --runs 2 --out "$R/ab.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback" | sed "s/^/[$tag]/" | cut -c1-200 | tee -a "$R/audit.log"
done
python3 - "$R" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,math,statistics as st,collections,pathlib
R=pathlib.Path(sys.argv[1]); rounds=collections.defaultdict(list)
for l in open(R/"ab.jsonl"):
    r=json.loads(l); boot,shape=r["tag"].rsplit("-",1); rounds[(boot,shape,r["run"])].append(r)
runs=collections.defaultdict(list)
for (boot,shape,run),rs in rounds.items(): runs[(boot,shape)].append(sum(x["completion_tokens"] for x in rs)/max(x["round_wall_s"] for x in rs))
boots=sorted({b for b,_ in runs}, key=lambda b:int(b[1:]))
T={1:12.71,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306}
for shape in ("c1","c4"):
    bm={b:st.mean(runs[(b,shape)]) for b in boots if (b,shape) in runs}
    A=[v for b,v in bm.items() if b[0]=="A"]; B=[v for b,v in bm.items() if b[0]=="B"]
    print(f"{shape}: boot means " + " ".join(f"{b}={bm[b]:.1f}" for b in bm))
    if len(A)>1 and len(B)>1:
        d=st.mean(B)-st.mean(A); se=math.sqrt(st.variance(A)/len(A)+st.variance(B)/len(B)); ci=T.get(min(len(A),len(B))-1,2.0)*se
        print(f"{shape}: daily (A) {st.mean(A):.1f}, tier ON idle (B) {st.mean(B):.1f} -> B-A {d/st.mean(A)*100:+.2f} %, 95 % CI [{(d-ci)/st.mean(A)*100:+.2f}, {(d+ci)/st.mean(A)*100:+.2f}] % (n {len(A)}+{len(B)} boots)")
PY
[ $rc = 0 ] && finish DONE || finish FAILED
