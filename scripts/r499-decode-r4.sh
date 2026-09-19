#!/usr/bin/env bash
# R499 — codex decode round 4 (vcruz305/exllamav3 decode patches + upstream a3b249543) on the daily image.
#
# WHY (2026-09-18 public-repo survey; user: "Decode speed and kv pool size", "isn't there more things to try"): vcruz305's
# native exllamav3 fork (DGX Spark recipe) removed host syncs from the MTP loop (their harness: host glue 8.4 -> 3.1 ms/round),
# pruned the DRAFT lm_head to 64k columns (verify keeps the full head) and stored the hyper-connection mixer weights int8.
# Our c1 d3 step is ~14.8 ms with ~1.5 ms of draft/sampling host gaps and a 1.7 ms mixer. Codex r4
# (flan/patches/exllamav3/decode-kernels/r4, box-ab-spec.md) ported them opt-in, default off.
#
# IMAGE tabbyapi:decode-kernels-r4 = the daily decode-kernels-r2 (shared-expert overlap) + r4 (Dockerfile.box). Launcher-r498
# (= live r491 + DYN off + CKPT_NAME default). Every arm: overlap 1 (= daily), explicit r4 selectors.
#   gate 0  selectors import, gr_mix_v2_int8 binding; CPU tests; int8 mixer kernel vs dequantized reference on both cards
#   group I (identity-preserving: greedy token-identical by construction): EXL3_DRAFT_PINNED_STAGING=1 EXL3_BATCH_VERIFY=1
#           EXL3_MTP_HEAD_N=65536 (EXL3_MTP_DEVICE_DRAFT / EXL3_EMBED_GPU stay 0: the GPU embedding mirror costs 1.18 GiB and
#           the served pool leaves 0.9 GiB on cuda:0) — OFF / ON / OFF2 / ON2, fingerprints canonical on EVERY arm, plus
#           fallback probes (greedy + presence penalty, stop string) identical OFF vs ON
#   group M (numerics change): EXL3_HC_MIX_V2_INT8=1 — OFF / ON / OFF2 / ON2, fingerprints recorded (not gated), VRAM delta
#   each arm: fn_bench code + prose c1/c4 2,048 x 2, multiprompt 12 x kind 2,048 c1/c4
# RUN: sudo systemd-run --unit=r499-decode-r4 --collect -p RuntimeMaxSec=86400 -E HOME=$HOME /bin/bash /srv/qwen5090/r499-decode-r4.sh
# 2026-09-18 23:30 CEST (user: "Stop using 3.05. From now on stay on 2.5"): moved to the 2.50bpw daily — checkpoint
# qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab, pool 786,432, launcher-r511, canonical c1 ae890c45d1000582 / 30k 2aa8d1024daece5c (R511).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r499-decode-r4; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
LIVE=/srv/qwen5090/launch-flashnext.sh
CAND=/srv/qwen5090/launch-flashnext-r511.sh
CFG=/srv/qwen5090/flashnext-config.yml
CTX=/srv/qwen5090/build/decode-kernels-r4
IMG=tabbyapi:decode-kernels-r4
BASE_ENV="EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1"
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r499] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
finish(){ sudo docker rm -f r499-test >/dev/null 2>&1
  if [ "$BOOTED" = 1 ]; then log "restoring the daily (unchanged live launcher)"
    sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done   # any daily (R511 changes the served id)
    log "daily: $(served_id) $(cfgline) VRAM free $(vram)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R499 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$CAND" "$LM" "$CTX/Dockerfile.box" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/probes/fn_needle_oai.py /srv/qwen5090/probes/multiprompt.py; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
export GPU_QUEUE_NAME=r499-decode-r4
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id) $(cfgline)"

log "=== build $IMG (daily keeps serving) ==="
( cd "$CTX" && sudo docker build -f Dockerfile.box -t "$IMG" . ) > "$R/build.log" 2>&1 || { log "BUILD FAILED: $(grep -aE 'error|Error' "$R/build.log" | tail -3 | cut -c1-240)"; finish ABORTED; exit 3; }
log "built $IMG $(sudo docker image inspect -f '{{.Id}}' "$IMG" | cut -c8-19)"
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1
for i in $(seq 30); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 2; done
# boot TAG REPLAY CACHE -> 0 serving with that config and image, 1 no boot, 2 mismatch
boot(){ local tag=$1 sel=$2 cache=786432 i st lp got
  log "boot $tag: r4 selectors: $sel"
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" IMG="$IMG" CACHE=$cache EXTRA_ENV="$BASE_ENV $sel" bash "$CAND" >> "$R/boot-$tag-$cache.log" 2>&1 ) & lp=$!
  for i in $(seq 360); do
    if [ "$(served_id)" = "$MODEL" ]; then wait $lp; got=$(cfgline)
      case "$got" in *"cache_size: $cache cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] "*"chunk_size: 2048 vision: true"*) ;; *) log "ABORT: config '$got'"; return 2;; esac
      [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = "$IMG" ] || { log "ABORT: container image is not $IMG"; return 2; }
      for kv in $sel; do [ "$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c "^$kv$")" = 1 ] || { log "ABORT: container env lacks $kv"; return 2; }; done
      log "UP $tag @ $cache: VRAM free MiB $(vram)"; return 0; fi
    st=$(cstatus); case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 60 ] && { sleep 3; continue; }
      log "NO BOOT $tag @ $cache ($st): $(sudo docker logs --tail 60 flashnext 2>&1 | grep -aE 'RuntimeError|Error|VRAM' | tail -1 | cut -c1-220)"
      sudo docker logs flashnext > "$R/boot-$tag-$cache.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1;; esac
    [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 0)" -ge 1 ] && {
      log "NO BOOT $tag @ $cache (restart loop): $(sudo docker logs --tail 60 flashnext 2>&1 | grep -aE 'RuntimeError|Error|VRAM' | tail -1 | cut -c1-220)"
      sudo docker logs flashnext > "$R/boot-$tag-$cache.docker.log" 2>&1; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1; }
    sleep 3; done
  log "NO BOOT $tag @ $cache (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }
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
fp(){ local a b; a=$(greedy "$1"); b=$(greedy30k "$1"); log "[$1] fingerprints c1 $a / 30k $b (canonical ae890c45d1000582 / 2aa8d1024daece5c)"
  [ "$a" = ae890c45d1000582 ] && [ "$b" = 2aa8d1024daece5c ]; }
speed(){ local t=$1 kind
  for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$t-$kind" --kind $kind --tokens 2048 --conc 1 4 --runs 2 \
      --out "$R/records-$t.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$t $kind]/" | cut -c1-260 | tee -a "$R/audit.log"; done
  python3 /srv/qwen5090/probes/multiprompt.py --url "$API" --model "$MODEL" --tag "$t" --tokens 2048 --conc 4 --out "$R/multiprompt.jsonl" 2>&1 | tee -a "$R/audit.log"
}

EV=(); for kv in $BASE_ENV; do EV+=(-e "$kv"); done
log "=== gate 0: selectors, CPU tests, int8 mixer kernel vs reference on both cards ==="
sudo docker run --rm --name r499-test --gpus all --ipc=host -v "$R":/results "${EV[@]}" --entrypoint bash "$IMG" -lc '
set -e
python3 -c "from exllamav3.ext import exllamav3_ext as e; assert hasattr(e, \"gr_mix_v2_int8\"); print(\"binding ok\")"
# test_round4_cpu.py reads the codex worktree layout (out/…, ref/served-src) and cannot run in the image; it passes in
# a local worktree (2026-09-18 21:10 CEST, rc 0). First box try aborted on it (FileNotFoundError).
python3 /opt/decode-kernels-r4/mixer_int8_parity.py --sites 96 --rows 1 --seed 1234 --json /results/mixer-int8-synthetic-96.json | tail -3
for g in 0 1; do python3 /opt/decode-kernels-r4/tests/test_gr_mix_v2_int8.py --device cuda:$g --warmup 50 --iterations 500 --json /results/mixer-gpu$g.json | tail -4; done
' > "$R/gate0.log" 2>&1; rc=$?
tail -14 "$R/gate0.log" | cut -c1-220 | tee -a "$R/audit.log"
[ $rc = 0 ] || { log "GATE 0 FAIL (rc $rc)"; finish ABORTED; exit 3; }
fallback(){ local t=$1
  python3 - "$API" "$MODEL" "$R/fallback-$t.json" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,hashlib,urllib.request
api,model,out=sys.argv[1:4]; res={}
cases={"presence":{"temperature":0,"presence_penalty":0.5,"max_tokens":200,"min_tokens":200},
       "rep":{"temperature":0,"repetition_penalty":1.1,"max_tokens":200,"min_tokens":200},
       "stop":{"temperature":0,"max_tokens":400,"stop":["\n\n\n","def test_"]}}
for name,extra in cases.items():
    body={"model":model,"messages":[{"role":"user","content":"Write a Python function that merges overlapping intervals, then tests for it."}],**extra}
    d=json.loads(urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600).read())
    m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or "")
    res[name]=hashlib.sha256(t.encode()).hexdigest()[:16]+f" n={d['usage']['completion_tokens']} fin={d['choices'][0].get('finish_reason')}"
json.dump(res,open(out,"w")); print("fallback", res)
PY
}
OFFSEL="EXL3_DRAFT_PINNED_STAGING=0 EXL3_BATCH_VERIFY=0 EXL3_MTP_DEVICE_DRAFT=0 EXL3_EMBED_GPU=0 EXL3_HC_MIX_V2_INT8=0"
ISEL="EXL3_DRAFT_PINNED_STAGING=1 EXL3_BATCH_VERIFY=1 EXL3_MTP_DEVICE_DRAFT=0 EXL3_EMBED_GPU=0 EXL3_HC_MIX_V2_INT8=0 EXL3_MTP_HEAD_N=65536"
MSEL="EXL3_DRAFT_PINNED_STAGING=0 EXL3_BATCH_VERIFY=0 EXL3_MTP_DEVICE_DRAFT=0 EXL3_EMBED_GPU=0 EXL3_HC_MIX_V2_INT8=1"
log "=== group I: pinned staging + batched verify + pruned draft head (identity-preserving) ==="
for arm in "I-OFF|$OFFSEL" "I-ON|$ISEL" "I-OFF2|$OFFSEL" "I-ON2|$ISEL"; do
  tag=${arm%%|*}; sel=${arm#*|}
  boot "$tag" "$sel" || { finish ABORTED; exit 3; }
  fp "$tag" || { log "[$tag] FINGERPRINT NOT CANONICAL — identity broken"; finish ABORTED; exit 3; }
  fallback "$tag"
  speed "$tag"
done
python3 - "$R" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys
r=sys.argv[1]; f={t:json.load(open(f"{r}/fallback-{t}.json")) for t in ("I-OFF","I-ON","I-OFF2","I-ON2")}
bad=[k for k in f["I-OFF"] if len({f[t][k] for t in f})!=1]
print("fallback identity:", "PASS" if not bad else f"FAIL {bad}")
PY
log "=== group M: int8 V2 mixer weights (numerics change) ==="
for arm in "M-OFF|$OFFSEL" "M-ON|$MSEL" "M-OFF2|$OFFSEL" "M-ON2|$MSEL"; do
  tag=${arm%%|*}; sel=${arm#*|}
  boot "$tag" "$sel" || { finish ABORTED; exit 3; }
  fp "$tag" || log "[$tag] fingerprint differs from canonical (expected for int8; recorded)"
  speed "$tag"
done
finish DONE
