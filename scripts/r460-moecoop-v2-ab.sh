#!/usr/bin/env bash
# R460 — codex MoE coop decode-kernel V2 (flan/patches/exllamav3/prompts/moe-coop-decode-kernel.md; overlay
# flan/patches/exllamav3/moecoop/, image …-mtpfix2-moecoopv2 = served daily + exl3_moe_coop_v2_kernel.cuh, rebuilt extension).
# Codex's own scenario: MoE kernel time −8…−18 % (rot untouched), i.e. ~2–4 % of all-kernel time per step; mode 1 is meant to be
# bit-identical to V1 (same geometry, K partitions, fold cadence, sum expressions; only bounded work loops, batched arrivals and
# compact scratch); mode 2 = V1's served wide-B rule (slots >= 128) reproduced, an "experimental order".
# Phase 1: codex's GPU test inside the image (one real layer's 512 experts, R in {1,2,4,8,16}, V1 vs V2 bitwise + timing), mode 1
# then mode 2, warm and cold. Phase 2: serving A/B on :8022 with the SAME image, arms OFF (V2=0) / ON1 / ON2 / OFF2, per arm:
# c1 greedy fingerprint (256-token prompt, expect 1474eee2f5945248), 30k greedy (4a255910dee2d9c5), fn_bench ladder c1/c4/c8 at
# 2048 forced tokens twice. Phase 3: GSM8K n=200 c8 (R355 parameters) on ON1 if its fingerprints matched. Daily restored at the end.
set -uo pipefail
export HOME=${HOME:?}
R=${RDIR:-/srv/qwen5090/results/2026-09-17-r460-moecoop-v2-ab}; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-3.05bpw
API=http://127.0.0.1:8022/v1
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
IMG=${IMG:-tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap-ppipe-nosync-mtpfix2-moecoopv2}
TEST=${TEST:-/opt/moe-coop-v2/test_moe_coop_v2.py}   # R462: /opt/moe-coop-v3/test_moe_coop_v2.py (round-2 overlay)
CTRL=${CTRL:-OFF}; CAND=${CAND:-ON1}                 # arms compared for the GSM8K decision (R462: ON1 vs ON3)
K="EXL3_HOST_GAP_REWIND=1+EXL3_HC_MIX_V2=1+EXL3_HC_MIX_V2_MIN_R=1+EXL3_LS_PREFILL_PIPELINE=1"   # the daily's four opt-ins
ARMS=${ARMS:-"OFF=$IMG|$K+EXL3_MOE_COOP_V2=0 ON1=$IMG|$K+EXL3_MOE_COOP_V2=1 ON2=$IMG|$K+EXL3_MOE_COOP_V2=2 OFF2=$IMG|$K+EXL3_MOE_COOP_V2=0"}
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
CKPT=/srv/qwen5090/models/$MODEL
BOOTED=0
log(){ echo "$(date -Is) [r460] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
wait_id(){ local want=$1 i st; for i in $(seq 240); do [ "$(served_id)" = "$want" ] && return 0
  st=$(cstatus); case "$st" in Restarting*|Exited*) log "container $st"; sudo docker logs --tail 12 flashnext 2>&1 | grep -aE "Error|error|Traceback|CUDA" | tail -4 | cut -c1-240 | tee -a "$R/audit.log"; return 1;; esac
  sleep 2; done; return 1; }
boot(){ local img=${2%%|*} env=""; [ "$2" != "${2%%|*}" ] && env=${2#*|}; env=${env//+/ }
  BOOTED=1; log "booting arm $1 IMG=$img EXTRA_ENV='$env'"
  IMG="$img" EXTRA_ENV="$env" bash "$LIVE" >> "$R/audit.log" 2>&1 || { log "BOOT FAILED"; return 1; }
  wait_id "$MODEL" || { log "BOOT UNVERIFIED"; return 1; }
  [ "$(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" = "$img" ] || { log "ABORT: container image is not $img"; return 1; }
  for kv in $env; do [ "$(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c "^$kv$")" = 1 ] || { log "ABORT: container env lacks $kv"; return 1; }; done
  log "up: $1 on $img $env"; }
ladder(){ python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1" --kind code --tokens 2048 --conc 1 4 8 --runs 2 \
    --out "$R/records-$1.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$1]/" | tee -a "$R/audit.log"; }
greedy30k(){ python3 - "$API" "$MODEL" "$R/greedy30k-$1.json" "$1" <<'PY2' | tee -a "$R/audit.log"
import json,sys,hashlib,urllib.request,random
api,model,out,tag=sys.argv[1:5]
rng=random.Random(4242); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
body=" ".join(rng.choice(words) for _ in range(23000))
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
fp(){ python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; print(hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16])' "$1" 2>/dev/null || echo none; }
finish(){ sudo docker rm -f r460-gputest >/dev/null 2>&1 || true
  if [ "$BOOTED" = 1 ]; then log "restoring the daily (promoted config)"; env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    wait_id "$MODEL" && log "RESTORED: serving $MODEL on $(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1)" || log "RESTORE UNVERIFIED: $(served_id)"; fi; log "=== R460 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" /srv/qwen5090/probes/fn_bench.py "$LM" "$CKPT/config.json"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
GPU_QUEUE_NAME=r460-moecoop-v2-ab
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id); image $IMG = $(sudo docker image inspect -f '{{.Id}}' "$IMG" | cut -c8-19)"
BOOTED=1; sudo docker stop -t 30 flashnext >/dev/null 2>&1; sleep 3
# ---- phase 1: codex's GPU test inside the image (GPU0), one real layer ----
gputest(){ local tag=$1; shift
  log "gpu test $tag: $*"
  sudo docker run --rm --name r460-gputest --gpus '"device=0"' --ipc=host -v /srv/qwen5090/models:/models:ro -v "$R":/results \
    --entrypoint python3 "$IMG" "$TEST" --json "/results/gputest-$tag.json" "$@" > "$R/gputest-$tag.log" 2>&1
  local rc=$?; log "gpu test $tag exit=$rc"; grep -aE "PASS|FAIL|mismatch|bitwise|max_abs|speed|V1|V2|Error" "$R/gputest-$tag.log" | head -40 | sed "s/^/[gputest-$tag] /" | cut -c1-230 | tee -a "$R/audit.log"
  return $rc; }
# GPUTEST_ARGS: extra flags for every gpu test (R460b: --skip-stage-events, because the harness's per-stage CUDA-graph node
# inspection died with "CUDA Runtime error 1" on the real layer); ONLY_GPUTEST=1 stops after phase 1 (daily restored).
# GPUTEST_SET: "tag:arg,arg,... tag2:..." overrides the default three runs (R462: mode 3 warm/cold with the regression gate)
if [ -n "${GPUTEST_SET:-}" ]; then for spec in $GPUTEST_SET; do gputest "${spec%%:*}" $(echo "${spec#*:}" | tr ',' ' ') ${GPUTEST_ARGS:-}; done
else
gputest mode1 --mode 1 ${GPUTEST_ARGS:-}
gputest mode1-cold --mode 1 --cold ${GPUTEST_ARGS:-}
gputest mode2 --mode 2 ${GPUTEST_ARGS:-}
fi
[ "${ONLY_GPUTEST:-0}" = 1 ] && { finish DONE; exit 0; }
# ---- phase 2: serving A/B ----
for arm in $ARMS; do name=${arm%%=*}; env=${arm#*=}
  boot "$name" "$env" || { log "arm $name skipped (boot failed)"; continue; }
  log "V2 log lines in container: $(sudo docker logs flashnext 2>&1 | grep -aic 'moe.coop.v2\|MOE_COOP_V2')"
  greedy "$name"; greedy30k "$name"; ladder "$name"
done
python3 - "$R" <<'PY' | tee -a "$R/audit.log"
import json,sys,pathlib,hashlib
R=pathlib.Path(sys.argv[1])
def sha(p):
    d=json.load(open(p)); m=d["choices"][0]["message"]; return hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16]
print("fingerprints:", {p.stem.split("-",1)[1]:sha(p) for p in sorted(R.glob("greedy-*.json"))}, "(expect 1474eee2f5945248)")
print("30k fingerprints:", {p.stem.split("-",1)[1]:sha(p) for p in sorted(R.glob("greedy30k-*.json"))}, "(expect 4a255910dee2d9c5)")
for p in sorted(R.glob("records-*.jsonl")):
    rows=[json.loads(l) for l in open(p) if l.strip()]
    out={}
    for c in sorted({r.get("conc") for r in rows if r.get("ok")}):
        g=[r for r in rows if r.get("ok") and r.get("conc")==c]
        aggs=[]
        for run in sorted({r.get("run") for r in g}):
            gr=[r for r in g if r.get("run")==run]; wall=gr[0].get("round_wall_s") or 0
            aggs.append(round(sum(r["completion_tokens"] for r in gr)/wall,1) if wall else None)
        dec=sorted(r["decode_tps"] for r in g if r.get("decode_tps"))
        out[f"c{c}"]={"agg_per_run":aggs,"decode_med":dec[len(dec)//2] if dec else None}
    print(p.stem, json.dumps(out))
PY
# ---- phase 3: GSM8K on ON1 if it is byte-identical ----
if [ "$(fp "$R/greedy-$CAND.json")" = "$(fp "$R/greedy-$CTRL.json")" ] && [ "$(fp "$R/greedy30k-$CAND.json")" = "$(fp "$R/greedy30k-$CTRL.json")" ]; then
  arm=$(for a in $ARMS; do [ "${a%%=*}" = "$CAND" ] && echo "$a"; done | head -1); boot "GSM8K-$CAND" "${arm#*=}" && {
    log "=== GSM8K n=200 (R355 parameters) num_concurrent=8 on $CAND ==="
    "$LM" --model local-chat-completions --model_args "base_url=$API/chat/completions,model=$MODEL,tokenizer=$CKPT,num_concurrent=8,max_retries=1,tokenized_requests=False" \
      --tasks gsm8k --num_fewshot 5 --limit 200 --apply_chat_template --gen_kwargs temperature=0,max_gen_toks=8192 --output_path "$R/ev-gsm8k-$CAND" > "$R/ev-gsm8k-$CAND.log" 2>&1
    log "lm_eval exit=$?"
    python3 - "$R/ev-gsm8k-$CAND" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,pathlib
f=sorted(pathlib.Path(sys.argv[1]).rglob("results_*.json"))
if not f: raise SystemExit("no results file")
d=json.loads(f[0].read_text()); s=d["results"]["gsm8k"]
print(f"GSM8K candidate: strict={s.get('exact_match,strict-match')} flexible={s.get('exact_match,flexible-extract')}  (daily c8 = 0.935, SE ~0.03)")
PY
  }
else log "$CAND is NOT byte-identical to $CTRL: GSM8K skipped (quality of a differing kernel is a separate decision)"; fi
finish DONE
