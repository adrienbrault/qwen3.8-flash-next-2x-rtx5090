#!/usr/bin/env bash
# R562 — c8 decode profile on the 8-slot config (user 2026-09-19: "profile c8 to know what to improve"). R519 profiled c1 d3 / c4 d3 /
# c6 d1 on the 786,432-era stack; since then: ring, bf16 GDN state, decode kernels r6, 8 slots (R558: c8 674 / 639 t/s, c8 / c6
# 1.24) and R560 (depth 2 at c6 = 18 verify rows falls off a cliff: prose c6 ~327 vs ~520 t/s at depth 1). Live image + live
# EXTRA_ENV, split 30/30, 8,8, chunk 2048, the R465 harness at 1k context, 64 warmup + 8 settle + 64 capture steps, wall then
# profiler mode for: c4 d3 (16 rows, reference), c6 d1 (12), c8 d1 (16, the served c8), c6 d2 (18), c8 d2 (24), c8 d3 (32).
# Output per shape: kernels.txt (per kernel), kernels.json (per device); summary groups kernel time by family and card.
# GPU TIMEBOX 20 min after the lock. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME
R=/srv/qwen5090/results/2026-09-19-r562-profile-c8; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
HARNESS=/srv/qwen5090/probes/r465/profile_decode_events.py
METER=/srv/qwen5090/probes/r465/events_meter.py
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
TUNEDIR=/srv/qwen5090/.exl3cache
API=http://127.0.0.1:8022/v1
BOOTED=0
log(){ echo "$(date -Is) [r562] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
finish(){ sudo docker rm -f r562-probe >/dev/null 2>&1 || true
  if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"
    env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R562 $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
for f in "$LIVE" "$HARNESS" "$METER" /srv/qwen5090/models/$MODEL/config.json; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
ENVS=$(sed -n 's/^EXTRA_ENV=\${EXTRA_ENV:-\(.*\)}$/\1/p' "$LIVE")
[ -n "$ENVS" ] || { log "ABORT: cannot read the daily EXTRA_ENV from $LIVE"; exit 3; }
DENV=(); for kv in $ENVS; do DENV+=(-e "$kv"); done
IMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE"); [ -n "$IMG" ] || { log "ABORT: no live image"; exit 3; }
export GPU_QUEUE_NAME=r562-profile-c8
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
END=$(( $(date +%s) + 1200 ))
log "lock held; timebox ends $(date -Is -d @$END); image $IMG; env: $ENVS"
BOOTED=1
sudo docker stop -t 30 flashnext >/dev/null 2>&1; sudo docker rm -f flashnext >/dev/null 2>&1; sleep 3
RUN="$R/run-$(date +%H%M%S)"; mkdir -p "$RUN"
profile(){ local sub=$1; shift; local left=$(( END - $(date +%s) ))
  [ $left -lt 90 ] && { log "SKIP $sub: timebox ($left s left)"; return 1; }
  log "shape set $sub: $* (timeout ${left}s)"
  timeout $left sudo docker run --rm --name r562-probe --gpus all --ipc=host --shm-size=16g \
    -v "$TUNEDIR":/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -v /srv/qwen5090/models:/models:ro -v "$HARNESS":/probe/profile_decode_events.py:ro -v "$METER":/probe/events_meter.py:ro -v "$RUN":/results "${DENV[@]}" \
    --entrypoint python3 "$IMG" /probe/profile_decode_events.py --model /models/$MODEL --out /results/$sub \
    --cache-quant 8,8 --gpu-split 30,30 --max-chunk-size 2048 --tokens 128 --warmup-steps 64 --settle-steps 8 --capture-steps 64 "$@" > "$RUN/$sub.log" 2>&1
  local rc=$?; sudo docker rm -f r562-probe >/dev/null 2>&1; log "shape set $sub exit $rc"; return $rc; }
for arm in wall profiler; do
  profile c8d1-$arm --contexts 1024 --batch 8 --draft 1 --capture-mode $arm
  profile c8d2-$arm --contexts 1024 --batch 8 --draft 2 --capture-mode $arm
  profile c6d2-$arm --contexts 1024 --batch 6 --draft 2 --capture-mode $arm
  profile c6d1-$arm --contexts 1024 --batch 6 --draft 1 --capture-mode $arm
  profile c4d3-$arm --contexts 1024 --batch 4 --draft 3 --capture-mode $arm
  profile c8d3-$arm --contexts 1024 --batch 8 --draft 3 --capture-mode $arm
done
python3 - "$RUN" <<'PY' | tee "$RUN/summary.txt" | tee -a "$R/audit.log"
import json,sys,pathlib,re,collections
run=pathlib.Path(sys.argv[1])
FAM=[("moe_coop",r"moe_coop"),("routing",r"routing|topk"),("exl3_gemm",r"exl3_m?gemm"),("hc_mixer",r"gr_v2|gr_mix|hc_apply|hc_mix"),
     ("gdn",r"gated_delta|conv1d|gdn_|state_rewind|conv_rewind"),("attention",r"attn|qsa|flash"),("int8_gemv",r"int8"),("cublas_cutlass",r"cutlass|gemv|gemm|sm80|sm90|ampere"),
     ("sampling_argmax",r"argmax|sampl|softmax"),("elementwise_other",r".")]
def fam(n):
    for f,p in FAM:
        if re.search(p,n,re.I): return f
wall={}
for d in sorted(p for p in run.iterdir() if p.is_dir() and p.name.endswith("-wall")):
    for f in d.rglob("kernels.json"):
        j=json.loads(f.read_text()); c=j["capture"]; wall[(d.name[:-5], f.parent.name)]=c["elapsed_ms_per_step"]
for d in sorted(p for p in run.iterdir() if p.is_dir() and p.name.endswith("-profiler")):
    sh=d.name[:-9]
    for f in d.rglob("kernels.json"):
        j=json.loads(f.read_text()); c=j["capture"]; n=c["iterations"]
        tok=sum(c.get("streamed_during_per_job") or [0])/n
        g=collections.defaultdict(float); dev=collections.defaultdict(float); top=[]
        for k in j["kernels"]:
            ms=k["total_cuda_us"]/1000/n; g[fam(k["name"])]+=ms
            for dv,v in (k.get("devices") or {}).items(): dev[dv]+=v["total_cuda_us"]/1000/n
            top.append((ms,k["name"][:90],k["launches"]//n))
        kt=sum(g.values()); w=wall.get((sh, f.parent.name))
        print(f"== {sh} [{f.parent.name}]: wall {w and round(w,2)} ms/step (profiler {c['elapsed_ms_per_step']:.2f}), tokens/step {tok:.2f}, t/s {tok/(w or c['elapsed_ms_per_step'])*1000:.0f}; kernel {kt:.2f} ms/step (cuda:0 {dev.get('0',0):.2f} / cuda:1 {dev.get('1',0):.2f}), non-kernel {c['elapsed_ms_per_step']-kt:.2f}")
        print("   "+" | ".join(f"{k} {v:.2f}" for k,v in sorted(g.items(),key=lambda x:-x[1])))
        for ms,nm,l in sorted(top,reverse=True)[:8]: print(f"     {ms:6.3f} ms x{l:4d}  {nm}")
PY
finish DONE
