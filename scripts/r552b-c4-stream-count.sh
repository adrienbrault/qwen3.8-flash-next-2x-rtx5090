#!/usr/bin/env bash
# R552 — why do 2 of 4 code c4 requests stop at 1,861 tokens with finish_reason "stop" despite min_tokens 2048 (every c4
# code round since at least R538; skews every c4 code figure)? One boot of the live launcher (tier off), then:
#   c4 x 2 rounds of the fn_bench code prompt (identical prompts, as fn_bench sends them) and c4 x 1 round with distinct
#   prompts; c1 x 2 of the same prompt; non-streamed so the full text, finish_reason, stop_str/eos_reason and usage are kept
#   per request; tabby log kept. Reading: which token/string ended the short ones, and whether min_tokens is honoured.
# R552b: R552 (non-streamed) had every request reach max_new_tokens (finish length, 14/14). This replays fn_bench.one
# itself (streamed, include_usage) and records server tokens vs client frames per request. GPU TIMEBOX 10 min. RUN (queued): popped by r515-queue-chain from flashnext-queue.txt.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-19-r552b-c4-stream-count; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r552b] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
finish(){ [ "$BOOTED" = 1 ] && sudo docker logs flashnext > "$R/docker.log" 2>&1
  if [ "$BOOTED" = 1 ] && [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
  elif [ "$BOOTED" = 1 ]; then log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
    "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    for i in $(seq 240); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done; log "daily: $(served_id)"; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R552b $1 ==="; }
trap 'log SIGTERM; finish ABORTED; exit 4' TERM
[ -e /srv/qwen5090/probes/fn_bench.py ] || { log "ABORT: fn_bench missing"; exit 3; }
export GPU_QUEUE_NAME=r552b-c4-stream-count
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
BOOTED=1; sudo docker rm -f flashnext >/dev/null 2>&1; sleep 2
"${CLEAN_ENV[@]}" NVME_TIER= bash "$LIVE" > "$R/boot.log" 2>&1 || { log "NO BOOT"; finish FAILED; exit 3; }
for i in $(seq 150); do [ "$(served_id)" = "$MODEL" ] && break; sleep 2; done
[ "$(served_id)" = "$MODEL" ] || { log "NO BOOT"; finish FAILED; exit 3; }
log "UP: $(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")"
timeout 900 python3 - "$API" "$MODEL" "$R" <<'PY' 2>&1 | tee -a "$R/audit.log"
import json,sys,threading,importlib.util
api,model,R=sys.argv[1:4]
spec=importlib.util.spec_from_file_location("fb","/srv/qwen5090/probes/fn_bench.py"); fb=importlib.util.module_from_spec(spec); sys.modules["fb"]=fb; spec.loader.exec_module(fb)
P=fb.ASK_CODE.strip()+" Begin now."
out=open(R+"/requests.jsonl","a")
# exactly fn_bench's streamed request (one(), include_usage, min_tokens = max_tokens = 2048), c4 x 3 rounds and c1 x 1:
# is the 1,861 a server stop, or client frames counted when the usage chunk is missing?
for tag,conc in (("c4-1",4),("c4-2",4),("c4-3",4),("c1",1)):
    sink=[]; ths=[threading.Thread(target=fb.one,args=(i,api,model,P,2048,True,sink,900)) for i in range(conc)]
    [t.start() for t in ths]; [t.join() for t in ths]
    for r in sorted(sink,key=lambda r:r["i"]):
        r["tag"]=tag; out.write(json.dumps(r)+"\n"); out.flush()
        print(f"[{tag}] req {r['i']}: completion_tokens {r.get('completion_tokens')} server {r.get('server_completion_tokens')} frames {r.get('client_frames')} finish {r.get('finish_reason')} prompt {r.get('prompt_tokens')} {r.get('err','')}")
PY
log "requests: $(wc -l < "$R/requests.jsonl")"
finish DONE
