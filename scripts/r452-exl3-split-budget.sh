#!/usr/bin/env bash
# R452 — the 8-bit pool ceiling under a different split budget (user: "Is that the max?"; KV floor stays 8-bit).
# Served: gpu_split [30, 30], cache 262,144 at 8,8 → VRAM free 947 MiB on GPU0, 2,511 MiB on GPU1. The loader fills GPU0 to
# its budget first and GPU1 takes the remainder (below budget), so GPU1's headroom is unreachable unless GPU0's budget drops.
# Arms: [29, 32] @ 327,680 and [29.5, 32] @ 327,680 (each +65,536 tokens = ~+1 GiB of 8-bit cache spread over the cards).
# Per booted arm: VRAM free, c1 + 30k greedy fingerprints (a cache-size change must NOT alter them), 120k uncached prefill
# wall (chunk temporaries must still fit), c8 decode 2048 tokens. A failed boot is a result. Restores the daily at the end.
set -uo pipefail
export HOME=${HOME:?}
R=/srv/qwen5090/results/2026-09-17-r452-exl3-cache-bits; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
CP=/srv/qwen5090/launch-flashnext-r452-bits.sh
MODEL=qwen3.8-flash-next-exl3-3.05bpw
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw
API=http://127.0.0.1:8022/v1
LM=/srv/qwen5090/venv-lmeval/bin/lm_eval
CANON_SHA=750e1459e177c47e
ARMS=${ARMS:-"29,32@327680 29.5,32@327680"}
BOOTED=0
log(){ echo "$(date -Is) [r452] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
wait_id(){ local want=$1 i st; for i in $(seq 240); do [ "$(served_id)" = "$want" ] && return 0
  st=$(cstatus); case "$st" in Restarting*|Exited*) log "container $st -- last error lines:"; sudo docker logs --tail 40 flashnext 2>&1 | grep -aE "Error|error|Insufficient|memory|Value|Input should" | tail -3 | cut -c1-200 | tee -a "$R/audit.log"; return 1;; esac
  sleep 2; done; return 1; }
boot(){ # $1 split "a,b"  $2 cache tokens
  local a=${1%,*} b=${1#*,}
  sudo sed "s/^  gpu_split: \[30, 30\]/  gpu_split: [$a, $b]/" "$LIVE" | sudo tee "$CP" >/dev/null; sudo chmod +x "$CP"
  sudo grep -qE "^  gpu_split: \[$a, $b\]" "$CP" || { log "ABORT: launcher copy edit failed for split $1"; return 1; }
  BOOTED=1; log "booting gpu_split [$a, $b], cache_size $2"
  CACHE=$2 bash "$CP" >> "$R/audit.log" 2>&1 || { log "BOOT FAILED: split $1 cache $2"; return 1; }
  wait_id "$MODEL" || { log "BOOT UNVERIFIED: split $1 cache $2 ($(served_id))"; return 1; }
  sudo grep -qE "^  gpu_split: \[$a, $b\]" /srv/qwen5090/flashnext-config.yml || { log "ABORT: generated gpu_split is not [$a, $b]"; return 1; }
  sudo grep -qE "^  cache_size: $2$" /srv/qwen5090/flashnext-config.yml || { log "ABORT: generated cache_size is not $2"; return 1; }
  log "up: split [$a, $b] cache $2, VRAM used/free MiB $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tr '\n' ' ')"; }
fingerprint(){ # $1 tag
  curl -sN -m 900 "$API/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":512,\"min_tokens\":512,\"temperature\":0,\"stream\":true,
         \"messages\":[{\"role\":\"user\",\"content\":\"Write a Python LRU cache with type hints. Code only.\"}]}" \
    | python3 -c '
import sys, json
r, c = [], []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data:"): continue
    p = line[5:].strip()
    if p == "[DONE]": break
    try: o = json.loads(p)
    except Exception: continue
    for ch in o.get("choices") or []:
        d = ch.get("delta") or ch.get("message") or {}
        r.append(d.get("reasoning_content") or d.get("reasoning") or "")
        c.append(d.get("content") or "")
sys.stdout.write("R:" + "".join(r) + "\nC:" + "".join(c))' > "$R/greedy-c1-$1.txt"
  local got; got=$(shasum -a 256 "$R/greedy-c1-$1.txt" | cut -c1-16)
  [ "$got" = "$CANON_SHA" ] && log "[$1] c1 greedy fingerprint $got == canonical (8-bit daily)" || log "[$1] c1 greedy fingerprint $got != canonical $CANON_SHA (expected for a bit change; $(wc -c < "$R/greedy-c1-$1.txt") bytes)"; }
arm(){ # $1 tag
  log "=== $1: c1 greedy fingerprint (must equal canonical) ==="; fingerprint "$1"
  log "=== $1: 30k greedy fingerprint (must equal 4a255910dee2d9c5) ==="
  python3 - "$API" "$MODEL" "$R/greedy30k-$1.json" "$1" <<'PY2' 2>&1 | tee -a "$R/audit.log"
import json,sys,hashlib,urllib.request,random
api,model,out,tag=sys.argv[1:5]
rng=random.Random(4242); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
body=" ".join(rng.choice(words) for _ in range(23000))  # ~30k tokens, deterministic (r432 construction)
msg=body+"\n\nSummarize the passage above in one sentence, then write a haiku about it."
req=json.dumps({"model":model,"temperature":0,"max_tokens":128,"min_tokens":128,"messages":[{"role":"user","content":msg}]}).encode()
r=urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=req,headers={"Content-Type":"application/json"}),timeout=900)
d=json.loads(r.read()); open(out,"w").write(json.dumps(d)); m=d["choices"][0]["message"]
t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or "")
print(tag,"30k greedy sha",hashlib.sha256(t.encode()).hexdigest()[:16],"(canonical 4a255910dee2d9c5) usage",d.get("usage"))
PY2
  log "=== $1: 120k uncached prefill wall (fresh nonce, max_tokens 1) ==="
  python3 - "$API" "$MODEL" "$1" <<'PY' 2>&1 | tee -a "$R/audit.log"
import sys,json,time,urllib.request,random
api,model,arm=sys.argv[1:4]; n=92000
random.seed(time.time_ns()%100000); words=["alpha","beta","gamma","delta","kappa","sigma","omega","theta","lambda","zeta"]
body=f"nonce{random.randint(0,10**9)} "+" ".join(random.choice(words) for _ in range(n))
req={"model":model,"max_tokens":1,"temperature":0,"messages":[{"role":"user","content":body}]}
t=time.time(); d=json.load(urllib.request.urlopen(urllib.request.Request(api+"/chat/completions",data=json.dumps(req).encode(),headers={"Content-Type":"application/json"}),timeout=1800))
print(json.dumps({"arm":arm,"tag":"120k","prompt_tokens":d.get("usage",{}).get("prompt_tokens"),"prefill_wall_s":round(time.time()-t,3)}))
PY
  log "=== $1: c8 decode 2048 tokens ==="
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-c8" --kind code --tokens 2048 --conc 8 --runs 1     --out "$R/records-$1-c8.jsonl" 2>&1 | grep -E "^  c=|FAILED|Traceback|Error" | sed "s/^/[$1]/" | tee -a "$R/audit.log"
}
finish(){
  if [ "$BOOTED" = 1 ]; then log "restoring the daily ([30, 30] @ 262,144)"; bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
    wait_id "$MODEL" && log "RESTORED: serving $MODEL on $(sudo docker ps --format '{{.Image}}' -f name=flashnext | head -1) $(sudo grep -E '^  (gpu_split|cache_size):' /srv/qwen5090/flashnext-config.yml | tr '\n' ' ')" || log "RESTORE UNVERIFIED: $(served_id)"; fi
  log "=== R452 $1 ==="; }
trap 'log "SIGTERM"; finish ABORTED; exit 4' TERM
for f in "$LIVE" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
grep -qE '^  gpu_split: \[30, 30\]' "$LIVE" || { log "ABORT: live launcher heredoc gpu_split line not at the expected shape"; exit 3; }
grep -qE '^CACHE=\$\{CACHE:-262144\}$' "$LIVE" || { log "ABORT: live launcher CACHE default not at the expected shape"; exit 3; }
GPU_QUEUE_NAME=r452-exl3-cache-bits
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
log "lock held; served at entry: $(served_id); arms: $ARMS"
for a in $ARMS; do split=${a%@*}; cache=${a#*@}; tag="s${split/,/-}-$cache"
  if boot "$split" "$cache"; then arm "$tag"; else log "RESULT $tag: does not boot"; fi
done
finish DONE
