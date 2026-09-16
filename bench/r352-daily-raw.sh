#!/usr/bin/env bash
# R352 — the daily's 8x38k admission, captured RAW. Is it the engine or the instrument?
#
# WHY. Twice now the daily answered an 8 x 38k-context admission arm with only one or two requests producing
# text: the records showed a usage block reporting 512 completion tokens, no text deltas, no error, ~16 s each.
# But the daily's own log shows `200 OK` for every one of those requests. Those two statements cannot both be
# about a server that refused the work, so this run captures the literal SSE for a few such requests and counts
# content deltas per request -- the instrument is on trial here, not the engine.
#
# The probe reads `delta.content` and `delta.reasoning_content`. If vLLM puts the text somewhere else, or sends
# the completion as one non-streamed object, the capture says so in plain text.
#
# RUN: sudo systemd-run --unit=r352-daily-raw --collect -p User=adrienbrault -p RuntimeMaxSec=7200 \
#        bash /srv/qwen5090/r352-daily-raw.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r352-daily-raw; mkdir -p "$R"
FN_L=/srv/qwen5090/launch-flashnext-r340.sh
log(){ echo "$(date -Is) [r352] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r352-raw
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R352 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

log "stopping Flash-Next to boot the daily"
STOP=1 bash "$FN_L" >/dev/null 2>&1 || true
FORCE_RESTORE=1 bash /srv/qwen5090/daily-restore-retry.sh >> "$R/daily-boot.log" 2>&1 || { log "DAILY BOOT FAILED"; finish ABORTED; exit 1; }
log "daily up: $(curl -s -m 10 http://127.0.0.1:8020/v1/models | head -c 80)"

python3 - "$R" <<'PY' | tee -a "$R/audit.log"
import json, os, subprocess, sys, threading, time
R = sys.argv[1]
FILL = ("Interval buffer kernel tensor latency scheduler prefill decode cache page slot quantisation attention "
        "recurrent window batch token stream fusion pipeline shard expert router gating norm rope. ")
PROMPT = (FILL * (40000 // 26)) + "\n\nNow write the complete source of a production-quality Python module: " \
         "every class, every method, full type hints, no commentary. Do not stop early."

def one(i):
    body = json.dumps({"model": "qwen3.8-27b", "max_tokens": 512, "min_tokens": 512, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": PROMPT + f"\n[variant {i}]"}]})
    path = f"{R}/raw-{i}.sse"
    # The body goes through a FILE. Passing it as argv raised `OSError: [Errno 7] Argument list too long` on the
    # first attempt, so the first r352 run captured eight empty files and proved nothing about the server.
    body_path = f"{R}/body-{i}.json"
    open(body_path, "w").write(body)
    t0 = time.time()
    proc = subprocess.run(["curl", "-sN", "-m", "900", "http://127.0.0.1:8020/v1/chat/completions",
                           "-H", "Content-Type: application/json", "--data-binary", f"@{body_path}"],
                          stdout=open(path, "wb"), stderr=subprocess.PIPE)
    if proc.returncode != 0 or os.path.getsize(path) == 0:
        print(f"  raw-{i}: curl rc={proc.returncode} wrote {os.path.getsize(path)} bytes "
              f"stderr={proc.stderr.decode()[:160]!r}", flush=True)
    # Count what actually arrived, by field name, from the raw bytes.
    n_content = n_reasoning = n_tool = n_usage = n_data = n_message_frames = 0
    keys = set()
    for raw in open(path, errors="replace"):
        if not raw.startswith("data:"):
            continue
        payload = raw[5:].strip()
        if payload == "[DONE]":
            continue
        n_data += 1
        try:
            c = json.loads(payload)
        except Exception:
            continue
        if c.get("usage"):
            n_usage += 1
        for ch in c.get("choices") or []:
            d = ch.get("delta") or {}
            m = ch.get("message") or {}
            keys.update(d.keys() or m.keys())
            if d.get("content") or m.get("content"): n_content += 1
            if d.get("reasoning_content") or m.get("reasoning_content"): n_reasoning += 1
            if d.get("tool_calls") or m.get("tool_calls"): n_tool += 1
            if m and not d: n_message_frames += 1
    print(f"  raw-{i}: {round(time.time()-t0,1)}s  data_frames={n_data} content_deltas={n_content} "
          f"reasoning_deltas={n_reasoning} tool_deltas={n_tool} usage_frames={n_usage} "
          f"single_message_frames={n_message_frames} keys={sorted(keys)}",
          flush=True)

ths = [threading.Thread(target=one, args=(i,)) for i in range(8)]
for t in ths: t.start()
for t in ths: t.join()
PY

sudo docker logs vllm-27b > "$R/daily-server-log.txt" 2>&1
log "daily log captured: $(wc -l < "$R/daily-server-log.txt") lines"
log "status codes seen: $(grep -ac '200 OK' "$R/daily-server-log.txt") x 200"

log "restoring Flash-Next"
sudo docker rm -f vllm-27b >/dev/null 2>&1 || true
for i in $(seq 24); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done
bash "$FN_L" >> "$R/audit.log" 2>&1 || log "FLASHNEXT BOOT FAILED"
finish DONE
