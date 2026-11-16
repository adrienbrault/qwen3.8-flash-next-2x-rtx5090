#!/usr/bin/env bash
# decodekinds-r1: decode speed by content on the served Flash-Next daily, for the public README.
# User 2026-11-16 ("lets do 1. - 4 ok"): (1) code / structured / prose c1 with thinking off on real prompts,
# MiaAI-Lab protocol; (2) vllm bench per category (Spec-Bench + MBPP code), thinking off, natural lengths,
# token-weighted decode, c1 and c4; (3) R828's code-edit probe re-measured on the 2.68 daily; (4) the
# thinking-on fn_bench decode as a separate "reasoning" row (also this session's control vs 2026-10-23).
# Measurement only: the served launcher, image and config, unchanged, booted fresh on :18022 (two passes,
# A and B). Stock power and the +4500 memory offset asserted (published numbers are at stock).
# Lock, drain, client stop and restore follow publish268-chain.sh. Budget ~75 min GPU; hard cap 3 h.
#
# Install + run on flan (see install.sh):
#   sudo systemd-run --unit=decodekinds-r1 --collect -p RuntimeMaxSec=10800 -p TimeoutStopSec=600 \
#     -p KillMode=control-group -p Environment=HOME=$HOME /usr/bin/bash /srv/qwen5090/decodekinds-r1/run.sh
set -euo pipefail
P=$(cd -- "$(dirname -- "$0")" && pwd)
D=/srv/qwen5090
MODEL=qwen3.8-flash-next-exl3-2.68bpw-selfcal
LIVE=$D/launch-flashnext.sh
RESTORE=$D/serve-flashnext-daily.sh
BASE=http://127.0.0.1:18022
API=$BASE/v1
CLEAN_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CLIENT_IMG=vllm/vllm-openai:v0.30.0
CLIENT_NAME=decodekinds-client
export HOME USER
export PATH="$HOME/.local/bin:$CLEAN_PATH" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
R=$D/results/$(date -u +%Y%m%dT%H%M%SZ)-decodekinds-r1
mkdir -p "$R"
log() { printf '%s [decodekinds] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$R/audit.log"; }

# ---- preparation, before the GPU lock ----
cp -a "$P" "$R/packet"
cp -a "$P/dataset" "$R/dataset"   # built by install.sh (prep-dataset.py)
log "dataset $(cat "$P/dataset.prep.json")"
cp "$LIVE" "$R/launcher.sh"
sha256sum "$LIVE" "$RESTORE" "$D/lib/gpu-queue.sh" "$D/lib/gateway-drain.sh" > "$R/host-helpers.sha256"
sudo docker image inspect -f '{{.Id}}' "$CLIENT_IMG" > "$R/client-image-id.txt"

export GPU_QUEUE_NAME=decodekinds-r1
source "$D/lib/gpu-queue.sh"       # registration BEFORE flock: the on-demand daemon stops the miner for it
source "$D/lib/gateway-drain.sh"
LOCK_HELD=0 MUTATED=0 STOPPED= CLEANING=0 PWR=
finish() {
  local rc=$1 cleanup_rc=0 c others
  [ "$CLEANING" = 0 ] || return 0
  CLEANING=1
  trap '' TERM INT HUP
  set +e
  [ -n "$PWR" ] && kill "$PWR" 2>/dev/null
  sudo docker rm -f "$CLIENT_NAME" >/dev/null 2>&1
  if [ "$LOCK_HELD" = 1 ]; then
    others=$(gpu_queue_others)
    if [ "$MUTATED" = 1 ] && sudo docker container inspect flashnext >/dev/null 2>&1; then
      sudo docker rm -f flashnext >> "$R/cleanup.log" 2>&1 || cleanup_rc=1
    fi
    if [ -n "$others" ]; then
      log "RESTORE DEFERRED: GPU queue continues: $others"
    elif [ "$MUTATED" = 1 ]; then
      log 'restoring the Flash-Next daily through serve-flashnext-daily.sh'
      if env -i HOME="$HOME" USER="$USER" PATH="$CLEAN_PATH" GPU_QUEUE_SELF="$$" bash "$RESTORE" > "$R/boot-restore.log" 2>&1; then
        log 'DAILY RESTORED :8022'
      else
        cleanup_rc=1; log 'RESTORE FAILED: inspect boot-restore.log; operator recovery required'
      fi
    fi
  fi
  for c in $STOPPED; do sudo docker start "$c" >> "$R/clients-restore.log" 2>&1 || cleanup_rc=1; done
  rm -f "$GPU_QUEUE_MARK"
  [ "$cleanup_rc" = 0 ] || rc=1
  printf '%s\n' "$rc" > "$R/exit.txt"
  log "DONE rc=$rc; results $R"
  return "$rc"
}
trap 'rc=$?; trap - EXIT; finish "$rc"; exit $?' EXIT
trap 'exit 143' TERM HUP
trap 'exit 130' INT
exec 9>"$D/gpu-exclusive.lock"
if ! flock -n 9; then log "waiting for the GPU lock: $(gpu_queue_others)"; flock 9; fi
LOCK_HELD=1
gateway_drain
gateway_wait_idle 900 >> "$R/audit.log" 2>&1
for c in hermes hermes-webui owui-proxy; do
  if [ "$(sudo docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null || true)" = true ]; then
    STOPPED="$STOPPED $c"
    sudo docker stop -t 30 "$c" >> "$R/clients-stop.log" 2>&1
  fi
done
log "lock held; direct clients stopped:${STOPPED:- none}"

# Published numbers are at stock power with the +4500 memory offset. The miner leaves a 400 W cap: undo it.
sudo bash "$D/daily-power.sh" stock > "$R/power-set.log" 2>&1
clocks() {
  timeout 30 sudo python3 "$P/clock-readback.py" > "$1"
  python3 - "$1" <<'PY'
import json, sys
cs = json.load(open(sys.argv[1]))
assert [c['index'] for c in cs] == [0, 1], cs
for c, stock in zip(cs, (600, 575)):
    assert c['core'] == 0 and c['memory'] == 4500, f'GPU {c["index"]} clock offsets {c["core"]}/{c["memory"]}'
    assert c['power_w'] == c['default_power_w'] == stock, f'GPU {c["index"]} power {c["power_w"]} != stock {stock}'
PY
}
clocks "$R/clocks-start.json"
nvidia-smi --query-gpu=timestamp,index,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,fan.speed \
  --format=csv,noheader -l 2 > "$R/gpu.csv" 2>&1 & PWR=$!

boot() {
  local tag=$1 dir=$R/$1
  mkdir -p "$dir"
  MUTATED=1
  sudo docker rm -f flashnext >/dev/null 2>&1 || true
  # Independent warm kernel cache per boot (publish268): writes cannot reach the daily's cache.
  mkdir "$dir/tune-cache"
  sudo cp -aL "$D/.exl3cache-tune268/." "$dir/tune-cache/"
  env -i HOME="$HOME" USER="$USER" PATH="$CLEAN_PATH" PORT=18022 R828_PREFLIGHT_DIR="$dir" TUNEDIR="$dir/tune-cache" \
    timeout 900 bash "$LIVE" > "$dir/boot.log" 2>&1
  alive
  sudo docker inspect -f '{{.Config.Image}} {{.Image}}' flashnext > "$dir/image.txt"
  log "BOOT OK $tag $(cat "$dir/image.txt")"
}
alive() {
  [ "$(curl --noproxy '*' -fsS -m 10 "$API/model" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')" = "$MODEL" ]
  [ "$(sudo docker inspect -f '{{.RestartCount}} {{.State.Status}}' flashnext)" = '0 running' ]
}
bench() {  # $1 pass, $2 concurrency
  local tag=$1-c$2 out=$R/$1/bench
  mkdir -p "$out"
  timeout -k 30 2400 sudo docker run --rm --name "$CLIENT_NAME" --network host \
    -e VLLM_NO_USAGE_STATS=1 -e VLLM_DO_NOT_TRACK=1 -e DO_NOT_TRACK=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_DATASETS_OFFLINE=1 -e VLLM_USE_RUST_BENCH=0 -e TABBY_FORCE_LEN=0 \
    -e TABBY_AUDIT_OUT="/out/$tag.requests.jsonl" -e TABBY_SAMPLES_OUT="/out/$tag.samples.tsv" \
    -v "$D/models/$MODEL":/model:ro -v "$R/dataset":/data:ro -v "$P":/probes:ro -v "$out":/out \
    --entrypoint python3 "$CLIENT_IMG" /probes/vllm_bench_tabby.py bench serve \
    --backend openai-chat --base-url "$BASE" --endpoint /v1/chat/completions --model "$MODEL" --tokenizer /model \
    --temperature 0 --request-rate inf --max-concurrency "$2" --num-warmups 0 --num-prompts 560 --no-oversample \
    --dataset-name spec_bench --dataset-path /data/decodekinds.jsonl --spec-bench-output-len 1024 --seed 7310 \
    --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir /out --result-filename "$tag.json" --disable-tqdm \
    > "$out/client-$tag.log" 2>&1
}

for pass in A B; do
  boot "$pass"
  # (4) + control: thinking-on decode exactly as publish268 unit a at c1 (2026-10-23: code 248.7, prose 251.4).
  for kind in code prose; do
    timeout -k 30 900 python3 "$P/vendor/fn_bench.py" --url "$API" --model "$MODEL" --tag "$pass-reasoning-$kind" \
      --kind "$kind" --distinct --tokens 1024 --warmup-runs 1 --conc 1 --runs 3 --out "$R/$pass/reasoning.jsonl" \
      > "$R/$pass/reasoning-$kind.log" 2>&1
  done
  python3 "$P/analyze.py" control "$R/$pass/reasoning.jsonl" | tee -a "$R/audit.log"
  # (1) MiaAI-Lab bench_decode protocol: c1, temperature 0, thinking off, 5 x 400-token cap.
  for shape in coding structured prose; do
    MIA_BASE=$BASE MIA_MODEL=$MODEL timeout -k 30 600 python3 "$P/vendor/mia/mia-prose-screen.py" \
      --shape "$shape" --output "$R/$pass/mia" > "$R/$pass/mia-$shape.log" 2>&1
  done
  log "[$pass] mia done"
  # (2) Spec-Bench + MBPP by category, thinking off, natural lengths (cap 1024, no min_tokens).
  bench "$pass" 1; log "[$pass] bench c1 done"
  bench "$pass" 4; log "[$pass] bench c4 done"
  # (3) R828's code-edit suite, unchanged (12 prompts, thinking off, 2,048 forced tokens, c1, warmup 1, 3 runs).
  timeout -k 30 1800 python3 -B "$P/vendor/r828/gate.py" client --url "$API" --model "$MODEL" --tag "$pass-edit" \
    --suite agent --out "$R/$pass/edit.jsonl" --conc 1 --runs 3 --warmup 1 > "$R/$pass/edit.log" 2>&1
  log "[$pass] edit done"
  alive
  sudo docker logs flashnext > "$R/$pass/container.log" 2>&1
  clocks "$R/$pass/clocks-after.json"
done
python3 "$P/analyze.py" report "$R" > "$R/summary.md" 2>&1
log "summary written"
