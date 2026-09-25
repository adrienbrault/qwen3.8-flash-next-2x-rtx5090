#!/usr/bin/env bash
# R731 (2026-09-25): the publishable rerun of R725/R725b, the standard `vllm bench serve` benchmark (vLLM v0.30.0
#   client + probes/vllm_bench_tabby.py) of the Flash-Next daily (:8022) on ShareGPT V3 and Spec-Bench.
# LINEAGE: R725 (ShareGPT) + R725b (Spec-Bench). The R725 review: the numbers and the shim
#   are faithful, but not publishable. The memory offset is not in the ShareGPT run's artifacts; each level used a
#   different sample (seed+c, disjoint Spec-Bench splits); Spec-Bench c2-c8 had 73-86 requests; there was one boot per
#   dataset; and R728 (window off) has since changed the daily. Every shim correction and every no-upload rule from R725
#   is kept unchanged. This unit changes only the protocol and the records:
#   (1) EVERY boot logs, to boots.tsv and audit.log: the core (GPC) and memory offsets per card, read through pynvml as
#       in r726/r730; the launcher md5; the served image tag + id; the container's EXL3_ env (file + sha); the
#       launcher's env-keys count, policy, cache and slots; the power limits; and the launcher's own `memory clock
#       offset ... readback` line. The offsets are read again after each cell. The expected state is core 0 and memory
#       +4500 on both cards (R726 KEEP-4500, R730 user decision: core OC dropped). Any other reading, or a launcher /
#       image / EXL3 env that differs from the first boot, ends the unit VOID at once: the matrix would mix configs.
#       The unit never writes an offset; the launcher's MEMOC block re-applies +4500 at every boot.
#   (2) The same sample at every level. ShareGPT: one --seed and one --num-prompts SG_N for every level (vLLM's sampler
#       shuffles with the seed and takes the first N that pass its length filter, so the set and the send order do not
#       depend on the concurrency). Spec-Bench: the original question.jsonl with --num-prompts SB_N (default all 480).
#       vLLM's SpecBench shuffles with its fixed seed 0 at every level, so the set and order are again the same. The
#       shim records each cell's sample (the order sent, sha256 of each prompt) to results/<tag>.samples.tsv, and the
#       summary FLAGS any cell whose manifest differs from its dataset's first cell.
#   (3) Spec-Bench SB_N = 480 requests per level (was 240 at c1, 73-86 at c2-c8).
#   (4) Replication and a cold cache: two passes, A and B, over the full matrix, and a FRESH boot of the live launcher
#       for EVERY cell (dataset x level x pass = 16 boots, env -i and NVME_TIER= as in r725/r730; a boot takes ~20-40 s).
#       The prompt cache is empty at every boot. The client's own warm-ups are off (--num-warmups 0, vLLM's default);
#       they would send the run's first prompt `conc` times and so revive it from the cache (R725: one cached request
#       per run). The unit instead warms each boot with `conc` concurrent NON-stream chat requests on short prompts
#       that are not in either dataset: they are outside parse_container's (stream) pattern and the serial range. The
#       summary reads each cell's own container log (container-<tag>.log) and records the cached prompt tokens and the
#       number of requests with any cache hit. Expected: 0; a repeated long ShareGPT prompt is the only legitimate
#       source of a hit.
#   (5) Headline per cell in review §5's README layout: output tok/s, req/s, TTFT p50/p99, TPOT p50/p99, 1000/TPOT p50,
#       E2E p50, tau = gen / (gen - accepted) = tokens per verify step (the definition of review §1 and of FINDINGS
#       R725), and depth = proposed / (gen - accepted) = the MTP depth in effect. Each pass gets its own table, then an
#       A/B table gives the spread per level. The R725 summary's dec-agg / @full / occ columns are removed (review §3:
#       @full was mis-specified; all three are house metrics).
# REQUEST SHAPE: unchanged from r725-std-bench.sh (see its header). backend openai-chat, stream + include_usage,
#   --temperature 0, thinking = template default. ShareGPT output = the reference-reply length; Spec-Bench = 256.
#   min_tokens = max_completion_tokens (the shim), request-rate inf, --max-concurrency c, percentiles 50/90/99. In the
#   results, ITL is per SSE frame (= per verify step); quote TPOT, not ITL.
# ORDER: pass A = ShareGPT c1 c2 c4 c8, then Spec-Bench c1 c2 c4 c8. Pass B = the same order.
# DECISION (pre-registered; the summary prints it; the last audit.log line is `DECISION: ...`):
#   VOID <why>          the instrument failed: a clock offset != core 0 / memory +4500 at a boot or after a cell, the
#                       launcher md5 / image / EXL3 env changed between boots, NVMe tier on, a boot failed, preflight
#                       failed, or a signal. The unit stops at the first one.
#   PUBLISHABLE         both passes complete: every cell has a result, rc 0, 0 failed, completed == num_prompts; no
#                       integrity flag (container lines == requests, client sum(out) == container sum(gen), Spec-Bench
#                       outputs all SB_OUT, the same sample manifest in every cell of a dataset); every cell has
#                       cached <= CACHE_MAX (1 %) of its prompt tokens; and the A/B spread
#                       |A - B| / mean(A, B) <= SPREAD_MAX (3 %) on output tok/s at every (dataset, level).
#   NOT-PUBLISHABLE <why>  otherwise, with the failing checks named.
#   Note: R592 records a 19.5 % boot-to-boot swing in single-stream fn_bench decode (1.5-2.1 % at c2-c8), so
#   c1 is the level most likely to break the 3 % rule. Operator decision before the run (2026-09-25): the rule gates
#   c2-c8 only (SPREAD_EXEMPT=1); c1 is published with its A/B spread beside it. The spread is reported at every level.
# NO UPLOADS: as R725. The client runs with VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1
#   TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1, never --gpus, and all data is local and
#   read-only. --network host is only for reaching 127.0.0.1:8022, and the preflights run with --network none. The unit
#   never pulls an image.
# DEPLOY: probes/vllm_bench_tabby.py (R731: +sample manifest), probes/std_bench_summary.py (R731 rewrite) and
#   probes/parse_container.py go to /srv/qwen5090/probes/. The unit only measures: it does not promote and does not
#   change the daily or any clock. Full raw output (client logs, result JSONs with --save-detailed, the manifests, a
#   boot log + EXL3 env + container log per cell, boots.tsv, runs.tsv, summary.txt/json) lands in $R.
# GPU BUDGET ~1 h 40-45 min. Per pass: ShareGPT SG_N=400 at R725's rates (1.11 / 1.44 / 1.80 / 2.29 req/s) = ~360 + 280 +
#   220 + 175 s ≈ 17 min; Spec-Bench 480 x 256 tokens at 244 / 344 / 448 / 522 tok/s = ~505 + 355 + 275 + 235 s
#   ≈ 23 min; 8 cells x ~55 s (stop + boot ~25-40 s, warm-up, client start ~15-22 s as measured in R725's audit.log gaps, log settle) ≈ 7 min. Two passes
#   ≈ 94 min, plus the preflight (CPU, before the lock) and the final restore (~1 min). RuntimeMaxSec >= 10800 when run
#   alone, >= 43200 in a chain. The trap restores the daily on TERM/INT/HUP.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
UNIT=${UNIT:-$(basename "$0" .sh)}
R=${R:-/srv/qwen5090/results/$(date +%F)-$UNIT}; mkdir -p "$R/results" "$R/probes" "$R/cells"
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
MDIR=/srv/qwen5090/models/$MODEL
DS=/srv/qwen5090/datasets/std-bench
SG=$DS/ShareGPT_V3_unfiltered_cleaned_split.json; SG_SHA=35f0e213
SB=$DS/spec_bench_question.jsonl;                 SB_SHA=4b6d33e7
CLIENT_IMG=${CLIENT_IMG:-vllm/vllm-openai:v0.30.0}
PROBES=/srv/qwen5090/probes
SHIM=$PROBES/vllm_bench_tabby.py
PARSE=$PROBES/parse_container.py
SUMMARY=$PROBES/std_bench_summary.py
PASSES=${PASSES:-"A B"}
DATASETS=${DATASETS:-"sharegpt specbench"}
CONCS=${CONCS:-"1 2 4 8"}
SG_N=${SG_N:-400}
SB_N=${SB_N:-480}
SEED=${SEED:-7310}
SB_OUT=${SB_OUT:-256}
WANT_GPC=${WANT_GPC:-"0 0"}
WANT_MEM=${WANT_MEM:-"4500 4500"}
# R731 review: run 2 ran at 400 W (the 27B launcher's cap, left over after a switch); the unit logged it but never gated.
# Flash-Next publishes at stock: every boot must read power.limit == power.default_limit on every card.
WANT_PWR=${WANT_PWR:-$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | awk '{printf "%s%.0f", (NR>1?" ":""), $1}')}
SPREAD_MAX=${SPREAD_MAX:-3}
SPREAD_EXEMPT=${SPREAD_EXEMPT-1}
CACHE_MAX=${CACHE_MAX:-0.01}
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-1200}
CLIENT_NAME=$UNIT-client
C=$R/cells   # per-cell boot logs, EXL3 env, container logs, client logs
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$UNIT
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
DECISION="VOID the unit ended before the summary"
FINISHED=0
finish(){ [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  sudo docker rm -f "$CLIENT_NAME" >/dev/null 2>&1; finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  [ "${BOOTED:-0}" = 1 ] && log "after restore: core offsets $(gpcoff); memory offsets $(memoff); served $(served_id || echo none)"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true
  log "=== r731 $1 ==="; log "DECISION: $DECISION"; }
void(){ DECISION="VOID $*"; echo "DECISION: $DECISION" >> "$R/summary.txt"; finish VOID; exit 3; }
trap 'log "signal"; DECISION="VOID signal (the run was interrupted)"; finish ABORTED; exit 4' TERM INT HUP
# per-card offsets, NVML index order (0 = ASUS, 1 = HP; r730). The unit only reads them.
gpcoff(){ timeout 30 sudo python3 -c 'import pynvml as N;N.nvmlInit();print(*[N.nvmlDeviceGetGpcClkVfOffset(N.nvmlDeviceGetHandleByIndex(i)) for i in range(N.nvmlDeviceGetCount())])' 2>/dev/null || echo "?"; }
memoff(){ timeout 30 sudo python3 -c 'import pynvml as N;N.nvmlInit();print(*[N.nvmlDeviceGetMemClkVfOffset(N.nvmlDeviceGetHandleByIndex(i)) for i in range(N.nvmlDeviceGetCount())])' 2>/dev/null || echo "?"; }

for f in "$LIVE" "$SHIM" "$PARSE" "$SUMMARY" "$SG" "$SB" "$MDIR/tokenizer.json" "$MDIR/tokenizer_config.json"; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
# snapshot the inputs: a probe edited while the unit is queued must not change what this run measures
cp "$SHIM" "$PARSE" "$SUMMARY" "$R/probes/"
sha(){ sha256sum "$1" | cut -c1-8; }
[ "$(sha "$SG")" = "$SG_SHA" ] || { log "ABORT: ShareGPT sha256 $(sha "$SG") != $SG_SHA"; exit 3; }
[ "$(sha "$SB")" = "$SB_SHA" ] || { log "ABORT: Spec-Bench sha256 $(sha "$SB") != $SB_SHA"; exit 3; }
SB_ROWS=$(grep -c . "$SB")
[ "$SB_N" -le "$SB_ROWS" ] || { log "ABORT: SB_N $SB_N > $SB_ROWS Spec-Bench rows"; exit 3; }
sudo docker image inspect "$CLIENT_IMG" >/dev/null 2>&1 || { log "ABORT: $CLIENT_IMG not on flan (the unit never pulls)"; exit 3; }
CENV=(-e VLLM_NO_USAGE_STATS=1 -e VLLM_DO_NOT_TRACK=1 -e DO_NOT_TRACK=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e HF_DATASETS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e VLLM_USE_RUST_BENCH=0 -e TABBY_FORCE_LEN=1)

# --- preflight (CPU only, no network, before the GPU lock): the shim patches this image (manifest recorder included),
# the tokenizer loads offline, and the full Spec-Bench file loads through the shim's pandas-free reader
sudo docker run --rm --network none "${CENV[@]}" -e TABBY_SAMPLES_OUT=/tmp/preflight.samples.tsv \
  -v "$MDIR":/model:ro -v "$DS":/data:ro -v "$R/probes":/probes:ro --entrypoint python3 "$CLIENT_IMG" -c '
import sys; sys.path.insert(0, "/probes")
import vllm_bench_tabby as s; s.install()
import vllm.benchmarks.serve as S
assert S.get_samples.__name__ == "get_samples_recorded", "manifest recorder not installed"
from vllm.tokenizers import get_tokenizer
t = get_tokenizer("/model")
m = t.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True, tokenize=False)
print("preflight: tokenizer", type(t).__name__, "vocab", len(t), "| template tail", repr(m[-48:]))
from vllm.benchmarks.datasets import datasets as D
d = D.SpecBench(dataset_path="/data/spec_bench_question.jsonl")
print("preflight: specbench rows", len(d.data))' > "$R/preflight.log" 2>&1
grep -q '^shim: patched.*sample manifest ON' "$R/preflight.log" && grep -q '^preflight: tokenizer' "$R/preflight.log" \
  && grep -qE "^preflight: specbench rows $SB_ROWS\$" "$R/preflight.log" \
  || { log "ABORT: preflight failed: $(grep -avE '^\s*$' "$R/preflight.log" | tail -2 | cut -c1-200)"; exit 3; }
grep -aE '^(shim|preflight):' "$R/preflight.log" | sed 's/^/  /' | tee -a "$R/audit.log"

gpu_lock
log "lock held; served at entry: $(served_id || echo none); passes $PASSES; datasets $DATASETS; concs $CONCS; SG_N $SG_N seed $SEED; SB_N $SB_N out $SB_OUT; offsets now core $(gpcoff) / memory $(memoff) (want $WANT_GPC / $WANT_MEM)"
cp "$LIVE" "$R/launcher-at-lock.sh"
printf 'tag\tpass\tdataset\tconc\tt_boot\tlauncher_md5\timage\timage_id\tenv_n\tenv_sha\tkeys_n\twindow\tslots\tcache\tpolicy\tpower_limit_w\tgpc_boot\tmem_boot\tgpc_after\tmem_after\tvram_free\tmem_line\n' > "$R/boots.tsv"
printf 'tag\tpass\tdataset\tconc\tserial_lo\tserial_hi\twarmups\tcatfile\trc\tcontainer\tn_req\tt0\tt1\n' > "$R/runs.tsv"
REF_MD5= REF_IMG= REF_ENV=
# B_* = the current boot's record; written to boots.tsv once the cell is over (with the after-cell offsets)
bootrow(){ printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$B_TAG" "$B_PASS" "$B_DS" "$B_CONC" "$B_T" "$B_MD5" "$B_IMG" "$B_IMGID" "$B_ENVN" "$B_ENVSHA" "$B_KEYS" "$B_WIN" \
  "$B_SLOTS" "$B_CACHE" "$B_POLICY" "$B_PWR" "$B_GPC" "$B_MEM" "${1:--}" "${2:--}" "$B_FREE" "$B_MEMLINE" >> "$R/boots.tsv"; }

boot(){ local tag=$1 bl
  B_TAG=$tag B_PASS=$2 B_DS=$3 B_CONC=$4 B_T=$(date -Is); bl=$C/boot-$tag.log
  served_stop; wait_unserved 45
  B_MD5=$(md5sum "$LIVE" | cut -c1-32)
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= bash "$LIVE" > "$bl" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { sudo docker logs flashnext > "$C/container-$tag.log" 2>&1
      void "[$tag] NO BOOT: $(grep -aE 'Insufficient VRAM|out of memory|Error|ABORT|NO BOOT' "$bl" | tail -1 | cut -c1-160)"; }
  sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$C/env-$tag.txt"
  B_IMG=$(sudo docker inspect -f '{{.Config.Image}}' flashnext 2>/dev/null); B_IMGID=$(sudo docker inspect -f '{{.Image}}' flashnext 2>/dev/null | cut -c1-19)
  B_ENVN=$(wc -l < "$C/env-$tag.txt"); B_ENVSHA=$(sha256sum "$C/env-$tag.txt" | cut -c1-12)
  B_KEYS=$(grep -aoE 'env keys \([0-9]+\)' "$bl" | tail -1 | grep -oE '[0-9]+'); B_WIN=$(grep -c '^EXL3_MTP_KV_WINDOW=' "$C/env-$tag.txt")
  B_SLOTS=$(grep -aoE 'slots [0-9]+' "$bl" | tail -1 | cut -d' ' -f2); B_CACHE=$(grep -aoE 'cache [0-9]+' "$bl" | tail -1 | cut -d' ' -f2)
  B_POLICY=$(grep -aoE "policy '[^']*'" "$bl" | tail -1 | sed "s/^policy //; s/'//g")
  B_PWR=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | awk '{printf "%s%.0f", (NR>1?" ":""), $1}')
  B_FREE=$(grep -aoE 'VRAM free MiB [0-9/]+' "$bl" | tail -1 | awk '{print $4}')
  B_MEMLINE=$(grep -aoE 'memory clock offset: .*' "$bl" | tail -1 | tr '\t' ' ')
  B_GPC=$(gpcoff); B_MEM=$(memoff)
  log "[$tag] UP: image $B_IMG ($B_IMGID); launcher md5 $B_MD5; keys $B_KEYS (container EXL3_ env $B_ENVN, sha $B_ENVSHA, window $B_WIN); slots $B_SLOTS cache $B_CACHE policy '$B_POLICY'; power $B_PWR W; core offsets $B_GPC; memory offsets $B_MEM (launcher: ${B_MEMLINE:-no memory-offset line}); VRAM free $B_FREE"
  if grep -q '^EXL3_NVME_TIER=' "$C/env-$tag.txt"; then bootrow; void "[$tag] NVMe tier on"; fi
  if [ "$B_PWR" != "$WANT_PWR" ]; then bootrow; void "[$tag] power limits at boot '$B_PWR' W, want stock '$WANT_PWR' W"; fi
  if [ "$B_GPC" != "$WANT_GPC" ] || [ "$B_MEM" != "$WANT_MEM" ]; then bootrow
    void "[$tag] clock offsets at boot: core '$B_GPC' memory '$B_MEM', want core '$WANT_GPC' memory '$WANT_MEM'"; fi
  if [ -z "$REF_MD5" ]; then REF_MD5=$B_MD5 REF_IMG=$B_IMGID REF_ENV=$B_ENVSHA
  elif [ "$B_MD5/$B_IMGID/$B_ENVSHA" != "$REF_MD5/$REF_IMG/$REF_ENV" ]; then bootrow
    void "[$tag] config changed mid-run: launcher/image/env $B_MD5/$B_IMGID/$B_ENVSHA vs first boot $REF_MD5/$REF_IMG/$REF_ENV"; fi; }

# conc concurrent non-stream chat requests on short prompts that are in neither dataset: they warm the kernels at this
# batch size, never touch a measured prompt's cache pages, and do not match parse_container's (stream) pattern
warm(){ python3 - "$MODEL" "$2" "$1" <<'EOF'
import json, sys, threading, urllib.request
model, conc, tag = sys.argv[1], max(int(sys.argv[2]), 1), sys.argv[3]
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
got = []
def one(i):
    body = {"model": model, "stream": False, "temperature": 0, "max_tokens": 64, "min_tokens": 64,
            "messages": [{"role": "user", "content": f"Warm-up {tag} slot {i}: describe a lighthouse at dusk in three sentences."}]}
    try:
        rq = urllib.request.Request("http://127.0.0.1:8022/v1/chat/completions", json.dumps(body).encode(),
                                    {"Content-Type": "application/json"})
        got.append((json.load(op.open(rq, timeout=180)).get("usage") or {}).get("completion_tokens"))
    except Exception as e:
        print(f"warm-up error: {e.__class__.__name__}: {e}"[:200])
ts = [threading.Thread(target=one, args=(i,)) for i in range(conc)]
[t.start() for t in ts]; [t.join() for t in ts]
print(f"{len(got)}/{conc} ok, completion tokens {got}")
EOF
}

# parsed (stream) request lines in this cell's container log: "<count with serial > lo> <max serial>"
lines(){ sudo docker logs flashnext > "$C/container-$1.log" 2>&1
  python3 "$R/probes/parse_container.py" "$C/container-$1.log" | python3 -c 'import sys, json
lo = int(sys.argv[1]); s = [json.loads(l)["serial"] for l in sys.stdin if l.strip()]
print(sum(v > lo for v in s), max(s or [0]))' "$2"; }

first=1
cell(){ local ps=$1 ds=$2 conc=$3 n=$4 rc lo hi k cnt t0 t1 tag=$1-$2-c$3; shift 4
  boot "$tag" "$ps" "$ds" "$conc"
  log "[$tag] warm-up: $(warm "$tag" "$conc" 2>&1 | tail -1)"
  read -r _ lo < <(lines "$tag" 0); lo=${lo:-0}
  t0=$(date -Is)
  timeout -k 30 "$CLIENT_TIMEOUT" sudo docker run --rm --name "$CLIENT_NAME" --network host "${CENV[@]}" \
    -e TABBY_SAMPLES_OUT="/out/$tag.samples.tsv" \
    -v "$MDIR":/model:ro -v "$DS":/data:ro -v "$R/probes":/probes:ro -v "$R/results":/out \
    --entrypoint python3 "$CLIENT_IMG" /probes/vllm_bench_tabby.py bench serve \
    --backend openai-chat --base-url http://127.0.0.1:8022 --endpoint /v1/chat/completions \
    --model "$MODEL" --tokenizer /model --temperature 0 \
    --request-rate inf --max-concurrency "$conc" --num-warmups 0 --num-prompts "$n" --no-oversample \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir /out --result-filename "$tag.json" --disable-tqdm \
    --metadata "unit=$UNIT" "tag=$tag" "pass=$ps" "dataset=$ds" "conc=$conc" "num_warmups=0" \
      "unit_warmup=${conc}x non-stream out-of-set" "boot=fresh per cell" "thinking=template-default" \
      "length_forcing=min_tokens" "client=vllm-v0.30.0+vllm_bench_tabby" \
    "$@" > "$C/client-$tag.log" 2>&1
  rc=$?; t1=$(date -Is); sudo docker rm -f "$CLIENT_NAME" >/dev/null 2>&1
  # docker's stdout is block-buffered: wait (<= 60 s) until the container log holds a line for every request
  for k in $(seq 20); do read -r cnt hi < <(lines "$tag" "$lo"); cnt=${cnt:-0} hi=${hi:-$lo}; [ "$cnt" -ge "$n" ] && break; sleep 3; done
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$ps" "$ds" "$conc" "$lo" "$hi" 0 - "$rc" \
    "cells/container-$tag.log" "$n" "$t0" "$t1" >> "$R/runs.tsv"
  local ga ma fr; ga=$(gpcoff); ma=$(memoff); bootrow "$ga" "$ma"
  # run 1 (16:25 UTC) shared the daily with a live agent session: 70 of 470 requests in the first cell were not the
  # client's. Every client request carries min_tokens (the shim forces it) and the warm-ups are non-stream with min_tokens,
  # so a header after this cell's first serial without min_tokens is foreign traffic.
  fr=$(python3 - "$C/container-$tag.log" "$lo" <<'PY'
import re, sys
t = open(sys.argv[1], errors="replace").read(); lo = int(sys.argv[2])
h = re.findall(r"INFO:\s+#(\d+) [\w/.-]+ \([\w-]+\): [\d,]+ prompt tokens ·\s+(.*?)(?=\n\S|\Z)", t, re.S)  # any endpoint, stream or not (R731 review)
print(sum(1 for n, rest in h if int(n) > lo and "min_tokens" not in rest))
PY
)
  printf '%s\t%s\n' "$tag" "${fr:-?}" >> "$R/foreign.tsv"
  [ "${fr:-0}" = 0 ] || log "[$tag] WARN: ${fr} foreign (non-benchmark) requests hit the daily during this cell"
  log "[$tag] rc $rc; $(grep -c '^shim: patched' "$C/client-$tag.log") shim line; $(grep -aoE '^shim: [0-9]+ samples' "$C/client-$tag.log" | cut -d' ' -f2) sampled; server lines $cnt/$n (serials $lo..$hi); $(grep -aE '^(Successful requests|Failed requests|Benchmark duration|Output token throughput|Median TPOT)' "$C/client-$tag.log" | sed -E 's/ {2,}/ /g' | paste -sd';' -); OOM $(grep -acE 'OutOfMemoryError|out of memory' "$C/container-$tag.log") TORCH_CHECK $(grep -acE 'TORCH_CHECK|c10::Error' "$C/container-$tag.log") tracebacks $(grep -ac Traceback "$C/container-$tag.log") unsupported $(grep -aci 'unsupported' "$C/container-$tag.log"); offsets after core $ga memory $ma"
  # the shim's main() path first runs here: a client that produced nothing must not burn fifteen more boots
  if [ $first = 1 ]; then first=0
    [ -s "$R/results/$tag.json" ] && [ -s "$R/results/$tag.samples.tsv" ] && grep -q '^shim: patched' "$C/client-$tag.log" \
      || void "first client run produced no result/manifest: $(grep -avE '^\s*$' "$C/client-$tag.log" | tail -2 | cut -c1-200)"; fi
  [ "$ga" = "$WANT_GPC" ] && [ "$ma" = "$WANT_MEM" ] || void "[$tag] clock offsets after the cell: core '$ga' memory '$ma'"; }

for ps in $PASSES; do
  for ds in $DATASETS; do
    for c in $CONCS; do
      case $ds in
        sharegpt)  cell "$ps" sharegpt "$c" "$SG_N" --dataset-name sharegpt --dataset-path "/data/$(basename "$SG")" --seed "$SEED" ;;
        specbench) cell "$ps" specbench "$c" "$SB_N" --dataset-name spec_bench --dataset-path "/data/$(basename "$SB")" \
                     --spec-bench-output-len "$SB_OUT" --seed "$SEED" ;;
        *) void "unknown dataset $ds" ;;
      esac
    done
  done
done

python3 "$R/probes/std_bench_summary.py" --runs "$R/runs.tsv" --results "$R/results" --boots "$R/boots.tsv" \
  --sb-data "$SB" --sb-out "$SB_OUT" --want-gpc "$WANT_GPC" --want-mem "$WANT_MEM" --cache-max "$CACHE_MAX" \
  --spread-max "$SPREAD_MAX" --spread-exempt "$SPREAD_EXEMPT" --decide "$PASSES" --json "$R/summary.json" > "$R/summary.txt" 2>&1
sed 's/^/  /' "$R/summary.txt" | tee -a "$R/audit.log"
d=$(grep -a '^DECISION: ' "$R/summary.txt" | tail -1 | sed 's/^DECISION: //')
DECISION=${d:-"VOID the summary printed no decision: $(tail -1 "$R/summary.txt" | cut -c1-200)"}
FOREIGN=$(awk -F'\t' '$2 != "0" {printf "%s%s=%s", (n++ ? ", " : ""), $1, $2}' "$R/foreign.tsv" 2>/dev/null)
if [ -n "$FOREIGN" ]; then log "foreign traffic: $FOREIGN"
  case "$DECISION" in PUBLISHABLE*) DECISION="NOT-PUBLISHABLE foreign (non-benchmark) requests in cells: $FOREIGN";; esac; fi
finish DONE
