#!/usr/bin/env bash
# launch-flashnext.sh -- serve Qwen3.8-Flash-Next (EXL3 3.05bpw) on 2x RTX 5090 via TabbyAPI + ExLlamaV3.
#
# This is the "daily" for the Flash-Next track. Every setting below is either a measured choice or the
# engine's default; the reasons are in docs/CONFIG.md and the raw numbers in docs/MEASUREMENTS.md.
#
#   VARIANTS
#     ./launch-flashnext.sh                 # serve on 0.0.0.0:8022 (the Mac reaches it at <host>:8022)
#     DRAFT=1 ./launch-flashnext.sh         # draft depth 1 instead of 3 (better at 4 concurrent requests)
#     PORT=8023 ./launch-flashnext.sh
#     STOP=1 ./launch-flashnext.sh          # stop the server
#
#   MEASURED ON 2026-09-16 (2x RTX 5090, stock power 600/575 W, EXL3 3.05bpw, 8-bit KV)
#     decode, 1 stream      191 t/s        (draft depth 3)
#     decode, 8 streams     299 t/s aggregate, 43 per stream
#     boot to serving       33 s load + <1 s warm kernels (both caches warm)
#     agent TTFT, warm      0.31-0.48 s on a 20k-token conversation, flat as it grows
#
# WHY THE CACHES MATTER. The first inference after a load costs 34-50 s on a cold box -- Triton compiling
# kernels plus exllamav3's own coop autotuner exploring GEMM shapes. Both persist to disk if told where:
#   TRITON_CACHE_DIR        -> Triton's kernel cache (the vendored FLA kernels set cache_results=True)
#   EXLLAMAV3_TUNE_CACHE    -> coop_autotune_v1.bin (exllamav3/exllamav3_ext/quant/coop_autotune.cu:75-77)
# With an empty cache the warmup measured 21.5 s; warm, 0.85 s. That is the difference between a 80 s and a
# 31 s start, and it is why the caches are mounted rather than left inside the container.
#
# WHY THERE IS A SAMPLER PRESET. TabbyAPI has no sampling fallbacks unless `sampling.override_preset` names
# one, and it says so at boot: "Requests that omit them run untruncated: temperature 1.0, top_k 0, top_p 1.0,
# min_p 0". A client that sends no sampler (DSH sends none) then samples this 3.05 bpw checkpoint at raw
# T=1.0 with no truncation and no penalty, and the reasoning spirals: 2026-09-16 a DSH request degenerated
# into multilingual word salad after ~10k tokens and the model emitted an end-of-thinking tag inside the
# debris, so the tag closed the reasoning channel mid-thought and the rest of the spiral arrived as the
# visible answer. The same request with a sampler runs coherent and calls its tool (see R338). The preset
# below supplies FALLBACKS only (force: false), so a client that sends its own sampler keeps it.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

NAME=flashnext
PORT=${PORT:-8022}
# TWO SEPARATE NUMBERS, and conflating them is a mistake:
#   MAXLEN caps ONE request -- set it to the model's native window (max_position_embeddings 262,144).
#   CACHE  is the page pool SHARED across requests -- it should exceed MAXLEN so several long sessions can
#          coexist. It is a load-time allocation, not a VRAM-derived value.
# The first version of this launcher copied 131,072 into both from a TEST config, halving the served window for
# no reason. Measured cost at q8 KV (from the checkpoint geometry: 12 full-attention layers, 12.00 KiB/token KV
# plus 3.75 KiB/token of always-fp16 QSA planes = 15.75 KiB/token):
#     131,072 -> 1.79 GiB      262,144 -> 3.58 GiB      393,216 -> 5.38 GiB
# MEASURED, not inferred: cache_size 393,216 FAILS the boot with
#     "RuntimeError: Insufficient VRAM in split for model and cache"  (R337, 2026-09-16)
# The earlier "~6.08 GiB budget / 444k tokens" figure was derived from FREE VRAM and was wrong -- free VRAM is
# not allocatable VRAM. exllamav3 autosplits the model across the two GPUs and needs the weights plus the whole
# cache to fit in that split, so the ceiling is well below the arithmetic. 262,144 boots; treat it as the cap.
MAXLEN=${MAXLEN:-262144}
CACHE=${CACHE:-262144}
DRAFT=${DRAFT:-3}
IMG=tabbyapi:53da7919-rqcount              # TabbyAPI 53da7919 + ExLlamaV3 v1.5.0 + the R338 requeue token-count fix, built by flan/docker/Dockerfile.tabbyapi
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw
MODEL=qwen3.8-flash-next-exl3-3.05bpw
TUNEDIR=/srv/qwen5090/.exl3cache           # kernel caches (Triton + coop autotune); survives container replacement
CFG=/srv/qwen5090/flashnext-config.yml
SAMP_PRESET=qwen38_thinking
SAMP_DIR=/srv/qwen5090/sampler_overrides   # mounted into the container's cwd-relative sampler_overrides/
LOG=/srv/qwen5090/logs/flashnext-$PORT.log
mkdir -p /srv/qwen5090/logs "$TUNEDIR" "$SAMP_DIR"

log(){ echo "$(date -Is) [flashnext] $*" | tee -a "$LOG"; }

if [ "${STOP:-0}" = 1 ]; then
  sudo docker rm -f "$NAME" >/dev/null 2>&1 && log "stopped" || log "was not running"
  exit 0
fi

[ -d "$CKPT" ] || { log "ABORT: checkpoint missing at $CKPT"; exit 3; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }

# --- sampler fallbacks ------------------------------------------------------------------------------
# Quoted heredoc: this file is pure data, so nothing in it may be expanded. Values are the checkpoint's
# thinking-mode recommendation (temperature 0.6, top_p 0.95, top_k 20) -- the same triple the 27B vLLM daily
# applies through --override-generation-config. `force: false` on every entry: these apply only to requests
# that omit the parameter, so a client that samples deliberately is never overridden.
cat > "$SAMP_DIR/$SAMP_PRESET.yml" <<'YML'
# Fallback sampler for Qwen3.8-Flash-Next thinking mode. Measured 2026-09-16: without it, DSH's requests
# (which send max_tokens only) sample at temperature 1.0 with no truncation and degenerate.
temperature:
  override: 0.6
  force: false
top_k:
  override: 20
  force: false
top_p:
  override: 0.95
  force: false
YML

# --- config -----------------------------------------------------------------------------------------
# CTX = 262,144 tokens (the model's max_position_embeddings). At 256-token pages that is 1024 pages. Sizing note
# for a seeded page pool: a 32k-token prompt costs 131 pages and a 256-token output a couple more, so roughly
# seven such jobs coexist; short agent turns cost ~2 pages each and the page count is not the binding limit.
cat > "$CFG" <<YML
model:
  model_dir: /models
  model_name: $MODEL
  backend: exllamav3
  max_seq_len: $MAXLEN
  cache_size: $CACHE
  cache_mode: 8,8
  # 8 slots. exllamav3 clamps the generator's max_batch_size to cache.num_slots, and TABBY DERIVES 4 FOR A
  # RECURRENT MODEL (128 otherwise). This checkpoint carries GDN recurrent state, so it takes the 4 path and a
  # c8 test would silently measure c4 without this line. (backends/exllamav3/model.py:400)
  max_batch_size: 8
  # qwen4_exp FORBIDS tensor parallel in this engine:
  #   NotImplementedError: Tensor-parallel is not currently implemented for Qwen4ExpForConditionalGeneration
  # Layer split is the only mode, and it SERIALIZES the two cards: measured alternating 100%/0% utilisation,
  # so only one GPU computes at a time. That is the ceiling on aggregate throughput here.
  tensor_parallel: false
  gpu_split: [30, 30]        # a YAML LIST, not "30,30" -- a string fails pydantic with type=list_type
  gpu_split_auto: false      # explicit rather than autosplit: TabbyAPI #405 applies autosplit_reserve to device 0 only
  cpu_moe_offload_layers: 0  # zero offload is the point; offloading experts costs decode rate (see docs/CONFIG.md)
  cpu_moe_split_experts: 0   # set explicitly: a stale nonzero elsewhere would silently change the baseline
  chunk_size: 2048
  output_chunking: true
  # VISION IS AVAILABLE IN THIS CHECKPOINT and TabbyAPI defaults it OFF. The weight index carries 987
  # vision tensors (model.visual.patch_embed.proj.weight, model.visual.blocks.N.attn.{k,v,o}_proj.{mul1,
  # trellis,suh,svh}) plus a vision_config, image_token_id, vision_start/end_token_id, and the checkpoint
  # ships preprocessor_config.json and video_preprocessor_config.json. The weights live inside the main
  # shards rather than a separate vision_*.safetensors, so their absence from a file listing is not evidence
  # that vision is missing. Verified present 2026-09-16.
  vision: true
  # REASONING PARSER. Without this the model's thinking arrives INLINE in \`content\` with its ends
  # delimited by tags, and \`reasoning_content\` is null -- verified 2026-09-16, a reply came back as
  # 'We need to respond to user: ... \n\nMAC REACHES FLAN'. Harnesses render the two fields separately, so
  # leaving this off leaks the reasoning into the visible answer. The default start/end tokens ( and
  # ) match this checkpoint's chat template, so they are not overridden.
  reasoning: true
  # thinking_token_budget is an accepted ALIAS for reasoning_budget_tokens. PROVENANCE CORRECTED 2026-09-16:
  # this cap did NOT come from the vLLM daily, which has no server-side reasoning budget at all. The only
  # 32768 in that stack is CLIENT-side -- scripts/miniswe/qwen38-local.yaml sets \`max_tokens: 32768\` so a
  # runaway thinking loop releases its slot and the harness recovers from finish_reason=length. That is a
  # different mechanism: it ends the request, where this budget forces the end-of-thinking tag INTO the
  # stream mid-thought. Keep the cap (it bounds a spiral), but do not read it as the daily's behaviour.
  # Not the cause of the 2026-09-16 DSH breakage: that request produced ~17k reasoning tokens and the cap is
  # 32768, so it never fired -- the sampler did it (R338).
  reasoning_budget_tokens: 32768
  # TOOL-CALL PARSER. Without this the model's tool calls are NOT parsed into the OpenAI \`tool_calls\` field:
  # they arrive as raw text in \`content\` and the response finishes with \`finish_reason: "stop"\` instead of
  # \`"tool_calls"\`. Verified 2026-09-16: a request carrying a \`tools\` array came back as
  # \`tool_calls: null\`, finish_reason "stop", content "\n\n<tool_call>\n<function=bash>\n<parameter=command>\n
  # ls -la\n</parameter>\n</function>\n</tool_call>". An agent harness that expects structured calls cannot
  # execute that, which is what surfaced as "the bash call failed with a generic transport error".
  # qwen3_coder matches this checkpoint's chat template byte for byte -- the template itself specifies
  # <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>, and the parser in
  # endpoints/OAI/utils/toolcall_formats/qwen3_coder.py consumes exactly that.
  tool_format: qwen3_coder
  # vision_offload stays false: the tower is small enough to sit in VRAM. Set true only if free VRAM gets
  # tight, at the cost of streaming vision weights from host RAM on every image.
draft_model:
  draft_mode: mtp
  # THE SCHEMA FIELD IS \`draft_num_tokens\`. \`num_draft_tokens\` is not a schema field and is silently ignored,
  # which would run the default depth while the config appeared to say otherwise.
  draft_num_tokens: $DRAFT
  # draft_cache_mode accepts only FP16/Q8/Q6/Q4 -- pair syntax like "8,8" is rejected by the draft schema.
  draft_cache_mode: Q8
  dynamic_draft: false       # measured loss: 184 vs 191 t/s at c1, 229 vs 258 at c4
memory:
  sysmem_recurrent_cache: 4096
  sysmem_kv_cache: 0
sampling:
  # Fallbacks for clients that send no sampler at all -- see the header. Without this line TabbyAPI warns at
  # boot and serves every such request at temperature 1.0 with no truncation (R338).
  override_preset: $SAMP_PRESET
YML

# --- memory hygiene ---------------------------------------------------------------------------------
# A killed vLLM leaves a multi-GB /dev/shm/vllm_offload_*.mmap behind. On 2026-09-15 one 16 GB orphan left the box
# with 40 GB of 60 GB and earlyoom SIGKILLed a 32.6 GB ngram_ram load -- no traceback, no dmesg line, and it read
# as an engine bug for a day. Sweep orphans (never live segments) before allocating anything.
for f in /dev/shm/vllm_offload_*.mmap /dev/shm/psm_*; do
  [ -e "$f" ] || continue
  sudo fuser -s "$f" 2>/dev/null || sudo rm -f "$f"
done

# --- start ------------------------------------------------------------------------------------------
# The vLLM 27B daily and this model CANNOT coexist: the 27B daily is TP=2 across both cards and Flash-Next
# needs both cards resident. Whichever is being served owns the box.
if sudo docker ps --format '{{.Names}}' | grep -qx vllm-27b; then
  log "stopping the vLLM 27B daily -- it holds both GPUs and cannot coexist with this model"
  sudo docker rm -f vllm-27b >/dev/null 2>&1
fi
sudo docker rm -f "$NAME" >/dev/null 2>&1
for i in $(seq 24); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done
log "starting on 0.0.0.0:$PORT, draft depth $DRAFT"
sudo docker run -d --name "$NAME" --gpus all --ipc=host --shm-size=16g --restart unless-stopped \
  -v "$TUNEDIR":/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
  -p 0.0.0.0:$PORT:$PORT \
  -v /srv/qwen5090/models:/models:ro -v "$CFG":/app/config.yml:ro \
  -v "$SAMP_DIR/$SAMP_PRESET.yml":/app/sampler_overrides/$SAMP_PRESET.yml:ro \
  --entrypoint python3 "$IMG" main.py --host 0.0.0.0 --port $PORT --disable-auth true >/dev/null 2>&1 \
  || { log "docker run FAILED"; exit 1; }

up=0
for i in $(seq 90); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
  sudo docker ps --format '{{.Names}}' | grep -qx "$NAME" || break
  sleep 5
done
sudo docker logs "$NAME" > "$LOG.docker" 2>&1
[ "$up" = 1 ] || { log "NO BOOT -- tail:"; tail -15 "$LOG.docker" | cut -c1-180 | sed 's/^/  /' | tee -a "$LOG"; exit 1; }

# Warm the kernels OUTSIDE any measurement: the first inference is the expensive one, and a client that
# happens to be first would otherwise pay it. Logged so the cost is visible rather than folklore.
W=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"Warmup.\",\"max_tokens\":16,\"temperature\":0}" \
  -o /dev/null -w "%{time_total}" 2>/dev/null || echo "?")
log "UP on $PORT | warmup ${W}s | VRAM free MiB $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/')"
log "OpenAI-compatible base URL: http://$(hostname -I | awk '{print $1}'):$PORT/v1"
