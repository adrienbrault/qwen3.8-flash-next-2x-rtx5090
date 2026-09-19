#!/usr/bin/env bash
# R495 prep (2026-09-18, user "what about this checkpoint? https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw"):
# fetch r0b0tlab's EXL3 2.50 bpw (ExLlamaV3 v1.5.0, -b 2.50 -mb 4 -vb 6 -hq -ngb 3; config bits 2.52, head 6, mtp 4,
# vision 6). NO GPU, NO lock, download only — safe beside the daily and the queued units.
# WHY: GPU weights 41.5 GiB vs 49 GiB for the served turboderp 3.05bpw_h5_ng5 (head 5, mtp 3); the served 360,448-token
# pool is only ~5.4 GiB of VRAM (15.75 KiB/token), so ~7.5 GiB of freed weights is worth up to ~+500k tokens of pool on
# paper, plus ~17 % fewer routed-expert bytes streamed per decode token and a 4-bit (vs 3-bit) MTP head. n-gram table
# 18.5 GiB at 3 bpw (ours 31 GiB) -> ngram_ram becomes affordable. Card quality (their harness, 3-bit KV, 3090 + CPU
# experts): GSM8K 79/80, HumanEval 40/40, IFEval 34/39, NIAH 262k PASS. Not comparable to ours — R495 reruns our gates at 8,8.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-18-r495-fetch-2p50; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
DEST=/srv/qwen5090/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
REPO=r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw
NEED_GB=75
mkdir -p "$DEST"
avail=$(df --output=avail -BG "$DEST" | tail -1 | tr -dc 0-9)
[ "${avail:-0}" -ge "$NEED_GB" ] || { log "ABORT: ${avail}G free, need ${NEED_GB}G"; exit 3; }
log "=== R495 fetch start: $REPO @ main -> $DEST (64.6 GB), ${avail}G free ==="
hf download "$REPO" --local-dir "$DEST" > "$R/download.log" 2>&1
rc=$?
log "download rc=$rc, on disk $(du -sh --apparent-size "$DEST" 2>/dev/null | cut -f1), shards $(find "$DEST" -name 'model-*.safetensors' | wc -l)/6, ngram $(ls "$DEST"/ngram_embedding.safetensors 2>/dev/null | wc -l)"
python3 -c "import json;q=json.load(open('$DEST/config.json'))['quantization_config'];print(q)" 2>&1 | tee -a "$R/audit.log"
log "=== R495 fetch DONE (rc=$rc) ==="
