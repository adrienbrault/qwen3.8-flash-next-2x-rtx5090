#!/usr/bin/env bash
# R498 prep — graft r0b0tlab's 4-bit MTP head onto the served turboderp 3.05bpw checkpoint. NO GPU, NO lock (CPU + disk only,
# inside the daily image for torch/safetensors; the daily keeps serving).
#
# WHY (2026-09-18 public-repo survey, user: "isn't there more things to try"): peonist-ai/halogen 0.6.0 re-quantized the MTP
# head at 8 bits: prose draft acceptance 51 -> 59 %, +4 % prose decode. The served pack's MTP layer is 3-bit (config mtp_bits 3);
# r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw (downloaded for R495b) ships a 4-bit one of the same architecture (same tensor names,
# plus the MTP hyper-connection mixer inside its shard — turboderp ships that in mtp_hyper_connection_mixer_patch.safetensors).
# Verification is exact and the verify shape is fixed by the depth policy, so greedy output must stay canonical; only the
# acceptance (= decode speed) can move. R498 measures it.
#
# LAYOUT of the hybrid (/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw-mtp4): symlinks to every 3.05 file except shard 7
# and the MTP mixer patch; model-00007-of-00007.safetensors rewritten without mtp.* (7,184 tensors kept); mtp-r0b0tlab-4bit.safetensors
# = all 6,203 mtp.* tensors of the 2.50 pack's last shard; config.json mtp_bits 4. exllamav3 globs every *.safetensors in the
# directory (loader/safetensors.py:437-458) and warns on duplicate keys, so the build asserts the key sets are disjoint.
set -euo pipefail
M=/srv/qwen5090/models
A=$M/qwen3.8-flash-next-exl3-3.05bpw
B=$M/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
H=$M/qwen3.8-flash-next-exl3-3.05bpw-mtp4
IMG=tabbyapi:decode-kernels-r2
R=/srv/qwen5090/results/2026-09-18-r498-mtp4-graft; mkdir -p "$R"
log(){ echo "$(date -Is) [r498-build] $*" | tee -a "$R/build.log"; }
[ -e "$H" ] && { log "ABORT: $H exists"; exit 3; }
mkdir -p "$H"
for f in "$A"/*; do b=$(basename "$f")
  case "$b" in model-00007-of-00007.safetensors|mtp_hyper_connection_mixer_patch.safetensors|config.json) ;; *) ln -s "../$(basename "$A")/$b" "$H/$b";; esac   # relative links: the launcher mounts the models tree at /models
done
python3 - "$A/config.json" "$H/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); assert c["quantization_config"]["mtp_bits"]==3
c["quantization_config"]["mtp_bits"]=4; c["quantization_config"]["mtp_source"]="r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw"
json.dump(c,open(sys.argv[2],"w"),indent=2)
PY
log "symlinks + config written; rewriting shard 7 and extracting the 4-bit MTP (container $IMG)"
sudo docker run -i --rm --entrypoint python3 -v "$M":"$M" "$IMG" - "$A" "$B" "$H" <<'PY' 2>&1 | tee -a "$R/build.log"
import sys, json
from safetensors import safe_open
from safetensors.torch import save_file
A, B, H = sys.argv[1:4]
src7 = f"{A}/model-00007-of-00007.safetensors"
with safe_open(src7, "pt") as f:
    meta = f.metadata(); keep = {k: f.get_tensor(k) for k in f.keys() if not k.startswith("mtp.")}
    n_old_mtp = sum(1 for k in f.keys() if k.startswith("mtp."))
save_file(keep, f"{H}/model-00007-of-00007.safetensors", metadata=meta)
print("shard 7: kept", len(keep), "dropped mtp", n_old_mtp)
idx = json.load(open(f"{B}/model.safetensors.index.json"))["weight_map"]
shards = sorted({v for k, v in idx.items() if k.startswith("mtp.")}); assert len(shards) == 1, shards
with safe_open(f"{B}/{shards[0]}", "pt") as f:
    mtp = {k: f.get_tensor(k) for k in f.keys() if k.startswith("mtp.")}; meta_b = f.metadata()
save_file(mtp, f"{H}/mtp-r0b0tlab-4bit.safetensors", metadata=meta_b)
print("mtp file:", len(mtp), "tensors from", shards[0])
# disjointness across every safetensors file of the hybrid
import glob, collections
seen = collections.Counter()
for p in glob.glob(f"{H}/*.safetensors"):
    if p.endswith("ngram_embedding.safetensors"): continue
    with safe_open(p, "pt") as f: seen.update(f.keys())
dup = [k for k, c in seen.items() if c > 1]
assert not dup, dup[:5]
print("tensor keys", sum(seen.values()), "duplicates 0; mtp keys", sum(1 for k in seen if k.startswith("mtp.")))
PY
for f in model-00007-of-00007.safetensors mtp-r0b0tlab-4bit.safetensors; do [ -s "$H/$f" ] || { log "ABORT: $f missing"; exit 3; }; done
log "=== R498 graft built: $H ($(du -sh --apparent-size -L "$H" | cut -f1) resolved) ==="
