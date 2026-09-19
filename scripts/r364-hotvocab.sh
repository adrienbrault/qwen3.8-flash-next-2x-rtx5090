#!/usr/bin/env bash
# R364 — MTP hot vocabulary (upstream PR #303, ported): build, generate the map, A/B it.
#
# WHAT IS UNDER TEST. The MTP draft head's forward is on the critical path of every decode step. The served
# configuration reads 207 t/s at c1, 250 aggregate at c4 and 313 at c8 (2026-09-16). #303 restricts the draft head
# to a hot vocabulary subset; its author measured up to 22% faster MTP decoding upstream. The port's own note says
# 22% is not a forecast here — this run replaces that with a measured number.
#
# ARMS, in the order the port's plan prescribes:
#   served      tabbyapi:qsa-cid (the enabled configuration) — the reference this experiment must not degrade
#   patched-off same image with the port — the disabled path, which must be BYTE-IDENTICAL to `served`
#   patched-on  same image with the map (fp16, validation off) — the treatment
#
# A changed *proposal* is expected in the treatment; a changed greedy **target** output is not, and would have to be
# explained before any throughput number is believed.
#
# RUN: sudo systemd-run --unit=r364-hotvocab --collect -p User=adrienbrault -p RuntimeMaxSec=21600 \
#        bash /srv/qwen5090/r364-hotvocab.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r364-hotvocab; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
L=/srv/qwen5090/launch-flashnext-r340.sh   # FROZEN SNAPSHOT for the arms: its defaults are historical by design
LIVE=/srv/qwen5090/launch-flashnext.sh     # the live launcher, used for RESTORES: see the note below
CKPT=/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw
CORPUS=/srv/qwen5090/hotvocab-corpus
MAP=/srv/qwen5090/mtp-hot-blocks.txt
log(){ echo "$(date -Is) [r364] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r364-hotvocab
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/greedy-compare.sh   # greedy_same(): directory-safe identity checks
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R364 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

# --- 1. a representative corpus: this box's own code and prose -------------------------------------
if [ ! -d "$CORPUS" ]; then
  mkdir -p "$CORPUS"
  cat /srv/qwen5090/probes/*.py > "$CORPUS/code.txt" 2>/dev/null
  cat /srv/qwen5090/*.sh >> "$CORPUS/code.txt" 2>/dev/null
  python3 - "$CORPUS/prose.txt" <<'PY'
import json, sys
out = open(sys.argv[1], "w")
n = 0
for line in open("/srv/qwen5090/r156-corpus.jsonl", errors="ignore"):
    try: d = json.loads(line)
    except Exception: continue
    t = d.get("text") or d.get("prompt") or ""
    if t: out.write(t + "\n"); n += 1
    if n >= 4000: break
out.close()
PY
  log "corpus built: $(wc -c < "$CORPUS/code.txt") bytes code, $(wc -c < "$CORPUS/prose.txt") bytes prose"
fi

greedy(){  # <out-prefix>
  python3 /srv/qwen5090/probes/hotvocab-greedy-capture.py --url "$API" --model "$MODEL" --out-dir "$1" \
    >> "$R/audit.log" 2>&1 || log "  capture FAILED for $1"
  local f; f=$(ls "$1"/* 2>/dev/null | head -1)
  if [ -n "$f" ]; then log "  $1: $(wc -c < "$f") bytes, sha $(shasum -a 256 "$f" | cut -c1-16)"; else log "  $1: no capture"; fi
}

probes(){  # <tag>
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-narrow" --kind code \
     --tokens 2048 --conc 1 4 8 --runs 1 --out "$R/records-$1.jsonl" 2>&1 | grep -E "^  c=|FAILED" | tee -a "$R/audit.log"
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$1-deep" --kind code \
     --tokens 1024 --ctx 120000 --conc 4 --runs 1 --out "$R/records-$1.jsonl" 2>&1 | grep -E "^  c=|FAILED" | tee -a "$R/audit.log"
}

# --- 2. arm `served`: the enabled configuration ----------------------------------------------------
log "=== arm served: the launcher's default image, i.e. the enabled configuration ==="
# NO IMG OVERRIDE: the launcher default IS the served configuration, and pinning it here is how this arm
# drifted from what the box actually serves when PR #337 was promoted.
# RESTORES GO THROUGH $LIVE, NOT $L. $L is a frozen snapshot taken for the R340 experiment, so its `IMG`
# default is whatever was promoted on the day it was frozen -- which is how the 2026-09-16 chain ended on
# tabbyapi:qsa-cid after PR #337 had been promoted, while every script logged "restoring the served
# configuration". A restore written against a snapshot restores a historical default, not the served one.
bash "$LIVE" >> "$R/audit.log" 2>&1 || { log "served boot FAILED"; finish ABORTED; exit 1; }
greedy "$R/greedy-served"
probes served

# --- 3. the ported image, plus its map -------------------------------------------------------------
# BUILD ON THE SERVED BASE. The recipe defaults to tabbyapi:qsa-cid, but the box now serves
# tabbyapi:qsa-cid-pr337, and a treatment built on the older base would carry PR #337 as a second difference in a
# comparison whose whole point is "the feature, nothing else".
HOTVOCAB_IMG=tabbyapi:qsa-cid-pr337-hotvocab
log "=== building $HOTVOCAB_IMG (BASE=tabbyapi:qsa-cid-pr337, REBASED recipe) ==="
# THE REBASE, because the first attempt failed for a reason that had nothing to do with the feature: the patch applied
# fine (with a 58-line offset, CID's patch having added lines above the hunks) and then `sha256sum --check` failed on
# generator.py and job.py, because the manifest had been generated against a PRISTINE v1.5.0 tree while the image
# carries v1.5.0 + CID. `FILE: FAILED` was sha256sum's output, not patch's, and the log read like a code failure.
# The rebased recipe verifies the served-image pre-patch hashes, applies the rebased patch, and checks a manifest
# regenerated for this baseline. Verified offline before installing: patch exit 0, zero failed hunks, all seven
# checksums OK.
if ! (cd /srv/qwen5090/hotvocab-rebase && sudo docker build -f out/Dockerfile.hotvocab-rebased -t "$HOTVOCAB_IMG" \
        --build-arg BASE=tabbyapi:qsa-cid-pr337 . >> "$R/build.log" 2>&1); then
  log "BUILD FAILED"; tail -6 "$R/build.log" | cut -c1-170 | tee -a "$R/audit.log"; finish ABORTED; exit 1
fi
log "image built"

log "=== generating the hot-vocab map (4096 groups) from the corpus; no model forward needed ==="
sudo docker run --rm -v "$CKPT":/models/qwen3.8-flash-next-exl3-3.05bpw:ro -v "$CORPUS":/corpus:ro \
  -v /srv/qwen5090:/out --entrypoint python3 "$HOTVOCAB_IMG" \
  /opt/hotvocab/build_mtp_hot_blocks.py -m /models/qwen3.8-flash-next-exl3-3.05bpw -c /corpus \
  -b 4096 -o /out/mtp-hot-blocks.txt >> "$R/audit.log" 2>&1
[ -s "$MAP" ] && log "map: $(wc -c < "$MAP") bytes, $(head -1 "$MAP" | cut -c1-120)" \
              || { log "MAP GENERATION FAILED"; finish ABORTED; exit 1; }

# --- 4. arm `patched-off`: the disabled path must be byte-identical ---------------------------------
log "=== arm patched-off: same image, feature disabled (must match served exactly) ==="
IMG="$HOTVOCAB_IMG" bash "$L" >> "$R/audit.log" 2>&1 || { log "patched-off boot FAILED"; finish ABORTED; exit 1; }
greedy "$R/greedy-patched-off"
probes patched-off

# --- 5. arm `patched-on`: the treatment --------------------------------------------------------------
log "=== arm patched-on: map mounted, fp16, sub-head validation off ==="
IMG="$HOTVOCAB_IMG" HOTVOCAB_MAP="$MAP" bash "$L" >> "$R/audit.log" 2>&1 || { log "patched-on boot FAILED"; finish ABORTED; exit 1; }
sudo docker exec flashnext sh -c 'echo "EXL3_MTP_HOT_BLOCKS=$EXL3_MTP_HOT_BLOCKS dtype=$EXL3_MTP_HOT_EMBED_DTYPE validate=$EXL3_MTP_VALIDATE_SUBHEAD"' >> "$R/audit.log" 2>&1
greedy "$R/greedy-patched-on"
probes patched-on

# Acceptance is in the server's own log line per request; the probe does not record it.
sudo docker logs --since 90m flashnext > "$R/server-log.txt" 2>&1
grep -aoE "draft [0-9]+/[0-9]+ accepted \([0-9]+%\)" "$R/server-log.txt" | tail -12 | tee -a "$R/audit.log"

# --- 6. gates and comparison -------------------------------------------------------------------------
log "=== gates ==="
if greedy_same "$R/greedy-served" "$R/greedy-patched-off"; then
  log "PASS served == patched-off (byte-identical): the port's disabled path changes nothing"
else
  log "FAIL served != patched-off — do not read the treatment column"
fi
if greedy_same "$R/greedy-patched-on" "$R/greedy-served"; then
  log "PASS patched-on == served: greedy target output unchanged with the feature enabled"
else
  log "NOTE patched-on differs from served — the port's plan says a changed proposal is expected but a changed"
  log "     greedy target output needs an explanation before the throughput columns are believed"
fi
log "restoring the enabled configuration (tabbyapi:qsa-cid, no hot vocab)"
# No IMG override: restore to whatever the launcher considers served, which is the point of a restore.
bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE FAILED"
finish DONE
