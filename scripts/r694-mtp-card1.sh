#!/usr/bin/env bash
# R694 (2026-09-24) — Flash-Next MTP draft component on cuda:1 (decode-push track B M4, the "keep" item of pipespec/EV-MODEL §7).
# The layer split runs the target 24/24 but the MTP block (draft chain + catch-up) sits with the target's front on cuda:0, so
# every draft step waits behind card 0 and card 1 idles 74-75 % of the step (R680). Launcher knob DRAFT_GPU_SPLIT="0, 32" emits
# `draft_gpu_split: [0, 32]` (TabbyAPI backends/exllamav3/model.py:285, 780-786 -> Model.load_gen(use_per_device=...)).
# Prediction (estimate, EV-MODEL): about -0.3 ms/step at c1 (3 chain steps x 0.45 -> ~0.30 ms), ~+2 %; less at c4/c8.
# Arms, one session, alternating 3 pairs: OFF = the daily; ON = the same launcher with DRAFT_GPU_SPLIT="0, 32".
# Per boot: free VRAM after boot, fn_greedy (identity vs the first OFF boot; greedy verification makes drafts output-neutral, so
# identity is required), canonical gate (fn_gate.sh RUNS=3), OOM count, free VRAM after.
# DECISION (pre-registered): promote if ON boots in all 3 pairs with 0 OOM lines, both cards >= 150 MiB free after the gate,
# greedy identical, leg B 8/8 x3 and ON/OFF >= 0.98, and leg A ON/OFF mean over the 3 pairs >= 1.015 at c1 OR c4 with no
# leg-A cell mean < 0.99. Otherwise hold and report per-cell ratios.
# GPU ~40 min. Queue-chained; the daily is restored at the end (this unit does not promote).
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-24-r694-mtp-card1; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
GREEDY=/srv/qwen5090/probes/fn_greedy.py
GATE=/srv/qwen5090/probes/fn_gate.sh
log(){ echo "$(date -Is) [r694] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=r694-mtp-card1
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ sudo docker logs flashnext > "$R/container-final.log" 2>&1 || true
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== R694 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$GATE" "$GREEDY"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
grep -q "DRAFT_GPU_SPLIT:+draft_gpu_split" "$LIVE" || { log "ABORT: live launcher lacks the DRAFT_GPU_SPLIT knob"; exit 3; }

gpu_lock
log "lock held; served at entry: $(served_id || echo none); host: $(free -m | awk '/Mem:/{print "avail "$7" shared "$5} /Swap:/{print "swap "$3"/"$2}' | tr '\n' ' ')"
NONCE=$(( $(date +%s) % 100000 ))
oom(){ sudo docker logs flashnext 2>&1 | grep -acE 'OutOfMemoryError|out of memory|graph.cu'; }
arm(){ local tag=$1 pair=$2; shift 2
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { log "[$tag] NO BOOT"; sudo docker logs flashnext > "$R/container-$tag.log" 2>&1; return 1; }
  log "[$tag] booted; free $(vram_free); draft split in config: $(sudo docker logs flashnext 2>&1 | grep -ac 'draft_gpu_split') log lines; $(grep -aoE 'draft_gpu_split: \[[^]]*\]' "$R/boot-$tag.log" | head -1)"
  local gt=$tag; [ "$tag" = A1 ] && gt=ref
  python3 "$GREEDY" --url http://127.0.0.1:8022 --tag "$gt" --out "$R/greedy.jsonl" > "$R/greedy-$tag.log" 2>&1
  [ "$gt" = ref ] || python3 "$GREEDY" --compare --ref ref --out "$R/greedy.jsonl" 2>&1 | grep -aE "^GREEDY $gt vs ref|MISSING $gt " | sed "s/^/  [$tag] /" | tee -a "$R/audit.log"
  RUNS=3 bash "$GATE" "$R/gate-$tag" http://127.0.0.1:8022/v1 "$MODEL" $(( (NONCE + pair * 7919) % 100000 )) > "$R/gate-$tag.out" 2>&1
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  log "[$tag] gate done; rows A $(wc -l < "$R/gate-$tag/bench-A.jsonl" 2>/dev/null) B $(grep -c '"ok": true' "$R/gate-$tag/bench-B.jsonl" 2>/dev/null)/$(wc -l < "$R/gate-$tag/bench-B.jsonl" 2>/dev/null); OOM $(oom); tracebacks $(grep -ac Traceback "$R/container-$tag.log"); free after $(vram_free)"; }
for p in 1 2 3; do
  arm A$p $p
  arm B$p $p DRAFT_GPU_SPLIT="0, 32"
done
# per-cell medians of per-request decode_tps (fn_bench rows), ON/OFF per pair
python3 - "$R" <<'EOF' 2>&1 | tee -a "$R/audit.log"
import json, sys, statistics as st, os
R = sys.argv[1]
def cells(tag, leg):
    p = f"{R}/gate-{tag}/bench-{leg}.jsonl"; out = {}
    if not os.path.exists(p): return out
    for ln in open(p):
        try: d = json.loads(ln)
        except Exception: continue
        if not d.get("ok"): continue
        c = d.get("conc"); v = d.get("decode_tps") or d.get("per_stream_decode_tps")
        if c is None or v is None: continue
        out.setdefault(c, []).append(v)
    return {c: st.median(v) for c, v in out.items()}
for leg in ("A", "B"):
    for p in (1, 2, 3):
        a, b = cells(f"A{p}", leg), cells(f"B{p}", leg)
        for c in sorted(set(a) | set(b)):
            r = (b.get(c, 0) / a[c]) if a.get(c) else float("nan")
            print(f"RATIO leg {leg} pair {p} c{c}: OFF {a.get(c, float('nan')):.1f} ON {b.get(c, float('nan')):.1f} ON/OFF {r:.3f}")
EOF
finish DONE
