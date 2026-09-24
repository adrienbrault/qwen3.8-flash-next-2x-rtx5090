#!/usr/bin/env bash
# R717c (2026-09-25): does rows32's c6-c8 gain survive context? R717's served A/B (+3-5 % at c6-c8) used short prompts;
# R717b's canonical gate c8 cell (~3.1k-token distinct prompts) read +0.9 %, and the review traced it to depth-2
# acceptance falling with context (tokens/step x1.215 vs step cost x1.222). The pool trade (983,040, -1.6 %) is only
# worth asking the user for if the gain holds at the contexts agents run at.
#   A = the daily (stack-r3, 999,424, policy [[4,3],[5,2],[8,1]]); P = rows32 (stack-r3-rows32 + R717's B env,
#   policy [[4,3],[8,2]], CACHE=983040 = the trade the user would decide). ABAB x3 pairs (A P P A A P), NVME_TIER=.
#   Per boot: c6 and c8, code and prose, ctx 4000 / 16000 / 32000, fn_bench --unique --distinct --salt 424242 (a different filler per stream, the same prompts in every arm;
#   the warm-up run prefills, the 2 recorded runs hit the prefix cache),
#   1,024 forced tokens, 1 warm-up run + 2 runs; per-stream decode = median decode_tps over the recorded requests.
# DECISION (pre-registered): per (c, kind, ctx) B/A = mean over the 3 pairs.
#   WORTH ASKING  every c6/c8 cell at ctx 16000 and 32000 >= 1.03 (code and prose), none < 1.00
#   NOT WORTH IT  any c6/c8 cell at ctx 16000/32000 < 1.00, or the 16k/32k mean over cells < 1.015
#   otherwise     MARGINAL (report; the user decides with the table)
# Also: 0 OOM / TORCH_CHECK / tracebacks per boot, UP-line and post-run free per card. GPU ~75 min. Daily restored.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
UNIT=${UNIT:-$(basename "$0" .sh)}
R=${R:-/srv/qwen5090/results/$(date +%F)-$UNIT}; mkdir -p "$R"
R717=${R717:-/srv/qwen5090/results/2026-09-24-r717-rows32-r4}
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
API=http://127.0.0.1:8022/v1
BENCH=/srv/qwen5090/probes/fn_bench.py
BASE=tabbyapi:stack-r3
IMG=tabbyapi:stack-r3-rows32
POLICY_A='[[4, 3], [5, 2], [8, 1]]'
POLICY_B='[[4, 3], [8, 2]]'
POOL_P=${POOL_P:-983040}
CTXS=${CTXS:-"4000 16000 32000"}
ORDER=${ORDER:-"A1 P1 P2 A2 A3 P3"}
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$UNIT
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== r717c $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$R717/audit.log" "$BENCH"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
AENV=$(sed -nE 's/.*\]   A env \([0-9]+ keys\): //p' "$R717/audit.log" | head -1)
BENV=$(sed -nE 's/.*\]   B env \([0-9]+ keys\): //p' "$R717/audit.log" | head -1)
[ -n "$AENV" ] && [ -n "$BENV" ] || { log "ABORT: could not read R717's A/B envs"; exit 3; }
for i in "$BASE" "$IMG"; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: image $i missing"; exit 3; }; done
vram_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | xargs; }
log "A env $(echo $AENV | wc -w) keys; B env $(echo $BENV | wc -w) keys; P pool $POOL_P; ctx $CTXS; order $ORDER"

gpu_lock
log "lock held; served at entry: $(served_id || echo none)"
REC=$R/records.jsonl
boot(){ local tag=$1; shift
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { log "[$tag] NO BOOT"; sudo docker logs flashnext > "$R/container-$tag.log" 2>&1; return 1; }
  sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$R/env-$tag.txt"
  grep -q '^EXL3_NVME_TIER=' "$R/env-$tag.txt" && { log "[$tag] ABORT: NVMe tier on"; return 1; }
  log "[$tag] booted $(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'); $(wc -l < "$R/env-$tag.txt") EXL3 keys; $(grep -aoE "cache [0-9]+" "$R/boot-$tag.log" | tail -1); $(grep -aoE "policy '[^']*'" "$R/boot-$tag.log" | tail -1); $(grep -aoE 'VRAM free MiB [0-9/]+' "$R/boot-$tag.log" | tail -1)"; }
run_boot(){ local tag=$1
  for ctx in $CTXS; do for conc in 6 8; do for kind in code prose; do
    python3 "$BENCH" --url "$API" --model "$MODEL" --tag "$tag-c$conc-$kind-x$ctx" --kind $kind --unique --distinct --salt 424242 --ctx $ctx \
      --tokens 1024 --warmup-runs 1 --runs 2 --conc $conc --out "$REC" > "$R/bench-$tag-c$conc-$kind-x$ctx.log" 2>&1
  done; done; done
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  log "[$tag] done; OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$tag.log"); TORCH_CHECK $(grep -acE 'TORCH_CHECK|c10::Error' "$R/container-$tag.log"); tracebacks $(grep -ac Traceback "$R/container-$tag.log"); free after $(vram_free) MiB; ok $(grep -c "\"tag\": \"$tag-" "$REC" 2>/dev/null) records"; }
for t in $ORDER; do
  case $t in
    A*) boot $t IMG="$BASE" "EXTRA_ENV=$AENV" "DRAFT_POLICY=$POLICY_A" || { finish NO-BOOT; exit 3; } ;;
    P*) boot $t IMG="$IMG" "EXTRA_ENV=$BENV" "DRAFT_POLICY=$POLICY_B" CACHE=$POOL_P || { finish NO-BOOT; exit 3; } ;;
  esac
  run_boot $t
done

python3 - "$REC" "$CTXS" <<'EOF' | tee "$R/summary.txt" | tee -a "$R/audit.log"
import json, sys, statistics as st
rec, ctxs = sys.argv[1], [int(x) for x in sys.argv[2].split()]
dec = {}
for ln in open(rec):
    try: r = json.loads(ln)
    except Exception: continue
    if r.get("ok") and r.get("decode_tps") and r.get("tag"):
        dec.setdefault(r["tag"], []).append(r["decode_tps"])
med = lambda t: st.median(dec[t]) if dec.get(t) else None
pairs = [("A1", "P1"), ("A2", "P2"), ("A3", "P3")]
cells, long_means, verdict_bad, verdict_all = {}, [], False, True
for ctx in ctxs:
    for c in (6, 8):
        for kind in ("code", "prose"):
            vals = [(med(f"{a}-c{c}-{kind}-x{ctx}"), med(f"{b}-c{c}-{kind}-x{ctx}")) for a, b in pairs]
            if any(x is None or y is None for x, y in vals):
                print(f"CELL c{c} {kind} ctx {ctx}: MISSING {vals}"); verdict_all = False; continue
            pr = [y / x for x, y in vals]; m = st.mean(pr)
            print(f"CELL c{c} {kind} ctx {ctx}: A {' '.join(f'{x:.1f}' for x, _ in vals)} | P {' '.join(f'{y:.1f}' for _, y in vals)} | "
                  f"P/A pairs {' '.join(f'{p:.3f}' for p in pr)} mean {m:.3f}")
            if ctx >= 16000:
                long_means.append(m)
                if m < 1.03: verdict_all = False
                if m < 1.00: verdict_bad = True
lm = st.mean(long_means) if long_means else 0.0
if verdict_bad or lm < 1.015:
    print(f"DECISION: NOT WORTH IT (16k/32k mean {lm:.3f}; a cell below 1.00: {verdict_bad})")
elif verdict_all:
    print(f"DECISION: WORTH ASKING (every c6/c8 cell at 16k/32k >= 1.03; mean {lm:.3f})")
else:
    print(f"DECISION: MARGINAL (16k/32k mean {lm:.3f}; the user decides with the table)")
EOF
finish DONE
