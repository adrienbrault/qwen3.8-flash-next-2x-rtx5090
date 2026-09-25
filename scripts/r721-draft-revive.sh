#!/usr/bin/env bash
# R721 (2026-09-25): do prompts revived from the prompt cache lose MTP draft acceptance because of the draft-KV window?
# R718's review (opus-decode/review-r718/out/REVIEW.md) found a within-boot cell-order effect: a stream prompt served in
# an earlier cell and revived from the prompt cache in a later one drafts ~2.33 tokens/step instead of ~2.82 at c6-c8
# 4k code (accepted/proposed ~0.65 vs ~0.92), at an identical step rate; fresh-prompt c8 4k = 1,022.6 tok/s aggregate.
# Leading hypothesis: EXL3_MTP_KV_WINDOW's per-slot ring cache (not page-indexed) zero-fills pages whose slot tags were
# lost, so a revived prefix drafts against zeros. Test: the same c4 -> c6 -> c8 4k code sequence (R718 R1's order, one
# salt, so c6 revives streams 0-3 and c8 revives 0-5), then a c8 cell on a never-seen salt (the in-boot fresh control),
# with the window as served (W) and with the window unset (N: the page-indexed draft cache, which revives with the
# prefix). N needs more draft KV (~1.25 KiB/token over the whole pool), so N boots at the first pool that fits from
# POOLS_N; pool size does not enter acceptance. Order W1 N1 N2 W2, NVME_TIER=, per-boot prefix cache is cold at boot.
# METRIC per boot (container per-request lines, parse_container.py): mean accepted/proposed over the recorded requests of
#   c8 (6 of 8 streams revived from c6; R718 predicts ~(6 x 0.65 + 2 x 0.92) / 8 = 0.72) and of c8fresh (0 revived),
#   gap = c8 / c8fresh. Sanity: the c8 warm-up round must show 6 requests with cached tokens > 0 (logged).
# DECISION (pre-registered):
#   WINDOW     both W boots gap <= 0.88 and both N boots gap >= 0.95  -> the window's revive path is the cause; fix it
#   NOT-WINDOW both N boots gap <= 0.88                               -> the cause is elsewhere (lease/tags are innocent)
#   otherwise  MIXED (report the table)
# Also logs decode_tps per cell (fn_bench records) and 0 OOM / tracebacks per boot. GPU ~50 min. Daily restored.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
UNIT=${UNIT:-$(basename "$0" .sh)}
R=${R:-/srv/qwen5090/results/$(date +%F)-$UNIT}; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
API=http://127.0.0.1:8022/v1
BENCH=/srv/qwen5090/probes/fn_bench.py
PARSE=${PARSE:-/srv/qwen5090/probes/parse_container.py}
SALT=${SALT:-424242}
FRESH_SALT=${FRESH_SALT:-515151}
POOLS_N=${POOLS_N:-"950272 917504 884736"}
ORDER=${ORDER:-"W1 N1 N2 W2"}
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=$UNIT
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== r721 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$BENCH" "$PARSE"; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
# the served env, resolved from the live launcher's default line; N = the same minus the window key
WENV=$(sed -nE 's/^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$/\1/p' "$LIVE" | head -1)
echo "$WENV" | grep -q 'EXL3_MTP_KV_WINDOW=' || { log "ABORT: served env has no EXL3_MTP_KV_WINDOW ($WENV)"; exit 3; }
NENV=$(echo "$WENV" | tr ' ' '\n' | grep -v '^EXL3_MTP_KV_WINDOW=' | xargs)
log "W env $(echo $WENV | wc -w) keys; N env $(echo $NENV | wc -w) keys; salt $SALT fresh $FRESH_SALT; N pools $POOLS_N; order $ORDER"
vram_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | xargs; }

gpu_lock
log "lock held; served at entry: $(served_id || echo none)"
REC=$R/records.jsonl
try_boot(){ local tag=$1; shift
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= "$@" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { sudo docker logs flashnext > "$R/container-$tag-noboot.log" 2>&1; return 1; }
  sudo docker exec flashnext env 2>/dev/null | grep -E '^EXL3_' | sort > "$R/env-$tag.txt"
  grep -q '^EXL3_NVME_TIER=' "$R/env-$tag.txt" && { log "[$tag] ABORT: NVMe tier on"; return 2; }
  log "[$tag] booted; $(wc -l < "$R/env-$tag.txt") EXL3 keys; window $(grep -c '^EXL3_MTP_KV_WINDOW=' "$R/env-$tag.txt"); $(grep -aoE "cache [0-9]+" "$R/boot-$tag.log" | tail -1); $(grep -aoE 'VRAM free MiB [0-9/]+' "$R/boot-$tag.log" | tail -1)"; }
cell(){ local tag=$1 conc=$2 salt=$3
  python3 "$BENCH" --url "$API" --model "$MODEL" --tag "$tag" --kind code --unique --distinct --salt $salt --ctx 4000 \
    --tokens 1024 --warmup-runs 1 --runs 2 --conc $conc --timeout 900 --out "$REC" > "$R/bench-$tag.log" 2>&1; }
run_boot(){ local tag=$1
  cell "$tag-c4" 4 $SALT; cell "$tag-c6" 6 $SALT; cell "$tag-c8" 8 $SALT; cell "$tag-c8fresh" 8 $FRESH_SALT
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  python3 "$PARSE" "$R/container-$tag.log" --min-gen 1 > "$R/requests-$tag.jsonl"
  log "[$tag] done; OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$tag.log"); tracebacks $(grep -ac Traceback "$R/container-$tag.log"); free after $(vram_free) MiB; $(wc -l < "$R/requests-$tag.jsonl") request lines"; }
for t in $ORDER; do
  case $t in
    W*) try_boot $t || { log "[$t] NO BOOT"; finish NO-BOOT; exit 3; } ;;
    N*) ok=0; for p in $POOLS_N; do
          try_boot $t "EXTRA_ENV=$NENV" CACHE=$p; rc=$?
          [ $rc = 0 ] && { ok=1; log "[$t] pool $p"; break; }
          [ $rc = 2 ] && { finish ABORTED; exit 3; }
          log "[$t] pool $p did not boot"; done
        [ $ok = 1 ] || { log "[$t] NO BOOT at any pool"; finish NO-BOOT; exit 3; } ;;
  esac
  run_boot $t
done

# per-request lines are in serial order; each fn_bench cell = 1 warm-up round + 2 recorded rounds of `conc` requests.
# Map them back by position: c4 (12), c6 (18), c8 (24), c8fresh (24) after the boot's own probe/warm-up requests.
python3 - "$R" $ORDER <<'EOF' | tee "$R/summary.txt" | tee -a "$R/audit.log"
import json, sys, statistics as st
R, order = sys.argv[1], sys.argv[2:]
def load(tag):
    rs = [json.loads(l) for l in open(f"{R}/requests-{tag}.jsonl")]
    return [r for r in rs if r["gen"] >= 1000]          # the 1,024-token bench requests only
res = {}
for tag in order:
    rs = load(tag)
    if len(rs) < 78:
        print(f"BOOT {tag}: only {len(rs)} bench request lines (expected 78) -> VOID"); res[tag] = None; continue
    rs = rs[-78:]
    cells, i = {}, 0
    for name, c in (("c4", 4), ("c6", 6), ("c8", 8), ("c8fresh", 8)):
        n = 3 * c; blk = rs[i:i + n]; i += n
        cells[name] = [blk[k * c:(k + 1) * c] for k in range(3)]   # rounds: warm-up, run 0, run 1
    acc = lambda r: r["acc"] / r["prop"] if r["prop"] else 0.0
    nseen = sum(1 for r in cells["c8"][0] if r["cached"] > 0)
    rev = [r for rnd in cells["c8"][1:] for r in rnd]
    fresh = [r for rnd in cells["c8fresh"][1:] for r in rnd]
    ms, mf = st.mean(map(acc, rev)), st.mean(map(acc, fresh))
    gap = ms / mf if mf else 0
    tps = lambda rr: st.mean(r["tps"] for rnd in rr[1:] for r in rnd)
    print(f"BOOT {tag}: c8 warm-up cache hits {nseen}/8 (expect 6); acc c8 {ms:.3f} (n={len(rev)}) c8fresh {mf:.3f} "
          f"(n={len(fresh)}) gap {gap:.3f} | per-stream T/s c4 {tps(cells['c4']):.1f} c6 {tps(cells['c6']):.1f} "
          f"c8 {tps(cells['c8']):.1f} c8fresh {tps(cells['c8fresh']):.1f}")
    res[tag] = gap
W = [g for t, g in res.items() if t.startswith("W")]; N = [g for t, g in res.items() if t.startswith("N")]
if None in W + N or len(W) < 2 or len(N) < 2:
    print("DECISION: VOID (a boot is missing)")
elif all(g <= 0.88 for g in W) and all(g >= 0.95 for g in N):
    print(f"DECISION: WINDOW (W gaps {W}, N gaps {N}): the draft window's revive path loses acceptance")
elif all(g <= 0.88 for g in N):
    print(f"DECISION: NOT-WINDOW (N gaps {N}): revived prompts lose acceptance without the window too")
else:
    print(f"DECISION: MIXED (W gaps {W}, N gaps {N})")
EOF
finish DONE
