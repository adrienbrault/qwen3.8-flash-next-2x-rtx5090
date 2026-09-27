#!/usr/bin/env bash
# RUN VIA r787-chain.sh ONLY: standalone, a stop while queued leaves :8022 down (no queued-signal restore; PRELAUNCH-R787 #6).
# R787a (2026-09-27): the README decode curve (docs/img/decode-scaling.svg and the solid lines of std-bench.svg)
# re-measured on the R785 daily: tabbyapi:rebase-dev-r3, pool 901,120, split [30, 30], 42 EXL3 keys (+ EXL3_GR_MIX_TILED=1),
# TUNEDIR /srv/qwen5090/.exl3cache-rebase-dev-r3, live launcher md5 e3db755f. Copy of r719-decode-curve.sh as run for
# R719b (2026-09-25, results 2026-09-25-r719-decode-curve on flan, published as 2026-09-25-r719b-decode-curve).
# INSTRUMENT (unchanged from R719/R719b): two boots of the live launcher (env -i, NVME_TIER= : tier off), fn_bench --distinct
# (every concurrent request its own suffix), greedy, 1,024 forced tokens, 1 warm-up + 3 recorded rounds per shape,
# c1..c8, code and prose. Tags NEW1-c<n>-<kind> / NEW2-c<n>-<kind> (bench/plot.py decode_rates(..., "NEW") reads them).
# Metrics as R704/R719: per-stream decode (median of per-request (n-1)/(t_last-t_first)), decode aggregate (sum of the
# concurrent requests' decode rates), round-wall aggregate (labelled secondary), TTFT separately.
# CHANGES AGAINST r719-decode-curve.sh (provenance and hygiene only, the measurement is byte-for-byte the same loop):
#   - R-number, results dir ($R787_DATE-r787a-decode-curve; R719b wrote into a dir named r719, renamed on publication).
#   - Every boot must BE the R785 daily at the published regime (r787-common.sh r787_boot_ok): image rebase-dev-r3, 42 keys
#     incl. EXL3_GR_MIX_TILED, pool 901,120, split [30, 30], the rebase-dev-r3 TUNEDIR, tier off, power == stock default
#     limits, core 0, memory +4500 (the launcher applies them; the unit reads them). A miss ends the unit VOID, never a
#     mixed curve. The live launcher md5 is checked before and after the lock. Replaces r719's stale "stack flags" grep.
#   - Clocks / power re-read after each boot's 16 shapes (logged; a drift VOIDs the unit).
#   - A boot that does not come up VOIDs the unit (r719's arm returned 1 and went on, which could publish a one-boot curve).
#   - Direct :8022 clients stopped for the unit (hermes, hermes-webui, owui-proxy; r787-common.sh), Olla drained
#     (gateway_drain + gateway_wait_idle); a report-only foreign-request count per boot from the container log.
#   - launcher-at-lock.sh copied into $R. container-NEW<b>.log kept (the R719 review had no container logs to check).
# GPU ~18 min (R719b: 07:59:38 lock -> 08:16:23 done, two boots) + restore when run alone.
# RUN: normally from r787-chain.sh. Alone:
#   sudo systemd-run --unit=r787a-decode-curve --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=1800 \
#     -p Environment=HOME=$HOME /usr/bin/bash /srv/qwen5090/r787a-decode-curve.sh
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
UNIT=r787a
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
. /srv/qwen5090/lib/serve-ctl.sh
. /srv/qwen5090/r787-common.sh
R=/srv/qwen5090/results/$R787_DATE-r787a-decode-curve
[ -e "$R/audit.log" ] && R=$R-$(date +%H%M)   # a re-run never appends to an earlier run's records.jsonl
mkdir -p "$R"
LIVE=$R787_LIVE
MODEL=$R787_MODEL
API=http://127.0.0.1:8022/v1
SCTL_LOG="$R/audit.log"
for f in "$LIVE" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/lib/gpu-queue.sh /srv/qwen5090/lib/gateway-drain.sh; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
why=$(r787_launcher_ok) || { log "ABORT (before the lock, nothing touched): $why"; exit 3; }
export GPU_QUEUE_NAME=r787a-decode-curve
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/gateway-drain.sh
VERDICT="VOID the unit ended before the analysis"
FINISHED=0
finish(){ [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  [ -n "$(served_id)" ] && sudo docker logs flashnext > "$R/container-final.log" 2>&1
  finish_restore "$LIVE"; r787_unquiesce; log "$R787_QMSG"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true
  log "=== R787a $1 ==="; log "VERDICT: $VERDICT"; }
void(){ VERDICT="VOID $*"; finish VOID; exit 3; }
trap 'log "signal"; VERDICT="VOID signal"; finish ABORTED; exit 4' TERM INT HUP
gpu_lock
gateway_drain
r787_quiesce
log "lock held; served at entry: $(served_id || echo none); $R787_QMSG"
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || log "WARN: Olla still has requests in flight after 900 s"
why=$(r787_launcher_ok) || void "under the lock: $why"
cp "$LIVE" "$R/launcher-at-lock.sh"
arm(){ local tag=$1 L=$2 pv ck cko
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= bash "$L" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { sudo docker logs flashnext > "$R/container-$tag.log" 2>&1; void "[$tag] NO BOOT"; }
  pv=$(r787_boot_ok "$R/boot-$tag.log") || { log "[$tag] $pv"; void "[$tag] not the R785 daily at the published regime: ${pv#*BAD:}"; }
  log "[$tag] booted: $pv; free $(vram_free)"
  for conc in 1 2 3 4 5 6 7 8; do for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-c$conc-$kind" --kind $kind --distinct \
      --tokens 1024 --warmup-runs 1 --conc $conc --runs 3 --out "$R/records.jsonl" > "$R/bench-$tag-c$conc-$kind.log" 2>&1
  done; done
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  ck=$(r787_clocks_ok) && cko=0 || cko=1
  log "[$tag] done; OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$tag.log"); restarts $(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null); foreign (report-only) $(r787_foreign "$R/container-$tag.log"); after: $ck; free after $(vram_free)"
  [ "$cko" = 0 ] || void "[$tag] clocks / power drifted during the arm: $ck"; }
arm NEW1 "$LIVE"; arm NEW2 "$LIVE"
python3 - "$R/records.jsonl" "$R/curve.tsv" <<'PY' 2>&1 | tee "$R/analysis.txt" | tee -a "$R/audit.log"
import json,sys,collections,statistics as st
tok=collections.defaultdict(int); wall={}; dec=collections.defaultdict(list); rounds=collections.defaultdict(list); ttft=collections.defaultdict(list)
for l in open(sys.argv[1]):
    r=json.loads(l)
    if not r.get("ok"): continue
    k=(r["tag"],r["run"]); tok[k]+=r["completion_tokens"] or 0; wall[k]=r["round_wall_s"]
    if r.get("decode_tps"): dec[r["tag"]].append(r["decode_tps"]); rounds[k].append(r["decode_tps"])
    if r.get("ttft_s") is not None: ttft[r["tag"]].append(r["ttft_s"])
agg=collections.defaultdict(list); dagg=collections.defaultdict(list)
for k,v in tok.items(): agg[k[0]].append(v/wall[k])
for k,v in rounds.items(): dagg[k[0]].append(sum(v))
def m(d, conc, kind, f):
    v=[x for b in (1,2) for x in d.get(f"NEW{b}-c{conc}-{kind}",[])]; return f(v) if v else float("nan")
def per_boot(d, conc, kind, f):
    return [f(d[t]) if d.get(t) else float("nan") for t in (f"NEW1-c{conc}-{kind}", f"NEW2-c{conc}-{kind}")]
out=open(sys.argv[2],"w"); out.write("conc\tkind\twallagg\tdecode_stream\tdecode_agg\tttft\n")
for kind in ("code","prose"):
    for c in range(1,9):
        v=[m(d,c,kind,f) for d,f in ((agg,st.mean),(dec,st.median),(dagg,st.mean),(ttft,st.median))]
        b=per_boot(dagg,c,kind,st.mean)
        print(f"{kind} c{c}: decode/stream {v[1]:.1f} | decode aggregate {v[2]:.0f} (boots {b[0]:.0f} / {b[1]:.0f}) | round-wall aggregate {v[0]:.0f} | TTFT {v[3]:.2f}s")
        out.write(f"{c}\t{kind}\t"+"\t".join(f"{x:.2f}" for x in v)+"\n")
PY
VERDICT="DONE $(python3 -c 'import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; print(sum(1 for x in r if x.get("ok")), "/", len(r), "ok records (want 432)")' "$R/records.jsonl" 2>/dev/null)"
finish DONE
