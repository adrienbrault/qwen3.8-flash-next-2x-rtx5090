#!/usr/bin/env bash
# RUN VIA r787-chain.sh ONLY: standalone, a stop while queued leaves :8022 down (no queued-signal restore; PRELAUNCH-R787 #6).
# R787c (2026-09-27): the README prefill figure's "decode at depth" lines (docs/img/prefill.svg, dashed) re-measured on the
# R785 daily (tabbyapi:rebase-dev-r3, pool 901,120, split [30, 30], 42 EXL3 keys incl. EXL3_GR_MIX_TILED=1, TUNEDIR
# /srv/qwen5090/.exl3cache-rebase-dev-r3, live launcher md5 e3db755f). Copy of r554-depth-decode.sh (R554, 2026-09-19).
# INSTRUMENT (unchanged from R554): one boot of the live launcher, tier off; fn_bench, forced 2,048 tokens, greedy, salted
# unique filler (--unique, a distinct salt per invocation), --runs 2 and NO warm-up round (fn_bench's --warmup-runs
# defaults to 0; R554's header said "after fn_bench's warm-up round", which was wrong: run0 is the cold prefill and run1
# decodes on the now-cached prefix, and bench/plot.py averages both runs' decode_tps). c1 code and prose at 0 / ~100k /
# ~200k prompt tokens, then c4 prose at the 133,000 prose target (4 distinct ~100k contexts = ~400k resident).
# Tags c1-<kind>-<ctx> and c4-prose-133000 (bench/plot.py depth_decode keeps c1-* and groups by kind and prompt_tokens).
# ONE PROTOCOL CORRECTION: the code targets. R554 asked 0 / 133,000 / 266,000 for both kinds, meaning ~0 / ~100k / ~200k
#   server-counted tokens at prose's ~0.75 tokens per budget unit. Code filler counts ~1.35 tokens per unit (R554: 133,000 ->
#   179,575 tokens), so code landed at ~180k instead of ~100k, and 266,000 (~359k tokens) exceeded the window: HTTP 400 on
#   both runs, no record. Code now asks 0 / 74,000 / 148,000 (~100k / ~200k at 1.35; ~222k even at 1.5, inside the
#   262,144 window), so both kinds read decode at the depths the header always meant. Prose targets unchanged.
# OTHER CHANGES AGAINST r554-depth-decode.sh:
#   - Built on lib/serve-ctl.sh (served_stop / wait_served_id / finish_restore) instead of private copies; the results dir
#     is no longer hardcoded ($R787_DATE-r787c-depth-decode).
#   - Provenance gate on the boot (r787-common.sh r787_boot_ok) and a clocks re-read at the end; any miss -> VOID.
#   - The summary printed n_ok / tokens_median / aggregate_tps, which are round-summary keys fn_bench prints to stdout and
#     never writes to the per-request JSONL, so every one read None; it now prints completion_tokens and wall_tps.
#   - Direct :8022 clients stopped (hermes, hermes-webui, owui-proxy), Olla drained, report-only foreign count.
# GPU ~6 min (R554: 15:34:55 lock -> ~15:39 end; the extra code point adds a ~7 s ~100k prefill + 2 x 2,048 tokens).
# RUN: normally from r787-chain.sh. Alone:
#   sudo systemd-run --unit=r787c-depth-decode --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=1800 \
#     -p Environment=HOME=$HOME /usr/bin/bash /srv/qwen5090/r787c-depth-decode.sh
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
UNIT=r787c
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
. /srv/qwen5090/lib/serve-ctl.sh
. /srv/qwen5090/r787-common.sh
R=/srv/qwen5090/results/$R787_DATE-r787c-depth-decode
[ -e "$R/audit.log" ] && R=$R-$(date +%H%M)   # a re-run never appends to an earlier run's depth.jsonl
mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=$R787_MODEL
LIVE=$R787_LIVE
CFG=/srv/qwen5090/flashnext-config.yml
FB=/srv/qwen5090/probes/fn_bench.py
SCTL_LOG="$R/audit.log"
for f in "$LIVE" "$FB" /srv/qwen5090/lib/gpu-queue.sh /srv/qwen5090/lib/gateway-drain.sh; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
why=$(r787_launcher_ok) || { log "ABORT (before the lock, nothing touched): $why"; exit 3; }
export GPU_QUEUE_NAME=r787c-depth-decode
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/gateway-drain.sh
VERDICT="VOID the unit ended before the probe finished"
FINISHED=0
finish(){ [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  [ -n "$(served_id)" ] && sudo docker logs flashnext > "$R/docker-final.log" 2>&1
  finish_restore "$LIVE"; r787_unquiesce; log "$R787_QMSG"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true
  log "=== R787c $1 ==="; log "VERDICT: $VERDICT"; }
void(){ VERDICT="VOID $*"; finish VOID; exit 3; }
trap 'log "signal"; VERDICT="VOID signal"; finish ABORTED; exit 4' TERM INT HUP
gpu_lock
gateway_drain
r787_quiesce
log "lock held; live image $(sed -n 's/^DAILY_IMG=\(.*\)$/\1/p' "$LIVE"); served at entry: $(served_id || echo none); $R787_QMSG"
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || log "WARN: Olla still has requests in flight after 900 s"
why=$(r787_launcher_ok) || void "under the lock: $why"
cp "$LIVE" "$R/launcher-at-lock.sh"
SALT=$(( $(date +%s) % 100000 ))
served_stop; wait_unserved 45
env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= bash "$LIVE" > "$R/boot.log" 2>&1 \
  && wait_served_id "$MODEL" 200 8 || { sudo docker logs flashnext > "$R/docker-noboot.log" 2>&1; void "NO BOOT"; }
pv=$(r787_boot_ok "$R/boot.log") || { log "$pv"; void "not the R785 daily at the published regime: ${pv#*BAD:}"; }
log "UP: $pv"
log "UP: $(sudo grep -E '^  (cache_size|chunk_size|max_batch_size):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '); VRAM free $(vram_free)"
n=0
for kind in code prose; do
  case $kind in code) targets="0 74000 148000";; prose) targets="0 133000 266000";; esac
  for c in $targets; do n=$((n+1))
    timeout 900 python3 "$FB" --url "$API" --model "$MODEL" --tag "c1-$kind-$c" --kind $kind --tokens 2048 --conc 1 --runs 2 --ctx $c --unique \
      --salt $(( SALT + n*1009 )) --out "$R/depth.jsonl" >> "$R/fn_bench.log" 2>&1 || log "fn_bench c1 $kind $c rc $?"
  done; done
timeout 1200 python3 "$FB" --url "$API" --model "$MODEL" --tag "c4-prose-133000" --kind prose --tokens 2048 --conc 4 --runs 2 --ctx 133000 --unique \
  --salt $(( SALT + 7777 )) --out "$R/depth.jsonl" >> "$R/fn_bench.log" 2>&1 || log "fn_bench c4 rc $?"
python3 - "$R/depth.jsonl" <<'PY' 2>&1 | tee "$R/summary.txt" | tee -a "$R/audit.log"
import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    print(f"{r.get('tag')} run{r.get('run')}: prompt {r.get('prompt_tokens')} tokens, ok {r.get('ok')}, tokens {r.get('completion_tokens')}, ttft {r.get('ttft_s')} s, per-stream decode {r.get('decode_tps')}, wall {r.get('wall_tps')} t/s")
PY
alive=0; [ "$(served_id)" = "$MODEL" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext)" = 0 ] && alive=1
[ "$alive" = 1 ] && log "server alive after the probe" || log "SERVER NOT ALIVE after the probe"
sudo docker logs flashnext > "$R/docker-final.log" 2>&1
ck=$(r787_clocks_ok) && cko=0 || cko=1
log "after the probe: OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/docker-final.log"); foreign (report-only) $(r787_foreign "$R/docker-final.log"); $ck; free $(vram_free)"
[ "$cko" = 0 ] || void "clocks / power drifted during the probe: $ck"
VERDICT="DONE $(python3 -c 'import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; c1=[x for x in r if x.get("tag","").startswith("c1-") and x.get("ok") and x.get("decode_tps")]; print(len(c1), "/ 12 c1 records with a decode rate,", len(r) - len(c1), "other")' "$R/depth.jsonl" 2>/dev/null)$([ "$alive" = 1 ] || echo '; SERVER NOT ALIVE at the end')"
finish DONE
