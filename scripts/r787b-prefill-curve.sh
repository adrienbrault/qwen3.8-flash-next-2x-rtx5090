#!/usr/bin/env bash
# RUN VIA r787-chain.sh ONLY: standalone, a stop while queued leaves :8022 down (no queued-signal restore; PRELAUNCH-R787 #6).
# R787b (2026-09-27): the README prefill figure's prefill line (docs/img/prefill.svg, purple) re-measured on the R785 daily
# (tabbyapi:rebase-dev-r3, pool 901,120, split [30, 30], 42 EXL3 keys incl. EXL3_GR_MIX_TILED=1 = tiled HC prefill, TUNEDIR
# /srv/qwen5090/.exl3cache-rebase-dev-r3, live launcher md5 e3db755f). Copy of r580-decode-curve.sh (R580 try 2,
# 2026-09-20, published as 2026-09-20-r580-decode-curve-try2), PREFILL PART ONLY.
# INSTRUMENT (unchanged from R580): one boot of the live launcher (env -i, NVME_TIER= : tier off, every prompt cold), a
# greedy c1 fingerprint (256 forced tokens, provenance only), then cold prefill at TRUE prompt lengths: fn_bench --ctx is
# a filler budget (~0.75 tokens per unit for prose), so the words-to-tokens ratio is calibrated once on this boot from
# the server's usage counter (tag `cal`, budget 60,000), then each target 30k / 60k / 120k / 200k / 240k gets the budget
# that lands on it, three salted --unique runs each (tags pf-<target>, 64 forced tokens), and the server-counted prompt
# tokens are reported, never the budget. Records: $R/prefill.jsonl (raw) and $R/prefill.tsv, for bench/plot.py.
# WHY PREFILL ONLY (decided 2026-09-27): bench/plot.py reads only prefill.jsonl from R580's dir (its records.jsonl path
# is just the parent). R580's decode curve ran fn_bench WITHOUT --distinct (one prompt for all streams of a round; R707
# review: reads 5-12 % fast per step), which R719/R719b replaced; R787a re-measures that curve with the current
# instrument. Keeping R580's curve would cost ~10 min of GPU for a second, contradicting curve nobody draws.
# CHANGES AGAINST r580-decode-curve.sh besides dropping the curve:
#   - Built on lib/serve-ctl.sh (served_stop / wait_served_id / finish_restore) instead of r580's private served_id / up /
#     finish: r580's served_id printed "<no answer>" where serve-ctl's prints nothing, so the two cannot be mixed, and
#     r580's finish appended the restore launcher's stdout (whose last line is the LAN address) to audit.log.
#   - The results dir is no longer hardcoded to one date ($R787_DATE-r787b-prefill-curve).
#   - Provenance gate on the boot (r787-common.sh r787_boot_ok: image, 42 keys + tiled, pool, split, TUNEDIR, tier off,
#     stock power, core 0, memory +4500) and a clocks re-read after the ladder; any miss ends the unit VOID.
#   - Direct :8022 clients stopped (hermes, hermes-webui, owui-proxy), Olla drained, report-only foreign count.
#   - The prefill loop, pf / pfstat, the salts, the budgets and the tags are r580's, unchanged.
# GPU ~6 min (R580 try 2: boot + calibration + 15 prefills in ~4 min at ~10k t/s; the 200k / 240k points dominate).
# RUN: normally from r787-chain.sh. Alone:
#   sudo systemd-run --unit=r787b-prefill-curve --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=1800 \
#     -p Environment=HOME=$HOME /usr/bin/bash /srv/qwen5090/r787b-prefill-curve.sh
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
UNIT=r787b
log(){ echo "$(date -Is) [$UNIT] $*" | tee -a "$R/audit.log"; }
. /srv/qwen5090/lib/serve-ctl.sh
. /srv/qwen5090/r787-common.sh
R=/srv/qwen5090/results/$R787_DATE-r787b-prefill-curve
[ -e "$R/audit.log" ] && R=$R-$(date +%H%M)   # a re-run never appends to an earlier run's prefill.jsonl
mkdir -p "$R"
API=http://127.0.0.1:8022/v1
NEWM=$R787_MODEL
LIVE=$R787_LIVE
CFG=/srv/qwen5090/flashnext-config.yml
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
SCTL_LOG="$R/audit.log"
cfgline(){ sudo grep -E '^  (cache_size|cache_mode|max_batch_size|gpu_split|chunk_size|vision):' $CFG | awk '{$1=$1; print}' | tr '\n' ' '; }
alive(){ [ "$(served_id)" = "$NEWM" ] && [ "$(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null || echo 9)" = 0 ]; }
for f in "$LIVE" /srv/qwen5090/probes/fn_bench.py /srv/qwen5090/lib/gpu-queue.sh /srv/qwen5090/lib/gateway-drain.sh; do
  [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
why=$(r787_launcher_ok) || { log "ABORT (before the lock, nothing touched): $why"; exit 3; }
export GPU_QUEUE_NAME=r787b-prefill-curve
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/gateway-drain.sh
VERDICT="VOID the unit ended before the ladder finished"
FINISHED=0
finish(){ [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  [ -n "$(served_id)" ] && sudo docker logs flashnext > "$R/docker-final.log" 2>&1
  finish_restore "$LIVE"; r787_unquiesce; log "$R787_QMSG"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true
  log "=== R787b $1 ==="; log "VERDICT: $VERDICT"; }
void(){ VERDICT="VOID $*"; finish VOID; exit 3; }
trap 'log "signal"; VERDICT="VOID signal"; finish ABORTED; exit 4' TERM INT HUP
gpu_lock
gateway_drain
r787_quiesce
log "lock held; served at entry: $(served_id || echo none) $(cfgline); $R787_QMSG"
gateway_wait_idle 900 >> "$R/audit.log" 2>&1 || log "WARN: Olla still has requests in flight after 900 s"
why=$(r787_launcher_ok) || void "under the lock: $why"
cp "$LIVE" "$R/launcher-at-lock.sh"
greedy(){ curl -s -m 900 "$API/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$NEWM" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }

log "=== one boot of the live launcher, tier off ==="
served_stop; wait_unserved 45
env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= bash "$LIVE" > "$R/boot-S.log" 2>&1 \
  && wait_served_id "$NEWM" 200 8 || { sudo docker logs flashnext > "$R/docker-noboot.log" 2>&1; void "NO BOOT"; }
pv=$(r787_boot_ok "$R/boot-S.log") || { log "$pv"; void "not the R785 daily at the published regime: ${pv#*BAD:}"; }
pl=$(sudo grep -E 'draft_num_tokens_by_batch' $CFG | awk '{$1=$1; print}')
log "UP: $pv"
log "UP: $(cfgline); $pl; VRAM free $(vram_free); c1 fingerprint $(greedy S)"

# Cold prefill at TRUE prompt lengths, to the top of the window (r580, unchanged). fn_bench's --ctx is a filler budget,
# not a token count: the salted passage is ctx/1.6 words, which lands near 0.75 x ctx tokens. So calibrate the ratio once
# on this boot from the server's own usage counter, then ask for the budget that lands on each target, and report the
# prompt tokens the server counted, never the budget. Three salted runs per depth, tier off, so all are cold.
SALT=$(( $(date +%s) % 100000 ))
pf(){ # pf TAG BUDGET SALT_K -> one salted cold prefill into prefill.jsonl
  python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$NEWM" --tag "$1" --kind prose --tokens 64 \
    --conc 1 --runs 1 --ctx "$2" --unique --salt $(( SALT + $2/1000 + $3*997 + RANDOM )) \
    --out "$R/prefill.jsonl" > "$R/prefill-$1-$3.log" 2>&1; }
pfstat(){ python3 -c "
import json,sys,statistics as st
rows=[json.loads(l) for l in open(sys.argv[1])]
rows=[r for r in rows if r.get('tag')==sys.argv[2] and r.get('ttft_s') and r.get('prompt_tokens')]
if not rows: print('none'); raise SystemExit
print(f\"{st.mean(r['prompt_tokens'] for r in rows):.0f}\t{st.mean(r['prompt_tokens']/r['ttft_s'] for r in rows):.0f}\t{st.mean(r['ttft_s'] for r in rows):.2f}\t{len(rows)}\")
" "$R/prefill.jsonl" "$1" 2>/dev/null || echo none; }

pf cal 60000 0
CAL=$(pfstat cal | cut -f1)
case "$CAL" in ''|none|0) void "prefill calibration produced no usable record";; esac
log "prefill calibration: budget 60,000 -> $CAL prompt tokens ($(python3 -c "print(f'{$CAL/60000:.4f}')") tokens per budget unit)"

: > "$R/prefill.tsv"; printf "target\tbudget\tprompt_tokens\ttps\tttft_s\tn\n" >> "$R/prefill.tsv"
for T in 30000 60000 120000 200000 240000; do
  B=$(python3 -c "print(round($T * 60000 / $CAL))")
  for k in 1 2 3; do pf "pf-$T" "$B" $k; done
  alive || void "server not alive after the $T-token prefills"
  ST=$(pfstat "pf-$T")
  case "$ST" in none) log "prefill $T: no usable record (budget $B; the server may have refused the length)";;
    *) log "prefill target $T: budget $B -> $(echo "$ST" | awk -F'\t' '{printf "%s prompt tokens, %s t/s, TTFT %s s (n %s)", $1, $2, $3, $4}')"
       printf "%s\t%s\t%s\n" "$T" "$B" "$ST" >> "$R/prefill.tsv";; esac
done

sudo docker logs flashnext > "$R/docker-final.log" 2>&1
ck=$(r787_clocks_ok) && cko=0 || cko=1
log "after the ladder: OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/docker-final.log"); restarts $(sudo docker inspect -f '{{.RestartCount}}' flashnext 2>/dev/null); foreign (report-only) $(r787_foreign "$R/docker-final.log"); $ck; VRAM free at the end: $(vram_free)"
[ "$cko" = 0 ] || void "clocks / power drifted during the ladder: $ck"
VERDICT="DONE $(( $(grep -c . "$R/prefill.tsv") - 1 ))/5 prefill targets with records (want 5)"
finish DONE
