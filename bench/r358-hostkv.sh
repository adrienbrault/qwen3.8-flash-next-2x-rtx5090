#!/usr/bin/env bash
# R358 — whether a host KV tier helps deep-context concurrency, the weakest measured axis on this box.
#
# THE WEAKNESS, MEASURED. Eight concurrent jobs of 78,233-79,139 tokens each (~628k of unrelated context) fit in
# this box's 262,144-token VRAM pool only by queueing: all eight complete, but the first token takes 43-82 s
# (r345). `sysmem_kv_cache` is the last config knob that could plausibly change that without engine work — it puts
# a second-tier KV cache in host RAM — and it has been 0 in every measurement so far.
#
# EXPECTATION, STATED BEFORE THE RUN: a host tier cannot make more than 262,144 tokens of *active* context fit, so
# it should not change admission. What it can change is recomputation and prefix reuse, so the discriminating arms
# are the ones with repeated prefixes and re-prefill, not the unique-context one.
#
# Arms: tier 0 (the served baseline) and tier 4096 MiB, each with
#   a) 8 x unique deep contexts     — admission and TTFT, expected unchanged
#   b) 1 x 131k context twice       — whether a long prefix stays usable between requests
#   c) 4 x shared 78k prefix        — the realistic fan-out shape
#
# RUN: sudo systemd-run --unit=r358-hostkv --collect -p User=adrienbrault -p RuntimeMaxSec=14400 \
#        bash /srv/qwen5090/r358-hostkv.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-16-r358-hostkv; mkdir -p "$R"
API=http://127.0.0.1:8022/v1
MODEL=qwen3.8-flash-next-exl3-3.05bpw
L=/srv/qwen5090/launch-flashnext-r340.sh
log(){ echo "$(date -Is) [r358] $*" | tee -a "$R/audit.log"; }

export GPU_QUEUE_NAME=r358-hostkv
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock
flock -n 9 || { log "queued behind: $(gpu_queue_others)"; flock 9; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R358 $1 ==="; }
trap 'log "### SIGTERM ###"; finish ABORTED; exit 4' TERM

arm(){  # arm <tag> <sys_kv_mib>
  local tag=$1 kv=$2
  log "booting $tag: SYS_KV=$kv"
  SYS_KV="$kv" bash "$L" >> "$R/audit.log" 2>&1 || { log "$tag BOOT FAILED"; return 1; }
  curl -sf -m 8 "$API/model" >/dev/null || { log "$tag: no server"; return 1; }
  log "  served config says sysmem_kv_cache: $(sudo grep 'sysmem_kv_cache' /srv/qwen5090/flashnext-config.yml | tr -d ' ')"
  P="python3 /srv/qwen5090/probes/fn_bench.py --url $API --model $MODEL --kind code --out $R/records-$tag.jsonl --tag $tag"
  log "  a) 8 x unique deep contexts (512 forced tokens)"
  $P-deep --tokens 512 --ctx 30000 --conc 8 --runs 1 --unique 2>&1 | tee -a "$R/audit.log"
  log "  b) the same 131k context twice, 1k forced (does the prefix stay usable?)"
  $P-cold --tokens 1024 --ctx 120000 --conc 1 --runs 2 2>&1 | tee -a "$R/audit.log"
  log "  c) 4 x shared 78k prefix, 1k forced"
  $P-shared --tokens 1024 --ctx 120000 --conc 4 --runs 1 2>&1 | tee -a "$R/audit.log"
}

arm tier0 0
arm tier4096 4096

log "restoring the baseline (SYS_KV=0) as the served configuration"
SYS_KV=0 bash "$L" >> "$R/audit.log" 2>&1 || log "BASELINE BOOT FAILED"
finish DONE
