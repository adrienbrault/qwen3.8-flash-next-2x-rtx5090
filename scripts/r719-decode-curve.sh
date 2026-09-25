#!/usr/bin/env bash
# R719 (2026-09-25): the README decode curve re-measured on the CURRENT daily (stack-r3-rows32, pool 983,040, policy
# [[4, 3], [8, 2]]), two boots of the live launcher. Same instrument as R704 (fn_bench, greedy, 1,024 forced tokens, 1 warm-up
# + 3 recorded rounds per shape, c1..c8, code and prose, NVMe tier off) EXCEPT --distinct: every concurrent request gets
# its own suffix, so streams cannot share a trajectory (R707 review: R704's one-prompt-for-all rounds read 5-12 % fast
# per step). Metrics as R704: per-stream decode (median of per-request (n-1)/(t_last-t_first)), decode aggregate (sum of
# the concurrent requests' decode rates), round-wall aggregate (labelled secondary), TTFT separately. GPU ~25 min.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/$(date +%F)-r719-decode-curve; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
API=http://127.0.0.1:8022/v1
log(){ echo "$(date -Is) [r719] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=r719-decode-curve
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ sudo docker logs flashnext > "$R/container-final.log" 2>&1 || true
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== R719 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
gpu_lock
log "lock held; served at entry: $(served_id || echo none)"
arm(){ local tag=$1 L=$2
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= bash "$L" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { log "[$tag] NO BOOT"; return 1; }
  log "[$tag] booted $(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'); stack flags $(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -cE '^EXL3_(HC_MIX_V3|MOE_COOP_V3)='); free $(vram_free)"
  for conc in 1 2 3 4 5 6 7 8; do for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-c$conc-$kind" --kind $kind --distinct \
      --tokens 1024 --warmup-runs 1 --conc $conc --runs 3 --out "$R/records.jsonl" > "$R/bench-$tag-c$conc-$kind.log" 2>&1
  done; done
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  log "[$tag] done; OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$tag.log"); free after $(vram_free)"; }
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
finish DONE
