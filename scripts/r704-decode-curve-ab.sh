#!/usr/bin/env bash
# R704 (2026-09-24) — the README's decode curve (R580 instrument: fn_bench, greedy, 1,024 tokens, 1 warm-up + 3 recorded
# rounds per shape, c1..c8, code and prose, NVMe tier off) on the PREVIOUS served configuration
# (launch-flashnext.sh.pre-r701 = slotfix-r1 with R694's draft component on cuda:1) against the CURRENT one (stack-r2,
# R701), ABAB, one session. R694 and R701 were judged by fn_gate per-request medians; the published figure was a fn_bench
# round-wall aggregate (README: c8 prose 630, R580), and a per-request median times the stream count is not that metric.
# Three metrics come out of the same records: round-wall aggregate = all streams' tokens / round wall; per-stream decode =
# median of per-request (n-1)/(t_last-t_first), the streaming rate after the first token; decode aggregate = sum of the
# concurrent requests' decode rates (TTFT and straggler tails excluded). TTFT median reported separately.
# Measurement only. GPU ~45 min. Queue-chained; the daily is restored at the end.
set -uo pipefail
export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-24-r704-decode-curve-ab; mkdir -p "$R"
LIVE=/srv/qwen5090/launch-flashnext.sh
OLD=/srv/qwen5090/launch-flashnext.sh.pre-r701
MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
API=http://127.0.0.1:8022/v1
log(){ echo "$(date -Is) [r704] $*" | tee -a "$R/audit.log"; }
export GPU_QUEUE_NAME=r704-decode-curve-ab
. /srv/qwen5090/lib/gpu-queue.sh
. /srv/qwen5090/lib/serve-ctl.sh
SCTL_LOG="$R/audit.log"
finish(){ sudo docker logs flashnext > "$R/container-final.log" 2>&1 || true
  finish_restore "$LIVE"; rm -f "${GPU_QUEUE_MARK:-/nonexistent}"
  sudo chown -R "$(stat -c %U /srv/qwen5090/results)" "$R" 2>/dev/null || true; log "=== R704 $1 ==="; }
trap 'log "signal"; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$OLD" /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done
gpu_lock
log "lock held; served at entry: $(served_id || echo none)"
arm(){ local tag=$1 L=$2
  served_stop; wait_unserved 45
  env -i HOME="$HOME" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin NVME_TIER= bash "$L" > "$R/boot-$tag.log" 2>&1 \
    && wait_served_id "$MODEL" 200 8 || { log "[$tag] NO BOOT"; return 1; }
  log "[$tag] booted $(sudo docker ps --filter name=^flashnext$ --format '{{.Image}}'); stack flags $(sudo docker inspect flashnext --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -cE '^EXL3_(HC_MIX_V3|MOE_COOP_V3)='); free $(vram_free)"
  for conc in 1 2 3 4 5 6 7 8; do for kind in code prose; do
    python3 /srv/qwen5090/probes/fn_bench.py --url "$API" --model "$MODEL" --tag "$tag-c$conc-$kind" --kind $kind \
      --tokens 1024 --warmup-runs 1 --conc $conc --runs 3 --out "$R/records.jsonl" > "$R/bench-$tag-c$conc-$kind.log" 2>&1
  done; done
  sudo docker logs flashnext > "$R/container-$tag.log" 2>&1
  log "[$tag] done; OOM $(grep -acE 'OutOfMemoryError|out of memory' "$R/container-$tag.log"); free after $(vram_free)"; }
arm OLD1 "$OLD"; arm NEW1 "$LIVE"; arm OLD2 "$OLD"; arm NEW2 "$LIVE"
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
def m(arm, d, conc, kind, f=st.mean):
    v=[x for b in (1,2) for x in d.get(f"{arm}{b}-c{conc}-{kind}",[])]; return f(v) if v else float("nan")
out=open(sys.argv[2],"w"); out.write("conc\tkind\told_wallagg\tnew_wallagg\told_decode_stream\tnew_decode_stream\told_decode_agg\tnew_decode_agg\told_ttft\tnew_ttft\n")
for kind in ("code","prose"):
    for c in range(1,9):
        v=[m(a,d,c,kind,f) for d,f in ((agg,st.mean),(dec,st.median),(dagg,st.mean),(ttft,st.median)) for a in ("OLD","NEW")]
        print(f"{kind} c{c}: decode/stream OLD {v[2]:.1f} NEW {v[3]:.1f} ({v[3]/v[2]:.3f}x) | decode aggregate OLD {v[4]:.0f} NEW {v[5]:.0f} | round-wall aggregate OLD {v[0]:.0f} NEW {v[1]:.0f} ({v[1]/v[0]:.3f}x) | TTFT OLD {v[6]:.2f}s NEW {v[7]:.2f}s")
        out.write(f"{c}\t{kind}\t"+"\t".join(f"{x:.2f}" for x in v)+"\n")
PY
finish DONE
