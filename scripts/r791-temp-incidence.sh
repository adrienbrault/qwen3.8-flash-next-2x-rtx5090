#!/usr/bin/env bash
# R791 (2026-09-28): loop INCIDENCE at the sampler fallback 1.0 vs 0.6 on ordinary Hermes turns, no prefix (R790 review
# "next": R790's probes only measured loop continuation / forced rescue). Client-only on the served :8022 daily.
#   Turns: R781's Hermes session snapshot (17 messages), every assistant turn: --upto 1 3 5 8 10 12 14 16, n 4 per
#   temperature per turn (conc 4), order alternating per turn (even index 1.0 first, odd 0.6 first) -> 32 rows per arm.
#   One session = one task family (a three.js scene): variety is limited, stated with the result.
#   Per row: loop-think injection (the r4 message in the stream), repeated-text fraction of the reasoning (share of
#   reasoning lines >= 20 chars that repeat an earlier line; >= 0.5 = looped even when no detector fired), reasoning
#   chars, class (tool_call / content / reasoning_only), wall.
# Run: bash /srv/qwen5090/r791-temp-incidence.sh   (as the operator user, ~30 min)
set -uo pipefail
D=/srv/qwen5090; P=$D/probes
R=$D/results/$(date +%F)-r791-temp-incidence-$(date +%H%M); mkdir -p "$R"
M=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab; API=http://127.0.0.1:8022/v1
CTL=$D/results/2026-09-27-r781-control
log(){ echo "$(date -Is) [r791] $*" | tee -a "$R/audit.log"; }
cp "$0" "$P/replay_hermes_turn.py" "$R/"
grep -q -- '--temperature' "$P/replay_hermes_turn.py" || { log "ABORT: probe lacks --temperature"; exit 3; }
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
log "start; live launcher md5 $(md5sum < $D/launch-flashnext.sh | cut -c1-32)"
arm(){ local u=$1 t=$2
  python3 "$P/replay_hermes_turn.py" --url $API --model $M --session "$CTL/session.json" --db "$CTL/state.db" --upto $u \
    --tools "$P/hermes-tools-r779.json" --n 4 --conc 4 --temperature $t \
    --tag "u$u-t$t" --out "$R/u$u-t$t.jsonl" > "$R/u$u-t$t.txt" 2>&1; log "upto $u temp $t rc $?"; }
i=0
for u in 1 3 5 8 10 12 14 16; do
  if [ $((i % 2)) = 0 ]; then arm $u 1.0; arm $u 0.6; else arm $u 0.6; arm $u 1.0; fi
  i=$((i + 1))
done
sudo docker logs --since "$T0" flashnext > "$R/container.log" 2>&1
python3 - "$R" <<'PY' | tee "$R/summary.txt"
import json, glob, sys, collections, statistics as st
R = sys.argv[1]
MSG = "I am repeating myself, so I will stop thinking here and act on what I have."
def rep(t):
    ls = [l.strip() for l in t.split("\n") if len(l.strip()) >= 20]
    seen, d = set(), 0
    for l in ls:
        d += l in seen; seen.add(l)
    return d / len(ls) if ls else 0.0
rows = collections.defaultdict(list)
for f in sorted(glob.glob(f"{R}/u*-t*.jsonl")):
    u, t = f.rsplit("/", 1)[1][:-6].split("-t")
    for l in open(f):
        r = json.loads(l); x = r.get("reasoning") or ""
        rows[t].append(dict(u=u, inj=MSG in x + (r.get("content") or ""), rep=rep(x), rc=len(x), cls=r["class"],
                            wall=r.get("wall"), finish=r.get("finish")))
        print(f"{u} t{t} {r['class']:<14} finish {r.get('finish')!s:<6} inj {int(MSG in x)} rep {rep(x):.2f} rchars {len(x):>6} wall {r.get('wall')}")
for t, L in sorted(rows.items()):
    n = len(L); looped = [r for r in L if r["inj"] or r["rep"] >= 0.5]
    print(f"ARM t{t}: n {n}; looped (injection or rep>=0.5) {len(looped)} ({sorted(set(r['u'] for r in looped))}); "
          f"injections {sum(r['inj'] for r in L)}; rep>=0.5 without injection {sum((not r['inj']) and r['rep'] >= 0.5 for r in L)}; "
          f"answered {sum(r['cls'] in ('tool_call', 'content') for r in L)}; reasoning chars median {st.median(r['rc'] for r in L):.0f} "
          f"p90 {sorted(r['rc'] for r in L)[int(0.9 * n) - 1]}; wall median {st.median(r['wall'] or 0 for r in L):.1f} s")
PY
log "container log: reasoning loop injections $(grep -ac 'reasoning loop detected' "$R/container.log"), token loop stops $(grep -ac 'loop was detected' "$R/container.log"); done $R"
