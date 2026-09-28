#!/usr/bin/env bash
# R790 (2026-09-28): does the served sampler fallback (temperature 1.0 since 2026-09-27 20:18 UTC) hurt loop-think
# recovery, and does it change loop entry? Closes open questions 2-3 of the R789 review.
#   The 2026-09-28-loopthink-temp-0011 probe read 7/16 forced rows carrying the loop into the answer at the preset (1.0)
#   vs 0/16 at explicit 0.6, but preset-vs-explicit was confounded with 1.0-vs-0.6 and it ran on the 42-key daily.
#   Here, on the served 39-key daily (client only: no GPU lock, the daily keeps serving):
#     forced  (replay_hermes_turn --forced-loop, 40 prefilled copies, "17*23"): explicit 1.0 / explicit 0.6 / preset,
#             n 8 per batch, two rounds in order e1.0 e0.6 pre | e0.6 pre e1.0  -> n 16 per arm
#     p107    (R781's Hermes session, turn 16, R782's prefix): explicit 1.0 / explicit 0.6, n 8 each -> loop ENTRY
#   Score: forced -> clean '391' answer vs looped line in the answer; p107 -> injected (loop-think fired) / escaped / no-loop,
#   and answered (tool call or content). The container log over the run is kept.
# Run: bash /srv/qwen5090/r790-loopthink-temp.sh   (as the operator user, ~15 min)
set -uo pipefail
D=/srv/qwen5090; P=$D/probes
R=$D/results/$(date +%F)-r790-loopthink-temp-$(date +%H%M); mkdir -p "$R"
M=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab; API=http://127.0.0.1:8022/v1
CTL=$D/results/2026-09-27-r781-control
P107=$(ls -d $D/results/2026-09-27-r782-loopthink2-*/ | tail -1)prefix-p107.txt
log(){ echo "$(date -Is) [r790] $*" | tee -a "$R/audit.log"; }
cp "$0" "$P/replay_hermes_turn.py" "$R/"
grep -q -- '--temperature' "$P/replay_hermes_turn.py" || { log "ABORT: probe lacks --temperature"; exit 3; }
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
log "start; live launcher md5 $(md5sum < $D/launch-flashnext.sh | cut -c1-32); keys $(sudo docker exec flashnext env | grep -c '^EXL3_')"
forced(){ local arm=$1 i=$2 a=; [ "$arm" = pre ] || a="--temperature $arm"
  python3 "$P/replay_hermes_turn.py" --url $API --model $M --forced-loop --n 8 --conc 4 --max-tokens 12000 $a \
    --tag "forced-$arm-$i" --out "$R/forced-$arm-$i.jsonl" > "$R/forced-$arm-$i.txt" 2>&1; log "forced $arm round $i rc $?"; }
p107(){ local t=$1
  python3 "$P/replay_hermes_turn.py" --url $API --model $M --session "$CTL/session.json" --db "$CTL/state.db" --upto 16 \
    --tools "$P/hermes-tools-r779.json" --prefix-file "$P107" --n 8 --conc 4 --temperature $t \
    --tag "p107-$t" --out "$R/p107-$t.jsonl" > "$R/p107-$t.txt" 2>&1; log "p107 $t rc $?"; }
forced 1.0 1; forced 0.6 1; forced pre 1
p107 1.0; p107 0.6
forced 0.6 2; forced pre 2; forced 1.0 2
sudo docker logs --since "$T0" flashnext > "$R/container.log" 2>&1
python3 - "$R" <<'PY' | tee "$R/summary.txt"
import json, glob, sys, collections
R = sys.argv[1]
MSG = "I am repeating myself, so I will stop thinking here and act on what I have."
agg = collections.defaultdict(lambda: collections.Counter())
for f in sorted(glob.glob(f"{R}/forced-*.jsonl")):
    arm = f.rsplit("/", 1)[1].split("-")[1]
    for l in open(f):
        r = json.loads(l); c = r.get("content") or ""
        k = agg["forced " + arm]; k["n"] += 1
        k["clean 391"] += ("391" in c and "quick brown fox" not in c)
        k["loop in answer"] += ("quick brown fox" in c)
        k["error"] += (r["class"] == "error")
for f in sorted(glob.glob(f"{R}/p107-*.jsonl")):
    arm = f.rsplit("/", 1)[1][5:-6]
    for l in open(f):
        r = json.loads(l); t = (r.get("reasoning") or "") + (r.get("content") or "")
        k = agg["p107 " + arm]; k["n"] += 1
        k["injected"] += MSG in t
        k["escaped (>=3 cone copies, no injection)"] += (MSG not in t and (r.get("reasoning") or "").count("Better approach: make the hill a cone") >= 3)
        k["answered"] += r["class"] in ("tool_call", "content")
        k["class " + r["class"]] += 1
for a, k in sorted(agg.items()):
    print(a, dict(k))
PY
log "loop stops in log: reasoning $(grep -ac 'reasoning loop detected' "$R/container.log"), token $(grep -ac 'loop was detected' "$R/container.log"); done $R"
