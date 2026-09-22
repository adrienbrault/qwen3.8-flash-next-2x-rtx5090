#!/usr/bin/env bash
# fn_gate.sh — the canonical Flash-Next gate: ONE script that produces the honest publishable table
# for whatever is serving on the API right now (2026-09-22).
#
# WHY THIS EXISTS. The repo's decode numbers accumulated across five instruments and two documented
# traps this week:
#   * every 256-token gate sits inside a start-of-generation TRANSIENT that reads ~30 % high
#     (R638->R583 correction): steady state begins around 1,024 generated tokens, so the gate
#     forces >=1,024 per request by default;
#   * per-request rates must be MEDIANS or time-weighted, never means — production traffic is
#     bimodal and a 176 t/s mean over it was really 65.6 (the backlog's measurement rule);
#   * a gate must also record WHAT served: model id, image, resolved env-key count — an arm
#     measured against the wrong boot is a table about nothing.
#
# Shapes (matching the promoted-config conventions):
#   leg A  ctx ~4k,   c1/c4/c8, tokens 1024, runs 2, warmup 1  — the headline decode table
#   leg B  ctx ~26k,  c4,       tokens 1024, runs 2, warmup 1  — the production shape
#
# Every request lands in $OUT/*.jsonl (one JSON object each: tokens, wall, ttft, decode window,
# server vs client counts) — the summary is derived, never the record.
#
# usage: fn_gate.sh OUT_DIR [URL] [MODEL] [SALT]
#   e.g.  fn_gate.sh /srv/qwen5090/results/2026-09-22-gate \
#           http://127.0.0.1:8022/v1 qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
set -uo pipefail
OUT=${1:?usage: fn_gate.sh OUT_DIR [URL] [MODEL] [SALT]}
API=${2:-http://127.0.0.1:8022/v1}
MODEL=${3:-}
SALT=${4:-424242}
TOKENS=${TOKENS:-1024}
RUNS=${RUNS:-2}
PROBE=${PROBE:-/srv/qwen5090/probes/fn_bench.py}
mkdir -p "$OUT"
LOG="$OUT/gate-audit.log"
log(){ echo "$(date -Is) [gate] $*" | tee -a "$LOG"; }

[ "$TOKENS" -ge 1024 ] || { log "ABORT: TOKENS=$TOKENS <1024 sits inside the publish transient (R638); screening A/Bs live elsewhere"; exit 3; }

served_id(){ curl -s -m 8 "$API/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null; }
SID=$(served_id)
[ -n "$SID" ] || { log "ABORT: nothing answering at $API"; exit 3; }
[ -n "$MODEL" ] && [ "$SID" != "$MODEL" ] && { log "ABORT: API serves '$SID', gate asked for '$MODEL' — a table about the wrong model is worse than none"; exit 3; }
MODEL=${MODEL:-$SID}
IMG=$(sudo docker ps --filter "name=^flashnext$" --format '{{.Image}}' 2>/dev/null | head -1)
log "serving: $SID | image: ${IMG:-<not flashnext>} | tokens $TOKENS x$RUNS salt $SALT"
# Resolved env keys from the CURRENT launcher log — the count catches an env-drop arm (R614) even
# when the caller meant well.
grep -aoE 'env keys \([0-9]+\)' /srv/qwen5090/logs/flashnext-*.log 2>/dev/null | tail -1 | tee -a "$LOG" || true

leg(){ local tag=$1; shift
  log "leg $tag: fn_bench $*"
  python3 "$PROBE" --url "$API" --model "$MODEL" --tag "$tag" \
    --tokens "$TOKENS" --warmup-runs 1 --unique --distinct --salt "$SALT" \
    --out "$OUT/bench-$tag.jsonl" "$@" >> "$OUT/bench-$tag.log" 2>&1 \
    && log "leg $tag: $(wc -l < "$OUT/bench-$tag.jsonl") rows" \
    || log "leg $tag FAILED (see bench-$tag.log)"
}

leg A --kind prose --ctx 4000  --conc 1 4 8 --runs "$RUNS"
leg B --kind prose --ctx 26000 --conc 4     --runs "$RUNS"

python3 - "$OUT" <<'PY' | tee -a "$LOG"
import json, os, statistics as st, sys
OUT = sys.argv[1]
def recs(tag):
    p = os.path.join(OUT, f"bench-{tag}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def med(v): return st.median(v) if v else None
def row(tag, conc):
    ok = [r for r in recs(tag) if r.get("ok") and r.get("conc") == conc]
    if not ok: return None
    acc = [r["completion_tokens"]/r["client_frames"] for r in ok if r.get("client_frames")]
    return dict(
        n=len(ok),
        decode=med([r["decode_tps"] for r in ok if r.get("decode_tps")]),
        wall=med([r["wall_tps"] for r in ok if r.get("wall_tps")]),
        ttft=med([r["ttft_s"] for r in ok if r.get("ttft_s") is not None]),
        acc=med(acc),
        tokens=med([r["completion_tokens"] for r in ok if r.get("completion_tokens")]),
        prompt=ok[0].get("prompt_tokens"),
        finish={f: sum(1 for r in ok if r.get("finish_reason")==f) for f in {r.get("finish_reason") for r in ok}},
    )
print("\n=== GATE TABLE (medians over per-request rows; steady-state decode) ===")
print(f"{'leg':>4} {'conc':>4} {'n':>3} {'prompt':>7} {'tokens':>7} {'decode':>7} {'wall':>7} {'ttft':>6} {'acc':>6}  finishes")
for tag, concs in (("A",[1,4,8]), ("B",[4])):
    for c in concs:
        r = row(tag, c)
        if not r:
            print(f"{tag:>4} {c:>4}   — no rows"); continue
        g = lambda v,f="{:.1f}": f.format(v) if v is not None else "-"
        print(f"{tag:>4} {c:>4} {r['n']:>3} {g(r['prompt'],'{:.0f}'):>7} {g(r['tokens'],'{:.0f}'):>7}"
              f" {g(r['decode']):>7} {g(r['wall']):>7} {g(r['ttft'],'{:.2f}'):>6} {g(r['acc'],'{:.2f}'):>6}  {r['finish']}")
# Transient quantifier, when a same-shape 256-token sibling exists: callers may drop
# bench-A256.jsonl alongside; ratio <1 says the old gate read high.
p256 = os.path.join(OUT, "bench-A256.jsonl")
if os.path.exists(p256):
    a1 = [r["decode_tps"] for r in recs("A") if r.get("ok") and r.get("conc")==1 and r.get("decode_tps")]
    b1 = [r["decode_tps"] for r in recs("A256") if r.get("ok") and r.get("conc")==1 and r.get("decode_tps")]
    if a1 and b1:
        print(f"\n256-token transient read: c1 {st.median(b1):.1f} vs 1024-token {st.median(a1):.1f}"
              f"  ({(st.median(b1)/st.median(a1)-1)*100:+.1f}% inflation)")
PY
log "=== GATE DONE ==="
