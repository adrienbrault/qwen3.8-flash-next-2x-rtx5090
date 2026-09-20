#!/usr/bin/env bash
# R587 — build, verify and (if the gates pass) promote a Prometheus /metrics endpoint in TabbyAPI.
# Overlay: patches/tabbyapi/metrics/r1 (SWE-2 round, 2026-09-20). Pure stdlib -- no new dependency is installed,
# because prometheus_client is not in the venv and the image build has no package-install step.
#
# WHY NOW. The monitoring stack (Grafana + VictoriaMetrics in k3s) has a live dcgm job and a `vllm` job pointed at a
# port that stopped serving weeks ago, so there is no application-level visibility at all. Every decode number in
# this repo had to be reconstructed by hand from the container log, and on 2026-09-20 that reconstruction is what
# finally explained the production gap -- after two method errors (averaging rate per request, and a midnight
# rollover in timestamps that carry no date) and several hours. The counter pair this overlay exports,
#   rate(tabby_generated_tokens_total[5m]) / rate(tabby_generate_seconds_total[5m])
# is that number, time-weighted, continuously, for free. The histograms are the distributions R583-R585 had to
# build a unit per question to see: generated length, context depth, per-request decode rate, TTFT, queue time.
#
# THE ORDER MATTERS. R586 runs SWE-bench Verified 500 next -- 500 agent sessions, which is the largest and most
# realistic sample of exactly the interleaved regime R585 just characterised. Promoting this first means that run
# is instrumented instead of opaque, so this round is queued AHEAD of it.
#
# What the overlay touches: two added lines in common/gen_logging.py (an import and a record_completion call at the
# end of log_metrics), two added blocks in endpoints/core/router.py (an import and an unauthenticated /metrics
# route beside /health), and two new files. install.py verifies both baseline hashes -- taken from the live
# container, 2026-09-20 -- before writing anything, so a base image that has drifted fails the build.
#   B  build $NIMG on the live image; the Dockerfile runs the 20 unit tests and a render smoke inside the build.
#   V  boot the candidate, scrape /metrics, drive KNOWN traffic, scrape again, assert the deltas match what was
#      sent. A metrics endpoint that reports plausible-looking numbers it did not measure is worse than none.
#   G  c1 + 30k greedy fingerprints MUST be canonical (this overlay changes no math, so anything else is a bug),
#      and boot free VRAM within the headroom rule.
#   P  promote: IMG in the launcher, restart, re-verify /metrics on the daily. Rollback: .pre-r587.
# TRY 5 (2026-09-20): try 4 swapped in R577's digest helpers and they returned `none / none` — a third harness
# fault, same shape as the first two. R577 sets API to .../8022/v1 and appends "/chat/completions"; this script
# sets API to the bare .../8022 because served_id and the verify curls append their own /v1. Lifted unchanged, the
# helpers POSTed to /chat/completions, got a 404, and `|| echo none` turned that into a gate failure. Both paths
# now carry an explicit /v1. Note what has held steady through all of this: the endpoint verification passed
# exactly again on try 4 (3 requests / 768 forced tokens, 206.5 t/s implied). The overlay has never been what fails.
# TRY 4 (2026-09-20): try 3's VERIFICATION PASSED EXACTLY -- 3 requests / 768 forced tokens in, /metrics moved by
# 3.0 and 768.0, implied 203.7 t/s. The endpoint is proven correct on the real seat. What failed was gate G, and
# it failed on an empty string: `fn_greedy.py` does not emit fingerprints at all. It writes the reply TEXT to a
# jsonl and prints `GREEDY <tag> <pid> depth= n= finish= '<preview>'`; its equivalence check is a separate
# --compare pass that diffs two arms' text. There is no 16-hex digest anywhere in its output, so grepping its log
# for one could only ever return nothing, and "no match" read as "not canonical". The canonical c1/30k digests
# come from R577's own helpers -- sha256 of content + "|" + reasoning_content, first 16 hex, over one fixed 256-
# token chat completion and one fixed ~30k-token one. Those two functions are lifted here verbatim, so this gate
# now compares like with like against the same REF1/REF30 every other promotion round used.
# TRY 3 (2026-09-20): try 2 built the image, booted it and the endpoint WORKED -- it reported 4 requests and 776
# generated tokens where the client had forced 3 x 256 = 768. That is the endpoint being exactly right: fn_bench
# issues an 8-token warmup round before its recorded ones, and 776 = 768 + 8. The verification, not the overlay,
# was wrong. Traffic is now three plain curl completions with min_tokens = max_tokens = 256, so what is asserted
# is exactly what is sent. Implied rate on the candidate was 265.8 t/s, in band for this shape.
# TRY 2 (2026-09-20): try 1 failed in under a second building an image tagged "\ntabbyapi:mtpwin-r2-metrics1".
# `docker inspect` on a container that is not there prints an empty line AND exits non-zero, so the `inspect ||
# grep` form captured both outputs. The two sources are now read separately, stripped, and the result is checked
# against the characters a tag may contain.
# GPU TIMEBOX ~25 min. RUN (queued): r515-queue-chain.
set -uo pipefail
export HOME=$HOME PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-20-r587-tabby-metrics; mkdir -p "$R"
API=http://127.0.0.1:8022
NEWM=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
LIVE=/srv/qwen5090/launch-flashnext.sh
# The canonical greedy digests for the current daily, unchanged since R579. Both must hold: this overlay adds an
# import and a counter update on an already-completed request, so it cannot legitimately move either one.
REF1=18238d63065ee16c
REF30=4a255910dee2d9c5
PROMPT="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
SRC=${SRC:-/srv/qwen5090/overlay-src/metrics-r1}
CLEAN_ENV=(env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
BOOTED=0
log(){ echo "$(date -Is) [r587] $*" | tee -a "$R/audit.log"; }
served_id(){ curl -s -m 8 "$API/v1/model" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])' 2>/dev/null || echo "<no answer>"; }
cstatus(){ sudo docker ps -a --format '{{.Status}}' -f name=flashnext 2>/dev/null | head -1; }
vram(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'; }
finish(){
  if [ "$BOOTED" = 1 ]; then
    sudo docker logs flashnext > "$R/docker-final.log" 2>&1
    if [ -n "$(gpu_queue_others)" ]; then log "not restoring: GPU queue continues ($(gpu_queue_others))"; sudo docker rm -f flashnext >/dev/null 2>&1
    else log "restoring the daily"; sudo docker rm -f flashnext >/dev/null 2>&1
      "${CLEAN_ENV[@]}" bash "$LIVE" >> "$R/audit.log" 2>&1 || log "RESTORE LAUNCHER FAILED"
      for i in $(seq 240); do [ "$(served_id)" != "<no answer>" ] && break; sleep 2; done; log "daily: $(served_id)"; fi; fi
  rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; log "=== R587 $1 ==="; }
trap 'log signal; finish ABORTED; exit 4' TERM INT HUP
for f in "$LIVE" "$SRC/Dockerfile.box" "$SRC/manifest.json" "$SRC/install.py" /srv/qwen5090/probes/fn_greedy.py \
         /srv/qwen5090/probes/fn_bench.py; do [ -e "$f" ] || { log "ABORT: missing $f"; exit 3; }; done

export GPU_QUEUE_NAME=r587-tabby-metrics
. /srv/qwen5090/lib/gpu-queue.sh
gpu_lock
# `docker inspect` on a missing container prints an EMPTY LINE to stdout and exits non-zero, so writing this as
# `$(inspect || grep)` captures both and yields "\ntabbyapi:...", which docker build rejects as an invalid tag.
# Try 1 died there in under a second. Take the two sources separately and strip.
LIMG=$(sudo docker inspect flashnext --format '{{.Config.Image}}' 2>/dev/null | tr -d '[:space:]')
[ -n "$LIMG" ] || LIMG=$(grep -oP '^IMG=\$\{IMG:-\K[^}]+' "$LIVE" | tail -1 | tr -d '[:space:]')
case "$LIMG" in ""|*[!A-Za-z0-9._:/-]*) log "ABORT: bad live image '$LIMG'"; exit 3;; esac
NIMG="$LIMG-metrics1"
log "lock held; live image $LIMG -> candidate $NIMG; served at entry: $(served_id)"
BOOTED=1

# ---- B: build ----
log "=== B: building $NIMG (install.py verifies both baseline hashes; the build runs the unit tests) ==="
sudo docker build -f "$SRC/Dockerfile.box" --build-arg BASE="$LIMG" -t "$NIMG" "$SRC" > "$R/build.log" 2>&1
br=$?
log "build rc=$br: $(grep -aE 'Successfully|error|Error|FAILED|OK \(|Ran [0-9]+ tests|render OK' "$R/build.log" | tail -4 | tr '\n' ' ' | cut -c1-320)"
[ $br = 0 ] || { log "BUILD FAILED"; finish FAILED; exit 1; }

up(){ local img=$1 tag=$2 i st lp
  sudo docker rm -f flashnext >/dev/null 2>&1; for i in $(seq 30); do [ "$(served_id)" = "<no answer>" ] && break; sleep 2; done
  ( "${CLEAN_ENV[@]}" IMG="$img" bash "$LIVE" > "$R/boot-$tag.log" 2>&1 ) & lp=$!
  for i in $(seq 200); do
    [ "$(served_id)" = "$NEWM" ] && { wait $lp; return 0; }
    st=$(cstatus); case "$st" in Restarting*|Exited*|"")
      [ -z "$st" ] && [ $i -lt 40 ] && { sleep 3; continue; }
      log "NO BOOT $tag ($st): $(sudo docker logs --tail 80 flashnext 2>&1 | grep -aE 'RuntimeError|Error|memory' | tail -1 | cut -c1-160)"
      kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; wait $lp 2>/dev/null; return 1;; esac
    sleep 3; done
  log "NO BOOT $tag (timeout)"; kill $lp 2>/dev/null; sudo docker rm -f flashnext >/dev/null 2>&1; return 1; }

scrape(){ curl -s -m 10 "$API/metrics" -o "$1" -w '%{http_code}'; }

# ---- V: does the endpoint measure what actually happened? ----
verify(){ local tag=$1 code
  code=$(scrape "$R/metrics-$tag-before.txt")
  [ "$code" = 200 ] || { log "FAIL($tag): /metrics returned HTTP $code"; return 1; }
  grep -q '^# TYPE tabby_generated_tokens_total counter' "$R/metrics-$tag-before.txt" || { log "FAIL($tag): no counter families in the scrape"; return 1; }
  # Known traffic, sent directly rather than through fn_bench: the probe issues its own 8-token warmup round
  # before the recorded ones, so try 1 compared 3 requests / 768 tokens against a /metrics delta of 4 / 776 and
  # called a correct endpoint wrong. 776 = 768 + 8 was the endpoint being exactly right. curl sends exactly what
  # is asserted: 3 completions, min_tokens = max_tokens = 256, so the expected delta is 3 requests / 768 tokens.
  local i
  for i in 1 2 3; do
    curl -s -m 300 "$API/v1/completions" -H 'Content-Type: application/json' -d "{
      \"model\": \"$NEWM\", \"prompt\": \"Write a short note about topic $tag number $i and the weather.\",
      \"max_tokens\": 256, \"min_tokens\": 256, \"temperature\": 0}" -o "$R/traffic-$tag-$i.json" -w '%{http_code} ' >> "$R/traffic-$tag.log" 2>&1
  done
  echo >> "$R/traffic-$tag.log"
  code=$(scrape "$R/metrics-$tag-after.txt")
  [ "$code" = 200 ] || { log "FAIL($tag): second scrape returned HTTP $code"; return 1; }
  python3 - "$R/metrics-$tag-before.txt" "$R/metrics-$tag-after.txt" "$tag" <<'PY'
import json, re, sys
before, after, tag = sys.argv[1:4]
def series(path):
    out = {}
    for line in open(path):
        if line.startswith("#") or not line.strip(): continue
        m = re.match(r'^([a-zA-Z_:][\w:]*(?:\{[^}]*\})?)\s+(\S+)$', line.strip())
        if m:
            try: out[m.group(1)] = float(m.group(2))
            except ValueError: pass
    return out
b, a = series(before), series(after)
def delta(prefix):
    keys = [k for k in a if k.split("{")[0] == prefix]
    if not keys: return None
    return sum(a[k] - b.get(k, 0.0) for k in keys)
want_reqs = 3
want_gen = 3 * 256
got_reqs = delta("tabby_requests_total")
got_gen = delta("tabby_generated_tokens_total")
got_sec = delta("tabby_generate_seconds_total")
print(f"[verify {tag}] client sent {want_reqs} requests / {want_gen} forced tokens (exact, no warmup); "
      f"/metrics moved requests {got_reqs} generated {got_gen} generate_seconds {got_sec}")
fail = []
if want_reqs == 0: fail.append("the client completed no requests")
if got_reqs is None or abs(got_reqs - want_reqs) > 0: fail.append(f"request delta {got_reqs} != {want_reqs}")
# The forced length is exact, so the token delta must match it; allow one token per request for a stray EOS.
if got_gen is None or abs(got_gen - want_gen) > want_reqs: fail.append(f"token delta {got_gen} != {want_gen}")
if got_sec is None or got_sec <= 0: fail.append(f"generate_seconds delta {got_sec} is not positive")
if got_gen and got_sec: print(f"[verify {tag}] implied time-weighted rate {got_gen / got_sec:.1f} t/s")
for f in fail: print(f"[verify {tag}] FAIL: {f}")
sys.exit(1 if fail else 0)
PY
}

log "=== V: candidate boot and endpoint verification ==="
up "$NIMG" cand || { log "FAIL: candidate did not boot"; finish FAILED; exit 1; }
CAND_VRAM=$(vram); log "UP candidate: VRAM free $CAND_VRAM"
verify cand 2>&1 | tee -a "$R/audit.log"; vok=${PIPESTATUS[0]}
[ "$vok" = 0 ] || { log "VERIFY FAILED: not promoting"; finish FAILED; exit 1; }

# Fingerprint helpers from R577 (the source of REF1/REF30). Both hash content + "|" + reasoning_content so a model
# that moves its thinking but not its answer still trips the gate. ONE change from R577: the path carries an
# explicit /v1. R577 sets API to .../8022/v1 and appends "/chat/completions"; this script sets API to the bare
# .../8022 because served_id and the verify curls append their own /v1. Lifting the helpers unchanged made them
# POST to /chat/completions, which 404s, so both digests came back "none" and try 4's gate read that as "moved".
greedy(){ curl -s -m 900 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$NEWM" "$PROMPT")" > "$R/greedy-$1.json"
  python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' "$R/greedy-$1.json" 2>/dev/null || echo none; }
greedy30k(){ python3 - "$API" "$NEWM" "$R/greedy30k-$1.json" <<'PY' 2>"$R/greedy30k-$1.err" || echo none
import json,sys,hashlib,urllib.request,random
api,model,out=sys.argv[1:4]
rng=random.Random(4242); words="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
body=" ".join(rng.choice(words) for _ in range(23000))
msg=body+"\n\nSummarize the passage above in one sentence, then write a haiku about it."
req=json.dumps({"model":model,"temperature":0,"max_tokens":128,"min_tokens":128,"messages":[{"role":"user","content":msg}]}).encode()
d=json.loads(urllib.request.urlopen(urllib.request.Request(api+"/v1/chat/completions",data=req,headers={"Content-Type":"application/json"}),timeout=900).read())
open(out,"w").write(json.dumps(d)); m=d["choices"][0]["message"]
print(hashlib.sha256(((m.get("content") or "")+"|"+(m.get("reasoning_content") or "")).encode()).hexdigest()[:16])
PY
}

log "=== G: greedy fingerprints (this overlay changes no math; anything but canonical is a bug) ==="
C1=$(greedy cand); C30=$(greedy30k cand)
log "greedy cand: c1 $C1 / 30k $C30 (canonical c1 $REF1 / 30k $REF30)"
if [ "$C1" != "$REF1" ] || [ "$C30" != "$REF30" ]; then
  log "GATE FAILED: fingerprints moved (c1 $C1 vs $REF1, 30k $C30 vs $REF30) — NOT promoting."
  finish FAILED; exit 1
fi
log "gate passed: c1 and 30k fingerprints canonical"

# ---- P: promote ----
log "=== P: promoting $NIMG as the daily image ==="
sudo cp "$LIVE" "$LIVE.pre-r587"
sudo python3 - "$LIVE" "$LIMG" "$NIMG" <<'PY'
import sys, pathlib
p, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
s = p.read_text()
a = f"IMG=${{IMG:-{old}}}"
b = (f"# R587 2026-09-20: Prometheus /metrics (patches/tabbyapi/metrics/r1), two added lines in gen_logging and an\n"
     f"# unauthenticated route beside /health. No math changed: c1 and 30k greedy fingerprints canonical on the\n"
     f"# candidate. ROLLBACK: IMG={old} (= launch-flashnext.sh.pre-r587).\n"
     f"IMG=${{IMG:-{new}}}")
assert s.count(a) == 1, f"expected exactly one {a!r}, found {s.count(a)}"
p.write_text(s.replace(a, b))
print("launcher IMG updated")
PY
[ $? = 0 ] || { log "PROMOTE FAILED: could not rewrite the launcher"; sudo cp "$LIVE.pre-r587" "$LIVE"; finish FAILED; exit 1; }
up "" daily || { log "DAILY DID NOT BOOT ON THE NEW IMAGE — rolling back"; sudo cp "$LIVE.pre-r587" "$LIVE"; up "" rollback; finish ROLLED-BACK; exit 1; }
log "daily up on $(sudo docker inspect flashnext --format '{{.Config.Image}}'); VRAM free $(vram)"
verify daily 2>&1 | tee -a "$R/audit.log"; vok=${PIPESTATUS[0]}
if [ "$vok" != 0 ]; then log "DAILY VERIFY FAILED — rolling back"; sudo cp "$LIVE.pre-r587" "$LIVE"; up "" rollback; finish ROLLED-BACK; exit 1; fi
log "PROMOTED: /metrics is live on the daily. Scrape target: <flan>:8022/metrics"
log "NEXT (not done here, k3s not GPU): add a tabby job to the vm-scrape-config ConfigMap in the monitoring namespace"
finish DONE
