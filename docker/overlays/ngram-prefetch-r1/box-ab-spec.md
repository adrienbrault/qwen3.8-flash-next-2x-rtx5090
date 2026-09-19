# ngram-prefetch r1 — box build, timing instrument runs, fingerprints, A/B and tier

Run on flan under the normal GPU queue: source `/srv/qwen5090/lib/gpu-queue.sh` before the flock
and chain the steps without restoring the daily in between (`flan/OPERATIONS.md` §12). All output
goes under `R=/srv/qwen5090/results/<date>-rNNN-ngram-prefetch/`, full raw output, never grep-only.
Probe helpers (`greedy`, `greedy30k`, `served_id`, `cfgline`, `vram`, the decode aggregation) are
the ones in `flan/r540-promote-r6.sh`; reuse them.

What this round is: one default-off env flag `EXL3_NGRAM_PREFETCH2=1` that (a) starts the PLE row
gather of every decode/verify forward as soon as its id history exists (right after the MTP draft
readback, instead of staging inline at the verify forward's start — the served stock path declines
prefetch for decode-sized inputs, so the whole gather is exposed today), and (b) makes the
prefill staging acquire its buffers on the worker thread, so issuing A_(i+1) never waits on
chunk i's cold gather or its forward's still-in-flight H2D uploads. Plus a timing instrument,
`EXL3_NGRAM_TIMING=1` (also on implicitly with PREFETCH2): per PLE forward, the time the forward
spends waiting for its rows (prefetch `future.result()` or inline `_stage`), split prefill vs
decode, aggregated by `NGramEmbedding.timing_report()` and printed every
`EXL3_NGRAM_TIMING_INTERVAL` (default 30 s) seconds by the generator.

The change is timing-only by construction (the staged set is consumed only when its history
matches what `forward()` builds; a mismatch is discarded). The flag-on arm must reproduce the
canonical fingerprints; any difference is a failure, not a quality question.

Variables:

```bash
SRC=/srv/qwen5090/overlay-src/ngram-prefetch-r1          # rsync of this directory
LIVE=/srv/qwen5090/launch-flashnext.sh
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")      # tabbyapi:nvme-tier-r4-e3det-r6-rawk
NIMG="$LIMG-ngramprefetch"
C1=e7fb377c987d685c; C30=4a255910dee2d9c5                # canonical fingerprints (check the live launcher's round)
```

Time budget: boot ~1 min; the 15-minute A/B cap is steps 5–8 (one fingerprint boot pair, two
prefill runs, one `mp_decode` boot each, compare) with the timing-instrument boot first.

## 1. Build (before the lock)

```bash
cd "$SRC" && sudo docker build -f Dockerfile.box --build-arg BASE="$LIMG" -t "$NIMG" . > "$R/build.log" 2>&1
```

The patch is Python only (four files, no extension rebuild). The build aborts on a patch hash
mismatch, a touched file that differs from its pinned baseline, a non-zero `patch -p1 --fuzz=0`
exit or a `.rej`, a post-patch hash mismatch, or a selector that does not read `False` unset /
`True` with `EXL3_NGRAM_PREFETCH2=1` / `EXL3_NGRAM_TIMING=1`. Keep `build.log`.

## 2. In-image runtime harness (lock held, daily down, ~1 min)

```bash
for G in 0 1; do
  sudo docker run --rm --gpus all --entrypoint python3 -v "$R":/out "$NIMG" \
    /opt/ngram-prefetch-r1/tests/gpu_ngram_prefetch.py --device cuda:$G --json /out/gpu$G.json \
    > "$R/harness-gpu$G.log" 2>&1
  echo "gpu$G exit $?"
done
```

Both must exit 0. The harness builds a synthetic trellis table in VRAM (no real 18.5 GiB table
needed) and checks, with the real dequant kernel and real CUDA events: forward == torch
reference; stock prefetch consumed exact; PREFETCH2 deferred staging (two sets queued back to
back, consumed out of queue order) exact; event reuse across re-staging exact; the verify-window
hook exact and consumed (stats "hit"); discard changes nothing; the timing report present. The
harness points the n-gram ext functions at its bundled faithful port (`fake_ext.py`) so the run
is independent of the checkpoint; if a reviewer prefers the real table, that is step 4 anyway.
This is the first real run of the deferred-acquire path against CUDA events: a failure here is a
boot blocker caught before serving.

## 3. Flag-off serving identity (G1)

Candidate launcher: the live launcher with `$LIMG` replaced by `$NIMG` in both places (IMG
default and the tier-default condition), as `r540-promote-r6.sh` builds it. With a non-served
image name the launcher attaches no NVMe tier, which keeps the daily's tier namespace untouched
during the whole experiment.

Boot it with `EXTRA_ENV` unchanged (selectors unset) at pool 983,040. Check `cfgline`
(`cache_size: 983040 cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] ... chunk_size: 2048
vision: true`) and the container env. `greedy boot` must equal `$C1` and `greedy30k boot` must
equal `$C30`. Record per-card free VRAM (expected: unchanged — the staging sets already exist in
the stock path; only the timing list, ~4096 floats worst case, is new). Keep the boot log. This
is the reference for every later arm.

## 4. Timing instrument on the flag-off arm (no serving change, ~2 min)

Restart the step-3 boot (same launcher) with
`EXTRA_ENV="<live env> EXL3_NGRAM_TIMING=1 EXL3_NGRAM_TIMING_INTERVAL=10"`.

1. Fingerprints again (`greedy` = `$C1`, `greedy30k` = `$C30`): the instrument alone must not
   move output.
2. Cold prefill at the 60k fn_bench target with `--salt`:
   `fn_bench.py --kind prose --tokens 64 --conc 1 --runs 1 --ctx 60000 --unique --salt $S1`.
   After the run, `sudo docker logs $NAME | grep "ngram timing"` — record the last summary
   lines: the `prefill exposed wait` (per-call wait inside the PLE forward, ms) is the number
   step 1 of the task is after. Expect roughly 0 today on the daily because the stock prefetch
   stages chunk k+1 before it runs; the p90 tail is the interesting cell (page-cache misses).
3. c1 (one stream, 512 tokens) and c4 (`--conc 4`) decode rounds the same way; the
   `decode/verify exposed wait` cell is the exposed gather the PREFETCH2 verify hook targets.
   R465 measured 0.72 ms/step mean (p90 1.66) at c4 d3 — this boot re-measures it on the
   current stack, which is what this overlay targets.

## 5. Flag-on fingerprints (G1 for the flag)

Flag-on arm = the step-3 candidate with `EXTRA_ENV="<live env> EXL3_NGRAM_PREFETCH2=1"`.
Boot, `greedy` = `$C1` and `greedy30k` = `$C30`, per-card VRAM. A difference stops the round:
the staged-set consumption is history-matched, so any output change is a bug, not a tuning
question.

## 6. Cold prefill A/B (flag off vs on, ~4 min)

Four runs, interleaved OFF/ON, at the 60k and 120k fn_bench targets, each with its own salt
(R507 lesson: `--unique` alone shares filler seeds across runs):

```bash
for arm in OFF ON; do
  FLAG=""; [ "$arm" = ON ] && FLAG="EXL3_NGRAM_PREFETCH2=1"
  boot step3/step5 launcher with EXTRA_ENV="<live env> $FLAG"
  fn_bench.py --kind prose --tokens 64 --conc 1 --runs 1 --ctx 60000  --unique --salt $S --out "$R/prefill_$arm.jsonl"
  fn_bench.py --kind prose --tokens 64 --conc 1 --runs 1 --ctx 120000 --unique --salt $S2 --out "$R/prefill_$arm.jsonl"
done
```

R484 measured the RAM ceiling at +6 % cold prefill on a 22.6k prompt and +1–2 % decode; R534's
daily reads 9,330 / 9,864 t/s (60k / 120k). Success here is prefill ≥ flag-off on both sizes
with the mechanism visible in the timing instrument's prefill cell shrinking (the chunk k+1
gather already overlaps block 0 in the stock pipeline, so the expected prefill gain is the p90
tail only — if the instrument shows the prefill wait already ~0 at p90, record it as measured:
it means prefill has nothing left to hide and the decode cell is the whole value).

## 7. mp_decode A/B (flag off vs on, ~6 min)

One boot per arm, the R545 pattern:

```bash
# OFF boot:  mp_decode.py run --url ... --tag A1 --tokens 512 --out "$R/mp.jsonl"   (24 code + 24 prose, c1 then c4)
# ON boot:   mp_decode.py run --url ... --tag B1 --tokens 512 --out "$R/mp.jsonl"
mp_decode.py compare --a A --b B "$R/mp.jsonl"
```

Expected: decode_tps at c1 and c4 moves by up to the exposed decode wait share of the step
(0.72 ms of a 14 ms c1 step / 22 ms c4 step ⇒ up to ~+5 % c1, ~+3 % c4 at the mean; the measured
expectation is less, because the verify gather is ~16 rows warm). A regression beyond −1 %
fails the round.

## 8. Timing instrument on the flag-on arm (~1 min)

Restart the ON boot with `EXL3_NGRAM_TIMING=1 EXL3_NGRAM_TIMING_INTERVAL=10` and repeat one c1
and one c4 round. The `decode/verify exposed wait` cell should now be near zero (the gather ran
during the draft chain / batch assembly) — this is the direct evidence the mechanism works. Save
both logs.

## 9. NVMe tier compatibility (flag on, scratch directory, ~2 min)

`T=/srv/qwen5090/fast/exl3-nvme-ngramprefetch-test`, `NVME_TIER=$T NVME_TIER_GB=16`.

1. Boot flag on with the tier at pool 983,040. `engine_identity()` (unchanged served code)
   already lists every EXL3_* variable in the namespace, so this arm's namespace is new; the
   first 30k request must cold-prefill and equal `$C30`.
2. Restart the same arm with the same directory; the tier must restore (`restored_pages > 0`)
   and the same request must equal `$C30`.
3. Boot flag OFF with the same directory: the log must show the flag-on namespace removed at
   open (`stale namespaces removed` ≥ 1), no restore, `greedy30k` = `$C30`.
4. `sudo rm -rf "$T"`.

## 10. Gates and end of chain

- G0 fingerprints: steps 3/4/5 all equal `$C1` / `$C30` (byte-identity is the quality gate).
- G2 agentic-edit smoke on the ON arm: 4 × `6/6 ok` (greedy/sample, c1/c4) — batched verify and
  requeue paths run the PREFETCH2 hook under real stop-string rewinds.
- Long greedy identity ON vs OFF, both at pool 983,040: the 30k fingerprint prompt with
  `max_tokens 2048, min_tokens 2048`, byte-identical; plus one 120k-needle request 5/5.
- Restore the daily only if no other unit is queued. Write `$R/audit.log`: build result, harness
  pass per card, fingerprints per arm, timing cells (prefill/decode) OFF vs ON, prefill table,
  mp_decode table with paired CI, agentic-edit, tier sequence.

## Promotion notes

If the A/B holds (fingerprints identical, decode within −1 %, prefill ≥ 0) the promoted launcher
adds `EXL3_NGRAM_PREFETCH2=1` to `EXTRA_ENV` (image can stay `$NIMG` or the served image with
the overlay applied); fingerprints stay `$C1`/`$C30`. The env change opens a new tier namespace
(the daily's tier restarts cold once). If the instrument shows the decode cell already ~0
(i.e. nothing exposed) or the prefill p90 already ~0, do not promote: the mechanism has nothing
left to hide at the served shapes and the result is a no-op record in FINDINGS.
