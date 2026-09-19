# Box A/B: NVMe prefix tier on the served chain (round 4), then round 3b's steps

## R4. Round 4 on `tabbyapi:mtp-pruned-r1-tc1-plefix` (about 27 minutes of GPU time; give the unit a 30+ minute timebox)

R526 try 5 passed every round-3b step on `stack-r4-e3r2` + ple-ckpt-clone r1, with the pruned-draft flags stripped. Round 4 runs the same tier on the served chain, with the daily's env unchanged. `impl-status.md` §R4 has the details. Four of the five tier files are byte-identical to R526 try 5. `generator.py` is mtp-pruned-r1's plus the same five tier hunks.

### Build (no GPU)

From this directory (`out/nvme-tier-r4/`, on the box under `overlay-src/nvme-tier-r4`):

```
sudo docker build -f Dockerfile.box --build-arg BASE=tabbyapi:mtp-pruned-r1-tc1-plefix -t tabbyapi:nvme-tier-r4 .
```

- The build prints `nvme-tier install: exllamav3/modules/ple.py: ple-ckpt-clone r1 (…) present (the tested file)`, then `stack-r4-e3r2+mtp-pruned-r1: 5 files installed`, then `nvme-tier smoke OK (stack-r4-e3r2+mtp-pruned-r1): …; ple stash fix present; …`. Any other variant name is a FAIL: the base is not the one you meant.
- Do not layer ple-ckpt-clone again; the base already has it. Building with no `BASE` (plain `stack-r4-e3r2`) now fails at `install.py` by design ("lacks ple-ckpt-clone r1").
- `launcher-nvme-tier.patch` is regenerated against the served launcher (repo `flan/launch-flashnext-tabby.sh` as of R530). It applies with `--fuzz=0`.

### Differences from the R526 unit

- **Do not strip** `EXL3_MTP_DEVICE_DRAFT=1 EXL3_EMBED_GPU=1 EXL3_EMBED_GPU_PRUNED=1` from `EXTRA_ENV`. Every boot uses the daily's `EXTRA_ENV` verbatim. The point of this run is the tier with the device draft chain on.
- `IMG=tabbyapi:nvme-tier-r4`, and fresh tier directories `/srv/qwen5090/fast/exl3-nvme-r4a` and `exl3-nvme-r4b`. The namespace differs from R526's anyway: `engine_identity` hashes every installed source.
- Drop the plefix layering step and the `case "$LIMG"` pruned/stack tolerance: the image is built on the live image.

### Steps

| # | min | step | pass condition |
|---|-----|------|----------------|
| 0 | 2 | OFF boot of `tabbyapi:nvme-tier-r4`, no `NVME_TIER` | no ` -- nvme tier` line. c1 and 30k fingerprints equal the daily's (canonical `e7fb377c987d685c` / `4a255910dee2d9c5`; if R530 recorded others, those). This is the "tier off = pruned-draft generator" gate on hardware; `test_merge_pruned.py` is its CPU counterpart |
| 1-10 | 14 | round 3b steps 1-10 below, with `IMG=tabbyapi:nvme-tier-r4` and the r4 directories | as below, with one change: step 2's fingerprints must equal step 0's |
| 11 | 10 | counterbalanced idle-tier decode: boots in the order daily image OFF, r4 ON (drained tier, dir r4a), r4 ON, daily image OFF | fn_bench code c1/c4, 2048 × 2 per boot. Mean ON vs mean OFF within −1.5 % at c1 and c4 (R526 measured −0.8 % / −1.9 % from one boot each, which is not enough to promote) |

Additional pass conditions for this run, because the device chain is new to the tier:
- Step 6 (`verify`): the restored output hash equals the pre-restart warm hash for both sizes, with the device chain on. Also compare with a cold run: R526 try 5 had restored = warm = cold.
- Step 8 (decode while a 120k chain drains): same bounds (−3 % c1 and c4).
- The docker log has no `EXL3_EMBED_GPU_PRUNED=1 inactive` warning on any boot. The pruned mirror must be active, as on the daily.

If step 0 fails, the tier is not involved yet, because `EXL3_NVME_TIER` is unset. Stop and send me both fingerprints plus the `sha256sum` of the installed `generator/generator.py`. It must be `b03ce7fe…`.

---

# Round 3b box A/B (rerun after R526, 14 minutes of GPU time)

This section is kept as run for R526. Its rebuild command below (no `BASE`) no longer builds, because the base must carry ple-ckpt-clone r1. Use §R4's build, and read `nvme-tier-r3` as `nvme-tier-r4` in paths.

R526 ran round 3. Fingerprints, drain, crash restart, idle decode and the churn cap all passed. Restore never hit (FAIL 1), a cold 30k produced an immediate EOS (FAIL 2), and decode during a drain lost 7.3 % at c1 (FAIL 3). `impl-status.md` §0 has the cause and the fixes. This spec is the rerun. Its steps are those of round 3, with added log lines, a FAIL 2 control and tighter pass conditions.

## Rebuild (no GPU)

From this directory (`out/nvme-tier-r3/`), same command and tag as R526:

```
sudo docker build -f Dockerfile.box -t tabbyapi:nvme-tier-r3 .
```

- The change is Python only; `install.py` and the build smoke test are unchanged.
- The build prints `nvme-tier smoke OK (stack-r4-e3r2)`.
- `manifest.json` pins the new overlay hashes.
- The launcher is your regenerated R525 patch (`NVME_TIER` / `NVME_TIER_GB`). Add nothing else: the new knobs below default to the values this spec wants.

Knobs, all optional; the defaults are what this spec measures:

| env | default | meaning |
|---|---|---|
| `EXL3_NVME_TIER_PUMP_PCT` | 1.0 | per-iteration copy-out budget, in % of the iteration's own duration (FAIL 3) |
| `EXL3_NVME_TIER_PUMP_PAGES` | 4 | at most this many pages per iteration; `0` = no per-iteration copies (idle drain only) |
| `EXL3_NVME_TIER_VERIFY` | 1 | 1: checkpoints are copied into a tier-owned buffer, checked for source changes, and read back after the write; 2: pages are read back too |
| `EXL3_NVME_TIER_SCAN` | 1 | at open, read and check every stored checkpoint in the background; prints ` -- nvme tier: open scan: ...` |
| `EXL3_NVME_TIER_LOG_LOOKUPS` | 1 | one ` -- nvme tier: lookup <outcome>: ...` line per allocation that found prompt pages on disk |

Host prep: same as R526, with fresh directories.

```
df -h /srv/qwen5090/fast
sudo rm -rf /srv/qwen5090/fast/exl3-nvme-r3a /srv/qwen5090/fast/exl3-nvme-r3b
R=/srv/qwen5090/results/$(date +%F)-nvme-tier-r3b; mkdir -p $R
R526=<the R526 results dir>        # holds R526's fill.json (the 30k prompt that ended in EOS)
T=out/nvme-tier-r3/tests/gpu_nvme_ab.py
```

`EXTRA_ENV` is the daily's R525 env, identical on every boot.

## Order and budget

| # | min | step | exact command | pass condition |
|---|-----|------|---------------|----------------|
| 1 | 1 | boot, tier ON, fresh dir | `IMG=tabbyapi:nvme-tier-r3 NVME_TIER=/srv/qwen5090/fast/exl3-nvme-r3a ./launch-flashnext-nvme.sh` | ` -- nvme tier: ON at /nvme-tier/ns-… (… 0 pages, 0 checkpoints …)` |
| 2 | 1 | fingerprints, cold, on the fresh boot | the R517/R526 G1 invocation | c1 `e7fb377c987d685c` and 30k `4a255910dee2d9c5` under R525 env, i.e. equal to the tier-OFF boot, as in R526 |
| 2b | 0.5 | FAIL 2 control, cold, tier on | `python3 $T first-token --out $R --fill $R526/fill.json --size 30000 --tag r3b-on-cold` | records the first token and top-5 logprobs of R526's 30k prompt. The comparison is at the end of the slot |
| 3 | 1.5 | fill 30k + 120k, cold then warm | `python3 $T fill --out $R --sizes 30000,120000` | warm cached > 0 |
| 4 | 0.5 | idle drain | `python3 $T wait-drained --out $R --container flashnext` | drained line with `errors 0`, `write-verify 0`, `stash-mutated 0`, and `pump N pages (x ms/page)`. `views-copied` > 0 is expected: the PLE id contexts are snapshotted at put |
| 5 | 1 | crash restart | `sudo docker rm -f flashnext && IMG=tabbyapi:nvme-tier-r3 NVME_TIER=/srv/qwen5090/fast/exl3-nvme-r3a ./launch-flashnext-nvme.sh` | the ON line has the step-4 counts and `0 torn`. **New:** within seconds, ` -- nvme tier: open scan: M/M checkpoints intact in … s`. Any `failed (…)` there is a FAIL, and its reason (`digest`, `header`, …) is the answer |
| 6 | 0.5 | restore from disk | `python3 $T verify --out $R` | `VERIFY PASS`: for both sizes the output hash equals the pre-restart warm hash and `cached_tokens` is within one page of the warm count (both ~99 %). The docker log shows ` -- nvme tier: lookup hit: …` for each request, and the next drained line reads `restored N pages + 2 ckpts … aborts 0 … read-fail 0` |
| 7 | 3 | idle decode | fn_bench code c1/c4, 2048 × 2 | within −1.5 % of the week's daily (R526: 219.6 / 504.0 vs OFF 219.0 / 497.3) |
| 8 | 2 | decode while a fresh 120k chain drains | `python3 $T decode --out $R --ntok 120000 --max-tokens 1024` | c1 during-drain chunks/s within −3 % of after-drain (R526: −7.3 %); c4 within −3 % |
| 9 | 1 | 2 GiB cap, fresh dir | `sudo docker rm -f flashnext && IMG=tabbyapi:nvme-tier-r3 NVME_TIER=/srv/qwen5090/fast/exl3-nvme-r3b NVME_TIER_GB=2 ./launch-flashnext-nvme.sh` | ON line with `cap 2.0 GiB` |
| 10 | 2.5 | churn | `python3 $T churn --out $R --dir /srv/qwen5090/fast/exl3-nvme-r3b --cap-gb 2 --n 5 --ntok 40000` | `CHURN … PASS` (as R526) |
| – | 0.5 | capture | `sudo docker logs flashnext > $R/docker-boot3.log 2>&1` (and the boot-2 log before step 9: `sudo docker logs flashnext > $R/docker-boot2.log 2>&1`) | – |

After the slot, on the restored daily (tier off, no extra boot):

```
python3 $T first-token --out $R --fill $R526/fill.json --size 30000 --tag daily-off-cold
```

**FAIL 2 verdict:** compare `first_token.jsonl` rows `r3b-on-cold` and `daily-off-cold`, plus R526's cold ("\n\n") and after-restart (EOS) outputs.

- If the top-2 logprob margin between "\n\n" and EOS is small (below about 0.5 nats) in either run, the EOS in R526 was a prefill-numerics near-tie (E3 is not bitwise deterministic, R524). It was not caused by the tier.
- If the tier-on cold run gives EOS with a large margin while the daily gives "\n\n", keep the tier off and send me both logs.

## Reading the new log lines

- `lookup <outcome>`, one of:
  - `hit`: restored.
  - `no-disk-ckpt`: pages on disk but no checkpoint on the prefix.
  - `ram-as-deep` / `in-ram`: RAM already had it.
  - `load-read:<check>` / `load-decode:<exc>`: the checkpoint read or decode failed, and `<check>` names the failed test.
  - `position a!=b`: the stored state is at another position.
  - `put-dropped`.
- `restore aborted (…)`: a page read failed mid-restore. The restored pages are reset and the disk checkpoint is dropped, so the job runs from VRAM/RAM only, cold for a restart.
- `quarantined ckpt|page <key> at segment s offset o: read check '<check>' failed twice`.
- `write verify failed for …`: the file did not hold the hashed bytes right after the write, which points at I/O.
- `checkpoint … changed while it was being serialized`: something wrote into a stash after `put`. The fix in this round should make this impossible; report it if it appears.

## What each gate proves

These are unchanged from round 3.

- **Step 2:** the tier adds only copies of finished pages. Pump copies are ordered on the compute stream, so the forward pass is unchanged.
- **Steps 4-5:** the drain completes before the kill, and the restart finds every record whole. The new open scan also reads every checkpoint payload.
- **Step 6:** bit-exact pages and checkpoints restore the same state as the warm VRAM hit.
  - The PLE id context is now snapshotted at `put` when the tier is on, in RAM and on disk alike. So the pre-restart warm request and the post-restart disk hit resume from the same state.
  - The R526 warm hashes are not comparable. They resumed from a RAM checkpoint whose PLE context was a live view of the slot (impl-status §0.2).
- **Step 8:** the pump is now budgeted by measured time.
- **Step 10:** the cap holds.

## FAIL handling

- Step 5 scan reports failures: stop, save both logs, and send me the failure reasons (`digest` means the payload bytes differ from the write-time hash).
- Step 6 `lookup load-…` or `restore aborted`: save the logs. The line names the failed check.
- Step 6 with cached tokens matching but the hash different: rerun `verify`. A disk hit should repeat its own hash, so report both hashes.
- Step 8 still below −3 %: rerun step 8 with `EXL3_NVME_TIER_PUMP_PCT=0.5`, or with `EXL3_NVME_TIER_PUMP_PAGES=0` (the bound: idle drain only).

## Optional second slot: stacked with recurrent-tip-r1

Unchanged from round 3: steps 1-7 with `IMG=tabbyapi:nvme-tier-r3-tip`, `EXTRA_ENV="<R525 env> EXL3_RECURRENT_TIP_STASH=1"` and a fresh dir. If the tip image carries the later PLE fix in `recurrent_tip.py` only, `install.py` notes the difference and installs. If tip's `generator.py` changed too, it refuses; send me the new file.
