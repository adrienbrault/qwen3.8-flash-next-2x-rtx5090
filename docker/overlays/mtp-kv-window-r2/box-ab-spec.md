# mtp-kv-window r1: box A/B spec (at most 30 min of GPU)

Budget by branch. Moved placement at S0: S1 with 4 boots (15 min) + S2 with ≤ 6 boots (12 min) + tier check (3 min) = 30 min; S3 does not run. Normal placement: S1 with 3 boots (about 12 min) + S2, which is expected to stop at its first step (about 3 min) + S3 (6 min) + tier check (3 min) = about 24 min.

Anchor: the daily's shape since R561, 8 slots at pool 966,656 (59 × 16,384), cache 8,8, split [30, 30], draft policy `[[4, 3], [8, 1]]`, NVMe tier OFF for every A/B step. The overlay is pinned to `tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16` (the R548 image). It does not install on the R565 image `tabbyapi:ngram-prefetch-r1-gdnbf16`: that overlay also patches `generator/generator.py`, so `install.py` stops on the baseline hash (see impl-status.md, "Stacking"). Every arm below runs in the new image. The control arm is the same image with the flag unset, so `EXL3_NGRAM_PREFETCH2` is absent from all arms. Before a promotion the patch has to be rebased onto the R565 generator and this spec's S1 has to be repeated there.

## Build

```
docker build -f Dockerfile.box --build-arg BASE=tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16 -t tabbyapi:mtpwin-r1 .
```

The build context is `out/mtp-kv-window-r1/`. Pass: the last RUN line prints `mtp-kv-window r1 imports through <ext path>`. If `install.py` exits non-zero, the base is not the pinned image.

## Arms

| arm | env added to the R548 EXTRA_ENV |
| --- | --- |
| OFF | none (flag unset) |
| W4 | `EXL3_MTP_KV_WINDOW=4096` (17 pages per slot) |
| W16 | `EXL3_MTP_KV_WINDOW=16384` (65 pages per slot) |

Launcher: the live launcher with `IMG=tabbyapi:mtpwin-r1 MAXBS=8 CACHE=966656`, tier unset, and EXTRA_ENV = the R548 list plus the arm's variable. Keep one results directory per unit, for example `/srv/qwen5090/results/<date>-rNNN-mtp-kv-window/`, with full raw outputs.

## S0: boot checks (every boot of every arm)

1. The container log must contain one line from the draft cache: ` -- EXL3_MTP_KV_WINDOW: draft cache 8 slots x 17 pages = 34816 tokens (requested 966656)` for W4, or `x 65 pages = 133120 tokens` for W16. OFF prints nothing.
   - If it says `16 slots`, TabbyAPI builds the draft Cache without `max_batch_size`. The window pool is then twice as large (69,632 or 266,240 tokens) and the arm is still valid. Record it and continue.
   - `EXL3_MTP_KV_WINDOW: the draft cache has N window slots but max_batch_size is 8`: STOP. TabbyAPI builds the draft Cache with fewer slots than the generator uses.
2. Record free VRAM per card at boot, read the same way as the R546/R548 ladders, and record where layer 23 sits. There are two possible outcomes, and the spec handles both.
   - **Normal placement** (layer 23 stays on cuda:1). W4 frees +1,041 MiB on cuda:0 and W16 frees +932 MiB, both against OFF. cuda:1 stays within ±16 MiB, because the MTP layer and its cache live on cuda:0 (R539 census).
   - **Moved placement** (layer 23 lands on cuda:0). This is the R543/R544 signature: cuda:0 free falls and cuda:1 free rises by roughly a layer's weights plus its 1.08 GiB of cache. It is the likely outcome if TabbyAPI loads the draft model (and its cache) before the main model's layer split is decided: R543 moved the layer at an unchanged [30, 30] split with less memory freed on cuda:0 than this window frees. If the draft model loads after the main model, placement cannot move, and cuda:0 simply keeps the freed memory idle.
   - Either way, this is not a failure. It decides which control S1 compares against.

## S1: same pool 966,656 (OFF, W4, W16, plus one control boot if placement moved; about 15 min)

0. **Placement control (only if S0 found moved placement).** Boot OFF with layer 23 on cuda:0, the way R544b did: lower the OFF pool in 16,384 steps until the layer moves, or force it with the split. Record its c1 fingerprint (the moved-placement fingerprint; R544b showed placement, not pool, sets it). S1.1 compares moved W arms against this control, never against OFF at normal placement.
1. **Fingerprints.** Run the canonical c1 request. Its prompt plus 2,048 forced tokens stays inside one ring cycle of both windows (4,352 and 16,640 tokens), so the draft layer reads the same K/V in every arm, drafts the same tokens and verifies with the same layout.
   - The c1 fingerprint must equal the OFF boot at the same placement (OFF itself, or the S1.0 control). A difference means STOP: this is the check that catches a wrong block table or a stray write into the main cache.
   - The 30k fingerprint is logged but has no pass rule. Beyond one cycle the drafts differ, acceptance differs, and so does the verify layout (which positions share a verify forward). This engine's greedy text depends on that layout: R537 and R542 fingerprints changed with the draft depth.
2. **Acceptance and decode at short context.** Run `mp_decode.py run --tokens 512 --n 24` (c1 + c4, 24 code and 24 prose prompts, all shorter than one cycle). Use the box copy's c8 mode if it has one. Otherwise add `fn_bench.py --conc 8 --distinct --kind code --tokens 1024 --runs 2 --warmup-runs 1`.
   - Record acceptance from the TabbyAPI per-request draft stats in the container log (accepted and rejected draft tokens), plus tokens per SSE frame from the JSONL.
   - Short-context acceptance must match OFF. The draft input is identical there, so the only spread is the run-to-run noise that OFF shows against itself.
   - When placement moved, the decode rates compare a different layer split, so read them against the S1.0 control as well.
3. **Acceptance and decode at depth.** One cold `fn_bench.py --ctx 100000 --kind code --tokens 1024 --conc 1 --salt <boot>` request. Then one c8 round of 8 distinct prompts at `--ctx 16000` (about 130k resident, every sequence deeper than W4's cycle and close to W16's) with `--tokens 1024`.
   - Record the acceptance and the decode rate for each.
   - This is where the window acts. Expect W4 below W16, and W16 at or below OFF.
4. **Needles, W arms only.** Run G3 needles at about 131k and about 240k, and require 5/5 each. The window never changes the target's logits for a given verify forward. It can change which tokens get verified together, so this checks that deep-context answers still hold.

Pick W for S2. The default is W16. Take W4 only if its deep c8 acceptance is within 1 pp of W16's: the pool arithmetic is the same (see below), and W4 leaves 110 MiB more free on cuda:0.

## S2: pool ladder with the chosen W (≤ 6 boots, about 12 min)

Climb from 966,656 in +16,384 steps: 983,040, 999,424, 1,015,808, 1,032,192, 1,048,576. At each step, record:

- free VRAM per card at boot, and where layer 23 sits (placement can flip back to normal as the pool grows, because layer 23's cache share grows with it);
- survival: a cold 120k prefill, then a c1 → c8 ramp with one request per batch size, so every decode graph shape is captured fresh, then one c8 round;
- the c1 fingerprint (it must equal the OFF or control fingerprint of the same placement).

Headroom rule, placement-aware. The 32 MiB rule of R546/R548 compared each card with the same card of the live pool. It stood in for R548 try 2's graph-capture OOM, and it is meaningless once a layer moves (cuda:0 then legitimately holds about 1 GiB less free).

- The reference is the free VRAM of the **tightest card** of the OFF boot at 966,656, normal placement (S1's OFF arm). It is an absolute per-card floor and does not change with placement: do not switch it to the S1.0 control mid-run. Under moved placement cuda:0 nets roughly +1,041 MiB minus layer 23's weights and its 1,080 MiB of cache, so the first step may fail the floor on cuda:0. That is a legitimate k = 0, not a spec defect.
- Every step must keep **both** cards at or above that reference minus 32 MiB at boot, **and** pass the survival run.
- Stop at the first failing step, or at a boot failure ("Insufficient VRAM in split").
- Report k as the number of steps above 966,656 that pass, and name the placement of the top step.

Expected:

- **Normal placement at every step: k = 0.** Every pool step costs cuda:1 7 layers × 16,384 × 1,172 B = 128.2 MiB, and the flag frees nothing on cuda:1. cuda:1 bound the pool at every earlier boot of this model (R539, R541, R546, R548), so the first step fails.
- **Moved placement:** each step costs 6 layers × 16,384 × 1,172 B = 109.9 MiB per card. cuda:1 gains layer 23's weights and cache, and cuda:0 pays for them out of the window's ~1 GiB plus its previous idle memory. The equal-bytes figure is +4 steps (1,032,192); the ladder shows how much of it the tighter card actually allows.
- Moved placement is the only way the window turns into pool. It changes greedy text (R544b), so a pool won this way needs new canonical fingerprints and the full promotion gates.

## S3: forced placement move (only if S0 found normal placement and S2 gives k = 0; about 6 min)

If TabbyAPI loads the draft model after the main model, the freed cuda:0 memory never reaches the main model's split, and placement cannot move by itself.

1. Flag on (chosen W). Use a split with more budget on cuda:0 than [30, 30], starting from [31, 29]. Confirm from the boot VRAM that layer 23 moved.
2. Control: flag OFF at the same split and 966,656. Record whether it boots, where its layers sit and its c1 fingerprint. This separates the window's contribution from the split change's.
3. The step 1 boot's c1 fingerprint must equal the step 2 control's (S1.1's rule at the moved placement).
4. Ladder up from 966,656 with S2's rules.

## Tier restore check (W arm, tier ON, about 3 min, run inside S1's W boot or the final boot)

This is the GT gate shape: prime a 30k request, restart the container, replay it, and also run it warm (same session) and cold (salted).

- Required: no traceback; the restore brings back the cached tokens (as R548 GT did); the window line appears; all three requests finish.
- Recorded: acceptance of the restored request against warm and cold. The window cannot rebuild draft K/V for a restored span: there is no target state for tokens that were never prefilled, and the tier stores main pages only. Pages the slot does not hold come back as zeros (impl-status.md, "restored prefixes"), so expect lower acceptance on the restored request until new tokens fill the window.
- Not a pass rule: text identity between warm, cold and restored at 30k. With the window on, the draft content at 30k depends on the request's cache history, so the verify layout does too. Log the first differing character, as R544 did.

## Decision rule

**Pricing acceptance.**

- c8 runs depth 1: 8 jobs × 2 verify rows = 16 rows, and the step wall is about flat (R562: 20.05 ms). A job emits 1 + α tokens per step, where α is the acceptance rate of its one draft token. A drop of Δα costs Δα / (1 + α) of c8 throughput. At α ≈ 0.6 that is about 0.6 % per pp; the coordinator's "X pp costs about X %" is the upper bound.
- c1 and c4 run depth 3, where there is no closed form, so mp_decode's paired decode rate is the measurement.

**Gain.** Pool steps realised in S2 or S3 are k, each +1.69 % of the pool (16,384 / 966,656).

**Candidate for the full 8-boot AB and the promotion gates, only if all of these hold:**

| check | threshold |
| --- | --- |
| S0 | window line present; placement recorded |
| S1.1 | c1 fingerprint equal to the OFF boot at the same placement (OFF, or the S1.0 / S3 control) |
| short context: mp_decode c1/c4 paired geo-mean and fn_bench c8 aggregate | each ≥ −1 % (expected 0: identical draft input) |
| short-context acceptance | equal to OFF within OFF's own spread |
| deep context (100k c1, 8 × 16k c8): paired decode or the acceptance-priced c8 loss | ≤ 0.25 × (k × 1.69 %), i.e. at most a quarter of the pool gain |
| needles 131k / 240k | 5/5 each |
| pool gain | k ≥ 2 (≥ +3.4 %) |

With k = 0 (no passing step at any placement in S2 or S3) the round closes as no gain, whatever the acceptance.

**Against the zero-code alternative.** `draft_cache_mode: Q6` or `Q4` in TabbyAPI's draft_model section frees 1.68 % or 3.36 % of the cache bytes: equal-bytes pools of 983,040 (+1 step) and 999,424 (+2 steps). The window frees 7.2 to 8.0 % (+4 steps). Two things apply to both:

- Q6 and Q4 also free only cuda:0 bytes, so they meet the same placement limit.
- Q6 and Q4 break the standing rule that no KV cache goes below 8-bit (user, 2026-09-17), unless the user exempts the draft layer.

The window stacks with Q6 only on its own 39 to 149 MiB pool, so stacking them is not worth a separate arm.
