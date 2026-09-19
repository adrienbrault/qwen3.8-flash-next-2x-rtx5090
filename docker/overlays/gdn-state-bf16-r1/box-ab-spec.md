# GDN state bf16 round 1 — box build, harness, ladder, A/B and gates

Run on flan under the normal GPU queue: source `/srv/qwen5090/lib/gpu-queue.sh` before the flock and chain the steps without restoring the daily in between (`flan/OPERATIONS.md` §12). All output goes under `R=/srv/qwen5090/results/<date>-rNNN-gdn-bf16/`, full raw output, never grep-only. No step below has been run. Probe scripts and helper functions (`greedy`, `greedy30k`, `served_id`, `cfgline`, `vram`, the decode aggregation) are the ones in `flan/r540-promote-r6.sh`; reuse them rather than rewriting them.

Variables:

```bash
SRC=/srv/qwen5090/overlay-src/gdn-state-bf16-r1          # rsync of flan/patches/exllamav3/gdn-state-bf16/r1
LIVE=/srv/qwen5090/launch-flashnext.sh
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")      # tabbyapi:nvme-tier-r4-e3det, or ...-r6 after R540
NIMG="$LIMG-gdnbf16"
MANIFEST=manifest.json; [ "$LIMG" = tabbyapi:nvme-tier-r4-e3det-r6 ] && MANIFEST=manifest-on-r6.json
C1=e7fb377c987d685c; C30=4a255910dee2d9c5                # live canonical fingerprints (check the live launcher's round)
```

## 1. Build (before the lock)

```bash
cd "$SRC" && sudo docker build -f Dockerfile.box --build-arg BASE="$LIMG" --build-arg MANIFEST="$MANIFEST" -t "$NIMG" . > "$R/build.log" 2>&1
sudo docker run --rm --entrypoint python3 "$NIMG" -c "import exllamav3.modules.gated_delta_net as g; from exllamav3.ext import exllamav3_ext as e; assert g._gdn_state_bf16 is False; assert hasattr(e, 'exl3_moe_prefill_e3_det') and hasattr(e, 'gr_mix_v2_int8'); print('ok')"
```

The build aborts on: patch hash mismatch; any touched file differing from its pinned baseline; a non-zero `patch -p1 --fuzz=0` exit or a `.rej`; a post-patch hash mismatch; a rebuilt `.so` outside site-packages or without `exl3_moe_prefill_e3_det`, `gr_mix_v2_int8`, `cuda_recurrent_gated_delta_rule`, `batched_state_rewind`; `_gdn_state_bf16` not `False` with the variable unset. This is the first nvcc compile of the round. Keep `build.log`. If R540 is live, also assert `gr_mix_v2_int8_regrid` in the second command.

## 2. Kernel harness on both cards (lock held, daily down)

The served image does not contain the harness, so mount the tests directory into it for the dump.

```bash
for G in 0 1; do
  sudo docker run --rm --gpus all --entrypoint python3 -v "$SRC/tests":/t:ro -v "$R":/out "$LIMG" \
    /t/gpu_gdn_bf16.py --device cuda:$G --dump /out/served-fp32-gpu$G.pt --json /out/dump-gpu$G.json
  sudo docker run --rm --gpus all --entrypoint python3 -v "$R":/out "$NIMG" \
    /opt/gdn-state-bf16-r1/tests/gpu_gdn_bf16.py --device cuda:$G --ref /out/served-fp32-gpu$G.pt --json /out/harness-gpu$G.json
done
```

All four runs must exit 0. The patched run fails (exit 1) on any of:
- `flag_off_identity_vs_served`: the fp32 path of the patched extension against the served image, `torch.equal` on outputs and all state planes: decode bsz 1/3/4 (v-split 4 and 1), verify q2/q4 with history, 8-token multi-token without history, the generic 64x64 kernel (decode, verify, multi-token), and the rewind job path with `EXL3_HOST_GAP_REWIND` off and on.
- `in_process_identity`: `gated_delta_rule_fn` chunked prefill vs a direct vendored fla call (fp32), fp32 rewind vs a torch copy. The cross-image comparison excludes the fla prefill because its Triton kernels autotune per process.
- `structure`: verify window == the same tokens decoded one at a time (fp32 and bf16, `torch.equal` of every intermediate state and output); an MTP schedule with random acceptance and rewinds through `GDNLayerState` jobs == sequential decode of the accepted tokens (fp32 and bf16); a bf16 rewind copies exactly one plane (sentinel planes and the next slot unchanged, both host-gap modes); allocation dtypes, `get_checkpoint_size` (fp32 = served formula, bf16 = half the state term); the import-time selector off when unset and on with `=1`.

The same JSON carries the numerics (informational):
- `decode_error`: 512 decode steps at 4 slots, bf16 vs fp32 state, at steps 1..512: per decay class (slow |g| 1e-4..1e-3, medium 1e-2..1e-1, fast 0.3..2) the relative Frobenius state error per (slot, head), the same ratio for rounding the fp32 state once (`state_rel_one_rounding`, the floor), their ratio (`state_growth_vs_one_rounding`), max abs state error, the core output relative error over the last 16 steps, and the minimum output cosine so far.
- `mtp_error`: 128 verify windows of 4 tokens with random acceptance and rewinds, bf16 vs fp32.
- `prefill_error`: a 32,768-token prefill in 2,048-token chunks through the served fla path (state rounded once per chunk), then 64 decode steps.
- `bench_us`: recurrent-kernel time per launch, fp32 vs bf16, bsz 1/4, q1 and q4-with-history. Direction only.

Stop and report before any boot if `out_cos_min_so_far` at step 512 is below 0.99 or the slow-class `out_rel_last16` max is above 0.05. Errors that size in one layer's core output compound across 36 layers. Record the numbers either way.

## 3. Flag-off serving identity (G1)

Candidate launcher: the live launcher with `$LIMG` replaced by `$NIMG` in both places (IMG default and the tier-default condition), exactly as `r540-promote-r6.sh` builds it. With a non-served image name the launcher attaches no NVMe tier, which keeps the daily's tier namespace untouched during the whole experiment.

Boot it with `EXTRA_ENV` unchanged (selector unset) at 819,200. Check `cfgline` (`cache_size: 819200 cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] ... chunk_size: 2048 vision: true`) and the container env. `greedy boot` must equal `$C1` and `greedy30k boot` must equal `$C30`. Any difference means the default-off path is not byte-identical: stop. Record `nvidia-smi --query-gpu=memory.used,memory.free` per card and keep the boot log (layer loading lines). This is the placement reference for step 4.

## 4. Flag-on boots and pool ladder

Flag-on arm = the step-3 candidate with `EXTRA_ENV="<live env> EXL3_GDN_STATE_BF16=1"`.

1. Boot flag-on at 819,200 twice (two containers, same pool). Record per-card used/free VRAM against step 3; the expected drop is about 864 MiB summed over both cards, split by where the GDN layers sit. Record both boots' `greedy` and `greedy30k` fingerprints. The two flag-on boots must agree with each other (the bf16 path is deterministic); they are expected to differ from `$C1`/`$C30`.
2. Compare the boot logs' layer placement and per-card usage with step 3. If a layer moved between cards, note it: the ladder and the A/B then compare different placements (the R489 confound, `gdnstate/r2/impl-status-gdn-state-r2.md`).
3. Ladder, flag on: `CACHE=$((819200 + k*16384))` for k = 1, 2, 3, ... until the first boot that fails (`Insufficient VRAM in split for model and cache`, a restart, or no answer). At every step that boots: `cfgline`, per-card free VRAM, `greedy` (must equal the flag-on fingerprint from 4.1), one cold 120,000-token prefill (`fn_bench.py --kind prose --tokens 64 --conc 1 --runs 1 --ctx 120000 --unique`), and one c4 decode round (`fn_bench.py --kind code --tokens 2048 --conc 4 --runs 1`) to prove runtime headroom. `POOL_ON` = the highest step that passes all of these. Arithmetic expectation (`impl-status.md`, VRAM): +3 steps = 868,352; +4 = 884,736 only if the limiting card had slack.
4. Control, flag off: one boot at `819200 + 16384`. R525 set 819,200 as the flag-off maximum; if this boots and passes the same checks, the flag-off ceiling moved and the ladder gain is measured from the new ceiling, not from 819,200.

## 5. A/B decode

Eight boots at 819,200, interleaved OFF/ON/OFF/ON/..., so the pool is the same in both arms. Per boot, the R540 decode block: `fn_bench.py --kind code|prose --tokens 2048 --warmup-runs 1 --conc 1 4 --runs 2`. Report per arm and cell (code/prose × c1/c4) the mean, and the paired ON−OFF difference with a 95 % CI across the four pairs, as in R538. Decode regressions beyond −3 % at c1 or c4 fail. One more flag-on boot at `POOL_ON` repeats the c1/c4 cells to check that the pool size does not change the decode rate.

The bf16 state halves the recurrent kernel's state traffic (two reads and one write of the state per token per layer), so a small decode gain is possible; `bench_us` from step 2 shows the direction at kernel level.

## 6. Quality gates (flag on, at POOL_ON)

Run in this order and stop at the first failure.

- G2 agentic-edit: `agentic-edit.py --tag GB --conc 1 4 --modes greedy sampled`; 4 × `6/6 ok`.
- G3 needles: `fn_needle_oai.py --ctx-tokens 30720 122880 --fracs 0.08 0.3 0.55 0.8 0.96` (30k/120k, the gate asked for here) and the R540 pair `--ctx-tokens 131072 240000` with the same fracs; `5/5 retrieved` on all four contexts.
- G4 tool-eval 69 × 4 (`tool-eval-bench --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8`): mean ≥ 82.0 (the R540 floor). Report it against the R530 boot spread 84.5–88.0; a result below 84.5 is a flag for a second boot before any promotion.
- G5 GSM8K n = 500 through `nostop_proxy.py` (R540 command, 5-shot, chat template, `temperature=0,max_gen_toks=8192`): flexible-extract ≥ 0.970 (R530 0.974). Keep `--log_samples` and compute the paired discordance (ON-correct/OFF-wrong vs ON-wrong/OFF-correct) against the flag-off samples; if no flag-off n = 500 samples from the served chain are kept, run G5 once on the step-3 flag-off boot as well.
- Long-context greedy divergence vs flag off. Three prompts, greedy, `temperature 0`: (i) the canonical `greedy30k` prompt with `max_tokens 1024, min_tokens 1024`; (ii) a 120k prose haystack from `fn_needle_oai.py` with its question, `max_tokens 512`; (iii) the R540 c1 code prompt with `max_tokens 2048, min_tokens 2048`. Run each on a flag-off boot (step 3 or a step-5 OFF boot) and on two flag-on boots. Save the full responses. Report, per prompt: the first differing character offset and word index ON vs OFF, `difflib.SequenceMatcher(None, on, off).ratio()`, and whether the final answer (needle value, code tests) is the same. Controls: OFF vs OFF across two boots must be identical (canonical fingerprints), ON vs ON across two boots must be identical. There is no pass threshold for the ON−OFF divergence itself: the change is not bit-exact, and G2–G5 decide quality. An ON answer that is wrong where OFF is right on (ii) or (iii) fails the round.

## 7. NVMe tier compatibility (flag on, scratch directory)

Use a fresh directory so the daily's tier is never opened: `T=/srv/qwen5090/fast/exl3-nvme-gdnbf16-test`, `NVME_TIER=$T NVME_TIER_GB=16`.

1. Boot flag on with the tier. Send the `greedy30k` prompt, wait for the tier to drain on idle (the ` -- nvme tier:` log line reports written checkpoints). Save the response.
2. Restart the same arm with the same directory. Send the same prompt. The tier must report a restore (`restored_checkpoints` > 0 / `lookup_hit` in the tier log or metrics), there must be no `TypeError` from `GDNLayerState.unstash` in the container log, and the response must equal step 1's.
3. Boot flag off with the same directory. The log must show `stale namespaces removed` ≥ 1 at open (the bf16 namespace is deleted), no restore, and `greedy30k` must equal `$C30`.
4. Boot flag on again with the same directory: again a new namespace, no hit on the first request.
5. `sudo rm -rf "$T"`.

The namespace separation comes from `engine_identity()` (the `EXL3_GDN_STATE_BF16=1` entry) and from `recurrent_layout[].checkpoint_bytes` (1,572,864 B + conv part vs 3,145,728 B + conv part for the state term per layer). Record the namespace digest from each boot's log.

## 8. End of chain

Restore the daily only if no other unit is queued (`daily-restore-retry.sh` / `gpu_queue_others`, OPERATIONS §12). Write `$R/audit.log` with: build result, harness pass/fail per card and the headline numerics (slow/medium/fast state and output error at 512 steps, MTP, 32k prefill, kernel µs), step-3 fingerprints, flag-on fingerprints, per-card VRAM for both arms, `POOL_ON`, the A/B table, G2–G5 results, the divergence table and the tier sequence.

## Promotion notes

The flag-on arm cannot pass R540's G1 (byte identity with the live daily); step 3 replaces it for the default-off image, and G2–G5 plus the divergence controls decide the flag. A promoted launcher would carry `IMG=$NIMG`, `EXL3_GDN_STATE_BF16=1` in `EXTRA_ENV` and `CACHE=$POOL_ON`; its flag-on `greedy`/`greedy30k` become the new canonical fingerprints. Promotion opens a new tier namespace (the image sources and the env both change), so the daily's tier restarts cold. Rollback = the previous launcher; its image and namespace are unchanged, but its tier directory was GC'd by the promoted boot and restarts cold as well. If deeper MTP is tried next, each extra draft depth costs 216 MiB of recurrent state with the flag on instead of 432 MiB.
