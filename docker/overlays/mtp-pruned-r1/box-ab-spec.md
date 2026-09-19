# Box A/B spec: MTP device draft chain on the pruned embedding mirror, round 1

**Image:** `tabbyapi:mtp-pruned-r1`. Build it before the timed window: `sudo docker build -t tabbyapi:mtp-pruned-r1 -f Dockerfile.box .`, with this directory as the build context. The overlay is pure Python, so there is no extension rebuild. The build fails if the base is not `stack-r4-e3r2`, byte for byte, for the three changed files.

**Arms:**
- OFF runs the new image with the daily `EXTRA_ENV` unchanged. This is R517: `EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1 EXL3_DRAFT_PINNED_STAGING=1 EXL3_BATCH_VERIFY=1 EXL3_MTP_HEAD_N=65536 EXL3_MOE_PREFILL_E3=1`. This arm also checks that the new image's default path is byte-identical to the daily.
- ON uses the same image and the same `EXTRA_ENV` plus `EXL3_MTP_DEVICE_DRAFT=1 EXL3_EMBED_GPU=1 EXL3_EMBED_GPU_PRUNED=1`.

Everything else follows launcher-r517: pool 786,432 @ 8,8, 4 slots, split [30, 30], policy [[4,3],[8,1]], vision on and port 8022. Use a non-daily port if the launcher allows it.

**Time budget:** about 13 minutes of GPU time, boots included. The steps are ordered so that an identity failure stops the run before any served boot.

| # | step | minutes | gate |
|---|---|---|---|
| 0 | Stop the served container. | 0.2 | — |
| 1 | Standalone GPU tests in the new image (one `docker run`, below): `gpu_unit_pruned_embed.py`, then `gpu_chain_identity.py`. | 3.5 | G0 |
| 2 | ON arm: boot, then check the boot log and VRAM. | 1.2 | G1 |
| 3 | ON arm: fingerprints, c1 greedy then 30k. | 0.4 | G2 |
| 4 | ON arm: fn_bench, in the order c1 code, c1 prose, c4 code, c4 prose. 2,048 forced tokens each, two runs per shape. | 2.0 | report |
| 5 | ON arm: c4 stress pass, then VRAM free and a log grep. | 0.5 | G4 |
| 6 | OFF arm: repeat steps 2-5 with the flags unset. | 4.1 | G1-G4 |
| 7 | Restore the daily with the daily launcher. This falls outside the window. | — | — |

If G0 fails, skip steps 2-6 and send back the JSONs. The daily restore costs one boot.

## Step 1: standalone tests (served container stopped)

```
R=/srv/qwen5090/results/$(date +%F)-mtp-pruned-r1; mkdir -p $R
sudo docker run --rm --gpus all --ipc=host --shm-size=16g \
  -e EXL3_HOST_GAP_REWIND=1 -e EXL3_HC_MIX_V2=1 -e EXL3_HC_MIX_V2_MIN_R=1 -e EXL3_LS_PREFILL_PIPELINE=1 \
  -e EXL3_MOE_COOP_V2=1 -e EXL3_SHARED_EXPERT_OVERLAP=1 -e EXL3_DRAFT_PINNED_STAGING=1 -e EXL3_BATCH_VERIFY=1 \
  -e EXL3_MTP_HEAD_N=65536 -e EXL3_MOE_PREFILL_E3=1 \
  -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
  -v /srv/qwen5090/models:/models:ro -v $R:/out \
  --entrypoint sh tabbyapi:mtp-pruned-r1 -c \
  "python3 /opt/mtp-pruned-r1/tests/gpu_unit_pruned_embed.py --out /out/gpu-unit.json 2>&1 | tee /out/gpu-unit.log && \
   python3 /opt/mtp-pruned-r1/tests/gpu_chain_identity.py --out /out 2>&1 | tee /out/chain.log"
```

The tune-cache mount is the launcher's `TUNEDIR` (`/srv/qwen5090/.exl3cache`, launcher line 217). Without it the first forward re-tunes kernels and eats into the budget. Do not set the three new flags here. `gpu_chain_identity.py` selects each arm itself by setting the module globals, and each arm gets a fresh Generator. The script's defaults are the 2.50bpw checkpoint and a cache of 131,072 tokens. The smaller cache leaves room on the head card for the `full` arm, which is the r4 unpruned 1.27 GB mirror. The script skips `full` and says so if there is no room.

**G0: all of these must hold:**
- `gpu-unit.log` ends with `ALL PASS`. The unit test checks five things:
  - the mirror rows equal the host rows bit for bit;
  - the VRAM added equals 65,536 x 2,560 x 2 B;
  - the pruned forward, the full-mirror forward and the host path agree bitwise, for in-set IDs and for out-of-set IDs (N-1, N, vocab-1, the special-token range, mixed batches of 16);
  - the out-of-set lookup does not block the host while about 100 ms of device work is queued;
  - a second same-shape out-of-set lookup waits for the first upload before reusing the pinned buffer, and both uploads hold exactly their own rows (checked at the raw-row level, in the table dtype);
  - the out-of-set forward output equals the full-mirror forward output bitwise. The script also prints an `INFO` line comparing against a host-cast reference; that line does not gate;
  - the microbenchmark table is present.
- `chain.log` ends with `PASS`. `chain-summary.json` must show:
  - `compare.pruned.draft_windows_identical` and `jobs_identical` are true. That means every drafted window, output token, accepted and rejected count and `draft_stats` row is identical to `off` across 8 prompts at c1 and at c4.
  - The same holds for `compare.full`, if that arm ran.
  - `pruned_host_fallback.host_calls` > 0. This proves real verified tokens outside the 65,536 rows went through the fallback. The prompts are ChatML-framed, so generation opens with the reasoning special token. Several prompts are CJK or Cyrillic.
- Record these for the report:
  - `chain.c1.median_us` and `chain.c4.median_us` per arm. This is the chain microbenchmark: the draft call with both cards synchronized around it.
  - `tok_s`.
  - `vram_free_mib_before_after`.
  - `gpu-unit.json` `bench_us`.

## Steps 2-5: served arms

**G1, boot.** The ON log must contain:

`EXL3_EMBED_GPU_PRUNED=1: embedding mirror 65536 of 248320 rows (320.0 MiB, prefix) on cuda:1`

The second number is the embedding table's row count, 248,320 per the brief. If it differs, report it but do not fail the arm.

It must not contain `EXL3_EMBED_GPU_PRUNED=1 inactive`. If `inactive` appears, the arm silently ran the host chain. Abort the arm; the rest of the line gives the reason.

The OFF log must contain neither line.

Both lines are stdlib `logging` warnings from `exllamav3.generator.generator`. With no handler configured they reach stderr, and so `docker logs`, through Python's last-resort handler. If the ON log contains neither line, grep it for `EXL3_EMBED_GPU_PRUNED`. If that finds nothing, treat the arm as inactive.

Record the launcher's post-warmup `VRAM free MiB` line. The expected values:
- OFF: 2,015 / 1,099, the same as R514/R517.
- ON: cuda:0 unchanged, and cuda:1 about 1,099 − 320 = about 779 MiB (±30).

A cuda:1 drop far above 320 MiB means something else was allocated. Report it.

**G2, fingerprints.** Both arms must produce c1 greedy `ae890c45d1000582` and 30k `4a255910dee2d9c5`. These are the daily's canonical values since R517. Any other hash fails the arm. The overlay changes where embedding rows come from, not their values, so no numerics change is allowed.

**fn_bench.** Use the usual shapes: code and prose, 2,048 forced tokens, c1 and c4, two runs each, in that order on both arms. Report the mean per shape as ON vs OFF. The survey forecast for the device chain (P2b) is −0.6 to −1.2 ms per step at c1 and −0.3 to −0.6 ms at c4 d3. It has never been measured on this box, so treat it as a hypothesis. Noise across boots has been ±1-2 % (R499/R513 pairs).

**G4, c4 stress.**
1. Send 4 concurrent requests. Each has a salted ~30k-token prompt and 512 greedy output tokens. This puts the prefill workspaces, 4 recurrent slots and the MTP chain on the cards at once, which is the VRAM peak the reduced cuda:1 headroom has to survive.
2. Then read `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`.
3. Grep the container log for `out of memory`, `CUDA error` and `Traceback`.

Pass means all 4 requests complete, there are no error lines, and the container is still up. Report free VRAM per card after the pass for both arms. The ON − OFF difference should stay about 320 MiB on cuda:1.

## What decides promotion

Promotion needs all of the following:
- G0 through G4 green;
- fingerprints canonical;
- ON c1 above OFF by more than the run-to-run spread in both code and prose, and c4 no worse.

The flag changes no numerics, so under the track's rule a byte-identical result does not need the quality gate. A cuda:1 free floor under about 600 MiB after the stress pass is a reason to hold even if speed is up, because the pool was sized against 1,099 MiB of headroom.
