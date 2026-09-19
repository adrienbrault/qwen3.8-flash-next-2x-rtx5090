# Decode kernels round 6 — box build and A/B

Run this on flan under the normal GPU queue: source `lib/gpu-queue.sh` before the flock and chain the steps without restoring the daily in between. Put all output under `/srv/qwen5090/results/<date>-r6-decode-kernels/`. The base is the daily image `tabbyapi:nvme-tier-r4-e3det` (R535). No step below has been run.

## 1. Build

```bash
cd <synced copy of docker/overlays/decode-kernels-r6>
sudo docker build -f overlay/Dockerfile.box \
  --build-arg BASE=tabbyapi:nvme-tier-r4-e3det -t tabbyapi:decode-kernels-r6 . > build.log 2>&1
```

The build aborts on any of these:
- patch hash mismatch;
- any of the five served files differing from its pinned baseline;
- a non-zero `patch -p1 --fuzz=0` exit, or a `.rej` file;
- a post-patch hash mismatch;
- a rebuilt `.so` that lacks `gr_mix_v2_regrid`, `gr_mix_v2_int8_regrid`, `gr_mix_v2_int8` or `exl3_moe_prefill_e3_det`, or that is not in site-packages;
- `GatedResidual.STATE_REGRID` not defaulting to `False`.

This is the first nvcc compile of the round. Keep `build.log` in the results directory.

## 2. Both-card kernel equality

```bash
R=/srv/qwen5090/results/<date>-r6-decode-kernels
for G in 0 1; do
  sudo docker run --rm --gpus all -v "$R":/out tabbyapi:decode-kernels-r6 \
    python3 /opt/decode-kernels-r6/tests/test_r6_kernels.py \
      --device cuda:$G --iterations 200 --json /out/kernels-gpu$G.json
done
```

Both runs must exit 0. Each run checks the following with `torch.equal`, for rows 1..16:
- GDN B/A against the served eight-warp kernel, with and without bias;
- `hc_apply` against the served 256-thread kernel, for half/float `y` and comb/no-comb;
- V2 state, served vs re-gridded, for both `gr_mix_v2` (fp16) and `gr_mix_v2_int8` (the daily path). Each variant is checked in site and final form, with half and float `mixed`, and on all four outputs (`dots`, `state`, `post`, `mixed`).

`launch_check` must report `found: true` for every probe. If a probe reports `found: false`, the candidate fell back to the served kernel and the equality result proves nothing. If the profiler itself cannot start (CUPTI), rerun with `--no-launch-check` and confirm the kernel names from the Nsight capture in section 3. The JSON `timing_us` block holds median CUDA-event times of the served and re-gridded int8/fp16 mixer entry points at R = 1, 4, 16. Use these to check direction only.

Stop on any mismatch, non-finite output, or missing launch.

## 3. Serving A/B (one bundle)

Use one image (`tabbyapi:decode-kernels-r6`) for every arm and the unchanged daily launcher and config (layer split `[30, 30]`, cache 819,200, NVMe tier as the daily). The daily env is the base for every arm:

```text
DAILY='EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1 EXL3_DRAFT_PINNED_STAGING=1 EXL3_BATCH_VERIFY=1 EXL3_MTP_HEAD_N=65536 EXL3_MOE_PREFILL_E3=1 EXL3_HC_MIX_V2_INT8=1 EXL3_MTP_DEVICE_DRAFT=1 EXL3_EMBED_GPU=1 EXL3_EMBED_GPU_PRUNED=1 EXL3_MOE_PREFILL_E3_DET=1'
A: EXTRA_ENV="$DAILY"
B: EXTRA_ENV="$DAILY EXL3_GDN_BA_WARP1=1 EXL3_HC_APPLY_WARP1=1 EXL3_GR_STATE_REGRID=1"
```

`EXL3_GR_FREE_UP_H` does not exist in r6. Run 8 boots in the order A B B A B A A B, as in R533. Restart between arms, because every selector is read once per process.

Per boot, run:
- c1 and 30k greedy fingerprints;
- `fn_bench` decode code/prose at c1 and c4, 2,048 tokens, with a per-invocation `--salt`;
- MTP proposed/accepted totals.

Per arm, run once: the 12-prompt `multiprompt.py` probe at c1/c4.

Correctness gates. Any failure is a NO-GO:
- Every boot of both arms reproduces the daily fingerprints: c1 `e7fb377c987d685c`, 30k `4a255910dee2d9c5` (R525–R535). If arm A drifts, the boot is invalid, not the candidate.
- B's MTP accepted/proposed totals equal A's for the same prompts.
- Idle and loaded VRAM are equal within noise. r6 adds no persistent tensors.
- No CUDA error, no graph-replay failure.

Performance reading:
- Paired B/A ratio with a 95 % CI per cell (code/prose × c1/c4).
- A plausible gain shows up at c1. At c4 the B/A GDN launch falls back by construction (16 verify rows = 192 CTAs ≥ 170 SMs), so only `hc_apply` and the state re-grid can move c4.
- The only numeric expectation is −72.7 µs/step for B/A at c1/d3 (r5 planning estimate; see `impl-status.md`). No gain is predicted for `hc_apply` or the state re-grid.

Profile one short steady c1/d3 and c4/d3 decode window per arm. Expected geometry in B:
- `gdn_ba_gemv_kernel<1>` with `grid (96,4)` at c1/d3; the served `<8>` with `(12,16)` at c4/d3;
- `hc_apply_kernel<4,…,32>` with `(20,4)` at c1/d3 and `(10,16)` at c4/d3;
- `gr_v2_state_regrid_kernel` with `(11,R)` site / `(10,R)` final, block 32. Arm A must show `gr_v2_state_kernel` with `(R)` × 128.

## 4. Attribution (only if B is faster at c1 with a CI excluding 0)

Run a 4-boot A / G / G / A with `G: EXTRA_ENV="$DAILY EXL3_GDN_BA_WARP1=1"` to separate the B/A change from the two launch-floor re-grids. If G matches B, promote only the GDN selector. Fewer default-on knobs is the tie-breaker. If B is flat or slower, stop. Do not run per-selector quartets on a null bundle.

## 5. Decision and rollback

Promotion uses the usual Flash-Next gates on the promoted env: canonical fingerprints, agent replay, needles, tool-eval, GSM8K through the no-stop proxy. The launcher change is `IMG=tabbyapi:decode-kernels-r6` plus the chosen selectors in `EXTRA_ENV`. Mirror it to the public Flash-Next repo in the same session.

Rollback takes two steps:
1. Keep the r6 image and drop the selectors. That is byte-identical to the served launches.
2. Or restore `IMG=tabbyapi:nvme-tier-r4-e3det`, which was `scripts/launch-flashnext.sh` before R540.
