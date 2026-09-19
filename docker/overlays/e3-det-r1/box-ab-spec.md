# Box A/B spec: e3-det r1 (deterministic E3 grouped MoE prefill)

Nothing here has run on the box. The GPU budget is 15 minutes, counted from the first GPU step (the live-daily baseline) to the daily's restore. The image build runs first and does not use the GPU, so the daily keeps serving while it builds.

Flag: `EXL3_MOE_PREFILL_E3_DET=1`. It only acts where `EXL3_MOE_PREFILL_E3=1` is also set and a prefill chunk has at least 512 rows. With the flag unset, the image runs exactly what `tabbyapi:stack-r4-e3r2` runs.

## 0. Build (no GPU, about 3-6 min, extension rebuilt for sm_120)

```bash
cd <checkout>/out/e3-det-r1          # build context = this directory
sudo docker build --build-arg BASE=tabbyapi:stack-r4-e3r2 -f Dockerfile.box -t tabbyapi:e3-det-r1 . 2>&1 | tee build.log
```

The Dockerfile has no heredocs. The build fails on any of the following:

- `install.py`: one of the five replaced files is not the stack-r4-e3r2 copy (sha256 in `manifest.json` `pre`), or the overlay payload or the installed copy has the wrong hash.
- The sm_120 JIT build of the whole extension fails (via `build_extension.py`, the stack's own helper).
- The import assertion fails. It checks that `exl3_moe_prefill_e3_det` and `exl3_moe_prefill_e3_det_reduce` exist in the site-packages `.so`, and that the E3 and int8-mixer symbols are still there.
- `block_sparse_mlp.MOE_PREFILL_E3_DET` does not default to `False`.

Archive `build.log`, the image digest and the rebuilt `.so` sha256.

A build failure or a failed assertion is a NO-GO. Send back the compiler output. The CUDA was only checked locally with clang's CUDA front end, not nvcc (see impl-status).

## GPU steps (15 min cap)

`LIVE_ENV` below is the live launcher's EXTRA_ENV, exactly: `EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1 EXL3_DRAFT_PINNED_STAGING=1 EXL3_BATCH_VERIFY=1 EXL3_MTP_HEAD_N=65536 EXL3_MOE_PREFILL_E3=1 EXL3_HC_MIX_V2_INT8=1`.

| t (min) | step | est. |
|---|---|---|
| 0.0 | **B0 live-daily baseline**: salted cold prefill on the running daily (atomic E3). Run `fn_bench --unique --salt <new random per invocation>` once per context: 60k twice, then 120k twice. Record TTFT and prefill tok/s. | 1.0 |
| 1.0 | Stop the daily; both GPUs free | 0.3 |
| 1.3 | **K1 + E1**, process 1: `gpu_e3_det.py --kernel --e2e` (command below). Stop here on FAIL. | 4.0 |
| 5.3 | **K2**, process 2: `gpu_e3_det.py --kernel --compare` with reduced repeats (cross-process hashes) | 2.0 |
| 7.3 | **Boot A**: served, DET on. Record free VRAM after boot. Take the c1 and 30k fingerprints as the first requests after boot, then stop. | 1.8 |
| 9.1 | **Boot B**: fresh boot, same image and env. Record free VRAM after boot. Take the c1 and 30k fingerprints first. | 1.5 |
| 10.6 | Boot B: salted cold prefill 60k ×2, 120k ×2 (same fn_bench form as B0), sampling free VRAM during the 120k runs | 1.0 |
| 11.6 | Boot B: decode, fn_bench code c1 and c4, 2,048 tokens × 2 runs | 1.0 |
| 12.6 | Restore the daily with `env -i`, so no EXL3 knobs leak into it | 1.0 |
| 13.6 | end (1.4 min slack) | |

Commands (in-image paths):

```bash
IMG=tabbyapi:e3-det-r1; R=/srv/qwen5090/results/<date>-e3-det-r1; mkdir -p $R
ENVF=""; for kv in $LIVE_ENV; do ENVF="$ENVF -e $kv"; done     # NOT EXL3_MOE_PREFILL_E3_DET: the script flips it in-process
RUN="sudo docker run --rm --gpus all --ipc=host --shm-size=16g -v /srv/qwen5090/models:/models:ro \
  -v /srv/qwen5090/.exl3cache:/exl3-cache -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
  $ENVF -v $R:/out --entrypoint python3 $IMG /opt/e3-det-r1/tests/gpu_e3_det.py \
  --model /models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab"
# K1 + E1 (one process): kernel phase (K=2/3/4 layers x 512/1,024/2,048 rows, 20 repeats) + cold 8k/30k x 3 reps DET on, then DET off
$RUN --kernel --e2e --out /out/gpu_e3_det_p1.json 2>&1 | tee $R/gpu_e3_det_p1.log
# K2 (second process): same inputs, det sha256 must equal p1's
$RUN --kernel --repeats 3 --iters 5 --warmup 2 --compare /out/gpu_e3_det_p1.json --out /out/gpu_e3_det_p2.json 2>&1 | tee $R/gpu_e3_det_p2.log
# Served boots A and B
env -i PATH=$PATH IMG=$IMG EXTRA_ENV="$LIVE_ENV EXL3_MOE_PREFILL_E3_DET=1" /srv/qwen5090/launch-flashnext.sh
```

The served boots use the live launcher unchanged, with `IMG` and `EXTRA_ENV` overridden, on the same port, pool (819,200 @ 8,8), 4 slots, split [30, 30], chunk 2048, draft policy and vision. Check that `EXL3_MOE_PREFILL_E3_DET=1` is present in the container env (`docker inspect`) on both boots.

## Gates

| gate | pass condition | on fail |
|---|---|---|
| G0 build | image builds, install hashes OK, import assertion OK | NO-GO, send the compiler log |
| G1 kernel determinism | p1 verdict PASS. For every (layer K2/K3/K4, rows 512/1,024/2,048): 20 DET repeats bitwise equal, 5 DET runs with a concurrent side-stream GEMM equal to them, outputs finite, NRMSE(DET, atomic E3) < 1e-5, DET peak allocation ≤ atomic + 4 MiB | NO-GO. An NRMSE ≥ 1e-5 means a slot/index/epilogue bug, not noise. |
| G2 e2e determinism | cold 8k and 30k: 3 DET reps identical (`det_first_diff` all null) | NO-GO |
| G3 cross-process | p2 `cross_process.equal` all true | NO-GO |
| G4 fingerprints | c1 on A and B equals the live canonical `e7fb377c987d685c` (the short c1 prompt never reaches E3, so the flag must not move it). The 30k fingerprint on A equals B. This is the new DET 30k canonical, expected to differ from `4a255910dee2d9c5`. | c1 moved: NO-GO (default-path leak). 30k A ≠ B: NO-GO. |
| G5 cold prefill | mean tok/s of boot B ≥ 0.97 × B0's mean, separately at 60k and 120k | see below |
| G5b VRAM | free VRAM per card after boot A and B, and during the 120k cold prefill (`nvidia-smi --query-gpu=memory.used,memory.total`), no more than 50 MiB below the live daily's (R525: 1,973 / 1,057 MiB free at boot, 1,215 / 569 under load). DET allocates one 210 MB slot block where atomic E3 allocated two 105 MB blocks: equal bytes, but the caching allocator's reserved footprint could differ. | NO-GO for promotion. The flag stays usable for identity A/Bs. |
| G6 decode | code c1 and c4 aggregate within ±2 % of the live daily's last published boot means (R525: c1 ~224, c4 ~503). Decode never enters E3 (rows < 512), so this only guards against image or build drift. | investigate before promoting |

Informational (no gate):

- Atomic E3's distinct outputs over 20 repeats (expect > 1).
- The atomic reps' `first_diff` in E1: shows the baseline differs. It can agree by chance on a given prompt.
- `det_vs_atomic_first_diff`.
- `det_over_atomic` per layer and row count.

Record the kernel `det_over_atomic` at 2,048 rows in the result file. It predicts G5: about 1.0 + 0.6 × (ratio − 1) end to end, because MoE is roughly 60 % of a 2,048-row chunk.

On G5: `impl-status.md` estimates −1 % to −3.5 % end to end. If G5 misses by less than 1 point, report the number and do not promote; round 2 has two levers, both listed in `impl-status.md`. The flag is still usable for identity A/Bs on long prompts, which is the brief's first purpose: every byte-identity gate on long prompts is void without it.

## Promotion (if G0-G6 and G5b pass)

Promotion is autonomous per the Flash-Next policy:

- Daily = `IMG=tabbyapi:e3-det-r1` + live EXTRA_ENV + `EXL3_MOE_PREFILL_E3_DET=1`.
- New 30k canonical = the G4 hash.
- The c1 canonical is unchanged.
- Rollback = the current launcher (`tabbyapi:stack-r4-e3r2`, no DET flag). The same image with the flag removed is byte-identical to it.
