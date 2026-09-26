# R739 and R745: a no-cache rebuild of the chain reproduces the served sources and kernels; 8 dependencies drifted and are now pinned

Results `2026-09-26-r739-build-verify-flashnext` (R739, 2026-09-26 02:08 to 03:05 UTC) and `2026-09-26-r745-constraints-verify` (R745, 07:03 to 07:10 UTC) on the serving host. Driver: [`docker/build-chain.sh`](../../docker/build-chain.sh). Raw records: [`2026-09-26-r739-build-verify/`](2026-09-26-r739-build-verify/).

## R739: the chain at 07141f3, built from scratch

The repository at `07141f3` (the issue #1 fixes), exported with `git archive` into an empty directory, built with `NO_CACHE=1 MAX_JOBS=12 IMAGE_REPO=tabbyapi-verify bash docker/build-chain.sh`. All 35 layers built with no cached step, and each `FROM` resolved to the previous `tabbyapi-verify` tag. The CUDA devel base (`nvidia/cuda:12.8.1-devel-ubuntu24.04`) was already on the host, so its pull is not in the figures below.

The layer logs show the three issue #1 fixes at work: the mixer V2 round 1 layer (`-hcmix1`) followed by round 2 (`-hcmix2`), the `memfix.py` hash check before and after (`1a5af53b` to `7f6e4b34`), and the `/tmp/build` removal with the import check in the `-bszn` and `-mixstate` layers.

| Compared, served `tabbyapi:stack-r3-rows32` vs rebuilt | Result |
|---|---|
| 554 ExLlamaV3 source files (`.py`, `.cu`, `.cuh`, `.cpp`, `.h`) and 160 TabbyAPI files under `/app`, by hash | identical |
| SASS of every function in the compiled `exllamav3_ext` (1,633 function bodies) | identical |
| Image environment (`.Config.Env`, which carries the `EXL3_*` defaults) | identical |
| The `exllamav3_ext` `.so` file as bytes | differs (the SASS is identical, so the difference is outside the kernels) |
| Versions of 100 installed Python distributions | 8 differ |

The 8 were `filelock` (3.32.7 to 4.0.3), `fsspec`, `huggingface_hub`, `idna`, `multidict`, `networkx`, `starlette` and `uvicorn`. All came from the one unpinned `pip install ".[cu12]"` of TabbyAPI in `Dockerfile.tabbyapi-qsa-cid`: TabbyAPI's own dependencies have no upper bound, and the rest arrive through torch, fastapi, anyio and aiohttp. torch, triton, exllamav3, flash-linear-attention, tokenizers, transformers and numpy matched. The effect of the newer `starlette` and `uvicorn` on serving was not measured. Diff: [`manifest.diff`](2026-09-26-r739-build-verify/manifest.diff); kernel and environment hashes: [`kernel-env-check.txt`](2026-09-26-r739-build-verify/kernel-env-check.txt).

Cost on a Ryzen 7 9800X3D, niced beside the serving engine: 57 min 7 s with `MAX_JOBS=12`, and 59.2 GB of disk beyond the CUDA base. The final image is 48.3 GB against the served 45.0 GB; the extra `-hcmix1` layer accounts for about 1.3 GB of the difference.

## R745: the dependencies pinned

[`docker/constraints-stack-r3-rows32.txt`](../../docker/constraints-stack-r3-rows32.txt) holds the versions of the served image (`pip list --format=freeze`), except torch, exllamav3 and TabbyAPI, which the TabbyAPI project pins by wheel URL. `Dockerfile.tabbyapi-qsa-cid` installs with `-c` on it, pins pip to 26.2.1, and fails the build when a named package resolves to another version.

No later layer installs Python packages, so R745 rebuilt that first layer alone with `--no-cache` (6 min 27 s) from `4598beb`: `constraints held: 83 pins`, and its 87 installed distributions equal the served image's 87 ([`r745-compare.txt`](2026-09-26-r739-build-verify/r745-compare.txt)).

Still unpinned: the apt packages of the CUDA base and the base itself, which is pulled by tag.
