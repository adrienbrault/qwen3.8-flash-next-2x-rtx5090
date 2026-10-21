# Experimental ExLlamaV3 1.6.4 rebase

Prepared 2026-10-21. The served image and checkpoint remain unchanged. This packet pins ExLlamaV3 1.6.4, a compatible TabbyAPI commit and the official Python 3.12/CUDA 12.8/Torch 2.9.0 wheel. It carries server fixes and the batch-indexed MTP selector while using the new upstream kernels and scheduler. [PATCH-AUDIT.md](PATCH-AUDIT.md) distinguishes retained code from native replacements and known differences. No performance or accuracy gain is established by the source port alone.

Run `bash build.sh` on the Linux host containing the exact parent image. `prepare.py` verifies upstream archive and wheel SHA256 values before applying the source patches; the build runs CPU import, result-handler and draft-schema checks. The output is a candidate image, never an installed serving configuration. The checkpoint manifests pin both self-calibrated 2.18 and 2.68 bpw revisions for later comparison against the existing 2.50 bpw pack.
