# Experimental exact-name policy and FLy-style rescue

The served image `tabbyapi:flashnext-names-fast-r1` is pf2lock r3 plus the Python files in `overlay/payload/`. `overlay/manifest.json` pins every predecessor and candidate SHA256. `overlay/install.py` checks all files and compiles all Python before replacement; it preserves originals under `/opt/flashnext-names-r1/original`. Native extension bytes are unchanged.

`EXL3_FLASHNEXT_FAST=1` makes `<served-model>-fast` available. Only that exact name selects entropy-gated W=1 rescue on eligible speculative windows. The default name uses native acceptance. The policy follows live job/requeue transitions and isolates RAM and NVMe cache domains; disk namespaces change with engine source. Probability/logit, stateful, forced/minimum-token, healing, checkpoint and lookup windows retain native fallback paths.

The method comes from [Li et al., FLy](https://arxiv.org/abs/2511.22972); [THIRD_PARTY.md](../../../THIRD_PARTY.md#fly-method-and-opt-in-name-overlay) records licences and reference origins. [Screen and ship evidence](../../../bench/results/headline-r1.md#experimental-alias-and-promotion-evidence) records temperature-dependent timing and the change in output variety.

`bash docker/overlays/flashnext-names-fast-r1/build-image.sh` requires the exact local pf2lock r3 parent digest and refuses an existing destination tag. `bash docker/build-chain.sh` rebuilds the full source chain and authenticates predecessor file bytes. Reconstructed image IDs differ and require a reviewed launcher pin update. A full GPU/image rebuild was not performed during this publication preparation.
