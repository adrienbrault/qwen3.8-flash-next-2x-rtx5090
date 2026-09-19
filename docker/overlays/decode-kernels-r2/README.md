# Decode kernels round 2

Outcome: Option A (shared-expert stream overlap) is implemented behind
`EXL3_SHARED_EXPERT_OVERLAP=1`, default off. The implementation is CPU-integrity-checked but has not
been compiled or run on a GPU. Option B is not exactly composable from the served GEMM kernels;
Option C is incompatible because shared matrices are K=5 while routed matrices are K=3.

- `source-map-and-options.md` — complete source map, precision/order, corrected K/payload analysis,
  and Options A/B/C disposition.
- `impl-status-decode-kernels-r2.md` — claim-by-claim file:line status and GPU-unverified list.
- `served-source.patch` — reviewable four-file native diff.
- `overlay/` — hash-gated payload, Docker build and forced extension rebuild.
- `tests/test_shared_expert_overlap.py` — checkpoint-backed two-card rows-1..16 `torch.equal` test
  and one-layer rows-1/4/16 microbenchmark.
- `box-ab-spec.md` — GPU test plus OFF/ON/OFF2/ON2 serving protocol.
- `overlap-estimates.tsv` — measured inputs and explicitly derived ceilings/projections.
- `verify_local.py`, `local-validation.json` — reproducible CPU-only validation and result.
