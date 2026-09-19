# Decode kernels round 4

Opt-in overlay for the served exllamav3 tree. No GPU box was contacted and no performance or
numerical measurement was fabricated.

Contents:

- `served-source.patch` — zero-fuzz patch against `ref/served-src/exllamav3`.
- `overlay/exllamav3/` — the nine complete replacement files.
- `overlay/manifest.json` — served/overlay SHA-256 hashes.
- `impl-status-decode-kernels-r4.md` — implementation, output/RNG contracts, source-derived head
  and VRAM estimates, interaction audit, and unverified work.
- `box-ab-spec.md` — clean sm_120 build plus OFF/ON/OFF2/ON2 serving gates.
- `mixer_int8_parity.py` — 96-site, real-dimension synthetic fp16/int8 audit.
- `tests/test_round4_cpu.py` and `verify_local.py` — CPU-only structural checks.
- `tests/test_gr_mix_v2_int8.py` — box-only CUDA parity/error/timing test.

Selectors, all default off/unset:

```text
EXL3_DRAFT_PINNED_STAGING=1
EXL3_BATCH_VERIFY=1
EXL3_MTP_DEVICE_DRAFT=1
EXL3_EMBED_GPU=1
EXL3_MTP_HEAD_N=65536
EXL3_HC_MIX_V2_INT8=1
```

Local validation:

```bash
python3 out/decode-kernels-r4/verify_local.py
```

The local result is in `local-validation.json`: required checks pass; torch and CUDA are unavailable,
so numerical and GPU gates remain pending.
