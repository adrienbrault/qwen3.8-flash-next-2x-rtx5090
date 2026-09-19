#!/usr/bin/env python3
"""Import once to JIT-build sm_120, install the rebuilt extension, and assert its bindings."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

from exllamav3.ext import exllamav3_ext


REQUIRED = (
    # round 6
    "gdn_ba_gemv", "hc_apply", "gr_mix_v2_regrid", "gr_mix_v2_int8_regrid",
    # served chain that must survive the rebuild
    "gr_mix_v2", "gr_mix_v2_int8", "exl3_moe_prefill_e3_det",
)
missing = [name for name in REQUIRED if not hasattr(exllamav3_ext, name)]
if missing:
    raise SystemExit(f"rebuilt extension lacks {missing}")

spec = importlib.util.find_spec("exllamav3")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("cannot locate installed exllamav3 package")
site_root = Path(next(iter(spec.submodule_search_locations))).resolve().parent
built = Path(exllamav3_ext.__file__).resolve()
destination = site_root / built.name
if built != destination:
    shutil.copy2(built, destination)
print(f"installed rebuilt extension: {destination}")
