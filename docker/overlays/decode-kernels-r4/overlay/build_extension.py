#!/usr/bin/env python3
"""Import edited CUDA sources once and install the resulting JIT extension."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

from exllamav3.ext import exllamav3_ext


spec = importlib.util.find_spec("exllamav3")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("cannot locate installed exllamav3 package")
site_root = Path(next(iter(spec.submodule_search_locations))).resolve().parent
built = Path(exllamav3_ext.__file__).resolve()
destination = site_root / built.name
if built != destination:
    shutil.copy2(built, destination)
print(f"installed rebuilt extension: {destination}")
