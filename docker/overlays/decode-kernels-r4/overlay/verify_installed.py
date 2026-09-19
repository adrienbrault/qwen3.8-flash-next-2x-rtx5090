#!/usr/bin/env python3
"""Refuse to overlay any installed exllamav3 tree other than the served baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE = Path("/opt/venv/lib/python3.12/site-packages/exllamav3")
MANIFEST = Path("/opt/decode-kernels-r4/manifest.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads(MANIFEST.read_text())
bad = []
for relative, record in manifest["files"].items():
    path = PACKAGE / relative
    actual = sha256(path) if path.is_file() else "missing"
    if actual != record["baseline_sha256"]:
        bad.append((relative, record["baseline_sha256"], actual))
if bad:
    for relative, expected, actual in bad:
        print(f"baseline mismatch {relative}: expected {expected}, got {actual}")
    raise SystemExit(1)
print(f"verified {len(manifest['files'])} served baseline files")
