#!/usr/bin/env python3
"""Install the decode-kernels-r2 payload, refusing an unknown source baseline."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_root() -> Path:
    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("cannot locate installed exllamav3 package")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    root = package_root()
    for relative, expected in manifest["files"].items():
        target = root / relative
        payload = HERE / "payload" / relative
        actual_baseline = sha256(target) if target.is_file() else "MISSING"
        actual_payload = sha256(payload) if payload.is_file() else "MISSING"
        if actual_baseline != expected["baseline_sha256"]:
            raise SystemExit(f"baseline hash mismatch: {target}: {actual_baseline}")
        if actual_payload != expected["payload_sha256"]:
            raise SystemExit(f"payload hash mismatch: {payload}: {actual_payload}")

    for relative in manifest["files"]:
        shutil.copy2(HERE / "payload" / relative, root / relative)

    removed = []
    for pattern in ("exllamav3_ext*.so", "exllamav3_ext*.pyd"):
        for candidate in root.parent.glob(pattern):
            candidate.unlink()
            removed.append(str(candidate))
    print(json.dumps({"installed": list(manifest["files"]), "removed_extensions": removed}))


if __name__ == "__main__":
    main()
