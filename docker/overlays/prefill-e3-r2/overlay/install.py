#!/usr/bin/env python3
"""Hash-pin, install E3 sources, and patch the served exllamav3 package."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
ROUND = HERE.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_root() -> Path:
    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("cannot locate installed exllamav3 package")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    patch = ROUND / "served-source.patch"
    if sha256(patch) != manifest["patch_sha256"]:
        raise SystemExit("prefill E3 patch hash mismatch")
    root = package_root()
    for relative, record in manifest["patched_files"].items():
        target = root / relative
        actual = sha256(target) if target.is_file() else "MISSING"
        if actual != record["baseline_sha256"]:
            raise SystemExit(f"baseline hash mismatch: {relative}: {actual}")

    additions = ROUND / "overlay" / "exllamav3"
    for relative, expected in manifest["added_files"].items():
        source = additions / relative
        if sha256(source) != expected:
            raise SystemExit(f"overlay source hash mismatch: {relative}")
        target = root / relative
        if target.exists() and sha256(target) != expected:
            raise SystemExit(f"refusing to replace unexpected existing source: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    with patch.open("rb") as stream:
        subprocess.run(
            ["patch", "-p1", "--fuzz=0", "--no-backup-if-mismatch"],
            cwd=root,
            stdin=stream,
            check=True,
        )
    for relative, record in manifest["patched_files"].items():
        actual = sha256(root / relative)
        if actual != record["overlay_sha256"]:
            raise SystemExit(f"patched hash mismatch: {relative}: {actual}")

    removed = []
    for pattern in ("exllamav3_ext*.so", "exllamav3_ext*.pyd"):
        for candidate in root.parent.glob(pattern):
            candidate.unlink()
            removed.append(str(candidate))
    print(json.dumps({
        "patched": list(manifest["patched_files"]),
        "added": list(manifest["added_files"]),
        "removed_extensions": removed,
    }))


if __name__ == "__main__":
    main()
