#!/usr/bin/env python3
"""Hash-pin and apply the mtp-kv-window round-1 patch (EXL3_MTP_KV_WINDOW) to the installed exllamav3 package.

Aborts unless the patch and every touched file match manifest.json exactly (a file with a null baseline is new and
must not exist yet), applies with `patch -p1 --fuzz=0` (a non-zero exit code aborts), then checks every patched file
against its pinned post-patch hash. The patch is Python only, so the prebuilt extension stays in place.

--root points at an exllamav3 package directory instead of the installed one (CPU tests).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_root() -> Path:
    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("cannot locate installed exllamav3 package")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def locate(name: str) -> Path:
    # Docker copies everything into one directory; the repo keeps the patch one level up
    for candidate in (HERE / name, HERE.parent / name):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"missing {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type = Path, default = None)
    parser.add_argument("--manifest", default = "manifest.json")
    args = parser.parse_args()

    manifest = json.loads(locate(args.manifest).read_text())
    source_patch = locate("served-source.patch")
    if sha256(source_patch) != manifest["patch_sha256"]:
        raise SystemExit("mtp-kv-window r2 patch hash mismatch")
    root = args.root.resolve() if args.root else package_root()

    for relative, record in manifest["files"].items():
        target = root / relative
        if record["baseline_sha256"] is None:
            if target.exists():
                raise SystemExit(f"new file already present: {relative}")
            continue
        actual = sha256(target) if target.is_file() else "MISSING"
        if actual != record["baseline_sha256"]:
            raise SystemExit(f"baseline hash mismatch: {relative}: {actual}")

    with source_patch.open("rb") as stream:
        proc = subprocess.run(
            ["patch", "-p1", "--fuzz=0", "--no-backup-if-mismatch"],
            cwd = root,
            stdin = stream,
        )
    if proc.returncode != 0:
        raise SystemExit(f"patch exited with {proc.returncode}")
    rejects = sorted(str(p) for p in root.rglob("*.rej"))
    if rejects:
        raise SystemExit(f"patch left reject files: {rejects}")

    for relative, record in manifest["files"].items():
        actual = sha256(root / relative)
        if actual != record["overlay_sha256"]:
            raise SystemExit(f"overlay hash mismatch: {relative}: {actual}")

    # Drop compiled bytecode of the replaced sources so the image carries none of the served versions
    removed = []
    for relative in manifest["files"]:
        cache_dir = (root / relative).parent / "__pycache__"
        stem = Path(relative).stem
        for candidate in cache_dir.glob(f"{stem}.*.pyc") if cache_dir.is_dir() else []:
            candidate.unlink()
            removed.append(str(candidate))
    print(json.dumps({"patched": list(manifest["files"]), "removed_bytecode": removed}))


if __name__ == "__main__":
    main()
