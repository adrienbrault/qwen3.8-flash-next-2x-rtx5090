#!/usr/bin/env python3
"""Fail-closed source overlay installer. Run in a disposable derived image."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, help="Parent of the installed exllamav3 package")
    p.add_argument("--disable-precompiled", action="store_true")
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "manifest.json").read_text())
    if a.root is None:
        spec = importlib.util.find_spec("exllamav3")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("Cannot locate exllamav3; provide --root")
        a.root = Path(next(iter(spec.submodule_search_locations))).parent
    # Validate EVERYTHING before changing any source file; never accept fuzz or drift.
    for f in manifest["files"]:
        src, dst = here / "files" / f["path"], a.root / f["path"]
        if sha(src) != f["sha256"]:
            raise RuntimeError(f"Corrupt overlay: {src}")
        before = sha(dst) if dst.exists() else None
        if before != f["before_sha256"]:
            raise RuntimeError(f"Baseline mismatch: {dst}: {before}, expected {f['before_sha256']}. Do not force; rebase/re-audit first.")
    report = {"root": str(a.root), "files": manifest["files"]}
    for f in manifest["files"]:
        dst = a.root / f["path"]; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(here / "files" / f["path"], dst)
        assert sha(dst) == f["sha256"]
        print("PASS installed", dst, f["sha256"])
    if a.disable_precompiled:
        spec = importlib.util.find_spec("exllamav3_ext")
        if spec is not None and spec.origin:
            binary = Path(spec.origin)
            if binary.suffix != ".so":
                raise RuntimeError(f"Unexpected precompiled extension: {binary}")
            backup = binary.with_name(binary.name + ".before-moe-v2")
            if backup.exists():
                raise RuntimeError(f"Backup already exists: {backup}")
            report["disabled_binary"] = {"path": str(binary), "sha256": sha(binary), "backup": str(backup)}
            binary.rename(backup)
            print("PASS disabled stale extension", binary)
    (here / "installed.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
