#!/usr/bin/env python3
"""Hash-pin and apply the gdn-state-bf16 round-1 patch (EXL3_GDN_STATE_BF16) to the installed exllamav3 package.

Aborts unless the patch and every touched file match manifest.json exactly, applies with
fuzz 0, then checks every patched file against its pinned post-patch hash.

--root points at an exllamav3 package directory instead of the installed one (CPU tests).
--manifest selects the hash pins: manifest.json = the served R535 tree (default),
manifest-on-r6.json = the same patch stacked on decode-kernels r6 (gdn.cu baseline = r6 overlay).
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
    # Docker copies everything into one directory; the repo keeps the patch one level up.
    for candidate in (HERE / name, HERE.parent / name):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"missing {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--keep-extension", action="store_true",
                        help="do not delete prebuilt exllamav3_ext binaries next to the package")
    args = parser.parse_args()

    manifest = json.loads(locate(args.manifest).read_text())
    source_patch = locate("served-source.patch")
    if sha256(source_patch) != manifest["patch_sha256"]:
        raise SystemExit("gdn-state-bf16 r1 patch hash mismatch")
    root = args.root.resolve() if args.root else package_root()

    for relative, record in manifest["files"].items():
        target = root / relative
        actual = sha256(target) if target.is_file() else "MISSING"
        if actual != record["baseline_sha256"]:
            raise SystemExit(f"baseline hash mismatch: {relative}: {actual}")

    with source_patch.open("rb") as stream:
        subprocess.run(
            ["patch", "-p1", "--fuzz=0", "--no-backup-if-mismatch"],
            cwd=root,
            stdin=stream,
            check=True,
        )
    rejects = sorted(str(p) for p in root.rglob("*.rej"))
    if rejects:
        raise SystemExit(f"patch left reject files: {rejects}")

    for relative, record in manifest["files"].items():
        actual = sha256(root / relative)
        if actual != record["overlay_sha256"]:
            raise SystemExit(f"overlay hash mismatch: {relative}: {actual}")

    removed = []
    if not args.keep_extension:
        for pattern in ("exllamav3_ext*.so", "exllamav3_ext*.pyd"):
            for candidate in root.parent.glob(pattern):
                candidate.unlink()
                removed.append(str(candidate))
    print(json.dumps({"patched": list(manifest["files"]), "removed_extensions": removed}))


if __name__ == "__main__":
    main()
