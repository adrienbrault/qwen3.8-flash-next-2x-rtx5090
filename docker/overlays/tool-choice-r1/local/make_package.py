"""
Regenerate manifest.json and served-source.patch for tool-choice-r1 from the
overlay/ tree and the unpatched reference tree. Local tool, not shipped.

  python make_package.py --ref ../../../ref/served-src/tabbyapi
"""

import argparse
import difflib
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OVERLAY = os.path.join(PKG, "overlay")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    args = ap.parse_args()

    files = []
    for dirpath, _, names in os.walk(OVERLAY):
        for name in names:
            if name.endswith(".pyc"):
                continue
            files.append(os.path.relpath(os.path.join(dirpath, name), OVERLAY))
    files.sort()

    entries = []
    patch = []
    for rel in files:
        over = os.path.join(OVERLAY, rel)
        base = os.path.join(args.ref, rel)
        has_base = os.path.isfile(base)
        entries.append(
            {
                "path": rel,
                "baseline_sha256": sha256(base) if has_base else None,
                "overlay_sha256": sha256(over),
            }
        )
        old = open(base).read().splitlines(keepends=True) if has_base else []
        new = open(over).read().splitlines(keepends=True)
        patch.extend(
            difflib.unified_diff(
                old,
                new,
                fromfile=f"a/{rel}" if has_base else "/dev/null",
                tofile=f"b/{rel}",
            )
        )

    manifest = {
        "name": "tool-choice-r1",
        "root": "/app",
        "note": "paths relative to /app; baseline_sha256 null = new file",
        "files": entries,
    }
    with open(os.path.join(PKG, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    with open(os.path.join(PKG, "served-source.patch"), "w") as f:
        f.writelines(patch)
    print(f"{len(entries)} files; manifest.json and served-source.patch written")


if __name__ == "__main__":
    main()
