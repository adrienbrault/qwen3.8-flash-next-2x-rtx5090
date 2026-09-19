#!/usr/bin/env python3
"""
Install the tool-choice-r1 overlay into a TabbyAPI tree (default /app).

For every file in manifest.json:
  - baseline_sha256 set:  the installed file must exist with exactly that hash
  - baseline_sha256 null: the file is new and must not exist yet
  - a file that already has the overlay hash is accepted (re-run on a patched tree)
Then copies the overlay files in and checks every installed hash against
overlay_sha256. Any mismatch exits non-zero, which fails the image build.
All checks run before anything is written.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/app")
    ap.add_argument("--manifest", default=os.path.join(here, "manifest.json"))
    ap.add_argument("--overlay", default=os.path.join(here, "overlay"))
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    errors = []
    to_copy = []
    for entry in manifest["files"]:
        rel = entry["path"]
        src = os.path.join(args.overlay, rel)
        dst = os.path.join(args.root, rel)
        base, over = entry["baseline_sha256"], entry["overlay_sha256"]

        if not os.path.isfile(src):
            errors.append(f"{rel}: missing from overlay")
            continue
        if sha256(src) != over:
            errors.append(f"{rel}: overlay file does not match manifest overlay_sha256")
            continue

        if os.path.exists(dst):
            current = sha256(dst)
            if current == over:
                print(f"already applied: {rel}")
                continue
            if base is None:
                errors.append(f"{rel}: new file expected, but one exists (sha256 {current})")
                continue
            if current != base:
                errors.append(f"{rel}: baseline mismatch: expected {base}, found {current}")
                continue
        elif base is not None:
            errors.append(f"{rel}: baseline file missing")
            continue
        to_copy.append((rel, src, dst))

    if errors:
        for e in errors:
            print(f"ERROR {e}", file=sys.stderr)
        print("tool-choice-r1 overlay NOT installed: baseline verification failed", file=sys.stderr)
        return 1

    for rel, src, dst in to_copy:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"installed: {rel}")

    bad = [e["path"] for e in manifest["files"] if sha256(os.path.join(args.root, e["path"])) != e["overlay_sha256"]]
    if bad:
        for rel in bad:
            print(f"ERROR {rel}: installed file does not match overlay_sha256", file=sys.stderr)
        return 1

    print(f"tool-choice-r1 overlay verified: {len(manifest['files'])} files under {args.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
