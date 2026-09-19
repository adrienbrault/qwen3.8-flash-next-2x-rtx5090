#!/usr/bin/env python3
"""
Install the NVMe-tier overlay (round 3b tier, round 4 packaging) into the installed exllamav3 package.

The installed tree must match exactly one baseline set in manifest.json. The variant is picked from the installed files'
hashes, never from the image tag:
  stack-r4-e3r2                    the stack image
  stack-r4-e3r2+recurrent-tip-r1   the stack image with recurrent-tip-r1 installed first
  stack-r4-e3r2+mtp-pruned-r1      mtp-pruned-r1 and its descendants (-tc1, -tc1-plefix: the served chain; tool-choice
                                   touches TabbyAPI only, ple-ckpt-clone touches modules/ple.py only)
Before anything else, every "requires" entry must hold on any variant: modules/ple.py must carry the ple-ckpt-clone r1
stash fix (R526: without it the tier refused checkpoints that changed while they were being serialized).
Every baseline file is verified before anything is copied (a file listed with a null baseline must be absent; a
"present" file must exist, any version);
every copied file is verified after. A tree that already carries this overlay (any variant) is a no-op.
Soft dependencies ("depends") are only reported. Exits non-zero on any mismatch; nothing is copied then.

  python3 install.py [--pkg-dir DIR] [--check]
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent


def sha(p: pathlib.Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def site_root(pkg_dir: str | None) -> pathlib.Path:
    if pkg_dir:
        pkg = pathlib.Path(pkg_dir).resolve()
    else:
        spec = importlib.util.find_spec("exllamav3")  # locates the package without importing it
        if spec is None or not spec.submodule_search_locations:
            sys.exit("nvme-tier install: exllamav3 is not installed")
        pkg = pathlib.Path(next(iter(spec.submodule_search_locations))).resolve()
    if pkg.name != "exllamav3" or not (pkg / "generator" / "generator.py").is_file():
        sys.exit(f"nvme-tier install: {pkg} is not an exllamav3 package directory")
    return pkg.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg-dir", help = "installed exllamav3 directory (default: found on sys.path)")
    ap.add_argument("--check", action = "store_true", help = "verify only; copy nothing")
    args = ap.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text())
    root = site_root(args.pkg_dir)

    for rel, req in manifest.get("requires", {}).items():
        p = root / rel
        text = p.read_text() if p.is_file() else ""
        if req["contains"] not in text:
            sys.exit(f"nvme-tier install: {rel} lacks {req['why']}: the installed file "
                     f"({(sha(p) or 'absent')[:16]}) does not contain\n  {req['contains'].strip()}\n"
                     "Build ple-ckpt-clone r1 into the base first (its Dockerfile.box), then this overlay on top.")
        got = sha(p)
        print(f"nvme-tier install: {rel}: {req['why']} present "
              f"({'the tested file' if got == req.get('sha256_tested') else 'note: not the tested file ' + got[:16]})")

    for name, v in manifest["variants"].items():
        if all(sha(root / rel) == f["sha256"] for rel, f in v["post"].items()):
            print(f"nvme-tier install: already installed ({name}) in {root / 'exllamav3'}")
            return 0

    matches, report = [], []
    for name, v in manifest["variants"].items():
        bad = [f"{rel}: installed {(sha(root / rel) or 'absent')[:16]} != {(want or 'absent')[:16]}"
               for rel, want in v["pre"].items() if sha(root / rel) != want]
        bad += [f"{rel}: absent (required)" for rel in v.get("present", []) if not (root / rel).is_file()]
        if bad:
            report.append(f"  {name}: " + "; ".join(bad))
        else:
            matches.append(name)
    if len(matches) != 1:
        sys.exit("nvme-tier install: the installed tree matches " +
                 ("no" if not matches else "more than one") + " baseline:\n" + "\n".join(report))
    name = matches[0]
    variant = manifest["variants"][name]
    for rel, want in {**manifest.get("depends", {}), **variant.get("depends", {})}.items():
        got = sha(root / rel)
        if got != want:
            print(f"nvme-tier install: note: {rel} differs from the file this round was tested with "
              f"({(got or 'absent')[:16]} != {(want or 'absent')[:16]})")
    if args.check:
        print(f"nvme-tier install: baseline OK ({name}); --check, nothing copied")
        return 0
    for rel, f in variant["post"].items():
        dst = root / rel
        dst.parent.mkdir(parents = True, exist_ok = True)
        shutil.copyfile(HERE / f["from"], dst)
        for pyc in (dst.parent / "__pycache__").glob(dst.stem + ".*.pyc") if (dst.parent / "__pycache__").is_dir() else []:
            pyc.unlink()
    bad = [rel for rel, f in variant["post"].items() if sha(root / rel) != f["sha256"]]
    if bad:
        sys.exit("nvme-tier install: post-install hash mismatch: " + ", ".join(bad))
    print(f"nvme-tier install: {name}: {len(variant['post'])} files installed into {root / 'exllamav3'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
