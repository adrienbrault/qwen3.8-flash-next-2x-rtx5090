#!/usr/bin/env python3
"""Install the MTP pruned-embedding-mirror round 1 overlay onto tabbyapi:stack-r4-e3r2: verify every changed file is exactly
the stack's copy and every new file is absent, copy the overlay into the installed package, verify every result hash.
Fails the build on any mismatch. Pure Python: no extension rebuild."""
import hashlib, importlib.util, json, pathlib, shutil, sys
here = pathlib.Path(__file__).resolve().parent
m = json.loads((here / "manifest.json").read_text())
spec = importlib.util.find_spec("exllamav3")
pkg = pathlib.Path(next(iter(spec.submodule_search_locations))).resolve()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
bad = [f"{f}: installed {sha(pkg / f)[:16] if (pkg / f).exists() else 'MISSING'} != stack {h[:16]}"
       for f, h in m["pre"].items() if not (pkg / f).exists() or sha(pkg / f) != h]
bad += [f"{f}: already present" for f in m["new"] if (pkg / f).exists()]
if bad: sys.exit("mtp-pruned-r1 install: base is not stack-r4-e3r2: " + "; ".join(bad))
for f in m["post"]:
    src = here / "overlay/exllamav3" / f
    if sha(src) != m["post"][f]: sys.exit(f"mtp-pruned-r1 install: overlay file {f} does not match the manifest")
    (pkg / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, pkg / f)
bad = [f for f, h in m["post"].items() if sha(pkg / f) != h]
if bad: sys.exit("mtp-pruned-r1 install: post-install hash mismatch: " + ", ".join(bad))
print(f"mtp-pruned-r1 installed into {pkg}: {len(m['post'])} files ({len(m['new'])} new)")
