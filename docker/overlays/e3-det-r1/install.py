#!/usr/bin/env python3
"""e3-det r1 onto tabbyapi:stack-r4-e3r2: verify every file it replaces is exactly the stack-r4-e3r2 copy, copy the overlay
into the installed exllamav3 package, verify every result hash. Fails the build on any mismatch."""
import hashlib, importlib.util, json, pathlib, shutil, sys
here = pathlib.Path(__file__).resolve().parent
m = json.loads((here / "manifest.json").read_text())
spec = importlib.util.find_spec("exllamav3")
if spec is None or not spec.submodule_search_locations:
    sys.exit("e3-det install: cannot locate the installed exllamav3 package")
pkg = pathlib.Path(next(iter(spec.submodule_search_locations))).resolve()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
bad = [f"{f}: installed {sha(pkg / f)[:16] if (pkg / f).exists() else 'MISSING'} != stack-r4-e3r2 {h[:16]}"
       for f, h in m["pre"].items() if not (pkg / f).exists() or sha(pkg / f) != h]
if bad: sys.exit("e3-det install: base is not tabbyapi:stack-r4-e3r2: " + "; ".join(bad))
bad = [f for f, h in m["post"].items() if sha(here / "overlay/exllamav3" / f) != h]
if bad: sys.exit("e3-det install: overlay payload hash mismatch: " + ", ".join(bad))
for f in m["post"]:
    shutil.copyfile(here / "overlay/exllamav3" / f, pkg / f)
bad = [f for f, h in m["post"].items() if sha(pkg / f) != h]
if bad: sys.exit("e3-det install: post-install hash mismatch: " + ", ".join(bad))
print(f"e3-det r1 installed into {pkg}: {len(m['post'])} files")
