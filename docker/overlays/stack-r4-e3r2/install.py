#!/usr/bin/env python3
"""Stack E3 grouped prefill r2 onto tabbyapi:decode-kernels-r4: verify the two shared files are exactly r4's, copy the merged
files and E3's new kernels into the installed package, verify every result hash. Fails the build on any mismatch."""
import hashlib, importlib.util, json, pathlib, shutil, sys
here = pathlib.Path(__file__).resolve().parent
m = json.loads((here / "manifest.json").read_text())
spec = importlib.util.find_spec("exllamav3")
pkg = pathlib.Path(next(iter(spec.submodule_search_locations))).resolve()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
bad = [f"{f}: installed {sha(pkg / f)[:16]} != r4 {h[:16]}" for f, h in m["pre"].items() if sha(pkg / f) != h]
if bad: sys.exit("stack install: base is not decode-kernels-r4: " + "; ".join(bad))
for f in m["post"]:
    (pkg / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(here / "overlay/exllamav3" / f, pkg / f)
bad = [f for f, h in m["post"].items() if sha(pkg / f) != h]
if bad: sys.exit("stack install: post-install hash mismatch: " + ", ".join(bad))
print(f"stack r4+e3r2 installed into {pkg}: {len(m['post'])} files")
