#!/usr/bin/env python3
"""Landing check for prefill-merge r1 inside the built image (CPU, no GPU, EXL3_* stripped).

Asserts where the code comes from and that it is the patched code: the installed exllamav3's touched files hash to
SHA256SUMS.src, the new module's revision string, the hooks are present in the functions that must carry them, and
both knobs are off by default. Imports the real package (the image's extension), not the test stubs.
Prints one line `prefill-merge-r1 landed: ...` on success (the build script greps it); exits non-zero otherwise.
"""
import hashlib
import importlib.util
import inspect
import os
import sys

for k in list(os.environ):
    if k.startswith("EXL3_"):
        del os.environ[k]

here = os.path.dirname(os.path.abspath(__file__))
site = os.path.dirname(importlib.util.find_spec("exllamav3").origin)
want = {}
for line in open(os.path.join(here, "SHA256SUMS.src")):
    h, f = line.split()
    want[f] = h
bad = []
for f, h in sorted(want.items()):
    got = hashlib.sha256(open(os.path.join(site, f), "rb").read()).hexdigest()
    if got != h:
        bad.append(f"{f} {got[:12]} != {h[:12]}")
if bad:
    sys.exit("prefill-merge-r1 LANDING FAILED: installed files differ from src/: " + "; ".join(bad))

import exllamav3  # noqa: E402
import exllamav3.cache.prefill_merge as pm  # noqa: E402
from exllamav3.generator.job import Job  # noqa: E402
from exllamav3.modules.gated_delta_net import GatedDeltaNet, GDNState  # noqa: E402
from exllamav3.modules.ple import PLELayer  # noqa: E402
from exllamav3.generator import disk_cache  # noqa: E402

checks = {
    "revision": pm.REVISION == "prefill-merge-r1",
    "knobs off by default": not pm.merge_enabled() and not pm.async_enabled(),
    "Job.prefill hook": "_pm.try_split(" in inspect.getsource(Job.prefill)
                         and "_pm.stash_split(" in inspect.getsource(Job.prefill),
    "GDN split": hasattr(GatedDeltaNet, "_pm_split_conv_rule")
                 and "_prefill_split" in inspect.getsource(GatedDeltaNet.forward),
    "GDNState.stash async hook": "_pm.async_stash(" in inspect.getsource(GDNState.stash),
    "GDNState.unstash waits": "_pm.wait_stash(" in inspect.getsource(GDNState.unstash),
    "PLE split": hasattr(PLELayer, "_pm_split_streams")
                 and "_prefill_split" in inspect.getsource(PLELayer.forward),
    "serialize_stash waits": "_pm_ready" in inspect.getsource(disk_cache.serialize_stash),
    "loaded from site-packages": os.path.dirname(exllamav3.__file__) == site,
}
failed = [k for k, v in checks.items() if not v]
if failed:
    sys.exit("prefill-merge-r1 LANDING FAILED: " + ", ".join(failed))
print(f"prefill-merge-r1 landed: {len(want)} files match SHA256SUMS.src in {site}; "
      f"exllamav3 {getattr(exllamav3, '__version__', '?')}; hooks {len(checks)}/{len(checks)}; "
      f"EXL3_PREFILL_MERGE and EXL3_STASH_ASYNC default off", flush = True)
