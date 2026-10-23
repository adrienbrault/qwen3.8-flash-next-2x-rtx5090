#!/usr/bin/env python3
"""tune268 SASS identity: base.txt rebuilt.txt report.txt (sass_hashes.py lists).
Every (function, hash) of the served .so must be in the rebuilt .so unchanged; added functions must be the tune268
kernels only (K6 dense V2 twins, the *_kernel_h half-integer MoE kernels). Exit 1 otherwise."""
import re
import sys

EXPECTED_ADDED = 37   # K6 twins: 2 kernels x 2 c_fp32 x 6 shapes = 24; *_kernel_h<2>: V2 4 + V3 5 + r2 4 = 13
ALLOWED_ADDED = re.compile(r"(exl3_m?gemm_v2_kernelILi6E|_kernel_hILi2E)")   # mangled: <6, ...> / _h<2, ...>


def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if line:
            name, h = line.rsplit(" ", 1)
            d.setdefault(name, set()).add(h)
    return d


base, new = load(sys.argv[1]), load(sys.argv[2])
changed = sorted(n for n in base if new.get(n) != base[n])          # equality: no extra body for a served name
added = sorted(n for n in new if n not in base)
bad_added = [n for n in added if not ALLOWED_ADDED.search(n)]
with open(sys.argv[3], "w") as f:
    f.write(f"functions base {len(base)} rebuilt {len(new)} unchanged {len(base) - len(changed)} "
            f"changed-or-missing {len(changed)} added {len(added)} (unexpected {len(bad_added)})\n")
    for n in changed:
        f.write(f"CHANGED {n}\n")
    for n in added:
        f.write(f"ADDED {n}\n")
print("tune268: " + open(sys.argv[3]).readline().strip())
for n in (changed + bad_added)[:20]:
    print("  ", n)
if changed or bad_added or not base or len(added) != EXPECTED_ADDED:
    print("tune268: SASS IDENTITY FAILED (expected every served function unchanged and only tune268 kernels added)")
    sys.exit(1)
print("tune268: every served function's SASS unchanged; added only the tune268 kernels")
