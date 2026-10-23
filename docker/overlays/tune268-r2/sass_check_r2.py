#!/usr/bin/env python3
"""tune268-r2 SASS identity: base.txt rebuilt.txt report.txt (sass_hashes.py lists).
Every (function, hash) of the r1 .so (served image tune268-r1, incl. its 37 r1 kernels) must be unchanged in the
rebuilt .so; exactly 14 functions may be added (codex 2b: 2a = 5, E3 half = 9). Exit 1 otherwise."""
import sys

EXPECTED_ADDED = 14


def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if line:
            name, h = line.rsplit(" ", 1)
            d.setdefault(name, set()).add(h)
    return d


base, new = load(sys.argv[1]), load(sys.argv[2])
changed = sorted(n for n in base if new.get(n) != base[n])
added = sorted(n for n in new if n not in base)
with open(sys.argv[3], "w") as f:
    f.write(f"functions base {len(base)} rebuilt {len(new)} unchanged {len(base) - len(changed)} "
            f"changed-or-missing {len(changed)} added {len(added)} (expected {EXPECTED_ADDED})\n")
    for n in changed:
        f.write(f"CHANGED {n}\n")
    for n in added:
        f.write(f"ADDED {n}\n")
print("tune268-r2: " + open(sys.argv[3]).readline().strip())
if changed or not base or len(added) != EXPECTED_ADDED:
    for n in (changed + added)[:30]:
        print("  ", n)
    print("tune268-r2: SASS IDENTITY FAILED")
    sys.exit(1)
print("tune268-r2: every r1 function's SASS unchanged; 14 functions added")
