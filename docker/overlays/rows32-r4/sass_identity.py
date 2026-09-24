#!/usr/bin/env python3
"""SASS identity for a host-only patch: base.txt rebuilt.txt report.txt (sass_hashes.py lists).
Every (function, hash) of the base .so must be in the rebuilt .so and no function may be added. The report file gets
one summary line and one line per changed / added function; exit 1 on any."""
import sys


def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if line:
            name, h = line.rsplit(" ", 1)
            d.setdefault(name, set()).add(h)
    return d


base, new = load(sys.argv[1]), load(sys.argv[2])
changed = sorted(n for n in base if n not in new or not base[n] <= new[n])
added = sorted(n for n in new if n not in base)
with open(sys.argv[3], "w") as f:
    f.write(f"functions base {len(base)} rebuilt {len(new)} unchanged {len(base) - len(changed)} "
            f"changed-or-missing {len(changed)} added {len(added)}\n")
    for n in changed:
        f.write(f"CHANGED {n}\n")
    for n in added:
        f.write(f"ADDED {n}\n")
print("rows32: " + open(sys.argv[3]).readline().strip())
for n in (changed + added)[:20]:
    print("  ", n)
if changed or added or not base:
    print("rows32: SASS IDENTITY FAILED (host-only patch: expected 0 changed, 0 added, base non-empty)")
    sys.exit(1)
print("rows32: SASS identical, no function added (host-only patch)")
