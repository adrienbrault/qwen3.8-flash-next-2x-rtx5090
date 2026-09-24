#!/usr/bin/env python3
"""moefast r3 served-SASS identity: base (stack-r2) vs rebuilt (stack-r2 + moefast-r3.patch) function hashes
(sass_hashes.py output files). Every base function must keep its SASS; the only new functions are r2's kernels A/B
(namespace exl3_moe_coop_r2_ns, 3 K x 3 codebooks x 2 tile widths x 2 kernels = 36) and r3's head kernel (1).
Exit 1 on anything else."""
import re, sys


def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if line:
            name, h = line.rsplit(" ", 1)
            d.setdefault(name, set()).add(h)
    return d


base, new = load(sys.argv[1]), load(sys.argv[2])
changed = [n for n in base if n not in new or not base[n] <= new[n]]
added = sorted(n for n in new if n not in base)
r2ab = [n for n in added if "exl3_moe_coop_r2_ns" in n and re.search(r"exl3_moe_coop_r2_[ab]_kernel", n)]
head = [n for n in added if "exl3_moe_coop_r2_ns" in n and "exl3_moe_coop_r2_head_kernel" in n]
other = [n for n in added if n not in r2ab and n not in head]
print(f"moefast-r3: base functions {len(base)}; unchanged {len(base) - len(changed)}; changed or missing {len(changed)}; "
      f"new {len(added)} (r2 A/B {len(r2ab)}, r3 head {len(head)}, other {len(other)})")
for n in changed[:20]:
    print("  CHANGED", n)
for n in other[:20]:
    print("  UNEXPECTED NEW", n)
if changed or other or len(r2ab) != 36 or len(head) != 1:
    print("moefast-r3: SERVED SASS IDENTITY FAILED (expected 0 changed, 36 r2 A/B, 1 head, 0 other)")
    sys.exit(1)
print("moefast-r3: served SASS identical (whitespace collapsed); 36 r2 kernels + 1 head kernel added")
