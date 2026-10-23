#!/usr/bin/env python3
"""cuobjdump -sass text on stdin -> "<function> <sha256 of whitespace-collapsed instruction text>" per function, sorted.
Whitespace is collapsed because cuobjdump pads columns to the longest line of the cubin. Duplicate function names
(the same kernel in several cubins) keep one line per distinct body. tune268: exits 1 when a function has no
instruction line (a cuobjdump format change would otherwise hash every body to sha256("") and pass an identity check)."""
import hashlib, re, sys
txt = sys.stdin.read()
out = set()
empty = []
for p in re.split(r"\n\s*Function : ", txt)[1:]:
    name = p.split("\n", 1)[0].strip()
    body = [" ".join(l.split()) for l in p.split("\n")[1:] if re.match(r"\s*/\*[0-9a-f]{4,}\*/", l)]
    if not body:
        empty.append(name)
    out.add(f"{name} {hashlib.sha256(chr(10).join(body).encode()).hexdigest()}")
print("\n".join(sorted(out)))
if empty or not out:
    print(f"sass_hashes: {len(empty)} functions without instruction lines, {len(out)} functions", file=sys.stderr)
    sys.exit(1)
