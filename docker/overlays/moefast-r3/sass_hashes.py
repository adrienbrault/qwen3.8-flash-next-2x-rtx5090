#!/usr/bin/env python3
"""cuobjdump -sass text (a FILE argument, or stdin) -> "<function> <sha256 of whitespace-collapsed instruction text>"
per function, sorted. Whitespace is collapsed because cuobjdump pads columns to the longest line of the cubin.
Duplicate function names (the same kernel in several cubins) keep one line per distinct body.
Same method as patches/exllamav3/pdl-l2-r1/sass_hashes.py (R705)."""
import hashlib, re, sys
txt = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
out = set()
for p in re.split(r"\n\s*Function : ", txt)[1:]:
    name = p.split("\n", 1)[0].strip()
    body = [" ".join(l.split()) for l in p.split("\n")[1:] if re.match(r"\s*/\*[0-9a-f]{4,}\*/", l)]
    out.add(f"{name} {hashlib.sha256(chr(10).join(body).encode()).hexdigest()}")
print("\n".join(sorted(out)))
