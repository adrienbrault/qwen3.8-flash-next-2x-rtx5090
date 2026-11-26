#!/usr/bin/env python3
"""rebase165 SASS / function inventory (CPU only, no GPU).

  sass_inventory.py hashes  < cuobjdump-sass.txt  > hashes.txt
      "<function> <sha256 of whitespace-collapsed instruction text>" per function body, sorted (the tune268
      sass_hashes.py format; exits 1 when a function has no instruction line).
  sass_inventory.py compare served-hashes.txt new-hashes.txt report.txt
      Per function name: same / changed / only-served / only-new, with a summary header and the tune268 /
      served-twin kernel families called out. Informational: exits 0 (the operator decides). Exits 2 only on
      malformed input.
  sass_inventory.py pybind served-dir.txt new-dir.txt [allowed-dropped.txt]
      Every attribute the served extension exports must exist in the new one, except the names listed in
      allowed-dropped.txt (intentional drops, see AUDIT.md). Exits 1 on a missing name.
  sass_inventory.py require-tune268 sm_120-hashes.txt
      Require tune268 K6 dense, K2.5 MoE coop and K2.5 E3 kernel symbols.
"""
import hashlib
import re
import sys

# Served-only kernel families (tune268 + earlier perf rounds) to call out in the report. Substring match on
# the (mangled or demangled) function name
FAMILIES = {
    "dense_v2 (densegemm r1/r2, tune268 K6 twins)": ("dense_v2", "gemm_v2", "gemv_v2", "exl3_gemm_kernel_v2", "exl3_gemv_kernel_v2"),
    "moe coop v2/v3/r2 (+ tune268 half-K)": ("moe_coop_v2", "moe_coop_v3", "moe_coop_r2", "coop_v2", "coop_v3", "coop_r2"),
    "moe prefill E3 (+ tune268-r2 half)": ("prefill_e3", "e3_"),
    "hc mix v2/v3": ("gr_mix_v2", "gr_mix_v3", "hc_mix_v3", "mix_v2", "mix_v3"),
    "gdn": ("gated_delta", "scan_replay", "gdn_ba_gemv", "conv1d_update"),
    "attention / qsa": ("attn", "qsa"),
}

# Mangled CUDA C++ names. Li6E is the dense K6 template argument. E3 uses the
# tag 5 for K2.5. Each half-K coop kernel has its own _h symbol.
REQUIRED_TUNE268 = {
    "dense K6 GEMM": r"exl3_gemm_v2_kernelILi6E",
    "dense K6 MGEMM": r"exl3_mgemm_v2_kernelILi6E",
    **{f"coop {version} half-K stage {stage.upper()}": pattern + "ILi2E"
       for stage in ("a", "b")
       for version, pattern in (("v2", rf"exl3_moe_coop_v2_ns.*exl3_moe_coop_{stage}_kernel_h"),
                                ("v3", rf"exl3_moe_coop_v3_{stage}_kernel_h"),
                                ("r2", rf"exl3_moe_coop_r2_{stage}_kernel_h"))},
    "E3 K2.5 gate/up": r"e3_gateup_kernelILi5ELi5E",
    "E3 K2.5 down": r"e3_down_kernelILi5E",
    "E3 K2.5 deterministic down": r"e3_down_det_kernelILi5E",
}


def require_tune268(path):
    names = load(path)
    missing = []
    for family, pattern in REQUIRED_TUNE268.items():
        matches = [name for name in names if re.search(pattern, name)]
        print(f"{family}: {len(matches)} sm_120 symbols", flush=True)
        if not matches:
            missing.append(family)
    if missing:
        raise SystemExit(f"missing required sm_120 kernels in {path}: {', '.join(missing)}")


def hashes(txt):
    out = set()
    empty = []
    for p in re.split(r"\n\s*Function : ", txt)[1:]:
        name = p.split("\n", 1)[0].strip()
        body = [" ".join(l.split()) for l in p.split("\n")[1:] if re.match(r"\s*/\*[0-9a-f]{4,}\*/", l)]
        if not body:
            empty.append(name)
        out.add(f"{name} {hashlib.sha256(chr(10).join(body).encode()).hexdigest()}")
    return sorted(out), empty


def load(path):
    d = {}
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        name, _, h = line.rpartition(" ")
        if not name or not re.fullmatch(r"[0-9a-f]{64}", h):
            print(f"malformed line in {path}: {line[:120]}", file = sys.stderr)
            sys.exit(2)
        d.setdefault(name, set()).add(h)
    return d


def compare(a_path, b_path, out_path):
    a, b = load(a_path), load(b_path)
    same = sorted(n for n in a if n in b and a[n] == b[n])
    changed = sorted(n for n in a if n in b and a[n] != b[n])
    only_a = sorted(n for n in a if n not in b)
    only_b = sorted(n for n in b if n not in a)
    lines = [
        f"served functions {len(a)}, new functions {len(b)}",
        f"same SASS {len(same)}, changed SASS {len(changed)}, only in served {len(only_a)}, only in new {len(only_b)}",
        "",
    ]
    for fam, keys in FAMILIES.items():
        sel = lambda names: [n for n in names if any(k in n for k in keys)]
        lines.append(f"[{fam}] same {len(sel(same))}, changed {len(sel(changed))}, "
                     f"only-served {len(sel(only_a))}, only-new {len(sel(only_b))}")
    for title, names in (("CHANGED", changed), ("ONLY-SERVED", only_a), ("ONLY-NEW", only_b), ("SAME", same)):
        lines.append("")
        lines.append(f"== {title} ({len(names)})")
        lines.extend(names)
    open(out_path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:3 + len(FAMILIES)]))


def pybind(a_path, b_path, allow_path = None):
    a = {l.strip() for l in open(a_path) if l.strip()}
    b = {l.strip() for l in open(b_path) if l.strip()}
    allow = {l.strip() for l in open(allow_path) if l.strip() and not l.startswith("#")} if allow_path else set()
    for n in sorted(allow & (a - b)):
        print(f"  dropped (allowed) {n}")
    missing = sorted(a - b - allow)
    added = sorted(b - a)
    print(f"pybind: served {len(a)}, new {len(b)}, missing in new {len(missing)}, added {len(added)}")
    for n in missing:
        print(f"  MISSING {n}")
    for n in added:
        print(f"  added {n}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "hashes":
        out, empty = hashes(sys.stdin.read())
        print("\n".join(out))
        if empty or not out:
            print(f"sass_inventory: {len(empty)} functions without instruction lines, {len(out)} functions", file = sys.stderr)
            sys.exit(1)
    elif cmd == "compare" and len(sys.argv) == 5:
        compare(*sys.argv[2:5])
    elif cmd == "require-tune268" and len(sys.argv) == 3:
        require_tune268(sys.argv[2])
    elif cmd == "pybind" and len(sys.argv) in (4, 5):
        pybind(*sys.argv[2:])
    else:
        print(__doc__, file = sys.stderr)
        sys.exit(2)
