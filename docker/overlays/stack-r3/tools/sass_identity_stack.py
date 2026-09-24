#!/usr/bin/env python3
"""stack-r3 served-SASS identity: base (tabbyapi:stack-r2) vs rebuilt (stack-r2 + the series) function hashes, both
produced by sass_hashes.py ("<function> <sha256 of whitespace-collapsed instruction text>" per line).

  sass_identity_stack.py <base-sass.txt> <rebuilt-sass.txt> --include "mf3 dg1"

Rules:
  * every base function keeps its SASS (the same body hash under the same name). Functions in anonymous namespaces
    are matched with the TU hash in their mangled name blanked (_GLOBAL__N__<hex>_..._cu_<hex>): editing a .cu file
    can change that hash without changing the kernel.
  * the only new functions are the included components' kernels, with exact counts where the component fixes them:
      core  hcfast r2   hcv4::*                                  >= 1 (count printed; r1's hcv3::* must stay)
      core  latchain    cuda_recurrent_gated_delta_rule_kernel_128_rr   8
      mf3   moefast r3  exl3_moe_coop_r2_ns A/B 36 + head 1
      dg*   densegemm   exl3_gemm_v2 12 + exl3_mgemm_v2 12 + exl3_gemv_v2 8
Exit 0 = identity holds and the new set is exactly the expected one; 1 otherwise.
"""
import argparse
import re
import sys

ANON = [(re.compile(r"_GLOBAL__N__[0-9a-f]+_"), "_GLOBAL__N__X_"), (re.compile(r"_cu_[0-9a-f]{8}"), "_cu_X")]


def norm(name):
    for rx, rep in ANON:
        name = rx.sub(rep, name)
    return name


def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if line:
            name, h = line.rsplit(" ", 1)
            d.setdefault(norm(name), set()).add(h)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("rebuilt")
    ap.add_argument("--include", default="")
    a = ap.parse_args()
    inc = set(a.include.replace(",", " ").split())
    base, new = load(a.base), load(a.rebuilt)
    changed = sorted(n for n in base if n not in new or not base[n] <= new[n])
    added = sorted(n for n in new if n not in base)
    groups = {
        "hcv4": [n for n in added if n.startswith("_ZN4hcv4")],
        "rr": [n for n in added if "cuda_recurrent_gated_delta_rule_kernel_128_rr" in n],
        "moe-r2-ab": [n for n in added if "exl3_moe_coop_r2_ns" in n and re.search(r"exl3_moe_coop_r2_[ab]_kernel", n)],
        "moe-r2-head": [n for n in added if "exl3_moe_coop_r2_ns" in n and "exl3_moe_coop_r2_head_kernel" in n],
        "gemm-v2": [n for n in added if n.startswith("_Z19exl3_gemm_v2_kernel")],
        "mgemm-v2": [n for n in added if n.startswith("_Z20exl3_mgemm_v2_kernel")],
        "gemv-v2": [n for n in added if n.startswith("_Z19exl3_gemv_v2_kernel")],
    }
    dg = bool(inc & {"dg1", "dg2"})
    want = {"hcv4": ">=1", "rr": 8,
            "moe-r2-ab": 36 if "mf3" in inc else 0, "moe-r2-head": 1 if "mf3" in inc else 0,
            "gemm-v2": 12 if dg else 0, "mgemm-v2": 12 if dg else 0, "gemv-v2": 8 if dg else 0}
    grouped = {n for v in groups.values() for n in v}
    other = [n for n in added if n not in grouped]
    bad = []
    for g, w in want.items():
        got = len(groups[g])
        ok = got >= 1 if w == ">=1" else got == w
        if not ok:
            bad.append(f"{g}: {got} new, expected {w}")
    hcv3_base = [n for n in base if n.startswith(("_ZN4hcv3", "_ZN11hcv3_served"))]
    print(f"stack-r3 SASS: base functions {len(base)}; unchanged {len(base) - len(changed)}; changed or missing {len(changed)}; "
          f"new {len(added)} = " + ", ".join(f"{g} {len(v)}" for g, v in groups.items()) + f", other {len(other)}; "
          f"hcfast r1 kernels in base {len(hcv3_base)} (must be unchanged)")
    for n in changed[:30]:
        print("  CHANGED", n)
    for n in other[:30]:
        print("  UNEXPECTED NEW", n)
    for b in bad:
        print("  COUNT", b)
    if changed or other or bad:
        print(f"stack-r3: SERVED SASS IDENTITY FAILED (INCLUDE='{' '.join(sorted(inc))}')")
        return 1
    print(f"stack-r3: served SASS identical (anonymous-namespace hashes blanked); new kernels exactly the INCLUDE='{' '.join(sorted(inc))}' set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
