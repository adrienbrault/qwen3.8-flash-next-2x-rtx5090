#!/usr/bin/env python3
"""stack-r3 install STACK bound (R710 lesson): every NEW kernel of an included component keeps its per-thread stack
frame within its served reference + 64 bytes, read from `cuobjdump -res-usage` STACK (spills and demoted arrays land
there; LOCAL stays 0 and hid R710's 544-1832-byte twins).

  stack_bound.py --base <res-usage of the stack-r2 .so> --new <res-usage of the rebuilt .so> --include "mf3 dg1"
                 [--margin 64]

Reference per new kernel: the served kernel(s) of the same family whose template arguments START WITH the new
kernel's (densegemm's gemv twin <K,fp32,cb,mmode,cfg> vs the served <...,smem>; latchain's _rr vs the served _128);
if none, the maximum STACK over the served family (hcfast r2 hcv4 vs the served hcv3 mixers, moefast r2 vs the
served moefast r1 V3 kernels). Enforced rules fail the install; report-only rules print their rows:
  hc2   hcv4::*                                   ref family hcv3 / hcv3_served            enforced
  lc    cuda_recurrent_..._kernel_128_rr<...>     ref cuda_recurrent_..._kernel_128<...>   enforced
  mf3   exl3_moe_coop_r2_ns::*                    ref family exl3_moe_coop_v3_ns::*        enforced when mf3
  dg    exl3_gemv_v2_kernel<...>                  ref exl3_gemv_kernel<..., smem>          enforced when dg1|dg2
  dg    exl3_{gemm,mgemm}_v2_kernel<...>          ref the served 16-row shape (compile_bar.py's mapping)
                                                   enforced when dg2; REPORT-ONLY when dg1: r1's gemm twins spill
                                                   (R710) and only mode 3 (gemv twin) may run on a dg1 image
Last line: "STACK-BOUND PASS|FAIL <n ok>/<n enforced> enforced; <n> report-only (<n over>) over bar".
Exit 0 on PASS, 1 on FAIL, 2 when an included component has no new kernels in --new (wrong build).
"""
import argparse
import re
import sys


def res_usage(path):
    """{mangled: STACK} (max over duplicate cubins)."""
    out = {}
    lines = open(path, errors="replace").read().splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"\s*Function (\S+?):?\s*$", line)
        if not m or i + 1 >= len(lines):
            continue
        s = re.search(r"STACK:(\d+)", lines[i + 1])
        if not s:
            continue
        name = m.group(1).rstrip(":")
        out[name] = max(int(s.group(1)), out.get(name, 0))
    return out


def split_name(mangled):
    """(qualified name without length prefixes, template-argument string up to the return type) of a mangled kernel.
    _Z16exl3_gemv_kernelILi4E...ELb0EEvPK.. -> ('exl3_gemv_kernel', 'ILi4E...ELb0EE'); _ZN4hcv4..kernelI..EEEv.. ->
    ('hcv4::..kernel', 'I..EEE'). Kernels return void, so the argument list ends at the first 'Ev'."""
    if not mangled.startswith("_Z"):
        return mangled, ""
    s = 2
    nested = mangled[s:s + 1] == "N"
    s += nested
    parts = []
    while s < len(mangled) and mangled[s].isdigit():
        n = re.match(r"\d+", mangled[s:]).group(0)
        k = int(n)
        s += len(n)
        parts.append(mangled[s:s + k])
        s += k
        if not nested:
            break
    rest = mangled[s:]
    targs = rest[:rest.index("Ev") + 1] if rest.startswith("I") and "Ev" in rest else ""
    return "::".join(parts), targs


# Dense twin <-> served shape key (compile_bar.py's KRE): kind, bits, c_fp32, cb, TM, TK, TN, SH, FS
KRE = re.compile(r"_Z\d+(exl3_(?:m?gemm)(?:_v2)?_kernel)ILi(\d+)ELb([01])ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EE")

RULES = [
    # key, component, new-kernel test, family test (served reference set)
    ("hcv4", "core", lambda q: q.startswith("hcv4::"), lambda q: q.startswith(("hcv3::", "hcv3_served::"))),
    ("rr", "core", lambda q: q == "cuda_recurrent_gated_delta_rule_kernel_128_rr",
     lambda q: q == "cuda_recurrent_gated_delta_rule_kernel_128"),
    ("moe-r2", "mf3", lambda q: q.startswith("exl3_moe_coop_r2_ns::"), lambda q: q.startswith("exl3_moe_coop_v3_ns::")),
    ("gemv-twin", "dg", lambda q: q == "exl3_gemv_v2_kernel", lambda q: q == "exl3_gemv_kernel"),
    ("gemm-twin", "dg-gemm", lambda q: q in ("exl3_gemm_v2_kernel", "exl3_mgemm_v2_kernel"), None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--include", default="")
    ap.add_argument("--margin", type=int, default=64)
    a = ap.parse_args()
    inc = set(a.include.replace(",", " ").split())
    base, new = res_usage(a.base), res_usage(a.new)
    added = {n: s for n, s in new.items() if n not in base}
    enforce = {"core": True, "mf3": "mf3" in inc, "dg": bool(inc & {"dg1", "dg2"}), "dg-gemm": "dg2" in inc}
    present = {"core": True, "mf3": "mf3" in inc, "dg": bool(inc & {"dg1", "dg2"}), "dg-gemm": bool(inc & {"dg1", "dg2"})}
    base_q = {}
    for n, s in base.items():
        q, t = split_name(n)
        base_q.setdefault(q, []).append((t, s, n))
    n_enf = ok_enf = n_rep = over_rep = 0
    missing = []
    print(f"{'rule':10s} {'kernel':70s} {'STACK':>5s} {'ref':>5s} {'bar':>5s}  verdict (reference)")
    for key, comp, is_new, fam in RULES:
        rows = []
        for n, s in sorted(added.items()):
            q, t = split_name(n)
            if not is_new(q):
                continue
            if key == "gemm-twin":
                k = KRE.match(n)
                ref, how = None, "no served shape"
                if k:
                    kind, bits, c32, cb, tm, tk, tn, sh, fs = k.groups()
                    want = (kind.replace("_v2", ""), bits, c32, cb, "16", tk, tn, sh, fs)
                    for bn, bs in base.items():
                        bk = KRE.match(bn)
                        if bk and bk.groups() == want:
                            ref, how = bs, f"served 16-row shape {bn[:48]}"
                            break
            else:
                cands = [(bt, bs, bn) for bq, lst in base_q.items() if fam(bq) for (bt, bs, bn) in lst]
                exact = [c for c in cands if t and c[0].startswith(t[:-1] if t.endswith("E") else t)]
                if exact:
                    ref, how = max(c[1] for c in exact), f"template-prefix match ({len(exact)})"
                elif cands:
                    ref, how = max(c[1] for c in cands), f"family max over {len(cands)} served"
                else:
                    ref, how = None, "NO served reference"
            rows.append((n, s, ref, how))
        if present[comp] and not rows:
            missing.append(key)
        for n, s, ref, how in rows:
            bar = None if ref is None else ref + a.margin
            good = bar is not None and s <= bar
            if enforce[comp]:
                n_enf += 1; ok_enf += good
                verdict = "ok" if good else "FAIL"
            else:
                n_rep += 1; over_rep += not good
                verdict = ("ok" if good else "over") + " (report-only)"
            print(f"{key:10s} {n[:70]:70s} {s:5d} {'-' if ref is None else ref:>5} {'-' if bar is None else bar:>5}  {verdict} ({how})")
    if missing:
        print(f"STACK-BOUND NO-DATA: included components without new kernels in {a.new}: {missing}")
        sys.exit(2)
    passed = ok_enf == n_enf
    print(f"STACK-BOUND {'PASS' if passed else 'FAIL'} {ok_enf}/{n_enf} enforced; {n_rep} report-only ({over_rep} over bar); margin {a.margin} B")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
