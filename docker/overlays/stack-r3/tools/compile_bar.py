#!/usr/bin/env python3
"""densegemm r2 compile bar: the gemm / mgemm V2 twins must not run out of local memory (R710).

R710 found every r1 gemm / mgemm twin with a 544-1832 byte per-thread stack frame (the served kernels of the same
shapes: 8-80) and REG 40 or 255: the fragment and pipeline state lived in local memory and the twins ran 3-6x slower.
`cuobjdump -res-usage` counts spills and demoted arrays under STACK, not LOCAL (r1 checked LOCAL).

  compile_bar.py --base <served res-usage> --twins <twin res-usage> [--sass <twin cuobjdump -sass>]
                 [--label NAME] [--margin 64]

Reference for a twin exl3_{gemm,mgemm}_v2_kernel<4, c_fp32, cb 2, TM, TK, TN, SH, FS> is the served
exl3_{gemm,mgemm}_kernel<4, c_fp32, 2, 16, TK, TN, SH, FS> (rows32 twins, TM 32, use the 16-row served shape).
Bar: twin STACK <= reference STACK + margin, for every 16-row twin (gates the EXL3_DENSE_V2 arms).
rows32 twins (EXL3_DENSE_ROWS32, piece 4a) get the same bar, reported separately as ROWS32 PASS|FAIL.
Last line: "BAR <label> PASS|FAIL rows16 <ok>/<n> rows32 <ok>/<n> max_stack16 <bytes> ROWS32 PASS|FAIL".
Exit 0 when the 16-row twins pass, 1 when they fail, 2 no data.
"""
import argparse
import re
import sys

KRE = re.compile(r"_Z\d+(exl3_(?:m?gemm)(?:_v2)?_kernel)ILi(\d+)ELb([01])ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EE")


def res_usage(path):
    """{(kind, bits, c_fp32, cb, TM, TK, TN, SH, FS): (REG, STACK)} (max over duplicate cubins)."""
    out = {}
    lines = open(path, errors="replace").read().splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"\s*Function (\S+):", line)
        if not m or i + 1 >= len(lines):
            continue
        k = KRE.match(m.group(1))
        if not k:
            continue
        d = dict(re.findall(r"(REG|STACK|LOCAL):(\d+)", lines[i + 1]))
        if "REG" not in d or "STACK" not in d:
            continue
        key = (k.group(1),) + tuple(int(x) for x in k.groups()[1:])
        reg, stack = int(d["REG"]), int(d["STACK"])
        old = out.get(key)
        out[key] = (max(reg, old[0]), max(stack, old[1])) if old else (reg, stack)
    return out


def sass_counts(path):
    """{key: (LDL, STL, CALL)} per twin function in a cuobjdump -sass dump."""
    out = {}
    txt = open(path, errors="replace").read()
    for part in re.split(r"\n\s*Function : ", txt)[1:]:
        name = part.split("\n", 1)[0].strip()
        k = KRE.match(name)
        if not k:
            continue
        key = (k.group(1),) + tuple(int(x) for x in k.groups()[1:])
        body = part.split("\n\t\t.....")[0]
        out[key] = (len(re.findall(r"\bLDL\b", body)), len(re.findall(r"\bSTL\b", body)),
                    len(re.findall(r"\bCALL\.", body)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--twins", required=True)
    ap.add_argument("--sass")
    ap.add_argument("--label", default="twins")
    ap.add_argument("--margin", type=int, default=64)
    a = ap.parse_args()
    base = res_usage(a.base)
    twins = {k: v for k, v in res_usage(a.twins).items() if k[0].endswith("_v2_kernel") and k[1] == 4 and k[3] == 2}
    sass = sass_counts(a.sass) if a.sass else {}
    if not twins:
        print(f"BAR {a.label} NO-DATA (no K4 cb2 gemm/mgemm twins in {a.twins})")
        sys.exit(2)
    ok16 = n16 = ok32 = n32 = 0
    max16 = 0
    print(f"{'twin':52s} {'REG':>4s} {'STACK':>6s} | {'served REG':>10s} {'STACK':>6s} | {'LDL':>4s} {'STL':>4s} {'CALL':>4s} | verdict")
    for key in sorted(twins, key=lambda k: (k[0], k[4], k[5], k[6], k[2])):
        kind, bits, c32, cb, tm, tk, tn, sh, fs = key
        reg, stack = twins[key]
        ref = base.get((kind.replace("_v2", ""), bits, c32, cb, 16, tk, tn, sh, fs))
        if ref is None:
            verdict = "NO-REF"
            ok = False
        else:
            ok = stack <= ref[1] + a.margin
            verdict = "ok" if ok else f"FAIL (> {ref[1]} + {a.margin})"
        if tm == 16:
            n16 += 1; ok16 += ok; max16 = max(max16, stack)
        else:
            n32 += 1; ok32 += ok
            verdict += " (rows32)"
        s = sass.get(key)
        sc = f"{s[0]:4d} {s[1]:4d} {s[2]:4d}" if s else f"{'-':>4s} {'-':>4s} {'-':>4s}"
        rr = f"{ref[0]:10d} {ref[1]:6d}" if ref else f"{'-':>10s} {'-':>6s}"
        name = f"{kind}<fp32={c32},{tm},{tk},{tn},{sh},{fs}>"
        print(f"{name:52s} {reg:4d} {stack:6d} | {rr} | {sc} | {verdict}")
    passed = n16 > 0 and ok16 == n16
    passed32 = n32 > 0 and ok32 == n32
    print(f"BAR {a.label} {'PASS' if passed else 'FAIL'} rows16 {ok16}/{n16} rows32 {ok32}/{n32} max_stack16 {max16} "
          f"ROWS32 {'PASS' if passed32 else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
