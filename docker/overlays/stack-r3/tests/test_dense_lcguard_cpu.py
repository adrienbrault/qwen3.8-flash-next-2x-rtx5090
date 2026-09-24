#!/usr/bin/env python3
"""stack-r3 CPU test of the densegemm side-branch guard (series/06-dense-lcguard.patch), no GPU, no torch.

The guard: a dense call captured on latchain's QSA side branch (EXL3_LC_QSA_FORK) passes locks = the device's lock
base + LC_LOCKS_ALT_OFFSET. dense_v2_launch_gemm / dense_v2_launch_gemv detect that from the call's own locks
argument (kernel_args[6]) and give the call its own V2 scratch: the gemv twin counts in the upper half of hctr (the
main branch in the lower half), the gemm twin writes a separate side slot region (allocated only when a gemm twin can
run AND EXL3_LC_QSA_FORK=1); a side call that does not fit takes the served kernel. Checked on a patched tree:

  A. compile + run (host C++ compiler) of code EXTRACTED from the tree, not retyped:
     - exl3_gemm_gr's locks line (`get_locks(device) + lc_gemm_locks_offset`) and kernelArgs initializer
       (quant/exl3_gemm.cu); the standalone exl3_gemv entry's locks line and kernel_args (quant/exl3_gemv.cu);
     - dense_v2_on_side_branch (exl3_dense_v2.cuh), lc_side_branch and lc_fork_requested (exl3_dense_v2.cu), the
       hctr selection line of dense_v2_launch_gemv, and the workspace layout lines of ws_ready;
     - MAX_TILES_C (exl3_devctx.cuh), LC_LOCKS_ALT_OFFSET (libtorch/lc_fork.h), DENSE_V2_* sizes.
     Cases: offset 0 (every main-branch call; every call without latchain) -> main; LC_LOCKS_ALT_OFFSET -> side;
     standalone exl3_gemv -> main; the side hctr pointer is base + H/2, the main one base; logged once per device;
     workspace byte ranges (main slots, side slots, hctr lower half, hctr upper half) pairwise disjoint in every
     mode x fork configuration, side slots 0 bytes unless a gemm twin can run and the fork is requested; the largest
     served gemv call (size_n 2560 = 20 blocks) and index_qk (5 blocks) fit a half.
  B. source lints: `bool side = lc_side_branch(kernel_args, device);` is the first statement of both launchers;
     dense_v2_launch_mgemm is untouched (22-argument layout; exl3_mgemm_gr never runs on the side branch) and keeps
     g_ws.slots; the main-branch statements of both launchers are the r1/r2 statements (the diff against the
     unpatched densegemm tree only adds side handling); the densegemm files reference no latchain symbol in code;
     the only non-zero exl3_gemm_set_locks_offset call is latchain's side-branch call, reset by RAII; bindings
     expose dense_v2_lc_guard.
  C. resource model of the concurrency windows: QF window (main: q/k/v projections; side: index_qk), moefast
     shared-expert window (side stream: shared gate_up mgemm + shared.down gemv; main: fp16 router + routed coop A,
     joined before stage B): with the guard no two concurrent twins share an array; without it modes 1/2 at 16 rows
     share slots (R712 finding) and an unfused-q/k/v config shares hctr in modes 1/3 (so the model is not vacuous).

  test_dense_lcguard_cpu.py --tree <exllamav3 package dir with the series applied incl. dg1|dg2> [--untouched <same
                            tree without 06-dense-lcguard>]
  test_dense_lcguard_cpu.py --build --src S [--patches P --series DIR]  (rebuilds dg1 and dg2 trees, with and without
                            the guard)
Exit 0 = all PASS.
"""
import argparse, difflib, os, re, shutil, subprocess, sys, tempfile

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def strip_comments(src):
    """Code only: string literals, block and line comments removed (a message that says 'latchain' is not a use)."""
    src = re.sub(r'"(?:\\.|[^"\\\n])*"', '""', src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def extract_block(src, start_pat, what):
    """Text from the first match of start_pat through its balanced closing brace (and a following ';')."""
    m = re.search(start_pat, src)
    if not m:
        raise SystemExit(f"cannot find {what}")
    j = src.index("{", m.end() - 1 if src[m.end() - 1] == "{" else m.end())
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                end = k + 1 + (src[k + 1:k + 2] == ";")
                return src[m.start():end]
    raise SystemExit(f"unbalanced {what}")


def compiler():
    for c in ("c++", "g++", "clang++"):
        if shutil.which(c):
            return c
    return None


def one_line(src, pat, what):
    m = re.search(pat, src)
    if not m:
        raise SystemExit(f"cannot find {what}")
    return m.group(0)


def part_a(tree, work):
    q = os.path.join(tree, "exllamav3_ext", "quant")
    gemm = open(os.path.join(q, "exl3_gemm.cu")).read()
    gemv = open(os.path.join(q, "exl3_gemv.cu")).read()
    hdr = open(os.path.join(q, "exl3_dense_v2.cuh")).read()
    dv2 = open(os.path.join(q, "exl3_dense_v2.cu")).read()
    devctx = open(os.path.join(q, "exl3_devctx.cuh")).read()
    lcf_path = os.path.join(tree, "exllamav3_ext", "libtorch", "lc_fork.h")
    has_lc = os.path.exists(lcf_path)

    defs = "\n".join([one_line(devctx, r"#define MAX_TILES_C .*", "MAX_TILES_C"),
                      one_line(open(lcf_path).read(), r"#define LC_LOCKS_ALT_OFFSET .*", "LC_LOCKS_ALT_OFFSET") if has_lc
                      else "#define LC_LOCKS_ALT_OFFSET (MAX_TILES_C / 2)   // latchain absent: the value it would use",
                      one_line(hdr, r"#define DENSE_V2_GEMV_MAX_HBLOCKS .*", "MAX_HBLOCKS"),
                      one_line(dv2, r"#define DENSE_V2_WS_BYTES .*", "WS_BYTES"),
                      one_line(dv2, r"#define DENSE_V2_WS_BYTES_ROWS32 .*", "WS_BYTES_ROWS32"),
                      one_line(dv2, r"#define DENSE_V2_WS_SIDE_BYTES .*", "WS_SIDE_BYTES")])
    gr = extract_block(gemm, r"\nint exl3_gemm_gr\s*\(", "exl3_gemm_gr")
    locks_line = one_line(gr, r"int\* locks = DevCtx::instance\(\)\.get_locks\(device\)[^;]*;", "exl3_gemm_gr locks")
    kargs = extract_block(gr, r"void\* kernelArgs\[\] =\s*\{", "exl3_gemm_gr kernelArgs")
    gv = extract_block(gemv, r"\nvoid exl3_gemv\s*\(", "exl3_gemv")
    gv_locks = one_line(gv, r"int\* locks = DevCtx::instance\(\)\.get_locks\(device\);", "exl3_gemv locks")
    gv_args = extract_block(gv, r"void\* kernel_args\[\] =\s*\{", "exl3_gemv kernel_args")
    helper = extract_block(hdr, r"inline bool dense_v2_on_side_branch\(", "dense_v2_on_side_branch")
    side_fn = extract_block(dv2, r"static bool lc_side_branch\(", "lc_side_branch")
    fork_fn = extract_block(dv2, r"static bool lc_fork_requested\(", "lc_fork_requested")
    gemv_fn = extract_block(dv2, r"\nvoid\* dense_v2_launch_gemv\n\(", "dense_v2_launch_gemv")
    hctr_line = one_line(gemv_fn, r"int\* hctr = g_ws\[device\]\.hctr[^;]*;", "hctr selection")
    ws_fn = extract_block(dv2, r"\nstatic bool ws_ready\(", "ws_ready")
    side_bytes_line = one_line(ws_fn, r"size_t side_bytes = [^;]*;", "side_bytes")
    layout = [one_line(ws_fn, r"w\.side_slots = [^;]*;", "side_slots"), one_line(ws_fn, r"w\.hctr = [^;]*;", "hctr")]
    check(("lc_gemm_locks_offset" in locks_line) == has_lc,
          f"exl3_gemm_gr locks line is the {'latchain' if has_lc else 'served'} form: {locks_line}")

    prog = f"""
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
{defs}
#define MAX_DEVICES 16
struct half {{ unsigned short x; }};
static std::vector<int> g_locks_buf((size_t) MAX_TILES_C * 2 + 64);
struct DevCtx {{
    static DevCtx& instance() {{ static DevCtx d; return d; }}
    int* get_locks(int) {{ return g_locks_buf.data(); }}
}};
static thread_local int lc_gemm_locks_offset = 0;
static int g_mode = 0, g_rows32 = 0;
static bool dense_v2_gemm_on() {{ return g_mode == 1 || g_mode == 2; }}
static bool dense_v2_rows32() {{ return g_rows32 == 1; }}
struct W {{ float* slots; size_t slot_bytes; float* side_slots; size_t side_bytes; int* hctr; }};
static W g_ws[MAX_DEVICES];
{helper}
{side_fn}
{fork_fn}
static bool gemm_gr_side(int offset, void*** args_out = nullptr)
{{
    lc_gemm_locks_offset = offset;
    int device = 0;
    const half* A_ptr = nullptr; const uint16_t* B_ptr = nullptr; void* C_ptr = nullptr;
    int size_m = 4, size_k = 2560, size_n = 640;
    const half* suh_ptr = nullptr; half* A_had_ptr = nullptr; const half* svh_ptr = nullptr;
    {locks_line}
    {kargs}
    bool side = lc_side_branch(kernelArgs, device);
    lc_gemm_locks_offset = 0;
    return side;
}}
static bool gemv_entry_side()
{{
    int device = 0;
    const half* A_ptr = nullptr; const uint16_t* B_ptr = nullptr; void* C_ptr = nullptr;
    int size_m = 4, size_k = 2560, size_n = 2560;
    const half* suh_ptr = nullptr; half* A_had_ptr = nullptr; const half* svh_ptr = nullptr;
    {gv_locks}
    {gv_args}
    return lc_side_branch(kernel_args, device);
}}
static int* hctr_for(bool side, int device)
{{
    {hctr_line}
    return hctr;
}}
// ws_ready's layout arithmetic on a fake base address
static bool layout_ok(int mode, int rows32, bool fork, char* msg)
{{
    g_mode = mode; g_rows32 = rows32;
    if (fork) setenv("EXL3_LC_QSA_FORK", "1", 1); else unsetenv("EXL3_LC_QSA_FORK");
    size_t slot_bytes = dense_v2_rows32() ? DENSE_V2_WS_BYTES_ROWS32 : DENSE_V2_WS_BYTES;
    {side_bytes_line}
    size_t hctr_bytes = (size_t) DENSE_V2_GEMV_MAX_HBLOCKS * sizeof(int);
    static std::vector<char> mem;
    mem.assign(slot_bytes + side_bytes + hctr_bytes, 0);
    char* p = mem.data();
    W& w = g_ws[0];
    w.slots = (float*) p; w.slot_bytes = slot_bytes; w.side_bytes = side_bytes;
    {layout[0]}
    {layout[1]}
    struct R {{ const char* n; uintptr_t a, b; }} r[4] = {{
        {{"slots", (uintptr_t) w.slots, (uintptr_t) w.slots + slot_bytes}},
        {{"side", (uintptr_t) (w.side_slots ? w.side_slots : (float*) p), (uintptr_t) (w.side_slots ? (char*) w.side_slots + side_bytes : p)}},
        {{"hctr-lo", (uintptr_t) hctr_for(false, 0), (uintptr_t) (hctr_for(false, 0) + DENSE_V2_GEMV_MAX_HBLOCKS / 2)}},
        {{"hctr-hi", (uintptr_t) hctr_for(true, 0), (uintptr_t) (hctr_for(true, 0) + DENSE_V2_GEMV_MAX_HBLOCKS / 2)}} }};
    bool ok = true;
    for (int i = 0; i < 4; ++i)
    {{
        if (r[i].b < r[i].a || r[i].a < (uintptr_t) p || r[i].b > (uintptr_t) p + mem.size()) ok = false;
        for (int j = i + 1; j < 4; ++j)
            if (r[i].a < r[j].b && r[j].a < r[i].b && r[i].b > r[i].a && r[j].b > r[j].a) ok = false;
    }}
    bool want_side = (mode == 1 || mode == 2 || rows32) && fork;
    if ((side_bytes != 0) != want_side) ok = false;
    snprintf(msg, 200, "mode %d rows32 %d fork %d: side slots %zu B, ranges disjoint and inside the allocation", mode, rows32, (int) fork, side_bytes);
    return ok;
}}
int main()
{{
    int bad = 0;
    auto ck = [&](bool c, const char* m) {{ std::printf("%s %s\\n", c ? "PASS" : "FAIL", m); bad += !c; }};
    ck(!gemm_gr_side(0), "C++: exl3_gemm_gr at offset 0 (main branch / no latchain) -> main scratch");
    ck(gemm_gr_side(LC_LOCKS_ALT_OFFSET), "C++: exl3_gemm_gr at LC_LOCKS_ALT_OFFSET (latchain side branch) -> side scratch");
    ck(gemm_gr_side(1), "C++: any non-zero offset -> side");
    ck(!gemv_entry_side(), "C++: standalone exl3_gemv entry (base locks) -> main");
    ck(!gemm_gr_side(0), "C++: offset reset to 0 -> main again");
    char msg[256];
    for (int mode = 0; mode <= 3; ++mode) for (int r32 = 0; r32 <= 1; ++r32) for (int f = 0; f <= 1; ++f)
    {{ bool ok = layout_ok(mode, r32, f, msg); ck(ok, msg); }}
    g_mode = 3; layout_ok(3, 0, true, msg);
    ck(hctr_for(true, 0) == hctr_for(false, 0) + DENSE_V2_GEMV_MAX_HBLOCKS / 2, "C++: side hctr = main hctr + H/2");
    ck(20 <= DENSE_V2_GEMV_MAX_HBLOCKS / 2 && 5 <= DENSE_V2_GEMV_MAX_HBLOCKS / 2, "C++: 20 (largest served gemv) and 5 (index_qk) blocks fit a half");
    return bad ? 1 : 0;
}}
"""
    cpp = os.path.join(work, "lcguard_harness.cpp")
    open(cpp, "w").write(prog)
    cc = compiler()
    if not cc:
        check(False, "no host C++ compiler (c++/g++/clang++) for part A")
        return
    exe = os.path.join(work, "lcguard_harness")
    r = subprocess.run([cc, "-std=c++17", "-O1", "-Wall", "-Wno-unused-variable", "-Wno-unused-but-set-variable",
                        "-Wno-unused-function", "-o", exe, cpp], capture_output=True, text=True)
    check(r.returncode == 0, f"C++ harness compiles ({cc}) with code extracted from the tree"
          + ("" if r.returncode == 0 else ":\n" + r.stderr[-2500:]))
    if r.returncode != 0:
        return
    r = subprocess.run([exe], capture_output=True, text=True)
    print(r.stdout.rstrip())
    logs = [l for l in r.stderr.splitlines() if "densegemm lcguard" in l]
    check(r.returncode == 0, "C++ harness cases")
    check(len(logs) == 1, f"the side path is logged once per device (got {len(logs)} lines)")


def fn_body(src, fn):
    m = re.search(r"\nvoid\* " + fn + r"\n\(.*?\)\n\{\n", src, re.S)
    return extract_block(src, r"\nvoid\* " + fn + r"\n\(", fn)[len(m.group(0)):] if m else ""


def part_b(tree, untouched):
    ext = os.path.join(tree, "exllamav3_ext")
    dv2 = open(os.path.join(ext, "quant", "exl3_dense_v2.cu")).read()
    hdr = open(os.path.join(ext, "quant", "exl3_dense_v2.cuh")).read()
    for fn in ("dense_v2_launch_gemm", "dense_v2_launch_gemv"):
        body = fn_body(dv2, fn)
        first = body.split("\n")[0].strip() if body else ""
        check(first.startswith("bool side = lc_side_branch(kernel_args, device);"), f"{fn}: side detection is the first statement")
    mg = fn_body(dv2, "dense_v2_launch_mgemm")
    check("side" not in strip_comments(mg) and "g_ws[device].slots" in mg,
          "dense_v2_launch_mgemm untouched (22-arg layout; mgemm never on the side branch), keeps g_ws.slots")
    code = strip_comments(dv2) + strip_comments(hdr)
    hits = [s for s in ("lc_gemm_locks_offset", "LC_LOCKS_ALT_OFFSET", "lc_fork_get", "lc_fork_slot", "LcFork",
                        "exl3_gemm_set_locks_offset") if re.search(r"\b" + s + r"\b", code)]
    check(not hits, f"densegemm files reference no latchain symbol in code ({hits or 'none'}; the fork flag is read as an env string)")
    b = open(os.path.join(ext, "bindings.cpp")).read()
    check(b.count('m.attr("dense_v2_lc_guard") = DENSE_V2_LC_GUARD;') == 1 and "#define DENSE_V2_LC_GUARD 1" in hdr,
          "bindings expose dense_v2_lc_guard = DENSE_V2_LC_GUARD (1)")
    calls = []
    for root, _, files in os.walk(ext):
        for f in files:
            if f.endswith((".cu", ".cpp", ".cuh", ".h")):
                raw = open(os.path.join(root, f), errors="replace").read()
                for m in re.finditer(r"exl3_gemm_set_locks_offset\(([^)]*)\)", strip_comments(raw)):
                    calls.append((f, m.group(1).strip()))
    nonzero = [c for c in calls if c[1] not in ("0", "int offset")]
    if os.path.exists(os.path.join(ext, "libtorch", "lc_fork.h")):
        att = open(os.path.join(ext, "libtorch", "attention.cpp")).read()
        check(nonzero == [("attention.cpp", "LC_LOCKS_ALT_OFFSET")]
              and "struct LocksReset { ~LocksReset() { exl3_gemm_set_locks_offset(0); } } locks_reset;" in att,
              f"the only non-zero exl3_gemm_set_locks_offset is latchain's side-branch call, RAII-reset ({calls})")
    if untouched:
        old = open(os.path.join(untouched, "exllamav3_ext", "quant", "exl3_dense_v2.cu")).read()
        for fn in ("dense_v2_launch_gemm", "dense_v2_launch_gemv", "dense_v2_launch_mgemm"):
            a, bb = fn_body(old, fn).split("\n"), fn_body(dv2, fn).split("\n")
            removed = [l for l in difflib.ndiff(a, bb) if l.startswith("- ")]
            added = [l[2:] for l in difflib.ndiff(a, bb) if l.startswith("+ ")]
            if fn == "dense_v2_launch_mgemm":
                check(not removed and not added, f"{fn}: byte-identical to the unguarded densegemm")
                continue
            # every removed line is a main-branch statement re-expressed with an explicit main branch
            allowed_removed = {
                "    if (!ws_ready(device, stream) || need > g_ws[device].slot_bytes) return nullptr;",
                "    float* ws = g_ws[device].slots;",
                "    if (size_n % 128 || size_n / 128 > DENSE_V2_GEMV_MAX_HBLOCKS) return nullptr;",
                "    int* hctr = g_ws[device].hctr;",
            }
            bad = [l for l in removed if l[2:] not in allowed_removed]
            check(not bad, f"{fn}: only the ws / hctr selection lines change (removed: {len(removed)}, unexpected: {bad})")
            text = "\n".join(added)
            if fn == "dense_v2_launch_gemm":
                ok = ("if (!ws_ready(device, stream)) return nullptr;" in text
                      and "if (!side && need > g_ws[device].slot_bytes) return nullptr;" in text
                      and "float* ws = side ? g_ws[device].side_slots : g_ws[device].slots;" in text)
            else:
                ok = ("int* hctr = g_ws[device].hctr + (side ? DENSE_V2_GEMV_MAX_HBLOCKS / 2 : 0);" in text
                      and "size_n / 128 > DENSE_V2_GEMV_MAX_HBLOCKS / 2" in text)
            check(ok, f"{fn}: main branch (side == false) keeps the unguarded conditions and scratch pointer")


def part_c():
    """Resources per V2 twin: gemm / mgemm -> slots, gemv -> hctr. Guarded side-branch calls use slots-side /
    hctr-hi. Two calls race when they run concurrently and share an array."""
    def res(path, rows, mode, side, guard):
        if path == "mgemm":
            r = "slots" if mode in (1, 2) else None
        elif path == "fp16":
            r = None
        elif rows <= 8:
            r = "hctr" if mode in (1, 3) else None
        else:
            r = "slots" if mode in (1, 2) else None
        if r and side and guard:
            r += "-side"
        return r
    windows = {
        # (main-branch calls, side-branch calls): the side calls of the QF window carry the latchain offset; the
        # moefast shared-expert stream does not (offset 0) but its window holds no dense main call
        "QF window, served (fused qkv mgemm)": ([("qkv", "mgemm")], [("index_qk", "gemm")], True),
        "QF window, unfused q/k/v gemm_gr": ([("q", "gemm"), ("k", "gemm"), ("v", "gemm")], [("index_qk", "gemm")], True),
        "moefast shared window (EARLY or late)": ([("router", "fp16"), ("routed A", "coop")],
                                                  [("shared gate_up", "mgemm"), ("shared.down", "gemm")], False),
    }
    raced_somewhere = False
    for wname, (mains, sides, lc_side) in windows.items():
        for mode in (1, 2, 3):
            for rows in (4, 16):
                for guard in (True, False):
                    m_res = {res(p, rows, mode, False, guard) for _, p in mains if p != "coop"} - {None}
                    s_res = {res(p, rows, mode, lc_side, guard) for _, p in sides} - {None}
                    shared = m_res & s_res
                    if guard:
                        check(not shared, f"model {wname}, mode {mode}, {rows} rows, guarded: no shared array "
                              f"(main {sorted(m_res) or '-'}, side {sorted(s_res) or '-'})")
                    elif shared:
                        raced_somewhere = True
                        print(f"INFO model {wname}, mode {mode}, {rows} rows, UNGUARDED: shares {sorted(shared)}")
    check(raced_somewhere, "model: without the guard some configuration races (the model is not vacuous)")


def build_trees(a, work):
    stack = os.path.join(work, "stack")
    shutil.copytree(a.src, stack, ignore=shutil.ignore_patterns("__pycache__"))
    for p in ("hcfast-r1/hcfast-r1.patch", "moefast-r1/moefast-r1.patch"):
        subprocess.run(["patch", "-s", "-p1", "--fuzz=0", "--batch", "--forward", "--no-backup-if-mismatch", "-d", stack,
                        "-i", os.path.join(a.patches, p)], check=True)
    trees = []
    for inc in ("dg1", "dg2"):
        t = os.path.join(work, inc)
        shutil.copytree(stack, t)
        subprocess.run(["bash", os.path.join(a.series, "apply-series.sh"), t], check=True, env=dict(os.environ, INCLUDE=inc),
                       stdout=subprocess.DEVNULL)
        u = os.path.join(work, inc + "-noguard")
        shutil.copytree(t, u)
        subprocess.run(["patch", "-s", "-R", "-p1", "--fuzz=0", "--batch", "--no-backup-if-mismatch", "-d", u, "-i",
                        os.path.join(a.series, "06-dense-lcguard.patch")], check=True)
        trees.append((inc, t, u))
    return trees


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree")
    ap.add_argument("--untouched")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--src", help="the installed exllamav3 package of tabbyapi:slotfix-r1 (required with --build)")
    ap.add_argument("--patches", default=os.path.join(here, "..", ".."), help="docker/overlays (holds hcfast-r1/ and moefast-r1/)")
    ap.add_argument("--series", default=os.path.join(here, "..", "series"))
    a = ap.parse_args()
    if a.build and not a.src:
        ap.error("--build needs --src")
    work = tempfile.mkdtemp(prefix="lcguard.", dir=os.environ.get("TMPDIR") or None)
    try:
        trees = [("tree", a.tree, a.untouched)] if a.tree else build_trees(a, work) if a.build else None
        if not trees:
            ap.error("--tree or --build")
        for name, t, u in trees:
            print(f"== {name}: {t}")
            wd = os.path.join(work, "cc-" + name)
            os.makedirs(wd, exist_ok=True)
            part_a(t, wd)
            part_b(t, u)
        print("== resource model")
        part_c()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("LCGUARD CPU " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)})"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
