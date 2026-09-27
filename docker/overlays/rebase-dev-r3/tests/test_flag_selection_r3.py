"""rebase-dev r3 tests (CPU only, extension and triton stubbed): upstream dev 73a6229..5783a93 (v1.5.2) on the r2 port.

What r3 adds over r2 is upstream's per-device shared-memory budget (Turing sm_75, PR #325 + rework f4db698) and the
loader's shared file stream (5783a93). These tests pin the properties impl-status.md argues for sm_120:

  * every compiled EXL3 GEMM shape fits the request cap, so the new shape filter is a no-op at 90 KiB (sm_120 opt-in is
    99 KiB, request = min(opt-in, 90 KiB) = 90 KiB = SMEM_MAX), and the served densegemm V2 twins still ask for SMEM_MAX
  * the one merge resolution (exl3_gemm.cu) keeps the lcguard locks offset next to upstream's smem_max
  * smem.pick_config takes the stock (first) candidate when it fits, probes once per key, and the served launch sites
    do not force a tile the new prefill rule would now honour differently
  * the BC-graph compile check only declines kernels above the device limit, and every served compile site is intact
"""
import ast
import glob
import inspect
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import _runner  # noqa: E402  (stubs exllamav3.ext and triton)

_runner.clear_env_flags()
import exllamav3  # noqa: E402

PKG = os.path.dirname(os.path.abspath(exllamav3.__file__))
EXT = os.path.join(PKG, "exllamav3_ext")
RESULTS = []


def src(rel):
    return open(os.path.join(PKG, rel)).read()


def check(name, fn):
    try:
        out = fn()
        RESULTS.append((name, True))
        print(f"PASS {name}" + (f"\n     {out}" if out else ""))
    except Exception as ex:  # noqa: BLE001
        import traceback
        RESULTS.append((name, False))
        print(f"FAIL {name}")
        traceback.print_exc()


def t_version():
    from exllamav3.version import __version__ as v
    assert v == "1.5.2", v
    return "exllamav3 1.5.2 (dev 5783a93)"


def gemm_smem(tm, tk, tn, ss, fs, bits, half, had):
    # Python port of exl3_kernel_map.cuh exl3_gemm_smem_bytes() (EXL3_GEMM_BASE_THREADS = 256)
    tbm, tbk, tbn = tm // 16, tk // 16, tn // 16
    fnw = 2 * tbn // (256 // 32)
    tu = 16 * bits + (8 if half else 0)
    a, b, c = tm * tk, tbk * tbn * tu, 4 * 256 * fnw * tbm
    c = max(c, tn * tm if had else 0)
    return ss * (2 * a + 2 * b) + 4 * c


def t_gemm_shapes_fit():
    km = open(os.path.join(EXT, "quant", "exl3_kernel_map.cuh")).read()
    shapes = {int(n): tuple(int(x) for x in v.split(","))
              for n, v in re.findall(r"#define EXL3_GEMM_SHAPE_(\d+)\s+([\d,\s]+)\n", km)}
    assert sorted(shapes) == [1, 2, 3, 4], shapes
    assert "EXL3_GEMM_BASE_THREADS 256" in km
    arch = open(os.path.join(EXT, "arch.cuh")).read()
    assert "#define EXL3_SMEM_MAX_DEFAULT (90 * 1024)" in arch
    inner = open(os.path.join(EXT, "quant", "exl3_gemm_inner.cuh")).read()
    assert "#define SMEM_MAX (90 * 1024)" in inner
    dev = open(os.path.join(EXT, "quant", "exl3_devctx.cu")).read()
    assert "return MIN(get_smem_max(device), EXL3_SMEM_MAX_DEFAULT);" in dev
    assert "cudaDevAttrMaxSharedMemoryPerBlockOptin" in dev
    worst = max(gemm_smem(*shapes[s], k, h, had) for s in shapes for k in range(1, 9) for h in (0, 1) for had in (0, 1))
    assert worst <= 90 * 1024, worst
    # sm_120 opt-in 101376 B -> request = min(101376, 92160) = 92160 = SMEM_MAX: the filter keeps every shape
    return f"largest GEMM footprint {worst} B (shape 4, K 8.5) <= 92160 B request cap: shape_compat on sm_120 = divisibility only"


def t_gemm_resolution():
    g = open(os.path.join(EXT, "quant", "exl3_gemm.cu")).read()
    assert g.count("int* locks = DevCtx::instance().get_locks(device) + lc_gemm_locks_offset;") == 1
    assert g.count("int smem_max = DevCtx::instance().get_smem_request(device);") == 2
    assert not re.search(r"\bSMEM_MAX\s*[,)]", g), "a launch in exl3_gemm.cu still passes SMEM_MAX"
    assert g.count("!half_k && ") == 2, "densegemm V2 twin gates changed"
    assert g.count("exl3_gemm_shape_compat(candidate_shape_idx, size_m, size_k, size_n, K, half_k)") == 2
    moe = open(os.path.join(EXT, "quant", "exl3_moe.cu")).read()
    assert "int smem_max = DevCtx::instance().get_smem_request(device);" in moe
    d2 = open(os.path.join(EXT, "quant", "exl3_dense_v2.cu")).read()
    assert d2.count("SMEM_MAX") >= 3, "densegemm V2 launches no longer request SMEM_MAX"
    return "lcguard offset + upstream smem_max both present; V2 twins still request SMEM_MAX (= the request cap on sm_120)"


class _FakeCuda:
    """Just enough of torch.cuda for smem.py on CPU (an sm_120 with 99 KiB opt-in)."""
    def __init__(self, optin):
        self.optin = optin

    def get_device_properties(self, idx):
        import types
        return types.SimpleNamespace(shared_memory_per_block_optin = self.optin, major = 12)

    def get_device_capability(self, idx):
        return (12, 0)

    def current_device(self):
        return 0


def t_pick_config():
    import torch
    import importlib
    import exllamav3.modules.attention_fn.smem as smem
    real = torch.cuda
    os.environ.pop("EXL3_TRITON_SMEM_LIMIT", None)
    smem = importlib.reload(smem)
    try:
        torch.cuda = _FakeCuda(101376)
        assert smem.smem_limit(0) == 101376
        cands = smem.tile_ladder(64, 32, 4, 2)
        assert cands[0] == (64, 32, 4, 2)
        probes = []

        def probe(c):
            probes.append(c)
            return 60000 if c == cands[0] else 20000
        assert smem.pick_config(0, "k", ("key",), cands, probe) == cands[0]
        assert smem.pick_config(0, "k", ("key",), cands, probe) == cands[0]
        assert probes == [cands[0]], probes   # one probe, then cached
        # a stock footprint over the limit walks down (only happens below sm_80-class budgets)
        probes.clear()
        big = lambda c: 120000 if c == cands[0] else 20000  # noqa: E731
        assert smem.pick_config(0, "k2", ("key",), cands, big) == cands[1]
        assert smem.halving_ladder(128)[0] == 128
    finally:
        torch.cuda = real
    return "stock candidate first, one probe per key, cached; a candidate is dropped only above the device opt-in"


def t_torch_optin_attr():
    """smem.py reads props.shared_memory_per_block_optin and falls back to 96 KiB (below the 99 KiB sm_120 opt-in) when
    the attribute is missing. torch binds it for CUDA builds (torch/csrc/cuda/Module.cpp, v2.14.0, `#ifndef USE_ROCM`);
    the .pyi omits it and CPU-only torch has no _CudaDeviceProperties, so this is only decidable on a CUDA torch
    (the image; landing_r3.py asserts it there too)."""
    import torch
    cls = getattr(torch._C, "_CudaDeviceProperties", None)
    if cls is None or not torch.version.cuda:
        return f"n/a here (CPU-only torch {torch.__version__}); asserted in the image by landing_r3.py"
    assert hasattr(cls, "shared_memory_per_block_optin"), "CUDA torch without shared_memory_per_block_optin"
    return f"torch {torch.__version__} (CUDA {torch.version.cuda}) binds shared_memory_per_block_optin"


def t_bc_compile_check():
    s = src("modules/attention_fn/bc_attn.py")
    assert "if ck.metadata.shared > limit:" in s and "raise BCKernelTooLarge(" in s
    assert "limit = smem_limit(device)" in s
    n = len(re.findall(r"_compile_kernel\(dev, ", s))
    assert n == 12, n   # decode split/combine/update, QSA stage/raw-append/pool-update/fewq/expand/sparse-split/sparse-combine, ring append/pool-update
    # served constexprs and stage knobs unchanged
    assert "_LC_QSA_SPLIT_STAGES" in s and "_LC_QSA_COMBINE_STAGES or 1" in s and "BLOCK_ROWS" in s and "ROWS_SUB" in s \
        and "D_SUB" in s and "V_DIM" in s
    assert "except BCKernelTooLarge:" in s
    tk = open(os.path.join(EXT, "triton_kernel.cpp")).read()
    assert "func_set_attribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes)" in tk
    return f"{n} served BC compile sites; decline only when shared > opt-in, which the driver already refuses at load"


def t_no_forced_tiles():
    """The prefill rule changed from `if qc is not None` (forced block_n overridden) to `... and block_n is None`.
    No caller in the package forces a tile on the Triton attention entry points, so the stock config is unchanged."""
    tp = src("modules/attention_fn/triton_paged.py")
    assert "if qc is not None and block_n is None:" in tp
    entry = {"paged_attn_triton_prefill", "paged_attn_triton_decode", "varlen_attn_triton", "dsa_indexer_scores", "dsa_attn"}
    forced = []
    for path in glob.glob(os.path.join(PKG, "**", "*.py"), recursive = True):
        if "/conversion/" in path:
            continue
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
                if name in entry:
                    kws = {k.arg for k in node.keywords} & {"block_m", "block_n", "num_stages", "num_warps", "block_h"}
                    if kws:
                        forced.append((os.path.relpath(path, PKG), node.lineno, name, sorted(kws)))
    assert not forced, forced
    return "no in-tree caller forces block_m/block_n/num_stages/num_warps on the Triton attention entry points"


def t_hc_tiled_gate():
    s = src("modules/hyperconnections.py")
    assert "and torch.cuda.get_device_capability(dev)[0] >= 8" in s
    import exllamav3.modules.hyperconnections as hc
    assert hc._gr_mix_tiled_enable is True
    return "tiled HC prefill: cc major >= 8 added (sm_120 = 12), default still on"


def t_stloader():
    s = open(os.path.join(EXT, "stloader.cpp")).read()
    assert "ssize_t br = pread(fileno(file)" in s
    assert "const int streams = 1;" in s and "if (i > 0 && handles[i] == handles[0]) continue;" in s
    # every read goes through read_range (positional on Linux), so the shared FILE* has no seek state to race on
    assert s.count("read_range(file,") == 3, s.count("read_range(file,")
    assert "fseek(" not in s.replace("_fseeki64", "")
    return "Linux: one FILE* per shard shared by the 8 workers, all reads pread (host memory only)"


def t_new_knobs_not_served():
    for k in ("EXL3_TRITON_SMEM_LIMIT", "EXL3_TRITON_SMEM_DEBUG"):
        assert k not in os.environ
    return "EXL3_TRITON_SMEM_LIMIT / _DEBUG are unset (debug-only; a limit would shrink served tiles)"


check("version = upstream 1.5.2 (dev 5783a93)", t_version)
check("every compiled EXL3 GEMM shape fits the 90 KiB request cap (shape filter is a no-op on sm_120)", t_gemm_shapes_fit)
check("exl3_gemm.cu merge resolution: lcguard locks offset + upstream smem_max", t_gemm_resolution)
check("smem.pick_config: stock first, one probe per key", t_pick_config)
check("torch exposes shared_memory_per_block_optin", t_torch_optin_attr)
check("BC-graph compile check: served sites intact, decline only above the opt-in", t_bc_compile_check)
check("no in-tree caller forces Triton attention tiles", t_no_forced_tiles)
check("hyperconnections tiled gate", t_hc_tiled_gate)
check("stloader shared stream is pread-only", t_stloader)
check("new upstream env knobs unset", t_new_knobs_not_served)

passed = sum(1 for _, ok in RESULTS if ok)
print(f"\n== {passed}/{len(RESULTS)} r3 tests passed ==")
sys.exit(0 if passed == len(RESULTS) else 1)
