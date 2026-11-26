"""rebase167 NATIVE static checks (stdlib only: runs on any host, and in the image with no GPU).

Package location: $REBASE167_PKG (the exllamav3/ package directory), else the installed package found by
importlib.util.find_spec without importing it.

* build flags: util/cuda_flags.cuda_cflags() with EXLLAMA_EXT_LINEINFO=1 EXLLAMA_EXT_COMPRESS=0 equals the served
  nvcc flag list exactly (so unchanged kernels keep their served SASS); without the knobs it differs only by
  -lineinfo / --compress-mode (documented flag difference of upstream 9b498413);
* sources: extension_sources(hip=False) holds every served-only translation unit, no rocm/ unit and no
  same-stem .cpp/.cu pair (ninja object collision);
* pybind: every name the served extension registered (served_pybind_names.txt: m.def / m.attr / py::class_
  from served bindings.cpp + libtorch/*_bc.h) is still registered by the integrated sources;
* conflict markers: none in the extension sources or modules/.
"""
import importlib.util
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVED_NVCC = ["-lineinfo", "-O3", "--use_fast_math", "-Xcudafe", "--diag_suppress=177", "-Xcudafe", "--diag_suppress=20012"]

SERVED_ONLY_TUS = [
    "hc_mix_v3.cu",
    "quant/exl3_dense_v2.cu",
    "quant/comp_units/exl3_dense_v2_inst_gemm.cu",
    "quant/comp_units/exl3_dense_v2_inst_gemv.cu",
    "quant/comp_units/exl3_dense_v2_inst_mgemm.cu",
    "quant/comp_units/exl3_dense_v2_inst_k6.cu",
    "quant/comp_units/exl3_moe_coop_inst_k2.cu",
    "quant/comp_units/exl3_moe_coop_inst_k3.cu",
    "quant/comp_units/exl3_moe_coop_inst_k4.cu",
    "quant/comp_units/exl3_moe_coop_inst_h2_tuned.cu",
    "quant/comp_units/exl3_moe_coop_inst_h2_uncap.cu",
    "quant/comp_units/exl3_moe_prefill_inst_h2_static.cu",
    "quant/exl3_moe_prefill_e3.cpp",
    "quant/exl3_moe_prefill_e3__cuda.cu",
    "quant/coop_autotune.cu",
    "quant/exl3_moe_coop.cu",
]


def _pkg() -> Path:
    p = os.environ.get("REBASE167_PKG")
    if p:
        return Path(p)
    spec = importlib.util.find_spec("exllamav3")
    assert spec and spec.submodule_search_locations, "set REBASE167_PKG to the exllamav3 package directory"
    return Path(list(spec.submodule_search_locations)[0])


def _cuda_flags():
    spec = importlib.util.spec_from_file_location("cuda_flags_t", _pkg() / "util" / "cuda_flags.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_nvcc_flags_match_served(monkeypatch):
    cf = _cuda_flags()
    monkeypatch.setenv("EXLLAMA_EXT_LINEINFO", "1")
    monkeypatch.setenv("EXLLAMA_EXT_COMPRESS", "0")
    assert cf.cuda_cflags(debug = False, hip = False) == SERVED_NVCC


def test_nvcc_flags_upstream_default_differs_only_by_lineinfo_and_compress(monkeypatch):
    cf = _cuda_flags()
    monkeypatch.delenv("EXLLAMA_EXT_LINEINFO", raising = False)
    monkeypatch.setenv("EXLLAMA_EXT_COMPRESS", "0")
    assert cf.cuda_cflags(debug = False, hip = False) == SERVED_NVCC[1:]


def test_sources_cover_served_units_and_no_collisions():
    cf = _cuda_flags()
    base = _pkg() / "exllamav3_ext"
    srcs = [Path(s) for s in cf.extension_sources(str(base), hip = False)]
    rel = {s.relative_to(base).as_posix() for s in srcs}
    missing = [t for t in SERVED_ONLY_TUS if t not in rel]
    assert not missing, missing
    assert not [r for r in rel if r.startswith("rocm/")]
    stems = {}
    for s in srcs:
        k = s.with_suffix("")
        assert k not in stems, f"same-stem collision {stems[k]} / {s}"
        stems[k] = s


def _registered_names():
    base = _pkg() / "exllamav3_ext"
    text = (base / "bindings.cpp").read_text()
    for h in sorted((base / "libtorch").glob("*_bc.h")):
        text += h.read_text()
    names = set(re.findall(r'm\.(?:def|attr)\(\s*"([A-Za-z0-9_]+)"', text))
    names |= set(re.findall(r'py::class_<[^>]*>\(\s*m\s*,\s*"([A-Za-z0-9_]+)"', text))
    return names


def _names(fn):
    return {l.strip() for l in (HERE / fn).read_text().splitlines() if l.strip() and not l.startswith("#")}


def test_served_pybind_names_still_registered():
    served = _names("served_pybind_names.txt")
    dropped = _names("pybind_dropped_ok.txt")
    now = _registered_names()
    missing = sorted(served - now - dropped)
    assert not missing, f"served pybind names not registered any more: {missing}"


def test_no_conflict_markers():
    pkg = _pkg()
    bad = []
    for sub in ("exllamav3_ext", "modules"):
        for p in (pkg / sub).rglob("*"):
            if p.suffix in (".py", ".cu", ".cuh", ".cpp", ".h") and p.is_file():
                for i, line in enumerate(p.read_text(errors = "replace").splitlines(), 1):
                    if line.startswith(("<<<<<<< ", ">>>>>>> ", "||||||| ")):
                        bad.append(f"{p}:{i}")
    assert not bad, bad[:20]


def test_gdn_bf16_per_token_rounding_wired():
    """Served bf16 recurrent-state semantics (round the carried state to bf16 after every token, EXL3_GDN_BF16_ROUND
    unset / token) in both scan bodies, which the forward scan and the rewind replay share; selected for forward
    and replay launches alike, with EXL3_GDN_BF16_ROUND=pass the explicit once-per-pass arm"""
    src = (_pkg() / "exllamav3_ext" / "gdn.cu").read_text()
    assert "st[e] = state;" not in src
    assert src.count("st[e] = state_carry<TS>(state);") == 2
    assert "state_carry<bf16_tok>(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }" in src
    assert 'std::getenv("EXL3_GDN_BF16_ROUND")' in src
    assert "LAUNCH_RULE_T(bf16_tok, K, __VA_ARGS__)" in src
    assert "scan_replay_kernel<__VA_ARGS__, bf16_tok>" in src
    assert "scan_replay_kernel_128<__VA_ARGS__, bf16_tok>" in src
    assert src.count("const bool bf16_token = state_bf16 && gdn_bf16_token_round();") == 2
