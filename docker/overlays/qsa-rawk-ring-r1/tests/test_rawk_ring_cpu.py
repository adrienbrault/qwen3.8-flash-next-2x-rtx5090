"""CPU-only checks for qsa-rawk-ring r1 (pytest). CPU torch, no Triton, no CUDA, no network.

The served source defaults to the local extraction of tabbyapi:nvme-tier-r4-e3det (R535); override with
RAWK_RING_SERVED_SRC. Tests that need it skip when it is absent. The Triton kernels (served and patched) run
through tests/fake_triton.py, a sequential CPU interpreter of the tl subset they use.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap

import pytest
import torch


HERE = Path(__file__).resolve().parent
R1 = HERE.parent
SERVED = Path(os.environ.get("RAWK_RING_SERVED_SRC", "served-src-r535/exllamav3"))
PATCH = R1 / "served-source.patch"
MANIFEST = json.loads((R1 / "overlay/manifest.json").read_text())
FILES = sorted(MANIFEST["files"])
# Extra served files the runtime tests need next to the patched ones (unchanged by the patch)
SUPPORT = ["constants.py", "cache/cache.py", "cache/fp16.py", "cache/quant.py",
           "modules/attention_fn/mla_triton.py", "generator/disk_cache.py"]

sys.path.insert(0, str(HERE))
import fake_triton as ft      # noqa: E402
import rawk_scenarios as sc   # noqa: E402

needs_served = pytest.mark.skipif(not SERVED.is_dir(), reason = f"served source not found at {SERVED}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_files(tmp_path: Path, files) -> Path:
    target = tmp_path / "exllamav3"
    for relative in files:
        (target / relative).parent.mkdir(parents = True, exist_ok = True)
        shutil.copy2(SERVED / relative, target / relative)
    return target


def apply(root: Path, patch: Path, *extra) -> subprocess.CompletedProcess:
    with patch.open("rb") as stream:
        return subprocess.run(["patch", "-p1", "-F0", "--no-backup-if-mismatch", *extra], cwd = root,
                              stdin = stream, capture_output = True)


@pytest.fixture(scope = "module")
def patched(tmp_path_factory) -> Path:
    root = copy_files(tmp_path_factory.mktemp("patched"), FILES + SUPPORT)
    proc = apply(root, PATCH)
    assert proc.returncode == 0, proc.stdout.decode() + proc.stderr.decode()
    return root


@pytest.fixture(scope = "module")
def kernels(patched: Path):
    return ft.kernel_set(patched / "modules/attention_fn/qsa_triton.py",
                         patched / "modules/attention_fn/mla_triton.py")


def functions(path: Path) -> dict:
    src = path.read_text()
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.setdefault(node.name, ast.get_source_segment(src, node))
    return out


def params(src: str, name: str) -> list[str]:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args]
    raise KeyError(name)


def ring_rows(root: Path) -> int:
    fns = functions(root / "cache/qsa.py")
    ns = {"_BC_MAX_QLEN": 16}
    exec(fns["rawk_ring_rows"], ns)
    return ns["rawk_ring_rows"](4)


# ---- patch + manifest -------------------------------------------------------------------------------------------

def test_patch_hash_pinned():
    assert sha256(PATCH) == MANIFEST["patch_sha256"]
    assert MANIFEST["base_image"] == "tabbyapi:nvme-tier-r4-e3det"
    assert MANIFEST["selectors"] == {"EXL3_QSA_RAWK_RING": "cache/qsa.py"}


def test_patch_touches_exactly_manifest_files():
    text = PATCH.read_text()
    assert sorted(re.findall(r"^\+\+\+ b/(\S+)", text, flags = re.M)) == FILES
    assert sorted(re.findall(r"^--- a/(\S+)", text, flags = re.M)) == FILES
    # Python only: no extension rebuild, no generator/tier/C++ change
    assert not any(p.endswith((".cu", ".cpp", ".h", ".cuh")) or p.startswith("generator/") for p in FILES)


@needs_served
def test_served_baseline_hashes():
    for relative, record in MANIFEST["files"].items():
        assert sha256(SERVED / relative) == record["baseline_sha256"], relative


@needs_served
def test_patch_applies_fuzz0_exit_code_and_overlay_hashes(tmp_path: Path):
    root = copy_files(tmp_path, FILES)
    proc = apply(root, PATCH)
    assert proc.returncode == 0, proc.stdout.decode() + proc.stderr.decode()
    assert not list(root.rglob("*.rej")) and not list(root.rglob("*.orig"))
    for relative, record in MANIFEST["files"].items():
        assert sha256(root / relative) == record["overlay_sha256"], relative
    again = apply(root, PATCH, "--forward", "--dry-run")
    assert again.returncode != 0


@needs_served
def test_install_py_on_scratch_root(tmp_path: Path):
    root = copy_files(tmp_path, FILES)
    cmd = [sys.executable, str(R1 / "overlay/install.py"), "--root", str(root)]
    proc = subprocess.run(cmd, capture_output = True, text = True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for relative, record in MANIFEST["files"].items():
        assert sha256(root / relative) == record["overlay_sha256"]
    again = subprocess.run(cmd, capture_output = True, text = True)
    assert again.returncode != 0 and "baseline hash mismatch" in again.stdout + again.stderr


@needs_served
def test_install_py_rejects_foreign_baseline(tmp_path: Path):
    root = copy_files(tmp_path, FILES)
    target = root / "modules/qsa_indexer.py"
    target.write_text(target.read_text() + "\n# drift\n")
    proc = subprocess.run([sys.executable, str(R1 / "overlay/install.py"), "--root", str(root)],
                          capture_output = True, text = True)
    assert proc.returncode != 0 and "baseline hash mismatch" in proc.stdout + proc.stderr


@needs_served
def test_stacks_with_other_rounds():
    """decode-kernels r6, gdn-state-bf16 r1 and adaptive-draft r1 touch disjoint files."""
    base = R1.parent.parent
    for other in ("decode-kernels/r6", "gdn-state-bf16/r1", "adaptive-draft/r1"):
        p = base / other / "served-source.patch"
        if not p.is_file():
            continue
        touched = {t.removeprefix("exllamav3/") for t in re.findall(r"^\+\+\+ b/(\S+)", p.read_text(), flags = re.M)}
        assert not touched & set(FILES), (other, touched & set(FILES))


# ---- flag-off identity (source level) -----------------------------------------------------------------------------

@needs_served
def test_served_functions_unchanged(patched: Path):
    """Every function or class the served files define is byte-identical in the patched files, except the three
    methods/classes that gain a ring branch."""
    changed = {"QSAPlanes", "_init_planes", "copy_page", "update_planes", "update_planes_ref",
               "select_indices_paged_ref", "BCAttn", "_configure_qsa", "QSAIndexer"}
    for relative in FILES:
        served = functions(SERVED / relative)
        new = functions(patched / relative)
        for name, src in served.items():
            if name in changed:
                continue
            assert new.get(name) == src, f"{relative}: {name} changed"


@needs_served
def test_update_planes_flag_off_path_is_served(patched: Path):
    served = functions(SERVED / "modules/qsa_indexer.py")["update_planes"]
    new = functions(patched / "modules/qsa_indexer.py")["update_planes"]
    start = new.index("            if getattr(layer, \"raw_ring_rows\", 0):\n")
    end = new.index("                return q\n", start) + len("                return q\n")
    assert new[:start] + new[end:] == served


@needs_served
def test_configure_qsa_flag_off_path_is_served(patched: Path):
    served = functions(SERVED / "modules/attention_fn/bc_attn.py")["_configure_qsa"]
    new = functions(patched / "modules/attention_fn/bc_attn.py")["_configure_qsa"]
    start = new.index("        ring = getattr(self.qsa_layer, \"raw_ring_rows\", 0)\n")
    else_ = new.index("        else:\n", start)
    body_end = new.index("\n\n        wts = scores", else_)
    body = new[else_ + len("        else:\n"):body_end]
    dedented = "\n".join(line[4:] if line.startswith("    ") else line for line in body.split("\n"))
    assert new[:start] + dedented + new[body_end:] == served


def test_ring_kernels_keep_served_runtime_signatures(patched: Path):
    """The C++ launch passes fixed argument vectors and patches graph parameters by index (raw append 2/3/4,
    pool 4/5/6): the ring kernels must take the same runtime parameters in the same order."""
    q = (patched / "modules/attention_fn/qsa_triton.py").read_text()
    m = (patched / "modules/attention_fn/mla_triton.py").read_text()
    assert params(q, "_qsa_raw_ring_append_kernel")[:6] == params(m, "_mla_plane_update_kernel")[:6]
    assert params(q, "_qsa_raw_ring_append_kernel")[6:] == ["page_size", "D", "RING"]
    assert params(q, "_qsa_pool_update_ring_bc_kernel") == params(q, "_qsa_pool_update_kernel") + ["RING"]
    eager = params(q, "_qsa_pool_update_ring_kernel")
    assert eager == params(q, "_qsa_pool_update_kernel")[:1] + ["rows_new"] + \
        params(q, "_qsa_pool_update_kernel")[1:] + ["RING"]


def test_aot_signature_dicts_match_kernel_parameters(patched: Path):
    src = (patched / "modules/attention_fn/bc_attn.py").read_text()
    fn = functions(patched / "modules/attention_fn/bc_attn.py")["_qsa_ring_graph_kernels"]
    q = (patched / "modules/attention_fn/qsa_triton.py").read_text()
    calls = [n for n in ast.walk(ast.parse(fn)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_compile_kernel"]
    assert len(calls) == 2
    for call, kname in zip(calls, ["_qsa_raw_ring_append_kernel", "_qsa_pool_update_ring_bc_kernel"]):
        assert call.args[1].id == kname
        sig = call.args[2]            # {runtime} | {constexpr}
        runtime = [k.value for k in sig.left.keys]
        constexprs = [e.value for e in sig.right.generators[0].iter.elts]
        assert runtime + constexprs == params(q, kname)
        consts = [kw.arg for kw in call.args[3].keywords]
        assert sorted(consts) == sorted(constexprs)
    assert "q_len + cr - 1 <= ring" in fn
    assert src.count("_qsa_ring_graph_kernels(") == 2   # definition + the one call in _configure_qsa


def test_ring_pool_math_is_the_served_math_line_for_line(patched: Path):
    """Bit-exactness on the GPU rests on the ring pool kernels performing the served kernel's operations in the same
    order: everything from the fp16 mean to the stores is textually identical, and the accumulation loop differs
    only in where a row pointer comes from."""
    fns = functions(patched / "modules/attention_fn/qsa_triton.py")
    served = fns["_qsa_pool_update_kernel"]
    tail = served[served.index("    # fp32 mean -> fp16"):]
    head = served[served.index("    b = tl.program_id(0)"):served.index("    for j in range(P):")]
    for name in ("_qsa_pool_update_ring_bc_kernel", "_qsa_pool_update_ring_kernel"):
        src = fns[name]
        assert src.endswith(tail), name
        assert head in src, name
    loop = lambda s: s[s.index("    for j in range(P):"):s.index("    # fp32 mean -> fp16")]
    served_loop = loop(served).replace(
        "            row = raw_plane + (phys * page_size + tok % page_size) * D\n",
        "            row = raw_plane + (phys * RING + tok % RING) * D\n")
    assert loop(fns["_qsa_pool_update_ring_bc_kernel"]) == served_loop
    eager = loop(fns["_qsa_pool_update_ring_kernel"])
    assert "if tok >= pos0:" in eager and "rows_new + (b * append_len + (tok - pos0)) * D" in eager


def test_eager_launch_order_pool_before_append(patched: Path):
    fn = functions(patched / "modules/attention_fn/qsa_triton.py")["qsa_ring_plane_update"]
    assert fn.index("_qsa_pool_update_ring_kernel[") < fn.index("_qsa_raw_ring_append_kernel[")


def test_verify_window_guard_present(patched: Path):
    fn = functions(patched / "modules/qsa_indexer.py")["update_planes"]
    assert 'params.get("recurrent_history") and seqlen > ring - cr + 1' in fn


# ---- allocation, copy_page and storage through the real cache classes (stubbed extension) ---------------------

_ALLOC_SCRIPT = textwrap.dedent('''
    import importlib, json, sys, types
    from types import SimpleNamespace
    import torch
    root = sys.argv[1]
    pkg = types.ModuleType("exllamav3"); pkg.__path__ = [root]; sys.modules["exllamav3"] = pkg
    ext = types.ModuleType("exllamav3.ext"); ext.exllamav3_ext = SimpleNamespace(); sys.modules["exllamav3.ext"] = ext
    model = types.ModuleType("exllamav3.model"); model.Config = object; sys.modules["exllamav3.model"] = model
    cpkg = types.ModuleType("exllamav3.cache"); cpkg.__path__ = [root + "/cache"]; sys.modules["exllamav3.cache"] = cpkg
    qsa = importlib.import_module("exllamav3.cache.qsa")
    att = SimpleNamespace(num_kv_heads = 2, head_dim = 256,
                          qsa_indexer = SimpleNamespace(head_dim = 128, compress_ratio = 4))
    out = {}
    for cls, kw in ((qsa.CacheLayer_qsa_quant, dict(k_bits = 8, v_bits = 8)), (qsa.CacheLayer_qsa, {})):
        a = cls(None, att, 0, 4 * 256, **kw); a.alloc(torch.device("cpu"))
        b = cls(None, att, 0, 4 * 256, **kw); b.alloc(torch.device("cpu"))
        for t in a.get_tensors():
            t.copy_(torch.randint(-100, 100, t.shape).to(t.dtype))
        rec = {
            "ring": getattr(a, "raw_ring_rows", None),
            "shapes": [list(t.shape) for t in a.get_tensors()],
            "dtypes": [str(t.dtype) for t in a.get_tensors()],
            "storage": int(a.storage_size()),
            "page_bytes": [t[0].numel() * t.element_size() for t in a.get_tensors()],
        }
        b.copy_page(a, 1, 2, 132)
        rec["copy_aligned"] = []
        for ta, tb in zip(a.get_tensors(), b.get_tensors()):
            n = {256: 132, 64: 33}.get(tb.shape[1])
            rec["copy_aligned"].append(None if n is None else bool(torch.equal(tb[2, :n], ta[1, :n])))
        rec["raw_untouched_after_copy"] = bool(torch.count_nonzero(b.raw_k) == 0)
        before = [t.clone() for t in b.get_tensors()]
        try:
            b.copy_page(a, 1, 3, 130)
            rec["copy_misaligned"] = "ok"
        except RuntimeError as e:
            rec["copy_misaligned"] = "raised"
        rec["misaligned_left_dest_untouched"] = all(bool(torch.equal(x, y)) for x, y in zip(before, b.get_tensors()))
        out[cls.__name__] = rec
    print(json.dumps(out))
''')


def run_alloc(root: Path, flag: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "EXL3_QSA_RAWK_RING"}
    if flag is not None:
        env["EXL3_QSA_RAWK_RING"] = flag
    proc = subprocess.run([sys.executable, "-c", _ALLOC_SCRIPT, str(root)], capture_output = True, text = True, env = env)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope = "module")
def served_tree(tmp_path_factory) -> Path:
    return copy_files(tmp_path_factory.mktemp("served"), FILES + SUPPORT)


@needs_served
def test_flag_off_allocation_identical_to_served(patched: Path, served_tree: Path):
    served = run_alloc(served_tree, None)
    for flag in (None, "0", ""):
        off = run_alloc(patched, flag)
        for cls in ("CacheLayer_qsa_quant", "CacheLayer_qsa"):
            assert off[cls]["ring"] == 0
            for key in ("shapes", "dtypes", "storage", "page_bytes", "copy_aligned"):
                assert off[cls][key] == served[cls][key], (cls, key)
            assert served[cls]["copy_misaligned"] == off[cls]["copy_misaligned"] == "ok"
    assert served["CacheLayer_qsa_quant"]["shapes"] == [[4, 256, 128], [4, 256, 128], [4, 256, 16], [4, 256, 16],
                                                        [4, 256, 128], [4, 64, 128]]


@needs_served
def test_flag_on_allocation_ring(patched: Path, served_tree: Path):
    served = run_alloc(served_tree, None)
    on = run_alloc(patched, "1")
    rows = ring_rows(patched)
    assert rows == 20
    for cls in ("CacheLayer_qsa_quant", "CacheLayer_qsa"):
        rec = on[cls]
        assert rec["ring"] == rows
        assert rec["shapes"][-2] == [4, rows, 128] and rec["shapes"][-1] == [4, 64, 128]
        assert rec["shapes"][:-2] == served[cls]["shapes"][:-2]
        assert rec["storage"] == served[cls]["storage"] - 4 * (256 - rows) * 128 * 2
        assert rec["page_bytes"][-2] == rows * 128 * 2 and rec["page_bytes"][-2] % 16 == 0   # cache_rotate chunks
        # block-aligned copy: K/V and pooled copied, no raw rows; misaligned copy refused before any write
        assert all(rec["copy_aligned"][:-2]) and rec["copy_aligned"][-1] and rec["copy_aligned"][-2] is None
        assert rec["raw_untouched_after_copy"]
        assert rec["copy_misaligned"] == "raised" and rec["misaligned_left_dest_untouched"]


def test_bytes_per_token_and_projected_pool():
    per_layer_served = 512 + 512 + 32 + 32 + 256 + 64
    per_layer_ring = per_layer_served - 256 + 20 * 256 // 256
    assert (per_layer_served, per_layer_ring) == (1408, 1172)
    total = per_layer_served * 13 * 819_200
    saved = (256 - 20) * 13 * 819_200
    assert saved == 2_513_305_600
    pool = total / (per_layer_ring * 13)
    assert int(pool) // 256 * 256 == 984_064
    assert 819_200 + 10 * 16_384 <= pool < 819_200 + 11 * 16_384


@needs_served
def test_nvme_namespace_sees_the_flag(patched: Path):
    """engine_identity() (unchanged served code) puts every EXL3_* variable except EXL3_NVME_TIER* in the tier
    namespace, so flag on and flag off never share a namespace."""
    fns = functions(patched / "generator/disk_cache.py")
    src = (patched / "generator/disk_cache.py").read_text()
    ns = {"__file__": str(patched / "generator/disk_cache.py")}
    exec("import hashlib, os\nfrom pathlib import Path\n" +
         re.search(r"^_ENGINE_SUFFIXES = .*$", src, flags = re.M).group(0) + "\n" +
         re.search(r"^_ENV_EXCLUDE_PREFIXES = .*$", src, flags = re.M).group(0) + "\n" +
         fns["engine_identity"], ns)
    on = ns["engine_identity"](patched, {"EXL3_QSA_RAWK_RING": "1", "EXL3_NVME_TIER": "/x"})
    off = ns["engine_identity"](patched, {"EXL3_NVME_TIER": "/x"})
    assert on["env"] == [["EXL3_QSA_RAWK_RING", "1"]] and off["env"] == []
    assert on["sources_sha256"] == off["sources_sha256"]


# ---- kernel semantics through the CPU interpreter -----------------------------------------------------------------

@pytest.mark.parametrize("scenario", sc.POSITIVE, ids = [f.__name__ for f in sc.POSITIVE])
def test_ring_pooled_plane_equals_full_plane(kernels, patched: Path, scenario):
    ft.reset_races()
    r = scenario(kernels, "cpu", ring_rows(patched))
    assert r["pooled_equal"], r
    assert r["calls"] > 0
    assert ft.races() == []


@pytest.mark.parametrize("scenario", sc.NEGATIVE, ids = [f.__name__ for f in sc.NEGATIVE])
def test_controls_detect_a_ring_that_loses_rows(kernels, patched: Path, scenario):
    r = scenario(kernels, "cpu", ring_rows(patched))
    assert not r["pooled_equal"], r


def test_interpreter_matches_torch_reference_pooling(kernels):
    """The interpreter runs the served pool kernel like the eager torch reference (update_planes_ref arithmetic):
    fp32 mean -> fp16, RMS norm with the +1 bias, NEOX rope at the block start on the leading ROPE_R dims."""
    w = sc.World(kernels, "cpu", 4, 20, seed = 21)
    w.new_seq("a", 2)
    w.call(["a"], [0], 64, "eager")
    raw = w.ref.raw[w.tables["a"][0], :64].float()
    mean = raw.view(16, 4, 128).mean(1).half().float()
    rstd = torch.rsqrt((mean * mean).mean(-1, keepdim = True) + w.eps)
    y = (mean * rstd * (w.k_norm_w.float() + 1.0)).half()
    pos = torch.arange(0, 64, 4, dtype = torch.float32).unsqueeze(1)
    fr = w.inv_freq.unsqueeze(0) * pos
    cos, sin = (torch.cos(fr) * w.attn_factor).half(), (torch.sin(fr) * w.attn_factor).half()
    h = w.rope_r // 2
    lo, hi = y[:, :h], y[:, h:2 * h]
    ref = torch.cat((lo * cos - hi * sin, hi * cos + lo * sin, y[:, 2 * h:]), dim = 1)
    got = w.ref.pooled[w.tables["a"][0], :16]
    assert torch.allclose(got.float(), ref.float(), atol = 2e-3, rtol = 0)
    assert torch.equal(w.ref.pooled, w.rg.pooled)


def test_ring_append_keeps_only_last_rows(kernels):
    w = sc.World(kernels, "cpu", 4, 20, seed = 22)
    w.new_seq("a", 2)
    w.call(["a"], [0], 100, "eager")
    page = w.tables["a"][0]
    for t in range(80, 100):
        assert torch.equal(w.rg.raw[page, t % 20], w.ref.raw[page, t])
