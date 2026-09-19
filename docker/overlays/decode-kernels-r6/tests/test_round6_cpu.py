"""CPU-only round-6 checks (pytest). No torch, no CUDA, no network.

The served source is read from R6_SERVED_SRC, a directory holding the exllamav3 package extracted
from tabbyapi:nvme-tier-r4-e3det. Tests that need it skip when the variable is unset or the
directory is absent.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import py_compile
import re
import shutil
import subprocess
import sys

import pytest


HERE = Path(__file__).resolve().parent
R6 = HERE.parent
_SERVED_ENV = os.environ.get("R6_SERVED_SRC")
SERVED = Path(_SERVED_ENV) if _SERVED_ENV else None
PATCH = R6 / "served-source.patch"
MANIFEST = json.loads((R6 / "overlay/manifest.json").read_text())
DAILY_ENV = (
    "EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 "
    "EXL3_MOE_COOP_V2=1 EXL3_SHARED_EXPERT_OVERLAP=1 EXL3_DRAFT_PINNED_STAGING=1 EXL3_BATCH_VERIFY=1 "
    "EXL3_MTP_HEAD_N=65536 EXL3_MOE_PREFILL_E3=1 EXL3_HC_MIX_V2_INT8=1 EXL3_MTP_DEVICE_DRAFT=1 "
    "EXL3_EMBED_GPU=1 EXL3_EMBED_GPU_PRUNED=1 EXL3_MOE_PREFILL_E3_DET=1"
)

needs_served = pytest.mark.skipif(SERVED is None or not SERVED.is_dir(),
                                  reason="set R6_SERVED_SRC to the served exllamav3 package directory")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_touched(tmp_path: Path) -> Path:
    """Copy only the files the patch touches; the rest of the tree is irrelevant to patch(1)."""
    target = tmp_path / "exllamav3"
    for relative in MANIFEST["files"]:
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SERVED / relative, target / relative)
    return target


def apply_patch(root: Path) -> subprocess.CompletedProcess:
    with PATCH.open("rb") as stream:
        return subprocess.run(["patch", "-p1", "-F0"], cwd=root, stdin=stream, capture_output=True)


@pytest.fixture
def patched(tmp_path: Path) -> Path:
    root = copy_touched(tmp_path)
    proc = apply_patch(root)
    assert proc.returncode == 0, proc.stdout.decode() + proc.stderr.decode()
    return root


# ---- patch + manifest ----------------------------------------------------------------

def test_patch_hash_pinned():
    assert sha256(PATCH) == MANIFEST["patch_sha256"]
    assert MANIFEST["base_image"] == "tabbyapi:nvme-tier-r4-e3det"


def test_patch_touches_exactly_manifest_files():
    touched = re.findall(r"^\+\+\+ b/(\S+)", PATCH.read_text(), flags=re.M)
    assert sorted(touched) == sorted(MANIFEST["files"])


@needs_served
def test_served_baseline_hashes():
    for relative, record in MANIFEST["files"].items():
        assert sha256(SERVED / relative) == record["baseline_sha256"], relative


@needs_served
def test_patch_applies_fuzz0_exit_code_and_overlay_hashes(patched: Path):
    assert not list(patched.rglob("*.rej"))
    for relative, record in MANIFEST["files"].items():
        assert sha256(patched / relative) == record["overlay_sha256"], relative


@needs_served
def test_patch_does_not_reapply(patched: Path):
    # A second forward application must fail (guards against a silently double-patched image).
    with PATCH.open("rb") as stream:
        proc = subprocess.run(["patch", "-p1", "-F0", "--forward", "--dry-run"], cwd=patched,
                              stdin=stream, capture_output=True)
    assert proc.returncode != 0
    out = (proc.stdout + proc.stderr).decode(errors="replace").lower()
    assert any(w in out for w in ("reversed", "already applied", "skipping", "previously applied")), out


@needs_served
def test_install_py_on_scratch_root(tmp_path: Path):
    root = copy_touched(tmp_path)
    proc = subprocess.run([sys.executable, str(R6 / "overlay/install.py"), "--root", str(root), "--keep-extension"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for relative, record in MANIFEST["files"].items():
        assert sha256(root / relative) == record["overlay_sha256"]
    again = subprocess.run([sys.executable, str(R6 / "overlay/install.py"), "--root", str(root), "--keep-extension"],
                           capture_output=True, text=True)
    assert again.returncode != 0 and "baseline hash mismatch" in (again.stdout + again.stderr)


@needs_served
def test_install_py_rejects_foreign_baseline(tmp_path: Path):
    root = copy_touched(tmp_path)
    target = root / "modules/hyperconnections.py"
    target.write_text(target.read_text() + "\n# drift\n")
    proc = subprocess.run([sys.executable, str(R6 / "overlay/install.py"), "--root", str(root), "--keep-extension"],
                          capture_output=True, text=True)
    assert proc.returncode != 0 and "baseline hash mismatch" in (proc.stdout + proc.stderr)
    # Nothing was patched.
    assert sha256(root / "exllamav3_ext/gdn.cu") == MANIFEST["files"]["exllamav3_ext/gdn.cu"]["baseline_sha256"]


# ---- selector default-off (structural) ------------------------------------------------

@needs_served
def test_python_selector_literal_default_off(patched: Path):
    module = patched / "modules/hyperconnections.py"
    py_compile.compile(str(module), doraise=True)
    source = module.read_text()
    assert 'STATE_REGRID = os.environ.get("EXL3_GR_STATE_REGRID", "0") == "1"' in source
    # Unset selector: the int8 branch resolves to the served entry point, the fp16 branch to the served choice.
    assert "mix_fn = ext.gr_mix_v2_int8_regrid if self.STATE_REGRID else ext.gr_mix_v2_int8" in source
    assert "ext.gr_mix_v2_regrid if self.STATE_REGRID else (\n" \
           "                        ext.gr_mix_v2_fused if self.DECODE_FUSE else ext.gr_mix_v2" in source
    assert "FREE_UP_H" not in source and "_checkpoint_up" not in source


@needs_served
def test_cpp_selectors_literal_one_only(patched: Path):
    gdn = (patched / "exllamav3_ext/gdn.cu").read_text()
    hc = (patched / "exllamav3_ext/hc_mix.cu").read_text()
    assert 'std::getenv("EXL3_GDN_BA_WARP1")' in gdn
    assert 'std::getenv("EXL3_HC_APPLY_WARP1")' in hc
    for text in (gdn, hc):
        assert "value && value[0] == '1' && value[1] == '\\0'" in text
    # Served entry points pass state_regrid = false; only *_regrid entry points can enable it.
    assert "gr_mix_v2_impl(streams, fn, upt, w, rms_eps, dots, state, post, mixed, false);" in hc
    assert "gr_mix_v2_int8_impl(streams, fn_q, fn_s, up_q, up_s, w, rms_eps, dots, state, post, mixed, false);" in hc
    assert hc.count("state_regrid = prop->major == 12 && prop->minor == 0;") == 2


def test_no_selector_enabled_by_default_in_dockerfile():
    text = (R6 / "overlay/Dockerfile.box").read_text()
    for key in ("EXL3_GDN_BA_WARP1", "EXL3_HC_APPLY_WARP1", "EXL3_GR_STATE_REGRID", "EXL3_GR_FREE_UP_H"):
        assert not re.search(rf"^ENV\s+{key}=1", text, flags=re.M)


@needs_served
def test_unpatched_served_has_no_round6_symbols():
    for relative in MANIFEST["files"]:
        text = (SERVED / relative).read_text()
        for token in ("EXL3_GDN_BA_WARP1", "EXL3_HC_APPLY_WARP1", "EXL3_GR_STATE_REGRID", "_regrid"):
            assert token not in text, (relative, token)


# ---- arithmetic-order and binding contracts -------------------------------------------

@needs_served
def test_exact_order_contracts(patched: Path):
    gdn = (patched / "exllamav3_ext/gdn.cu").read_text()
    hc = (patched / "exllamav3_ext/hc_mix.cu").read_text()
    assert "for (int j = lane; j < k / 2; j += 32)" in gdn
    assert "for (int offset = 16; offset > 0; offset >>= 1)" in gdn
    assert "int row = blockIdx.x * WARPS + warp;" in gdn
    assert "c += THREADS" in hc
    # The served state kernel and the re-grid kernel carry the same per-output FMA chain.
    chain = "for (int h = 0; h < H; ++h) v = fmaf(rmr[h], dr[(size_t) i * H + h], v);"
    assert hc.count(chain) == 2
    assert hc.count("v *= 1.0f / (float) H;") >= 2
    # The int8 launcher branches between the two state kernels, nothing else changes there.
    launch_i8 = hc[hc.index("static void gr_v2_launch_i8"):hc.index("static bool gr_v2_launch_fused")]
    assert "gr_v2_state_regrid_kernel<<<" in launch_i8 and "gr_v2_state_kernel<<<R, GR_THREADS_A" in launch_i8
    assert "gr_v2_dots_i8_kernel<B><<<" in launch_i8 and "gr_v2_up_i8_kernel<B, true>" in launch_i8


@needs_served
def test_bindings_keep_served_chain(patched: Path):
    bindings = (patched / "exllamav3_ext/bindings.cpp").read_text()
    assert "<<<" not in bindings
    for name in ("gr_mix_v2_regrid", "gr_mix_v2_int8_regrid", "gr_mix_v2_int8", "exl3_moe_prefill_e3_det"):
        assert f'm.def("{name}"' in bindings, name
    header = (patched / "exllamav3_ext/hc_mix.cuh").read_text()
    assert "void gr_mix_v2_int8_regrid" in header and "void gr_mix_v2_regrid" in header


def test_dockerfile_contract():
    text = (R6 / "overlay/Dockerfile.box").read_text()
    assert re.search(r"^ARG BASE=tabbyapi:nvme-tier-r4-e3det$", text, flags=re.M)
    assert "ENV TORCH_CUDA_ARCH_LIST=12.0" in text
    assert "MAX_JOBS=8" in text
    assert "rm -rf ${TORCH_EXTENSIONS_DIR}" in text
    for name in ("gr_mix_v2_int8_regrid", "gr_mix_v2_regrid", "exl3_moe_prefill_e3_det"):
        assert f"hasattr(e, '{name}')" in text
    build = (R6 / "overlay/build_extension.py").read_text()
    assert '"exl3_moe_prefill_e3_det"' in build and '"gr_mix_v2_int8_regrid"' in build


# ---- the central question, pinned against the served source ---------------------------

@needs_served
def test_daily_env_routes_decode_through_int8_mixer():
    src = (SERVED / "modules/hyperconnections.py").read_text()
    assert "EXL3_HC_MIX_V2_INT8=1" in DAILY_ENV and "EXL3_HC_MIX_V2_MIN_R=1" in DAILY_ENV
    # INT8 + MIN_R<=1 releases the fp16 decode copies and keeps up_h/proj_h for prefill.
    assert "if self.MIX_V2_INT8 and self.MIX_V2:" in src
    assert "if self.MIX_V2_MIN_R <= 1:\n            self.fn_h = None\n            self.upx_h = None" in src
    assert "g = torch.matmul(t, self.up_h.t())" in src
    # The int8 launcher reuses the served fp32 state kernel.
    hc = (SERVED / "exllamav3_ext/hc_mix.cu").read_text()
    launch_i8 = hc[hc.index("static void gr_v2_launch_i8"):hc.index("static bool gr_v2_launch_fused")]
    assert "gr_v2_state_kernel<<<R, GR_THREADS_A, 0, stream>>>" in launch_i8


# ---- GPU harness is importable and plannable without torch/CUDA ------------------------

def test_harness_dry_run_without_torch(tmp_path: Path):
    out = tmp_path / "plan.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    proc = subprocess.run([sys.executable, "-S", str(HERE / "test_r6_kernels.py"), "--dry-run", "--json", str(out)],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(out.read_text())
    plan = payload["plan"]
    assert plan["rows"] == list(range(1, 17))
    assert len(plan["gdn_ba"]) == 32 and len(plan["hc_apply"]) == 64 and len(plan["gr_state"]) == 128
    assert {c["weights"] for c in plan["gr_state"]} == {"fp16", "int8"}
    # Geometry the box spec quotes: c1/d3 = 4 rows.
    assert next(c for c in plan["gdn_ba"] if c["rows"] == 4)["served_blocks"] == 48
    hc4 = next(c for c in plan["hc_apply"] if c["rows"] == 4)
    assert hc4["served_blocks"] == 12 and hc4["warp1_blocks"] == 80
    assert next(c for c in plan["gr_state"] if c["rows"] == 4 and c["post"])["regrid_blocks"] == 44
    assert next(c for c in plan["gdn_ba"] if c["rows"] == 16)["warp1_selected"] is False


def test_harness_module_has_no_top_level_torch_import():
    text = (HERE / "test_r6_kernels.py").read_text()
    assert not re.search(r"^(import torch|from torch|from exllamav3)", text, flags=re.M)
