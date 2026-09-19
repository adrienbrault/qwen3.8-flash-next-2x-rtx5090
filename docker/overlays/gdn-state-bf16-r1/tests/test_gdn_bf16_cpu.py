"""CPU-only checks for gdn-state-bf16 r1 (pytest). No torch, no CUDA, no network.

The served source defaults to the local extraction of tabbyapi:nvme-tier-r4-e3det (R535); override with
GDN_BF16_SERVED_SRC. Tests that need it skip when it is absent.
"""
from __future__ import annotations

import ast
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
R1 = HERE.parent
PATCHES = R1.parent.parent
SERVED = Path(os.environ.get("GDN_BF16_SERVED_SRC", "served-src-r535/exllamav3"))
PATCH = R1 / "served-source.patch"
MANIFEST = json.loads((R1 / "overlay/manifest.json").read_text())
R6_PATCH = PATCHES / "decode-kernels/r6/served-source.patch"
ADAPTIVE_PATCH = PATCHES / "adaptive-draft/r1/served-source.patch"

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


@pytest.fixture
def patched(tmp_path: Path) -> Path:
    root = copy_files(tmp_path, MANIFEST["files"])
    proc = apply(root, PATCH)
    assert proc.returncode == 0, proc.stdout.decode() + proc.stderr.decode()
    return root


def region(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i:text.index(end, i)]


# ---- patch + manifest -------------------------------------------------------------------------------------------

def test_patch_hash_pinned():
    assert sha256(PATCH) == MANIFEST["patch_sha256"]
    assert MANIFEST["base_image"] == "tabbyapi:nvme-tier-r4-e3det"
    assert MANIFEST["selectors"] == {"EXL3_GDN_STATE_BF16": "modules/gated_delta_net.py"}


def test_patch_touches_exactly_manifest_files():
    text = PATCH.read_text()
    touched = re.findall(r"^\+\+\+ b/(\S+)", text, flags = re.M)
    assert sorted(touched) == sorted(MANIFEST["files"])
    assert sorted(re.findall(r"^--- a/(\S+)", text, flags = re.M)) == sorted(MANIFEST["files"])
    # generator/*.py belong to the adaptive-draft round; this round stays out of them
    assert not any(p.startswith("generator/") for p in touched)


@needs_served
def test_served_baseline_hashes():
    for relative, record in MANIFEST["files"].items():
        assert sha256(SERVED / relative) == record["baseline_sha256"], relative


@needs_served
def test_patch_applies_fuzz0_exit_code_and_overlay_hashes(patched: Path):
    assert not list(patched.rglob("*.rej")) and not list(patched.rglob("*.orig"))
    for relative, record in MANIFEST["files"].items():
        assert sha256(patched / relative) == record["overlay_sha256"], relative


@needs_served
def test_patch_does_not_reapply(patched: Path):
    proc = apply(patched, PATCH, "--forward", "--dry-run")
    assert proc.returncode != 0


@needs_served
def test_install_py_on_scratch_root(tmp_path: Path):
    root = copy_files(tmp_path, MANIFEST["files"])
    cmd = [sys.executable, str(R1 / "overlay/install.py"), "--root", str(root), "--keep-extension"]
    proc = subprocess.run(cmd, capture_output = True, text = True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for relative, record in MANIFEST["files"].items():
        assert sha256(root / relative) == record["overlay_sha256"]
    again = subprocess.run(cmd, capture_output = True, text = True)
    assert again.returncode != 0 and "baseline hash mismatch" in again.stdout + again.stderr


@needs_served
def test_install_py_rejects_foreign_baseline(tmp_path: Path):
    root = copy_files(tmp_path, MANIFEST["files"])
    target = root / "modules/gated_delta_net_fn/gated_delta_rule.py"
    target.write_text(target.read_text() + "\n# drift\n")
    proc = subprocess.run([sys.executable, str(R1 / "overlay/install.py"), "--root", str(root), "--keep-extension"],
                          capture_output = True, text = True)
    assert proc.returncode != 0 and "baseline hash mismatch" in proc.stdout + proc.stderr
    assert sha256(root / "exllamav3_ext/gdn.cu") == MANIFEST["files"]["exllamav3_ext/gdn.cu"]["baseline_sha256"]


# ---- stacking with the rounds in flight -------------------------------------------------------------------------

MANIFEST_R6 = json.loads((R1 / "overlay/manifest-on-r6.json").read_text())


@needs_served
@pytest.mark.skipif(not R6_PATCH.is_file(), reason = "decode-kernels r6 patch not present")
def test_install_py_on_r6_stacked_root(tmp_path: Path):
    """--manifest manifest-on-r6.json: same patch bytes, pinned on top of decode-kernels r6"""
    assert MANIFEST_R6["patch_sha256"] == MANIFEST["patch_sha256"]
    assert sorted(MANIFEST_R6["files"]) == sorted(MANIFEST["files"])
    r6_files = re.findall(r"^\+\+\+ b/(\S+)", R6_PATCH.read_text(), flags = re.M)
    root = copy_files(tmp_path, sorted(set(r6_files) | set(MANIFEST["files"])))
    assert apply(root, R6_PATCH).returncode == 0
    cmd = [sys.executable, str(R1 / "overlay/install.py"), "--root", str(root), "--keep-extension"]
    wrong = subprocess.run(cmd, capture_output = True, text = True)
    assert wrong.returncode != 0 and "baseline hash mismatch" in wrong.stdout + wrong.stderr
    proc = subprocess.run(cmd + ["--manifest", "manifest-on-r6.json"], capture_output = True, text = True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for relative, record in MANIFEST_R6["files"].items():
        assert sha256(root / relative) == record["overlay_sha256"], relative
    # the r6 selector survives in the stacked gdn.cu
    assert 'std::getenv("EXL3_GDN_BA_WARP1")' in (root / "exllamav3_ext/gdn.cu").read_text()


@needs_served
@pytest.mark.skipif(not R6_PATCH.is_file(), reason = "decode-kernels r6 patch not present")
def test_stacks_with_decode_kernels_r6_both_orders(tmp_path: Path):
    r6_files = re.findall(r"^\+\+\+ b/(\S+)", R6_PATCH.read_text(), flags = re.M)
    files = sorted(set(r6_files) | set(MANIFEST["files"]))
    results = []
    for order in ((R6_PATCH, PATCH), (PATCH, R6_PATCH)):
        root = copy_files(tmp_path / f"o{len(results)}", files)
        for p in order:
            proc = apply(root, p)
            assert proc.returncode == 0, (p, proc.stdout.decode() + proc.stderr.decode())
        assert not list(root.rglob("*.rej"))
        results.append({f: sha256(root / f) for f in files})
    assert results[0] == results[1]


@pytest.mark.skipif(not ADAPTIVE_PATCH.is_file(), reason = "adaptive-draft r1 patch not present")
def test_disjoint_from_adaptive_draft():
    theirs = {re.sub(r"^exllamav3/", "", p) for p in re.findall(r"^\+\+\+ b/(\S+)", ADAPTIVE_PATCH.read_text(), flags = re.M)}
    assert not theirs & set(MANIFEST["files"])


# ---- default off: Python --------------------------------------------------------------------------------------

@needs_served
def test_python_compiles(patched: Path):
    for relative in MANIFEST["files"]:
        if relative.endswith(".py"):
            py_compile.compile(str(patched / relative), doraise = True)


@needs_served
def test_selector_literal_default_off(patched: Path):
    src = (patched / "modules/gated_delta_net.py").read_text()
    assert '_gdn_state_bf16 = os.environ.get("EXL3_GDN_STATE_BF16", "0") == "1"' in src
    assert src.count("EXL3_GDN_STATE_BF16") >= 1
    # Only GatedDeltaNet sets the attribute, fp32 unless the selector is on, and KDA always fp32
    assert "self.recurrent_state_dtype = torch.bfloat16 if _gdn_state_bf16 and not self.kda else torch.float" in src
    assert src.count("recurrent_state_dtype") == 2
    # GDNLayerState falls back to the served fp32 for modules without the attribute (Mamba2 reuses the class)
    assert 'dtype = getattr(module, "recurrent_state_dtype", torch.float),' in src
    served = (SERVED / "modules/mamba2.py").read_text()
    assert "from .gated_delta_net import GDNLayerState" in served and "recurrent_state_dtype" not in served


@needs_served
def test_state_words_identity_for_fp32(patched: Path):
    """Extract _state_words and evaluate it on stand-in tensors: fp32 keeps the served element count"""
    src = (patched / "modules/gated_delta_net.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_state_words")
    fn.args.args[0].annotation = None
    ns = {}
    exec(compile(ast.Module(body = [fn], type_ignores = []), "x", "exec"), ns)

    class T:
        def __init__(self, size):
            self.size = size

        def element_size(self):
            return self.size

    plane = 48 * 128 * 128
    assert ns["_state_words"](T(4), plane) == plane                    # served job size, unchanged
    assert ns["_state_words"](T(2), plane) == plane // 2               # bf16: same bytes as the plane
    assert (ns["_state_words"](T(2), plane) * 4) == plane * 2
    assert ns["_state_words"](T(2), plane) % 4 == 0                    # batched_state_rewind float4 contract


@needs_served
def test_rewind_jobs_route_through_state_words(patched: Path):
    src = (patched / "modules/gated_delta_net.py").read_text()
    body = region(src, "    def rewind_state_job(", "    def stash(")
    assert body.count("_state_words(") == 2
    served = region((SERVED / "modules/gated_delta_net.py").read_text(), "    def rewind_state_job(", "    def stash(")
    reverted = body.replace("_state_words(t, t.numel() // shape[0] // shape[1])", "t.numel() // shape[0] // shape[1]")
    reverted = reverted.replace("_state_words(self.recurrent_state, self.recurrent_state[slot, 0].numel())",
                                "self.recurrent_state[slot, 0].numel()")
    assert reverted == served


@needs_served
def test_checkpoint_size_and_unstash(patched: Path):
    src = (patched / "modules/gated_delta_net.py").read_text()
    assert "self.module.num_v_heads * self.module.k_head_dim * self.module.v_head_dim * self.recurrent_state.element_size()" in src
    unstash = region(src, "    def unstash(self, slot, stashed", "    def tp_export(")
    assert "if s.dtype != self.recurrent_state.dtype:" in unstash and "raise TypeError" in unstash
    # stash is untouched: a checkpoint keeps the live dtype
    assert region(src, "    def stash(self, slot", "    def unstash(") == \
        region((SERVED / "modules/gated_delta_net.py").read_text(), "    def stash(self, slot", "    def unstash(")


@needs_served
def test_prefill_fp32_passes_same_state_object(patched: Path):
    src = (patched / "modules/gated_delta_net_fn/gated_delta_rule.py").read_text()
    assert "initial_state = state if state is None or state.dtype == torch.float else state.float()," in src
    # the KDA chunk call (fp32-only, asserts float32) is unchanged
    assert src.count("state.float()") == 1
    kda = region(src, "            from ...vendor.fla import chunk_kda", "        core_attn_out = torch.empty(")
    assert "initial_state = state,\n" in kda


# ---- default off: CUDA ----------------------------------------------------------------------------------------

KERNELS_START = "template <int MAX_HEAD_DIM, bool save_history, int V_SPLIT, bool MAMBA2 = false"
KERNELS_END = "void cuda_recurrent_gated_delta_rule_gr\n"
HOST_START = "void cuda_recurrent_gated_delta_rule_gr\n"
HOST_END = "void cuda_recurrent_gated_delta_rule\n("


def revert_kernels(text: str) -> str:
    """Map the templated kernels back to the served text: state_t -> float, as_float(*p) -> *p,
    store_state(p, x) -> *p = x. Equality with the served region proves the fp32 instantiation is the served
    kernel modulo an identity load/store wrapper."""
    text = text.replace(", typename state_t = float>", ">")
    text = text.replace("state_t* __restrict__ recurrent_state,      //", "float* __restrict__ recurrent_state,        //")
    text = text.replace("state_t*", "float*")
    text = re.sub(r"as_float\(\*(\w+)\)", r"*\1", text)
    text = re.sub(r"store_state\((\w+), (\w+)\);", r"*\1 = \2;", text)
    return text


@needs_served
def test_kernels_are_served_modulo_state_type(patched: Path):
    new = (patched / "exllamav3_ext/gdn.cu").read_text()
    old = (SERVED / "exllamav3_ext/gdn.cu").read_text()
    a = region(old, KERNELS_START, KERNELS_END)
    b = region(new, KERNELS_START, KERNELS_END)
    assert b != a
    assert revert_kernels(b) == a


@needs_served
def test_state_template_param_is_last_and_defaults_to_float(patched: Path):
    new = (patched / "exllamav3_ext/gdn.cu").read_text()
    assert "template <int MAX_HEAD_DIM, bool save_history, int V_SPLIT, bool MAMBA2 = false, typename state_t = float>" in new
    assert "template <bool save_history, int V_SPLIT, bool CHANNELWISE = false, typename state_t = float>" in new


@needs_served
def test_store_helpers(patched: Path):
    new = (patched / "exllamav3_ext/gdn.cu").read_text()
    assert "__device__ __forceinline__ void store_state(float* p, float x)\n{\n    *p = x;\n}" in new
    assert "__device__ __forceinline__ void store_state(bfloat16* p, float x)\n{\n    *p = __float2bfloat16_rn(x);\n}" in new
    # load widening reuses the served as_float overloads (float identity, bf16 exact widening)
    assert "__device__ __forceinline__ float as_float(float x)\n{\n    return x;\n}" in new


@needs_served
def test_host_dispatch_served_block_unchanged(patched: Path):
    new = region((patched / "exllamav3_ext/gdn.cu").read_text(), HOST_START, HOST_END)
    old = region((SERVED / "exllamav3_ext/gdn.cu").read_text(), HOST_START, HOST_END)
    # the bf16 block sits before the served macros and returns; removing it and the dtype-check change
    # must give back the served host function verbatim
    start = new.index("    // bf16 state: the same kernels instantiated")
    end = new.index("    #define KERNEL_ARGS                         \\")
    stripped = new[:start] + new[end:]
    stripped = stripped.replace(
        "    // fp32 state is the served layout; bf16 state is the EXL3_GDN_STATE_BF16 storage option\n"
        "    bool bf16_state = recurrent_state.dtype() == at::kBFloat16;\n"
        "    if (!bf16_state)\n"
        "    {\n"
        "        TORCH_CHECK_DTYPE(recurrent_state, kFloat);\n"
        "    }\n"
        "    TORCH_CHECK(!bf16_state || !channelwise,\n"
        "                \"cuda_recurrent_gated_delta_rule: bf16 recurrent_state is not supported with channelwise (KDA) decay\");\n",
        "    TORCH_CHECK_DTYPE(recurrent_state, kFloat);\n",
    )
    assert stripped == old
    bf16_block = new[start:end]
    assert "return;" in bf16_block and "cuda_check(cudaPeekAtLastError());" in bf16_block
    # served launches: no explicit state type, so state_t = float (the served instantiations)
    served_launches = re.findall(r"LAUNCH_RULE\((cuda_recurrent_gated_delta_rule_kernel\w*<[^>]*>)\)", old)
    assert len(served_launches) == 14
    assert re.findall(r"LAUNCH_RULE\((cuda_recurrent_gated_delta_rule_kernel\w*<[^>]*>)\)", stripped) == served_launches
    # bf16 launches: every one names bfloat16 explicitly, none is channelwise
    bf16_launches = re.findall(r"LAUNCH_RULE_BF16\((cuda_recurrent_gated_delta_rule_kernel\w*<[^>]*>)\)", bf16_block)
    assert len(bf16_launches) == 10
    assert all(l.endswith(", false, bfloat16>") for l in bf16_launches)
    assert "(bfloat16*) recurrent_state.data_ptr()" in bf16_block
    assert "GP_gdn_rule_state, 3" in bf16_block and "GP_gdn_rule_slots, 12" in bf16_block


@needs_served
def test_mamba2_and_rewind_kernels_untouched(patched: Path):
    new = (patched / "exllamav3_ext/gdn.cu").read_text()
    old = (SERVED / "exllamav3_ext/gdn.cu").read_text()
    for start, end in (("void cuda_recurrent_mamba2_gr", "void cuda_recurrent_mamba2\n"),
                       ("void batched_state_rewind_kernel", "void batched_conv_rewind("),
                       ("void batched_state_rewind(", "\n}\n")):
        assert region(new, start, end) == region(old, start, end), start
    assert region(new, "void cuda_recurrent_mamba2_gr", "void cuda_recurrent_mamba2\n").count(
        "TORCH_CHECK_DTYPE(recurrent_state, kFloat);") == 1


@needs_served
def test_no_other_state_consumers_assume_fp32():
    """The producer/consumer map: every C++ reader of recurrent_state outside gdn.cu only forwards the tensor."""
    cpp = (SERVED / "exllamav3_ext/libtorch/gated_delta_net.cpp").read_text()
    assert "TORCH_CHECK_DTYPE(recurrent_state" not in cpp and "(float*) recurrent_state" not in cpp
    assert cpp.count("recurrent_state.data_ptr()") == 4       # graph pointer patches (untyped void*)
    disk = (SERVED / "generator/disk_cache.py").read_text()
    # NVMe tier identity: every EXL3_* flag and the per-layer checkpoint bytes are in the namespace
    assert 'if k.startswith("EXL3_") and not k.startswith(_ENV_EXCLUDE_PREFIXES)' in disk
    assert '"checkpoint_bytes": layer.get_checkpoint_size(),' in disk
    # checkpoint payloads carry the dtype per tensor
    assert 'entries.append([str(t.dtype).replace("torch.", ""), list(t.shape), offset, nbytes])' in disk


# ---- packaging ------------------------------------------------------------------------------------------------

def test_dockerfile_contract():
    text = (R1 / "Dockerfile.box").read_text()
    assert re.search(r"^ARG BASE=tabbyapi:nvme-tier-r4-e3det$", text, flags = re.M)
    assert "ENV TORCH_CUDA_ARCH_LIST=12.0" in text
    assert "rm -rf /opt/gdn-state-bf16-r1/torch-extensions" in text
    assert "rm -f /opt/venv/lib/python3.12/site-packages/exllamav3_ext*.so" in text
    assert "build_extension.py" in text
    for binding in ("exl3_moe_prefill_e3_det", "gr_mix_v2_int8", "cuda_recurrent_gated_delta_rule", "batched_state_rewind"):
        assert f"hasattr(e, '{binding}')" in text
    assert "assert g._gdn_state_bf16 is False" in text
    assert not re.search(r"^ENV\s+EXL3_GDN_STATE_BF16", text, flags = re.M)
    assert "<<" not in text                                   # legacy builder: no heredocs
    for line in re.findall(r"^COPY (.+)$", text, flags = re.M):
        for src in line.split()[:-1]:
            assert (R1 / src).exists(), src


def test_overlay_files_present():
    for name in ("install.py", "manifest.json", "manifest-on-r6.json", "build_extension.py"):
        assert (R1 / "overlay" / name).is_file()
    for name in ("impl-status.md", "box-ab-spec.md", "served-source.patch", "Dockerfile.box"):
        assert (R1 / name).is_file()


def test_gpu_harness_cli():
    path = HERE / "gpu_gdn_bf16.py"
    py_compile.compile(str(path), doraise = True)
    src = path.read_text()
    for flag in ('"--device"', '"--json"', '"--dump"', '"--ref"', '"--steps"', '"--prefill-tokens"'):
        assert flag in src
    assert "return 0 if not failures else 1" in src
    assert 'failures += [f"identity:{k}" for k, v in ident.items() if not v]' in src
    assert 'ap.add_argument("--steps", type = int, default = 512)' in src
    assert 'ap.add_argument("--prefill-tokens", type = int, default = 32768)' in src
