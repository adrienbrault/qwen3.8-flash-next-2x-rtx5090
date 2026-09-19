#!/usr/bin/env python3
"""Local-only worktree verification; makes no CUDA/performance claim.

This is deliberately not copied into the image. Image-run checks live in
tests/test_prefill_e3_installed.py and only inspect installed or /opt files.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import py_compile
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
ROUND = ROOT / "out/prefill-e3-r2"
SERVED = ROOT / "ref/served-src/exllamav3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    checks: dict[str, bool] = {}
    manifest = json.loads((ROUND / "overlay/manifest.json").read_text())
    checks["refbase_image"] = manifest["base_image"] == "tabbyapi:decode-kernels-r2-refbase"
    checks["patch_hash"] = sha256(ROUND / "served-source.patch") == manifest["patch_sha256"]
    checks["added_hashes"] = all(
        sha256(ROUND / "overlay/exllamav3" / relative) == expected
        for relative, expected in manifest["added_files"].items()
    )
    checks["baseline_hashes"] = all(
        sha256(SERVED / relative) == record["baseline_sha256"]
        for relative, record in manifest["patched_files"].items()
    )

    with tempfile.TemporaryDirectory(prefix="prefill-e3-r2-") as temp:
        tree = Path(temp) / "exllamav3"
        shutil.copytree(SERVED, tree)
        subprocess.run(
            ["patch", "-p1", "--fuzz=0", "--no-backup-if-mismatch"],
            cwd=tree,
            stdin=(ROUND / "served-source.patch").open("rb"),
            check=True,
            stdout=subprocess.PIPE,
        )
        checks["patched_hashes"] = all(
            sha256(tree / relative) == record["overlay_sha256"]
            for relative, record in manifest["patched_files"].items()
        )
        py_compile.compile(str(tree / "modules/block_sparse_mlp.py"), doraise=True)
        checks["patched_python_compile"] = True

    python_files = [
        ROUND / "tests/bench_prefill_e3_layer.py",
        ROUND / "tests/inspect_checkpoint_k_layout.py",
        ROUND / "tests/test_prefill_e3_installed.py",
        ROUND / "overlay/install.py",
        ROUND / "overlay/build_extension.py",
    ]
    with tempfile.TemporaryDirectory(prefix="prefill-e3-r2-pyc-") as pyc_temp:
        for index, source in enumerate(python_files):
            py_compile.compile(
                str(source),
                cfile=str(Path(pyc_temp) / f"{index}.pyc"),
                doraise=True,
            )
    checks["artifact_python_compile"] = True

    cpp = (ROUND / "overlay/exllamav3/exllamav3_ext/quant/exl3_moe_prefill_e3.cpp").read_text()
    cu = (ROUND / "overlay/exllamav3/exllamav3_ext/quant/exl3_moe_prefill_e3.cu").read_text()
    patch = (ROUND / "served-source.patch").read_text()
    dockerfile = (ROUND / "Dockerfile.box").read_text()
    installed_test = (ROUND / "tests/test_prefill_e3_installed.py").read_text()
    benchmark = (ROUND / "tests/bench_prefill_e3_layer.py").read_text()
    checks["host_c10_cuda_check"] = (
        "#include <c10/cuda/CUDAException.h>" in cpp and "C10_CUDA_CHECK(" in cpp
    )
    checks["literal_default_off"] = (
        'os.environ.get("EXL3_MOE_PREFILL_E3", "0") == "1"' in patch
    )
    checks["decode_excluded"] = (
        "MOE_PREFILL_E3_MIN_ROWS = max(512" in patch
        and "num_tokens >= MOE_PREFILL_E3_MIN_ROWS" in patch
    )
    checks["all_k_independent"] = all(token in cu + patch for token in (
        "dq_dispatch<BITS0, E3_CB>",
        "dq_dispatch<BITS1, E3_CB>",
        "e3_launch_down<2>", "e3_launch_down<3>", "e3_launch_down<4>",
        "self.multi_gate.K, self.multi_up.K, self.multi_down.K",
    ))
    checks["mul1_only"] = "constexpr int E3_CB = 2" in cu
    checks["tile_64x128x32"] = all(token in cu for token in (
        "constexpr int E3_TILE_N = 128",
        "constexpr int E3_TILE_K = 32",
        "constexpr int E3_MB = 4",
    ))
    mapping_ok = True
    for bits0, bits1 in itertools.product((2, 3, 4), repeat=2):
        pw0, pw1 = bits0 * 16, bits1 * 16
        chunks0 = 2 * 8 * (pw0 // 8)
        chunks1 = 2 * 8 * (pw1 // 8)
        covered: list[int] = []
        for chunk in range(chunks0 + chunks1):
            stream = int(chunk >= chunks0)
            local = chunk - chunks0 if stream else chunk
            pc = (pw1 if stream else pw0) // 8
            pw = pw1 if stream else pw0
            half_tile = local // (8 * pc)
            rest = local - half_tile * 8 * pc
            warp = rest // pc
            vector = rest - warp * pc
            base = 2 * 8 * pw0 if stream else 0
            start = base + half_tile * 8 * pw + warp * pw + vector * 8
            covered.extend(range(start, start + 8))
        mapping_ok &= sorted(covered) == list(range(2 * 8 * (pw0 + pw1)))
    checks["mixed_k_b_stage_mapping"] = mapping_ok
    checks["round_root_build_context"] = (
        "COPY served-source.patch " in dockerfile
        and "COPY overlay/" in dockerfile
        and "COPY out/" not in dockerfile
    )
    checks["legacy_builder_safe"] = "<<" not in dockerfile
    checks["image_test_is_worktree_independent"] = (
        'Path("/opt/prefill-e3-r2")' in installed_test
        and "find_spec(\"exllamav3\")" in installed_test
        and "ref/served-src" not in installed_test
        and "out/prefill" not in installed_test
    )
    checks["benchmark_per_card_per_k"] = all(token in benchmark for token in (
        "target_k = [2, 3] + ([4] if 4 in available_k else [])",
        'row_counts != [512, 2048]',
        'use != [30.0, 30.0]',
        'torch.cuda.Event(enable_timing=True)',
        '"candidate_finite"',
    ))
    checks["cuda_unavailable"] = shutil.which("nvcc") is None
    checks["torch_unavailable"] = subprocess.run(
        ["python3", "-c", "import torch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode != 0
    checks["all_cpu_checks_pass"] = all(
        value for key, value in checks.items()
        if key not in {"cuda_unavailable", "torch_unavailable", "all_cpu_checks_pass"}
    )
    output = {"schema_version": 1, "scope": "local-only", "checks": checks}
    (ROUND / "local-validation.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if checks["all_cpu_checks_pass"] else 1)


if __name__ == "__main__":
    main()
