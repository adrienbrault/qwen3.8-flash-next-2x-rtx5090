#!/usr/bin/env python3
"""CPU/static smoke checks for the installed round-2 image artifact.

This script intentionally discovers exllamav3 through importlib and reads only
the installed package plus files copied to /opt/prefill-e3-r2. It has no
dependency on the source worktree layout.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ARTIFACT = Path("/opt/prefill-e3-r2")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_root() -> Path:
    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise AssertionError("cannot locate installed exllamav3 package")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def run_checks() -> dict[str, bool]:
    root = package_root()
    manifest = json.loads((ARTIFACT / "overlay/manifest.json").read_text())
    checks: dict[str, bool] = {}
    checks["refbase_image"] = manifest["base_image"] == "tabbyapi:decode-kernels-r2-refbase"
    checks["patch_payload_hash"] = (
        digest(ARTIFACT / "served-source.patch") == manifest["patch_sha256"]
    )
    checks["installed_patched_hashes"] = all(
        digest(root / relative) == record["overlay_sha256"]
        for relative, record in manifest["patched_files"].items()
    )
    checks["installed_added_hashes"] = all(
        digest(root / relative) == expected
        for relative, expected in manifest["added_files"].items()
    )

    py = (root / "modules/block_sparse_mlp.py").read_text()
    cu = (root / "exllamav3_ext/quant/exl3_moe_prefill_e3.cu").read_text()
    cpp = (root / "exllamav3_ext/quant/exl3_moe_prefill_e3.cpp").read_text()
    checks["literal_opt_in"] = 'os.environ.get("EXL3_MOE_PREFILL_E3", "0") == "1"' in py
    checks["decode_excluded"] = (
        "MOE_PREFILL_E3_MIN_ROWS = max(512" in py
        and "num_tokens >= MOE_PREFILL_E3_MIN_ROWS" in py
    )
    checks["independent_projection_k"] = all(token in py for token in (
        "self.multi_gate.K, self.multi_up.K, self.multi_down.K",
        "all(k in (2, 3, 4)",
    ))
    checks["k_specializations"] = all(token in cu for token in (
        "dq_dispatch<BITS0, E3_CB>",
        "dq_dispatch<BITS1, E3_CB>",
        "e3_launch_gateup<GATE_BITS, 2>",
        "e3_launch_gateup<GATE_BITS, 3>",
        "e3_launch_gateup<GATE_BITS, 4>",
        "e3_launch_down<2>",
        "e3_launch_down<3>",
        "e3_launch_down<4>",
    ))
    checks["mul1_codebook"] = "constexpr int E3_CB = 2" in cu
    checks["tile_64x128x32"] = all(token in cu for token in (
        "constexpr int E3_TILE_N = 128",
        "constexpr int E3_TILE_K = 32",
        "constexpr int E3_MB = 4",
    ))
    checks["host_checked_dispatch"] = all(token in cpp for token in (
        "C10_CUDA_CHECK(", "check_bits(gate_bits", "check_bits(up_bits",
        "check_bits(down_bits",
    ))
    return checks


def main() -> None:
    checks = run_checks()
    print(json.dumps({"schema_version": 1, "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
