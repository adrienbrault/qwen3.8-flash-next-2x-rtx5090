#!/usr/bin/env python3
"""CPU-only integrity checks for the r2 overlay; this does not claim CUDA validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import py_compile
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OVERLAY = HERE / "overlay"
SERVED = ROOT / "ref/served-src/exllamav3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((OVERLAY / "manifest.json").read_text())
    checks = []
    with tempfile.TemporaryDirectory(prefix="decode-kernels-r2-") as tmp:
        checkout = Path(tmp) / "exllamav3"
        checkout.mkdir()
        for relative, hashes in manifest["files"].items():
            source = SERVED / relative
            payload = OVERLAY / "payload" / relative
            assert sha256(source) == hashes["baseline_sha256"], relative
            assert sha256(payload) == hashes["payload_sha256"], relative
            target = checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            checks.append(f"hash:{relative}")
        subprocess.run(
            ["git", "apply", str(HERE / "served-source.patch")],
            cwd=checkout,
            check=True,
        )
        for relative in manifest["files"]:
            assert (checkout / relative).read_bytes() == (OVERLAY / "payload" / relative).read_bytes()
            checks.append(f"patch:{relative}")

    cpp = (OVERLAY / "payload/exllamav3_ext/libtorch/blocksparse_mlp.cpp").read_text()
    launcher = (OVERLAY / "payload/exllamav3_ext/quant/exl3_moe_coop.cu").read_text()
    for needle in (
        'std::getenv("EXL3_SHARED_EXPERT_OVERLAP")',
        "cudaStreamCreateWithFlags(&shared_stream, cudaStreamNonBlocking)",
        "cudaEventRecord(shared_input_ready, stream)",
        "cudaEventRecord(shared_done, shared_stream)",
    ):
        assert needle in cpp, needle
        checks.append(f"source:{needle}")
    assert launcher.count("cudaStreamWaitEvent(stream, wait_before_b, 0)") == 2
    checks.append("source:wait-before-B-v1-v2")

    with tempfile.TemporaryDirectory(prefix="decode-kernels-r2-pyc-") as pyc_dir:
        for script in (
            HERE / "tests/test_shared_expert_overlap.py",
            OVERLAY / "install.py",
            OVERLAY / "build_extension.py",
        ):
            py_compile.compile(
                str(script),
                cfile=str(Path(pyc_dir) / (script.name + "c")),
                doraise=True,
            )
            checks.append(f"py_compile:{script.relative_to(HERE)}")

    print(json.dumps({"status": "pass", "scope": "CPU-only", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
