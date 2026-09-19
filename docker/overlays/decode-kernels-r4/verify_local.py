#!/usr/bin/env python3
"""CPU-only structural validation for the round-4 overlay."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/decode-kernels-r4"
SERVED = ROOT / "ref/served-src/exllamav3"
OVERLAY = OUT / "overlay/exllamav3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((OUT / "overlay/manifest.json").read_text())
    files = manifest["files"]
    checks = {}
    checks["patch_hash"] = sha256(OUT / "served-source.patch") == manifest["patch_sha256"]
    checks["baseline_hashes"] = all(
        sha256(SERVED / rel) == record["baseline_sha256"] for rel, record in files.items()
    )
    checks["overlay_hashes"] = all(
        sha256(OVERLAY / rel) == record["overlay_sha256"] for rel, record in files.items()
    )

    dry = subprocess.run(
        ["patch", "--dry-run", "--fuzz=0", "-p1", "-d", str(SERVED)],
        stdin=(OUT / "served-source.patch").open("rb"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    checks["patch_dry_run"] = dry.returncode == 0
    with tempfile.TemporaryDirectory(prefix="decode-kernels-r4-") as td:
        staged = Path(td)
        for rel in files:
            (staged / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SERVED / rel, staged / rel)
        applied = subprocess.run(
            ["patch", "--fuzz=0", "-p1", "-d", str(staged)],
            stdin=(OUT / "served-source.patch").open("rb"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        checks["patch_applied_hashes"] = applied.returncode == 0 and all(
            sha256(staged / rel) == record["overlay_sha256"]
            for rel, record in files.items()
        )

    unit = subprocess.run(
        [
            sys.executable, "-m", "unittest", "discover", "-s", str(OUT / "tests"),
            "-p", "test_round4_cpu.py", "-v",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    checks["cpu_unit_tests"] = unit.returncode == 0

    generator = (OVERLAY / "generator/generator.py").read_text()
    overlap = (OVERLAY / "generator/draft_overlap.py").read_text()
    sampler = (OVERLAY / "generator/sampler/custom.py").read_text()
    embedding = (OVERLAY / "modules/embedding.py").read_text()
    mtp = (OVERLAY / "architecture/qwen4_exp_mtp.py").read_text()
    mixer = (OVERLAY / "modules/hyperconnections.py").read_text()
    cuda = (OVERLAY / "exllamav3_ext/hc_mix.cu").read_text()
    checks["flag_defaults_off"] = all(token in text for text, token in (
        (generator, 'EXL3_DRAFT_PINNED_STAGING", "0") == "1"'),
        (generator, 'EXL3_BATCH_VERIFY", "0") == "1"'),
        (generator, 'EXL3_MTP_DEVICE_DRAFT", "0") == "1"'),
        (embedding, 'EXL3_EMBED_GPU", "0") == "1"'),
        (mixer, 'EXL3_HC_MIX_V2_INT8", "0") == "1"'),
        (mtp, 'os.environ.get("EXL3_MTP_HEAD_N")'),
    ))
    checks["sampler_fallback_guards"] = all(token in generator + sampler + overlap for token in (
        "reqs_past_ids", "job.filters", "forced_ids", "return_probs", "return_top_tokens",
        "device_logit_mask", 'batch_verify_mode = "sampled"',
    ))
    checks["c10_cuda_check"] = (
        "#include <c10/cuda/CUDAException.h>" in cuda
        and cuda.count("C10_CUDA_CHECK(cudaPeekAtLastError())") >= 3
    )
    checks["cuda_binding_present"] = all(token in cuda for token in (
        "gr_v2_dots_i8_kernel", "gr_v2_up_i8_kernel", "void gr_mix_v2_int8",
    )) and "gr_mix_v2_int8" in (OVERLAY / "exllamav3_ext/bindings.cpp").read_text()
    checks["tabbyapi_overlay_absent"] = not (OUT / "overlay/tabbyapi").exists()
    checks["torch_available"] = importlib.util.find_spec("torch") is not None
    checks["cuda_toolkit_available"] = any(
        path.is_file() for path in (Path("/usr/local/cuda/bin/nvcc"), Path("/opt/cuda/bin/nvcc"))
    )

    result = {
        "checks": checks,
        "patch_dry_run_output": dry.stdout.decode(),
        "cpu_unit_test_output": unit.stdout.decode(),
        "note": (
            "Required local checks exclude torch/CUDA availability. The synthetic parity script "
            "and CUDA extension tests are supplied for the served image."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    required = {
        name: value for name, value in checks.items()
        if name not in {"torch_available", "cuda_toolkit_available"}
    }
    return 0 if all(required.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
