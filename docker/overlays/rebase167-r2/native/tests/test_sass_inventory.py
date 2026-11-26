"""Exercise the sm_120 symbol gate and failure propagation without nvcc."""

from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent if (HERE.parent / "sass_inventory.py").exists() else HERE.parent / "tools"
INVENTORY = TOOLS / "sass_inventory.py"


def test_tune268_inventory_requires_specific_variants():
    kernels = ["exl3_gemm_v2_kernelILi6E", "exl3_mgemm_v2_kernelILi6E",
               "e3_gateup_kernelILi5ELi5E", "e3_down_kernelILi5E", "e3_down_det_kernelILi5E"]
    for stage in ("a", "b"):
        kernels.extend([f"exl3_moe_coop_v2_ns22exl3_moe_coop_{stage}_kernel_hILi2E",
                        f"exl3_moe_coop_v3_{stage}_kernel_hILi2E",
                        f"exl3_moe_coop_r2_{stage}_kernel_hILi2E"])
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "inventory.txt"
        for omit in [None, *kernels]:
            path.write_text("".join(f"_Z{n}params {'a' * 64}\n" for n in kernels if n != omit))
            proc = subprocess.run([sys.executable, str(INVENTORY), "require-tune268", str(path)],
                                  capture_output=True, text=True)
            if omit is None:
                assert proc.returncode == 0, proc.stderr
            else:
                assert proc.returncode != 0 and "missing required sm_120 kernels" in proc.stderr


def test_dumper_failure_survives_valid_partial_sass():
    commands = [sys.executable, str(INVENTORY)]
    # The parser succeeds on a complete function emitted before the dumper fails.
    pipeline = r'''(printf '\nFunction : kernel\n/*0000*/ NOP;\n'; exit 17) | "$1" "$2" hashes'''
    proc = subprocess.run(["bash", "-o", "pipefail", "-c", pipeline, "bash", *commands],
                          capture_output=True, text=True)
    assert proc.stdout.startswith("kernel ")
    assert proc.returncode == 17
