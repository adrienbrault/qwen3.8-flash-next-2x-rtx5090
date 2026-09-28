"""CPU-only runner for test_loop_think.py outside the image (no torch, no exllamav3 wheel, no GPU).

  python3 run_offline.py APP_DIR [TEST_FILE]

APP_DIR is a TabbyAPI tree with the patch applied (the image's /app). The collector imports
exllamav3.generator.loop_detect.LoopDetector; this runner serves base/loop_detect.py (a copy of exllamav3's, which
needs torch only for an isinstance check) from a temporary stub package. TabbyAPI's optional-dependency probe runs
first, before the stubs are on sys.path, so the rest of TabbyAPI still sees exllamav3 and torch as absent and does
not import the real backend. Needs TabbyAPI's CPU dependencies (pydantic, fastapi, loguru, ...) in the interpreter.
"""
import os
import runpy
import shutil
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
app = os.path.abspath(sys.argv[1])
test = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(here, "test_loop_think.py")

os.chdir(app)
sys.path.insert(0, app)
import common.optional_dependencies as od  # noqa: E402  (probe before the stubs exist)

print("optional dependencies seen by TabbyAPI:", od.dependencies)

stubs = tempfile.mkdtemp(prefix="loopthink-stubs-")
os.makedirs(os.path.join(stubs, "exllamav3", "generator"))
os.makedirs(os.path.join(stubs, "torch"))
for pkg in ("exllamav3", "exllamav3/generator"):
    open(os.path.join(stubs, pkg, "__init__.py"), "w").close()
shutil.copy(os.path.join(here, "base", "loop_detect.py"), os.path.join(stubs, "exllamav3", "generator", "loop_detect.py"))
with open(os.path.join(stubs, "torch", "__init__.py"), "w") as fh:
    fh.write("class Tensor:  # loop_detect.py only uses torch.Tensor in an isinstance check\n    pass\n")
sys.path.insert(1, stubs)

sys.argv = [test]
try:
    runpy.run_path(test, run_name="__main__")
finally:
    shutil.rmtree(stubs, ignore_errors=True)
