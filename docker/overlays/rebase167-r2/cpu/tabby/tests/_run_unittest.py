"""Run TabbyAPI's own test modules (tests.test_x ...) with the rebase165 bootstrap (EXL3_STUB_EXT, EXL3_TREE).

unittest TestCases run as usual; plain pytest-style `def test_*()` functions without arguments (fixtures) are wrapped
as FunctionTestCases, since pytest is not in the image.
"""
import importlib
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

TREE, TABBY = _bootstrap.install()
_bootstrap.check_tree(TREE)
os.chdir(TABBY)
suite = unittest.TestSuite()
for name in sys.argv[1:]:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    try:
        mod = importlib.import_module(name)
    except Exception:  # noqa: BLE001  (the loader above already reports the import error)
        continue
    for fname, fn in sorted(vars(mod).items()):
        if fname.startswith("test_") and inspect.isfunction(fn) and fn.__module__ == mod.__name__ \
                and not inspect.signature(fn).parameters:
            suite.addTest(unittest.FunctionTestCase(fn, description=f"{name}.{fname}"))
result = unittest.TextTestRunner(verbosity=1).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
