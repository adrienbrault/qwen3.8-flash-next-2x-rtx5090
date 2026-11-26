"""Minimal pytest stand-in for the offline image (pytest is not installed there and the build has no network).

Supports what out/native/tests use: plain test_* functions, @pytest.fixture (incl. autouse, yield fixtures),
@pytest.mark.parametrize (stacked), pytest.importorskip / skip / raises, and the monkeypatch / tmp_path fixtures.
Usage: python3 _minipytest.py test_a.py [test_b.py ...]   -> exit 0 when nothing failed.
REBASE167_NO_SKIP=1: exit non-zero when anything was skipped as well (the image build: torch and the integrated
package must import there, so a skip means a test that did not run).
When the real pytest is importable, run-cpu-tests.sh uses it instead.
"""
import contextlib
import importlib.util
import inspect
import itertools
import os
import shutil
import sys
import tempfile
import traceback
import types


class _Skip(Exception):
    pass


def skip(reason = ""):
    raise _Skip(reason)


def importorskip(name, *args, **kwargs):
    try:
        return importlib.import_module(name)
    except Exception as e:  # noqa: BLE001 - a module that fails to import is a skip, as with pytest
        raise _Skip(f"cannot import {name}: {e}")


@contextlib.contextmanager
def raises(exc, match = None):
    try:
        yield
    except exc as e:
        if match is not None:
            import re
            assert re.search(match, str(e)), f"{e!r} does not match {match!r}"
        return
    raise AssertionError(f"DID NOT RAISE {exc}")


def fixture(fn = None, *, autouse = False, **_):
    def deco(f):
        f._mini_fixture = True
        f._mini_autouse = autouse
        return f
    return deco(fn) if fn is not None else deco


class _Mark:
    @staticmethod
    def parametrize(names, values, **_):
        names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)

        def deco(f):
            f._mini_params = getattr(f, "_mini_params", []) + [(names, list(values))]
            return f
        return deco

    def __getattr__(self, _):
        return lambda *a, **k: (lambda f: f)


mark = _Mark()


class MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value = None, raising = True):
        if isinstance(target, str):
            mod, _, attr = target.rpartition(".")
            target, name, value = importlib.import_module(mod), attr, name
        old = getattr(target, name, _MISSING)
        if old is _MISSING and raising:
            raise AttributeError(name)
        self._undo.append(lambda: setattr(target, name, old) if old is not _MISSING else delattr(target, name))
        setattr(target, name, value)

    def delattr(self, target, name, raising = True):
        old = getattr(target, name, _MISSING)
        if old is _MISSING:
            if raising:
                raise AttributeError(name)
            return
        self._undo.append(lambda: setattr(target, name, old))
        delattr(target, name)

    def setenv(self, name, value):
        old = os.environ.get(name)
        self._undo.append(lambda: os.environ.__setitem__(name, old) if old is not None else os.environ.pop(name, None))
        os.environ[name] = str(value)

    def delenv(self, name, raising = True):
        old = os.environ.get(name)
        if old is None and raising:
            raise KeyError(name)
        self._undo.append(lambda: os.environ.__setitem__(name, old) if old is not None else None)
        os.environ.pop(name, None)

    def undo(self):
        while self._undo:
            self._undo.pop()()


_MISSING = object()


def _install_shim():
    shim = types.ModuleType("pytest")
    for n in ("skip", "importorskip", "raises", "fixture", "mark"):
        setattr(shim, n, globals()[n])
    shim.MonkeyPatch = MonkeyPatch
    sys.modules["pytest"] = shim


def _run_file(path):
    name = "minitest_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except _Skip as e:
        print(f"SKIP {os.path.basename(path)} (module): {e}")
        return 0, 0, 1
    except ModuleNotFoundError as e:
        if e.name in ("torch", "triton", "exllamav3_ext"):   # host without the image's stack
            print(f"SKIP {os.path.basename(path)} (module): {e}")
            return 0, 0, 1
        raise
    fixtures = {k: v for k, v in vars(mod).items() if callable(v) and getattr(v, "_mini_fixture", False)}
    autouse = [k for k, v in fixtures.items() if v._mini_autouse]
    passed = failed = skipped = 0
    for tname, fn in sorted(vars(mod).items()):
        if not tname.startswith("test_") or not callable(fn):
            continue
        grids = getattr(fn, "_mini_params", [])
        combos = [dict()]
        for names, values in grids:
            nxt = []
            for c in combos:
                for v in values:
                    v = v if len(names) > 1 else (v,)
                    nxt.append({**c, **dict(zip(names, v))})
            combos = nxt
        for c in combos:
            label = f"{os.path.basename(path)}::{tname}" + (f"[{c}]" if c else "")
            mp = MonkeyPatch()
            tmp = tempfile.mkdtemp()
            gens = []
            try:
                def resolve(argname):
                    if argname == "monkeypatch":
                        return mp
                    if argname == "tmp_path":
                        import pathlib
                        return pathlib.Path(tmp)
                    f = fixtures[argname]
                    kw = {a: resolve(a) for a in inspect.signature(f).parameters}
                    r = f(**kw)
                    if inspect.isgenerator(r):
                        gens.append(r)
                        return next(r)
                    return r
                for a in autouse:
                    resolve(a)
                kwargs = {a: (c[a] if a in c else resolve(a)) for a in inspect.signature(fn).parameters}
                fn(**kwargs)
                passed += 1
                print(f"PASS {label}")
            except _Skip as e:
                skipped += 1
                print(f"SKIP {label}: {e}")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL {label}")
                traceback.print_exc()
            finally:
                for g in reversed(gens):
                    with contextlib.suppress(StopIteration):
                        next(g)
                mp.undo()
                shutil.rmtree(tmp, ignore_errors = True)
    return passed, failed, skipped


if __name__ == "__main__":
    _install_shim()
    sys.dont_write_bytecode = True
    tot = [0, 0, 0]
    for p in sys.argv[1:]:
        r = _run_file(p)
        tot = [a + b for a, b in zip(tot, r)]
    print(f"== minipytest: {tot[0]} passed, {tot[1]} failed, {tot[2]} skipped ==")
    no_skip = os.environ.get("REBASE167_NO_SKIP") == "1"
    if no_skip and tot[2]:
        print("== REBASE167_NO_SKIP=1: skipped tests count as failures ==")
    sys.exit(1 if tot[1] or (no_skip and tot[2]) else 0)
