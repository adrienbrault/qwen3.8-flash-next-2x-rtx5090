"""TabbyAPI -> exllamav3 call-site audit (CPU only, rebase-dev r3; r2's audit + the loop-think r4 surface).

Parses the served TabbyAPI backend (tests/tabby-backends/*.py, copied from the served image
tabbyapi:stack-r3-rows32 /app/backends/exllamav3) with `ast` and checks every call into
exllamav3 against the signatures of the tree under test:

  * every name tabby imports from exllamav3 exists
  * every keyword tabby passes is a parameter of the callee (or the callee takes **kwargs and
    forwards them to a class whose signature has it: AsyncGenerator -> Generator,
    AsyncJob -> Job)
  * positional arity fits

rebase-dev r3 adds (the base image is tabbyapi:stack-r3-rows32-tokcount-loopthink4, whose loop-think r4 lives OUTSIDE
backends/exllamav3, in endpoints/OAI/utils/chat_completion.py and common/sampling.py):

  * TABBY_EXTRA (colon-separated files or dirs; default tests/tabby-extra, copied from patches/tabbyapi/loop-think-r4/src;
    at build time /app/endpoints/OAI/utils/chat_completion.py:/app/common/sampling.py): every `from exllamav3... import`
    resolves, and every call of an imported exllamav3 class fits its __init__ (LoopDetector(window, max_period))
  * every hasattr(<obj>, "<name>") feature probe in the scanned TabbyAPI files that names an engine feature
    (constrain_output_now, generator.error) finds it on the engine class, since a missing one silently disables the
    feature instead of failing (reasoning budget / loop-think injection)
  * the loop-think contract: LoopDetector(W, P) exposes feed_many() and _total and detects a period-P loop once its W
    window fills; Job takes max_rq_tokens and stop_on_loop, builds LoopDetector(W, W // min_reps) from stop_on_loop, and
    passes stop_on_loop on to the requeued job (the engine detector restarts at every requeue, which r4's 2W guard
    relies on); AsyncJob.constrain_output_now forwards to Job.constrain_output_now

Tree under test: the package next to tests/ (the r3 port) by default, or EXL3_TREE=<dir holding
exllamav3/> (e.g. the dev-vanilla tree). The extension and triton are stubbed by _runner.
"""
import ast
import glob
import inspect
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.environ.get("EXL3_TREE") or os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, TREE)
import _runner  # noqa: E402,F401  (stubs exllamav3.ext and triton)

import exllamav3  # noqa: E402
assert os.path.dirname(os.path.dirname(os.path.abspath(exllamav3.__file__))) == os.path.abspath(TREE), \
    f"imported {exllamav3.__file__}, expected the tree under {TREE}"

from exllamav3 import AsyncGenerator, AsyncJob, Cache, Config, Generator, Model, Tokenizer  # noqa: E402
from exllamav3.generator.job import Job  # noqa: E402
import exllamav3.generator.sampler as sampler_mod  # noqa: E402
from exllamav3.cache.quant import CacheLayer_quant  # noqa: E402  (Cache forwards **kwargs to the layer type)

FAILS = []
CHECKED = []
# A keyword that no signature names and that only a trailing **kwargs swallows is silently ignored at runtime (or, when
# TabbyAPI probes the signature first, refused at load: num_draft_tokens_by_batch on upstream dev raises "Batch-indexed
# drafting requires the matching ExLlamaV3 patch"). Such sites fail unless ALLOW_ABSORBED (comma-separated keyword names)
# lists them as a known, handled gap.
ALLOW_ABSORBED = {k for k in os.environ.get("ALLOW_ABSORBED", "").split(",") if k}
ABSORBED = []


def absorbed(where, label, k):
    ABSORBED.append((where, label, k))
    if k not in ALLOW_ABSORBED:
        FAILS.append(f"{where}: {label}: {k} is only absorbed by **kwargs (silently ignored or refused at load)")


def params_of(fn):
    sig = inspect.signature(fn)
    ps = list(sig.parameters.values())
    if ps and ps[0].name in ("self", "cls"):
        ps = ps[1:]
    names = {p.name for p in ps if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    var_kw = any(p.kind == p.VAR_KEYWORD for p in ps)
    var_pos = any(p.kind == p.VAR_POSITIONAL for p in ps)
    npos = len([p for p in ps if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    return names, var_kw, var_pos, npos


def check_call(where, label, fn, call, forward = None):
    names, var_kw, var_pos, npos = params_of(fn)
    kws = [k.arg for k in call.keywords if k.arg is not None]
    npos_given = len([a for a in call.args if not isinstance(a, ast.Starred)])
    bad = []
    for k in kws:
        if k in names:
            continue
        if var_kw and forward is not None:
            fnames, fvar_kw, _, _ = params_of(forward)
            if k in fnames:
                continue
            if fvar_kw:
                absorbed(where, label, k)
                continue
            bad.append(f"{k} (not in {label} nor its forward target)")
        elif var_kw:
            absorbed(where, label, k)
        else:
            bad.append(k)
    if npos_given > npos and not var_pos:
        bad.append(f"{npos_given} positional args > {npos}")
    CHECKED.append((where, label, kws))
    if bad:
        FAILS.append(f"{where}: {label}: {', '.join(bad)}")


CALLEES = {
    "Config.from_directory": (Config.from_directory, None),
    "Model.from_config": (Model.from_config, None),
    "Tokenizer.from_config": (Tokenizer.from_config, None),
    "Cache": (Cache.__init__, CacheLayer_quant.__init__),
    "AsyncGenerator": (AsyncGenerator.__init__, Generator.__init__),
    "AsyncJob": (AsyncJob.__init__, Job.__init__),
    "load_gen": (Model.load_gen, None),
    "CustomSampler": (sampler_mod.CustomSampler.__init__, None),
}


def callee_key(call):
    f = call.func
    if isinstance(f, ast.Name):
        if f.id in CALLEES:
            return f.id
        if f.id.startswith("SS_") and hasattr(sampler_mod, f.id):
            return f.id
        return None
    if isinstance(f, ast.Attribute):
        if f.attr == "load_gen" and isinstance(f.value, ast.Attribute) and \
                f.value.attr in ("model", "draft_model", "vision_model"):
            return "load_gen"
        if isinstance(f.value, ast.Name):
            k = f"{f.value.id}.{f.attr}"
            if k in CALLEES:
                return k
    return None


BACKENDS = os.environ.get("TABBY_BACKENDS") or os.path.join(HERE, "tabby-backends")
files = sorted(glob.glob(os.path.join(BACKENDS, "*.py")))
assert files, f"{BACKENDS}/*.py missing"
for path in files:
    tree = ast.parse(open(path).read())
    base = os.path.basename(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("exllamav3"):
            mod = __import__(node.module, fromlist = ["_"])
            for a in node.names:
                if not hasattr(mod, a.name):
                    FAILS.append(f"{base}:{node.lineno}: from {node.module} import {a.name}: missing")
                else:
                    CHECKED.append((f"{base}:{node.lineno}", f"import {node.module}.{a.name}", []))
        if isinstance(node, ast.Call):
            key = callee_key(node)
            if key is None:
                continue
            if key.startswith("SS_"):
                check_call(f"{base}:{node.lineno}", key, getattr(sampler_mod, key).__init__, node)
            else:
                fn, fwd = CALLEES[key]
                check_call(f"{base}:{node.lineno}", key, fn, node, fwd)


# ---- rebase-dev r3: TabbyAPI files outside backends/exllamav3 (loop-think r4) ----
def _py_files(spec):
    out = []
    for part in [p for p in spec.split(":") if p]:
        if os.path.isdir(part):
            out += sorted(glob.glob(os.path.join(part, "**", "*.py"), recursive = True))
        else:
            assert os.path.isfile(part), f"TABBY_EXTRA entry {part} missing"
            out.append(part)
    return out


EXTRA = os.environ.get("TABBY_EXTRA") or os.path.join(HERE, "tabby-extra")
extra_files = _py_files(EXTRA)
assert extra_files, f"TABBY_EXTRA {EXTRA}: no .py files"
PROBE_FILES = files + extra_files
for path in extra_files:
    tree = ast.parse(open(path).read())
    base = os.path.relpath(path, EXTRA) if os.path.isdir(EXTRA) else os.path.basename(path)
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("exllamav3"):
            mod = __import__(node.module, fromlist = ["_"])
            for a in node.names:
                if not hasattr(mod, a.name):
                    FAILS.append(f"{base}:{node.lineno}: from {node.module} import {a.name}: missing")
                else:
                    imported[a.asname or a.name] = getattr(mod, a.name)
                    CHECKED.append((f"{base}:{node.lineno}", f"import {node.module}.{a.name}", []))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in imported \
                and inspect.isclass(imported[node.func.id]):
            check_call(f"{base}:{node.lineno}", node.func.id, imported[node.func.id].__init__, node)

# hasattr feature probes on engine objects: a missing attribute silently disables the feature
from exllamav3.generator.async_generator import AsyncGenerator as _AG  # noqa: E402
# model.py probes the objects it holds: `job` is an AsyncJob, `self.generator` an AsyncGenerator
ENGINE_PROBES = {"constrain_output_now": AsyncJob, "error": _AG}


def _engine_has(name):
    cls = ENGINE_PROBES[name]
    if hasattr(cls, name) or f"self.{name} =" in inspect.getsource(cls):
        return cls.__name__
    return None


seen_probes = set()
for path in PROBE_FILES:
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hasattr" \
                and len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value in ENGINE_PROBES:
            name = node.args[1].value
            seen_probes.add(name)
            owner = _engine_has(name)
            where = f"{os.path.basename(path)}:{node.lineno}"
            if owner is None:
                FAILS.append(f"{where}: hasattr probe '{name}' finds no engine class carrying it (feature silently off)")
            else:
                CHECKED.append((where, f"hasattr probe {name} -> {owner}", []))
if not seen_probes >= set(ENGINE_PROBES):
    FAILS.append(f"expected hasattr probes {sorted(ENGINE_PROBES)} in the TabbyAPI files, saw {sorted(seen_probes)} "
                 f"(TabbyAPI changed: review this audit)")

# loop-think contract (behaviour, not only names)
from exllamav3.generator.loop_detect import LoopDetector  # noqa: E402
try:
    names, _, _, npos = params_of(LoopDetector.__init__)
    assert npos >= 2 and {"window_size", "max_period"} <= names, (names, npos)
    d = LoopDetector(8, 4)
    assert hasattr(d, "_total") and d._total == 0
    assert not d.feed_many([1, 2, 3])
    assert d.feed_many([1, 2, 3, 1, 2, 3, 1, 2, 3]), "period-3 loop not detected once the 8-token window is full"
    assert d._total == 12, d._total
    long_d = LoopDetector(3 * 10, 10)          # r4's long detector shape: (3L, L)
    assert long_d.max_period == 10, long_d.max_period
    jsrc = inspect.getsource(Job)
    jn, _, _, _ = params_of(Job.__init__)
    assert {"max_rq_tokens", "stop_on_loop"} <= jn, jn
    assert "self.loop_detector = LoopDetector(window_size, window_size // min_reps)" in jsrc
    assert "stop_on_loop = self.stop_on_loop," in inspect.getsource(Job.prepare_for_requeue), \
        "requeued job no longer inherits stop_on_loop"
    assert "self.job.constrain_output_now(output)" in inspect.getsource(AsyncJob.constrain_output_now)
    assert callable(getattr(Job, "constrain_output_now", None))
    CHECKED.append(("loop_detect.py/job.py", "loop-think contract", []))
except Exception as ex:  # noqa: BLE001  (a missing attribute is a contract failure too)
    FAILS.append(f"loop-think contract: {ex!r}")

labels = sorted({c[1] for c in CHECKED})
print(f"tree: {os.path.abspath(TREE)}")
print(f"tabby backend: {os.path.abspath(BACKENDS)}")
print(f"tabby extra: {os.path.abspath(EXTRA)} ({len(extra_files)} files)")
print(f"checked {len(CHECKED)} import/call/probe sites over {len(files) + len(extra_files)} tabby files ({len(labels)} distinct callees)")
for where, label, kws in CHECKED:
    if kws:
        print(f"  {where} {label}({', '.join(kws)})")
for where, label, k in ABSORBED:
    print(f"  ABSORBED {where} {label}: {k}{' (allowed: ALLOW_ABSORBED)' if k in ALLOW_ABSORBED else ''}")
if FAILS:
    for f in FAILS:
        print("FAIL", f)
    print(f"== tabby call sites: {len(FAILS)} FAILURES ==")
    sys.exit(1)
print("PASS every tabby import resolves and every keyword is accepted")
print("== tabby call sites: PASS ==")
