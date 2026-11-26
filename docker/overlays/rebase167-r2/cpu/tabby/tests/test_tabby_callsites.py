"""TabbyAPI -> exllamav3 call-site audit for rebase165 (CPU only; no GPU, no network).

Derived from ref/image-opt-rebase-dev-r3/tests/test_tabby_callsites.py (rebase-dev r3), updated for TabbyAPI 564f8fb
+ the served features and for the exllamav3 v1.6.5 integration. It parses the TabbyAPI tree (TABBY_DIR, default /app)
with `ast` and checks every use of exllamav3 against the tree under test (EXL3_TREE, the directory holding the
exllamav3 package; default: whatever `import exllamav3` finds):

  1. every `from exllamav3... import X` in the scanned TabbyAPI files resolves
  2. every keyword passed to an engine constructor/method is a parameter of the callee, or of the class its **kwargs
     forward to (AsyncGenerator -> Generator, AsyncJob -> Job, Cache -> CacheLayer_quant); a keyword that only a
     trailing **kwargs swallows is silently ignored at runtime and FAILS (ALLOW_ABSORBED=k1,k2 to accept one)
  3. every attribute TabbyAPI reads or calls on an engine object exists (class attribute, or `self.<name> =` in the
     class source / its bases): AsyncJob.set_filters/set_sampler/set_banned_strings/constrain_output_now (fc25e6a phase
     switching, reasoning budget, loop-think), Job.filters_suspended/time_first_prefill, Generator.max_draft_tokens
     (c598516 context check), Tokenizer.single_id/encode/..., Model.warmup, Config.infer_params fields
  4. hasattr() feature probes in TabbyAPI that name an engine feature find it (a miss silently turns the feature off)
  5. served contracts:
     - Generator.__init__ names num_draft_tokens_by_batch (TabbyAPI refuses to load a batch policy otherwise)
     - exllamav3.cache_trace has request_context/encoded/render/attach/finished with the argument shapes TabbyAPI uses
     - exllamav3.generator.prompt_lookup.COUNTERS exists and every key reaches the EOS result (BASE_COUNTERS on the
       job + the adaptive lookup's metrics, published by the generator); TabbyAPI reads them with .get, so a key the
       engine does not publish shows as null in the R828_COUNTER line
     - loop-think: LoopDetector(window, max_period) with feed_many()/_total; Job builds its detector from stop_on_loop
       and a requeued job keeps stop_on_loop
     - Cache accepts recurrent_dtype (TabbyAPI passes it only for recurrent_state_fp32)

Exit status 0 = PASS. Prints every checked site.
"""
import ast
import glob
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402


def _read(path):
    with open(path) as f:
        return f.read()


TREE, TABBY = _bootstrap.install()
exllamav3 = _bootstrap.check_tree(TREE)

from exllamav3 import AsyncGenerator, AsyncJob, Cache, Config, Generator, Model, Tokenizer  # noqa: E402
from exllamav3.generator.job import Job  # noqa: E402
import exllamav3.generator.sampler as sampler_mod  # noqa: E402
from exllamav3.cache import CacheLayer_quant  # noqa: E402

FAILS = []
CHECKED = []
ABSORBED = []
ALLOW_ABSORBED = {k for k in os.environ.get("ALLOW_ABSORBED", "").split(",") if k}


def fail(msg):
    FAILS.append(msg)


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


def check_call(where, label, fn, call, forward=None):
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
                ABSORBED.append((where, label, k))
                if k not in ALLOW_ABSORBED:
                    bad.append(f"{k} (only absorbed by **kwargs)")
                continue
            bad.append(f"{k} (not in {label} nor its forward target)")
        elif var_kw:
            ABSORBED.append((where, label, k))
            if k not in ALLOW_ABSORBED:
                bad.append(f"{k} (only absorbed by **kwargs)")
        else:
            bad.append(k)
    if npos_given > npos and not var_pos:
        bad.append(f"{npos_given} positional args > {npos}")
    CHECKED.append((where, label, kws))
    if bad:
        fail(f"{where}: {label}: {', '.join(bad)}")


def class_has(cls, name):
    """Class attribute, or an instance attribute assigned in the class (or a base) source."""
    if hasattr(cls, name):
        return True
    for base in inspect.getmro(cls):
        if base is object:
            continue
        try:
            src = inspect.getsource(base)
        except (OSError, TypeError):
            continue
        if f"self.{name} =" in src or f"self.{name}:" in src:
            return True
    return False


# ---- 1/2: imports and calls ---------------------------------------------------------------------------------------
SCAN = sorted(glob.glob(os.path.join(TABBY, "backends", "exllamav3", "*.py"))) + [
    os.path.join(TABBY, p) for p in (
        "endpoints/OAI/utils/chat_completion.py",
        "endpoints/OAI/utils/tool_choice.py",
        "endpoints/OAI/router.py",
        "common/sampling.py",
        "common/gen_logging.py",
        "common/tokenize_offloop.py",
    )
]
for path in SCAN:
    assert os.path.isfile(path), f"{path} missing (TABBY_DIR={TABBY})"

CALLEES = {
    "Config.from_directory": (Config.from_directory, Config.__init__),  # **kwargs -> the arch config (layer_map)
    "Model.from_config": (Model.from_config, None),
    "Tokenizer.from_config": (Tokenizer.from_config, None),
    "Cache": (Cache.__init__, CacheLayer_quant.__init__),
    "AsyncGenerator": (AsyncGenerator.__init__, Generator.__init__),
    "AsyncJob": (AsyncJob.__init__, Job.__init__),
    "CustomSampler": (sampler_mod.CustomSampler.__init__, None),
}
# self.<holder>.<method>(...) calls on engine objects TabbyAPI holds
HELD = {
    ("model", "load_gen"): Model.load_gen,
    ("draft_model", "load_gen"): Model.load_gen,
    ("model", "warmup"): Model.warmup,
    ("model", "unload"): Model.unload,
    ("draft_model", "unload"): Model.unload,
    ("tokenizer", "encode"): Tokenizer.encode,
    ("tokenizer", "decode"): Tokenizer.decode,
    ("tokenizer", "single_id"): Tokenizer.single_id,
}

module_aliases = {}  # alias -> module object (from exllamav3 import cache_trace as r823_trace)
for path in SCAN:
    tree = ast.parse(_read(path))
    base = os.path.relpath(path, TABBY)
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("exllamav3"):
            try:
                mod = __import__(node.module, fromlist=["_"])
            except Exception as ex:  # noqa: BLE001
                fail(f"{base}:{node.lineno}: import {node.module}: {ex!r}")
                continue
            for a in node.names:
                if not hasattr(mod, a.name):
                    fail(f"{base}:{node.lineno}: from {node.module} import {a.name}: missing")
                    continue
                obj = getattr(mod, a.name)
                imported[a.asname or a.name] = obj
                if inspect.ismodule(obj):
                    module_aliases[a.asname or a.name] = obj
                CHECKED.append((f"{base}:{node.lineno}", f"import {node.module}.{a.name}", []))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        where = f"{base}:{node.lineno}"
        if isinstance(f, ast.Name):
            if f.id in CALLEES:
                fn, fwd = CALLEES[f.id]
                check_call(where, f.id, fn, node, fwd)
            elif f.id.startswith("SS_") and hasattr(sampler_mod, f.id):
                check_call(where, f.id, getattr(sampler_mod, f.id).__init__, node)
            elif f.id in imported and inspect.isclass(imported[f.id]):
                check_call(where, f.id, imported[f.id].__init__, node)
        elif isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Name):
                key = f"{f.value.id}.{f.attr}"
                if key in CALLEES:
                    fn, fwd = CALLEES[key]
                    check_call(where, key, fn, node, fwd)
                elif f.value.id in module_aliases:
                    mod = module_aliases[f.value.id]
                    if not hasattr(mod, f.attr):
                        fail(f"{where}: {mod.__name__}.{f.attr} missing")
                    else:
                        check_call(where, f"{mod.__name__}.{f.attr}", getattr(mod, f.attr), node)
            elif isinstance(f.value, ast.Attribute) and isinstance(f.value.value, ast.Name) \
                    and f.value.value.id == "self" and (f.value.attr, f.attr) in HELD:
                check_call(where, f"{f.value.attr}.{f.attr}", HELD[(f.value.attr, f.attr)], node)

# ---- 3: attributes TabbyAPI uses on engine objects -----------------------------------------------------------------
from exllamav3.model.config import InferParams  # noqa: E402

REQUIRED_ATTRS = {
    AsyncJob: ["cancel", "cancelled", "constrain_output_now", "set_filters", "set_sampler", "set_banned_strings",
               "queue", "job"],
    Job: ["filters_suspended", "time_first_prefill", "cached_pages", "cached_tokens", "sequences",
          "constrain_output_now", "set_filters", "set_sampler"],
    AsyncGenerator: ["close", "generator"],
    Generator: ["max_batch_size", "max_draft_tokens", "recurrent_checkpoint_interval",
                "recurrent_checkpoint_interval_pp", "recurrent_cache", "get_cache_stats"],
    Cache: ["max_num_tokens"],
    Config: ["eos_token_id_list", "infer_params", "model_classes"],
    InferParams: ["moe_cpu_offload", "moe_cpu_split", "moe_cpu_threads", "moe_cpu_pinned_arena",
                  "ngram_stream_from_disk", "embed_stream_from_disk", "vision_pinned"],
    Model: ["caps", "load_gen", "unload", "warmup"],
    Tokenizer: ["bos_token", "bos_token_id", "decode", "encode", "eos_token", "eos_token_id",
                "get_id_to_piece_list", "pad_token", "single_id", "tokenizer", "unk_token"],
}
for cls, names in REQUIRED_ATTRS.items():
    for name in names:
        if class_has(cls, name):
            CHECKED.append((cls.__name__, f"attr {name}", []))
        else:
            fail(f"{cls.__name__}.{name} missing (TabbyAPI uses it)")

# AsyncJob's phase-switch / injection methods must forward to the Job
for name in ("constrain_output_now", "set_filters", "set_sampler", "set_banned_strings"):
    if hasattr(AsyncJob, name):
        src = inspect.getsource(getattr(AsyncJob, name))
        if "self.job." not in src:
            fail(f"AsyncJob.{name} does not forward to self.job")

# ---- 4: hasattr feature probes ---------------------------------------------------------------------------------------
ENGINE_PROBES = {"error": AsyncGenerator, "flush_cache_to_disk": Generator}
OPTIONAL_PROBES = {"flush_cache_to_disk"}  # the daily runs without the disk tier
seen = set()
for path in SCAN:
    tree = ast.parse(_read(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hasattr" \
                and len(node.args) == 2 and isinstance(node.args[1], ast.Constant) \
                and node.args[1].value in ENGINE_PROBES:
            name = node.args[1].value
            seen.add(name)
            where = f"{os.path.relpath(path, TABBY)}:{node.lineno}"
            if class_has(ENGINE_PROBES[name], name):
                CHECKED.append((where, f"hasattr probe {name} -> {ENGINE_PROBES[name].__name__}", []))
            elif name in OPTIONAL_PROBES:
                CHECKED.append((where, f"hasattr probe {name}: absent (optional feature off)", []))
            else:
                fail(f"{where}: hasattr probe '{name}' finds nothing on {ENGINE_PROBES[name].__name__}")
if not seen >= set(ENGINE_PROBES):
    fail(f"expected hasattr probes {sorted(ENGINE_PROBES)}, saw {sorted(seen)} (TabbyAPI changed: review this audit)")

# ---- 5: served contracts ---------------------------------------------------------------------------------------------
try:
    assert "num_draft_tokens_by_batch" in inspect.signature(Generator.__init__).parameters, \
        "Generator.__init__ has no num_draft_tokens_by_batch (TabbyAPI raises 'Batch-indexed drafting requires the " \
        "matching ExLlamaV3 patch' with the daily's draft_num_tokens_by_batch)"
    CHECKED.append(("generator.py", "Generator(num_draft_tokens_by_batch)", []))
    # passed via **draft_head_kwargs only when draft_model.sparse_draft_head > 0 (opt-in arm, NATIVE T1)
    assert "sparse_draft_head" in inspect.signature(Generator.__init__).parameters, \
        "Generator.__init__ has no sparse_draft_head (TabbyAPI's opt-in sparse draft head arm)"
    CHECKED.append(("generator.py", "Generator(sparse_draft_head)", []))
except AssertionError as ex:
    fail(str(ex))

try:
    from exllamav3 import cache_trace as ct
    want = {
        "request_context": ["request", "data"],
        "encoded": ["ids", "path"],
        "render": ["request_id", "ids", "tokenizer", "params", "sampler"],
        "attach": ["job", "record"],
        "finished": ["job", "result", "finish"],
    }
    for fn, args in want.items():
        assert hasattr(ct, fn), f"cache_trace.{fn} missing"
        got = list(inspect.signature(getattr(ct, fn)).parameters)
        assert got[:len(args)] == args, f"cache_trace.{fn}{tuple(got)} != {tuple(args)}"
    CHECKED.append(("cache_trace.py", "R823 trace surface", []))
except Exception as ex:  # noqa: BLE001
    fail(f"cache_trace contract: {ex!r}")

try:
    from exllamav3.generator import prompt_lookup as pl
    keys = set(pl.COUNTERS)
    base = set(getattr(pl, "BASE_COUNTERS", ()))
    adaptive = set(pl.AdaptiveLookup().metrics)
    assert keys <= base | adaptive, f"COUNTERS not covered by BASE_COUNTERS + AdaptiveLookup.metrics: " \
                                    f"{sorted(keys - base - adaptive)}"
    assert callable(getattr(Job, "prompt_lookup_counters", None)), "Job.prompt_lookup_counters missing"
    jsrc = inspect.getsource(Job.prompt_lookup_counters)
    assert "BASE_COUNTERS" in jsrc and ("prompt_lookup_adaptive.metrics" in jsrc or "adaptive.metrics" in jsrc), jsrc
    if "if adaptive is not None else {}" in jsrc.replace("\n", " "):
        print("  NOTE Job.prompt_lookup_counters omits the adaptive counters when the job has no adaptive state: "
              "those keys are null in R828_COUNTER (TabbyAPI reads them with .get)")
    gsrc = inspect.getsource(Generator)
    assert "prompt_lookup_counters()" in gsrc, "the generator no longer publishes the counters into EOS results"
    CHECKED.append(("prompt_lookup.py/job.py/generator.py", f"R828 COUNTERS ({len(keys)} keys) reach EOS", []))
except Exception as ex:  # noqa: BLE001
    fail(f"R828 COUNTERS contract: {ex!r}")

try:
    from exllamav3.generator.window_loop_detect import WindowLoopDetector as LoopDetector
    names, _, _, npos = params_of(LoopDetector.__init__)
    assert npos >= 2 and {"window_size", "max_period"} <= names, (names, npos)
    d = LoopDetector(8, 4)
    assert hasattr(d, "_total") and d._total == 0
    assert not d.feed_many([1, 2, 3])
    assert d.feed_many([1, 2, 3, 1, 2, 3, 1, 2, 3]), "period-3 loop not detected once the 8-token window is full"
    assert d._total == 12, d._total
    long_d = LoopDetector(3 * 10, 10)
    assert long_d.max_period == 10, long_d.max_period
    jn, _, _, _ = params_of(Job.__init__)
    assert {"max_rq_tokens", "stop_on_loop", "filters", "banned_strings", "embeddings"} <= jn, jn
    jsrc = inspect.getsource(Job)
    assert "WindowLoopDetector(window, window // min_reps)" in jsrc
    assert "stop_on_loop = self.stop_on_loop" in inspect.getsource(Job.prepare_for_requeue).replace(
        "stop_on_loop=self.stop_on_loop", "stop_on_loop = self.stop_on_loop"), \
        "requeued job no longer inherits stop_on_loop"
    CHECKED.append(("window_loop_detect.py/job.py", "loop-think contract", []))
except Exception as ex:  # noqa: BLE001
    fail(f"loop-think contract: {ex!r}")

try:
    cn, _, _, _ = params_of(Cache.__init__)
    assert "recurrent_dtype" in cn, cn
    CHECKED.append(("cache.py", "Cache(recurrent_dtype)", []))
except AssertionError as ex:
    fail(f"Cache.__init__ has no recurrent_dtype: {ex}")

# ---- report ---------------------------------------------------------------------------------------------------------
print(f"exllamav3: {exllamav3.__file__} (version {getattr(__import__('exllamav3.version').version, '__version__', '?')})")
print(f"tabby:     {os.path.abspath(TABBY)}")
print(f"checked {len(CHECKED)} sites over {len(SCAN)} TabbyAPI files")
for where, label, kws in CHECKED:
    print(f"  {where} {label}" + (f"({', '.join(kws)})" if kws else ""))
for where, label, k in ABSORBED:
    print(f"  ABSORBED {where} {label}: {k}" + (" (allowed)" if k in ALLOW_ABSORBED else ""))
if FAILS:
    for f in FAILS:
        print("FAIL", f)
    print(f"== tabby call sites: {len(FAILS)} FAILURES ==")
    sys.exit(1)
print("== tabby call sites: PASS ==")
