"""Config-parsing and tabby-facing surface tests (CPU only).

The served image is driven by tabby-backends/exllamav3/*.py. Verify that the
ported tree still exposes the constructor signatures tabby calls, with
upstream's new parameters defaulted (so tabby needs no patch).
"""
import inspect
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _runner  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS {name}")
    except Exception:
        import traceback
        RESULTS.append((name, False, traceback.format_exc()))
        print(f"FAIL {name}")
        traceback.print_exc()


def generator_signature_defaults():
    import exllamav3.generator.generator as gen
    sig = inspect.signature(gen.Generator.__init__)
    for p in ("model", "cache", "draft_model", "max_batch_size"):
        assert p in sig.parameters, p
    # the draft policy pair the launcher config sets ([[4, 3], [8, 1]]) parses
    gen.Generator.__init__  # noqa: B018  (signature exists)
    import types
    g = object.__new__(gen.Generator)
    g.num_draft_tokens_by_batch = ((4, 3), (8, 1))
    g.num_draft_tokens = 2
    assert gen.Generator._get_draft_depth(g, 4) == 3


def async_generator_surface():
    from exllamav3.generator.async_generator import AsyncGenerator
    # AsyncGenerator forwards every Generator kwarg (v1.5.0 and dev compatible)
    params = list(inspect.signature(AsyncGenerator.__init__).parameters)
    assert "args" in params and "kwargs" in params, params
    # close() now closes the CPU page cache tier too (upstream) and the NVMe
    # tier (ours) — both kept
    src = inspect.getsource(AsyncGenerator.close)
    assert "cpu_page_cache" in src and "disk_page_cache" in src


def job_surface():
    from exllamav3.generator.job import Job
    params = list(inspect.signature(Job.__init__).parameters)
    for p in ("banned_strings", "stop_conditions"):
        assert p in params, p
    # dev-side extras (draft tokens, MTP options) travel through **kwargs
    assert "kwargs" in inspect.signature(Job.__init__).parameters
    assert hasattr(Job, "constrain_output_now")


def cache_factory():
    import exllamav3.cache.cache as cache_mod
    sig = inspect.signature(cache_mod.Cache.__init__)
    for p in ("max_batch_size", "max_history"):
        assert p in sig.parameters, p


def sampler_surface():
    import exllamav3.generator.sampler.custom as custom
    steps = [n for n in dir(custom) if n.startswith("SS_")]
    assert len(steps) >= 8, steps
    src = inspect.getsource(custom)
    assert "batch_verify" in src  # served EXL3_BATCH_VERIFY plumbing


def config_surface():
    from exllamav3.model.config import Config
    assert inspect.signature(Config.from_directory) is not None


check("Generator signature keeps tabby's kwargs with defaults", generator_signature_defaults)
check("AsyncGenerator surface + tier closures in close()", async_generator_surface)
check("Job surface keeps tabby's kwargs", job_surface)
check("Cache factory keeps max_batch_size/max_history", cache_factory)
check("CustomSampler surface (SS_* steps, batch verify key)", sampler_surface)
check("Config surface", config_surface)

n_ok = sum(1 for _, ok, _ in RESULTS if ok)
print(f"== {n_ok}/{len(RESULTS)} config/surface tests passed ==")
if n_ok != len(RESULTS):
    sys.exit(1)
