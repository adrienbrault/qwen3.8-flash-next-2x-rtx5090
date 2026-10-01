#!/usr/bin/env python3
"""CPU import landing: entire inherited stack plus R825c replacements."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '/app')
ROOT = Path(__file__).resolve().parent

def manifest(p):
    return {r: h for h, r in (s.split('  ', 1) for s in p.read_text().splitlines())}

expected = {}
for directory in ('/opt/r823-cachetrace', '/opt/r823b-cachetail',
                  '/opt/r823c-cachetail-inforward', '/opt/r825', '/opt/r825b', str(ROOT)):
    expected.update(manifest(Path(directory) / 'SHA256SUMS.exl3.src'))
pipe = importlib.import_module('exllamav3.generator.prefill_pipeline')
job = importlib.import_module('exllamav3.generator.job')
pm = importlib.import_module('exllamav3.cache.prefill_merge')
policy = importlib.import_module('exllamav3.cache.checkpoint_policy')
site = Path(pipe.__file__).resolve().parents[1]
for rel, want in expected.items():
    assert hashlib.sha256((site / rel).read_bytes()).hexdigest() == want, rel
for rel, want in manifest(Path('/opt/r823-cachetrace/SHA256SUMS.app.src')).items():
    assert hashlib.sha256((Path('/app') / rel).read_bytes()).hexdigest() == want, rel
for name in ('modules.gated_delta_net', 'modules.ple', 'cache.recurrent',
             'generator.generator', 'generator.async_generator', 'generator.pagetable',
             'cache.cache', 'cache_trace'):
    m = importlib.import_module('exllamav3.' + name)
    assert Path(m.__file__).resolve() == site / (name.replace('.', '/') + '.py')
assert pipe._pm is job._pm is pm
assert pipe.r823b_interval is job.r823b_interval is policy.interval
assert callable(pm.try_pipeline_capture)
assert callable(pipe.whole_window_ends)
assert callable(pipe.resume_window)
assert callable(pipe.close_window)
assert callable(job.Job.prepare_for_queue)
gen = importlib.import_module('exllamav3.generator.generator')
assert 'prepare_for_queue' in gen.Generator.enqueue.__code__.co_names
assert '_ls_deferred_prepare' not in gen.Generator.enqueue.__code__.co_names
trace = importlib.import_module('exllamav3.cache_trace')
assert '_r823_pending_page_digests' in trace.attach.__code__.co_names
assert os.environ.get('EXL3_PREFILL_RESUMABLE', '1') == '1'
assert os.environ.get('EXL3_PREFILL_WHOLE_PROMPT', '0') in ('0', '1')
print(json.dumps({'expected': expected, 'site': str(site)}, sort_keys=True))
print('R825c landing PASS: inherited stack and host preparation delta imported and byte verified')
