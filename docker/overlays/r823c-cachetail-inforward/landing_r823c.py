#!/usr/bin/env python3
"""CPU-only import landing; verify the combined trace, tail policy and chunk-capture manifest."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
ORIGINAL=Path('/opt/r823-cachetrace')
sys.path.insert(0,'/app')

def manifest(path):
    return dict((rel,h) for h,rel in (line.split('  ',1) for line in path.read_text().splitlines()))

trace=importlib.import_module('exllamav3.cache_trace')
gen=importlib.import_module('exllamav3.generator.generator')
job=importlib.import_module('exllamav3.generator.job')
pipe=importlib.import_module('exllamav3.generator.prefill_pipeline')
policy=importlib.import_module('exllamav3.cache.checkpoint_policy')
pt=importlib.import_module('exllamav3.generator.pagetable')
rc=importlib.import_module('exllamav3.cache.recurrent')
pm=importlib.import_module('exllamav3.cache.prefill_merge')
app=importlib.import_module('backends.exllamav3.model')
router=importlib.import_module('endpoints.OAI.router')
site=Path(gen.__file__).resolve().parents[1]
for m,rel in [(trace,'cache_trace.py'),(gen,'generator/generator.py'),(job,'generator/job.py'),
              (pipe,'generator/prefill_pipeline.py'),(policy,'cache/checkpoint_policy.py')]:
    assert Path(m.__file__).resolve()==site/rel,(m.__file__,rel)
assert all(m.r823_trace is trace for m in [job,pt,rc,pm,gen])
assert job.r823b_interval is pipe.r823b_interval is trace.r823b_interval is policy.interval
assert gen.r823b_read_tail is policy.read_tail
assert Path(app.__file__).resolve()==Path('/app/backends/exllamav3/model.py')
expected=manifest(ORIGINAL/'SHA256SUMS.exl3.src')
expected.update(manifest(Path('/opt/r823b-cachetail/SHA256SUMS.exl3.src')))
expected.update(manifest(ROOT/'SHA256SUMS.exl3.src'))
for rel in ('modules/gated_delta_net.py','modules/ple.py'):
    module=importlib.import_module('exllamav3.'+rel[:-3].replace('/', '.'))
    assert Path(module.__file__).resolve()==site/rel
assert pipe._pm is pm
assert hasattr(pm, 'try_boundary') and hasattr(pm, 'stash_boundary')
for root,entries in [(site,expected),(Path('/app'),manifest(ORIGINAL/'SHA256SUMS.app.src'))]:
    for rel,want in entries.items():
        got=hashlib.sha256((root/rel).read_bytes()).hexdigest()
        assert got==want,(str(root/rel),got,want)
print(json.dumps(dict(exl3=expected,policy_file=policy.__file__),sort_keys=True))
print('R823c landing: chunk capture + shared policy + cache trace imported and byte verified')
