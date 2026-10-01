#!/usr/bin/env python3
"""Import-time landing, CPU only: assert imported paths and exact reviewed source bytes."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/app')

def verify(root,name):
    result={}
    for row in (ROOT/name).read_text().splitlines():
        expected,rel=row.split('  ',1)
        p=root/rel
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual==expected,(str(p),actual,expected)
        result[rel]=actual
    return result

trace=importlib.import_module('exllamav3.cache_trace')
generator=importlib.import_module('exllamav3.generator.generator')
job=importlib.import_module('exllamav3.generator.job')
pt=importlib.import_module('exllamav3.generator.pagetable')
rc=importlib.import_module('exllamav3.cache.recurrent')
pm=importlib.import_module('exllamav3.cache.prefill_merge')
app=importlib.import_module('backends.exllamav3.model')
router=importlib.import_module('endpoints.OAI.router')
site=Path(generator.__file__).resolve().parents[1]
assert Path(trace.__file__).resolve()==site/'cache_trace.py'
assert Path(app.__file__).resolve()==Path('/app/backends/exllamav3/model.py')
assert all(m.r823_trace is trace for m in [job,pt,rc,pm,generator])
assert 'EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP' in Path(generator.__file__).read_text()
print(json.dumps({'generator_file':generator.__file__,'trace_file':trace.__file__,
                  'exl3':verify(site,'SHA256SUMS.exl3.src'),
                  'app':verify(Path('/app'),'SHA256SUMS.app.src')},sort_keys=True))
print('R823 landing: density seam + cache trace imported and byte verified')
