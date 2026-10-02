#!/usr/bin/env python3
"""CPU install self-test: execute actual overlay helpers and strict parser replays, no torch import."""
from pathlib import Path
import importlib.util
import json
import sys

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))

def helper():
    spec=importlib.util.spec_from_file_location('r828_lookup',P/'overlay/exllamav3/generator/prompt_lookup.py')
    m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)
    return m

def main():
    import gate
    m=helper()
    match=m.find_previous_suffix([7,8,9,1,2,3,7,8,9],3,3)
    assert m.open_lookup_chain(1,match,3)==(1,2,3)
    assert m.lookup_mask((1,2,3))==[False,True,True]
    assert m.open_lookup_chain(9,match,3) is None
    events=json.loads((P/'fixtures/real-sse.json').read_text())
    value,usage,_,_,_=gate.consume(['data: '+json.dumps(e)+'\n\n' for e in events]+['data: [DONE]\n\n'])
    assert usage['completion_tokens']>0 and value['finish']
    gate.completion_lines((P/'fixtures/parent-container.log').read_text())
    gate.counter_lines((P/'fixtures/candidate-counter.log').read_text())
    gate.read_rows(P/'fixtures/r813-records.jsonl')
    assert len(gate.agent_prompts())==12
    # Fail closed on a broken schema, not silent []/None defaults.
    try: gate.consume(['data: {"choices":[]}','data: [DONE]'])
    except AssertionError: pass
    else: raise AssertionError('malformed stream accepted')
    import unittest
    suite=unittest.defaultTestLoader.discover(str(P/'tests'))
    assert unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful(), 'CPU port regression'
    import replay_gate
    replay_gate.main()
    print('R828 CPU SELFTEST PASS (helpers + real SSE/R813/log replay)')

if __name__=='__main__': main()
