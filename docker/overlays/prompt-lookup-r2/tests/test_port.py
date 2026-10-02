import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import unittest
import tempfile

def parameterized(names, values):
    def decorate(fn):
        fn.cases = values
        return fn
    return decorate

raises = unittest.TestCase().assertRaises

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from selftest import helper
import gate
m=helper()
G=P/'overlay/exllamav3/generator/generator.py'
J=P/'overlay/exllamav3/generator/job.py'

def method(path,name):
    tree=ast.parse(path.read_text())
    return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)

def execute(node,globs=None):
    # Compile real production statements with only external collaborators stubbed.
    node=ast.fix_missing_locations(node)
    scope={} if globs is None else dict(globs)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual overlay>','exec'),scope)
    return scope[node.name]

@parameterized('depth,chain',[(1,(4,)),(2,(4,5)),(3,(4,5,6))])
def case_match_opener_and_depth(depth,chain):
    match=m.find_previous_suffix([1,2,3,4,5,6,9,1,2,3],3,3)
    assert match.continuation==(4,5,6)
    assert m.open_lookup_chain(4,match,depth)==chain
    assert m.open_lookup_chain(7,match,depth) is None
    assert m.lookup_mask(chain)==[False]+[True]*(depth-1)

@parameterized('history',[[1,2], [1,1,1,1], [1,2,1,2,1,2], [1,2,9,4,5,6,1,2,3]])
def case_no_full_continuation(history):
    assert m.find_previous_suffix(history,3,3) is None

def case_mm_budget_stop_ids_and_invalid_source():
    history=[7,8,9,500,501,502,7,8,9]
    assert m.find_previous_suffix(history,3,3,excluded_spans=[(3,6)]) is None
    match=m.find_previous_suffix(history,3,3)
    assert match.continuation==(500,501,502) # EOS IDs are proposals; target sampler decides.
    match=m.find_previous_suffix(history,3,3,max_proposal_tokens=2)
    assert m.open_lookup_chain(500,match,3) is None
    assert m.open_lookup_chain(500,match,2)==(500,501)
    assert m.build_lookup_match(history,1,2,3,3) is None
    assert m.build_lookup_match(history,0,9,3,3) is None
    assert m.build_lookup_match(history,0,3,3,3,excluded_spans=[(8,9)]) is None
    bad=history.copy(); bad[2]=99
    assert m.build_lookup_match(bad,0,3,3,3) is None
    with raises(ValueError): m.open_lookup_chain(1,None,0)
    with raises(ValueError): m.find_previous_suffix([],0,3)
    with raises(ValueError): m.build_lookup_match([],0,0,3,3,max_proposal_tokens=-1)

@parameterized('batch,depth',[(1,3),(3,3),(4,3),(5,2),(8,2)])
def case_real_policy(batch,depth):
    fn=execute(method(G,'_get_draft_depth'))
    assert fn(NS(num_draft_tokens_by_batch=((4,3),(8,2)),num_draft_tokens=3),batch)==depth

@parameterized('generated',[0,1,2,3])
def case_no_token_zero_path(generated):
    node=method(J,'get_prompt_lookup_match')
    # Resolve relative pure-helper import; no torch or GPU extension loaded.
    node.body=[n for n in node.body if not isinstance(n,ast.ImportFrom)]
    fn=execute(node,{'build_lookup_match':m.build_lookup_match})
    class IDs(list):
        def __getitem__(self,k): return super().__getitem__(k)
    ids=IDs([[7,8,9,1,2,3,7,8,9]])
    sam=NS(accept_tensor=lambda _: (0,3))
    job=NS(prompt_lookup_sam=sam,sequences=[NS(sequence_ids=NS(torch=lambda:ids))],
           rq_new_tokens=0,new_tokens=generated,max_new_tokens=512,embeddings=[])
    result=fn(job)
    assert (result is not None)==(generated>=3)

@parameterized('i,credit',[(0,0),(1,1),(2,1)])
def case_actual_acceptance_credit_excludes_opener(i,credit):
    tree=method(G,'iterate_gen')
    block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test).startswith('mask is not None'))
    fn=ast.FunctionDef(name='credit',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='job'),ast.arg(arg='mask'),ast.arg(arg='i')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[block],decorator_list=[])
    job=NS(lookup_accepted=0)
    execute(fn)(job,m.lookup_mask((1,2,3)),i)
    assert job.lookup_accepted==credit
    execute(fn)(job,None,i)
    assert job.lookup_accepted==credit

@parameterized('frontier,width,i,expected',[(255,3,0,[0,255]),(255,3,1,[1,255]),(255,3,2,[2,255]),(254,3,0,[0,254]),(254,3,3,[1,256])])
def case_actual_reject_remainder_across_page_boundary(frontier,width,i,expected):
    tree=method(G,'iterate_gen')
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='reject_remainder')
    fn=execute(node,{'batch_logits':NS(shape=(1,width+1,10)),'PAGE_SIZE':256})
    # seq.kv_position is committed position advanced by receive_sample; verify-page
    # write frontiers already include the unresolved speculative suffix.
    rejected=width-i
    committed=frontier+i+1
    full=frontier+width+1
    pages=[NS(kv_position=min(256,max(0,full-p*256))) for p in range(3)]
    seq=NS(kv_position=committed,allocated_pages=pages)
    rewinds=[]; job=NS(rejected_draft_tokens=0,sequences=[seq])
    state=NS(rewind=rewinds.append)
    assert fn(job,0,i,[state])==rejected
    assert job.rejected_draft_tokens==rejected
    assert sum(p.kv_position for p in pages)==committed
    assert rewinds==([rejected] if rejected else [])
    assert seq.kv_position==committed


def case_requeue_counters_are_real_state_keys():
    node=method(J,'prepare_for_requeue')
    state=next(n.value for n in ast.walk(node) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='rq_state' for x in n.targets))
    expanded=[v for k,v in zip(state.keys,state.values) if k is None]
    assert any(ast.unparse(v)=='self.prompt_lookup_counters()' for v in expanded)
    assert 'prompt_lookup_sam' in [k.value for k in state.keys if isinstance(k,ast.Constant)]
    fn=method(J,'prompt_lookup_counters'); fn.body=[n for n in fn.body if not isinstance(n,ast.ImportFrom)]
    obj=NS(**dict(zip(m.COUNTERS,(9,8,3,6,4))))
    assert execute(fn,{'COUNTERS':m.COUNTERS})(obj)==vars(obj)


def case_target_logits_calls_and_sampler_unchanged():
    old=method(P/'fixtures/base-generator.py','iterate_gen'); new=method(G,'iterate_gen')
    def target_calls(n):
        return [ast.dump(c,include_attributes=False) for c in ast.walk(n) if isinstance(c,ast.Call)
                and ast.unparse(c.func) in ('self.model.forward','job.receive_logits','job.receive_sample','self.draft_model.prefill')]
    assert target_calls(old)==target_calls(new)
    for name in ('receive_logits','receive_sample','get_input_ids_list'):
        assert ast.dump(method(P/'fixtures/base-job.py',name))==ast.dump(method(J,name))
    # Constructor's mode computations are unchanged, including device/pruned embedding eligibility.
    old=method(P/'fixtures/base-generator.py','iterate_draftmodel_mtp_gen'); new=method(G,'iterate_draftmodel_mtp_gen')
    for name in ('dev_draft','window','step_seqlens','params'):
        def assignments(n):
            return [ast.dump(x) for x in ast.walk(n) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in x.targets)]
        assert assignments(old)==assignments(new), name


def case_real_sse_and_broken_formats():
    events=json.loads((P/'fixtures/real-sse.json').read_text())
    lines=['data: '+json.dumps(e) for e in events]+['data: [DONE]']
    value,usage,_,_,frames=gate.consume(lines)
    assert value['finish'] and usage['completion_tokens']==32 and frames<=32
    for bad in (lines[:-1], ['data: {"error":"bad"}'], ['data: {"choices": []}','data: [DONE]'], ['data: {bad}']):
        with raises((AssertionError,ValueError)): gate.consume(bad)


def case_real_log_and_r813_replay():
    rows=gate.completion_lines((P/'fixtures/parent-container.log').read_text())
    assert {51,54} <= set(rows) and rows[51]['gen_tokens']==64
    assert gate.counter_lines((P/'fixtures/candidate-counter.log').read_text())[51]['lookup_accepted']==10
    assert len(gate.read_rows(P/'fixtures/r813-records.jsonl'))>0
    with raises(AssertionError): gate.completion_lines('#1 completions: 64 tokens generated weird schema')
    with raises((AssertionError,KeyError)): gate.counter_lines('R827_COUNTER {"label":"#1 completions"}')


def case_launcher_candidate():
    with tempfile.TemporaryDirectory() as directory:
        launcher_case(Path(directory))

def launcher_case(tmp_path):
    gate.prepare(P/'fixtures/live-launcher.sh',tmp_path)
    a=(tmp_path/'launch-lookup-0.sh').read_text(); b=(tmp_path/'launch-lookup-1.sh').read_text()
    assert a.replace('EXL3_PROMPT_LOOKUP=0','EXL3_PROMPT_LOOKUP=1')==b
    assert len(gate.selectors(a))==47
    assert len(gate.agent_prompts())==12


# unittest cases are natively collected by pytest, and also run without installing dependencies.
class TestPort(unittest.TestCase):
    pass

for _name, _fn in list(globals().items()):
    if not _name.startswith('case_'):
        continue
    for _index, _values in enumerate(getattr(_fn, 'cases', [()])):
        if not isinstance(_values, tuple):
            _values=(_values,)
        def _test(self, fn=_fn, values=_values):
            fn(*values)
        setattr(TestPort, 'test_' + _name.removeprefix('case_') + '_' + str(_index), _test)

del _fn, _name, _test, _values, _index

if __name__ == '__main__':
    unittest.main()
