#!/usr/bin/env python3
"""Real copied TabbyAPI module + real Job/Sequence/AsyncJob/trace on CPU.
Only model forward, page allocation, native extension and unrelated app deps are faked.
R825C_APP points at the copied /app tree; --tree=base replays the original failure.
"""
import ast
import asyncio
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import types
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent
PROVENANCE = json.loads((ROOT/'BASE-PROVENANCE.json').read_text())
APP = Path(os.environ.get('R825C_APP', PROVENANCE['app_tree']))
if not APP.is_dir(): APP = ROOT/'fixtures/app'
TREE = os.environ.get('R825C_TREE', 'src')

class Dummy:
    def __init__(self, *a, **k): pass
    def __getattr__(self, name): return lambda *a, **k: None


def stub(name, **exports):
    module = types.ModuleType(name)
    module.__path__ = []
    module.__dict__.update(exports)
    sys.modules[name] = module
    return module


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def methods(name, path, cls, names, globals):
    """GPU Generator methods only; Job/Sequence/AsyncJob/TabbyAPI import whole files."""
    tree = ast.parse(path.read_text())
    c = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls)
    c = copy.deepcopy(c)
    c.body = [x for x in c.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name in names]
    for fn in c.body: fn.decorator_list = []
    module = stub(name, **globals)
    node = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), c], type_ignores=[])
    exec(compile(ast.fix_missing_locations(node), str(path), 'exec'), module.__dict__)
    return module


def setup():
    stub('exllamav3')
    stub('exllamav3.generator')
    stub('exllamav3.util')
    stub('exllamav3.util.device_copy', to_device=lambda *a, **k: (_ for _ in ()).throw(AssertionError('device copy')))
    tensor = load('exllamav3.util.tensor', ROOT/'fixtures/util/tensor.py')
    stub('exllamav3.cache')
    stub('exllamav3.cache.checkpoint_policy', interval=lambda *a:256)
    trace = load('exllamav3.cache_trace', ROOT/os.environ.get('R825C_TRACE_TREE', TREE)/'cache_trace.py')
    sys.modules['exllamav3'].cache_trace = trace
    stub('exllamav3.constants', PAGE_SIZE=256)
    stub('exllamav3.cache', RecurrentCache=Dummy, prefill_merge=Dummy(), CacheLayer_quant=Dummy)
    stub('exllamav3.cache.cache', Cache=Dummy)
    stub('exllamav3.cache.checkpoint_policy', interval=lambda *a:256)
    def native_positions(pos, ids, merge, spans, grids):
        # CPU stand-in for the native host loop; these fixtures contain text IDs.
        # Real RoPE frequency math and the real Job caller run unchanged.
        assert pos.device.type == ids.device.type == 'cpu'
        pos.copy_(torch.arange(pos.shape[-1]).expand(3,-1))
        return pos.shape[-1]
    stub('exllamav3.ext', exllamav3_ext=types.SimpleNamespace(BC_SAM=Dummy,gen_mrope_pos_ids=native_positions))
    load('exllamav3.util.rope', ROOT/'fixtures/util/rope.py')
    stub('exllamav3.tokenizer', MMEmbedding=Dummy)
    stub('exllamav3.tokenizer.mm_embedding', FIRST_MM_EMBEDDING_INDEX=1000000)
    sys.modules['exllamav3.util'].profile_opt = Dummy()
    stub('exllamav3.generator.filter', Filter=Dummy)
    stub('exllamav3.generator.loop_detect', LoopDetector=Dummy)
    stub('exllamav3.generator.sampler', Sampler=Dummy, DefaultSampler=Dummy)
    stub('exllamav3.generator.draft_overlap', draw_sampling_seed=lambda *a:1)
    pagetable = load('exllamav3.generator.pagetable', ROOT/TREE/'generator/pagetable.py')
    job = load('exllamav3.generator.job', ROOT/TREE/'generator/job.py')
    generator = methods('exllamav3.generator.generator', ROOT/TREE/'generator/generator.py', 'Generator',
        {'enqueue','cancel','iterate_start_jobs','reap_failed_job','num_remaining_jobs'}, dict(torch=torch,Job=job.Job))
    asyncgen = load('exllamav3.generator.async_generator', ROOT/TREE/'generator/async_generator.py')
    exl = sys.modules['exllamav3']
    for n,v in dict(AsyncGenerator=asyncgen.AsyncGenerator,AsyncJob=asyncgen.AsyncJob,
                    Generator=generator.Generator, Cache=Dummy,Config=Dummy,Model=Dummy,Tokenizer=Dummy).items():
        setattr(exl,n,v)
    # Stub only imported dependencies; import the WHOLE unchanged model.py from the copied tree.
    source = APP/'backends/exllamav3/model.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == PROVENANCE['fixtures']['app/backends/exllamav3/model.py'], 'copied app model.py drift'
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(('backends.','common.','endpoints.')):
            stub(node.module, **{x.name:Dummy for x in node.names})
    stub('backends'); stub('backends.exllamav3')
    app = load('backends.exllamav3.model', source)
    assert Path(app.__file__) == source
    app.unwrap = lambda x,d=None: d if x is None else x
    app.xlogger = Dummy()
    for n in ('log_prompt','log_request_start','log_generation_params','log_metrics'):
        setattr(app,n,lambda *a, **k:None)
    app.format_settings = lambda *a:''
    app.validate_context_requirements = lambda *a:None
    app.ExLlamaV3Grammar = lambda:types.SimpleNamespace(filters=[],add_json_schema_filter=lambda *a,**k:None,
        add_regex_filter=lambda *a,**k:None,add_grammar_filter=lambda *a,**k:None)
    builder = types.SimpleNamespace(settings=[],build=lambda greedy:types.SimpleNamespace(steps=[]))
    app.ExllamaV3SamplerBuilder = types.SimpleNamespace(from_params=lambda *a:builder)
    app.status_display = types.SimpleNamespace(add_job=lambda *a:Dummy(), remove_job=lambda *a:None)
    return app, job, generator, asyncgen, pagetable, trace, tensor

APP_MOD, JOB, GEN, ASYNC, PT, TRACE, TENSOR = setup()

class Params:
    temperature=0; max_tokens=16; min_tokens=16; add_bos_token=False
    json_schema=None; regex_pattern=None; grammar_string=None; banned_strings=None
    token_healing=False; logprobs=1; top_logprobs=2; loop_detect_window=0
    def __init__(self): self.stop=[99]
    def model_dump(self, **k): return {}
    def param_source(self, n): return 'default'
    def get_stop_on_loop(self): return None

class Tokenizer:
    eos_token_id=99; bos_token_id=-1; bos_token=''
    def single_id(self, text): return -2
    def encode(self, text, **kw): return torch.tensor([[7,8]]) if text!='<tool_response>' else torch.empty((1,0),dtype=torch.long)
    def decode(self, ids): return ['unknown']

def mm_embedding():
    return types.SimpleNamespace(first_index=1000000,last_index=1000004,grid_thw=(1,2,2),mrope_merge_size=1)

class Disconnect:
    def __init__(self, hook=None): self.hook=hook;self.cleanup={};self.finished=[]
    async def add_cleanup_task(self, key, callback, args):
        self.cleanup[key]=(callback,args)
        if self.hook: await self.hook()
    async def poll(self): pass
    async def finish(self,key): self.finished.append(key);self.cleanup.pop(key,None)


def engine(held=True, mrope=False):
    g = GEN.Generator.__new__(GEN.Generator)
    owner = types.SimpleNamespace(sequences=[mm_embedding()])
    g._ls_prefill_runtime = types.SimpleNamespace(job=owner) if held else None
    g.pending_jobs=[];g.active_jobs=[owner] if held else [];g.job_serial=39
    g.pagetable=types.SimpleNamespace(max_pages=4096,referenced_pages={},num_unreferenced_pages=lambda:4096)
    g.num_draft_tokens_by_batch=None;g.max_num_draft_tokens=3;g.max_total_tokens=1048576
    g.recurrent_checkpoint_interval=4096;g.recurrent_checkpoint_interval_pp=32768;g.recurrent_checkpoint_tail_pp=6144
    g.recurrent_cache={};g.mtp_draft=True;g.max_batch_size=4;g.max_chunk_size=2048
    g.cache=types.SimpleNamespace(recurrent_state_cls=types.SimpleNamespace(guaranteed_rollback=0))
    g.padded_vocab_size=128;g.ngram_match_min=0;g.tokenizer=Tokenizer()
    g.rope_calls=[]
    rope_module=sys.modules['exllamav3.util.rope']
    cpu_rope=rope_module.RoPE('cpu',rope_module.RopeSettings(head_dim=8,rope_scaling={'mrope_section':[2,1,1]}))
    def rope(*a):
        assert a[0].device.type == 'cpu', 'MRoPE IDs must be CPU'
        g.rope_calls.append(a)
        freqs,offset=cpu_rope.get_mrope_freqs(*a)
        assert freqs.device.type == 'cpu'
        return freqs,offset
    g.model=types.SimpleNamespace(caps={'mrope':mrope},g_rope=types.SimpleNamespace(get_mrope_freqs=rope))
    g.on_queue_drained=lambda:None
    # Do not launch the background GPU iteration loop. Actual wrapper enqueue/cancel/queues remain in use.
    ag=ASYNC.AsyncGenerator.__new__(ASYNC.AsyncGenerator)
    ag.generator=g;ag.jobs={};ag.error=None;ag.condition=asyncio.Condition()
    return g, ag


def container(ag, n=205):
    c=APP_MOD.ExllamaV3Container.__new__(APP_MOD.ExllamaV3Container)
    c.tokenizer=ag.generator.tokenizer;c.generator=ag;c.max_seq_len=1048576
    c.hf_model=types.SimpleNamespace(add_bos_token=lambda:False,eos_tokens=lambda:[99])
    c.config=types.SimpleNamespace(eos_token_id_list=[99]);c.cache=types.SimpleNamespace(max_num_tokens=1048576)
    c.active_job_ids={};c.job_max_rq_tokens=lambda n:n+4
    async def encode(*a): return torch.arange(n,dtype=torch.long).unsqueeze(0)
    c._encode_prompt=encode
    c.handle_logprobs=lambda result,out:out.update(logprobs_content=['cpu-test'])
    c.handle_finish_chunk=lambda *a:dict(finish_reason='length',eos_reason='max_new_tokens')
    return c


class Frontend(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        TRACE.ENABLED=True;TRACE.REGISTRY.clear();TRACE.HISTORY.clear()
        TRACE.CONTEXT.set(dict(key='r823/r825p/candidate/short200',conversation='cpu',explicit=True,roles=[]))

    async def run_request(self, *, held=True, n=205, params=None, mrope=False, cancel=False, constrain=False):
        g,ag=engine(held,mrope);c=container(ag,n);self.g=g;self.ag=ag
        self.before=None
        async def at_registration():
            aj=c.active_job_ids['short'];j=aj.job
            self.before=copy.copy(j.__dict__)
            self.assertIsNotNone(j.sequences[0].page_hashes)
            self.assertIsNotNone(j.all_unique_hashes)
            self.assertIsNotNone(j.held_tokens)
            self.assertIsNotNone(j.time_enqueue)
            self.assertIsNone(j.sequences[0].allocated_pages)
            self.assertEqual(len(g.rope_calls),int(mrope))
            if constrain:
                self.assertTrue(c.constrain_generation_output('short','forced'))
                self.assertEqual(j.forced_ids.tolist(),[[7,8]])
                self.assertTrue(j.filters_suspended)
            if cancel:
                await aj.cancel()
                self.assertNotIn(j,g.pending_jobs)
                self.assertIsNotNone(g._ls_prefill_runtime)
                return
            g._ls_prefill_runtime=None
            # Drive the real admission method with fake allocation/activation; no device model.
            j.activate=lambda:None
            j.allocate_pages=lambda:None
            rs=[];g.iterate_start_jobs(rs)
            if mrope: self.assertEqual(g.rope_calls.__len__(),1)
            else: self.assertFalse(g.rope_calls)
            self.assertEqual(j.time_enqueue,self.before['time_enqueue'])
            if constrain:self.assertEqual(j.forced_ids.tolist(),[[7,8]])
            ag.deliver_results(rs)
            ag.deliver_results([dict(job=j,stage='streaming',eos=True,text='ok',token_ids=torch.tensor([[42]]))])
        d=Disconnect(at_registration)
        mm=types.SimpleNamespace(content=[mm_embedding()]) if mrope else None
        return [x async for x in c.generate_gen('short','cpu-prompt',params or Params(),d,mm)]

    async def test_real_generate_gen_midwindow_short205(self):
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture): out=await self.run_request()
        self.assertEqual(out[0]['text'],'ok');self.assertEqual(out[0]['token_ids'],[42])
        render=next(json.loads(x.split(' ',1)[1]) for x in capture.getvalue().splitlines() if '"event":"render"' in x)
        self.assertEqual(render['page_digests'],[]) # prepared EMPTY list: real failure was 205 tokens
        self.assertEqual(render['serial'],39)
        self.assertFalse(getattr(self.g.active_jobs[-1],'_ls_deferred_prepare',False))

    async def test_real_generate_gen_hashed_prompt_and_token_healing(self):
        p=Params();p.token_healing=True
        with contextlib.redirect_stdout(io.StringIO()): await self.run_request(n=1025,params=p)
        j=self.g.active_jobs[-1]
        self.assertEqual(len(j.sequences[0].sequence_ids),1024)
        self.assertEqual(len(j.sequences[0].page_hashes),3)
        self.assertEqual(j.return_top_tokens,2);self.assertTrue(j.return_probs)
        self.assertEqual(j.stop_tokens,{99});self.assertEqual(j.max_new_tokens,16)

    async def test_real_generate_gen_immediate_cpu_mrope_and_injection_survive(self):
        with contextlib.redirect_stdout(io.StringIO()): await self.run_request(mrope=True,constrain=True)
        freqs=self.g.active_jobs[-1].alt_rope_freqs
        self.assertIsInstance(freqs,torch.Tensor);self.assertEqual(freqs.device.type,'cpu')
        self.assertEqual(freqs.shape,(1,205,4));self.assertEqual(self.g.active_jobs[-1].alt_rope_offset,0)

    async def test_real_generate_gen_pending_cancel_does_not_drain_owner(self):
        with contextlib.redirect_stdout(io.StringIO()): out=await self.run_request(cancel=True)
        self.assertEqual(out,[]);self.assertEqual(self.ag.jobs,{})

    async def test_normal_render_identical_to_inherited_trace(self):
        base=load('exllamav3.inherited_trace',ROOT/'base/cache_trace.py');base.ENABLED=True;base._SALT=TRACE._SALT
        g,ag=engine(False);j=JOB.Job(input_ids=torch.arange(1025).unsqueeze(0),max_new_tokens=16)
        g.enqueue(j)
        values=[]
        for module in (base,TRACE):
            with patch.object(module,'emit',side_effect=lambda event,**v:values.append((event,v))):
                module.attach(j,dict(key='r823/test',conversation='cpu',n=1025))
        self.assertEqual(values[0],values[1])
        self.assertEqual(len(values[0][1]['page_digests']),4)

    async def test_real_failing_log_replay(self):
        evidence=json.loads((ROOT/'fixtures/late-arrival-failure.json').read_text())
        for s in ("'NoneType' object is not iterable",'r823_trace.attach(job.job, trace_record)',
                  '313, in stream_generate_completion','212, in _stream_collector','line 1559, in',
                  'race.py", line 131, in attach'):
            self.assertIn(s,evidence['stack_markers'])
        short=evidence['encode']
        allocation=evidence['allocation']
        self.assertEqual(short['n'],205)
        self.assertEqual(allocation['serial'],39)
        self.assertGreater(allocation['t'],short['t'])
        with contextlib.redirect_stdout(io.StringIO()):out=await self.run_request(n=short['n'])
        self.assertEqual(out[0]['text'],'ok')


class Contracts(unittest.IsolatedAsyncioTestCase):
    def setUp(self): TRACE.ENABLED=True

    async def test_trace_unknown_hashes_emit_once_at_allocation(self):
        g,ag=engine();j=JOB.Job(input_ids=torch.arange(1025).unsqueeze(0),max_new_tokens=16)
        j.generator=g;j.serial_number=39
        values=[]
        with patch.object(TRACE,'emit',side_effect=lambda event,**v:values.append((event,v))):
            TRACE.attach(j,dict(key='r823/test',conversation='cpu',n=1025))
            self.assertIsNone(values[0][1]['page_digests'])
            self.assertTrue(j._r823_pending_page_digests)
            j.prepare_for_queue(g,39)
            pt=types.SimpleNamespace(get_live_page=lambda h:None,metrics={})
            TRACE.before_allocation(j.sequences[0],pt,None)
            TRACE.before_allocation(j.sequences[0],pt,None)
        later=[v for event,v in values if event=='page_digests']
        self.assertEqual(len(later),1)
        self.assertEqual(later[0]['page_digests'],[TRACE.digest(h) for h in j.sequences[0].page_hashes])
        self.assertEqual(later[0]['serial'],39)
        self.assertFalse(j._r823_pending_page_digests)

    async def test_trace_disabled_does_not_attach_or_emit(self):
        g,ag=engine();j=JOB.Job(input_ids=torch.tensor([[1]]),max_new_tokens=16)
        with patch.object(TRACE,'ENABLED',False),patch.object(TRACE,'emit') as emit:
            TRACE.attach(j,dict(key='r823/test'))
        emit.assert_not_called();self.assertFalse(hasattr(j,'_r823_trace'))

    async def test_host_preparation_matches_synchronous_normal_and_requeue(self):
        # Host metadata and CPU hashes must be exactly the same as the inherited implementation.
        original=load('exllamav3.generator.original_job',ROOT/'base/generator/job.py')
        for n in (1,205,256,257,1025):
            for healing in (False,True):
                for max_new in (None,16):
                    with self.subTest(n=n,healing=healing,max_new=max_new):
                        g,ag=engine();g.max_total_tokens=8192
                        kwargs=dict(input_ids=torch.arange(n).unsqueeze(0),max_new_tokens=max_new,
                                    max_rq_tokens=20,token_healing=healing,return_top_tokens=2,stop_conditions=['end',99])
                        old=original.Job(**kwargs);new=JOB.Job(**kwargs)
                        old.prepare_for_queue(g,39)
                        new.prepare_for_queue(g,39)
                        for name in ('max_new_tokens','max_rq_tokens','skips','serial_number','stop_tokens',
                                     'stop_strings','alt_rope_freqs','alt_rope_offset','sam'):
                            self.assertEqual(getattr(old,name),getattr(new,name),name)
                        self.assertEqual(set(old.all_unique_hashes),set(new.all_unique_hashes))
                        for name in ('page_hashes','max_cached_pages','new_unique_pages'):
                            self.assertEqual(getattr(old.sequences[0],name),getattr(new.sequences[0],name))
                        before=new.held_tokens
                        new.held_text='already held';new.full_completion='keep me';new.sam=object()
                        sam=new.sam
                        new.prepare_for_queue(g,39,rq=True)
                        self.assertIs(new.held_tokens,before);self.assertIs(new.sam,sam)
                        self.assertEqual(new.held_text,'already held');self.assertEqual(new.full_completion,'keep me')

    async def test_host_phase_cannot_touch_page_or_recurrent_state_or_cuda(self):
        class Table:
            max_pages=4096
            def __getattr__(self,name): raise AssertionError('page table touch: '+name)
        class Cache:
            recurrent_state_cls=types.SimpleNamespace(guaranteed_rollback=256)
            def __getattr__(self,name): raise AssertionError('cache touch: '+name)
        g,ag=engine(mrope=True);g.pagetable=Table();g.cache=Cache();g.ngram_match_min=1
        def forbidden(*a,**k): raise AssertionError('GPU API')
        with patch.object(torch.cuda,'_lazy_init',forbidden),patch.object(torch.cuda,'synchronize',forbidden):
            aj=ASYNC.AsyncJob(ag,input_ids=torch.arange(1025).unsqueeze(0),max_new_tokens=16,
                             embeddings=[mm_embedding()],banned_strings=['bad'],filters=[types.SimpleNamespace(is_active=True)])
            aj.constrain_output_now(torch.tensor([[7,8]]))
        self.assertEqual(len(aj.job.sequences[0].page_hashes),4)
        self.assertIsNotNone(aj.job.sam);self.assertIsNone(aj.job.recurrent_state)
        self.assertEqual(len(g.rope_calls),1);self.assertFalse(hasattr(aj.job,'_ls_deferred_prepare'))
        self.assertFalse(torch.cuda.is_initialized())

    async def test_invalid_host_preparation_leaves_no_async_mapping_or_pending_job(self):
        for held in (False,True):
            g,ag=engine(held);g.pagetable.max_pages=0
            with self.assertRaisesRegex(AssertionError,'cannot be enqueued'):
                ASYNC.AsyncJob(ag,input_ids=torch.tensor([[1]]),max_new_tokens=16)
            self.assertEqual(ag.jobs,{});self.assertEqual(g.pending_jobs,[])
            self.assertEqual(g.job_serial,39)

    async def test_mrope_failure_is_immediate_and_leaves_owner_and_queues_intact(self):
        g,ag=engine(mrope=True)
        owner=g.active_jobs[0]
        def fail(*a): raise ValueError('mrope failure')
        g.model.g_rope.get_mrope_freqs=fail
        with self.assertRaisesRegex(ValueError,'mrope failure'):
            ASYNC.AsyncJob(ag,input_ids=torch.tensor([[1]]),max_new_tokens=16,embeddings=[mm_embedding()])
        self.assertEqual(g.active_jobs,[owner]);self.assertEqual(g.pending_jobs,[])
        self.assertEqual(ag.jobs,{});self.assertIsNone(ag.error)
        self.assertIs(g._ls_prefill_runtime.job,owner)

    async def test_real_generate_gen_latched_engine_error_is_not_masked_by_attach(self):
        g,ag=engine();ag.error=RuntimeError('latched engine failure')
        c=container(ag);recover=[]
        async def no_gpu_recovery(ex,job):recover.append(ex)
        c._recover_from_generation_error=no_gpu_recovery
        TRACE.CONTEXT.set(dict(key='r823/test',conversation='cpu',explicit=True,roles=[]))
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'latched engine failure'):
            async for _ in c.generate_gen('short','cpu',Params(),Disconnect()):pass
        self.assertEqual(recover,[ag.error]);self.assertEqual(ag.jobs,{})
        self.assertEqual(g.pending_jobs,[]);self.assertIsNotNone(g._ls_prefill_runtime)

    async def test_generator_fix_passes_real_frontend_with_original_strict_attach(self):
        original=load('exllamav3.original_trace',ROOT/'base/cache_trace.py');original.ENABLED=True
        original.CONTEXT.set(dict(key='r823/test',conversation='cpu',explicit=True,roles=[]))
        case=Frontend('test_real_generate_gen_midwindow_short205')
        with patch.object(sys.modules['exllamav3'],'cache_trace',original),contextlib.redirect_stdout(io.StringIO()):
            out=await case.run_request()
        self.assertEqual(out[0]['text'],'ok')


if __name__=='__main__': unittest.main(verbosity=2)
