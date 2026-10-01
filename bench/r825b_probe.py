#!/usr/bin/env python3
"""Operator GPU equality then cold A/B; offline result gate uses stdlib only."""
import argparse
import asyncio
import math
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# Unit archive places unchanged shared probes here.
sys.path.insert(0, str(ROOT/'probes'))
sys.path.insert(0, str(ROOT.parents[1]/'probes'))
ORDER = ('A', 'B', 'B', 'A')
SIZES = (20000, 50000, 90000)


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def analyze(path):
    data = json.loads(Path(path).read_text())
    equality_path = Path(path).with_name('equality.json')
    if not equality_path.is_file():
        raise ValueError('missing full equality evidence')
    equality = json.loads(equality_path.read_text())
    expected_equal = [(case, arm) for case in ('T35','T32','T65','20000','50000','90000','20001','50001','90113')
                      for arm in ('A','B','NO_SLAB')]
    if (not equality.get('passed') or
            [(r['case'],r['arm']) for r in equality.get('rows',[])] != expected_equal or
            any(not r.get('equal') or r.get('differences') or not r.get('positions')
                for r in equality.get('rows',[]))):
        raise ValueError('incomplete/failed equality matrix')
    rows = data['timings']
    if len(rows) != 36:
        raise ValueError('require three ABBA blocks at each of three sizes')
    expected = [(size, block, slot, arm) for size in SIZES for block in range(3)
                for slot, arm in enumerate(ORDER)]
    if [(r['size'], r['block'], r['slot'], r['arm']) for r in rows] != expected:
        raise ValueError('interleave/order mismatch')
    if not data['equality_passed'] or not data['daily_identity_passed']:
        raise ValueError('missing exact equality against LIVE baseline')
    for r in rows:
        if r['cached'] != 0 or r['prompt_tokens'] != r['size'] or r['engine_ms'] <= 0:
            raise ValueError('non-cold/incomplete timing')
        if r['alloc_retries'] != [0, 0] or any(x < 64*1024**2 for x in r['driver_free']):
            raise ValueError('memory/retry gate failed')
        if r['arm'] == 'B' and (r['fallbacks'] or r['serial_rows'] or
                                r['pipeline_windows'] != 1 or not r['final_pipeline']):
            raise ValueError('B did not run one whole prompt window')
    # Each ABBA block uses one immutable input in all four arms and clears caches.
    for i in range(0, len(rows), 4):
        if len({r['prompt_sha256'] for r in rows[i:i+4]}) != 1:
            raise ValueError('A/B prompt mismatch')
    result = {}
    for size in SIZES:
        selected = [r for r in rows if r['size'] == size]
        med = {arm: statistics.median(r['engine_ms'] for r in selected if r['arm'] == arm)
               for arm in ('A', 'B')}
        result[str(size)] = dict(median_ms=med, saved_pct=100*(1-med['B']/med['A']),
                                reps_per_arm=6)
    peer = equality.get('late_peer')
    if not peer or not peer.get('equal') or peer.get('differences') or not peer.get('positions'):
        raise ValueError('missing/failed late-peer equality')
    if not (peer['arrival_chunk'] >= 2 and peer['finish_chunk'] == peer['arrival_chunk'] + 1
            and peer['short_output_count'] == 32):
        raise ValueError('late peer was not enqueued during a retained window')
    for row in rows:
        gap = row.get('heartbeat_max_gap_ms')
        ticks = row.get('heartbeat_ticks')
        if not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap <= 0 or not ticks:
            raise ValueError('missing/invalid async heartbeat evidence')
    for size in SIZES:
        result[str(size)]['heartbeat_max_gap_ms'] = {
            arm: max(r['heartbeat_max_gap_ms'] for r in rows if r['size'] == size and r['arm'] == arm)
            for arm in ('A','B')}
    # Registration: exact gates first, >=5% saving AND <=500ms heartbeat gap.

    supported = all(r['saved_pct'] >= 5 and r['heartbeat_max_gap_ms']['B'] <= 500 for r in result.values())
    return dict(verdict='SUPPORTED' if supported else 'NOT-SUPPORTED', sizes=result,
                note='Engine time only; no promotion. Exact checkpoint/logit/token gates are mandatory.')


def gpu(a, baseline=False):
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.cache import prefill_merge as pm
    from exllamav3.generator import prefill_pipeline as pp
    from exllamav3 import cache_trace as trace
    from r823_reuse import load_fixtures
    from r823c_checkpoint_equal import equal_stash
    assert torch.cuda.device_count() == 2
    assert all(torch.cuda.get_device_capability(d) == (12, 0) for d in range(2))
    for k, v in {'EXL3_LS_PREFILL_PIPELINE':'1', 'EXL3_STASH_ASYNC':'1',
                 'EXL3_PREFILL_MERGE':'1', 'EXL3_RECURRENT_CHECKPOINT_INFORWARD':'1',
                 'EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP':'4096',
                 'EXL3_RECURRENT_CHECKPOINT_TAIL_PP':'12288'}.items():
        assert os.environ.get(k) == v, (k, os.environ.get(k))
    assert 'EXL3_NVME_TIER' not in os.environ
    # The served daily (launcher 8b644c17) does not set EXL3_PREFILL_NOSYNC: both arms use its default stage join.
    assert os.environ.get('EXL3_PREFILL_NOSYNC', '0') == '0', 'daily runs without EXL3_PREFILL_NOSYNC'
    expected_env = dict(line.split('=', 1) for line in (ROOT/'fixtures/live-container-env.txt').read_text().splitlines() if '=' in line)
    expected_env = {k: v for k, v in expected_env.items() if k.startswith(('EXL3_', 'EXLLAMAV3_'))}
    actual_env = {k: v for k, v in os.environ.items() if k.startswith(('EXL3_', 'EXLLAMAV3_'))
                  and k not in ('EXL3_PREFILL_WHOLE_PROMPT', 'EXL3_PREFILL_RESUMABLE')}
    assert actual_env == expected_env, (actual_env, expected_env)
    os.environ['EXL3_PREFILL_RESUMABLE'] = '1'
    os.environ['EXL3_PREFILL_WHOLE_PROMPT'] = '0'
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = Config.from_directory(a.model)
    model, draft = Model.from_config(cfg), Model.from_config(cfg, component='mtp')
    ckw = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
    cache = Cache(model, max_num_tokens=901120, max_batch_size=8, max_history=3, **ckw)
    dcache = Cache(draft, max_num_tokens=901120, max_batch_size=8, max_history=3, **ckw)
    draft.load(use_per_device=[0,32], max_chunk_size=2048, max_batch_size=8)
    model.load(use_per_device=[30,30], max_chunk_size=2048, max_batch_size=8)
    tokenizer = Tokenizer.from_config(cfg)
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
        max_batch_size=8, max_chunk_size=2048, draft_model=draft, draft_cache=dcache,
        num_draft_tokens=3, dynamic_draft_tokens=False, num_draft_tokens_by_batch=[(4,3),(8,2)],
        recurrent_cache_size=4096*1024**2, cpu_cache_size=0)
    layout = pp._layout(model)
    assert layout is not None
    actual = [[(m.key, inst, idx) for m, inst, idx in stage] for stage in layout[1]]
    assert json.loads(json.dumps(actual)) == json.loads(a.daily_layout.read_text())['stages']
    dump(a.out/'placement.json', dict(stages=actual, pool=901120, batch=8, history=3))
    manifest, payloads = load_fixtures(a.fixtures)
    body = tokenizer.encode(a.prompt_file.read_text(), add_bos=False)

    def ids_for(size, key):
        head = tokenizer.encode(f'R825 cold {key}:\n', add_bos=True)
        assert head.shape[1] < 256
        filler = body.repeat(1, size//body.shape[1]+2)[:, :size-head.shape[1]]
        return torch.cat((head, filler), dim=1)

    def sync():
        for d in range(2):
            torch.cuda.synchronize(d)

    def reset():
        assert gen.num_remaining_jobs() == 0
        pm.drain_worker()
        sync()
        gen.pagetable.reset_page_table()
        gen.recurrent_cache.clear()
        gen.recurrent_cache.update_total_size()

    events, logits = [], {}
    original_emit = trace.emit
    def emit(event, **values):
        if event in ('tail_capture', 'tail_fallback', 'prefill_forward'):
            events.append(dict(event=event, **values))
        original_emit(event, **values)
    trace.emit = emit
    window_count = [0]
    peer_boundary = {}
    primary = [None]
    original_window = pp.run_window
    def window(*args, **kwargs):
        runtime = args[2]
        if runtime.index == 0:
            window_count[0] += 1
        terminating = bool(getattr(runtime, 'resumable', False) and runtime.index > 0
                           and gen.pending_jobs and runtime.job is primary[0])
        value = original_window(*args, **kwargs)
        if terminating:
            assert not hasattr(runtime.job, '_ls_prefill_pipeline')
            peer_boundary['finish_chunk'] = runtime.index
        return value
    pp.run_window = window
    # Equality-only hooks: copy the complete final prompt logits and first
    # verification logits. Hooks are removed for every timing measurement.
    heads = [(m, m.forward) for m, _, _ in model.fwd_modules if m.caps.get('logits_output')]
    assert len(heads) == 1
    checking = [False]
    # Prefill marker: the daily carries no prefill-only param (_prefill_nosync exists only with
    # EXL3_PREFILL_NOSYNC=1), so mark the extent of Job.prefill; decode/verification run outside it.
    in_prefill = [0]
    prefill_owner = [None]
    original_job_prefill = Job.prefill
    def job_prefill(self, *args, **kwargs):
        in_prefill[0] += 1
        prefill_owner[0] = self
        try:
            return original_job_prefill(self, *args, **kwargs)
        finally:
            in_prefill[0] -= 1
    Job.prefill = job_prefill
    def logits_hook(fn):
        def wrapped(x, params, *args, **kwargs):
            y = fn(x, params, *args, **kwargs)
            if checking[0]:
                if in_prefill[0] and prefill_owner[0] is primary[0]:
                    logits['final_prompt'] = y.detach().cpu().clone()
                elif not in_prefill[0] and primary[0].sequences[0].prefill_complete and 'first_verify' not in logits:
                    logits['first_verify'] = y.detach().cpu().clone()
            return y
        return wrapped
    for m, fn in heads:
        m.forward = logits_hook(fn)

    def snapshots(end):
        result = {}
        pm.drain_worker()
        for key in gen.recurrent_cache.keys():
            st = gen.recurrent_cache[key]
            pm.wait_stash(st)
            if st['position'] > end:
                continue
            assert st['position'] not in result, 'duplicate snapshot position'
            result[st['position']] = {k: tuple(t.detach().cpu().clone() for t in v)
                if isinstance(v, (tuple, list)) else v for k, v in st.items()}
        return result

    def run(ids, check=False, warm=False, new_tokens=None, peer_ids=None):
        events.clear(); logits.clear(); window_count[0] = 0; checking[0] = check
        job = Job(input_ids=ids.clone(), max_new_tokens=new_tokens if new_tokens is not None else (32 if check else 1),
                  sampler=GreedySampler(), stop_conditions=[], seed=1234)
        primary[0] = job
        peer_boundary.clear()
        sync()
        retries = [torch.cuda.memory_stats(d).get('num_alloc_retries', 0) for d in range(2)]
        outputs, short_outputs, gaps, iterate_ms = [], [], [], []
        async def drive():
            # Run the actual served AsyncGenerator, injecting the already loaded
            # synchronous Generator to avoid a second stack/cache allocation.
            import exllamav3.generator.async_generator as am
            saved_factory, original_iterate = am.Generator, gen.iterate
            am.Generator = lambda *args, **kwargs: gen
            wrapper = am.AsyncGenerator()
            am.Generator = saved_factory
            previous = [time.perf_counter()]
            cold_end = [None]
            def observed_iterate():
                started = time.perf_counter()
                result = original_iterate()
                iterate_ms.append(1000*(time.perf_counter()-started))
                if job.time_first_token is not None and cold_end[0] is None:
                    cold_end[0] = time.perf_counter()
                return result
            gen.iterate = observed_iterate
            async def heartbeat():
                while cold_end[0] is None or previous[0] < cold_end[0]:
                    await asyncio.sleep(0.05)
                    now = time.perf_counter()
                    gaps.append(1000*(now-previous[0]))
                    previous[0] = now
            def attach(raw):
                wrapped = am.AsyncJob.__new__(am.AsyncJob)
                wrapped.generator, wrapped.job = wrapper, raw
                wrapped.queue, wrapped.cancelled = asyncio.Queue(), False
                wrapper.enqueue(wrapped)
                return wrapped
            async def consume(wrapped, dest):
                async for res in wrapped:
                    if 'token_ids' in res:
                        dest.extend(res['token_ids'].flatten().tolist())
            async def late_peer():
                while True:
                    rt = getattr(gen, '_ls_prefill_runtime', None)
                    if rt is not None and rt.job is job and rt.index >= 3:
                        assert rt.pending is not None and rt.pending.done()
                        peer_boundary['arrival_chunk'] = rt.index
                        short = Job(input_ids=peer_ids.clone(), max_new_tokens=32,
                                    sampler=GreedySampler(), stop_conditions=[], seed=1234)
                        await consume(attach(short), short_outputs)
                        return
                    if cold_end[0] is not None:
                        raise RuntimeError('late-peer window never engaged')
                    await asyncio.sleep(0.000001)
            beat = asyncio.create_task(heartbeat())
            # Start heartbeat before the generator task can issue the cold prefill.
            await asyncio.sleep(0.000001)
            consumer = asyncio.create_task(consume(attach(job), outputs))
            peer_task = asyncio.create_task(late_peer()) if peer_ids is not None else None
            try:
                await asyncio.wait_for(consumer, timeout=600)
                if peer_task is not None:
                    await asyncio.wait_for(peer_task, timeout=600)
                await asyncio.wait_for(beat, timeout=2)
            finally:
                for task in (beat, consumer, peer_task):
                    if task is not None and not task.done():
                        task.cancel()
                await wrapper.close()
                gen.iterate = original_iterate
        asyncio.run(drive())
        assert gen.num_remaining_jobs() == 0
        sync(); checking[0] = False
        cached = job.cached_pages*256 + job.cached_tokens
        if not warm:
            assert cached == 0
        record = dict(prompt_tokens=ids.shape[1], cached=cached,
            engine_ms=(job.time_first_token-job.time_first_prefill)*1000,
            prompt_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            alloc_retries=[torch.cuda.memory_stats(d).get('num_alloc_retries',0)-retries[d] for d in range(2)],
            driver_free=[torch.cuda.mem_get_info(d)[0] for d in range(2)],
            allocated=[torch.cuda.memory_allocated(d) for d in range(2)],
            reserved=[torch.cuda.memory_reserved(d) for d in range(2)],
            fallbacks=sum(e['event']=='tail_fallback' for e in events),
            pipeline_windows=window_count[0], heartbeat_max_gap_ms=max(gaps),
            heartbeat_ticks=len(gaps), iterate_max_ms=max(iterate_ms),
            peer_boundary=dict(peer_boundary), short_output_ids=short_outputs)
        fw = [e for e in events if e['event']=='prefill_forward' and e['prompt_prefill'] and e['serial'] == job.serial_number]
        assert fw and fw[0]['start'] == cached and fw[-1]['end'] == ids.shape[1]-1
        assert all(b['start']==x['end'] for x,b in zip(fw,fw[1:]))
        record['serial_rows'] = sum(e['end']-e['start'] for e in fw if not e['pipeline'])
        record['final_pipeline'] = fw[-1]['pipeline']
        record['crossed'] = [e['position'] for e in events if e['event']=='tail_capture' and e['crossed']]
        if check:
            assert len(outputs) == 32, len(outputs)
            assert set(logits) == {'final_prompt','first_verify'}, logits.keys()
            return dict(record=record, snapshots=snapshots(ids.shape[1]-1),
                        logits=dict(logits), output_ids=outputs)
        return record

    # Identical R818 fast-state warmup strings in each baseline/candidate boot.
    for n, text in [(20,'Note 04560: train fire door sand square fire snow ocean wind house cloud.'),
                    (29,'Note 12479: number fish metal horse bridge copper castle apple leaf music island square rain chair copper snow seed corner floor village.'),
                    (11,'Note 07311: bridge music.')]:
        ids = tokenizer.encode(text, add_bos=False)
        assert ids.shape[1] == n
        run(ids, new_tokens=8)
    reset()
    run(ids_for(90113, 'shape-warm'))
    reset()
    cases = []
    for name in ('T35','T32','T65'):
        row = next(r for r in manifest['requests'] if r['path']=='raw' and r['kind']=='seed' and r['case']==name)
        warm_ids = None
        if name == 'T35':
            w = next(r for r in manifest['requests'] if r['path']=='raw' and r['kind']=='warm' and r['case']==name)
            warm_ids = torch.tensor([payloads[w['payload']]['token_ids']], dtype=torch.long)
        cases.append((name, torch.tensor([payloads[row['payload']]['token_ids']],dtype=torch.long), warm_ids))
    for size in (*SIZES, 20001, 50001, 90113):
        cases.append((str(size), ids_for(size, f'equality-{size}'), None))
    if baseline:
        for name, ids, warm_ids in cases:
            reset()
            if warm_ids is not None:
                run(warm_ids)
            value = run(ids, check=True, warm=warm_ids is not None)
            torch.save(dict(ids=ids, warm_ids=warm_ids, **value), a.out/f'baseline-{name}.pt')
        reset()
        short_ids = ids_for(128, 'late-peer-short')
        short = run(short_ids, check=True)
        torch.save(dict(ids=short_ids, **short), a.out/'baseline-short.pt')
        dump(a.out/'baseline.json' , dict(passed=True, cases=[c[0] for c in cases], daily_image=True))
        return
    assert json.loads((a.reference/'baseline.json').read_text())['passed']
    equality = []
    for name, _, _ in cases:
        reference = torch.load(a.reference/f'baseline-{name}.pt', weights_only=False, map_location='cpu')
        for arm in ('A','B','NO_SLAB'):
            reset(); os.environ['EXL3_PREFILL_WHOLE_PROMPT'] = '0'
            if reference['warm_ids'] is not None:
                run(reference['warm_ids'])
            os.environ['EXL3_PREFILL_WHOLE_PROMPT'] = '0' if arm=='A' else '1'
            saved = pm.try_pipeline_capture
            if arm == 'NO_SLAB':
                pm.try_pipeline_capture = lambda *args, **kwargs: None
            try:
                got = run(reference['ids'], check=True, warm=reference['warm_ids'] is not None)
            finally:
                pm.try_pipeline_capture = saved
            differences = []
            if reference['snapshots'].keys() != got['snapshots'].keys():
                differences.append('checkpoint positions differ')
            for pos in reference['snapshots'].keys() & got['snapshots'].keys():
                differences += [f'{pos}: {e}' for e in equal_stash(reference['snapshots'][pos],got['snapshots'][pos])]
            for key in reference['logits']:
                x, y = reference['logits'][key], got['logits'][key]
                if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x,y) or not torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)):
                    differences.append(key+' logits differ')
            if reference['output_ids'] != got['output_ids']:
                differences.append('greedy output IDs differ')
            if arm == 'B' and (got['record']['fallbacks'] or not got['record']['final_pipeline'] or got['record']['serial_rows']):
                differences.append('whole-prompt path did not engage')
            if arm == 'NO_SLAB' and got['record']['fallbacks'] == 0:
                differences.append('forced fallback did not engage')
            row = dict(case=name, arm=arm, equal=not differences, differences=differences,
                       record=got['record'], positions=sorted(got['snapshots']))
            equality.append(row)
            dump(a.out/'equality.json', dict(passed=all(r['equal'] for r in equality), rows=equality))
            print(json.dumps(row), flush=True)
            if differences:
                raise RuntimeError('GPU equality failed; abort before A/B traffic')
    # A newly enqueued short request interrupts the retained solo window.
    reset()
    os.environ['EXL3_PREFILL_WHOLE_PROMPT'] = '1'
    long_ref = torch.load(a.reference/'baseline-90000.pt', weights_only=False, map_location='cpu')
    short_ref = torch.load(a.reference/'baseline-short.pt', weights_only=False, map_location='cpu')
    got = run(long_ref['ids'], check=True, peer_ids=short_ref['ids'])
    differences = []
    if got['snapshots'].keys() != long_ref['snapshots'].keys():
        differences.append('late-peer checkpoint positions differ')
    for pos in got['snapshots'].keys() & long_ref['snapshots'].keys():
        differences += [f'{pos}: {e}' for e in equal_stash(long_ref['snapshots'][pos], got['snapshots'][pos])]
    for key, x in long_ref['logits'].items():
        y = got['logits'][key]
        if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y) or not torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)):
            differences.append('late-peer '+key+' logits differ')
    if long_ref['output_ids'] != got['output_ids']:
        differences.append('late-peer long output IDs differ')
    if short_ref['output_ids'] != got['record']['short_output_ids']:
        differences.append('late-peer short output IDs differ')
    boundary = got['record']['peer_boundary']
    if boundary.get('finish_chunk') != boundary.get('arrival_chunk', -2) + 1:
        differences.append('late-peer did not finish retained lookahead at next boundary')
    late = dict(equal=not differences, differences=differences, positions=sorted(got['snapshots']),
                short_output_count=len(got['record']['short_output_ids']), record=got['record'], **boundary)
    dump(a.out/'equality.json', dict(passed=not differences and all(r['equal'] for r in equality), rows=equality, late_peer=late))
    print(json.dumps(late), flush=True)
    if differences:
        raise RuntimeError('late-peer equality failed; abort before timing')
    # Remove equality hooks: no logits CPU copies or checkpoint host snapshots in timings.
    for m, fn in heads:
        m.forward = fn
    timing = []
    for size in SIZES:
        for block in range(3):
            ids = ids_for(size, f'timed-{size}-{block}')
            for slot, arm in enumerate(ORDER):
                reset()
                os.environ['EXL3_PREFILL_WHOLE_PROMPT'] = '1' if arm=='B' else '0'
                rec = dict(run(ids), size=size, block=block, slot=slot, arm=arm)
                timing.append(rec)
                dump(a.out/'results.json', dict(equality_passed=True, daily_identity_passed=True,
                                                timings=timing))
                print(json.dumps(rec), flush=True)
    dump(a.out/'decision.json', analyze(a.out/'results.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('baseline','gpu'):
        c = sub.add_parser(name)
        for key in ('model','fixtures','out','daily-layout','prompt-file'):
            c.add_argument('--'+key, required=True, type=str if key=='model' else Path)
        if name=='gpu':
            c.add_argument('--reference',type=Path,required=True)
    c = sub.add_parser('analyze'); c.add_argument('--results',type=Path,required=True);c.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    if a.command=='analyze':
        dump(a.out, analyze(a.results))
    else:
        gpu(a, baseline=a.command=='baseline')

if __name__ == '__main__':
    main()
