#!/usr/bin/env python3
"""GPU step 0. Old cut versus in-forward checkpoints, same fixture IDs, one model load.
Run only under the unit's GPU lock, with daily down, disk off, and daily selectors.
Every tensor is torch.equal AND equal byte representations; no tolerance gate.
"""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True


def equal_stash(a, b):
    import torch
    errors = []
    if a.keys() != b.keys():
        return ['stash keys differ']
    for key, value in a.items():
        other = b[key]
        if isinstance(value, (tuple, list)):
            if len(value) != len(other):
                errors.append(f'{key}: tensor count'); continue
            for i, (x, y) in enumerate(zip(value, other)):
                if (x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y) or
                        not torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))):
                    errors.append(f'{key}/{i}: tensor differs')
        elif value != other:
            errors.append(f'{key}: scalar differs')
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.cache import prefill_merge as pm
    from exllamav3 import cache_trace as trace
    from r823_reuse import load_fixtures
    assert 'EXL3_NVME_TIER' not in os.environ, 'disk must be off'
    assert os.environ.get('EXL3_LS_PREFILL_PIPELINE') == '1', 'pipeline must be enabled'
    assert os.environ.get('EXL3_STASH_ASYNC') == '1', 'daily async mode required'
    os.environ['EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP'] = '4096'
    os.environ['EXL3_RECURRENT_CHECKPOINT_TAIL_PP'] = '12288'
    os.environ.pop('EXL3_RECURRENT_CHECKPOINT_INFORWARD', None)
    manifest, payloads = load_fixtures(a.fixtures)
    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    draft = Model.from_config(config, component='mtp')
    ckw = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
    cache = Cache(model, max_num_tokens=901120, max_batch_size=8, max_history=3, **ckw)
    dcache = Cache(draft, max_num_tokens=901120, max_batch_size=8, max_history=3, **ckw)
    draft.load(use_per_device=[0,32], max_chunk_size=2048)
    model.load(use_per_device=[30,30], max_chunk_size=2048, max_batch_size=8)
    gen = Generator(model=model, cache=cache, tokenizer=Tokenizer.from_config(config),
                    max_batch_size=8, max_chunk_size=2048, draft_model=draft,
                    draft_cache=dcache, num_draft_tokens=3, dynamic_draft_tokens=False, num_draft_tokens_by_batch=[(4,3),(8,2)],
                    recurrent_cache_size=4096*1024**2, cpu_cache_size=0)
    assert gen.recurrent_cache is not None
    # T35 exercises a nonzero cached base (2560); other seeds are cold, long and distinct.
    cases = [r for r in manifest['requests'] if r['path']=='raw' and
             r['kind']=='seed' and r['case'] in ('T35','T32','T65')]
    assert len(cases)==3
    rows=[]
    original_emit=trace.emit
    events=[]
    def emit(event, **values):
        if event in ('tail_capture','tail_fallback','prefill_forward'):
            events.append(dict(event=event, **values))
        original_emit(event, **values)
    trace.emit=emit
    def run(row):
        ids=torch.tensor([payloads[row['payload']]['token_ids']], dtype=torch.long)
        job=Job(input_ids=ids, max_new_tokens=1, sampler=GreedySampler(), stop_conditions=[])
        gen.enqueue(job)
        while gen.num_remaining_jobs():
            for result in gen.iterate():
                if result['stage']=='error': raise RuntimeError(str(result))
        pm.drain_worker()
    def reset():
        pm.drain_worker()
        gen.pagetable.reset_page_table();gen.recurrent_cache.clear();gen.recurrent_cache.update_total_size()
    def host_snapshots():
        snapshots={}
        for key in gen.recurrent_cache.keys():
            st=gen.recurrent_cache[key];pm.wait_stash(st)
            assert st['position'] not in snapshots, 'duplicate position'
            snapshots[st['position']]={k:tuple(t.detach().cpu().clone() for t in v)
                if isinstance(v,(tuple,list)) else v for k,v in st.items()}
        return snapshots
    for case in cases:
        arms={};engagement={}
        for arm in ('CUT','INFORWARD','NO_SLAB'):
            reset();os.environ.pop('EXL3_RECURRENT_CHECKPOINT_INFORWARD',None)
            if case['case']=='T35':
                warm=next(r for r in manifest['requests'] if r['path']=='raw' and r['case']=='T35' and r['kind']=='warm')
                run(warm)
            if arm!='CUT':os.environ['EXL3_RECURRENT_CHECKPOINT_INFORWARD']='1'
            events.clear()
            saved_try=pm.try_boundary
            if arm=='NO_SLAB':pm.try_boundary=lambda *args,**kwargs:None
            try:run(case)
            finally:pm.try_boundary=saved_try
            arms[arm]=host_snapshots()
            engagement[arm]=dict(captures=sum(e['event']=='tail_capture' for e in events),
                fallbacks=sum(e['event']=='tail_fallback' for e in events),
                pipeline=sum(e['event']=='prefill_forward' and e['pipeline'] for e in events))
        differences=[]
        for arm in ('INFORWARD','NO_SLAB'):
            if arms[arm].keys()!=arms['CUT'].keys():differences.append(f'{arm}: positions differ')
            for pos in arms['CUT'].keys() & arms[arm].keys():
                differences += [f'{arm}/{pos}: {e}' for e in equal_stash(arms['CUT'][pos], arms[arm][pos])]
        engaged=(engagement['INFORWARD']['captures']>0 and engagement['INFORWARD']['pipeline']>0 and
                 engagement['CUT']['captures']==0 and engagement['NO_SLAB']['captures']==0 and
                 engagement['NO_SLAB']['fallbacks']>0)
        rows.append(dict(case=case['case'], positions=sorted(arms['CUT']), engagement=engagement,
                         equal=not differences, engaged=engaged, differences=differences))
        print(json.dumps(rows[-1]),flush=True)
    passed=all(r['equal'] and r['engaged'] for r in rows)
    a.out.write_text(json.dumps(dict(passed=passed, rows=rows),indent=2)+'\n')
    print('R823c GPU checkpoint equality '+('PASS' if passed else 'FAIL'),flush=True)
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main())
