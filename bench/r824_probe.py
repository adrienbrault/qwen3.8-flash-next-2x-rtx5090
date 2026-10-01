#!/usr/bin/env python3
"""R824: offline R823-cache curve and operator-only cold CUDA-event profile.

Standard-library imports only until gpu(). No network, ssh, or git in this probe.
Completion deltas are cadence, never claimed as isolated forward GPU times.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import gzip
import statistics
import sys
import time

STAMP = re.compile(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+Z \[R823-cache\] (\{.*\})$')
DEPTHS = (0, 32768, 81920)
LIVE_MD5 = '8b644c17e60049a8069fc901b6f091fa'
IMAGE_ID = 'sha256:f5a3c35e2e47647f2200ff33ca822405036835f0729544fad11bad7c56c226b7'
MODEL = 'qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab'


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def trace_events(path):
    events = []
    for n, line in enumerate(Path(path).read_text(errors='replace').splitlines(), 1):
        if '[R823-cache] ' not in line:
            continue
        m = STAMP.match(line)
        if not m:
            raise ValueError(f'{path}:{n}: unexpected R823-cache format')
        d = json.loads(m[1])
        if d.get('v') != 1 or not isinstance(d.get('t'), (int, float)):
            raise ValueError(f'{path}:{n}: invalid trace version/time')
        if d['event'] == 'prefill_forward':
            for key in ('run', 'key', 'serial', 'start', 'end', 'pipeline', 'prompt_prefill'):
                if key not in d:
                    raise ValueError(f'{path}:{n}: missing {key}')
            if not (isinstance(d['pipeline'], bool) and isinstance(d['prompt_prefill'], bool)
                    and 0 <= d['start'] < d['end']):
                raise ValueError(f'{path}:{n}: invalid forward')
        events.append(dict(d, source_line=n))
    return events


def clients(path):
    records = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if len({r['key'] for r in records}) != len(records):
        raise ValueError(f'{path}: duplicate client keys')
    return {r['key']: r for r in records}


def identity(launcher, inspect, config):
    text = Path(launcher).read_text()
    if hashlib.md5(Path(launcher).read_bytes()).hexdigest() != LIVE_MD5:
        raise ValueError('foreign LIVE launcher (expected 8b644c17)')
    keys = dict(s.split('=',1) for s in re.search(r'^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$',text,re.M)[1].split())
    obj = json.loads(Path(inspect).read_text())[0]
    env = dict(s.split('=',1) for s in obj['Config']['Env'] if '=' in s)
    if obj['Image'] != IMAGE_ID or obj['State']['Status'] != 'running' or obj['RestartCount'] != 0:
        raise ValueError('LIVE image/running/restart mismatch')
    if any(env.get(k) != v for k,v in keys.items()) or env.get('EXL3_CACHE_TRACE') != '1':
        raise ValueError('LIVE selectors differ from pinned launcher')
    if 'EXL3_NVME_TIER' in env:
        raise ValueError('NVMe enabled')
    if any(m['Destination'].startswith(('/opt/venv/lib/python3.12/site-packages/exllamav3','/app/backends','/app/endpoints')) for m in obj['Mounts']):
        raise ValueError('foreign source mount')
    cfg = Path(config).read_text()
    for line in ('cache_size: 901120','max_batch_size: 8','chunk_size: 2048',
                 'sysmem_recurrent_cache: 4096','sysmem_kv_cache: 0'):
        if line not in cfg:
            raise ValueError('LIVE config '+line)
    return {k:v for k,v in env.items() if k.startswith(('EXL3_','EXLLAMA','CUDA_','TRITON_','PYTORCH_'))}


def curve(root, boots):
    rows, requests, sources = [], [], {}
    for boot in boots:
        log, client = Path(root)/f'container-{boot}.log', Path(root)/f'client-{boot}.jsonl'
        for p in (log, client):
            sources[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        groups = collections.defaultdict(list)
        for e in trace_events(log):
            if e['event'] == 'prefill_forward' and e['prompt_prefill']:
                groups[(e['run'], e['key'], e['serial'])].append(e)
        cs = clients(client)
        for (_, key, serial), es in groups.items():
            c = cs.get(key)
            if not c or c.get('status') != 'VALID' or c.get('kind') != 'seed' or c.get('case') == 'C2':
                continue
            u = c.get('usage') or {}
            if u.get('prompt_tokens', 0) < 30000 or u.get('prompt_tokens_details', {}).get('cached_tokens') != 0:
                continue
            if es[0]['start'] != 0 or any(b['start'] != a['end'] or b['t'] <= a['t'] for a, b in zip(es, es[1:])):
                raise ValueError(f'{boot} {key}: cold forward chain not contiguous')
            if es[-1]['end'] != u['prompt_tokens'] - 1:
                raise ValueError(f'{boot} {key}: incomplete prompt chain')
            requests.append(dict(boot=boot, key=key, serial=serial, prompt_tokens=u['prompt_tokens'],
                                 engine_s=u['prompt_time'], first_completion_t=es[0]['t'],
                                 last_completion_t=es[-1]['t'], first_chunk_unmeasurable=True))
            for a, b in zip(es, es[1:]):
                if not (a['pipeline'] and b['pipeline'] and b['end']-b['start'] == 2048):
                    continue
                # Coarse window checkpoint geometry of a cold solo job. The last
                # two chunks also terminate windows; show separately from interiors.
                coarse = b['start'] % 32768 == 0
                near_tail = b['end'] >= u['prompt_tokens']-1-4096
                rows.append(dict(boot=boot, key=key, start=b['start'], end=b['end'],
                                 ms=(b['t']-a['t'])*1000, coarse_restart=coarse,
                                 near_tail=near_tail, source_line=b['source_line']))
    bins = []
    for lo in range(0, 90112, 8192):
        selected = [r for r in rows if lo <= r['start'] < lo+8192 and not r['coarse_restart'] and not r['near_tail']]
        if selected:
            vals = sorted(r['ms'] for r in selected)
            bins.append(dict(start=lo, end=lo+8192, n=len(vals), median_ms=statistics.median(vals),
                             min_ms=min(vals), max_ms=max(vals), p90_ms=vals[int(.9*(len(vals)-1))]))
    return dict(sources=sources, requests=requests, rows=rows, bins=bins,
                restarts=[r for r in rows if r['coarse_restart']],
                note='post-forward completion cadence; first forward lacks a same-job preceding completion')


def merge_intervals(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b < a:
            raise ValueError('reversed GPU interval')
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


def interval_ms(intervals):
    return sum(b-a for a, b in merge_intervals(intervals))


def event_wrapper(torch, state, fn, category, device_fn, info_fn):
    """Current-stream envelopes; safe on worker threads; no internal join."""
    def wrapped(*a, **kw):
        d = device_fn(a, kw)
        if d.type != 'cuda':
            return fn(*a, **kw)
        stream = torch.cuda.current_stream(d)
        beg, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        info = info_fn(a, kw)
        with torch.cuda.device(d):
            beg.record(stream)
            host0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                host_ms = (time.perf_counter()-host0)*1000
                end.record(stream)
                state['events'].append((int(d.index), beg, end, category, info, host_ms))
    return wrapped


def decision(bands):
    """Pre-registered screening rule; shares are critical-card kernel times."""
    if len(bands) != 3 or any(not b['valid'] for b in bands):
        return 'INCONCLUSIVE: missing/inflated capture or placement mismatch'
    # Improvement is assumed locally, never inferred from the family share alone.
    scores = collections.defaultdict(list)
    for b in bands:
        shares = b['critical_shares']
        for family, fraction in [('moe_fat', .4), ('moe_thin', .3), ('attention', .3), ('gdn', .3)]:
            scores[family].append(shares.get(family, 0)*fraction)
        scores['host_or_pipeline'].append(b['both_idle_share']*.5)
    ranked = sorted(((statistics.mean(v), k) for k, v in scores.items()), reverse=True)
    if ranked[0][0] < .05:
        return 'STOP: no candidate clears 5% projected cold time saved'
    if ranked[0][0]-ranked[1][0] < .02:
        return 'TIE: inspect timeline/counterfactual before a kernel round'
    return f'NEXT={ranked[0][1]}; projected time saved={100*ranked[0][0]:.1f}% (screen, not measured gain)'


def kernel_family(name):
    if re.search(r'e3.*(gate_?up|down)', name, re.I):
        return 'moe_fat'
    if 'exl3_moe_kernel' in name:
        return 'moe_thin'
    if re.search(r'e3|rout|topk|expert|moe', name, re.I):
        return 'moe_other'
    if re.search(r'gated_delta|delta_rule|kkt|solve_tril|recompute_w_u|chunk_fwd|chunk_local|conv1d|causal_conv|gdn|l2norm', name, re.I):
        return 'gdn'
    if re.search(r'attn|qsa|flash|fmha|paged|lse', name, re.I):
        return 'attention'
    return 'dense_or_other'


def timeline(trace, forwards):
    es = trace.get('traceEvents', [])
    ranges = sorted((e for e in es if e.get('ph') == 'X' and e.get('cat') == 'user_annotation'
                     and e.get('name') == 'phase/job_prefill'), key=lambda e:e['ts'])
    if len(ranges) != len(forwards):
        raise ValueError('Job.prefill ranges/forward ledger mismatch (ignore phase/prefill_job wrapper)')
    gpu = [e for e in es if e.get('ph') == 'X' and e.get('cat','').lower() in ('kernel','gpu_memcpy','gpu_memset')]
    if not gpu:
        raise ValueError('no GPU activities: Kineto/CUPTI capture failed')
    rows = []
    for cpu, f in zip(ranges, forwards):
        if f['rows'] != 2048:
            continue
        lo, hi = cpu['ts'], cpu['ts']+cpu['dur']
        cards = collections.defaultdict(list)
        families = collections.defaultdict(lambda:collections.defaultdict(float))
        names = collections.Counter()
        for e in gpu:
            a, b = max(lo, e['ts']), min(hi, e['ts']+e['dur'])
            if b <= a:
                continue
            args = e.get('args', {})
            # Kineto's REAL PtoP memcpy schema uses inDevice/fromDevice/toDevice.
            d = str(args.get('device', args.get('inDevice', 'unknown')))
            cards[d].append((a/1000,b/1000))
            if e['cat'].lower() == 'kernel':
                families[d][kernel_family(e['name'])] += (b-a)/1000
                names[e['name']] += 1
        wall = (hi-lo)/1000
        busy = {d:interval_ms(iv) for d,iv in cards.items()}
        all_iv = [iv for intervals in cards.values() for iv in intervals]
        union = interval_ms(all_iv)
        critical = max(busy, key=busy.get) if busy else None
        valid = set(cards) == {'0','1'} and critical is not None and all(families[d]['moe_fat']>0 and families[d]['moe_thin']>0 for d in cards)
        rows.append(dict(start=f['kv_start'], rows=f['rows'], wall_ms=wall, per_card_busy_ms=busy,
                         both_idle_ms=max(0,wall-union), overlap_ms=max(0,sum(busy.values())-union),
                         both_idle_share=max(0,1-union/wall), critical_card=critical,
                         per_card_family_ms=dict(families), kernel_names=dict(names), valid=valid,
                         critical_shares={k:v/wall for k,v in families[critical].items()} if critical else {}))
    return rows


def analyze(root):
    root = Path(root)
    caps = sorted((root/'profile/prefill_b1_d3/ctx90113_cold').glob('cap*'))
    if len(caps) != 2:
        raise ValueError('expected two complete captures')
    data = []
    for cap in caps:
        with gzip.open(cap/'trace.json.gz','rt') as fh:
            rows = timeline(json.load(fh), json.loads((cap/'forwards.json').read_text()))
        kernels = json.loads((cap/'kernels.json').read_text())
        if kernels['cached'] != 0:
            raise ValueError('capture is not cold')
        event_file = json.loads((root/f'events-{cap.name}.json').read_text())
        events = event_file['events']
        if not all(any(e['category']=='stage0' and e.get('start')==d for e in events) for d in DEPTHS):
            raise ValueError('pipeline missing at required depths')
        actual = [next(r for r in rows if r['start']==d) for d in DEPTHS]
        # At 0 and 32k the requested chunk is a window fill. Decisions also
        # show the very next steady cadence (same KV depth to within 2048).
        interior = [next(r for r in rows if r['start']==d) for d in (2048,34816,81920)]
        selected_depths=set(DEPTHS+(2048,34816))
        envelopes=[dict(e,elapsed_ms=e['end_ms']-e['begin_ms']) for e in events if e.get('start') in selected_depths]
        data.append(dict(capture=cap.name, boundary_bands=actual, decision_bands=interior,
                         all_chunks=rows, cuda_envelopes=envelopes,memory=event_file['memory'],
                         decision=decision(interior)))
        if any(m['alloc_retries_delta'] for m in event_file['memory']):
            data[-1]['decision']='INCONCLUSIVE: allocator retries during capture'
    worker = json.loads((root/'profile/prefill_b1_d3/worker.json').read_text())
    timing = worker['results']['ctx90113_cold']
    timed = timing['timed']
    if len(timed) != 3 or timing.get('error') or timing.get('cache_mismatch') or any(r['cached'] for r in timed):
        raise ValueError('incomplete or non-cold timed records')
    timed_ms = statistics.median(r['x_ms'] for r in timed)
    for cap, d in zip(caps, data):
        k=json.loads((cap/'kernels.json').read_text())
        d['inflation'] = k['x_ms']/timed_ms
        if d['inflation'] > 1.15:
            d['decision'] = 'INCONCLUSIVE: profiler inflation >15%; use CUDA-event envelopes for the next measurement'
    if data[0]['decision'].split(';')[0] != data[1]['decision'].split(';')[0]:
        verdict = 'INCONCLUSIVE: captures disagree; inspect both retained timelines'
    else:
        verdict = data[0]['decision'].split(';')[0]
    return dict(captures=data, verdict=verdict, note='busy uses Kineto intervals; CUDA-event envelopes include launch starvation and are not kernel busy')


def gpu(args):
    """One load of the served stack; profile_decode --prefill entry path.

    Event pairs are recorded on the real current streams in BOTH host threads.
    Never synchronize a module/ext wrapper. Profiling remains process-local.
    Kineto traces from profile_decode are kept for actual kernel busy/idle checks.
    """
    import torch
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(d)!=(12,0) for d in range(2)):
        raise RuntimeError('expected two sm120 devices')
    spec = importlib.util.spec_from_file_location('r824_profile_decode', args.harness)
    pd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pd)
    from exllamav3.generator import prefill_pipeline as pp
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.cache.checkpoint_policy import interval
    args.out.mkdir(parents=True, exist_ok=True)
    original_load = pd.load_stack
    state = {'capture': 0, 'events': [], 'anchors': {}, 'undo': [], 'layout': None}
    original_install, original_remove = pd.Ranges.install, pd.Ranges.remove

    def load(a, b, depth, cache_tokens, metadata):
        # The R804 harness's b=1 load changed the layer boundary. Load b=8,
        # pool=901120, history=3 as the daily, but enqueue only ONE job.
        result = original_load(a, 8, depth, cache_tokens, metadata)
        gen = result[-1]
        if gen.recurrent_cache is None:
            raise RuntimeError('recurrent cache missing')
        if gen.max_chunk_size != 2048 or gen.num_draft_tokens != 3:
            raise RuntimeError('daily chunk/draft mismatch')
        model = result[0]
        layout = pp._layout(model)
        if layout is None:
            raise RuntimeError('served LS pipeline layout unavailable')
        stages = [[(m.key, inst, idx) for m, inst, idx in stage] for stage in layout[1]]
        state['layout'] = stages
        dump(args.out/'placement.json', dict(stages=stages, model_load_max_batch=8, pool=cache_tokens,
                                             pp_policy=[interval(gen, x, 90112) for x in (2048,32768,81920,88064)]))
        # Exact module partition is checked against an operator-captured daily
        # layout, not merely against use_per_device=[30,30].
        expected = json.loads(Path(args.daily_layout).read_text())['stages']
        if json.loads(json.dumps(stages)) != expected:
            raise RuntimeError('target module placement differs from LIVE daily')
        # R818 launcher warm-up before any shapes/timings. These are the exact
        # served raw completion strings and their calibrated tokenizer lengths.
        from exllamav3 import Job, GreedySampler
        for n, body in [(20,'Note 04560: train fire door sand square fire snow ocean wind house cloud.'),
                        (29,'Note 12479: number fish metal horse bridge copper castle apple leaf music island square rain chair copper snow seed corner floor village.'),
                        (11,'Note 07311: bridge music.')]:
            ids = result[-2].encode(body, add_bos=False)
            if ids.shape[-1] != n:
                raise RuntimeError('FASTWARM tokenizer identity')
            job = Job(input_ids=ids, max_new_tokens=8, sampler=GreedySampler(), stop_conditions=[], identifier=-n)
            gen.enqueue(job)
            for _ in range(64):
                res = gen.iterate()
                if any(r['stage']=='error' for r in res):
                    raise RuntimeError('FASTWARM failed')
                if not gen.num_active_jobs() and not gen.num_pending_jobs():
                    break
            else:
                raise RuntimeError('FASTWARM did not finish')
            pd.sync(torch)
        return result

    def event_call(fn, category, device_fn, info_fn):
        return event_wrapper(torch,state,fn,category,device_fn,info_fn)

    def install(self):
        original_install(self)
        state['capture'] += 1
        state['events'] = []
        state['memory_before'] = {d:torch.cuda.memory_stats(d).get('num_alloc_retries',0) for d in range(2)}
        for d in range(torch.cuda.device_count()):
            with torch.cuda.device(d):
                torch.cuda.synchronize(d)
                anchor = torch.cuda.Event(enable_timing=True)
                anchor.record()
                state['anchors'][d] = anchor
        def set_attr(obj, attr, value):
            state['undo'].append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, value)
        # Stage envelopes retain the producer's true depth, even when the
        # lookahead executes before Job has committed the preceding chunk.
        set_attr(pp._Runtime, '_stage0', event_call(pp._Runtime._stage0, 'stage0',
            lambda a,k: a[0].devices[0], lambda a,k: {'start':int(a[2]['cache_seqlens'][0]), 'rows':2048}))
        set_attr(pp._Runtime, '_modules', event_call(pp._Runtime._modules, 'stage_modules',
            lambda a,k: a[3][0][0].device,
            lambda a,k: {'start':int(a[2]['cache_seqlens'][0]), 'keys':[m.key for m,_,_ in a[3]]}) )
        # These leaf families exclude TransformerBlock and shared experts
        # nested inside BlockSparseMLP; never sum parent and child durations.
        categories = {'BlockSparseMLP':'moe', 'Attention':'attention', 'GatedDeltaNet':'gdn',
                      'GatedResidual':'hc', 'HyperConnection':'hc', 'PLELayer':'ple'}
        seen = set()
        todo = list(self.model.modules)
        while todo:
            m = todo.pop()
            if id(m) in seen:
                continue
            seen.add(id(m))
            todo.extend(getattr(m, 'modules', None) or [])
            cat = categories.get(type(m).__name__)
            if cat and m.device.type == 'cuda':
                def info(a,k,m=m):
                    params = a[1] if len(a)>1 else k['params']
                    return dict(start=int(params['cache_seqlens'][0]), key=m.key)
                set_attr(m, 'forward', event_call(m.forward, cat, lambda a,k,m=m:m.device, info))
        for name, cat in [('exl3_moe_prefill_e3_det','moe_fat'), ('exl3_moe','moe_thin'),
                          ('exl3_moe_prefill_e3_det_reduce','moe_reduce')]:
            set_attr(ext, name, event_call(getattr(ext,name), cat, lambda a,k:a[0].device,
                                           lambda a,k: {}))

    def remove(self):
        # pd.run_job syncs BOTH GPUs before Ranges.remove(). Event elapsed
        # reads occur only here, outside the measured interval.
        records = []
        for d, beg, end, category, info, host_ms in state['events']:
            anchor = state['anchors'][d]
            records.append(dict(device=d, begin_ms=anchor.elapsed_time(beg), end_ms=anchor.elapsed_time(end),
                                category=category, host_ms=host_ms, **info))
        memory = []
        for d in range(2):
            free,total=torch.cuda.mem_get_info(d)
            memory.append(dict(device=d,driver_free=free,total=total,allocated=torch.cuda.memory_allocated(d),
                               reserved=torch.cuda.memory_reserved(d),
                               alloc_retries_delta=torch.cuda.memory_stats(d).get('num_alloc_retries',0)-state['memory_before'][d]))
        dump(args.out/f'events-cap{state["capture"]}.json', dict(events=records, depths=DEPTHS,
                                                                placement=state['layout'],memory=memory))
        while state['undo']:
            obj, attr, old = state['undo'].pop()
            setattr(obj, attr, old)
        original_remove(self)

    pd.load_stack, pd.Ranges.install, pd.Ranges.remove = load, install, remove
    a = argparse.Namespace(model=args.model, out=args.out/'profile/prefill_b1_d3', batch=[1], draft=3, tokens=64,
        contexts=[90113], warmup_steps=8, settle_steps=8, max_chunk_size=2048, cache_quant='8,8',
        cache_tokens=901120, tensor_parallel=False, gpu_split='30,30', draft_gpu_split='0,32',
        prompt_file=args.prompt_file, prefill=True, one_process=True, prefix_tokens=0,
        timed_reps=3, capture_reps=2, phase_ranges=True, module_ranges=True, worker=True)
    a.out.mkdir(parents=True, exist_ok=True)
    pd.prefill_multi(a)
    print('R824 GPU capture complete; analyze event envelopes plus retained Kineto traces', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('curve')
    c.add_argument('--raw', type=Path, required=True)
    c.add_argument('--boots', nargs='+', default=['A1','A2','D1','D2'])
    c.add_argument('--out', type=Path, required=True)
    g = sub.add_parser('gpu')
    g.add_argument('--model', required=True)
    g.add_argument('--out', type=Path, required=True)
    g.add_argument('--harness', type=Path, required=True)
    g.add_argument('--daily-layout', type=Path, required=True)
    g.add_argument('--prompt-file', type=Path, required=True)
    report = sub.add_parser('analyze')
    report.add_argument('--root', type=Path, required=True)
    report.add_argument('--out', type=Path, required=True)
    pin = sub.add_parser('identity')
    pin.add_argument('--launcher', type=Path, required=True)
    pin.add_argument('--inspect', type=Path, required=True)
    pin.add_argument('--config', type=Path, required=True)
    pin.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'curve':
        data = curve(a.raw, a.boots)
        dump(a.out, data)
        print(f'{len(data["requests"])} solo cold requests, {len(data["rows"])} completion deltas')
        for b in data['bins']:
            print(f'{b["start"]:5}-{b["end"]:5}: n={b["n"]:3} median={b["median_ms"]:.3f} ms')
    elif a.command == 'gpu':
        gpu(a)
    elif a.command == 'analyze':
        result = analyze(a.root)
        dump(a.out, result)
        print(result['verdict'])
    else:
        dump(a.out, identity(a.launcher,a.inspect,a.config))
        print('R824 LIVE identity PASS')


if __name__ == '__main__':
    main()
