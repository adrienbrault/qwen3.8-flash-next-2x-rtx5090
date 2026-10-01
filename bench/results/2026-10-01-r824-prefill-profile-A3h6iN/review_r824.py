#!/usr/bin/env python3
"""Independent raw replay; no GPU imports. Exact kernel names and launch attribution."""
import argparse
import bisect
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import r824_probe as R


def corrected_family(name):
    family = R.kernel_family(name)
    if family == 'attention':
        if not re.search(r'attn|qsa|dsa|flash|fmha|cache_paged|(?:^|_)lse(?:_|$)', name, re.I):
            return 'dense_or_other'
    if name.startswith('_dsa_'):
        return 'attention'
    return family


def probe_events(path):
    # probe.log has NO Docker timestamp prefix; server container logs do.
    events = []
    for n, line in enumerate(Path(path).read_text(errors='replace').splitlines(), 1):
        if '[R823-cache] ' not in line:
            continue
        value = json.loads(line.split('[R823-cache] ', 1)[1])
        if value.get('v') != 1 or not isinstance(value.get('t'), (int, float)):
            raise ValueError(f'{path}:{n}: invalid event')
        events.append(dict(value, source_line=n))
    if not events:
        raise ValueError('no trace events')
    return events


def cadence(path):
    groups = collections.defaultdict(list)
    for e in probe_events(path):
        if e['event'] == 'prefill_forward' and e['prompt_prefill']:
            groups[(e['run'], e.get('key'), e['serial'])].append(e)
    rows = []
    for key, es in groups.items():
        if es[0]['start'] != 0 or es[-1]['end'] != 90112:
            continue
        if any(a['end'] != b['start'] or b['t'] <= a['t'] for a, b in zip(es, es[1:])):
            raise ValueError('broken cold chain')
        ds = [dict(start=b['start'], end=b['end'], pipeline=b['pipeline'],
                   ms=(b['t']-a['t'])*1000) for a, b in zip(es, es[1:])]
        steady = statistics.median(d['ms'] for d in ds if d['ms'] < 250)
        slow = [d for d in ds if d['ms'] > 250]
        rows.append(dict(identity=key, steady_median_ms=steady, slow=slow,
                         excess_ms=sum(d['ms']-steady for d in slow)))
    return rows


def decomposition(trace_path, forwards_path):
    with gzip.open(trace_path, 'rt') as f:
        events = json.load(f)['traceEvents']
    ranges = sorted((e for e in events if e.get('ph') == 'X' and
                     e.get('cat') == 'user_annotation' and e['name'] == 'phase/job_prefill'),
                    key=lambda e: e['ts'])
    fw = json.loads(forwards_path.read_text())
    assert len(ranges) == len(fw)
    launches = {e['args']['correlation']: e for e in events if e.get('cat') == 'cuda_runtime'
                and 'correlation' in e.get('args', {})}
    # Attribution is the launch CPU thread/range, never the GPU execution timestamp.
    mods = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in events:
        if e.get('cat') == 'user_annotation' and e.get('ph') == 'X' and e['name'].startswith('mod/'):
            mods[e['tid']][e['name']].append((e['ts'], e['ts']+e['dur']))
    for thread in mods.values():
        for intervals in thread.values():
            intervals.sort()
    rows = []
    for band, f in zip(ranges, fw):
        if f['kv_start'] not in (2048, 34816, 81920):
            continue
        lo, hi = band['ts'], band['ts']+band['dur']
        names = collections.defaultdict(collections.Counter)
        origins = collections.defaultdict(collections.Counter)
        for e in events:
            if e.get('cat') != 'kernel' or e.get('ph') != 'X':
                continue
            duration = max(0, min(hi, e['ts']+e['dur'])-max(lo, e['ts']))/1000
            if not duration:
                continue
            d, name = str(e['args']['device']), e['name']
            names[d][name] += duration
            launch = launches.get(e['args'].get('correlation'))
            origin = 'unattributed'
            if launch:
                matches = []
                for mod, intervals in mods[launch['tid']].items():
                    i = bisect.bisect_right(intervals, (launch['ts'], float('inf')))-1
                    if i >= 0 and intervals[i][0] <= launch['ts'] <= intervals[i][1]:
                        matches.append((intervals[i][1]-intervals[i][0], mod))
                if matches:
                    origin = min(matches)[1]
            if corrected_family(name) == 'dense_or_other':
                origins[d][origin] += duration
            if origin == 'mod/Attention' and re.search(r'gemm|gemv|cutlass::Kernel', name, re.I):
                origins[d]['attention_projection_gemm'] += duration
        corrected = {d: dict(collections.Counter()) for d in names}
        for d, ns in names.items():
            for n, v in ns.items():
                fam = corrected_family(n)
                corrected[d][fam] = corrected[d].get(fam, 0)+v
        rows.append(dict(start=f['kv_start'], wall_ms=band['dur']/1000,
                         names={d: dict(v) for d, v in names.items()}, corrected=corrected,
                         dense_origin={d: dict(v) for d, v in origins.items()}))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    r = R.analyze(a.raw/'gpu')
    for tag in ('pre', 'locked', 'restored'):
        R.identity(a.raw/'launch-live.sh', a.raw/f'inspect-{tag}.json', a.raw/f'config-{tag}.yml')
    assert not (a.raw/'VOID').exists()
    assert (a.raw/'clocks-start.json').read_bytes() == (a.raw/'clocks-end.json').read_bytes()
    r['cadence'] = cadence(a.raw/'probe.log')
    for c in r['captures']:
        cap = a.raw/'gpu/profile/prefill_b1_d3/ctx90113_cold'/c['capture']
        c['decomposition'] = decomposition(cap/'trace.json.gz', cap/'forwards.json')
        corrected_bands = []
        for band, row in zip(c['decision_bands'], c['decomposition']):
            corrected_bands.append(dict(band, critical_shares={k: v/band['wall_ms']
                for k, v in row['corrected'][band['critical_card']].items()}))
        c['corrected_decision'] = R.decision(corrected_bands)
        c['scores_original_pct'] = {k: 100*statistics.mean(b['critical_shares'].get(k,0)*v
            for b in c['decision_bands']) for k, v in
            [('moe_fat',.4),('moe_thin',.3),('attention',.3),('gdn',.3)]}
        c['scores_corrected_pct'] = {k: 100*statistics.mean(b['critical_shares'].get(k,0)*v
            for b in corrected_bands) for k, v in
            [('moe_fat',.4),('moe_thin',.3),('attention',.3),('gdn',.3)]}
    r['provenance'] = {str(p.relative_to(a.raw)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [a.raw/'probe.log', a.raw/'launch-live.sh',
                  *a.raw.glob('gpu/profile/prefill_b1_d3/ctx90113_cold/cap*/trace.json.gz')]}
    a.out.write_text(json.dumps(r, indent=2)+'\n')
    print(r['verdict'])
    for c in r['captures']:
        print(c['capture'], c['scores_original_pct'], c['scores_corrected_pct'], c['corrected_decision'])
        for row in c['decomposition']:
            print(row['start'], row['dense_origin'])
    print('cadence runs', len(r['cadence']), 'excess ms', [round(x['excess_ms'],3) for x in r['cadence']])

if __name__ == '__main__':
    main()
