#!/usr/bin/env python3
"""Recompute R827b/R828/R828c from published request rows and per-request counters.

No endpoint or daemon access. Counter ratios are ratios of sums; speed uses one
median per boot. Warmups marked run=-1 are excluded. Historical decisions remain
separate from the prospective eight-pair c3 confirmation.
"""
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent / 'results'
ROUNDS = {
    'r827b': '2026-10-02-r827-prompt-lookup-ygmGkW',
    'r828': '2026-10-02-r828-prompt-lookup-r3-em6iO4',
    'r828c': '2026-10-02-r828c-c3-confirm-d8bApo',
}
TAGS = ('OFF1', 'ON1', 'ON2', 'OFF2')


def rows(path):
    data = [json.loads(line) for line in path.read_text().splitlines()]
    assert data, path
    for row in data:
        assert row['ok'] and row['client_frames'] <= row['completion_tokens'], path
        assert math.isfinite(row['decode_tps']) and row['decode_tps'] > 0, path
        finish = row.get('finish_reason', row.get('value', {}).get('finish'))
        assert finish == 'length', (path, finish)
    return [row for row in data if row['run'] >= 0]


def counters(path):
    data = json.loads(path.read_text())
    rr = data['counters']
    assert len(rr) == data['measured_n']
    ss = lambda key: sum(r.get(key, 0) for r in rr)
    steps = ss('decode_steps')
    generated = ss('gen_tokens')
    accepted = generated - steps
    if 'lookup_accept_mtp_0' in rr[0]:
        by_source = sum(ss(f'lookup_accept_{source}_{position}')
                        for source in ('mtp', 'copy') for position in range(3))
        assert generated == steps + by_source
    assert ss('lookup_accepted') <= accepted
    assert ss('lookup_proposed') == 2 * ss('lookup_hits')
    if 'lookup_on_steps' in rr[0]:
        assert ss('lookup_on_steps') + ss('lookup_probe_steps') + ss('lookup_off_steps') == steps
        assert ss('lookup_skipped_forwards') == ss('lookup_proposed')
    values = {key: ss(key) for key in rr[0] if key.startswith('lookup_')}
    values.update(steps=steps, generated=generated,
                  ms_step=1000 * ss('gen_time') / steps, tokens_step=generated / steps,
                  copy_acceptance=(ss('lookup_accepted') / ss('lookup_proposed') if ss('lookup_proposed') else None))
    return values


def identity(root, suites):
    reference = {}
    for suite in suites:
        for line in (root / f'identity-OFF1-{suite}.jsonl').read_text().splitlines():
            r = json.loads(line); reference[(suite, r['pid'])] = r
    result = {}
    for tag in ('ON1', 'ON2', 'OFF2'):
        counts = {}
        for suite in suites:
            equal = 0; total = 0
            for line in (root / f'identity-{tag}-{suite}.jsonl').read_text().splitlines():
                row = json.loads(line); ref = reference[(suite, row['pid'])]
                assert row['request_sha256'] == ref['request_sha256']
                equal += row['value'] == ref['value']; total += 1
            counts[suite] = dict(equal=equal, total=total)
        result[tag] = counts
    return result


def abba(name):
    root = ROOT / ROUNDS[name]
    result = dict(shapes={}, identity=identity(root, ('fn', 'chat', 'agent') + (('fnstyle',) if name == 'r828' else ())))
    for shape in ('c1-code', 'c1-prose', 'c2-code', 'c2-prose', 'c3-code', 'c3-prose', 'agent-c1'):
        rates = {}; mechanism = {}; prompts = set()
        for tag in TAGS:
            path = root / (f'speed-{tag}-agent.jsonl' if shape == 'agent-c1' else f'fn-{tag}-{shape}.jsonl')
            data = rows(path)
            length = 2048 if shape == 'agent-c1' else 1024
            assert len(data) == (36 if shape == 'agent-c1' else 3 * int(shape[1]))
            assert all(r['completion_tokens'] == length for r in data)
            prompts.update(r['prompt_tokens'] for r in data)
            rates[tag] = st.median(r['decode_tps'] for r in data)
            mechanism[tag] = counters(root / f'counters-{tag}-{shape}.json')
        gains = {f'{on}/{off}': 100 * (rates[on] / rates[off] - 1)
                 for on, off in [('ON1','OFF1'), ('ON2','OFF2')] +
                 ([('ON1','OFF2'), ('ON2','OFF1')] if shape == 'agent-c1' else [])}
        result['shapes'][shape] = dict(medians=rates, gains_pct=gains, prompt_tokens=sorted(prompts), counters=mechanism)
    return result


def confirmation():
    root = ROOT / ROUNDS['r828c']; pairs = []
    for i in range(1, 9):
        rates = {}; mechanism = {}
        for arm in ('OFF','ON'):
            tag = f'P{i:02d}_{arm}'
            data = rows(root / f'fn-{tag}-c3-code.jsonl')
            assert len(data) == 9 and all(r['conc'] == 3 and r['completion_tokens'] == 1024 and
                r['temperature'] == 0 and r['prompt_tokens'] == 118 for r in data)
            rates[arm] = st.median(r['decode_tps'] for r in data)
            mechanism[arm] = counters(root / f'counters-{tag}-c3-code.json')
        pairs.append(dict(pair=i, order='OFF/ON' if i % 2 else 'ON/OFF', medians=rates,
                          g=math.log(rates['ON']/rates['OFF']), counters=mechanism))
    gains = [p['g'] for p in pairs]; mean = st.mean(gains); sd = st.stdev(gains)
    lower = mean - 1.8946 * sd / math.sqrt(8); floor = math.log(.99)
    return dict(pairs=pairs, mean=mean, sample_sd=sd, lower_bound=lower, floor=floor,
                decision='PASS' if lower > floor else 'FAIL')


def main():
    result = {name: abba(name) for name in ('r827b','r828')}
    result['r828c'] = confirmation()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
