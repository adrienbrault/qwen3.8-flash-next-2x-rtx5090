#!/usr/bin/env python3
"""R828c prospective c3 confirmation; no daemon, HTTP, or host mutations."""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import gate

N = 8
T = 1.8946
FLOOR = math.log(0.99)
ORDER = tuple((f'P{i:02d}_OFF', f'P{i:02d}_ON') if i % 2 else
              (f'P{i:02d}_ON', f'P{i:02d}_OFF') for i in range(1, N + 1))
ASSUMPTION = 'One-sided 95% paired-t bound assumes independent, approximately normal boot-pair log gains; small-sample power is not established.'


def rule(on, off):
    assert len(on) == len(off) == N, 'exactly eight new pairs required'
    assert all(math.isfinite(x) and x > 0 for x in on + off), 'invalid boot rate'
    gains = [math.log(a / b) for a, b in zip(on, off)]
    mean = statistics.mean(gains)
    sd = statistics.stdev(gains)  # sample sd, denominator N-1
    bound = mean - T * sd / math.sqrt(N)
    return dict(g=gains, mean=mean, sd=sd, lower_bound=bound, floor=FLOOR,
                n=N, t=T, DECISION='PASS' if bound > FLOOR else 'FAIL', assumption=ASSUMPTION)


def digest(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob('*')):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode() + b'\0' + p.read_bytes())
    return h.hexdigest()


def declare(root):
    root = Path(root)
    gate.dump(root / 'order.json', dict(order=ORDER, n=N, t=T, floor=FLOOR,
                                     rule='mean(log(ON/OFF))-1.8946*sample_sd/sqrt(8) > log(0.99)',
                                     assumption=ASSUMPTION))
    print('DECLARED ORDER: ' + ' ; '.join(' -> '.join(pair) for pair in ORDER))


def boot_rows(root, tag):
    rows = gate.read_rows(Path(root) / f'fn-{tag}-c3-code.jsonl')
    assert len(rows) == 9
    assert {(r['run'], r['i']) for r in rows} == {(r, i) for r in range(3) for i in range(3)}
    assert all(r['completion_tokens'] == r['max_tokens'] == r['min_tokens'] == 1024 and
               r['conc'] == 3 and r['tag'] == f'{tag}-c3-code' and
               r['temperature'] == 0 and r['ctx_requested'] == 0 and
               math.isfinite(r['decode_tps']) for r in rows), 'c3 request geometry drift'
    return statistics.median(r['decode_tps'] for r in rows)


def metrics(data, flag):
    assert data['n'] == 15 and data['measured_n'] == 9 and len(data['counters']) == 9
    rows = data['counters']
    for row in rows:
        gate.validate_counter(row)
    steps = sum(r['decode_steps'] for r in rows)
    matches = sum(r['lookup_match_' + b] for r in rows for b in ('3_7', '8_15', '16_31', '32_plus'))
    # Match buckets also include OFF sparse samples, which use an existing
    # whole-window readback. With PROBE-only c3, matches == added opener reads.
    # If residency changes in a valid new boot, retain it and publish bounds.
    reprobes = sum(r['lookup_reprobe_checks'] for r in rows)
    reads = [max(0, matches - reprobes), matches]
    if flag == '0':
        reads = [0, 0]
    return dict(checks=sum(r['lookup_checks'] for r in rows), opener_reads=reads,
                ineligible_steps=sum(r['lookup_ineligible_steps'] for r in rows),
                ms_step=1000 * sum(r['gen_time'] for r in rows) / steps,
                tokens_step=sum(r['gen_tokens'] for r in rows) / steps,
                steps=steps, matches=matches, reprobe_checks=reprobes)


def parse_boot(root, tag, out):
    root = Path(root)
    archived = json.loads((root / f'counters-{tag}-c3-code.json').read_text())
    flag = '0' if 'OFF' in tag else '1'
    assert len(archived['serials']) == 15 and len(set(archived['serials'])) == 15
    with Path(out).with_suffix('.txt').open('w') as log, contextlib.redirect_stdout(log):
        gate.cell(root / f'container-{tag}-c3-code.log', min(archived['serials']) - 1,
                  15, flag, out, measured=9)
    parsed = json.loads(Path(out).read_text())
    assert parsed == archived, 'raw counter replay differs from captured cell'
    return dict(median=boot_rows(root, tag), **metrics(parsed, flag))


def replay(source, out):
    """Preserve R828 REJECT, recheck inherited gates, replay real c3 logs."""
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Freeze served code, clients/prompts, parsers and full image ID against B.
    files = [gate.P / name for name in ('IMAGE_ID.env', 'gate.py', 'promote.py',
             'fn_bench.py', 'fn_driver.py', 'fn_greedy.py', 'chat_greedy.py',
             'install.py', 'selftest.py', 'Dockerfile')]
    files += [p for d in ('overlay', 'fixtures') for p in (gate.P / d).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    for path in files:
        rel = path.relative_to(gate.P)
        assert path.read_bytes() == (source / 'packet' / rel).read_bytes(), f'candidate/prompt/parser drift: {rel}'
    gate.dump(out / 'frozen-files.json', {str(p.relative_to(gate.P)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    # All files report() needs, plus real raw c3 logs. No fabricated counters.
    for pattern in ('identity-*.jsonl', 'fn-*.jsonl', 'speed-*-agent.jsonl', 'counters-*.json', 'container-*-c3-code.log'):
        for path in source.glob(pattern):
            shutil.copyfile(path, out / path.name)
    shutil.copyfile(source / 'decision.json', out / 'original-decision.json')
    with (out / 'inherited-gates.txt').open('w') as log, contextlib.redirect_stdout(log):
        result = gate.report(out)
    assert result == json.loads((out / 'original-decision.json').read_text()), 'R828 decision does not reproduce'
    assert result['DECISION'] == 'REJECT' and result['identity']['pass']
    assert all(s['pass_gate'] for name, s in result['shapes'].items() if name != 'c3-code'), 'inherited gate failed'
    assert not result['shapes']['c3-code']['pass_gate'], 'original REJECT must remain recorded'
    readings = {}
    for tag in gate.TAGS:
        readings[tag] = parse_boot(out, tag, out / f'replayed-{tag}.json')
    gate.dump(out / 'replay.json', readings)
    print(json.dumps(readings, indent=2))
    print('REAL R828 REPLAY PASS; original REJECT retained; other registered gates PASS')
    return readings


def boot_check(root, tag):
    root = Path(root)
    start = json.loads((root / f'inspect-{tag}.json').read_text())[0]
    end = json.loads((root / f'inspect-{tag}-end.json').read_text())[0]
    assert start['Id'] == end['Id'], 'container changed during boot'
    assert start['RestartCount'] == end['RestartCount'] == 0
    for path in root.glob('inspect-P??_*.json'):
        if path.name.endswith('-end.json') or path.name == f'inspect-{tag}.json':
            continue
        assert json.loads(path.read_text())[0]['Id'] != start['Id'], 'paired boots must be independent'


def report(root):
    root = Path(root)
    declaration = json.loads((root / 'order.json').read_text())
    assert declaration['order'] == [list(p) for p in ORDER], 'order differs from registration'
    expected = {f'fn-{tag}-c3-code.jsonl' for p in ORDER for tag in p}
    assert {p.name for p in root.glob('fn-P*-c3-code.jsonl')} == expected, 'missing or extra confirmatory boots'
    pairs, on, off = [], [], []
    for i, tags in enumerate(ORDER, 1):
        values = {}
        for tag in tags:
            boot_check(root, tag)
            values[tag.rsplit('_', 1)[1]] = parse_boot(root, tag, root / f'replayed-{tag}.json')
        pairs.append(dict(pair=i, order=tags, **values))
        on.append(values['ON']['median']); off.append(values['OFF']['median'])
    result = rule(on, off)
    for pair, gain in zip(pairs, result['g']):
        pair['g'] = gain
    result['pairs'] = pairs
    gate.dump(root / 'decision.json', result)
    print('| Pair | Order | ON t/s | OFF t/s | g | Checks ON/OFF | Opener reads ON/OFF | Ineligible ON/OFF | ms/step ON/OFF | tokens/step ON/OFF |')
    print('|---|---|---:|---:|---:|---|---|---|---|---|')
    def reads(v):
        a, b = v['opener_reads']
        return str(a) if a == b else f'[{a},{b}]'
    for p in pairs:
        a, b = p['ON'], p['OFF']
        print(f"| {p['pair']} | {' -> '.join(p['order'])} | {a['median']:.6f} | {b['median']:.6f} | {p['g']:.9f} | "
              f"{a['checks']}/{b['checks']} | {reads(a)}/{reads(b)} | {a['ineligible_steps']}/{b['ineligible_steps']} | "
              f"{a['ms_step']:.6f}/{b['ms_step']:.6f} | {a['tokens_step']:.6f}/{b['tokens_step']:.6f} |")
    print(f"mean(g)={result['mean']:.12g}; sample sd(g)={result['sd']:.12g}; lower bound={result['lower_bound']:.12g}; log(0.99)={FLOOR:.12g}")
    print(ASSUMPTION)
    print('Opener reads are inferred valid-match readbacks; intervals exclude unseparated OFF sparse matches. All counters use nine measured streams, ratios of sums.')
    print('DECISION ' + result['DECISION'] + '; original R828 REJECT unchanged')
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    for cmd in ('declare', 'report'):
        sub.add_parser(cmd).add_argument('--root', required=True)
    p = sub.add_parser('replay'); p.add_argument('--source', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('boot'); p.add_argument('--root', required=True); p.add_argument('--tag', required=True)
    sub.add_parser('digest').add_argument('--root', required=True)
    args = vars(parser.parse_args()); cmd = args.pop('cmd')
    if cmd == 'digest': print(digest(args['root']))
    else: {'declare': declare, 'report': report, 'replay': replay, 'boot': boot_check}[cmd](**args)


if __name__ == '__main__':
    main()
