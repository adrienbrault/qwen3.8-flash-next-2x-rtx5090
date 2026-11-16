#!/usr/bin/env python3
"""decodekinds-r1 analysis.
  analyze.py control <reasoning.jsonl>   exit 1 unless c1 thinking-on decode is within 3 % of 2026-10-23 (248.7 / 251.4)
  analyze.py report <results dir>        Markdown summary of both passes
Decode rate = token-weighted: sum(completion - 1) / sum(decode window), i.e. the solid-line definition in the
README chart, so a category with many short answers is not inflated by medians of per-request rates."""
import glob
import json
import statistics as st
import sys
from pathlib import Path

REF = {'code': 248.7, 'prose': 251.4}


def control(path):
    rows = [json.loads(l) for l in open(path)]
    bad = 0
    for kind, ref in REF.items():
        rs = [r for r in rows if r.get('tag', '').endswith('reasoning-' + kind) and r.get('ok') and r.get('decode_tps')]
        assert len(rs) >= 3, (kind, len(rs))
        assert all(r['completion_tokens'] == 1024 for r in rs), 'forced length not reached'
        med = st.median(r['decode_tps'] for r in rs)
        ok = abs(med / ref - 1) <= 0.03
        bad += not ok
        print(f'CONTROL {kind}: median {med:.1f} t/s vs 2026-10-23 {ref} ({100 * (med / ref - 1):+.1f} %) {"OK" if ok else "OUT OF BAND"}')
    sys.exit(1 if bad else 0)


def bench(out, tag, cats):
    res = json.load(open(f'{out}/{tag}.json'))
    samples = [l.split('\t') for l in open(f'{out}/{tag}.samples.tsv').read().splitlines()[1:]]
    audit = {}
    for l in open(f'{out}/{tag}.requests.jsonl'):
        a = json.loads(l)
        audit.setdefault(a['prompt_sha'], []).append(a)
    # bench results are in sample order; map each request to its category through the prompt sha
    shas = [s[1] for s in samples]
    assert len(shas) == len(res['output_lens']) == len(res['itls']), (len(shas), len(res['output_lens']))
    per = {}
    for sha, n, itl, ttft in zip(shas, res['output_lens'], res['itls'], res['ttfts']):
        c = cats[sha]
        d = per.setdefault(c, dict(tok=0, dec=0.0, n=0, outs=[], ttft=[]))
        if n and n > 1 and sum(itl) > 0:
            d['tok'] += n - 1
            d['dec'] += sum(itl)
        d['n'] += 1
        d['outs'].append(n or 0)
        d['ttft'].append(ttft)
    # validity from the shim's per-request audit
    reasoning = sum(a.get('reasoning_frames', 0) for v in audit.values() for a in v)
    failed = sum(not a.get('success') for v in audit.values() for a in v)
    finish = {}
    acc = {}
    for sha, v in audit.items():
        c = cats[sha]
        for a in v:
            finish.setdefault(c, {}).setdefault(a.get('finish_reason'), 0)
            finish[c][a.get('finish_reason')] += 1
            cd = a.get('completion_details') or {}
            x = acc.setdefault(c, [0, 0])
            x[0] += cd.get('accepted_prediction_tokens') or 0
            x[1] += cd.get('rejected_prediction_tokens') or 0
    return per, finish, acc, reasoning, failed, res


def report(root):
    root = Path(root)
    cats = json.load(open(root / 'dataset/categories.json'))
    print(f'# decodekinds-r1 summary ({root.name})\n')
    print('Client rates here are `(n - 1) / (latency - TTFT)`. They overstate answers of a few decode steps: the '
          "stream's last frame arrives ~0.1 ms after the one before it, so one step's tokens are counted without "
          'its time (+43 % on 26-token translations, +10-15 % on MBPP and extraction, 0-3 % above 150 tokens). '
          'The published per-category rates are step-based: generated tokens / decode steps from '
          '`server-steps.jsonl` (server_steps.py) divided by the mean inter-frame interval (bench/plot.py).\n')
    for p in ('A', 'B'):
        pd = root / p
        if not pd.exists():
            continue
        print(f'## pass {p}\n')
        rows = [json.loads(l) for l in open(pd / 'reasoning.jsonl')]
        for kind in REF:
            rs = [r['decode_tps'] for r in rows if r.get('tag', '').endswith('reasoning-' + kind) and r.get('decode_tps')]
            print(f'- reasoning (thinking on, 1,024 forced, c1) {kind}: median {st.median(rs):.1f} t/s (n={len(rs)})')
        for shape in ('coding', 'structured', 'prose'):
            f = pd / 'mia' / f'{shape}.json'
            if f.exists():
                m = json.load(open(f))
                runs = m['runs']
                det = [r['usage'].get('completion_tokens_details') or {} for r in runs]
                print(f'- MiaAI {shape} (thinking off, c1, 5 x cap 400): median {m["tok_s_median"]:.1f} t/s, '
                      f'tokens {[r["completion_tokens"] for r in runs]}, finish {sorted({r["finish_reason"] for r in runs})}, '
                      f'<think> in text {sum("<think>" in (r.get("text") or "") for r in runs)}, '
                      f'accepted/rejected drafts {sum(d.get("accepted_prediction_tokens") or 0 for d in det)}/'
                      f'{sum(d.get("rejected_prediction_tokens") or 0 for d in det)}')
        for c in (1, 4):
            tag = f'{p}-c{c}'
            if not (pd / 'bench' / f'{tag}.json').exists():
                continue
            per, finish, acc, reasoning, failed, res = bench(pd / 'bench', tag, cats)
            print(f'\n### vllm bench, thinking off, natural length (cap 1,024), c{c}: {res["completed"]} ok, {failed} failed, '
                  f'reasoning frames {reasoning}\n')
            print('| category | n | decode t/s per stream (token-weighted) | mean output tokens | TTFT p50 ms | finish | drafts accepted / (accepted + rejected) |')
            print('|---|---:|---:|---:|---:|---|---:|')
            allt = alld = 0
            for cat in sorted(per):
                d = per[cat]
                allt += d['tok']; alld += d['dec']
                a, r = acc.get(cat, [0, 0])
                rate = f'{a / (a + r):.2f}' if a + r else '-'
                print(f'| {cat} | {d["n"]} | {d["tok"] / d["dec"]:.1f} | {st.mean(d["outs"]):.0f} | '
                      f'{1000 * st.median(d["ttft"]):.0f} | {finish.get(cat)} | {rate} |')
            print(f'| **all** | {sum(d["n"] for d in per.values())} | {allt / alld:.1f} | | | | |')
        f = pd / 'edit.jsonl'
        if f.exists():
            rs = [json.loads(l) for l in open(f)]
            m = [r['decode_tps'] for r in rs if r.get('run', -1) >= 0 and r.get('decode_tps')]
            print(f'\n- R828 code edit (thinking off, 2,048 forced, c1, 36 measured): per-stream median {st.median(m):.1f} t/s '
                  f'(min {min(m):.1f}, max {max(m):.1f})')
        print()
    gpu = root / 'gpu.csv'
    if gpu.exists():
        temps = {}
        for l in open(gpu):
            x = [s.strip() for s in l.split(',')]
            if len(x) >= 5 and x[1] in ('0', '1'):
                try:
                    temps.setdefault(x[1], []).append(float(x[4]))
                except ValueError:
                    pass
        print('GPU max temperature: ' + ', '.join(f'GPU {k} {max(v):.0f} C' for k, v in sorted(temps.items())))


if __name__ == '__main__':
    {'control': control, 'report': report}[sys.argv[1]](sys.argv[2])
