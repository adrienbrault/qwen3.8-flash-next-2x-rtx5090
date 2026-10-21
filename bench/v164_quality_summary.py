#!/usr/bin/env python3
"""Validate the completed paired accuracy screen and export score-only records."""
import argparse, collections, csv, json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--export',type=Path);a=p.parse_args()
summary={}; cases={}; exports={'gsm-outcomes.csv':[], 'tool-outcomes.csv':[], 'needle-outcomes.csv':[]}
for arm in 'BCD':
    root=a.results/arm
    assert (root/'gsm8k-exit.txt').read_text().strip()=='0'
    assert (root/'tool-eval-exit.txt').read_text().strip()=='0'
    gs=[json.loads(l) for f in (root/'gsm8k').rglob('*.jsonl') for l in f.read_text().splitlines() if l.strip()]
    assert len(gs)==200
    cases[arm]={(g['doc_id'],g['filter']):g for g in gs};assert len(cases[arm])==200
    math={}
    for filt in ['strict-match','flexible-extract']:
        group=[g for g in gs if g['filter']==filt]
        assert {g['doc_id'] for g in group}==set(range(100))
        math[filt]=sum(g['exact_match'] for g in group)
    for g in gs:
        exports['gsm-outcomes.csv'].append(dict(arm=arm,doc_id=g['doc_id'],filter=g['filter'],exact_match=g['exact_match'],doc_hash=g['doc_hash'],prompt_hash=g['prompt_hash']))
    t=json.loads((root/'tool-eval.json').read_text());rs=t['scores']['scenario_results']
    assert t['status']=='completed' and len(rs)==69 and len({r['scenario_id'] for r in rs})==69
    assert sum(r['points'] for r in rs)==t['scores']['total_points']
    assert t['scores']['max_points']==138
    for r in rs:
        exports['tool-outcomes.csv'].append(dict(arm=arm,scenario_id=r['scenario_id'],status=r['status'],points=r['points']))
    ns=[json.loads(l) for l in (root/'needles.jsonl').read_text().splitlines() if l.strip()]
    assert len(ns)==3 and all('err' not in n for n in ns)
    for n in ns:
        exports['needle-outcomes.csv'].append(dict(arm=arm,frac=n['frac'],prompt_tokens=n['prompt_tokens'],completion_tokens=n['completion_tokens'],finish_reason=n['finish_reason'],hit=n['hit']))
    summary[arm]={'gsm8k_n':100,'gsm8k_correct':math,'tool_scenarios':69,'tool_points':t['scores']['total_points'],'tool_max_points':138,'tool_reported_score':t['final_score'],'tool_status_counts':dict(collections.Counter(r['status'] for r in rs)),'needle_hits':sum(n['hit'] for n in ns),'needle_n':3,'needle_actual_prompt_tokens':sorted({n['prompt_tokens'] for n in ns}),'needle_length_finishes':sum(n['finish_reason']=='length' for n in ns)}
for arm in 'CD':
    assert cases[arm].keys()==cases['B'].keys()
    for k,g in cases[arm].items():
        assert g['doc_hash']==cases['B'][k]['doc_hash'] and g['prompt_hash']==cases['B'][k]['prompt_hash']
    summary[arm]['paired_vs_B']={}
    for filt in ['strict-match','flexible-extract']:
        pairs=[(cases['B'][k]['exact_match'],g['exact_match']) for k,g in cases[arm].items() if k[1]==filt]
        summary[arm]['paired_vs_B'][filt]={'gained':sum(b==0 and c==1 for b,c in pairs),'lost':sum(b==1 and c==0 for b,c in pairs)}
if a.export:
    a.export.mkdir(parents=True,exist_ok=True)
    for name,rows in exports.items():
        with (a.export/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
    (a.export/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
