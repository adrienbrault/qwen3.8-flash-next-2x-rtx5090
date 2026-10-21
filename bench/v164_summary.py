#!/usr/bin/env python3
"""Summarize completed records, rejecting missing/failed/short benchmark requests."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results',type=Path);args=p.parse_args();report={}
for arm in 'ABCD':
 root=args.results/arm
 if not root.exists():report[arm]={'complete':False,'reason':'no measurement directory'};continue
 summary={'complete':True,'decode':{},'greedy':{}}
 def rows(file):return [json.loads(l) for l in (root/file).read_text().splitlines() if l.strip()]
 for kind in ['code','prose']:
  file=root/('decode-'+kind+'.jsonl')
  if not file.exists():summary['complete']=False;continue
  data=rows(file.name)
  good=len(data)==26 and all(x.get('ok') and x.get('finish_reason')=='length' and x.get('server_completion_tokens')==1024 for x in data)
  summary['complete'] &= good
  summary['decode'][kind]={'complete':good,'requests':len(data),'by_concurrency':{}}
  for c in [1,4,8]:
   rs=[x for x in data if x.get('conc')==c]
   runs=[]
   for run in [0,1]:
    same=[x for x in rs if x.get('run')==run]
    if len(same)!=c:continue
    runs.append({'decode_aggregate_tps':sum(x['decode_tps'] for x in same),'median_decode_tps':statistics.median(x['decode_tps'] for x in same),'median_ttft_s':statistics.median(x['ttft_s'] for x in same)})
   if runs:summary['decode'][kind]['by_concurrency'][c]={'runs':runs,'median_decode_aggregate_tps':statistics.median(x['decode_aggregate_tps'] for x in runs),'median_per_stream_decode_tps':statistics.median(x['median_decode_tps'] for x in runs)}
 if (root/'greedy.jsonl').exists():
  g=rows('greedy.jsonl');summary['greedy']={'records':len(g),'complete':len(g)==5 and all(x.get('text') and x.get('pred_n') for x in g)}
  summary['complete'] &= summary['greedy']['complete']
 if (root/'agents.jsonl').exists():
  data=rows('agents.jsonl');good=bool(data) and all(x.get('ok') and x.get('gen_tokens')==x.get('asked_tokens') for x in data)
  groups={}
  for x in data: groups.setdefault((x['session'],x['inst']),[]).append(x)
  good &= {k[0] for k in groups}==set(range(6)) and len(groups)==6
  good &= all(sorted(x['step'] for x in group)==list(range(group[0]['steps_asked'])) for group in groups.values())
  span=max(x['t_submit']+x['wall_s'] for x in data)-min(x['t_submit'] for x in data) if data else 0
  summary['agents']={'complete':good,'requests':len(data),'wall_span_s':span,'wall_output_tps':sum(x.get('gen_tokens',0) for x in data)/span if span else None,'median_ttft_s':statistics.median(x['ttft_s'] for x in data) if data else None,'max_sse_gap_s':max(x['sse_gap_max_s'] for x in data) if data else None}
  summary['complete'] &= good
 else:summary['complete']=False
 if (root/'model.json').exists():
  info=json.loads((root/'model.json').read_text());summary['model']=info['id'];summary['pool']=info['parameters']['cache_size'];summary['window']=info['parameters']['max_seq_len']
 report[arm]=summary
print(json.dumps(report,indent=2))
