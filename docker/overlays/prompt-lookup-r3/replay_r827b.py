#!/usr/bin/env python3
"""R827b request totals -> explicitly reconstructed step schedules, never raw step traces."""
import argparse
import json
from pathlib import Path
import random
from selftest import helper
m=helper()
P=Path(__file__).resolve().parent


def reconstruct(row, mode='balanced', seed=0):
    n=row['decode_steps'];h=row['lookup_hits'];a=row['lookup_accepted']
    assert row['lookup_checks']==n and row['lookup_proposed']==2*h and 0<=a<=2*h
    if mode=='balanced':
        hits=[i for i in range(n) if ((i+1)*h)//n>(i*h)//n]
    elif mode=='front':hits=list(range(h))
    elif mode=='back':hits=list(range(n-h,n))
    else:
        hits=sorted(random.Random(seed).sample(range(n),h))
    # Exact accepted-token mass; prefix lengths per hit are 0,1,2.
    accepted=[((i+1)*a)//h-(i*a)//h for i in range(h)] if h else []
    if mode=='front':accepted=sorted(accepted,reverse=True)
    elif mode=='back':accepted=sorted(accepted)
    elif mode=='shuffle':random.Random(seed+1).shuffle(accepted)
    events=[(False,0)]*n
    for i,acc in zip(hits,accepted):events[i]=(True,acc)
    assert sum(hit for hit,acc in events)==h and sum(acc for hit,acc in events)==a
    return events


def replay(row, mode='balanced', seed=0, eligible=True):
    a=m.AdaptiveLookup();states=[]
    for hit,accepted in reconstruct(row,mode,seed):
        state,check=a.begin(eligible);states.append(state)
        if check:
            match=m.PromptLookupMatch(0,3,(1,2,3)) if hit else None
            a.opened(1,match,3)
            if state!='OFF':
                for pos in range(accepted+1):a.observe(pos,pos+1)
        a.finish()
    return dict(label=row['label'],steps=len(states),on_steps=states.count('ON'),
                probe_steps=states.count('PROBE'),off_steps=states.count('OFF'),
                enable_transitions=a.metrics['lookup_enable_transitions'],
                disable_transitions=a.metrics['lookup_disable_transitions'])


def report():
    result=dict(kind='aggregate-constrained reconstruction; NOT observed chronological replay',
        limitations=['R827b recorded request totals, not per-step hits/acceptance or batch occupancy.',
                     'Copied-tail acceptance is a proxy: r3 PROBE shadow scores can be lower when MTP verification stops early.',
                     'Schedules preserve all request hit/proposal/acceptance totals; ordering and counterfactual target behavior are assumed.',
                     'c2/c3 request-only totals cannot reconstruct late singleton tails.'],shapes={})
    for shape in ('agent-c1','c1-prose','c1-code'):
        rows=[]
        for boot in ('ON1','ON2'):
            data=json.loads((P/'fixtures/r827b'/f'counters-{boot}-{shape}.json').read_text())
            rows+=data['counters']
        schedules={}
        for mode,seed in [('balanced',0),('front',0),('back',0)]+[('shuffle',n) for n in range(10)]:
            requests=[replay(r,mode,seed) for r in rows]
            steps=sum(r['steps'] for r in requests);on=sum(r['on_steps'] for r in requests)
            schedules[f'{mode}-{seed}']=dict(on_fraction=on/steps,on_steps=on,steps=steps,requests=requests)
        result['shapes'][shape]=dict(balanced_on_fraction=schedules['balanced-0']['on_fraction'],
            reconstruction_range=[min(v['on_fraction'] for v in schedules.values()),max(v['on_fraction'] for v in schedules.values())],
            schedules=schedules)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');a=ap.parse_args();data=report()
    if a.out:Path(a.out).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({shape:dict(predicted_on_fraction=d['balanced_on_fraction'],
                               reconstruction_range=d['reconstruction_range']) for shape,d in data['shapes'].items()},indent=2))
    print(data['kind'])


if __name__=='__main__':main()
