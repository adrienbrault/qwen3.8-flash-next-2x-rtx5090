#!/usr/bin/env python3
"""Synthetic gate regressions, explicitly not measurements. Run during install/dry-run."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import gate


def make(root):
    for tag in gate.TAGS:
        rate=103 if tag.startswith('ON') else 100
        for suite in ('fn','chat','agent','fnstyle'):
            rows=[]
            for pid,_,_ in gate.bodies('fixture-model',suite,True):
                rows.append(dict(tag=tag,pid=pid,suite=suite,ok=True,conc=1,run=0,
                                 completion_tokens=64,prompt_tokens=8192,client_frames=25,
                                 decode_tps=rate,value={'content':f'same {suite}/{pid}','finish':'length'},
                                 request_sha256='same-request-by-pid-'+pid,adaptive_on_steps=25 if tag.startswith('ON') else 0))
            (root/f'identity-{tag}-{suite}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        for shape in [f'c{c}-{kind}' for c in (1,2,3) for kind in ('code','prose')]+['agent-c1','identity-agent']:
            on=tag.startswith('ON')
            counter=dict(label='#1 chat/completions (stream)',gen_tokens=64,gen_time=.3,
                         decode_steps=25,lookup_checks=20 if on else 0,lookup_hits=8 if on else 0,
                         lookup_proposed=16 if on else 0,lookup_accepted=12 if on else 0)
            counter.update({key:0 for key in gate.COUNTERS if key not in counter})
            counter['lookup_on_steps' if on else 'lookup_off_steps']=25
            counter['lookup_all_hit_rounds']=counter['lookup_hits']
            counter['lookup_skipped_forwards']=counter['lookup_proposed']
            counter['lookup_accept_copy_1']=8 if on else 0
            counter['lookup_accept_copy_2']=4 if on else 0
            gate.dump(root/f'counters-{tag}-{shape}.json',dict(n=12 if shape=='identity-agent' else (48 if shape=='agent-c1' else 5*int(shape[1])),
                      measured_n=12 if shape=='identity-agent' else (36 if shape=='agent-c1' else 3*int(shape[1])),
                      counters=[counter],ms_step=12,tokens_step=2.56,proposal_hit_rate=.4 if on else 0,
                      accepted_lookup_tokens_step=.48 if on else 0))
        records=[]
        for run in range(3):
            for r in gate.agent_prompts():
                records.append(dict(tag=tag,conc=1,pid=r['pid'],run=run,ok=True,
                                    completion_tokens=2048,client_frames=700,decode_tps=rate))
        (root/f'speed-{tag}-agent.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        for c in (1,2,3):
            for kind in ('code','prose'):
                records=[dict(tag=f'{tag}-c{c}-{kind}',run=r,i=i,conc=c,ok=True,completion_tokens=1024,
                              server_completion_tokens=1024,client_frames=350,decode_tps=rate,
                              finish_reason='length',prompt_tokens=118)
                         for r in range(3) for i in range(c)]
                (root/f'fn-{tag}-c{c}-{kind}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))


def reject_mutation(mutator,expected=None):
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);make(root);mutator(root)
        try:
            with contextlib.redirect_stdout(io.StringIO()): result=gate.report(root)
        except (AssertionError,ValueError,KeyError,FileNotFoundError):
            assert expected is None;return
        assert expected and result['DECISION']==expected


def change_rates(root,tag,rate,shape='agent'):
    path=root/f'speed-{tag}-agent.jsonl' if shape=='agent' else root/f'fn-{tag}-{shape}.jsonl'
    rows=[json.loads(r) for r in path.read_text().splitlines()]
    for r in rows:r['decode_tps']=rate
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def main():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);make(root)
        with contextlib.redirect_stdout(io.StringIO()): result=gate.report(root)
        assert result['DECISION']=='PASS'
    reject_mutation(lambda r: change_rates(r,'ON2',101), 'REJECT')
    reject_mutation(lambda r: change_rates(r,'OFF2',104), 'REJECT') # EACH OFF comparator
    reject_mutation(lambda r: change_rates(r,'ON1',98,'c2-prose'), 'REJECT')
    reject_mutation(lambda r:(r/'identity-ON1-agent.jsonl').unlink())
    def identity_flip(root,tag='OFF2',n=1):
        path=root/f'identity-{tag}-fn.jsonl';rows=[json.loads(v) for v in path.read_text().splitlines()]
        for k in range(n):rows[k]['value']['content']='token-zero flip'
        path.write_text(''.join(json.dumps(v)+'\n' for v in rows))
    # R828b: OFF/OFF must be identical; an ON boot may diverge on <= gate.MAX_ON_DIVERGENCE of 24 prompts.
    reject_mutation(identity_flip,'INVALID')
    reject_mutation(lambda r:identity_flip(r,'ON1',gate.MAX_ON_DIVERGENCE),'PASS')
    reject_mutation(lambda r:identity_flip(r,'ON1',gate.MAX_ON_DIVERGENCE+1),'REJECT')
    def offpath_flip(root):
        identity_flip(root,'ON1')
        path=root/'identity-ON1-fn.jsonl'
        rows=[json.loads(v) for v in path.read_text().splitlines()]
        rows[0]['adaptive_on_steps']=0
        path.write_text(''.join(json.dumps(v)+'\n' for v in rows))
    reject_mutation(offpath_flip,'REJECT')
    # The intermediate OFF/OFF command must actually exit nonzero.
    import subprocess,sys
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);make(root);identity_flip(root)
        run=subprocess.run([sys.executable,'-B',str(gate.P/'gate.py'),'check-identity',
                            '--root',str(root),'--tag','OFF2'],capture_output=True,text=True)
        assert run.returncode==3 and 'INVALID' in run.stdout
    def step(a,hit=True,acc=2):
        state,check=a.begin(True)
        if check:
            match=gate.LOOKUP.PromptLookupMatch(0,3,(1,2,3)) if hit else None
            a.opened(1,match,3)
            if state!='OFF':
                for pos in range(acc+1):a.observe(pos,pos+1)
        a.finish();return state
    adaptive=gate.LOOKUP.AdaptiveLookup()
    assert 'ON' not in [step(adaptive,i%5==0,0) for i in range(400)]
    adaptive=gate.LOOKUP.AdaptiveLookup()
    assert [step(adaptive) for _ in range(100)].count('ON')==76
    def inactive(root):
        path=root/'counters-ON1-identity-agent.json';d=json.loads(path.read_text())
        for key in ('lookup_hits','lookup_accepted','lookup_proposed','lookup_all_hit_rounds','lookup_skipped_forwards','lookup_accept_copy_1','lookup_accept_copy_2'):
            d['counters'][0][key]=0
        gate.dump(path,d)
    reject_mutation(inactive)
    def missing_shape(root):
        path=root/'fn-ON2-c3-code.jsonl';path.write_text(path.read_text().splitlines()[0]+'\n')
    reject_mutation(missing_shape)
    print('R828 GATE REPLAY PASS: synthetic eligibility, both-boot speed, -1% floor, identity flip, inactive lookup, missing data')

if __name__=='__main__':main()
