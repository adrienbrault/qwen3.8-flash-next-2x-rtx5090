#!/usr/bin/env python3
"""R823 immutable SSE client, exact trace joins, technical validation and registered verdicts."""
import argparse
import collections
import concurrent.futures
import csv
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import threading
import urllib.request

sys.dont_write_bytecode=True
from r823_fixtures import canonical,lcp,SIZES,D
ORDER=('A','B','B','A')
ORIGINALS=('A1','B1','B2','A2')
EXPERIMENTS = {
    'r823': dict(order=ORDER, originals=ORIGINALS,
                 policies={'A': (32768, 0), 'B': (2048, 0)}),
    'r823c': dict(order=('A','C','D','D','C','A'),
                  originals=('A1','C1','D1','D2','C2','A2'),
                  policies={'A': (32768, 0), 'C': (4096, 12288), 'D': (4096, 12288)}),
    'r823b': dict(order=('A','B','C','C','B','A'),
                  originals=('A1','B1','C1','C2','B2','A2'),
                  policies={'A': (32768, 0), 'B': (16384, 0), 'C': (4096, 12288)}),
}


def checkpoint_policy(experiment, arm):
    return EXPERIMENTS[experiment]['policies'][arm]


def effective_interval(position, prompt_end, pp, tail=0):
    if position >= prompt_end - 4096:
        return 2048
    return 32768 if tail and position < prompt_end - tail else pp
VERDICTS=('SUPPORTED','RECOVERY-SUPPORTED / FIDELITY-UNRESOLVED','NOT-SUPPORTED','INCONCLUSIVE','INCOMPLETE')


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def sha(data): return hashlib.sha256(data).hexdigest()


def load_fixtures(root):
    root=Path(root)
    for row in (root/'SHA256SUMS').read_text().splitlines():
        h,name=row.split('  ',1)
        if sha((root/name).read_bytes())!=h: raise ValueError('fixture hash mismatch: '+name)
    manifest=json.loads((root/'manifest.json').read_text())
    payloads=json.loads(gzip.decompress((root/'payloads.json.gz').read_bytes()))
    keys=set(); previous={}
    for r in manifest['requests']:
        if r['key'] in keys: raise ValueError('duplicate fixture request key')
        keys.add(r['key']); p=payloads[r['payload']]
        if sha(p['body'].encode())!=r['payload'] or len(p['token_ids'])!=r['n']:
            raise ValueError('fixture payload identity/count drift')
        prior=previous.get(r['session'])
        if prior is not None and lcp(prior,p['token_ids'])!=r['lcp']:
            raise ValueError('fixture actual LCP drift')
        if r['d'] is not None and r['d']!=r['lcp']: raise ValueError('fixture boundary assertion failed')
        previous[r['session']]=p['token_ids']
    if manifest['seed']!=823 or manifest['request_count']!=sum(r['case']!='CANARY' for r in manifest['requests']):
        raise ValueError('fixture plan drift')
    return manifest,payloads


def opener(): return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url,body,headers=None):
    body=body if isinstance(body,str) else canonical(body)
    rq=urllib.request.Request(url,body.encode(),{'Content-Type':'application/json',**(headers or {})})
    return json.load(opener().open(rq,timeout=300))


def token_bearing(event):
    for c in event.get('choices',[]):
        d=c.get('delta') or {}
        if c.get('text') or any(d.get(k) for k in ('content','reasoning_content','reasoning','refusal')):
            return True
        for call in d.get('tool_calls') or []:
            fn=call.get('function') or {}
            if fn.get('name') or fn.get('arguments'): return True
        logs=c.get('logprobs') or {}
        if logs.get('content') or logs.get('tokens'): return True
    return False


def sse(response):
    data=[]
    for raw in response:
        line=raw.decode().rstrip('\r\n')
        if line.startswith('data:'): data.append(line[5:].lstrip())
        elif not line and data:
            value='\n'.join(data); data=[]
            if value=='[DONE]': return
            yield time.perf_counter(),json.loads(value)
    if data: raise ValueError('truncated SSE event')


def output(events):
    """Accumulate fields independently of SSE partitioning, including empty visible answers."""
    fields=collections.defaultdict(str); tokens=[]; finish=[]; logprobs=[]
    def add(prefix,value):
        if isinstance(value,str): fields[prefix]+=value
        elif isinstance(value,dict):
            for k,v in value.items():
                if k!='index': add(prefix+'/'+k,v)
        elif isinstance(value,list):
            for i,v in enumerate(value): add(prefix+'/'+str(v.get('index',i) if isinstance(v,dict) else i),v)
        elif value is not None: fields[prefix]+=canonical(value)
    for e in events:
        for c in e.get('choices',[]):
            if 'text' in c: add('text',c['text'])
            add('delta',c.get('delta') or {})
            if c.get('finish_reason') is not None: finish.append(c['finish_reason'])
            logs=c.get('logprobs') or {}
            content=logs.get('content') or []
            logprobs.extend(content)
            if content: tokens.extend([r.get('token_id',r.get('token')) for r in content])
            elif logs.get('tokens'):
                tokens.extend(logs['tokens'])
                for tok,lp,top in zip(logs['tokens'],logs.get('token_logprobs',[]),logs.get('top_logprobs',[])):
                    logprobs.append(dict(token=tok,logprob=lp,top_logprobs=[dict(token=k,logprob=v) for k,v in (top or {}).items()]))
    return dict(fields=dict(fields),tokens=tokens,finish=finish),logprobs


def identity(a,b): return canonical(a)==canonical(b)


def divergence(a,b,la=None,lb=None):
    aa=a.get('tokens',[]); bb=b.get('tokens',[]); i=lcp(aa,bb)
    out={'first_differing_token':i if aa!=bb else None,'bytes_equal':a.get('fields')==b.get('fields'),
         'finish_equal':a.get('finish')==b.get('finish')}
    for name,logs in [('A',la),('B',lb)]:
        if logs and i<len(logs):
            top=logs[i].get('top_logprobs',[])
            vals=sorted([r['logprob'] for r in top],reverse=True)
            out[name+'_top5']=top
            out[name+'_margin']=vals[0]-vals[1] if len(vals)>1 else None
    return out


def request_one(row,payload,url,barrier=None):
    first=None; events=[]
    headers={'x-r823-request':row['key'],'x-r823-conversation':row['session']}
    rq=urllib.request.Request(url+('/chat/completions' if row['path']=='chat' else '/completions'),
                              payload['body'].encode(),{'Content-Type':'application/json',**headers})
    op=opener()
    if barrier is not None: barrier.wait(timeout=10)
    t=time.perf_counter();epoch=time.time()
    result=dict(key=row['key'],payload=row['payload'],case=row['case'],path=row['path'],session=row['session'],
                turn=row['turn'],kind=row['kind'],dispatch_epoch=epoch,status='INVALID')
    try:
        with op.open(rq,timeout=300) as response:
            for stamp,e in sse(response):
                if first is None and token_bearing(e): first=stamp
                events.append(e)
        out,logs=output(events)
        usage=[e['usage'] for e in events if e.get('usage')]
        if first is None or not usage or not out['finish']: raise ValueError('missing token-bearing SSE/usage/finish')
        result.update(status='VALID',ttft_s=first-t,total_s=time.perf_counter()-t,usage=usage[-1],
                      output=out,logprobs=logs,events=events)
    except Exception as e: result['error']=f'{type(e).__name__}: {e}'
    return result


def run_client(a):
    manifest,payloads=load_fixtures(a.fixtures)
    rows=[r for r in manifest['requests'] if (r['case']=='CANARY')==a.canary]
    # Before traffic, validate every immutable prompt with the served tokenizer. This does not touch cache.
    verified=[]
    for h in dict.fromkeys(r['payload'] for r in rows):
        p=payloads[h]; body=json.loads(p['body'])
        prompt=post(a.url+'/apply-template',body)['prompt'] if 'messages' in body else p['prompt']
        observed=post(a.url+'/token/encode',{'text':prompt,'add_bos_token':False})['tokens']
        if observed!=p['token_ids']:
            raise ValueError('server render/tokenizer differs from offline fixture: '+h)
        verified.append({'payload':h,'n':len(observed),'observed_boundaries':'post-render trace authoritative',
                         'render_sha':sha(prompt.encode())})
    dump(a.out+'.preflight.json',verified)
    # C2 is paired by turn and kind, with a barrier after each pair. Seeds are paired too.
    groups=[]; used=set()
    for r in rows:
        if r['key'] in used: continue
        group=[r]
        if r['case']=='C2':
            group=[x for x in rows if x['case']=='C2' and x['path']==r['path'] and x['turn']==r['turn'] and x['kind']==r['kind']]
            if len(group)!=2: raise ValueError('C2 barrier plan malformed')
        used.update(x['key'] for x in group); groups.append(group)
    with open(a.out,'w') as f:
        for group in groups:
            barrier=threading.Barrier(2) if len(group)==2 else None
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                got=list(pool.map(lambda r:request_one(r,payloads[r['payload']],a.url,barrier),group))
            for record in got: f.write(canonical(record)+'\n'); f.flush()
    return 0 if all(r['status']=='VALID' for r in read_jsonl(a.out)) else 1


def read_jsonl(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def parse_traces(text):
    rows=[]
    for line in text.splitlines():
        if '[R823-cache]' in line:
            r=json.loads(line.split('[R823-cache]',1)[1].strip())
            if r.get('v')!=1 or not r.get('run') or not isinstance(r.get('t'),(int,float)):
                raise ValueError('malformed trace envelope')
            rows.append(r)
    if not rows: raise ValueError('no R823 trace')
    if len({r['run'] for r in rows})!=1: raise ValueError('mixed boot trace namespaces')
    return rows


def parse_server(text):
    # Wrapped Rich headers/completions, stripped timestamp prefixes, joined by explicit HTTP serial.
    text=re.sub(r'\x1b\[[0-9;]*m','',text)
    text=re.sub(r'(?m)^\d{4}-\d\d-\d\dT\S+\s+','',text)
    records=collections.defaultdict(dict)
    # The lookahead must stay on ONE line: with re.S, '\n.*?#' reached any later record and cut every body
    # after its first line, losing 'N new in X s' (R823 try 2 canary).
    pattern=r'#(\d+) (?:chat/)?completions(?: \(stream\))?:\s*(.*?)(?=\n[^\n]*?#\d+ (?:chat/)?completions|\Z)'
    for serial,body in re.findall(pattern,text,re.S):
        body=' '.join(body.split()); dest=records[int(serial)]
        n=re.search(r'([\d,]+) prompt tokens',body)
        if n: dest['n']=int(n[1].replace(',',''))
        n=re.search(r'prompt ([\d,]+) tokens.*?([\d,]+) new in ([\d.]+) s',body)
        if n: dest.update(n=int(n[1].replace(',','')),new=int(n[2].replace(',','')),engine_s=float(n[3]))
        n=re.search(r'first token ([\d.]+) s',body)
        if n: dest['server_ttft_s']=float(n[1])
    return dict(records)


def joins(traces,clients,fixtures):
    cr={r['key']:r for r in clients}
    if len(cr)!=len(clients): raise ValueError('duplicate client key')
    expected={r['key']:r for r in fixtures}
    if set(cr)!=set(expected): raise ValueError('missing/foreign client request keys')
    by=collections.defaultdict(list)
    for t in traces:
        if t.get('key') is not None: by[t['key']].append(t)
    renders=[t for t in traces if t['event']=='render']
    # The launcher's boot warm-ups ("Warmup." + FASTWARM N_p 20/29/11) are traced too: they precede every
    # fixture request. Any non-fixture request after the first fixture request is still foreign.
    ours=[t['serial'] for t in renders if str(t['key']).startswith('r823/')]
    if ours and any(not str(t['key']).startswith('r823/') and t['serial']>min(ours) for t in renders):
        raise ValueError('foreign server request during fixture traffic')
    renders=[t for t in renders if str(t['key']).startswith('r823/')]
    if {t['key'] for t in renders}!=set(expected) or len(renders)!=len(expected):
        raise ValueError('foreign/missing/duplicate server requests')
    result=[]
    for key,f in expected.items():
        c=cr[key]; events=by[key]
        one={}
        for name in ('render','allocation_before','allocation','finish','final'):
            rr=[t for t in events if t['event']==name]
            if len(rr)!=1: raise ValueError(f'{key}: missing/duplicate {name}')
            one[name]=rr[0]
        serials={t['serial'] for t in events if t.get('serial') is not None}
        if len(serials)!=1: raise ValueError('explicit key/Job serial mismatch')
        if c['status']!='VALID' or c['payload']!=f['payload']: raise ValueError('client failure/payload substitution')
        r=one['render']; fin=one['finish']; alloc=one['allocation']
        if r['n']!=f['n'] or r['lcp_prompt']!=f['lcp'] or r['heuristic'] or not r['greedy'] or r.get('sampler_mode')!='greedy':
            raise ValueError('actual render/LCP/sampler invalid')
        if f.get('kind')=='edit':
            expected_message=0 if f.get('case')=='EARLY-EDIT' else 3
            if r.get('changed_message_index')!=expected_message: raise ValueError('changed message boundary mismatch')
        if r['min_tokens']!=64 or r['max_tokens']!=64 or r['loop_detect_window']!=0:
            raise ValueError('length/sampler/loop mismatch')
        if fin['generated']!=64 or c['usage']['completion_tokens']!=64 or fin['prompt_tokens']!=f['n']:
            raise ValueError('output/prompt count mismatch')
        if f['lcp'] is not None and alloc['selected']>(f['lcp']//256)*256:
            raise ValueError('selected cache exceeds verified unchanged page prefix')
        if fin['cached_tokens']!=alloc['selected'] or not math.isfinite(c['ttft_s']):
            raise ValueError('cache count/timing mismatch')
        for t in events:
            if t.get('ram_bytes',0)>4096*1024**2 or t.get('ram_max',4096*1024**2)!=4096*1024**2:
                raise ValueError('logical RAM budget drift')
        before=one['allocation_before']
        losses=[x for x in traces if x['event'] in ('loss','kv_loss') and x['t']<=alloc['t']]
        held={x['matching_hash'] for x in before['candidates'] if x['present_RAM']}
        pages=set(r['page_digests'][:(f['lcp'] or 0)//256])
        latest_save={}
        for x in traces:
            if x['event']=='save' and x['t']<=alloc['t']: latest_save[x['digest']]=x['t']
        relevant=[x for x in losses if
                  (x['event']=='loss' and x.get('digest') in pages and x.get('digest') not in held and
                   (x.get('position') or 0)>alloc['selected'] and latest_save.get(x['digest'],-1)<=x['t']) or
                  (x['event']=='kv_loss' and before['pre_cap_kv_prefix']<(f['lcp'] or 0)//256*256 and x.get('digest') in pages)]
        result.append(dict(fixture=f,client=c,trace=one,events=events,relevant_losses=relevant,
                           gap=max(0,(f['lcp'] or 0)-alloc['selected'])))
    return result


def design_errors(attempts,records,experiment='r823'):
    cfg=EXPERIMENTS[experiment]; originals=cfg['originals']; order=cfg['order']; count=len(originals)
    why=[]
    if [r['boot'] for r in attempts[:count]]!=list(originals): why.append('original order is not '+' '.join(order))
    if len({r['boot'] for r in attempts})!=len(attempts): why.append('duplicate attempt')
    for r,arm in zip(attempts[:count],order):
        if r['arm']!=arm or r['replaces']!='-': why.append('original assignment changed')
    extras=[]
    for arm in cfg['policies']:
        bad=[r['boot'] for r in attempts[:count] if r['arm']==arm and records.get(r['boot'],{}).get('status')=='INVALID']
        if bad: extras.append((f'R{arm}',arm,bad[0]))
    for r,want in zip(attempts[count:],extras):
        if (r['boot'],r['arm'],r['replaces'])!=want: why.append('unauthorized replacement')
    if len(attempts)>count+len(extras): why.append('excess replacement')
    return why


def med(rows,field): return statistics.median(r[field] for r in rows)


def summarize(joined,arm,experiment='r823'):
    pp,tail=checkpoint_policy(experiment,arm)
    if experiment in ('r823b','r823c') and any(r['trace']['render'].get('policy_tail',0)!=tail for r in joined):
        raise ValueError('tail engagement/readback failed')
    if any(r['trace']['render']['policy_pp']!=pp or r['trace']['render']['policy_near']!=2048 or r['trace']['render']['chunk']!=2048 for r in joined):
        raise ValueError('density engagement/readback failed')
    groups=collections.defaultdict(list)
    for r in joined: groups[(r['fixture']['path'],r['fixture']['case'])].append(r)
    summary={}; mechanism=True; intended=False; loss_dominates=False; density=True; reuse_failure=False
    for (path,case),rows in groups.items():
        edits=[r for r in rows if r['fixture']['kind']=='edit']
        samples=[r for r in rows if r['fixture']['kind'] in ('edit','append')]
        seeds=[r for r in rows if r['fixture']['kind'] in ('seed','warm') and r['fixture']['lcp'] is None]
        if experiment in ('r823b','r823c'):
            # T35's long seed reuses its 2689-token warm prefix; still gate its cold-seed cost.
            seeds=[r for r in rows if r['fixture']['kind']=='seed']
        s=dict(ttft=statistics.median(r['client']['ttft_s'] for r in samples) if samples else None,
               gap=statistics.median(r['gap'] for r in edits) if edits else None,
               cold=statistics.median(r['trace']['finish']['engine_prefill_s'] for r in seeds),
               cold_client=statistics.median(r['client']['ttft_s'] for r in seeds),
               losses=sum(bool(r['relevant_losses']) for r in edits),turns=len(edits),
               ram_peak=max(t.get('ram_peak',0) for r in rows for t in r['events']))
        s['hit_reuse_failures']=[]
        for rr in rows:
            if rr['fixture']['kind']=='hit':
                n=rr['fixture']['n'];minimum=(n-1)//256*256
                if (n-1)%256==0:minimum-=256
                if rr['trace']['allocation']['selected']<minimum:
                    s['hit_reuse_failures'].append(rr['fixture']['key'])
                    if arm!='A':reuse_failure=True
        summary[path+'/'+case]=s
        if path=='raw' and case in D and case!='C2':
            target={'T35':35328,'T32':32768,'T65':65536}[case]
            hits=sum(r['trace']['allocation']['selected']==target for r in edits)
            clean=all(r['trace']['allocation_before']['pre_cap_kv_prefix'] >= r['fixture']['d']//256*256 and not r['relevant_losses'] for r in edits)
            if arm=='A':
                mechanism &= hits/len(edits)>=.9 and clean
                intended |= any(r['gap']>2304 for r in edits)
                loss_dominates |= not clean
        if arm!='A':
            for r in edits:
                held=[x for x in r['trace']['allocation_before']['candidates'] if x['present_RAM'] and x['anchored'] and not x['capped']]
                allowance=2304 if experiment=='r823' else effective_interval(r['fixture']['lcp']//256*256,r['fixture']['n']-1,pp,tail)+256
                if held and r['gap']>allowance and not r['relevant_losses']: density=False
                if case=='C2' and r['relevant_losses'] and r['gap']>allowance: reuse_failure=True
        for r in rows:
            saves=[x for x in r['events'] if x['event']=='save' and x['reason']=='interior']
            if any(x['effective_interval']!=effective_interval(x['position'],r['fixture']['n']-1,pp,tail) or
                   (x['position']-x['cached_base'])%x['effective_interval'] for x in saves): density=False
            # Completed eligible interior boundaries must actually appear, not just an env label.
            alloc=r['trace']['allocation']['selected']; end=r['fixture']['n']-1
            expected={pos for pos in range(alloc+2048,end-4096,2048)
                      if (pos-alloc)%effective_interval(pos,end,pp,tail)==0}
            actual={x['position'] for x in saves}
            if not expected.issubset(actual): density=False
        if case=='APPEND':
            for r in samples:
                minimum=((r['fixture']['lcp']-1)//256)*256
                # R803 aligned-end/MTP excludes one page when exactly aligned.
                if (r['fixture']['lcp']-1)%256==0: minimum-=256
                if r['trace']['allocation']['selected']<minimum:
                    if arm!='A': reuse_failure=True
                    else: loss_dominates=True
    extra = {}
    if experiment == 'r823c':
        pipeline = []
        for r in joined:
            all_forwards = [e for e in r['events'] if e['event'] == 'prefill_forward']
            if any(not isinstance(e.get('prompt_prefill'), bool) for e in all_forwards):
                raise ValueError('prefill trace phase missing')
            forwards = [e for e in all_forwards if e['prompt_prefill']]
            captures = [e for e in r['events'] if e['event'] == 'tail_capture']
            fallbacks = [e for e in r['events'] if e['event'] == 'tail_fallback']
            if r['trace']['render'].get('policy_inforward') != (arm == 'D'):
                raise ValueError('in-forward selector readback failed')
            if r['fixture']['n'] - 1 > r['trace']['allocation']['selected'] and not forwards:
                raise ValueError('missing per-request prefill trace')
            if any(e['end'] <= e['start'] or not isinstance(e['pipeline'], bool) for e in forwards):
                raise ValueError('invalid forward trace')
            if any(e['end'] != f['start'] for e, f in zip(forwards, forwards[1:])):
                raise ValueError('prefill trace gap/overlap')
            if forwards and forwards[0]['start'] != r['trace']['allocation']['selected']:
                raise ValueError('prefill trace start missing')
            if forwards and forwards[-1]['end'] != r['fixture']['n'] - 1:
                raise ValueError('prefill trace tail missing')
            if any(e['pipeline'] and e['end']-e['start']!=2048 for e in forwards):
                raise ValueError('pipeline trace chunk size')
            if arm != 'D' and (captures or fallbacks):
                raise ValueError('in-forward path engaged in old-path arm')
            pipeline.append(dict(key=r['fixture']['key'],
                forwards=len(forwards), pipeline_forwards=sum(e['pipeline'] for e in forwards),
                pipeline_rows=sum(e['end']-e['start'] for e in forwards if e['pipeline']),
                serial_rows=sum(e['end']-e['start'] for e in forwards if not e['pipeline']),
                captures=len(captures), crossed_captures=sum(e.get('crossed',False) for e in captures),
                fallbacks=len(fallbacks), replay_forwards=len(all_forwards)-len(forwards)))
        extra['pipeline_requests'] = pipeline
        # Evidence gate: actual crossed checkpoint publication on single-stream seeds.
        seeds = {r['fixture']['key'] for r in joined if r['fixture']['kind']=='seed' and
                 r['fixture']['case'] in ('T32','T65')}
        if arm == 'D':
            density &= all(p['pipeline_forwards'] > 0 and p['crossed_captures'] > 0
                           for p in pipeline if p['key'] in seeds)
    return dict(cases=summary,mechanism=mechanism,intended_gap=intended,loss_dominates=loss_dominates,
                density=density,reuse_failure=reuse_failure, **extra)


def validate(root,boot,arm,experiment='r823'):
    root=Path(root); errors=[]; out=dict(boot=boot,arm=arm,status='INVALID',errors=errors)
    try:
        if (root/f'void-{boot}.txt').exists(): raise ValueError((root/f'void-{boot}.txt').read_text().strip())
        identity=json.loads((root/f'identity-{boot}.json').read_text())
        if identity.get('errors'): raise ValueError('; '.join(identity['errors']))
        if not identity.get('verified'): raise ValueError('identity not verified')
        manifest,_=load_fixtures(root/'overlay/fixtures')
        fixtures=[r for r in manifest['requests'] if r['case']!='CANARY']
        traces=parse_traces((root/f'container-{boot}.log').read_text())
        joined=joins(traces,read_jsonl(root/f'client-{boot}.jsonl'),fixtures)
        server=parse_server((root/f'container-{boot}.log').read_text())
        for r in joined:
            http=r['trace']['render']['http_serial']
            sr=server[http];fin=r['trace']['finish']
            if sr['n']!=r['fixture']['n'] or sr['n']-sr['new']!=fin['cached_tokens']:
                raise ValueError('wrapped completion/trace reconciliation failed')
            r['server']=sr
        pre=json.loads((root/f'client-{boot}.jsonl.preflight.json').read_text())
        if {x['payload'] for x in pre}!={r['payload'] for r in fixtures}: raise ValueError('missing served-tokenizer preflight')
        out.update(summarize(joined,arm,experiment),joined=joined,hardware=json.loads((root/f'hardware-{boot}.json').read_text()))
        hw=out['hardware']
        if len(hw['vram_min_mib'])!=2 or not hw['samples'] or hw.get('oom') or not hw.get('coverage'):
            raise ValueError('missing telemetry, coverage, or OOM')
        out['status']='VALID'
    except (OSError,ValueError,TypeError,KeyError,IndexError,ZeroDivisionError) as e: errors.append(str(e))
    return out


def decision_r823(valid,errors=(),canary=True):
    out=dict(verdict='INCOMPLETE',real_cause='OPEN',errors=list(errors),comparisons=[],divergences=[])
    arms={a:[r for r in valid if r['arm']==a] for a in ('A','B')}
    if errors or not canary or any(len(v)!=2 for v in arms.values()): return out
    # Order within arm preserves original pairing A1:B1 and A2:B2, replacing only invalid original.
    for arm in arms: arms[arm].sort(key=lambda r:ORIGINALS.index(r.get('replaces',r['boot'])) if r.get('replaces','-')!='-' else ORIGINALS.index(r['boot']))
    identity_ok=True; gains=True; costs=True; mechanism=all(r['mechanism'] for r in arms['A'])
    if any(not r['density'] for r in valid):
        out['verdict']='NOT-SUPPORTED'; out['reason']='actual saves do not engage registered policy'; return out
    for a,b in zip(arms['A'],arms['B']):
        cases=[]
        for key,x in a['cases'].items():
            y=b['cases'][key]; case=key.split('/')[1]
            if case in ('T35','T32','T65','MOVING-EDIT','C2'):
                gain=x['ttft']-y['ttft']; ratio=y['ttft']/x['ttft']; gap=y['gap']/x['gap'] if x['gap'] else None
                passed=gain>=1.0 and ratio<=.75 and gap is not None and gap<=.5
                cases.append(dict(case=key,A=x,B=y,gain_s=gain,ratio=ratio,gap_ratio=gap,pass_gain=passed))
            if case=='APPEND': costs &= y['ttft']-x['ttft']<=.10 and y['ttft']<=x['ttft']*1.2
            costs &= y['cold']<=x['cold']*1.1
        costs &= all(bv>=av-64 for av,bv in zip(a['hardware']['vram_min_mib'],b['hardware']['vram_min_mib']))
        costs &= not b['reuse_failure']
        # Aggregate within each boot/case, then compare boot medians across exact-policy edited cases.
        primary=['raw/'+c for c in ('T35','T32','T65','MOVING-EDIT','C2')]
        am=statistics.median(a['cases'][k]['ttft'] for k in primary)
        bm=statistics.median(b['cases'][k]['ttft'] for k in primary)
        ag=statistics.median(a['cases'][k]['gap'] for k in primary)
        bg=statistics.median(b['cases'][k]['gap'] for k in primary)
        gains &= am-bm>=1.0 and bm<=am*.75 and bg<=ag*.5
        # 'Material' c2 regression has no registered numerical tolerance. Conservative: any increase blocks promotion.
        costs &= all(b['cases'][k]['ttft']<=a['cases'][k]['ttft'] for k in ('raw/C2','chat/C2'))
        out.setdefault('primary',[]).append(dict(A=a['boot'],B=b['boot'],A_ttft=am,B_ttft=bm,A_gap=ag,B_gap=bg))
        out['comparisons'].append(dict(A=a['boot'],B=b['boot'],cases=cases))
    # Every cross-arm, within-arm repeat and variant/HIT output must be byte/token identical.
    comparisons=[(arms['A'][0],arms['A'][1]),(arms['B'][0],arms['B'][1])]
    comparisons += [(a,b) for a in arms['A'] for b in arms['B']]
    for a,b in comparisons:
        aa={r['fixture']['key']:r for r in a['joined']}; bb={r['fixture']['key']:r for r in b['joined']}
        for key,r in aa.items():
            s=bb[key]
            if not identity(r['client']['output'],s['client']['output']):
                identity_ok=False
                out['divergences'].append(dict(key=key,A=a['boot'],B=b['boot'],
                    partition=[r['trace']['allocation']['selected'],s['trace']['allocation']['selected']],
                    **divergence(r['client']['output'],s['client']['output'],r['client']['logprobs'],s['client']['logprobs'])))
    for b in valid:
        by={r['fixture']['key']:r for r in b['joined']}
        for r in b['joined']:
            pair=r['fixture'].get('pair')
            if pair and not identity(r['client']['output'],by[pair]['client']['output']):
                identity_ok=False
                out['divergences'].append(dict(key=r['fixture']['key'],boot=b['boot'],pair=pair,
                                               **divergence(by[pair]['client']['output'],r['client']['output'])))
    out.update(mechanism=mechanism,gain=gains,cost=costs,strict_identity=identity_ok)
    if not mechanism and (not all(r['intended_gap'] for r in arms['A']) or any(r['loss_dominates'] for r in arms['A'])):
        out['verdict']='INCONCLUSIVE'
    elif not mechanism or not gains or not costs: out['verdict']='NOT-SUPPORTED'
    elif not identity_ok: out['verdict']='RECOVERY-SUPPORTED / FIDELITY-UNRESOLVED'
    else: out['verdict']='SUPPORTED'
    return out



def decision(valid,errors=(),canary=True,experiment='r823'):
    if experiment=='r823':
        return decision_r823(valid,errors,canary)
    cfg=EXPERIMENTS[experiment]
    out=dict(verdict='INCOMPLETE',real_cause='OPEN',errors=list(errors),candidates={})
    arms={arm:[r for r in valid if r['arm']==arm] for arm in cfg['policies']}
    if errors or not canary or any(len(v)!=2 for v in arms.values()): return out
    for rows in arms.values():
        rows.sort(key=lambda r: cfg['originals'].index(r['replaces'] if r.get('replaces','-')!='-' else r['boot']))
    # Fidelity includes ALL candidate pairs, so B/C drift cannot be hidden by independent verdicts.
    fidelity=True; divergences=[]
    for i,a in enumerate(valid):
        for b in valid[i+1:]:
            aa={r['fixture']['key']:r for r in a['joined']}; bb={r['fixture']['key']:r for r in b['joined']}
            if aa.keys()!=bb.keys():
                out['errors'].append('cross-boot fixture key mismatch');return out
            for key,r in aa.items():
                s=bb[key]
                if not identity(r['client']['output'],s['client']['output']):
                    fidelity=False
                    divergences.append(dict(key=key,A=a['boot'],B=b['boot'],
                        **divergence(r['client']['output'],s['client']['output'],r['client']['logprobs'],s['client']['logprobs'])))
    for b in valid:
        by={r['fixture']['key']:r for r in b['joined']}
        for r in b['joined']:
            pair=r['fixture'].get('pair')
            if pair and not identity(r['client']['output'],by[pair]['client']['output']):
                fidelity=False;divergences.append(dict(boot=b['boot'],key=r['fixture']['key'],pair=pair))
    mechanism=all(r['mechanism'] for r in arms['A'])
    for arm in cfg['policies']:
        if arm == 'A': continue
        gains=True;costs=True;comparisons=[]
        density=all(r['density'] for r in arms['A']+arms[arm])
        for a,b in zip(arms['A'],arms[arm]):
            rows=[]
            if a['cases'].keys()!=b['cases'].keys():
                out['errors'].append('cross-boot case mismatch');return out
            for key,x in a['cases'].items():
                y=b['cases'][key];case=key.split('/')[1]
                gain=x['ttft']-y['ttft'] if x['ttft'] is not None else None
                passed=None
                if case in ('T32','C2'):
                    passed=gain>=1.0 and y['ttft']<=x['ttft']*.75
                    gains &= passed
                cold_bound = 1.03 if experiment == 'r823c' and arm == 'D' and case != 'C2' else 1.1
                seed_ok=y['cold']<=x['cold']*cold_bound and y['cold_client']<=x['cold_client']*cold_bound
                append_ok=case!='APPEND' or (y['ttft']-x['ttft']<=.10 and y['ttft']<=x['ttft']*1.2)
                early_ok=case!='EARLY-EDIT' or y['ttft']<=x['ttft']*1.1
                costs &= seed_ok and append_ok and early_ok
                rows.append(dict(case=key,A=x,candidate=y,gain_s=gain,pass_gain=passed,
                                 pass_cold=seed_ok,cold_bound=cold_bound,pass_append=append_ok,pass_early=early_ok))
            costs &= all(bv>=av-64 for av,bv in zip(a['hardware']['vram_min_mib'],b['hardware']['vram_min_mib']))
            costs &= not b['reuse_failure']
            comparisons.append(dict(A=a['boot'],candidate=b['boot'],cases=rows))
        verdict='SUPPORTED'
        if not mechanism and (not all(r['intended_gap'] for r in arms['A']) or any(r['loss_dominates'] for r in arms['A'])):
            verdict='INCONCLUSIVE'
        elif not mechanism or not gains or not costs or not density: verdict='NOT-SUPPORTED'
        elif not fidelity: verdict='RECOVERY-SUPPORTED / FIDELITY-UNRESOLVED'
        out['candidates'][arm]=dict(verdict=verdict,gain=gains,cost=costs,density=density,comparisons=comparisons)
    supported=[a for a,v in out['candidates'].items() if v['verdict']=='SUPPORTED']
    out.update(strict_identity=fidelity,divergences=divergences,mechanism=mechanism,supported_arms=supported)
    out['verdict']='SUPPORTED' if supported else ('RECOVERY-SUPPORTED / FIDELITY-UNRESOLVED'
        if any(v['verdict']=='RECOVERY-SUPPORTED / FIDELITY-UNRESOLVED' for v in out['candidates'].values())
        else 'INCONCLUSIVE' if not mechanism else 'NOT-SUPPORTED')
    return out


def canary(root):
    root=Path(root)
    try:
        a=read_jsonl(root/'client-CS.jsonl'); b=read_jsonl(root/'client-CA.jsonl')
        manifest,_=load_fixtures(root/'overlay/fixtures'); fixtures=[r for r in manifest['requests'] if r['case']=='CANARY']
        joined=joins(parse_traces((root/'container-CA.log').read_text()),b,fixtures)
        if len(a)!=len(fixtures) or len(b)!=len(a): return False
        # Served canary's usage may not expose cached_tokens. Reconcile exact N-new from logs by HTTP id.
        for bt in ('CS','CA'):
            if (root/f'void-{bt}.txt').exists(): return False
            if not json.loads((root/f'identity-{bt}.json').read_text()).get('verified'): return False
        server=parse_server((root/'container-CS.log').read_text())
        before=parse_server((root/'container-CS-pre.log').read_text())
        after=[server[k] for k in sorted(server) if k>max(before,default=0)]
        if len(after)!=len(fixtures): return False
        by={r['key']:r for r in a}
        for r in joined:
            x=by[r['fixture']['key']]
            if x['status']!='VALID' or x['usage']['completion_tokens']!=64 or not identity(x['output'],r['client']['output']): return False
            record=after[fixtures.index(r['fixture'])]
            if record['n']!=r['fixture']['n'] or record['n']-record['new']!=r['trace']['allocation']['selected']: return False
        timing={'served_median_s':statistics.median(r['ttft_s'] for r in a),
                'trace_median_s':statistics.median(r['ttft_s'] for r in b)}
        dump(root/'canary-cost.json',timing)
        return True
    except (OSError,KeyError,ValueError,TypeError): return False


def equality_errors(root):
    """Step-0 evidence is mandatory when replaying the experiment offline too."""
    try:
        e=json.loads((Path(root)/'gpu-equality.json').read_text())
        rows=e['rows']
        if e['passed'] is not True or {r['case'] for r in rows} != {'T35','T32','T65'} or len(rows)!=3:
            return ['step 0 equality coverage/failure']
        for r in rows:
            if not r['positions'] or not r['equal'] or not r['engaged'] or r['differences']:
                return ['step 0 equality mismatch/disengaged']
            x=r['engagement']
            if (x['INFORWARD']['captures'] <= 0 or x['INFORWARD']['pipeline'] <= 0 or
                    x['CUT']['captures'] != 0 or x['NO_SLAB']['captures'] != 0 or
                    x['NO_SLAB']['fallbacks'] <= 0):
                return ['step 0 equality path coverage']
        return []
    except (OSError,ValueError,KeyError,TypeError):
        return ['step 0 equality artifact missing/invalid']


def decide(root,experiment='r823'):
    cfg=EXPERIMENTS[experiment]
    root=Path(root)
    with (root/'attempts.tsv').open() as f:
        attempts=list(csv.DictReader(f,delimiter='\t'))
    records={r['boot']:validate(root,r['boot'],r['arm'],experiment) for r in attempts}
    for r in attempts: records[r['boot']]['replaces']=r['replaces']
    errors=design_errors(attempts,records,experiment)
    if experiment=='r823c':errors += equality_errors(root)
    with (root/'plan.tsv').open() as f:
        plan=list(csv.DictReader(f,delimiter='\t'))
    if plan!=[dict(boot=b,arm=a) for b,a in zip(cfg['originals'],cfg['order'])]: errors.append('immutable plan drift')
    out=decision([r for r in records.values() if r['status']=='VALID'],errors,canary(root),experiment)
    out['boots']={b:{k:v for k,v in r.items() if k!='joined'} for b,r in records.items()}
    return out


def hardware(a):
    with Path(a.smi).open() as f:
        rows=list(csv.DictReader(f))
    cards=collections.defaultdict(list)
    for r in rows:
        r={k.strip():v.strip() for k,v in r.items()}
        stamp=datetime.datetime.strptime(r['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
        def field(prefix):
            value=next(v for k,v in r.items() if k.startswith(prefix))
            # The unit samples --format=csv (with units): '1099 MiB', '2662 MHz', '96.42 W', '0 %'.
            return float(value.split()[0])
        cards[r['index']].append(dict(t=stamp,free_mib=field('memory.free'),sm_mhz=field('clocks.current.sm'),
                                     memory_mhz=field('clocks.current.memory'),power_w=field('power.draw'),util_pct=field('utilization.gpu')))
    host=read_jsonl(a.host);clients=read_jsonl(a.client)
    lo=min(r['dispatch_epoch'] for r in clients);hi=max(r['dispatch_epoch']+r.get('total_s',0) for r in clients)
    def covered(stamps):
        return bool(stamps and min(stamps)<=lo+1 and max(stamps)>=hi-1 and max((y-x for x,y in zip(stamps,stamps[1:])),default=999)<2)
    coverage=covered([r['t'] for r in host]) and all(covered([r['t'] for r in cards[str(i)]]) for i in range(2))
    per_case={}
    for key in sorted({r['path']+'/'+r['case'] for r in clients}):
        rr=[r for r in clients if r['path']+'/'+r['case']==key]
        start=min(r['dispatch_epoch'] for r in rr);end=max(r['dispatch_epoch']+r.get('total_s',0) for r in rr)
        hh=[r for r in host if start<=r['t']<=end]
        per_case[key]=dict(window_epoch=[start,end],host_samples=len(hh),
                           rss_peak_kib=max((r['rss_kib'] for r in hh),default=None),
                           pinned_peak_kib=max((r['pinned_kib'] for r in hh),default=None),
                           locked_peak_kib=max((r['locked_kib'] for r in hh),default=None),gpu={})
        for i in range(2):
            gg=[r for r in cards[str(i)] if start<=r['t']<=end]
            if gg:
                per_case[key]['gpu'][str(i)]=dict(samples=len(gg),vram_min_mib=min(r['free_mib'] for r in gg),
                    **{k:statistics.mean(r[k] for r in gg) for k in ('sm_mhz','memory_mhz','power_w','util_pct')})
    dump(a.out,dict(samples=len(rows),vram_min_mib=[min(r['free_mib'] for r in cards[str(i)]) for i in range(2)],
                    rss_peak_kib=max(r['rss_kib'] for r in host),locked_peak_kib=max(r['locked_kib'] for r in host),
                    pinned_peak_kib=max(r['pinned_kib'] for r in host),per_case=per_case,
                    coverage=coverage,oom='out of memory' in Path(a.log).read_text().lower()))


def host_sample(a):
    with open(a.out,'w') as f:
        while True:
            txt=Path(f'/proc/{a.pid}/status').read_text(); vals={r.split(':')[0]:r.split(':')[1].strip() for r in txt.splitlines() if ':' in r}
            f.write(canonical(dict(t=time.time(),rss_kib=int(vals.get('VmRSS','0 kB').split()[0]),
                                   locked_kib=int(vals.get('VmLck','0 kB').split()[0]),
                                   pinned_kib=int(vals.get('VmPin','0 kB').split()[0])))+'\n'); f.flush(); time.sleep(.5)


def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('client');c.add_argument('--fixtures',required=True);c.add_argument('--out',required=True);c.add_argument('--url',default='http://127.0.0.1:8022/v1');c.add_argument('--canary',action='store_true')
    c=sub.add_parser('validate');c.add_argument('--root',required=True);c.add_argument('--boot',required=True);c.add_argument('--arm',choices=('A','B','C','D'),required=True);c.add_argument('--json')
    c.add_argument('--experiment',choices=EXPERIMENTS,default='r823')
    c=sub.add_parser('decide');c.add_argument('--root',required=True);c.add_argument('--json');c.add_argument('--experiment',choices=EXPERIMENTS,default='r823')
    c=sub.add_parser('preflight');c.add_argument('--fixtures',required=True)
    c=sub.add_parser('hardware')
    for k in ('smi','host','client','log','out'): c.add_argument('--'+k,required=True)
    c=sub.add_parser('host-sample');c.add_argument('--pid',type=int,required=True);c.add_argument('--out',required=True)
    a=ap.parse_args()
    if a.cmd=='client': return run_client(a)
    if a.cmd=='host-sample': return host_sample(a)
    if a.cmd=='hardware': hardware(a);return 0
    if a.cmd=='preflight':
        m,_=load_fixtures(a.fixtures);print(f'R823 fixtures OK: {m["request_count"]} measured + {m["canary_count"]} canary requests');return 0
    if a.cmd=='validate':
        out=validate(a.root,a.boot,a.arm,a.experiment)
        if a.json: dump(a.json,{k:v for k,v in out.items() if k!='joined'})
        print(out['status']+': '+'; '.join(out['errors']));return int(out['status']=='INVALID')
    try: out=decide(a.root,a.experiment)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as e:
        out=dict(verdict='INCOMPLETE',real_cause='OPEN',errors=[f'missing/malformed unit evidence: {e}'])
    if a.json: dump(a.json,out)
    print(json.dumps(out,indent=2,allow_nan=False));print(a.experiment.upper()+' VERDICT: '+out['verdict']);return 0

if __name__=='__main__': sys.exit(main())
