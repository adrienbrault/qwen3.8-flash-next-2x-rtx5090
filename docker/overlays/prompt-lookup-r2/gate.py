#!/usr/bin/env python3
"""R827 strict served clients, identity, replay parsers and pre-registered gate. Stdlib only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import statistics as st
import time
import urllib.request

P = Path(__file__).resolve().parent
LIVE_MD5 = '4d0764cf3d7362b1f4fd17dfb7be8e42'  # Public fixture pin; historical on-box pin is recorded in the write-up.
PARENT = 'tabbyapi:r825c-hostprepare'
PARENT_ID = 'sha256:aa04a1cbe94b60bbd91e3d69d15f5e1b6eb679f14885e599404de218d8a897c9'
IMAGE = 'tabbyapi:r827-prompt-lookup'
COUNTERS = ('decode_steps', 'lookup_checks', 'lookup_hits', 'lookup_proposed', 'lookup_accepted')
TAGS = ('OFF1', 'ON1', 'ON2', 'OFF2')

def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def env_map(lines):
    pairs = [x.split('=', 1) for x in lines if x.startswith('EXL3_')]
    assert all(len(x) == 2 for x in pairs), 'malformed env'
    assert len(dict(pairs)) == len(pairs), 'duplicate env'
    return dict(pairs)

def selectors(text):
    matches = re.findall(r'^EXTRA_ENV=\$\{EXTRA_ENV:-(.*)\}$', text, re.M)
    assert len(matches) == 1, 'selector assignment'
    items = matches[0].split()
    pairs = [v.split('=', 1) for v in items]
    assert all(len(v) == 2 for v in pairs) and len(dict(pairs)) == len(pairs)
    return dict(pairs)

def prepare(live, out):
    text = Path(live).read_text()
    assert hashlib.md5(text.encode()).hexdigest() in (LIVE_MD5, '262e9c31f714724409635fbac9df8ac2'), 'foreign live launcher'
    expected = env_map((P/'fixtures/live-container-env.txt').read_text().splitlines())
    defaults = selectors(text)
    assert len(defaults) == 46 and all(expected.get(k) == v for k,v in defaults.items())
    assert 'EXL3_PROMPT_LOOKUP' not in defaults
    assert text.count('DAILY_IMG=' + PARENT + '\n') == 1
    # The live launcher pins its image by full ID (R825C_IMAGE_ID, checked before boot): repin it to the candidate.
    pin = re.findall(r'^R825C_IMAGE_ID=(sha256:[0-9a-f]{64})$', text, flags=re.M)
    assert len(pin) == 1, 'live launcher must carry exactly one R825C_IMAGE_ID pin'
    cand_id = re.findall(r'^R827_IMAGE_ID=(sha256:[0-9a-f]{64})$', (P/'IMAGE_ID.env').read_text(), flags=re.M)
    assert len(cand_id) == 1, 'IMAGE_ID.env must carry the operator-built R827 image ID'
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    launchers = {}
    for flag in ('0', '1'):
        cand = text.replace('DAILY_IMG=' + PARENT + '\n', 'DAILY_IMG=' + IMAGE + '\n')
        cand = cand.replace('R825C_IMAGE_ID=' + pin[0] + '\n', 'R825C_IMAGE_ID=' + cand_id[0] + '\n')
        cand = re.sub(r'^(EXTRA_ENV=\$\{EXTRA_ENV:-.*)(\})$',
                      lambda m: m[1] + ' EXL3_PROMPT_LOOKUP=' + flag + m[2], cand, flags=re.M)
        assert len(selectors(cand)) == 47
        path = out / ('launch-lookup-' + flag + '.sh')
        path.write_text(cand)
        launchers[flag] = {'path': str(path), 'md5': hashlib.md5(cand.encode()).hexdigest()}
    a = (out/'launch-lookup-0.sh').read_text()
    b = (out/'launch-lookup-1.sh').read_text()
    assert a.replace('EXL3_PROMPT_LOOKUP=0', 'EXL3_PROMPT_LOOKUP=1') == b
    dump(out/'launchers.json', launchers)
    print(json.dumps(launchers))

def runtime(inspect, config, launcher, flag, image_id, boot=None):
    d = json.loads(Path(inspect).read_text())
    assert len(d) == 1
    d = d[0]
    assert d['State']['Status'] == 'running' and d['RestartCount'] == 0
    assert d['Image'] == image_id
    assert d['Config']['Image'] == (PARENT if flag == 'live' else IMAGE)
    expected = env_map((P/'fixtures/live-container-env.txt').read_text().splitlines())
    if flag != 'live': expected['EXL3_PROMPT_LOOKUP'] = flag
    actual = env_map(d['Config']['Env'])
    assert actual == expected, f'EXL3 env differs from LIVE capture: {actual.items() ^ expected.items()}'
    text = Path(launcher).read_text()
    assert len(selectors(text)) == (46 if flag == 'live' else 47)
    assert all(actual[k] == v for k,v in selectors(text).items())
    for mount in d['Mounts']:
        assert not mount['Destination'].startswith(('/opt/venv/lib/python3.12/site-packages/exllamav3', '/app/backends', '/app/common')), 'source bind overlay'
    c = Path(config).read_text()
    for key, value in {'cache_size': '901120', 'cache_mode': '8,8', 'max_batch_size': '8',
                       'gpu_split': '[30, 30]', 'draft_mode': 'mtp', 'draft_num_tokens': '3',
                       'draft_gpu_split': '[0, 32]', 'draft_num_tokens_by_batch': '[[4, 3], [8, 2]]',
                       'dynamic_draft': 'false'}.items():
        found = re.findall(r'^\s*' + key + r':\s*(.*?)\s*(?:#.*)?$', c, re.M)
        assert found == [value], (key, found, value)
    if boot:
        b = Path(boot).read_text()
        assert re.search(r'env keys \(47\):', b), 'missing resolved env log'
        assert 'FASTWARM ok' in b and 'FASTWARM FAILED' not in b, 'R818 warm-up failure'
    print(f'R827 RUNTIME PASS {flag}: {len(actual)} exact EXL3 env entries')

def joined(text):
    # Rich wraps long log lines at >=10 spaces; support docker timestamp prefixes too.
    return re.sub(r'\n(?:\d{4}-\S+Z )? {10,}', ' ', text)

def completion_lines(text):
    text = joined(text)
    rows = {}
    for line in text.splitlines():
        if re.search(r'#\d+ (?:chat/)?completions.*tokens generated', line):
            m = re.search(r'#(\d+) (?:chat/)?completions(?: \(stream\))?: ([\d,]+) tokens generated.*?draft ([\d,]+)/([\d,]+) accepted', line)
            assert m, f'unparsed completion line: {line[:160]}'
            serial,n,acc,prop = [int(v.replace(',', '')) for v in m.groups()]
            assert serial not in rows and 0 <= acc <= prop
            rows[serial] = dict(serial=serial, gen_tokens=n, draft_accept=acc, draft_proposed=prop)
    assert rows, 'no served completion lines'
    return rows

def counter_lines(text):
    rows = {}
    for line in joined(text).splitlines():
        if 'R827_COUNTER ' not in line: continue
        raw = line.split('R827_COUNTER ',1)[1]
        d, end = json.JSONDecoder().raw_decode(raw)
        assert not raw[end:].strip(), 'trailing counter data'
        serial = int(re.search(r'#(\d+)', d['label'])[1])
        assert serial not in rows, 'duplicate logical-request counters'
        for key in COUNTERS:
            assert type(d[key]) is int and d[key] >= 0, key
        assert d['decode_steps'] > 0 and d['gen_tokens'] > 0 and d['gen_time'] > 0
        assert d['lookup_hits'] <= d['lookup_checks'] <= d['decode_steps']
        assert d['lookup_accepted'] <= d['lookup_proposed'] <= 2*d['lookup_hits']
        rows[serial] = d
    return rows

def cell(log, lo, expected, flag, out, parent=False, measured=None):
    text = Path(log).read_text()
    for marker in ('Traceback', 'OutOfMemoryError', 'c10::Error', 'CUDA error', 'EXL3_PREFILL_MERGE disabled'):
        assert marker not in text, marker
    base = {k:v for k,v in completion_lines(text).items() if k > lo}
    assert len(base) == expected, f'foreign/missing completion lines: {len(base)} != {expected}'
    counters = {k:v for k,v in counter_lines(text).items() if k > lo}
    if not parent:
        assert set(counters) == set(base), 'missing counter/serial join'
        for serial, d in counters.items():
            assert d['gen_tokens'] == base[serial]['gen_tokens']
            if flag == '0': assert all(d[k] == 0 for k in COUNTERS if k != 'decode_steps'), 'OFF lookup activity'
    else:
        # The unpatched daily has no lookup counters. Its real parser still runs;
        # candidate counter parser is additionally exercised by selftest's replay.
        assert not counters, 'dry-run parent unexpectedly patched'
    measured = expected if measured is None else measured
    assert 0 < measured <= expected
    selected = sorted(base)[-measured:]  # warmups finish before measurement dispatch
    counters = {k:counters[k] for k in selected} if counters else {}
    data = {'n':len(base), 'measured_n':measured, 'serials':sorted(base),
            'measured_serials':selected, 'counters':list(counters.values())}
    data['stream_metrics'] = [dict(label=d['label'], ms_step=1000*d['gen_time']/d['decode_steps'],
        tokens_step=d['gen_tokens']/d['decode_steps'],
        proposal_hit_rate=d['lookup_hits']/d['lookup_checks'] if d['lookup_checks'] else 0,
        accepted_lookup_tokens_step=d['lookup_accepted']/d['decode_steps']) for d in counters.values()]
    if counters:
        steps = sum(d['decode_steps'] for d in counters.values())
        checks = sum(d['lookup_checks'] for d in counters.values())
        hits = sum(d['lookup_hits'] for d in counters.values())
        data.update(ms_step=1000*sum(d['gen_time'] for d in counters.values())/steps,
                    tokens_step=sum(d['gen_tokens'] for d in counters.values())/steps,
                    proposal_hit_rate=hits/checks if checks else 0,
                    accepted_lookup_tokens_step=sum(d['lookup_accepted'] for d in counters.values())/steps)
    dump(out, data)
    print(json.dumps({k:v for k,v in data.items() if k != 'counters'}))

def consume(lines):
    """Strict stream replay; channel bytes independent of speculative frame grouping."""
    parts = {'text': [], 'content': [], 'reasoning_content': [], 'reasoning': [], 'tool_calls': []}
    usage = None; finish = None; first = last = None; done = False; frames = 0
    for raw in lines:
        line = raw.decode('utf-8') if isinstance(raw,bytes) else raw
        line = line.strip()
        if not line or line.startswith(':'): continue
        assert line.startswith('data:'), f'unknown SSE line: {line[:80]}'
        data = line[5:].strip()
        if data == '[DONE]': done=True; break
        d = json.loads(data)
        assert not d.get('error'), d
        assert 'choices' in d, 'missing choices'
        if d.get('usage') is not None:
            usage = d['usage']; assert type(usage['completion_tokens']) is int
            assert type(usage['prompt_tokens']) is int
        visible = False
        for ch in d['choices']:
            assert ch['index'] == 0, 'unexpected choice'
            delta = ch.get('delta') or ch.get('message') or {}
            parts['text'].append(ch.get('text') or '')
            for k in ('content','reasoning_content','reasoning'):
                parts[k].append(delta.get(k) or '')
            if delta.get('tool_calls'): parts['tool_calls'].extend(delta['tool_calls'])
            visible |= bool(ch.get('text') or any(delta.get(k) for k in ('content','reasoning_content','reasoning','tool_calls')))
            finish = ch.get('finish_reason') or finish
        if visible:
            stamp = time.perf_counter(); first = first or stamp; last = stamp; frames+=1
    assert done and usage and finish and first is not None, 'truncated or empty SSE/usage/finish'
    assert usage['completion_tokens'] >= frames and usage['completion_tokens'] > 0
    value = {k: ''.join(v) for k,v in parts.items() if k!='tool_calls'}
    # IDs in tool calls vary per response; function name/arguments are the identity fields.
    value['tool_calls'] = [(t.get('index',0), t['function']['name'], t['function']['arguments']) for t in parts['tool_calls']]
    value['finish'] = finish
    return value, usage, first, last, frames

def request(url, body, path):
    t0 = time.perf_counter()
    req = urllib.request.Request(url+path, json.dumps(body).encode(), {'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=900) as response:
        value, usage, first, last, frames = consume(response)
    n = usage['completion_tokens']; window=last-first
    return dict(value=value, completion_tokens=n, prompt_tokens=usage['prompt_tokens'], client_frames=frames,
                ttft_s=first-t0, wall_s=time.perf_counter()-t0, decode_window_s=window,
                decode_tps=(n-1)/window if window>0 and n>1 else None,
                sha256=hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest(), ok=True)

def agent_prompts():
    """R501's real-file rewrite shape, twelve fixed requests; no glob of live files."""
    files = sorted((P/'fixtures/agent-sources').glob('*.py'))
    assert len(files)==6
    rows=[]
    for i,f in enumerate(files):
        source=''.join(f.read_text().splitlines(keepends=True)[:160])
        for edit in (0,1):
            context='\n\n'.join(f'Existing revision {v} of {f.name}:\n```python\n{source}```' for v in range(3))
            change=('append the comment `# reviewed` to the final line' if edit==0 else
                    'insert the comment `# reviewed` immediately before the final line')
            prompt=f'{context}\n\nRewrite the COMPLETE last revision with one small edit: {change}. Keep every other byte. Output only one Python code block.'
            rows.append(dict(pid=f'edit-{i}-{edit}',prompt=prompt,source_sha256=hashlib.sha256(source.encode()).hexdigest()))
    return rows

def bodies(model, suite, short=False):
    import fn_greedy, chat_greedy
    nt = 8 if short else 256
    out=[]
    if suite == 'fn':
        prompts=[(f'short{i}',s) for i,s in enumerate(fn_greedy.SHORT)]+[('long100k',fn_greedy.long_prompt())]
        for pid,prompt in prompts:
            out.append((pid,'/completions',dict(model=model,prompt=prompt,temperature=0,seed=0,max_tokens=nt)))
    elif suite=='chat':
        for pid,msg in {**chat_greedy.PROMPTS,**chat_greedy.LONG}.items():
            b=dict(model=model,messages=msg,temperature=0,max_tokens=8 if short else (5000 if pid=='long' else 512))
            if pid=='tool': b['tools']=[chat_greedy.TOOL]
            if pid=='long' and not short: b['min_tokens']=5000
            out.append((pid,'/chat/completions',b))
    elif suite=='agent':
        for r in agent_prompts():
            b=dict(model=model,messages=[{'role':'user','content':r['prompt']}],temperature=0,
                   enable_thinking=False,max_tokens=8 if short else 3072,min_tokens=8 if short else 3072,
                   loop_detect_window=0)
            out.append((r['pid'],'/chat/completions',b))
    else: raise ValueError(suite)
    for _,_,b in out:
        b.update(stream=True,stream_options={'include_usage':True})
    return out

def run_client(url,model,tag,suite,out,conc=1,runs=1,warmup=0,short=False):
    requests=bodies(model,suite,short)
    # Identity crosses requeue at 3072; speed uses 2048 to keep four boots <=75 min.
    if suite=='agent' and (runs>1 or warmup>0) and not short:
        for _,_,body in requests:
            body['min_tokens']=body['max_tokens']=2048
    assert len(requests)==(12 if suite=='agent' else 6)
    if suite!='agent': assert conc==1 and runs==1 and warmup==0
    with open(out,'a',buffering=1) as f:
        for run in range(-warmup,runs):
            for offset in range(0,len(requests),conc):
                batch=requests[offset:offset+conc]; start=time.perf_counter()
                with ThreadPoolExecutor(max_workers=conc) as pool:
                    results=list(pool.map(lambda x: request(url,x[2],x[1]),batch))
                wall=time.perf_counter()-start
                for (pid,path,body),r in zip(batch,results):
                    r.update(tag=tag,pid=pid,suite=suite,conc=conc,run=run,round_wall_s=wall,
                             request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest())
                    if suite=='agent':
                        assert r['completion_tokens']==body['max_tokens'], 'agent forced-length mismatch'
                    # Flush warmups too, marked run=-1 for audit; excluded by gate.
                    f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(f'R827 CLIENT PASS {tag} {suite} c{conc}')

def read_rows(path, require_decode=True):
    rows=[json.loads(l) for l in Path(path).read_text().splitlines()]
    assert rows and all(r['ok'] and not r.get('accounting_note') for r in rows), 'failed/accounting client'
    for r in rows:
        if 'server_completion_tokens' in r:
            assert r['server_completion_tokens']==r['completion_tokens']
            assert r['finish_reason'] and type(r['prompt_tokens']) is int
        assert r['completion_tokens']>0 and r['client_frames']>0
        assert r['client_frames']<=r['completion_tokens']
        if require_decode:
            assert r['decode_tps'] is not None and r['decode_tps']>0, 'missing decode measurement'
    return rows

# R827b (user 2026-10-02, option 1): lookup changes the draft tokens in the verify batch, which can flip greedy
# near-ties (R827 ON1: 1/24, edit-3-1 at char 2,132, a comment placed one line later). Identity is now a measured
# divergence count against OFF1, pre-registered before R827b data: OFF2 must equal OFF1 on 24/24 (the served config
# is deterministic at c1; otherwise the harness is unstable and the run is INVALID), and each ON boot may diverge
# from OFF1 on at most MAX_ON_DIVERGENCE of the 24 identity prompts.
MAX_ON_DIVERGENCE = 2

def first_diff(a, b):
    a, b = json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True)
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))

def identity(root, tags=TAGS):
    root=Path(root); result={'diverged':{}, 'pass':True}
    for suite,n in [('fn',6),('chat',6),('agent',12)]:
        data={}
        for tag in tags:
            rs=read_rows(root/f'identity-{tag}-{suite}.jsonl', require_decode=False)
            assert len(rs)==n and len({r['pid'] for r in rs})==n, (suite,tag)
            data[tag]={r['pid']:r for r in rs}
        ref=data['OFF1']
        for tag in tags[1:]:
            assert set(data[tag])==set(ref)
            for pid,r in ref.items():
                c=data[tag][pid]
                assert c['request_sha256']==r['request_sha256'], 'request bytes drift'
                if c['value']!=r['value'] or c['completion_tokens']!=r['completion_tokens']:
                    result['diverged'].setdefault(tag, []).append(dict(suite=suite, pid=pid, first_diff_char=first_diff(r['value'], c['value']),
                                                                      tokens=[r['completion_tokens'], c['completion_tokens']]))
    for tag in tags[1:]:
        k = len(result['diverged'].get(tag, []))
        ok = k == 0 if tag.startswith('OFF') else k <= MAX_ON_DIVERGENCE
        result[f'{tag}_vs_OFF1'] = f'{24-k}/24 identical' + ('' if ok else ' FAIL')
        result['pass'] &= ok
    return result

def report(root):
    root=Path(root); ids=identity(root); summary={}; gate=ids['pass']
    for shape in [f'c{c}-{k}' for c in (1,2,3) for k in ('code','prose')]+['agent-c1']:
        per_boot={}
        for tag in TAGS:
            if shape=='agent-c1':
                rows=read_rows(root/f'speed-{tag}-agent.jsonl')
                rows=[r for r in rows if r['run']>=0]
                assert all(r['completion_tokens']==2048 and r['conc']==1 and r['tag']==tag for r in rows)
                assert len(rows)==36 and {(r['run'],r['pid']) for r in rows}=={(n,p['pid']) for n in range(3) for p in agent_prompts()}
            else:
                rows=read_rows(root/f'fn-{tag}-{shape}.jsonl')
                c=int(shape[1]); assert len(rows)==3*c
                assert {(r['run'],r['i']) for r in rows}=={(n,i) for n in range(3) for i in range(c)}
                assert all(r['completion_tokens']==1024 and r['tag']==f'{tag}-{shape}' and r['conc']==c for r in rows)
            per_boot[tag]=st.median(r['decode_tps'] for r in rows)
            counter=json.loads((root/f'counters-{tag}-{shape}.json').read_text())
            assert counter['n']==(48 if shape=='agent-c1' else 5*int(shape[1]))
            assert counter['measured_n']==(36 if shape=='agent-c1' else 3*int(shape[1]))
            assert counter['counters'] and all(k in counter for k in ('ms_step','tokens_step','proposal_hit_rate','accepted_lookup_tokens_step'))
            if tag.startswith('ON') and shape=='agent-c1':
                assert counter['accepted_lookup_tokens_step']>0, 'agent speed lookup did not fire/accept'
        # Agent: EACH ON must beat EACH OFF by >=2%, not just pooled medians.
        pairs=[('ON1','OFF1'),('ON2','OFF2')]
        if shape=='agent-c1': pairs=[(a,b) for a in ('ON1','ON2') for b in ('OFF1','OFF2')]
        gains={f'{a}/{b}':per_boot[a]/per_boot[b]-1 for a,b in pairs}
        passed=all(v>= (0.02 if shape=='agent-c1' else -0.01) for v in gains.values())
        gate &= passed
        summary[shape]=dict(decode_stream_tps=per_boot,paired_gain=gains,pass_gate=passed,
                            counters={tag:json.loads((root/f'counters-{tag}-{shape}.json').read_text()) for tag in TAGS})
    for tag in ('ON1','ON2'):
        d=json.loads((root/f'counters-{tag}-identity-agent.json').read_text())
        assert sum(r['lookup_hits'] for r in d['counters'])>0 and sum(r['lookup_accepted'] for r in d['counters'])>0, 'identity agent lookup inactive'
    decision='PROMOTE-ELIGIBLE' if gate else 'REJECT'
    data=dict(identity=ids,shapes=summary,DECISION=decision,promotion=False)
    dump(root/'decision.json',data)
    print(json.dumps(ids)); print('DECISION '+decision+' (user decides; no promotion)')
    return data

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--live',required=True); p.add_argument('--out',required=True)
    p=sub.add_parser('runtime')
    for k in ('inspect','config','launcher','flag','image-id'): p.add_argument('--'+k,required=True)
    p.add_argument('--boot')
    p=sub.add_parser('serial'); p.add_argument('--log',required=True)
    p=sub.add_parser('cell')
    for k in ('log','out','flag'): p.add_argument('--'+k,required=True)
    p.add_argument('--measured',type=int); p.add_argument('--lo',type=int,required=True); p.add_argument('--expected',type=int,required=True); p.add_argument('--parent',action='store_true')
    p=sub.add_parser('client')
    for k in ('url','model','tag','suite','out'): p.add_argument('--'+k,required=True)
    p.add_argument('--conc',type=int,default=1); p.add_argument('--runs',type=int,default=1)
    p.add_argument('--warmup',type=int,default=0); p.add_argument('--short',action='store_true')
    p=sub.add_parser('validate'); p.add_argument('--rows',required=True); p.add_argument('--identity',action='store_true')
    p=sub.add_parser('check-identity'); p.add_argument('--root',required=True); p.add_argument('--tag',required=True)
    p=sub.add_parser('report'); p.add_argument('--root',required=True)
    a=vars(ap.parse_args()); cmd=a.pop('cmd')
    if cmd=='check-identity':
        print(json.dumps(identity(a['root'],('OFF1',a['tag'])))); return
    if cmd=='serial':
        text=joined(Path(a['log']).read_text()); print(max([int(x) for x in re.findall(r'#(\d+) (?:chat/)?completions',text)]+[0])); return
    if cmd=='validate': print(f'R827 ROWS PASS {len(read_rows(a["rows"], require_decode=not a["identity"]))}'); return
    {'prepare':prepare,'runtime':runtime,'cell':cell,'client':run_client,'report':report}[cmd](**a)

if __name__=='__main__': main()
