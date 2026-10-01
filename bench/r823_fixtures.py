#!/usr/bin/env python3
"""Immutable synthetic R823 fixtures. uv run --with tokenizers --with jinja2 ...
All trimmed pieces are decoded/re-encoded; all final IDs and LCPs are asserted.
"""
import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import re

SEED = 823
MODEL = 'qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab'
TOOLS = [{'type':'function','function':{'name':'inspect_records','description':'Inspect synthetic files',
          'parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path']}}}]
SAMPLER = dict(model=MODEL,stream=True,stream_options={'include_usage':True},temperature=0,
               top_k=0,top_p=1,min_p=0,seed=823,min_tokens=64,max_tokens=64,
               loop_detect_window=0,add_bos_token=False,token_healing=False)
SIZES = {'T35':[54465,56647,58364,59450,61765,63852,65627,69914],
         'T32':[67618,69208,71685,73685,74832,77809,79644,81615],
         'T65':[88703,88823,90050,90191,90496,91608]}
D = {'T35':48000,'T32':60000,'T65':81000,'C2':60000}


def canonical(obj):
    return json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def lcp(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x != y),min(len(a),len(b)))


class Builder:
    def __init__(self,model):
        from tokenizers import Tokenizer
        from jinja2 import Environment
        self.model = Path(model)
        self.tok = Tokenizer.from_file(str(self.model/'tokenizer.json'))
        env = Environment(trim_blocks=True,lstrip_blocks=True)
        env.globals['raise_exception'] = lambda s: (_ for _ in ()).throw(ValueError(s))
        # Compact JSON, insertion order. The server re-serializes tools through its pydantic ToolSpec, so
        # render() passes them in that model's field order (R823 try 3: 45/113 chat fixtures differed).
        env.policies['json.dumps_kwargs'] = {'ensure_ascii':False,'sort_keys':False}
        self.template = env.from_string((self.model/'chat_template.jinja').read_text())
        rng = random.Random(SEED)
        words = ['parser','record','checksum','queue','worker','buffer','anchor','comment','result','synthetic']
        self.pool = ''.join(f'// src/file_{i%29}.py row {i:05d}: {rng.choice(words)}={rng.randrange(100000):05d}; '
                            f'mock_result status={rng.choice(["ready","pending","verified"])}\n' for i in range(18000))
        self.pool_ids = self.encode(self.pool)
        self.rows=[]
        self.previous={}
        self.payloads={}

    def encode(self,s):
        return self.tok.encode(s,add_special_tokens=False).ids

    def decode(self,ids):
        s=self.tok.decode(ids,skip_special_tokens=False)
        assert self.encode(s)==ids,'round-trip failure'
        return s

    def tools(self,messages):
        tools=copy.deepcopy(TOOLS)
        root=re.search(r'\[root:([^\]]+)\]',messages[0]['content'])[1]
        tools[0]['function']['description']=root+' Inspect synthetic files'
        return tools

    def render(self,messages):
        # TabbyAPI ToolSpec(function: Function(name, description, parameters), type): that field order.
        served=[{'function':{k:t['function'][k] for k in ('name','description','parameters')},'type':t['type']}
                for t in self.tools(messages)]
        return self.template.render(messages=messages,tools=served,add_generation_prompt=True,
                                    enable_thinking=False,preserve_thinking=True,bos_token='',eos_token='<|im_end|>')

    def piece(self,n):
        assert n>=0
        return self.decode(self.pool_ids[:n])

    def ledger(self,root):
        return [{'role':'system','content':f'[root:{root}] synthetic repository inspection; summarize only.'},
                {'role':'user','content':'Inspect the mock records and source comments.'},
                {'role':'assistant','content':'','tool_calls':[{'id':'mock0','type':'function',
                    'function':{'name':'inspect_records','arguments':{'path':'src/mock.py'}}}]},
                {'role':'tool','tool_call_id':'mock0','content':'PLACEHOLDER'},
                {'role':'assistant','content':'Synthetic records loaded.'},
                {'role':'user','content':'Summarize the records in one paragraph.'}]

    def fit(self,messages,target):
        # Fit inside the tool content, preserving immutable transcript delimiters.
        m=copy.deepcopy(messages)
        original=m[3]['content']
        current=len(self.encode(self.render(m)))
        assert current<=target,(current,target)
        m[3]['content']=original+'\n'+self.piece(max(0,target-current-12))+'\n'
        for _ in range(8):
            delta=target-len(self.encode(self.render(m)))
            if delta==0:
                return m
            if delta>0:
                m[3]['content']+='7'*delta
            else:
                ids=self.encode(m[3]['content'])
                m[3]['content']=self.decode(ids[:delta])
        raise ValueError('could not fit exact N')

    def exact(self,root,n,d=None,value='0'):
        m=self.ledger(root)
        if d is None:
            m[3]['content']='mock results\n'
            return self.fit(m,n)
        m[3]['content']='\nR823_BOUNDARY\n'
        prefix=self.render(m).split('R823_BOUNDARY')[0]
        # Position digit at d; newline isolates the numeric marker from BPE merges.
        need=d-len(self.encode(prefix))-len(self.encode('R823_BOUNDARY\n'))
        m[3]['content']=self.piece(max(0,need-8))+'\nR823_BOUNDARY\n'
        for _ in range(12):
            pre=self.render(m).split('R823_BOUNDARY')[0]+'R823_BOUNDARY\n'
            delta=d-len(self.encode(pre))
            if delta==0: break
            if delta>0: m[3]['content']=m[3]['content'].replace('\nR823_BOUNDARY','7'*delta+'\nR823_BOUNDARY')
            else:
                head=m[3]['content'].split('\nR823_BOUNDARY')[0]
                m[3]['content']=self.decode(self.encode(head)[:delta])+'\nR823_BOUNDARY\n'
        assert len(self.encode(self.render(m).split('R823_BOUNDARY')[0]+'R823_BOUNDARY\n'))==d
        m[3]['content']+=value+'\n'
        return self.fit(m,n)

    def add(self,case,session,path,kind,turn,m,expected=None,pair=None):
        prompt=self.render(m)
        ids=self.encode(prompt)
        assert self.decode(ids)==prompt
        prev=self.previous.get(session)
        actual=None if prev is None else lcp(prev,ids)
        if expected is not None: assert actual==expected,(case,turn,actual,expected)
        body=dict(SAMPLER)
        if path=='chat':
            # API tool arguments are JSON strings; template receives parsed mappings.
            request_m=copy.deepcopy(m)
            for msg in request_m:
                for call in msg.get('tool_calls',[]):
                    call['function']['arguments']=canonical(call['function']['arguments'])
            body.update(messages=request_m,tools=self.tools(m),enable_thinking=False,
                        template_vars={'enable_thinking':False,'preserve_thinking':True},logprobs=True,top_logprobs=5)
        else:
            body.update(prompt=prompt,logprobs=5,top_logprobs=5)
        payload=canonical(body)
        h=sha(payload.encode())
        self.payloads.setdefault(h,dict(body=payload,token_ids=ids,prompt=prompt,messages=m))
        key=f'r823/{path}/{case}/{session.rsplit("/",1)[-1]}/{turn}/{kind}'
        starts=[]
        for i,msg in enumerate(m):
            content=msg.get('content','')
            at=prompt.find(content) if content else -1
            starts.append(dict(index=i,role=msg['role'],content_start=len(self.encode(prompt[:at])) if at>=0 else None))
        self.rows.append(dict(key=key,case=case,session=session,path=path,kind=kind,turn=turn,
                              payload=h,n=len(ids),lcp=actual,d=expected,pair=pair,boundaries=starts))
        self.previous[session]=ids
        return key

    def edited(self,case,session,path):
        sizes=SIZES[case] if case!='C2' else [67618,69208,71685,74832,77809,81615]
        d=D[case]
        if case=='T35':
            # Exact warm prefix of the long seed; hence a 2,560-token resume.
            seed=self.exact(session,sizes[0],d)
            text=self.decode(self.encode(self.render(seed))[:2689])
            # raw warm request is deliberately a trimmed serialized transcript.
            body=dict(SAMPLER,prompt=text,logprobs=5)
            h=sha(canonical(body).encode()); self.payloads[h]=dict(body=canonical(body),token_ids=self.encode(text),prompt=text,messages=[])
            self.rows.append(dict(key=f'r823/raw/T35/{session.rsplit("/",1)[-1]}/-1/warm',case=case,session=session,path=path,kind='warm',turn=-1,payload=h,n=2689,lcp=None,d=None,pair=None,boundaries=[]))
            self.previous[session]=self.encode(text)
        m=self.exact(session,sizes[0],d)
        self.add(case,session,path,'seed',0,m)
        for i,n in enumerate(sizes,1):
            m=self.exact(session,n,d,str(i))
            key=self.add(case,session,path,'edit',i,m,d)
            self.add(case,session,path,'hit',i,m,n,key)

    def append(self,session,path,canary=False):
        case='CANARY' if canary else 'APPEND'
        m=self.exact(session,4096 if canary else 50000)
        self.add(case,session,path,'seed',0,m)
        for i in range(1,3 if canary else 11):
            prior=self.encode(self.render(m))
            m += [{'role':'assistant','content':'Pre-generated synthetic summary.'},
                  {'role':'user','content':self.piece(3100 if not canary else 200)}]
            key=self.add(case,session,path,'append',i,m,len(prior))
            if canary: self.add(case,session,path,'hit',i,m,len(self.encode(self.render(m))),key)
        if not canary: assert 80000<=len(self.encode(self.render(m)))<=90000

    def moving(self,session,path):
        m=self.exact(session,50000)
        self.add('MOVING-EDIT',session,path,'seed',0,m)
        for i in range(1,11):
            prev=self.encode(self.render(m)); d=len(prev)-8000
            # Replace a single digit in the latest (growing) synthetic tool result.
            while not self.decode([prev[d]]).isdigit(): d+=1
            tool=m[3]['content']; rendered=self.render(m); start=rendered.index(tool)
            at=len(self.encode(rendered[:start])); local=d-at
            tids=self.encode(tool); before=self.decode(tids[:local]); after=self.decode(tids[local+1:])
            old=self.decode([tids[local]]); value=str((int(old)+1)%10)
            m[3]['content']=before+value+after
            m=self.fit(m,len(prev)+3000)
            key=self.add('MOVING-EDIT',session,path,'edit',i,m,d)
            self.add('MOVING-EDIT',session,path,'hit',i,m,len(self.encode(self.render(m))),key)

    def early(self,session):
        m=self.exact(session,60000)
        self.add('EARLY-EDIT',session,'raw','seed',0,m)
        for i in range(1,3):
            m[0]['content']=f'{i} early system marker '+m[0]['content']
            key=self.add('EARLY-EDIT',session,'raw','edit',i,m)
            assert self.rows[-1]['lcp']<2048
            self.add('EARLY-EDIT',session,'raw','hit',i,m,len(self.encode(self.render(m))),key)

    def build(self,nonce):
        for case in ['T35','T32','T65']:
            self.edited(case,f'{nonce}/raw/{case}/s0','raw')
        self.append(f'{nonce}/raw/APPEND/s0','raw')
        self.moving(f'{nonce}/raw/MOVING-EDIT/s0','raw')
        self.early(f'{nonce}/raw/EARLY-EDIT/s0')
        for s in ['s0','s1']: self.edited('C2',f'{nonce}/raw/C2/{s}','raw')
        for case in ['T32','APPEND','MOVING-EDIT']:
            session=f'{nonce}/chat/{case}/s0'
            if case=='T32': self.edited(case,session,'chat')
            elif case=='APPEND': self.append(session,'chat')
            else: self.moving(session,'chat')
        # C2 chat is fixed in the archive; runner requires verified post-render joins.
        for s in ['s0','s1']: self.edited('C2',f'{nonce}/chat/C2/{s}','chat')
        self.append(f'{nonce}/raw/CANARY/s0','raw',True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--nonce',default='synthetic-823-v1')
    a=ap.parse_args(); b=Builder(a.model); b.build(a.nonce)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    assets={p.name:sha(p.read_bytes()) for p in b.model.iterdir() if p.is_file()}
    manifest=dict(seed=SEED,nonce=a.nonce,assets=assets,requests=b.rows,
                  request_count=sum(r['case']!='CANARY' for r in b.rows),canary_count=sum(r['case']=='CANARY' for r in b.rows))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with open(out/'payloads.json.gz','wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
            z.write(canonical(b.payloads).encode())
    (out/'SHA256SUMS').write_text(''.join(f'{sha((out/n).read_bytes())}  {n}\n' for n in ['manifest.json','payloads.json.gz']))
    print(f'{len(b.rows)} requests; {len(b.payloads)} unique payloads; { (out/"payloads.json.gz").stat().st_size} compressed bytes')

if __name__=='__main__': main()
