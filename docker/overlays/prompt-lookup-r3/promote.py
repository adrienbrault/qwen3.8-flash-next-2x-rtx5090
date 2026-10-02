#!/usr/bin/env python3
"""Read-only post-promotion gates. Atomic install/rollback are owned by the unit."""
import argparse
import json
from pathlib import Path
import statistics
import gate


def verify(root):
    root=Path(root)
    boot=(root/'boot-promote.log').read_text()
    assert 'FASTWARM ok' in boot and 'FASTWARM FAILED' not in boot
    old=json.loads((root/'vram-reference.json').read_text())
    new=json.loads((root/'vram-promote.json').read_text())
    assert len(old)==len(new)==2 and all(type(x) is int and x>=0 for x in old+new)
    assert all(b>=a-32 for a,b in zip(old,new)), f'VRAM free {new} < {old} - 32 MiB'
    off=gate.read_rows(root/'fn-OFF2-c1-code.jsonl')
    on=gate.read_rows(root/'fn-PROMOTE-c1-code.jsonl')
    assert len(off)==len(on)==3 and all(r['completion_tokens']==1024 for r in off+on)
    ratio=statistics.median(r['decode_tps'] for r in on)/statistics.median(r['decode_tps'] for r in off)
    assert ratio>=.99, 'post-promotion fn c1 code below paired -1% floor'
    matches={}
    for suite,n in [('fn',6),('chat',6),('fnstyle',12)]:
        ref={r['pid']:r for r in gate.read_rows(root/f'identity-OFF2-{suite}.jsonl',False)}
        rows=gate.read_rows(root/f'identity-PROMOTE-{suite}.jsonl',False)
        assert len(ref)==len(rows)==n and len({r['pid'] for r in rows})==n
        assert {r['pid'] for r in rows}==set(ref)
        for r in rows:
            b=ref[r['pid']]
            assert r['request_sha256']==b['request_sha256']
            assert r['value']==b['value'] and r['completion_tokens']==b['completion_tokens'], 'post-promotion identity '+suite+'/'+r['pid']
        matches[suite]=f'{n}/{n}'
    result=dict(status='PASS',vram_reference=old,vram_free=new,fn_c1_code_ratio=ratio,identity=matches)
    gate.dump(root/'promotion-gates.json',result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    print(json.dumps(verify(ap.parse_args().root)))
