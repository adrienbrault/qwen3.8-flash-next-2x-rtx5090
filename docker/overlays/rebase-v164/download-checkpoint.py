#!/usr/bin/env python3
"""Download the pinned candidate to NVMe; never changes a serving checkpoint."""
import hashlib,json,sys,os
from pathlib import Path
from huggingface_hub import snapshot_download
variant=sys.argv[1] if len(sys.argv)>1 else '268'
assert variant in ('218','268')
manifest=json.loads(Path(__file__).with_name('checkpoint-218.json' if variant=='218' else 'checkpoint.json').read_text())
target=Path('/srv/qwen5090/models/qwen3.8-flash-next-exl3-'+('2.18' if variant=='218' else '2.68')+'bpw-selfcal')
assert str(target.parent.resolve()).startswith('/srv/qwen5090/fast/'), target.parent.resolve()
patterns=None
if variant=='218':
    reference=json.loads(Path(__file__).with_name('checkpoint.json').read_text())
    shared={e['path']:e for e in reference['files']}
    patterns=[]
    for e in manifest['files']:
        old=shared.get(e['path'])
        candidate=target.parent/'qwen3.8-flash-next-exl3-2.68bpw-selfcal'/e['path']
        if e['sha256'] and old and old['sha256']==e['sha256'] and candidate.is_file():
            target.mkdir(parents=True,exist_ok=True)
            link=target/e['path'];link.parent.mkdir(parents=True,exist_ok=True)
            if link.is_symlink() and link.readlink().is_absolute():link.unlink()
            if not link.exists():link.symlink_to(os.path.relpath(candidate,link.parent))
        else:patterns.append(e['path'])
snapshot_download(manifest['repository'],revision=manifest['revision'],local_dir=str(target),max_workers=4,allow_patterns=patterns)
for entry in manifest['files']:
    p=target/entry['path']
    assert p.is_file(),p
    if entry['size'] is not None: assert p.stat().st_size==entry['size'],p
    if entry['sha256']:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
        assert h.hexdigest()==entry['sha256'],p
print('SUMMARY flashnext-v164-checkpoint download and pinned file hashes PASS',flush=True)
