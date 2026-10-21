#!/usr/bin/env python3
"""Materialize exact upstream inputs and apply only the audited candidate patches."""
import hashlib,io,json,subprocess,tarfile,urllib.request,shutil
from pathlib import Path
root=Path(__file__).resolve().parent
m=json.loads((root/'manifest.json').read_text());out=root/'out';out.mkdir(exist_ok=True)
for filename, expected in m['patch_sha256'].items():
 assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==expected, filename
for name in ['engine','tabby']:
 entry=m[name];archive=out/(name+'.tar.gz')
 if not archive.exists():archive.write_bytes(urllib.request.urlopen('https://codeload.github.com/'+entry['repo']+'/tar.gz/'+entry['commit'],timeout=120).read())
 data=archive.read_bytes();assert hashlib.sha256(data).hexdigest()==entry['archive_sha256']
 target=out/name
 if target.exists():shutil.rmtree(target)
 target.mkdir()
 with tarfile.open(fileobj=io.BytesIO(data)) as t:
  for member in t.getmembers():
   parts=member.name.split('/',1)
   if len(parts)!=2 or not parts[1]:continue
   member.name=parts[1];t.extract(member,target,filter='data')
 subprocess.run(['git','apply','--unsafe-paths',str(root/(name+'.patch'))],cwd=target,check=True)
wheel=out/m['wheel']['filename']
if not wheel.exists():
 with urllib.request.urlopen(m['wheel']['url'],timeout=120) as src,wheel.open('wb') as dst:shutil.copyfileobj(src,dst)
assert hashlib.file_digest(wheel.open('rb'),'sha256').hexdigest()==m['wheel']['sha256']
print('Prepared pinned sources and wheel')
