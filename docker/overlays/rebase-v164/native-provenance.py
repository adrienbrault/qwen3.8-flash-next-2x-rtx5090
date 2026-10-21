#!/usr/bin/env python3
"""Read-only check: installed native extension equals the pinned official wheel member."""
import hashlib,importlib.util,json,zipfile
from pathlib import Path
root=Path('/opt/flashnext-v164')
m=json.loads((root/'manifest.json').read_text());wheel=root/m['wheel']['filename']
assert hashlib.file_digest(wheel.open('rb'),'sha256').hexdigest()==m['wheel']['sha256']
spec=importlib.util.find_spec('exllamav3_ext');assert spec and spec.origin
installed=Path(spec.origin);actual=hashlib.file_digest(installed.open('rb'),'sha256').hexdigest()
with zipfile.ZipFile(wheel) as z:
 members=[x for x in z.namelist() if Path(x).name==installed.name];assert len(members)==1,members
 expected=hashlib.sha256(z.read(members[0])).hexdigest()
assert actual==expected,(actual,expected)
print(json.dumps({'wheel_sha256':m['wheel']['sha256'],'native_path':str(installed),'native_sha256':actual,'matches_pinned_wheel':True},indent=2))
