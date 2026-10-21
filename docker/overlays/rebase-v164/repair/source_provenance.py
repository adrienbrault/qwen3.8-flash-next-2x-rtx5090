#!/usr/bin/env python3
"""Read-only receipt for the source-built recovery image, not the official wheel."""
import hashlib,importlib.metadata,json
from pathlib import Path
import torch,exllamav3,exllamav3_ext
p=Path(exllamav3_ext.__file__)
assert exllamav3_ext.flashnext_decode_capacity==32
assert hasattr(exllamav3_ext,'gr_mix_v2_int8_v3')
print(json.dumps({'kind':'source-built-sm120','engine_version':importlib.metadata.version('exllamav3'),
    'torch':torch.__version__,'torch_cuda':torch.version.cuda,'native_filename':p.name,
    'native_sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),
    'decode_capacity':exllamav3_ext.flashnext_decode_capacity,
    'manifest':json.loads(Path('/opt/flashnext-v164/manifest.json').read_text())},indent=2))
