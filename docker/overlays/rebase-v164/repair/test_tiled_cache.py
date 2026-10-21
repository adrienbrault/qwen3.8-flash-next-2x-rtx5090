#!/usr/bin/env python3
"""Exercise native table construction, opt-in reuse and re-prepare/unload lifetime."""
import os
from unittest.mock import patch
import torch
from exllamav3.modules.hyperconnections import GatedResidual
from exllamav3.modules import hyperconnections as hc
calls=[]
def quant(src,out,scale):
    calls.append(src.shape)
    # Stable fake of the native output interface. This test covers lifetime and
    # bypass behavior; GPU contact compares actual native table bytes.
    out.fill_(len(calls));scale.fill_(1)
def ws(shape,dtype):return torch.empty(shape,dtype=dtype)
def owner():
    m=GatedResidual.__new__(GatedResidual);m.modules=[];m.hc_mult=2;m.hidden_size=8;m.rank=4;m.tiled=True
    m.proj_h=torch.arange(64,dtype=torch.half).view(4,16)
    m.upx_h=torch.arange(64,dtype=torch.half).view(2,2,4,4)
    m._tiled_cache=None
    return m
with patch.object(hc.ext,'det_quant_weight',quant):
    m=owner()
    with patch.dict(os.environ,EXL3_GR_CACHE_TILED='0'):
        m._prepare_tiled_cache(False);a=m._tiled_tables(ws);b=m._tiled_tables(ws)
        assert len(calls)==4 and a[0] is not b[0]
    with patch.dict(os.environ,EXL3_GR_CACHE_TILED='1'):
        m._prepare_tiled_cache(False);n=len(calls);a=m._tiled_tables(ws);b=m._tiled_tables(ws)
        assert len(calls)==n and all(x is y for x,y in zip(a,b))
        m._prepare_tiled_cache(False);c=m._tiled_tables(ws)
        assert len(calls)==n+2 and c[0] is not a[0]
        m._prepare_tiled_cache(True)
        assert m._tiled_cache is None
        m._prepare_tiled_cache(False);m.unload();assert m._tiled_cache is None and m.proj_h is None
print('Native tiled-table OFF/reuse/re-prepare/conversion/unload checks PASS')
