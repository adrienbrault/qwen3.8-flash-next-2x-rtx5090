"""Native BC microbatch regression: output scratch aliases across successive calls."""
import types
import torch
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP, MAX_BSZN

for rows in (MAX_BSZN+1,2*MAX_BSZN,3*MAX_BSZN-1,4*MAX_BSZN):
    owner=types.SimpleNamespace(expert_size=5)
    owner.experts_cfg=types.SimpleNamespace(out_bszn=torch.empty((MAX_BSZN,5)))
    calls=[]
    class BC:
        def run_bszN(self,y,sel,rw):
            assert 0<y.shape[0]<=MAX_BSZN
            calls.append(y.shape[0])
            owner.experts_cfg.out_bszn.fill_(-9999)
            owner.experts_cfg.out_bszn[:y.shape[0]].copy_(y*rw+sel)
    owner.bc=BC()
    y=torch.arange(rows*5,dtype=torch.float).reshape(rows,5)
    sel=torch.arange(rows,dtype=torch.long).reshape(rows,1)
    rw=torch.arange(1,rows+1,dtype=torch.float).reshape(rows,1)
    result=BlockSparseMLP._run_decode_chunks(owner,y,sel,rw)
    torch.testing.assert_close(result,y*rw+sel)
    owner.experts_cfg.out_bszn.fill_(float('nan'))
    assert torch.isfinite(result).all(), 'returned output must not alias BC scratch'
    assert sum(calls)==rows
print('Native BC scratch lifetime, row order and partial final microbatch PASS')
