#!/usr/bin/env python3
import torch
from exllamav3.modules.embedding_pruned import PrunedEmbeddingMirror
for dtype in (torch.half,torch.bfloat16):
 w=torch.arange(24*8,dtype=torch.float).view(24,8).to(dtype).requires_grad_()
 for keep in (torch.arange(8),torch.tensor([2,4,6,8])):
  m=PrunedEmbeddingMirror(w,keep,'cpu')
  ids=keep[:2].view(2,1)
  assert torch.equal(m.lookup(w,ids,None),torch.nn.functional.embedding(ids,w))
  other=torch.tensor([[1],[23]])
  assert torch.equal(m.lookup(w,other,other),torch.nn.functional.embedding(other,w))
print('Exact prefix/nonprefix/outside-set embedding rows, both resident dtypes PASS')
# Delayed metadata consumer at the real MTP loop seam: each asynchronous H2D
# must retain that step's position after the shared base is advanced further.
import os
from types import SimpleNamespace as NS
from unittest.mock import patch
from exllamav3.generator.generator import Generator
import exllamav3.generator.generator as gm
seen=[]
class Draft:
 def device_draft_mirror(self):return NS(device=torch.device('cpu'))
 def target_embed(self):return NS(embedding=NS(weight=torch.zeros((8,2))))
 def forward(self,ids,params):
  seen.append(params['cache_seqlens'])
  return torch.zeros((1,1,4,2))
 def sample_from_state(self,state,params):return torch.tensor([[len(seen)+1]])
seq=NS(block_index_tensor=torch.zeros((1,16),dtype=torch.int32),kv_position=100)
job=NS(is_prefill_done=lambda:True,get_max_seq_len=lambda:100,sequences=[seq],mtp_last_hidden=torch.zeros((1,1,4,2)),time_first_token=1,embeddings=[],get_input_ids_list=lambda:[torch.tensor([[1]])])
g=NS(active_jobs=[job],num_draft_tokens=3,draft_calibrator=None,draft_input_ids_pinned=torch.zeros((1,1),dtype=torch.long),draft_ids_pinned=torch.zeros((1,3),dtype=torch.long),draft_model=Draft(),draft_cache=object(),model=NS(logit_layer_idx=0,modules=[NS(prepare_for_device=lambda x,p:x)]))
g._staging=lambda tag,rows,width=None:torch.empty((rows,) if width is None else (rows,width),dtype=torch.int32)
with patch.dict(os.environ,EXL3_MTP_DEVICE_DRAFT='1'),patch.object(torch.Tensor,'pin_memory',lambda self:self):
 ids=Generator.iterate_draftmodel_mtp_gen(g,[])
assert [t.item() for t in seen]==[100,101,102] and len({t.data_ptr() for t in seen})==3
assert ids.tolist()==[[2,3,4]]
print('Real MTP loop keeps per-step H2D metadata immutable PASS')
