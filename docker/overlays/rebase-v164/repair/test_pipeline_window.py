"""Longer one-ahead windows retain native checkpoint and last-page barriers."""
import os
from types import SimpleNamespace as NS
from unittest.mock import patch
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import torch
from exllamav3.generator.job import Job
from exllamav3.generator import native_prefill_pipeline as pp
seq=NS(kv_position=0,sequence_ids=range(50001))
gen=NS(max_chunk_size=2048,recurrent_checkpoint_interval=4096,recurrent_checkpoint_interval_pp=32768)
job=NS(sequences=[seq],generator=gen,cached_pages=0,last_recurrent_checkpoint_pos=None)
job.is_checkpoint_boundary=lambda **kw:Job.is_checkpoint_boundary(job,**kw)
with patch.dict(os.environ,{'EXL3_NATIVE_PREFILL_WINDOW':'8'}):
 assert pp.window_ends(job)==list(range(2048,16385,2048))
 seq.kv_position=16384
 assert pp.window_ends(job)==list(range(18432,32769,2048))
 seq.kv_position=0;gen.recurrent_checkpoint_interval_pp=3072
 assert pp.window_ends(job)==[2048,4096]
 seq.kv_position=4096;job.last_recurrent_checkpoint_pos=4096
 assert pp.window_ends(job)==[6144,8192]
 gen.recurrent_checkpoint_interval_pp=32768;seq.sequence_ids=range(12289);seq.kv_position=0;job.last_recurrent_checkpoint_pos=None
 assert pp.window_ends(job)==[2048,4096,6144,8192]  # native near-tail checkpoint wins
with patch.dict(os.environ,{'EXL3_NATIVE_PREFILL_WINDOW':'3'}):
 try:pp.window_ends(job)
 except ValueError:pass
 else:raise AssertionError('unbounded window admitted')
rt=pp._Runtime.__new__(pp._Runtime);rt.ends=list(range(2048,16385,2048));rt.index=0;rt.pending=None;rt.devices=('d0','d1');rt.mtp=True;rt.model=object()
events=[];ids=torch.zeros((1,2048),dtype=torch.long)
rt.job=NS(sequences=[NS(sequence_ids=NS(torch_slice=lambda a,b:ids))])
class Stream:
 def wait_event(self,e):pass
 def synchronize(self):pass
rt.streams=(Stream(),Stream())
def stage0(ids,p):
 events.append(('A',int(p['cache_seqlens'][0])))
 return ids,p,ids,None
rt.stage0=stage0;rt.stages=([],[]);rt.modules=lambda x,p,mods:events.append(('B',int(p['cache_seqlens'][0])))
rt.peer_copy=lambda source,device,stream:source
with ThreadPoolExecutor(max_workers=1) as pool:
 rt.executor=pool
 with patch.object(torch.cuda,'device',lambda d:nullcontext()),patch.object(torch.cuda,'stream',lambda s:nullcontext()),patch.object(pp,'advance_recurrent_states',lambda *args:None):
  for start in range(0,16384,2048):
   rt.forward(ids,{'cache_seqlens':torch.tensor([start],dtype=torch.int32)})
  rt.drain()
for start in range(0,16384,2048):assert events.count(('A',start))==events.count(('B',start))==1,(start,events)
assert rt.pending is None and rt.index==8
print('Eight-chunk one-future execution, native checkpoint and last-page barriers PASS')
