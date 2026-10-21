#!/usr/bin/env python3
"""Native checkpoint planning and single-issue two-chunk descriptor lifetime."""
from types import SimpleNamespace as NS
from unittest.mock import patch
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import torch
from exllamav3.generator.job import Job
from exllamav3.generator import native_prefill_pipeline as pp

seq=NS(kv_position=4096,sequence_ids=range(50001))
gen=NS(max_chunk_size=2048,recurrent_checkpoint_interval=2048,recurrent_checkpoint_interval_pp=3072)
job=NS(sequences=[seq],generator=gen,cached_pages=0,last_recurrent_checkpoint_pos=4096)
job.is_checkpoint_boundary=lambda **kw:Job.is_checkpoint_boundary(job,**kw)
assert pp.window_ends(job)==[6144,8192]
for anchor in (None,0,4096,5120):
 for pp_interval in (2048,3072,32768):
  job.last_recurrent_checkpoint_pos=anchor;gen.recurrent_checkpoint_interval_pp=pp_interval
  for pos in range(0,50100,256):
   seq.kv_position=pos
   assert job.is_checkpoint_boundary()==job.is_checkpoint_boundary(position=pos)
seq.kv_position=0;seq.sequence_ids=range(4097);job.last_recurrent_checkpoint_pos=None
gen.recurrent_checkpoint_interval=gen.recurrent_checkpoint_interval_pp=2048
assert pp.window_ends(job)==[2048]  # no stage-0 lookahead across first-chunk stash
seq.sequence_ids=range(4000);assert pp.window_ends(job)==[2048]

# A completed future must remain attached until chunk1 consumes it. Dropping it
# after join would run stage A1 twice and corrupt recurrent state.
rt=pp._Runtime.__new__(pp._Runtime)
rt.ends=[2048,4096];rt.index=0;rt.pending=None;rt.devices=('d0','d1');rt.mtp=True
rt.model=object();events=[];ids=torch.zeros((1,2048),dtype=torch.long)
rt.job=NS(sequences=[NS(sequence_ids=NS(torch_slice=lambda a,b:ids))])
class Stream:
 def wait_event(self,e):pass
 def synchronize(self):pass
rt.streams=(Stream(),Stream())
def stage0(ids,p):
 events.append(('A',int(p['cache_seqlens'][0])))
 return ids,p,ids,None
rt.stage0=stage0;rt.stages=([],[])
rt.modules=lambda x,p,mods:events.append(('B',int(p['cache_seqlens'][0])))
rt.peer_copy=lambda source,device,stream:source
with ThreadPoolExecutor(max_workers=1) as pool:
 rt.executor=pool
 with patch.object(torch.cuda,'device',lambda d:nullcontext()),patch.object(torch.cuda,'stream',lambda s:nullcontext()),patch.object(pp,'advance_recurrent_states',lambda *args:None):
  rt.forward(ids,{'cache_seqlens':torch.tensor([0],dtype=torch.int32)})
  assert rt.pending is not None and rt.pending.done()
  rt.forward(ids,{'cache_seqlens':torch.tensor([2048],dtype=torch.int32)})
  rt.drain()
assert events.count(('A',0))==events.count(('A',2048))==1,events
assert events.count(('B',0))==events.count(('B',2048))==1,events
# Drain must attempt both devices even if a worker or first stream fails.
from concurrent.futures import Future
rt.pending=Future();rt.pending.set_exception(RuntimeError('worker failed'))
joins=[]
class FailingStream:
 def __init__(self,i):self.i=i
 def synchronize(self):
  joins.append(self.i)
  if self.i==0:raise RuntimeError('stream failed')
rt.streams=(FailingStream(0),FailingStream(1))
with patch.object(torch.cuda,'device',lambda d:nullcontext()):
 try:rt.drain()
 except RuntimeError as exc:assert str(exc)=='worker failed'
 else:raise AssertionError('drain must fail')
assert joins==[0,1] and rt.pending is None
print('Native checkpoint intervals, last-page barrier and one-issue lookahead checks PASS')
