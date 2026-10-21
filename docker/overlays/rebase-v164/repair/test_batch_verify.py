#!/usr/bin/env python3
from types import SimpleNamespace as NS
from exllamav3 import GreedySampler
from exllamav3.generator.sampler.presets import ComboSampler
from exllamav3.generator.batch_verify import mode_for
s=ComboSampler(temperature=.6,top_k=20,top_p=.95)
assert not s.reqs_past_ids and s.batch_verify_mode=='sampled'
j=NS(sampler=s,new_tokens=2,min_new_tokens=128,forced_ids=None,checkpoint=None,filters=[],return_probs=False,return_top_tokens=0,sequences=[object()])
assert mode_for(j,4)=='sampled'
for key,value in [('checkpoint',{'offset':0,'explored_tokens':[7]}),('forced_ids',object()),('filters',[object()]),('return_probs',True),('return_top_tokens',1),('sequences',[object(),object()]),('new_tokens',-1),('new_tokens',0)]:
 old=getattr(j,key);setattr(j,key,value);assert mode_for(j,4) is None;setattr(j,key,old)
j.new_tokens=126;assert mode_for(j,4) is None
j.new_tokens=128;assert mode_for(j,4)=='sampled'
j.sampler=ComboSampler(rep_p=1.1,temperature=.6,top_k=20);assert mode_for(j,4) is None
j.sampler=GreedySampler();assert mode_for(j,4)=='greedy'
print('Actual simplified sampler/history and forced/filter/EOS-boundary eligibility PASS')
