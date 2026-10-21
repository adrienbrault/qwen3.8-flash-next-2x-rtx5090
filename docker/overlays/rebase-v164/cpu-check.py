#!/usr/bin/env python3
"""Exercise the actual rebased Tabby result handler and configuration schema on CPU."""
import sys
sys.path.insert(0, '/app')
from backends.exllamav3.model import ExllamaV3Container
from common.config_models import DraftModelConfig
from inspect import signature
from exllamav3 import Generator
assert 'num_draft_tokens_by_batch' in signature(Generator.__init__).parameters
cfg=DraftModelConfig.model_validate({'draft_mode':'mtp','draft_num_tokens':3,'draft_num_tokens_by_batch':[[4,3],[8,2]]})
assert cfg.draft_num_tokens_by_batch==[(4,3),(8,2)]
for bad in [[[8,2],[4,3]],[[4,0]],[[True,3]]]:
 try: DraftModelConfig.model_validate({'draft_mode':'mtp','draft_num_tokens':3,'draft_num_tokens_by_batch':bad})
 except ValueError: pass
 else: raise AssertionError(bad)
obj=ExllamaV3Container.__new__(ExllamaV3Container)
result={'eos_reason':'max_new_tokens','prompt_tokens':100,'cached_tokens':0,'time_prefill':1.0,'new_tokens':32,'time_generate':0.5,'time_enqueued':0.01,'accepted_draft_tokens':12,'rejected_draft_tokens':3}
finish=obj.handle_finish_chunk(result,'cpu-fixture','answer')
assert finish['gen_tokens']==32 and finish['finish_reason']=='length'
assert finish['draft_accept']==12 and finish['cached_tokens']==0
print('Real result handling, draft schema and engine signature PASS')

from common.gen_logging import log_metrics
log_metrics('cpu-fixture',finish,100,262144)
print('Full native completion result + logging path PASS')

# Reproduce the legal off-loop encode / token-endpoint interleaving found in review.
import threading,types
from exllamav3.tokenizer.tokenizer import Tokenizer
from tokenizers import Tokenizer as Backend,models
obj=Tokenizer.__new__(Tokenizer)
obj.tokenizer=Backend(models.WordLevel({'[UNK]':0,'hello':1},unk_token='[UNK]'))
obj.tokenizer.add_special_tokens(['<|im_start|>'])
obj.unspecial_piece_to_id={};obj._encode_lock=threading.RLock()
entered=threading.Event();release=threading.Event();second_started=threading.Event();second_done=threading.Event()
def hook(self,texts):
    if '<|im_start|>' in texts[0]:
        entered.set();assert release.wait(2)
    return [e.ids for e in self.tokenizer.encode_batch(texts,add_special_tokens=False)]
obj._encode_texts=types.MethodType(hook,obj)
outputs={}
def first():outputs['first']=obj.encode_part_base('<|im_start|>'*1800,True)
def second():
    second_started.set();outputs['second']=obj.encode_part_base('hello',False);second_done.set()
a=threading.Thread(target=first);b=threading.Thread(target=second);a.start();assert entered.wait(2)
b.start();assert second_started.wait(2);assert not second_done.wait(.05)
release.set();a.join(2);b.join(2);assert not a.is_alive() and not b.is_alive()
assert outputs['first'].count(obj.tokenizer.token_to_id('<|im_start|>'))==1800
assert outputs['second']==[1]
print('Concurrent special-token mode isolation PASS')

obj2=Tokenizer.__new__(Tokenizer)
obj2.tokenizer=Backend(models.WordLevel({'[UNK]':0,'hello':1},unk_token='[UNK]'))
obj2.tokenizer.add_special_tokens(['<|im_start|>'])
obj2.unspecial_piece_to_id={};obj2._encode_lock=threading.RLock()
text='<|im_start|>'*4000
first=obj2.encode_part_base(text,False);second=obj2.encode_part_base(text,False)
assert first==second and first.count(obj2.tokenizer.token_to_id('<|im_start|>'))==0
assert obj2.encode_part_base(text,True).count(obj2.tokenizer.token_to_id('<|im_start|>'))==4000
print('First long encode chunk-probe mode restoration PASS')
