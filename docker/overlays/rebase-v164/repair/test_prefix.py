"""Exercise actual native fallback/cropping and prefix cache lifetime on CPU."""
import types,weakref
from unittest.mock import patch
import torch
from exllamav3.model.model import Model
from exllamav3.architecture.qwen4_exp_mtp import Qwen4ExpMTPModel
from exllamav3.modules.draft_head_prefix import DraftHeadPrefix,tensor_version

class Head:
    def prepare_for_device(self,x,params):return x
    def forward(self,x,params):return torch.tensor([[[1.,2.,3.,99.]]])
class Target:
    loaded_tp=False
    config=types.SimpleNamespace(vocab_size=3)
    modules=[Head()]
    logit_layer_idx=0
    lm_head_argmax=Model.lm_head_argmax
mixer=types.SimpleNamespace(device='cpu',hc_mult=1,hidden_size=2,forward=lambda x,p:x.squeeze(-2))
target=Target();draft=Qwen4ExpMTPModel.__new__(Qwen4ExpMTPModel)
draft.stack_out=types.SimpleNamespace(mixer=mixer);draft.attached_model=weakref.ref(target);draft._head_prefix=None
params={'export_draft_conf':True}
ids=draft.sample_from_state(torch.ones((1,1,2)),params)
assert ids.tolist()==[[2]] and params['draft_conf'].tolist()==[[3.]],'native fallback must crop padding and export confidence'
prefix=DraftHeadPrefix(128);prefix.cache=['old'];prefix.key='old';prefix.checked={1}
assert prefix.prepare(types.SimpleNamespace(inner=None),256) is None
assert prefix.cache is None and prefix.key is None and not prefix.checked
with torch.inference_mode():t=torch.ones(1)
assert tensor_version(t) is None
# A loaded draft can be explicitly unloaded without retaining its copied CUDA head.
draft._head_prefix=prefix
with patch.object(Model,'unload',return_value=None) as base:
    draft.unload();base.assert_called_once()
assert draft._head_prefix is None
print('Native padding/confidence fallback, cache invalidation, inference tensor and unload regressions PASS')
