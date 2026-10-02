"""Run the ACTUAL served/ported MTP loop on CPU tensor doubles; no torch/CUDA import."""
import ast
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from selftest import helper
m=helper()


class Tensor:
    def __init__(self,data,cuda=False):
        self.data=[list(x) for x in data] if data and isinstance(data[0],(tuple,list)) else list(data)
        self.is_cuda=cuda;self.device='cuda' if cuda else 'cpu'
    @property
    def shape(self):
        return (len(self.data),len(self.data[0])) if self.data and isinstance(self.data[0],list) else (len(self.data),)
    def __getitem__(self,key): return View(self,key)
    def __setitem__(self,key,value):
        if isinstance(key,int) and len(self.shape)==1:self.data[key]=value
        else:View(self,key).copy_(value)
    def copy_(self,other):
        values=other.tolist() if isinstance(other,(Tensor,View)) else other
        self.data=[list(x) for x in values] if values and isinstance(values[0],list) else list(values)
        return self
    def is_pinned(self):return False
    def to(self,device,**kwargs):return Tensor(self.data,cuda=device=='cuda')
    def cpu(self):return Tensor(self.data)
    def view(self,*shape):
        values=[x for row in self.data for x in row] if len(self.shape)==2 else self.data
        return Tensor(values,self.is_cuda)
    def tolist(self):return self.data
    def __iadd__(self,v):self.data=[x+v for x in self.data];return self

class View(Tensor):
    def __init__(self,parent,key):
        self.parent=parent;self.key=key
        self.is_cuda=parent.is_cuda;self.device=parent.device
    @property
    def data(self):
        if isinstance(self.key,tuple):
            r,c=self.key
            rs=[self.parent.data[r]] if isinstance(r,int) else self.parent.data[r]
            return [row[c] for row in rs]
        return self.parent.data[self.key] if isinstance(self.key,slice) else [self.parent.data[self.key]]
    @data.setter
    def data(self,value):raise AssertionError('view replacement')
    def copy_(self,other):
        src=other.tolist() if isinstance(other,(Tensor,View)) else other
        if isinstance(self.key,tuple):
            r,c=self.key
            indices=[r] if isinstance(r,int) else list(range(len(self.parent.data)))[r]
            for i,idx in enumerate(indices):
                # Scalar row indexing in torch produces a vector (used for chain tail).
                values=src if isinstance(r,int) else src[i]
                self.parent.data[idx][c]=values
        else:
            indices=list(range(len(self.parent.data)))[self.key]
            for i,idx in enumerate(indices):self.parent.data[idx]=list(src[i]) if isinstance(src[i],list) else src[i]
        return self

class Torch:
    long='long'
    @staticmethod
    def empty(shape,**kwargs):return Tensor([[0]*shape[1] for _ in range(shape[0])],cuda=kwargs.get('device')=='cuda')
    @staticmethod
    def tensor(values,**kwargs):return Tensor(values)
    @staticmethod
    def cat(tensors,dim=0):
        assert dim==0
        return Tensor([row for t in tensors for row in t.data],cuda=tensors[0].is_cuda)


def loop(path):
    node=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='iterate_draftmodel_mtp_gen')
    # Relative helper import resolves to the same pure implementation above.
    node.body=[n for n in node.body if not isinstance(n,ast.ImportFrom)]
    g={'torch':Torch,'PAGE_SIZE':256,'_DRAFT_PINNED_STAGING':False,'_EMBED_GPU_PRUNED':True,
       '_MTP_DEVICE_DRAFT':True,'open_lookup_chain':m.open_lookup_chain,'lookup_mask':m.lookup_mask}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<production MTP>','exec'),g)
    return g[node.name]


def setup(enabled,hits,device,depth):
    jobs=[];calls=[]
    for hit in hits:
        match=m.PromptLookupMatch(0,3,(1,2,3)) if hit else None
        jobs.append(NS(is_prefill_done=lambda:True,get_max_seq_len=lambda:100,
                       sequences=[NS(kv_position=100,block_index_tensor=Tensor([[0]]))],
                       mtp_last_hidden=Tensor([[0]],cuda=device),time_first_token=1,embeddings=[],
                       get_input_ids_list=lambda:[Tensor([[9]])],get_prompt_lookup_match=lambda match=match:match,
                       lookup_checks=0,lookup_hits=0,lookup_proposed=0))
    def forward(ids,params):
        calls.append((params['draft_step'],ids.shape,ids.tolist()))
        return Tensor([[0] for _ in hits],cuda=device)
    def sample(state,params):
        return Tensor([[{0:1,1:7,2:8}[params['draft_step']]] for _ in hits],cuda=device)
    drafter=NS(target_embed=lambda:NS(can_embed_device_ids=lambda d:True),forward=forward,sample_from_state=sample)
    gen=NS(active_jobs=jobs,max_num_draft_tokens=3,prompt_lookup_enabled=enabled,draft_calibrator=None,
           draft_input_ids_pinned=Tensor([[0] for _ in hits]),draft_ids_pinned=Tensor([[0]*3 for _ in hits]),
           draft_model=drafter,model=NS(modules=[NS(prepare_for_device=lambda state,params:state)],logit_layer_idx=0),
           _pruned_embed=(None,'cuda'),draft_cache=object(),_draft_params=lambda p:p,
           _draft_tables=lambda name,b,p:(Tensor([[0]*p for _ in hits]),Tensor([100 for _ in hits])),
           draft_window_rows=lambda *args:None,_get_draft_depth=lambda _:depth,
           _draft_step_seqlens=lambda name,seq,idx:seq)
    return gen,jobs,calls

class TestMTPLoop(unittest.TestCase):
    def test_base_off_allhit_and_mixed_host_and_device(self):
        p=Path(__file__).resolve().parents[1]
        old=loop(p/'fixtures/base-generator.py');new=loop(p/'overlay/exllamav3/generator/generator.py')
        for device in (False,True):
            for depth in (2,3):
                for hits in ([True],[False],[True,False],[True,True]):
                    with self.subTest(device=device,depth=depth,hits=hits):
                        base,_,bc=setup(False,hits,device,depth)
                        off,oj,oc=setup(False,hits,device,depth)
                        b=old(base,[]).tolist();o=new(off,[]).tolist()
                        self.assertEqual(b,o);self.assertEqual(bc,oc)
                        self.assertTrue(all(j.lookup_checks==0 for j in oj))
                        on,jobs,calls=setup(True,hits,device,depth)
                        out=new(on,[]).tolist()
                        self.assertEqual(out,[([1,2,3] if hit else [1,7,8])[:depth] for hit in hits])
                        self.assertEqual(len(calls),1 if all(hits) else depth)
                        self.assertTrue(all(call[1][0]==len(hits) for call in calls)) # no miss compaction
                        self.assertEqual([j.lookup_proposed for j in jobs],[(depth-1) if hit else 0 for hit in hits])
                        self.assertEqual(on._prompt_lookup_round,{id(job):[False]+[True]*(depth-1) for job,hit in zip(jobs,hits) if hit})

if __name__=='__main__':unittest.main()
