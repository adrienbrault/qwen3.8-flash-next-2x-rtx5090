"""CPU regressions for actual overlap eligibility/wrapper and C++ resource methods.

Usage: python test_shared_overlap.py /path/to/source/exllamav3
No extension import, CUDA device or model is required. The C++ lifecycle methods
are compiled unchanged against a fail-injecting CUDA shim, not reimplemented.
"""
import ast
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[4] / 'ai/flashnext-v164/candidate-engine/exllamav3'
if (root / 'exllamav3').is_dir():
    root = root / 'exllamav3'
source = (root / 'modules/block_sparse_mlp.py').read_text()
cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'BlockSparseMLP')
methods = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('_shared_overlap_ok', 'forward')]
assert len(methods) == 2
for m in methods:
    m.decorator_list = []
    m.returns = None
    for arg in m.args.args:
        arg.annotation = None
ns = {'os': os, 'torch': types.SimpleNamespace(half='half'), 'MAX_BSZN': 32}
exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(root / 'modules/block_sparse_mlp.py'), 'exec'), ns)

class Input:
    def __init__(self, rows=4):
        self.shape = (1, rows, 2560)
        self.is_cuda = True
        self.dtype = 'half'
        self.device = 'cuda:1'
        self.contiguous = True
    def numel(self):
        return self.shape[0] * self.shape[1] * self.shape[2]
    def is_contiguous(self):
        return self.contiguous
    def view(self, *shape):
        return self

class BC:
    shared_early = True
    def __init__(self):
        self.pending = False
        self.calls = []
        self.start_failure = False
    def start_shared(self, y):
        assert not self.pending
        self.calls.append('start')
        self.pending = True
        if self.start_failure:
            raise RuntimeError('shared graph launch aborted')
    def drain_shared(self):
        self.calls.append('drain')
        self.pending = False

class Owner(types.SimpleNamespace):
    _shared_overlap_ok = ns['_shared_overlap_ok']
    forward = ns['forward']
    def _forward(self, x, params, out_dtype):
        self.bc.calls.append('router')
        if params.get('router_fail'):
            raise RuntimeError('router failed')
        if self.bc.pending:
            self.bc.calls.append('consume')
            self.bc.pending = False
        return x

def owner():
    return Owner(bc=BC(), bc_sh_exp=True, hidden_size=2560, device='cuda:1',
        cpu_offload=False, cpu_split_first=None, tp_reduce=False,
        intermediate_size=640, num_local_experts=512, num_experts=512,
        is_quantized=True, config=types.SimpleNamespace(infer_params=types.SimpleNamespace(no_reconstruct=False)),
        alt_residual_channel=False, latent_in=None, router_pre_norm=None,
        routed_pre_norm=None, routing_device=None, routing_gate=object())

previous = os.environ.get('EXL3_SHARED_EXPERT_OVERLAP')
try:
    for flag in ('', '0', 'true', '2'):
        os.environ['EXL3_SHARED_EXPERT_OVERLAP'] = flag
        o = owner(); x = Input()
        assert not o._shared_overlap_ok(x, {})
        assert o.forward(x, {}, None) is x and o.bc.calls == ['router']
    os.environ['EXL3_SHARED_EXPERT_OVERLAP'] = '1'
    for rows in (1, 4, 16, 32):
        o = owner(); x = Input(rows)
        assert o._shared_overlap_ok(x, {})
        assert o.forward(x, {}, None) is x
        assert o.bc.calls == ['start', 'router', 'consume'] and not o.bc.pending
    for attr, value in [('cpu_offload', True), ('cpu_split_first', 0), ('tp_reduce', True),
            ('intermediate_size', 0), ('num_local_experts', 0), ('is_quantized', False),
            ('alt_residual_channel', True), ('latent_in', object()), ('router_pre_norm', object()),
            ('routed_pre_norm', object()), ('routing_device', 'cuda:0'), ('routing_gate', None),
            ('bc_sh_exp', False), ('hidden_size', 2559)]:
        o = owner(); setattr(o, attr, value)
        assert not o._shared_overlap_ok(Input(), {}), attr
    for attr, value in [('is_cuda', False), ('dtype', 'bf16'), ('contiguous', False), ('device', 'cuda:0')]:
        x = Input(); setattr(x, attr, value)
        assert not owner()._shared_overlap_ok(x, {}), attr
    for rows in (0, 33, 512):
        assert not owner()._shared_overlap_ok(Input(rows), {})
    o = owner(); o.config.infer_params.no_reconstruct = True
    assert not o._shared_overlap_ok(Input(), {})
    o = owner(); o.bc.shared_early = False
    assert not o._shared_overlap_ok(Input(), {})
    o = owner(); o.bc = None
    assert not o._shared_overlap_ok(Input(), {})
    for key in ('autosplit_measure', 'tp_warmup', 'activate_all_experts', 'capture_sens'):
        assert not owner()._shared_overlap_ok(Input(), {key: True}), key
    assert not owner()._shared_overlap_ok(Input(), {'capture_sens': {}})
    for where in ('router', 'shared'):
        o = owner(); o.bc.start_failure = where == 'shared'
        try:
            o.forward(Input(), {'router_fail': where == 'router'}, None)
        except RuntimeError:
            pass
        else:
            raise AssertionError('expected injected failure')
        assert o.bc.calls[-1] == 'drain' and not o.bc.pending, (where, o.bc.calls)
        o.bc.start_failure = False
        assert o.forward(Input(), {}, None) is not None and not o.bc.pending
finally:
    if previous is None:
        os.environ.pop('EXL3_SHARED_EXPERT_OVERLAP', None)
    else:
        os.environ['EXL3_SHARED_EXPERT_OVERLAP'] = previous
print('Actual Python eligibility, native fallback and aborted-forward drain PASS')

cpp = (root / 'exllamav3_ext/libtorch/blocksparse_mlp.cpp').read_text()
def method(name):
    start = cpp.index(name)
    body = cpp.index('{', start)
    level = 1; end = body + 1
    while level:
        level += (cpp[end] == '{') - (cpp[end] == '}')
        end += 1
    return cpp[start:end]
actual = '\n'.join(method(n) for n in ('void BC_BlockSparseMLP::init_shared_overlap()',
    'void BC_BlockSparseMLP::drain_shared()', 'BC_BlockSparseMLP::~BC_BlockSparseMLP()'))
shim = r'''
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
using cudaStream_t = int*;
using cudaEvent_t = int*;
constexpr int cudaStreamNonBlocking=1, cudaEventDisableTiming=2, cudaSuccess=0;
int current_device=0;
int cudaGetDevice(int* d) {*d=current_device; return 0;}
int cudaSetDevice(int d) {current_device=d; return 0;}
int live_streams=0, live_events=0, creates=0, fail_at=0;
std::vector<std::string> events;
int cudaStreamCreateWithFlags(cudaStream_t* p,int) { if (++creates==fail_at) return 1; *p=new int(1); ++live_streams; return 0; }
int cudaEventCreateWithFlags(cudaEvent_t* p,int) { if (++creates==fail_at) return 1; *p=new int(1); ++live_events; return 0; }
int cudaStreamDestroy(cudaStream_t p) { events.push_back("destroy-stream"); delete p; --live_streams; return 0; }
int cudaEventDestroy(cudaEvent_t p) { events.push_back("destroy-event"); delete p; --live_events; return 0; }
int cudaStreamSynchronize(cudaStream_t p) { events.push_back(p ? "sync-side" : "sync-current"); return 0; }
#define C10_CUDA_CHECK(x) do { if ((x)) throw std::runtime_error("injected CUDA allocation failure"); } while(0)
namespace at {
struct Device { int index() const { return 1; } };
struct Tensor { bool defined=false; Tensor()=default; Tensor(bool d):defined(d){} bool is_cuda()const{return true;} Device device()const{return{};} };
namespace cuda { struct Stream { cudaStream_t stream()const{return nullptr;} }; Stream getCurrentCUDAStream(){return{};} }
}
namespace c10 { namespace cuda { struct CUDAGuard { CUDAGuard(int){} CUDAGuard(at::Device){} }; } }
struct BC_BlockSparseMLP {
    bool shared_early=false, shared_started=false;
    bool shared_experts=true, out_d_sh=true;
    int shared_overlap_device=-1;
    cudaStream_t shared_stream=nullptr;
    cudaEvent_t shared_input_ready=nullptr, shared_done=nullptr;
    at::Tensor yh, shared_input;
    void init_shared_overlap(); void drain_shared(); ~BC_BlockSparseMLP() noexcept;
};
'''
main = r'''
int main() {
#ifdef USE_ROCM
    setenv("EXL3_SHARED_EXPERT_OVERLAP","1",1);
    { BC_BlockSparseMLP bc; bc.init_shared_overlap(); assert(!bc.shared_early); }
    assert(creates==0 && live_streams==0 && live_events==0);
#else
    for (const char* flag : {"0", "2", "true"}) {
        setenv("EXL3_SHARED_EXPERT_OVERLAP",flag,1);
        { BC_BlockSparseMLP bc; bc.init_shared_overlap(); assert(!bc.shared_early); }
        assert(creates==0 && live_streams==0 && live_events==0);
    }
    setenv("EXL3_SHARED_EXPERT_OVERLAP","1",1);
    for (int failure=1; failure<=3; ++failure) {
        creates=0; fail_at=failure;
        try { BC_BlockSparseMLP bc; bc.init_shared_overlap(); assert(false); }
        catch(const std::runtime_error&) {}
        assert(live_streams==0 && live_events==0);
    }
    creates=0; fail_at=0; events.clear();
    {
        BC_BlockSparseMLP bc; bc.init_shared_overlap();
        assert(bc.shared_early && live_streams==1 && live_events==2);
        // Aborted early launch: no done event need have been recorded.
        bc.shared_started=true; bc.shared_input=at::Tensor(true);
        bc.drain_shared();
        assert(!bc.shared_started && !bc.shared_input.defined);
        assert(events==std::vector<std::string>({"sync-side","sync-current"}));
        events.clear();
    }
    assert(live_streams==0 && live_events==0 && current_device==0);
    assert(events==std::vector<std::string>({"sync-side","destroy-event","destroy-event","destroy-stream"}));
    for(int missing=0;missing<2;++missing) {
        BC_BlockSparseMLP bc; bc.shared_experts=missing!=0; bc.out_d_sh=missing==0;
        bc.init_shared_overlap(); assert(!bc.shared_early);
    }
#endif
    std::cout << "Actual C++ partial-allocation cleanup, abort drain, destructor ordering and ROCm isolation PASS\n";
}
'''
compiler = shutil.which('g++') or shutil.which('c++')
assert compiler, 'C++ compiler required for native resource lifecycle regression'
with tempfile.TemporaryDirectory(prefix='shared-overlap-cpu-') as scratch:
    path = Path(scratch)
    (path / 'test.cpp').write_text(shim + actual + main)
    for flags in ([], ['-DUSE_ROCM']):
        subprocess.run([compiler, '-std=c++17', *flags, str(path / 'test.cpp'), '-o', str(path / 'test')], check=True)
        subprocess.run([str(path / 'test')], check=True)

# The actual dispatch still defers the wait until B for integer kernels and for
# the native fractional fallback; empty-input output also reads shared output.
cu = (root / 'exllamav3_ext/quant/exl3_moe_coop.cu').read_text()
launch = cu[cu.index('void exl3_moe_coop_launch('):cu.index('MoeCoopParams exl3_moe_coop_prepare')]
assert 'stream, v2, wait_before_b)' in launch
assert 'stream, v2, wait_before_b, v3)' in launch
assert launch.index('cudaStreamWaitEvent') < launch.index('exl3_moe_coop_empty_kernel')
a = launch.rindex('cudaLaunchKernel(ka.kernel')
b = launch.rindex('cudaLaunchKernel(kb.kernel')
assert a < launch.rindex('cudaStreamWaitEvent') < b
assert 'stream, wait_before_b);' in cu[cu.index('void exl3_moe_coop_run'):]
print('Integer/generic/fractional/empty dispatch event seams PASS')
