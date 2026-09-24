#!/usr/bin/env python3
"""hcfast r1 host-side check of the two arithmetic claims V3 rests on (numpy, IEEE fp32, no GPU):
  1. the reduce-scatter over xor offsets 16/8/4/2/1 (hc_mix_v3.cu up, step 4) returns, for every value, the bits
     lane 0 holds after the served 5-level butterfly -- and the butterfly leaves identical bits in all 32 lanes;
     B = 1, 2, 4, 8 (4B values per lane), 3,000 random cases each, magnitudes 1e-3..1e3;
  2. the PRMT/FADD magic conversion 0x4B0000(q+128) - 8388736.0f equals (float) q bit for bit for all 256 int8.
FTZ does not matter: both sides perform the same fp32 additions on the same operand pairs.
Run: python3 test_hcfast_math_cpu.py   (needs numpy)
"""
import numpy as np
rng=np.random.default_rng(0)
def butterfly(x):
    x=x.copy()
    for off in (16,8,4,2,1):
        x = (x + x[np.arange(32)^off]).astype(np.float32)
    return x
def rs(x):
    NV=x.shape[1]; val=x.copy(); lanes=np.arange(32)
    for lvl in range(5):
        off=16>>lvl; cur=NV>>lvl
        if cur>=2:
            h=cur//2; upper=(lanes & off)!=0
            send=np.where(upper[:,None], val[:,:h], val[:,h:cur])
            keep=np.where(upper[:,None], val[:,h:cur], val[:,:h])
            recv=send[lanes^off]
            val[:,:h]=(keep+recv).astype(np.float32)
        else:
            val[:,0]=(val[:,0]+val[lanes^off,0]).astype(np.float32)
    span=32//NV
    return np.array([val[l,0] for l in range(32) if l & (span-1)==0],np.float32)
bad=0; n=0
for B in (1,2,4,8):
    NV=4*B
    for t in range(3000):
        x=(rng.standard_normal((32,NV))*10**rng.uniform(-3,3,(32,NV))).astype(np.float32)
        bf=butterfly(x)
        assert all(np.array_equal(bf[0].view(np.int32), bf[l].view(np.int32)) for l in range(32))
        n+=1
        if not np.array_equal(bf[0].view(np.int32), rs(x).view(np.int32)): bad+=1
print("reduce-scatter vs butterfly: cases",n,"mismatches",bad)
q=np.arange(-128,128,dtype=np.int32)
u=(((q+128)&0xFF).astype(np.uint32)|np.uint32(0x4B000000))
f=(u.view(np.float32)-np.float32(8388736.0)).astype(np.float32)
print("magic conversion exact for all 256 int8:", np.array_equal(f.view(np.int32), q.astype(np.float32).view(np.int32)))
assert bad == 0 and np.array_equal(f.view(np.int32), q.astype(np.float32).view(np.int32)); print("MATH PASS")
