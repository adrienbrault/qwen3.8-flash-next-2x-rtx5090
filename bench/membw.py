#!/usr/bin/env python3
"""DRAM bandwidth of ONE card: device-to-device copy of a buffer far larger than L2 (~96 MB on the 5090).
Bandwidth = 2 x bytes / time (read + write), in 10^9 bytes per second. Run it with exactly one visible GPU
(docker --gpus device=N); it samples that card's memory clock through NVML while the copies run, and checks the copy
is exact (a memory-OC corruption canary, not a proof).
  membw.py [--mib 1024] [--iters 300]  ->  MEMBW gbps=... ms=... copy_equal=... memclk_max=..."""
import argparse, threading, time
import torch

p = argparse.ArgumentParser()
p.add_argument("--mib", type=int, default=1024)
p.add_argument("--iters", type=int, default=300)
a = p.parse_args()
clk, stop = [], threading.Event()
def sample():
    try:
        import pynvml as N; N.nvmlInit(); h = N.nvmlDeviceGetHandleByIndex(0)
        while not stop.is_set():
            clk.append(N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_MEM)); time.sleep(0.02)
    except Exception as e:
        clk.append(f"nvml:{type(e).__name__}")
n = a.mib * 1024 * 1024 // 4
x = torch.empty(n, dtype=torch.float32, device=0).normal_(); y = torch.empty_like(x)
for _ in range(5): y.copy_(x)
torch.cuda.synchronize()
t = threading.Thread(target=sample); t.start()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(a.iters): y.copy_(x)
e.record(); torch.cuda.synchronize(); stop.set(); t.join()
ms = s.elapsed_time(e) / a.iters
nums = [c for c in clk if isinstance(c, int)]
print(f"MEMBW gbps={2 * n * 4 / (ms / 1e3) / 1e9:.1f} ms={ms:.3f} copy_equal={bool(torch.equal(x, y))} "
      f"memclk_max={max(nums) if nums else clk[:1]}", flush=True)
