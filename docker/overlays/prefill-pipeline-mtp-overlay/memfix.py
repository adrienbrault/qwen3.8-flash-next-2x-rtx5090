"""R441: the prefill pipeline's 320 MB free-VRAM guard reads torch.cuda.mem_get_info (device-level free), which is small once
the served cache + draft cache are allocated even though the caching allocator holds reusable reserved memory (R439/R440:
every window rejected there in the harness with the draft and in serving). Count the allocator's reusable reserve too."""
import sys
p = sys.argv[1]
s = open(p).read()
old = "    if any(torch.cuda.mem_get_info(d)[0] < 320 * 1024**2 for d in devices):"
new = ("    def _free_bytes(d):  # device free + allocator reserve not currently allocated (R441)\n"
       "        return torch.cuda.mem_get_info(d)[0] + torch.cuda.memory_reserved(d) - torch.cuda.memory_allocated(d)\n"
       "    if any(_free_bytes(d) < 320 * 1024**2 for d in devices):")
assert s.count(old) == 1, "guard line not found exactly once"
open(p, "w").write(s.replace(old, new))
print("memfix applied")
