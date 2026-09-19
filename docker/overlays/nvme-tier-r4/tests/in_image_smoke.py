#!/usr/bin/env python3
"""
Build-time smoke test for the NVMe-tier overlay inside the image (no GPU, stdlib + the image's torch).
Run by Dockerfile.box right after install.py; exits non-zero on any failure so the build stops.

  - the installed modules are the overlay's (site-packages, not a stray checkout) and import cleanly
  - the default path is inert: EXL3_NVME_TIER unset -> generator._NVME_TIER == ""
  - the multimodal id floor the tier filters on is the one the tokenizer uses
  - a SegmentStore round trip + reopen in a temp dir; a checkpoint payload round trip is bit-exact and not pickle
  - engine_identity over the installed package is stable and reasonably fast
  - RecurrentCache carries the on_evict seam; on the tip variant, recurrent_tip imports next to it
  - on the mtp-pruned-r1 variant, the installed generator.py is the pruned-draft one with the tier ported into it
  - modules/ple.py carries the ple-ckpt-clone r1 stash fix (every variant)
"""
import os
import pathlib
import sys
import tempfile
import time

assert not os.environ.get("EXL3_NVME_TIER"), "EXL3_NVME_TIER must be unset at build time"

import torch

import exllamav3.generator.disk_cache as dc
import exllamav3.generator.generator as gen
import exllamav3.generator.pagetable as pt
import exllamav3.cache.recurrent as rec
from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX

for m in (dc, gen, pt, rec):
    assert "/site-packages/" in m.__file__, m.__file__
assert gen._NVME_TIER == "", gen._NVME_TIER
assert dc.FIRST_MM_EMBEDDING_INDEX == FIRST_MM_EMBEDDING_INDEX
assert hasattr(rec.RecurrentCache, "on_evict")
assert hasattr(pt.PageTable, "chain_to_root")

gsrc = open(gen.__file__).read()
assert "DiskPageCache.install(self" in gsrc, "generator.py does not carry the tier"
pkg = pathlib.Path(dc.__file__).resolve().parents[1]
try:
    import exllamav3.cache.recurrent_tip as tip
    variant = "stack-r4-e3r2+recurrent-tip-r1"
    assert hasattr(tip, "TipPolicyMixin") and issubclass(tip.TipPolicyMixin, object)
    assert "TipStash.install(self)" in gsrc, "tip generator not ported"
except ImportError:
    variant = "stack-r4-e3r2"
if "_EMBED_GPU_PRUNED = " in gsrc:
    assert variant == "stack-r4-e3r2", "tip and mtp-pruned-r1 are not a tested combination"
    variant = "stack-r4-e3r2+mtp-pruned-r1"
    assert "def _prepare_pruned_embed(self)" in gsrc and (pkg / "modules" / "embedding_pruned.py").is_file(), \
        "pruned-draft generator without its embedding_pruned.py"
ple = (pkg / "modules" / "ple.py").read_text()
assert "self.id_state[slot, :self.ctx].clone())" in ple and "self.id_state[slot, :self.ctx].cpu())" not in ple, \
    "modules/ple.py lacks the ple-ckpt-clone r1 stash fix"

# Store round trip + reopen
with tempfile.TemporaryDirectory() as d:
    ns = {"smoke": 1}
    s = dc.SegmentStore(d, ns, capacity_bytes = 64 * 1024**2, segment_bytes = 1024**2, admission_min_pages = 2,
                        free_space_fn = lambda: 100.0)
    keys = [bytes([i + 1]) * 16 for i in range(4)]
    payload = lambda i: bytes([i]) * 8192
    cp = {
        "position": 1024, "checkpoint_size": 123,
        (3, 0): (torch.randn(1, 4, 8, 8, dtype = torch.float32), torch.randn(16, 4).to(torch.bfloat16)),
    }
    raw = b"".join(bytes(memoryview(p)) for p in dc.serialize_stash(cp))
    assert not raw.startswith(b"\x80"), "checkpoint payload looks like a pickle"
    assert s.store_checkpoint(keys[-1], keys, dc.serialize_stash(cp))
    for i, k in enumerate(keys):
        assert s.store_page(k, keys[i - 1] if i else None, payload(i))
    s.close()
    s = dc.SegmentStore(d, ns, capacity_bytes = 64 * 1024**2, segment_bytes = 1024**2, admission_min_pages = 2,
                        free_space_fn = lambda: 100.0)
    for i, k in enumerate(keys):
        assert s.fetch_page(k) == payload(i), i
    back = dc.deserialize_stash(s.fetch_checkpoint(keys[-1]))
    assert back["position"] == 1024 and back["checkpoint_size"] == 123
    for a, b in zip(cp[(3, 0)], back[(3, 0)]):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    assert s.disk_bytes() <= 64 * 1024**2
    s.close()

t0 = time.monotonic()
a = dc.engine_identity()
b = dc.engine_identity()
dt = (time.monotonic() - t0) / 2
assert a == b and a["source_files"] > 50, a
print(f"nvme-tier smoke OK ({variant}): {dc.__file__}; ple stash fix present; engine identity {a['sources_sha256'][:16]} over "
      f"{a['source_files']} files in {dt:.2f} s; mm floor {FIRST_MM_EMBEDDING_INDEX}")
