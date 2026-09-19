#!/usr/bin/env python3
"""Apply the NVMe-tier generator.py hunks to a generator.py (the served one, or recurrent-tip-r1's).

Every hunk is anchored on text that exists exactly once in both bases; the script fails if an anchor is missing or
ambiguous, so a moved base is caught instead of fuzzily patched.

usage: port_generator.py <in generator.py> <out generator.py>
"""
import sys

FLAG = (
    "# Persistent NVMe prefix tier (generator/disk_cache.py): EXL3_NVME_TIER=<absolute directory>. Unset = no tier:\n"
    "# disk_cache is never imported and every hook below is a `self.disk_page_cache is not None` test that is False.\n"
    "_NVME_TIER = os.environ.get(\"EXL3_NVME_TIER\", \"\")\n"
    "\n"
)

HUNKS = [
    # 1. flag, right before the class
    ("\nclass Generator:\n", "\n" + FLAG + "class Generator:\n"),
    # 2. construction, after the checkpoint interval setup (RecurrentCache and the page table exist)
    (
        "        self.recurrent_checkpoint_interval_pp = ceil_span(recurrent_checkpoint_interval_pp, self.max_chunk_size)\n",
        "        self.recurrent_checkpoint_interval_pp = ceil_span(recurrent_checkpoint_interval_pp, self.max_chunk_size)\n"
        "        self.disk_page_cache = None\n"
        "        if _NVME_TIER:\n"
        "            from .disk_cache import DiskPageCache\n"
        "            self.disk_page_cache = DiskPageCache.install(self, cache, draft_cache, _NVME_TIER)\n",
    ),
    # 3. per-iteration pump, before any job allocates pages
    (
        "        results = []\n        self.iterate_start_jobs(results)\n",
        "        # NVMe tier: stop the idle drain, then copy out a few pages of admitted chains (no file I/O here)\n"
        "        if self.disk_page_cache is not None:\n"
        "            self.disk_page_cache.pump()\n"
        "\n"
        "        results = []\n        self.iterate_start_jobs(results)\n",
    ),
    # 4. idle transition: pages move during defrag (drain stopped), then stay put until the next iterate
    (
        "        if self.recurrent_cache is not None:\n"
        "            self.recurrent_cache.prune_stranded()\n"
        "        self.pagetable.defrag()\n",
        "        if self.disk_page_cache is not None:\n"
        "            self.disk_page_cache.on_busy()\n"
        "        if self.recurrent_cache is not None:\n"
        "            self.recurrent_cache.prune_stranded()\n"
        "        self.pagetable.defrag()\n",
    ),
    (
        "        run_pending_swap_sweeps(self.model.config.infer_params)\n        malloc_trim()\n",
        "        run_pending_swap_sweeps(self.model.config.infer_params)\n        malloc_trim()\n"
        "        if self.disk_page_cache is not None:\n"
        "            self.disk_page_cache.on_idle()\n",
    ),
]


def port(text: str) -> str:
    for old, new in HUNKS:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"anchor found {n} times (need exactly 1): {old[:80]!r}")
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        text = f.read()
    with open(dst, "w") as f:
        f.write(port(text))
    print(f"ported {src} -> {dst}")
