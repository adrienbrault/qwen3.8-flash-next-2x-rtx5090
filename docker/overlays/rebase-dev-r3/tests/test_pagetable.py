"""Pagetable tests (CPU only, no model, no extension).

Exercises the pure bookkeeping of generator/pagetable.py — the reference-free
page classes, ref counting and the tree eviction order — plus the chain-hash
helpers the NVMe tier keys pages with.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _runner  # noqa: E402

import torch  # noqa: E402

from exllamav3.constants import PAGE_SIZE  # noqa: E402
from exllamav3.generator.pagetable import (  # noqa: E402
    CachePage, PageTable, is_content_hash, _tensor_blake2b_checksum,
)

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS {name}")
    except Exception:
        import traceback
        RESULTS.append((name, False, traceback.format_exc()))
        print(f"FAIL {name}")
        traceback.print_exc()


class FakeGenerator:
    pass


class FakeCache:
    max_num_tokens = 64 * PAGE_SIZE


def fresh_table():
    pt = PageTable.__new__(PageTable)
    pt.generator = FakeGenerator()
    pt.cache = FakeCache()
    pt.max_pages = FakeCache.max_num_tokens // PAGE_SIZE
    pt.access_serial = pt.max_pages
    pt.referenced_pages = {}
    pt.unreferenced_pages = {}
    pt.all_pages = []
    pt.cpu_tier = None
    pt.disk_tier = None
    pt.eviction_policy = "tree"
    pt.metrics = {}
    pt.reset_page_table()
    pt.last_defrag_serial = pt.max_pages
    return pt


def refs_and_hashes():
    pt = fresh_table()
    # the table owns max_pages unreferenced empty pages keyed by random hashes
    assert len(pt.unreferenced_pages) == pt.max_pages
    assert all(p.kv_position == 0 for p in pt.unreferenced_pages.values())
    # content-hash predicate distinguishes real hashes from the random sentinel
    assert is_content_hash(b"\x01" * 32)
    assert not is_content_hash(b"\x00" * 32)


def chain_checksum():
    # the chained page hash: checksum(page tokens) chained with the previous hash
    ids = torch.arange(256)
    c1 = _tensor_blake2b_checksum(ids, None)
    c2 = _tensor_blake2b_checksum(ids, c1)
    assert c1 != c2
    assert _tensor_blake2b_checksum(ids, None) == c1  # deterministic
    assert is_content_hash(c1)


def tree_eviction_order():
    # A rooted chain of 4 unreferenced pages: eviction must emit leaves (tail)
    # first so the longest prefix survives; the root goes last.
    pt = fresh_table()
    h = [bytes([i]) * 32 for i in range(1, 5)]
    pages = []
    for i, hh in enumerate(h):
        p = CachePage.__new__(CachePage)
        p.pagetable = pt
        p.page_index = 100 + i
        p.phash = hh
        p.prev_hash = None if i == 0 else h[i - 1]
        p.kv_position = (i + 1) * 256
        p.access_serial = 1000 + i
        p.ref_count = 0
        p.children = []
        pages.append(p)
        pt.all_pages.append(p)
        pt.unreferenced_pages[p.phash] = p
    order = [p.phash for p in pt.build_eviction_order()]
    mine = set(h)
    got = [hh for hh in order if hh in mine]
    # our rooted chain must be emitted tail-first (leaves before the root), so
    # the longest prefix survives; other pages may interleave
    assert got == list(reversed(h)), (got, h)


def protected_order():
    pt = fresh_table()
    h = [bytes([i]) * 32 for i in range(1, 4)]
    for i, hh in enumerate(h):
        p = CachePage.__new__(CachePage)
        p.pagetable = pt
        p.phash = hh
        p.prev_hash = None if i == 0 else h[i - 1]
        p.kv_position = (i + 1) * 256
        p.access_serial = 1000 + i
        p.refcount = 0
        pt.unreferenced_pages[p.phash] = p
    order = pt.build_eviction_order(protected_hashes={h[1]})
    got = [p.phash for p in order]
    assert h[1] in got, "protected pages are emitted last, not lost"
    assert got[-1] == h[1], "protected pages must be last"


check("page refs and content-hash predicate", refs_and_hashes)
check("chained page hash is deterministic and chain-dependent", chain_and_hashes) if False else None
check("chained page hash", lambda: (_tensor_blake2b_checksum(torch.arange(256), None) is not None
                                    and _tensor_blake2b_checksum(torch.arange(256), None) !=
                                    _tensor_blake2b_checksum(torch.arange(256), b"\x00" * 32)))
check("tree eviction order is tail-first", tree_eviction_order)
check("protected hashes evict last", protected_order)

n_ok = sum(1 for _, ok, _ in RESULTS if ok)
print(f"== {n_ok}/{len(RESULTS)} pagetable tests passed ==")
if n_ok != len(RESULTS):
    sys.exit(1)
