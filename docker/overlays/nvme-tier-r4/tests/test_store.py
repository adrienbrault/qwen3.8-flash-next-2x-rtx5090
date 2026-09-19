"""SegmentStore (storage core): records, recovery, cap, eviction order, reclaim, locking, identity."""
import hashlib
import os
import pathlib
import random
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

import harness

m = harness.load()
dc = m.disk_cache
SegmentStore = dc.SegmentStore


def h(i: int) -> bytes:
    # content-hash shaped: top 8 bytes non-zero
    return hashlib.blake2b(i.to_bytes(8, "little"), digest_size = 16).digest()


def payload(key: bytes, n: int) -> bytes:
    out = bytearray()
    seed = key
    while len(out) < n:
        seed = hashlib.blake2b(seed, digest_size = 64).digest()
        out += seed
    return bytes(out[:n])


def open_store(root, cap = 64 * 1024**2, seg = 1024**2, min_pages = 1, **kw):
    return SegmentStore(root, {"t": 1}, cap, 0.0, seg, min_pages, **kw)


def store_chain(s, keys, size = 20000, cp_at = None, cp_size = 30000):
    s.admit_chain(keys)
    prev = None
    for k in keys:
        assert s.store_page(k, prev, payload(k, size))
        prev = k
    for i in (cp_at if cp_at is not None else [len(keys)]):
        k = keys[i - 1]
        assert s.store_checkpoint(k, keys[:i], payload(k + b"cp", cp_size))


def du(root) -> int:
    total = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


def test_round_trip_and_restart(tmp_path):
    keys = [h(i) for i in range(10)]
    s = open_store(tmp_path)
    store_chain(s, keys, cp_at = [5, 10])
    assert s.longest_resumable_prefix(keys) == 10
    s.close()
    s = open_store(tmp_path)
    assert len(s.pages) == 10 and len(s.checkpoints) == 2
    assert s.longest_resumable_prefix(keys) == 10
    for k in keys:
        assert s.fetch_page(k) == payload(k, 20000)
    assert bytes(s.fetch_checkpoint(keys[4])) == payload(keys[4] + b"cp", 30000)
    assert s.disk_bytes() == s.disk_bytes_stat()
    s.close()


def test_torn_tail_truncated_only(tmp_path):
    keys = [h(i) for i in range(6)]
    s = open_store(tmp_path)
    store_chain(s, keys, cp_at = [3, 6])
    seg = sorted(s.path.glob("segment-*.bin"))[-1]
    s.close()
    size = seg.stat().st_size
    with open(seg, "r+b") as f:  # tear the last record (the 6-page checkpoint) in its payload
        f.truncate(size - 5000)
    s = open_store(tmp_path)
    assert s.metrics["torn_records"] == 1
    assert keys[5] not in s.checkpoints and keys[2] in s.checkpoints
    assert s.longest_resumable_prefix(keys) == 3
    assert all(s.fetch_page(k) == payload(k, 20000) for k in keys[:3])
    assert all(k not in s.pages for k in keys[3:])  # past the surviving anchor: stranded, pruned at open
    assert s.disk_bytes() == s.disk_bytes_stat()
    s.close()


def test_torn_payload_same_length_in_newest_segment_detected(tmp_path):
    """A record whose bytes are complete in length but wrong in content (unflushed page cache after a power loss) is
    caught by the full digest check of the newest segment at open."""
    keys = [h(i) for i in range(4)]
    s = open_store(tmp_path)
    store_chain(s, keys[:2])  # pages 0-1 + checkpoint at 2, then pages 2-3 + checkpoint at 4
    s.admit_chain(keys)
    s.store_page(keys[2], keys[1], payload(keys[2], 20000))
    s.store_page(keys[3], keys[2], payload(keys[3], 20000))
    s.store_checkpoint(keys[3], keys, payload(keys[3] + b"cp", 30000))
    e = s.pages[keys[2]]
    seg = s._segment_path(e.segment)
    off = e.offset + dc.HEADER.size + 100
    s.close()
    with open(seg, "r+b") as f:
        f.seek(off)
        f.write(b"\0" * 64)
    s = open_store(tmp_path)
    assert s.metrics["torn_records"] == 1
    assert keys[2] not in s.pages and keys[1] in s.pages
    assert list(s.checkpoints) == [keys[1]]  # everything after the torn record is gone; the 2-page chain survives
    assert s.longest_resumable_prefix(keys) == 2
    s.close()


def test_corrupt_old_segment_quarantined_on_read(tmp_path):
    keys = [h(i) for i in range(30)]
    s = open_store(tmp_path, seg = 128 * 1024)
    store_chain(s, keys)
    first = s.pages[keys[0]]
    assert first.segment != s.active_segment
    seg = s._segment_path(first.segment)
    s.close()
    with open(seg, "r+b") as f:
        f.seek(first.offset + dc.HEADER.size + 10)
        f.write(b"\xff" * 8)
    s = open_store(tmp_path, seg = 128 * 1024)  # header-only scan: not detected at open
    assert keys[0] in s.pages
    assert s.fetch_page(keys[0]) is None
    assert keys[0] not in s.pages and s.metrics["quarantined"] == 1
    assert s.fetch_page(keys[1]) == payload(keys[1], 20000)
    s.close()


def test_open_reads_only_headers_of_old_segments(tmp_path, monkeypatch):
    keys = [h(i) for i in range(60)]
    s = open_store(tmp_path, seg = 256 * 1024)
    store_chain(s, keys, size = 60000)
    newest = s.active_segment
    newest_bytes = s._seg_phys[newest]
    total = s.disk_bytes()
    s.close()
    read = [0]
    real = os.pread

    def counting(fd, n, off):
        b = real(fd, n, off)
        read[0] += len(b)
        return b
    monkeypatch.setattr(dc.os, "pread", counting)
    s = open_store(tmp_path, seg = 256 * 1024)
    assert len(s.pages) == 60
    assert read[0] <= newest_bytes + 128 * 60, (read[0], newest_bytes, total)
    assert read[0] < total / 2
    s.close()


def test_kill9_mid_write(tmp_path):
    """A child process appends 2 MB records as fast as it can and is SIGKILLed mid-stream. The reopened store keeps
    every complete record (all readable, digests OK) and drops at most the one torn tail record."""
    child = textwrap.dedent(f"""
        import os, sys, hashlib
        sys.path.insert(0, {str(harness.HERE)!r})
        os.environ["NVME_STACKED"] = "0"
        import harness
        dc = harness.load().disk_cache
        s = dc.SegmentStore({str(tmp_path)!r}, {{"t": 1}}, 1 << 34, 0.0, 16 << 20, 1)
        i = 0
        prev = None
        while True:
            k = hashlib.blake2b(i.to_bytes(8, "little"), digest_size=16).digest()
            s.admit_chain([k])
            s.store_page(k, None, os.urandom(2_000_000))
            s.store_checkpoint(k, [k], os.urandom(300_000))
            if i == 3:
                print("GO", flush=True)
            prev = k
            i += 1
    """)
    p = subprocess.Popen([sys.executable, "-c", child], stdout = subprocess.PIPE, env = dict(os.environ))
    assert p.stdout.readline().strip() == b"GO"
    time.sleep(random.uniform(0.2, 0.6))
    p.send_signal(signal.SIGKILL)
    p.wait()
    s = open_store(tmp_path, cap = 1 << 34, seg = 16 << 20)
    n = len(s.pages)
    assert n >= 4
    assert s.metrics["torn_records"] <= 1
    for k in list(s.pages):
        assert s.fetch_page(k) is not None
    assert s.disk_bytes() == s.disk_bytes_stat()
    s.close()


def test_cap_under_random_churn(tmp_path):
    """Hard cap on physical bytes under a long random admit/store workload at a small cap, checked after every
    append against the incremental counter, stat() and a du of the whole root; rewrite amplification bounded."""
    rng = random.Random(7)
    cap = 3 * 1024**2
    s = open_store(tmp_path, cap = cap, seg = 256 * 1024)
    assert s.segment_bytes == 256 * 1024
    chains = []
    n = 0
    for step in range(500):
        if chains and rng.random() < 0.4:
            base = rng.choice(chains)  # extend or branch an existing conversation
            cut = rng.randint(1, len(base))
            keys = base[:cut] + [h(10**6 + n + i) for i in range(rng.randint(1, 5))]
        else:
            keys = [h(n + i) for i in range(rng.randint(2, 8))]
        n += 20
        chains.append(keys)
        s.admit_chain(keys)
        prev = None
        for k in keys:
            s.store_page(k, prev, payload(k, rng.randint(8000, 40000)))
            prev = k
            assert s.disk_bytes() <= cap
        s.store_checkpoint(keys[-1], keys, payload(keys[-1] + b"c", 50000))
        assert s.disk_bytes() <= cap
        assert s.disk_bytes() == s.disk_bytes_stat()
        assert du(tmp_path) <= cap + 4096
    assert s.metrics["compactions"] > 0
    assert s.metrics["compaction_rewrite_bytes"] <= 3 * s.metrics["compaction_reclaim_bytes"]
    # index is consistent: every page and checkpoint readable
    for k in list(s.pages):
        assert s.fetch_page(k) is not None
    for k in list(s.checkpoints):
        assert s.fetch_checkpoint(k) is not None
    s.close()
    s = open_store(tmp_path, cap = cap, seg = 256 * 1024)
    assert s.disk_bytes() <= cap
    s.close()


def test_interior_checkpoints_evicted_before_leaves(tmp_path):
    cap = 2 * 1024**2
    s = open_store(tmp_path, cap = cap, seg = 128 * 1024)
    a = [h(i) for i in range(12)]
    store_chain(s, a, size = 30000, cp_at = [4, 8, 12], cp_size = 150000)  # conversation A, 3 checkpoints
    b = [h(100 + i) for i in range(6)]
    store_chain(s, b, size = 30000, cp_at = [6], cp_size = 150000)
    # fill until eviction starts
    i = 0
    while s.metrics["evicted_interior_checkpoints"] == 0 and s.metrics["pruned_pages"] == 0:
        c = [h(1000 + i * 10 + j) for j in range(3)]
        store_chain(s, c, size = 30000, cp_size = 20000)
        i += 1
    assert s.metrics["evicted_interior_checkpoints"] >= 1
    assert s.metrics["pruned_pages"] == 0  # no leaf was evicted while an interior checkpoint existed
    assert a[11] in s.checkpoints and b[5] in s.checkpoints  # newest point of each conversation kept
    assert a[3] not in s.checkpoints  # oldest interior point went first
    s.close()


def test_reads_and_index_queries_not_blocked_by_compaction(tmp_path, monkeypatch):
    s = open_store(tmp_path, cap = 64 * 1024**2, seg = 128 * 1024)
    keys = [h(i) for i in range(30)]
    store_chain(s, keys)
    # drop most of the first segment's records so it is compactable
    first = s.pages[keys[0]].segment
    with s.lock:
        victims = [e for e in list(s.pages.values()) if e.segment == first][1:]
    other = [h(500 + i) for i in range(3)]
    store_chain(s, other)
    gate = threading.Event()
    entered = threading.Event()
    real = SegmentStore._copy_range

    def slow_copy(*a):
        entered.set()
        gate.wait(5)
        return real(*a)
    monkeypatch.setattr(SegmentStore, "_copy_range", staticmethod(slow_copy))
    with s.lock:
        for e in victims:
            s._drop_page_entry(e)
    t = threading.Thread(target = s._compact_forward, args = (first,))
    t.start()
    assert entered.wait(5)
    t0 = time.monotonic()
    assert s.fetch_page(other[1]) == payload(other[1], 20000)
    assert s.should_store_page(h(9999)) is False
    assert s.admit_chain([h(9999)])
    assert s.longest_resumable_prefix(other) == 3
    assert time.monotonic() - t0 < 1.0
    gate.set()
    t.join()
    assert s.fetch_page(keys[0]) == payload(keys[0], 20000)  # forwarded record still readable
    assert s.disk_bytes() == s.disk_bytes_stat()
    s.close()


def test_read_racing_compaction_retries_new_location(tmp_path):
    s = open_store(tmp_path, cap = 64 * 1024**2, seg = 128 * 1024)
    keys = [h(i) for i in range(20)]
    store_chain(s, keys)
    e = s.pages[keys[0]]
    old = (e.segment, e.offset)
    with s.lock:
        for x in [x for x in list(s.pages.values()) if x.segment == old[0] and x.key != keys[0]]:
            s._drop_page_entry(x)
    store_chain(s, [h(700)])
    s._compact_forward(old[0])
    assert (e.segment, e.offset) != old
    assert not s._segment_path(old[0]).exists()
    assert s._read_at(e, old[0], old[1], bytearray(e.payload_len)) == "open"  # stale location fails cleanly
    assert s.fetch_page(keys[0]) == payload(keys[0], 20000)
    s.close()


def test_single_owner_lock_and_namespace_gc(tmp_path):
    s = open_store(tmp_path)
    with pytest.raises(RuntimeError):
        open_store(tmp_path)
    store_chain(s, [h(1), h(2)])
    old = s.path
    s.close()
    (tmp_path / "unrelated").mkdir()
    s2 = SegmentStore(tmp_path, {"t": 2}, 64 * 1024**2, 0.0, 1024**2, 1)
    assert not old.exists() and s2.metrics["namespace_gc"] == 1
    assert (tmp_path / "unrelated").exists()
    s2.close()


def test_free_space_floor_refuses(tmp_path):
    s = open_store(tmp_path, free_space_fn = lambda: 5.0)
    s.min_free_pct = 10.0
    s.admit_chain([h(1)])
    assert not s.store_page(h(1), None, b"x" * 5000)
    assert s.metrics["free_space_refusals"] == 1 and s.disk_bytes() == s._ns_bytes
    s.close()


def test_identity_mtime_insensitive_content_sensitive(tmp_path):
    d = harness.model_dir(tmp_path)
    a = dc._checkpoint_identity(d)
    os.utime(d / "model.safetensors", (1, 1))
    os.utime(d / "config.json", (1, 1))
    assert dc._checkpoint_identity(d) == a
    (d / "config.json").write_text('{"arch": "fake", "layers": 3}')
    assert dc._checkpoint_identity(d) != a


def test_engine_identity(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "generator").mkdir(parents = True)
    (pkg / "a.py").write_text("x = 1\n")
    (pkg / "k.cu").write_text("// kernel\n")
    base = dc.engine_identity(pkg, {"EXL3_MOE_PREFILL_E3": "1", "EXL3_NVME_TIER": "/a", "PATH": "/bin"})
    assert base["env"] == [["EXL3_MOE_PREFILL_E3", "1"]]
    assert dc.engine_identity(pkg, {"EXL3_MOE_PREFILL_E3": "1", "EXL3_NVME_TIER": "/b", "EXL3_NVME_TIER_GB": "2"}) == base
    assert dc.engine_identity(pkg, {"EXL3_MOE_PREFILL_E3": "0"}) != base
    (pkg / "k.cu").write_text("// kernel v2\n")
    assert dc.engine_identity(pkg, {"EXL3_MOE_PREFILL_E3": "1"})["sources_sha256"] != base["sources_sha256"]


def test_lowered_cap_enforced_at_open(tmp_path):
    s = open_store(tmp_path, cap = 64 * 1024**2, seg = 256 * 1024)
    for c in range(12):
        store_chain(s, [h(c * 100 + i) for i in range(10)], size = 30000, cp_size = 60000)
    big = s.disk_bytes()
    s.close()
    cap = big // 3
    s = open_store(tmp_path, cap = cap, seg = 256 * 1024)
    assert s.disk_bytes() <= cap and s.disk_bytes() == s.disk_bytes_stat()
    assert du(tmp_path) <= cap + 4096
    assert s.checkpoints  # newest conversations survive
    for k in list(s.pages):
        assert s.fetch_page(k) is not None
    s.close()
