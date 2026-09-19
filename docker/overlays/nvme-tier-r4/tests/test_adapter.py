"""DiskPageCache + the real PageTable / Sequence / RecurrentCache on CPU tensors: proactive copy-out, idle drain,
restart reuse (bit-exact pages and checkpoints), crash after the drained line, eviction path, failure handling,
generator-thread I/O discipline, the cap under churn, and (NVME_STACKED=1) the recurrent-tip-r1 seams."""
import builtins
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import torch

import harness
from harness import MiniEngine, tokens_for, PAGE_SIZE

m = harness.load()
dc = m.disk_cache


def prompt(seed, pages, extra = 100):
    return tokens_for(seed, pages * PAGE_SIZE + extra)


@pytest.fixture
def mdir(tmp_path):
    return harness.model_dir(tmp_path)


def test_default_hooks_inert_without_tier(tmp_path, mdir):
    e = MiniEngine(tmp_path / "tier", mdir, tier = False)
    assert e.pt.disk_tier is None and e.rc.disk_tier is None
    r1 = e.run(prompt(1, 12))
    r2 = e.run(prompt(1, 12) + tokens_for(2, 300))
    assert r2["cached_pages"] == 12 and r2["bad_pages"] == [] and r2["state_ok"]
    assert not (tmp_path / "tier").exists()


def test_proactive_copy_idle_drain_restart_bit_exact(tmp_path, mdir):
    t = prompt(10, 40)
    a = MiniEngine(tmp_path / "tier", mdir)
    r = a.run(t)
    assert a.tier.pending_copies() > 0  # admitted chain queued, pumped 4 pages per iteration
    a.idle()
    assert a.wait_drained()
    assert len(a.tier.backend.pages) == 40 and len(a.tier.backend.checkpoints) == 1
    a.close()
    b = MiniEngine(tmp_path / "tier", mdir)
    assert len(b.tier.backend.pages) == 40
    r = b.run(t + tokens_for(11, 300))
    assert r["cached_pages"] == 40
    assert r["bad_pages"] == [] and r["state_ok"] is True
    assert b.tier.metrics["restored_pages"] == 40 and b.tier.metrics["restored_checkpoints"] == 1
    assert b.tier.flush(10)
    assert b.tier._pool.available() == b.tier.staging_slots
    b.close()


def test_restart_after_sigkill_following_drained_line(tmp_path, mdir):
    """docker rm -f analogue: the child runs two conversations, goes idle, prints the tier's drained line, keeps
    writing (a long third conversation) and is SIGKILLed. The next process reuses both drained conversations bit for
    bit; the half-written third one is pruned or partially usable, never wrong."""
    child = textwrap.dedent(f"""
        import os, sys, time
        sys.path.insert(0, {str(harness.HERE)!r})
        import harness
        from harness import MiniEngine, tokens_for, PAGE_SIZE
        e = MiniEngine({str(tmp_path / "tier")!r}, {str(mdir)!r}, pages = 256)
        e.run(tokens_for(20, 40 * PAGE_SIZE + 100))
        e.run(tokens_for(21, 33 * PAGE_SIZE + 100))
        e.idle()
        assert e.wait_drained()
        e.tier._drained_logged = False
        e.idle()
        time.sleep(0.5)
        print("READY", flush = True)
        e.run(tokens_for(22, 120 * PAGE_SIZE + 100), checkpoints = [40, 80, 120])
        e.idle()
        time.sleep(30)
    """)
    p = subprocess.Popen([sys.executable, "-c", child], stdout = subprocess.PIPE, stderr = subprocess.STDOUT,
                         env = dict(os.environ))
    out = []
    while True:
        line = p.stdout.readline()
        if not line:
            break
        out.append(line)
        if line.strip() == b"READY":
            break
    assert any(b"nvme tier: drained" in l for l in out), out
    time.sleep(0.05)
    p.send_signal(signal.SIGKILL)
    p.wait()
    b = MiniEngine(tmp_path / "tier", mdir, pages = 256)
    for seed, n in ((20, 40), (21, 33)):
        r = b.run(tokens_for(seed, n * PAGE_SIZE + 100) + tokens_for(seed + 100, 300))
        assert r["cached_pages"] == n, (seed, r["cached_pages"])
        assert r["bad_pages"] == [] and r["state_ok"] is True
    r = b.run(tokens_for(22, 120 * PAGE_SIZE + 100) + tokens_for(5, 300))
    assert r["bad_pages"] == [] and r["state_ok"] in (None, True)
    assert b.tier.flush(10)
    assert b.tier.backend.disk_bytes() == b.tier.backend.disk_bytes_stat()
    b.close()


def test_eviction_path_and_disk_restore_under_vram_pressure(tmp_path, mdir):
    e = MiniEngine(tmp_path / "tier", mdir, pages = 48, pump_pages = 1)
    ta = prompt(30, 40)
    e.run(ta)
    tb = prompt(31, 40)
    e.run(tb)  # evicts most of A's pages; the evict hook copies the ones not on disk yet
    e.idle()
    assert e.wait_drained()
    m_ = e.tier.metrics
    assert m_["eviction_copies"] + m_["eviction_store_skips"] > 0
    r = e.run(ta + tokens_for(32, 300))
    assert r["bad_pages"] == [] and r["state_ok"] in (None, True)
    if m_["eviction_store_skips"] == 0:
        assert r["cached_pages"] == 40
    assert m_["restored_pages"] > 0
    e.close()


def test_deeper_disk_checkpoint_beats_shallower_ram_one(tmp_path, mdir):
    t = prompt(40, 40)
    a = MiniEngine(tmp_path / "tier", mdir)
    a.run(t, checkpoints = [16, 40])
    a.idle()
    assert a.wait_drained()
    a.close()
    b = MiniEngine(tmp_path / "tier", mdir)
    r16 = b.run(prompt(40, 16), checkpoints = [16])  # restores the 16-page prefix: VRAM + RAM now hold it
    assert r16["cached_pages"] == 16 and b.tier.metrics["restored_pages"] == 16
    r = b.run(t + tokens_for(41, 300))  # RAM has 16, only the disk has 40: the deeper one wins
    assert r["cached_pages"] == 40 and r["bad_pages"] == [] and r["state_ok"]
    assert b.tier.metrics["restored_checkpoints"] == 2
    assert b.tier.metrics["restored_pages"] == 40
    b.close()


def test_corrupt_page_mid_chain_falls_back_without_error(tmp_path, mdir):
    t = prompt(50, 40)
    a = MiniEngine(tmp_path / "tier", mdir, segment = 256 * 1024)
    a.run(t)
    a.idle()
    assert a.wait_drained()
    st = a.tier.backend
    ph = m.pagetable.Sequence(torch.tensor([t]), torch.tensor([t]))
    ph.prepare(False, 0)
    e20 = st.pages[ph.page_hashes[20]]
    seg = st._segment_path(e20.segment)
    off = e20.offset + dc.HEADER.size + 64
    a.close()
    with open(seg, "r+b") as f:
        f.seek(off)
        f.write(b"\x5a" * 32)
    b = MiniEngine(tmp_path / "tier", mdir, segment = 256 * 1024)
    if ph.page_hashes[20] not in b.tier.backend.pages:
        pytest.skip("corruption landed in the newest segment and was caught at open")
    r = b.run(t + tokens_for(51, 300))
    assert r["cached_pages"] == 0 and r["bad_pages"] == []
    assert b.tier.metrics["restore_failures"] == 1
    assert b.tier.metrics["quarantined"] == 1
    # the abort guard: the 20 pages restored before the bad one were reset and the disk checkpoint left RAM,
    # so the job ran cold (r["cached_pages"] == 0) with nothing half-restored left behind
    assert b.tier.metrics["restore_aborts"] == 1 and b.tier.metrics["read_fail_digest"] == 2
    assert ph.page_hashes[39] not in b.rc or b.rc.get(ph.page_hashes[39]) is not None
    b.tier.pump()
    assert b.tier.flush(10)
    assert b.tier._pool.available() == b.tier.staging_slots
    b.close()


def test_open_restore_batch_released_after_failed_allocation(tmp_path, mdir):
    t = prompt(60, 20)
    a = MiniEngine(tmp_path / "tier", mdir)
    a.run(t)
    a.idle()
    assert a.wait_drained()
    a.close()
    b = MiniEngine(tmp_path / "tier", mdir)
    seq = m.pagetable.Sequence(torch.tensor([t]), torch.tensor([t]))
    seq.prepare(False, 0)
    batch = b.tier.begin_restore(seq.page_hashes, b.pt)  # allocation "raises" before end_restore
    assert batch is not None and batch.futures
    time.sleep(0.2)
    b.tier.pump()  # next iteration releases it
    assert b.tier._open_batch is None
    assert b.tier._pool.available() == b.tier.staging_slots
    b.close()


def test_multimodal_chain_never_persisted(tmp_path, mdir):
    e = MiniEngine(tmp_path / "tier", mdir)
    t = prompt(70, 40)
    t[3 * PAGE_SIZE + 5] = harness.load().mm.FIRST_MM_EMBEDDING_INDEX + 17
    e.run(t)
    e.idle()
    assert e.wait_drained()
    assert e.tier.metrics["mm_refusals"] >= 1
    assert len(e.tier.backend.pages) == 0 and len(e.tier.backend.checkpoints) == 0
    e.close()


def test_placeholder_and_short_chains_refused(tmp_path, mdir):
    e = MiniEngine(tmp_path / "tier", mdir, min_pages = 8)
    placeholder = (7).to_bytes(16, "big")
    assert not e.tier.persist_checkpoint(placeholder, harness.state_value(placeholder, 256))
    assert e.tier.metrics["placeholder_refusals"] == 1
    e.run(prompt(71, 4))
    assert e.tier.metrics["admission_rejects"] >= 1 and e.tier.pending_copies() == 0
    e.close()


def test_idle_drain_stops_on_busy(tmp_path, mdir, monkeypatch):
    e = MiniEngine(tmp_path / "tier", mdir, pump_pages = 0)
    real = e.tier.backend.store_page

    def slow(*a, **k):
        time.sleep(0.03)
        return real(*a, **k)
    monkeypatch.setattr(e.tier.backend, "store_page", slow)
    e.run(prompt(80, 40))
    assert e.tier.pending_copies() == 40
    e.tier.on_idle()
    time.sleep(0.15)
    e.tier.on_busy()
    time.sleep(0.05)
    frozen = e.tier.pending_copies()
    time.sleep(0.3)
    assert e.tier.pending_copies() == frozen and 0 < frozen < 40
    e.tier.on_idle()
    assert e.wait_drained(30)
    assert len(e.tier.backend.pages) == 40
    e.close()


def test_generator_thread_never_touches_files(tmp_path, mdir, monkeypatch):
    t = prompt(90, 40)
    a = MiniEngine(tmp_path / "tier", mdir, pages = 48)
    main = threading.get_ident()
    calls = []

    def wrap(mod, name):
        real = getattr(mod, name)

        def f(*args, **kw):
            if threading.get_ident() == main:
                calls.append(name)
            return real(*args, **kw)
        monkeypatch.setattr(mod, name, f)
    for name in ("open", "read", "write", "writev", "pwrite", "pread", "preadv", "fsync", "unlink", "replace",
                 "rename", "truncate", "ftruncate", "statvfs", "stat", "lseek", "posix_fadvise"):
        if hasattr(os, name):
            wrap(os, name)
    wrap(builtins, "open")
    a.run(t)
    a.run(prompt(91, 40))  # VRAM pressure: eviction-path copies
    a.idle()
    assert a.wait_drained()
    a.run(t + tokens_for(92, 300))  # restore from disk (reads on reader threads)
    a.idle()
    assert a.wait_drained()
    monkeypatch.undo()
    assert calls == [], sorted(set(calls))
    assert a.tier.metrics["restored_pages"] > 0
    a.close()


def test_writer_errors_disable_writes_never_raise(tmp_path, mdir, monkeypatch):
    e = MiniEngine(tmp_path / "tier", mdir)

    def boom(*a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(e.tier.backend, "store_page", boom)
    monkeypatch.setattr(e.tier.backend, "store_checkpoint", boom)
    for i in range(6):
        e.run(prompt(100 + i, 12))
        e.idle()
        e.wait_drained(5)
    assert e.tier.metrics["writer_errors"] >= dc.WRITER_MAX_CONSECUTIVE_ERRORS
    assert e.tier.writes_disabled
    r = e.run(prompt(100, 12) + tokens_for(7, 300))  # still serves from VRAM
    assert r["cached_pages"] == 12 and r["state_ok"]
    e.close()


def test_cap_holds_through_adapter_churn(tmp_path, mdir):
    cap = 3 * 1024**2
    e = MiniEngine(tmp_path / "tier", mdir, pages = 64, cap = cap, segment = 256 * 1024)
    root = tmp_path / "tier"
    peak = [0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            total = 0
            for dirpath, _, files in os.walk(root):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(dirpath, f))
                    except OSError:
                        pass
            peak[0] = max(peak[0], total)
            time.sleep(0.001)
    th = threading.Thread(target = sample)
    th.start()
    try:
        for i in range(40):
            e.run(prompt(200 + i, 10 + (i % 7)))
            e.idle()
            e.wait_drained()
    finally:
        stop.set()
        th.join()
    b = e.tier.backend
    assert b.disk_bytes() <= cap and b.disk_bytes() == b.disk_bytes_stat()
    assert peak[0] <= cap + b.segment_bytes
    assert b.metrics["pruned_pages"] > 0 and b.metrics["compactions"] > 0
    e.close()


def test_checkpoint_payload_bit_exact_and_not_pickle():
    h = b"\x01" * 16
    s = harness.state_value(h, 4096)
    s[("nested", (1, 2))] = (torch.arange(5, dtype = torch.int64),)
    parts = dc.serialize_stash(s)
    blob = bytearray(b"".join(bytes(memoryview(p).cast("B")) for p in parts))
    assert blob[:4] == b"RCP1"
    back = dc.deserialize_stash(blob)
    assert harness.same_stash(s, back)
    with pytest.raises(ValueError):
        dc.deserialize_stash(bytearray(b"\x80\x04" + b"\0" * 64))


@pytest.mark.skipif(not harness.STACKED, reason = "needs recurrent-tip-r1 (NVME_STACKED=1)")
def test_tip_checkpoint_persisted_on_put(tmp_path, mdir):
    e = MiniEngine(tmp_path / "tier", mdir, tip_policy = True)
    t = prompt(300, 40)
    e.iterate_start()
    ids = torch.tensor([t])
    seq = m.pagetable.Sequence(ids, ids.clone())
    seq.prepare(False, 16)
    seq.allocate_pages(e.pt, e.rc)
    for i in range(40):
        p = seq.allocated_pages[i]
        e.write_page(p, seq.page_hashes[i])
        p.sequence[0].copy_(ids[0, i * PAGE_SIZE:(i + 1) * PAGE_SIZE])
        p.kv_position = PAGE_SIZE
        p.prev_hash = None if i == 0 else seq.allocated_pages[i - 1].phash
    key = seq.page_hashes[39]
    e.rc.put_tip(key, harness.state_value(key, 40 * PAGE_SIZE), min_parent_position = 1)
    assert e.rc.tip_meta[key].kind == "tip"
    e.pt.deallocate_pages(seq.allocated_pages)
    e.idle()
    assert e.wait_drained()
    assert key in e.tier.backend.checkpoints and len(e.tier.backend.pages) == 40
    e.close()


@pytest.mark.skipif(not harness.STACKED, reason = "needs recurrent-tip-r1 (NVME_STACKED=1)")
def test_tip_policy_eviction_persists_through_on_evict(tmp_path, mdir):
    e = MiniEngine(tmp_path / "tier", mdir, tip_policy = True, pages = 256)
    real_enqueue = e.tier._enqueue
    e.tier._enqueue = lambda item: False if item[0] == "checkpoint" else real_enqueue(item)
    size = harness.state_value(b"\0" * 16, 0)["checkpoint_size"]
    e.rc.max_size = 3 * size
    r1 = e.run(prompt(310, 20))
    e.run(prompt(311, 20))
    e.run(prompt(312, 20))
    assert e.tier.metrics["write_queue_skips"] >= 3 and len(e.tier.backend.checkpoints) == 0
    e.tier._enqueue = real_enqueue
    e.run(prompt(313, 20))  # tip policy makes room: tip_drop -> on_evict -> persisted now
    e.idle()
    assert e.wait_drained()
    assert e.tier.metrics["evict_persist"] >= 1
    assert r1["page_hashes"][19] in e.tier.backend.checkpoints
    assert sum(e.rc.tip_metrics[k] for k in e.rc.tip_metrics if k.startswith("evict_")) >= 1
    e.close()


@pytest.mark.skipif(not harness.STACKED, reason = "needs recurrent-tip-r1 (NVME_STACKED=1)")
def test_restart_restore_through_tip_aware_cache(tmp_path, mdir):
    t = prompt(320, 40)
    a = MiniEngine(tmp_path / "tier", mdir, tip_policy = True)
    a.run(t)
    a.idle()
    assert a.wait_drained()
    a.close()
    b = MiniEngine(tmp_path / "tier", mdir, tip_policy = True)
    r = b.run(t + tokens_for(321, 300))
    assert r["cached_pages"] == 40 and r["bad_pages"] == [] and r["state_ok"]
    key = r["page_hashes"][39]
    assert b.rc.tip_meta[key].kind == "std"
    b.close()


RESTORE_CHILD = """
import sys
sys.path.insert(0, {here!r})
import harness
from harness import MiniEngine, tokens_for, PAGE_SIZE
e = MiniEngine({tier!r}, {mdir!r}, pages = 256)
mode = sys.argv[1]
t = tokens_for(90, 60 * PAGE_SIZE + 100)
if mode == "write":
    e.run(t, checkpoints = [30, 50, 60])
    e.idle()
    assert e.wait_drained()
    print("WROTE", len(e.tier.backend.pages), len(e.tier.backend.checkpoints), flush = True)
else:
    r = e.run(t + tokens_for(91, 300))
    m = e.tier.metrics
    print("READ", r["cached_pages"], r["bad_pages"], r["state_ok"], m["restored_pages"], m["restored_checkpoints"],
          m["lookup_hit"], flush = True)
    print(e.tier.summary(), flush = True)
e.close()
"""


def test_restore_in_fresh_process_with_another_hash_seed(tmp_path, mdir):
    """R526 FAIL 1 regression guard: write in one process, restore in a fresh process with a different
    PYTHONHASHSEED (the page and checkpoint keys are content hashes, nothing per-process may enter them)."""
    child = RESTORE_CHILD.format(here = str(harness.HERE), tier = str(tmp_path / "tier"), mdir = str(mdir))
    out = {}
    for mode, seed in (("write", "11"), ("read", "12345")):
        env = dict(os.environ, PYTHONHASHSEED = seed)
        r = subprocess.run([sys.executable, "-c", child, mode], capture_output = True, text = True, env = env,
                           timeout = 120)
        assert r.returncode == 0, r.stdout + r.stderr
        out[mode] = r.stdout
    assert "WROTE 60 3" in out["write"], out["write"]
    line = [l for l in out["read"].splitlines() if l.startswith("READ")][0]
    assert line == "READ 60 [] True 60 1 1", out["read"]
    assert "lookups 1: hit 1" in out["read"], out["read"]


def test_checkpoint_read_failure_is_reported_and_job_runs_cold(tmp_path, mdir):
    """A checkpoint whose bytes on disk do not match its digest: the lookup says why (load-read:digest), the
    record is quarantined, the job runs cold with nothing restored, and the next prefill re-persists it."""
    t = prompt(95, 40)
    a = MiniEngine(tmp_path / "tier", mdir, segment = 256 * 1024)
    r0 = a.run(t)
    a.idle()
    assert a.wait_drained()
    st = a.tier.backend
    key = r0["page_hashes"][39]
    e = st.checkpoints[key]
    seg, off = st._segment_path(e.segment), e.offset + dc.HEADER.size + e.payload_len // 2
    a.close()
    with open(seg, "r+b") as f:
        f.seek(off)
        f.write(b"\xa5" * 16)
    b = MiniEngine(tmp_path / "tier", mdir, segment = 256 * 1024)
    if key not in b.tier.backend.checkpoints:
        pytest.skip("corruption landed in the newest segment and was caught at open")
    r = b.run(t + tokens_for(96, 300), checkpoints = [40, 41])  # a real prefill stashes at the same page again
    mt = b.tier.metrics
    assert r["cached_pages"] == 0 and r["bad_pages"] == []
    assert mt["lookup_load-failed"] == 1 and mt["read_fail_digest"] == 2 and mt["quarantined"] == 1
    assert mt["restored_pages"] == 0 and mt["restored_checkpoints"] == 0
    assert "load-read:digest" in b.tier.last_miss and "quarantined 1" in b.tier.summary()
    b.idle()
    assert b.wait_drained()
    assert key in b.tier.backend.checkpoints  # the cold run's checkpoint replaced the bad record
    b.close()


def test_write_verify_failure_drops_the_record(tmp_path, mdir, monkeypatch):
    e = MiniEngine(tmp_path / "tier", mdir)
    monkeypatch.setattr(dc.SegmentStore, "verify_record", lambda self, entry: "digest")
    e.run(prompt(97, 40))
    e.idle()
    e.wait_drained()
    assert e.tier.metrics["write_verify_failures"] == 1
    assert len(e.tier.backend.checkpoints) == 0 and e.tier.metrics["checkpoint_writes"] == 0
    e.close()


def test_pump_is_time_budgeted_and_zero_disables_it(tmp_path, mdir):
    """FAIL 3: the per-iteration pump spends at most EXL3_NVME_TIER_PUMP_PCT of the iteration time on copies;
    EXL3_NVME_TIER_PUMP_PAGES=0 turns it off. The idle drain still finishes every chain."""
    for pages, pct, lo in ((0, 1.0, 0), (4, 1.0, 0), (4, 100.0, 20)):
        e = MiniEngine(tmp_path / f"tier-{pages}-{pct}", mdir)
        e.tier.pump_frac = pct / 100.0
        e.tier.pump_pages = pages
        e.run(prompt(98, 40), pump_after = False)
        t0 = time.perf_counter()
        for _ in range(60):
            e.iterate_start()
            time.sleep(0.002)
        elapsed = time.perf_counter() - t0
        n = e.tier.metrics["pump_copies"]
        spent = e.tier.metrics["pump_seconds"]
        assert lo <= n, (pages, pct, n)
        if pages == 0:
            assert n == 0
        elif pct < 100:
            # never more than the budget earned over the loop plus one page
            assert spent <= e.tier.pump_frac * elapsed + 2 * e.tier._page_cost, (n, spent, elapsed)
        e.idle()
        assert e.wait_drained()
        assert len(e.tier.backend.pages) == 40
        e.close()


def test_open_scan_reports_and_quarantines_bad_checkpoints(tmp_path, mdir, capfd):
    t1, t2 = prompt(120, 40), prompt(121, 36)
    a = MiniEngine(tmp_path / "tier", mdir, segment = 256 * 1024)
    r1 = a.run(t1)
    a.run(t2)
    a.idle()
    assert a.wait_drained()
    st = a.tier.backend
    e = st.checkpoints[r1["page_hashes"][39]]
    seg, off = st._segment_path(e.segment), e.offset + dc.HEADER.size + 8
    a.close()
    with open(seg, "r+b") as f:
        f.seek(off)
        f.write(b"\x11" * 8)
    b = MiniEngine(tmp_path / "tier", mdir, segment = 256 * 1024, scan = True)
    if len(b.tier.backend.checkpoints) != 2:
        pytest.skip("corruption landed in the newest segment and was caught at open")
    b.tier._scan_thread.join(10)
    out = capfd.readouterr().out
    assert "open scan: 1/2 checkpoints intact, 1 failed (digest 1)" in out, out
    assert len(b.tier.backend.checkpoints) == 1 and b.tier.metrics["scan_bad"] == 1
    r = b.run(t2 + tokens_for(122, 300))
    assert r["cached_pages"] == 36 and r["state_ok"] is True
    b.close()


class _AliasState:
    """A recurrent state whose stash() returns, like PLELayerState.stash (modules/ple.py), a live view of
    CPU-resident state for one entry (the carried token-id context)."""

    def __init__(self, stashed, live):
        self.stashed = dict(stashed)
        self.stashed[(99, 0)] = (live[1, :3].cpu(),)  # .cpu() of a CPU tensor is the tensor itself
        self.position = stashed["position"]
        self.checkpoint_size = stashed["checkpoint_size"]

    def stash(self):
        return self.stashed


@pytest.mark.parametrize("fixed", [True, False])
def test_live_view_in_stash_is_snapshotted_at_put(tmp_path, mdir, monkeypatch, fixed):
    """R526 FAIL 1 root cause. The PLE id context is stashed as a live view of the slot; the next forward overwrites
    it in place. With the tier on, put() snapshots it, so RAM and disk hold the checkpoint-time ids. The control
    (fixed=False) shows what happened before: the writer persisted whatever the slot held when it got to it."""
    if not fixed:
        monkeypatch.setattr(dc.DiskPageCache, "own_stash", lambda self, state, stashed: None)
    gate = threading.Event()
    real = dc.DiskPageCache._snapshot_checkpoint

    def slow_snapshot(self, key, stashed):
        gate.wait(10)  # the writer reaches the checkpoint only after the "next forward" below
        return real(self, key, stashed)

    monkeypatch.setattr(dc.DiskPageCache, "_snapshot_checkpoint", slow_snapshot)
    t = prompt(130, 40)
    e = MiniEngine(tmp_path / "tier", mdir)
    r = e.run(t, checkpoints = [])
    h = r["page_hashes"][39]
    live = torch.full((4, 8), 7, dtype = torch.long)
    e.rc.put(h, _AliasState(harness.state_value(h, 40 * PAGE_SIZE), live))
    live[1, :3] = 999  # the slot's next forward
    ram = e.rc.get(h)[(99, 0)][0]
    assert ram.tolist() == ([7, 7, 7] if fixed else [999, 999, 999])
    gate.set()
    e.idle()
    assert e.wait_drained()
    e.close()
    b = MiniEngine(tmp_path / "tier", mdir)
    st, why = b.tier.load_checkpoint(h)
    assert why == "" and st[(99, 0)][0].tolist() == ([7, 7, 7] if fixed else [999, 999, 999])
    if fixed:
        assert harness.same_stash({k: v for k, v in st.items() if k != (99, 0)},
                                  harness.state_value(h, 40 * PAGE_SIZE))
    b.close()
