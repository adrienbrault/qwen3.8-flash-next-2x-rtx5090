"""
CPU-only checks for the EXL3_NGRAM_PREFETCH2 port onto the 1.6.7 RowTable (modules/row_table.py):
deferred prefetch (staging set acquired on the worker) returns the same rows as the upstream
non-deferred prefetch and as an inline lookup, the hit/miss/retire accounting matches, and a
deferred entry cancelled before it acquired a set retires cleanly.

Runs inside the built image without a GPU: pinned allocations are replaced by pageable ones
(_PinSet.grow is patched), the table is a RAM store and lookups decode on the CPU.
"""
import torch
import pytest

row_table = pytest.importorskip("exllamav3.modules.row_table")
RowTable = row_table.RowTable
_PinSet = row_table._PinSet


@pytest.fixture(autouse = True)
def _pageable_pins(monkeypatch):
    def grow(self, n, row_words, row_dtype):
        if self.uids is None or self.uids.numel() < n or self.packed.shape[1] != row_words \
                or self.packed.dtype != row_dtype:
            self.packed = torch.empty((n, row_words), dtype = row_dtype)
            self.inverse = torch.empty(n, dtype = torch.int64)
            self.heads = torch.empty(n, dtype = torch.int32)
            self.uids = torch.empty(n, dtype = torch.int64)
    monkeypatch.setattr(_PinSet, "grow", grow)


def _table(rows = 1000, words = 8):
    t = RowTable("t", ["t.weight"])
    store = torch.arange(rows * words, dtype = torch.float16).view(rows, words)
    t._set_stores([store])
    return t, store


def _resolve(ids, pin):
    return ids.view(-1), True


def _decode(rows, pin, U):
    return rows.float()


def _lookup(t, ids):
    return t.lookup(ids, ids.numel(), _resolve, _decode, torch.device("cpu"))


@pytest.mark.parametrize("deferred", [False, True])
def test_prefetch_hit_same_rows(deferred):
    t, store = _table()
    ids = torch.tensor([[5, 17, 999, 0, 5]], dtype = torch.int64)
    t.prefetch(ids, ids.numel(), _resolve, deferred = deferred)
    out = _lookup(t, ids)
    assert torch.equal(out, store[ids.view(-1)].float())
    assert t.prefetch_stats["hit"] == 1 and t.prefetch_stats["miss"] == 0
    assert not t._pending and all(not p.held for p in t._pins)
    t.close()


def test_deferred_miss_falls_back_inline():
    t, store = _table()
    a = torch.tensor([[1, 2, 3]], dtype = torch.int64)
    b = torch.tensor([[4, 5, 6]], dtype = torch.int64)
    t.prefetch(a, a.numel(), _resolve, deferred = True)
    out = _lookup(t, b)
    assert torch.equal(out, store[b.view(-1)].float())
    assert t.prefetch_stats["miss"] == 1
    out = _lookup(t, a)
    assert torch.equal(out, store[a.view(-1)].float())
    assert t.prefetch_stats["hit"] == 1
    t.close()


def test_deferred_retire_more_prefetches_than_sets():
    t, store = _table()
    idss = [torch.tensor([[i, i + 1]], dtype = torch.int64) for i in range(0, 40, 4)]
    for ids in idss:
        t.prefetch(ids, ids.numel(), _resolve, deferred = True)
    t.drain_prefetch()
    assert not t._pending
    assert all(not p.held for p in t._pins)
    assert len(t._pins) <= row_table.MAX_PIN_SETS
    for ids in idss:
        assert torch.equal(_lookup(t, ids), store[ids.view(-1)].float())
    t.close()


def test_retire_unacquired_deferred_entry():
    t, _ = _table()
    entry = {"ids": torch.zeros((1, 1), dtype = torch.int64), "pin": None, "future": None}

    class _F:
        def cancel(self):
            return True
    entry["future"] = _F()
    t._pending.append(entry)
    t._retire(entry)        # must not touch entry["pin"] (None)
    assert not t._pending and t.prefetch_stats["retired"] == 1


def test_deferred_stage_and_inline_miss_never_share_a_set():
    """rebase167 fix round 1: a lookup miss on the issuing thread and a deferred stage on the worker acquire staging
    sets concurrently; choosing a free set and marking it held must be one step. Unguarded (served prefetch2 r1 and
    2ea62832) both could take the same set, and the miss returned rows the worker was writing: about 11 % of
    iterations failed on a Mac CPU run, 0 of 5000 with the lock"""
    for _ in range(300):
        t, store = _table()
        a = torch.tensor([[1, 2, 3]], dtype = torch.int64)
        b = torch.tensor([[4, 5, 6]], dtype = torch.int64)
        t.prefetch(a, a.numel(), _resolve, deferred = True)
        assert torch.equal(_lookup(t, b), store[b.view(-1)].float())
        assert torch.equal(_lookup(t, a), store[a.view(-1)].float())
        assert not t._pending and all(not p.held for p in t._pins)
        t.close()
