"""CPU tests: Qwen4ExpMTPModel.draft_head_keep_ids() agrees with the pruned head sample_from_state uses."""
import types

import pytest
import torch

from _harness import load_mtp

VOCAB_PAD = 248320          # served lm_head width (trellis columns * 16)


def fake_lm(n_full = VOCAB_PAD, bias = None):
    inner = types.SimpleNamespace(
        trellis = torch.zeros((4, n_full // 16, 2), dtype = torch.int16),
        svh = torch.zeros(n_full, dtype = torch.half),
        suh = torch.zeros(16, dtype = torch.half),
        bias = bias,
    )
    return types.SimpleNamespace(inner = inner, device = torch.device("cpu"))


def fake_model(mod, lm):
    m = object.__new__(mod.Qwen4ExpMTPModel)
    target = types.SimpleNamespace(modules = [None, lm], logit_layer_idx = 1)
    m.attached_model = lambda: target       # attach_to stores a weakref; a closure behaves the same here
    return m


@pytest.mark.parametrize("head_n, expect", [(65536, 65536), (65600, 65536), (100, 0), (None, None), (0, None), (VOCAB_PAD, None)])
def test_keep_ids_matches_pruned_head(head_n, expect):
    flags = {} if head_n is None else {"EXL3_MTP_HEAD_N": head_n}
    mod = load_mtp("overlay", **flags)
    lm = fake_lm()
    m = fake_model(mod, lm)
    keep = m.draft_head_keep_ids()
    if expect in (None, 0):
        assert keep is None
        return
    assert torch.equal(keep, torch.arange(expect))
    # the head sample_from_state would use is the one cached by the same call
    trellis, svh, n_cols = m._pruned_head_cache
    assert n_cols == expect == keep.numel()
    assert trellis.shape[1] * 16 == n_cols and svh.shape[0] == n_cols
    # cached: second call returns the same object (the embedding mirror keys on it)
    assert m.draft_head_keep_ids() is keep


def test_keep_ids_none_with_bias():
    mod = load_mtp("overlay", EXL3_MTP_HEAD_N = 65536)
    m = fake_model(mod, fake_lm(bias = torch.zeros(1)))
    assert m.draft_head_keep_ids() is None


def test_keep_ids_none_unattached():
    mod = load_mtp("overlay", EXL3_MTP_HEAD_N = 65536)
    m = object.__new__(mod.Qwen4ExpMTPModel)
    m.attached_model = None
    assert m.draft_head_keep_ids() is None


def test_argmax_range_is_keep_set():
    """sample_from_state takes argmax over n_cols pruned logits: every draft ID is < n_cols by construction."""
    n_cols = 65536
    logits = torch.randn((4, 1, n_cols))
    ids = torch.argmax(logits, dim = -1)
    assert int(ids.max()) < n_cols and int(ids.min()) >= 0
