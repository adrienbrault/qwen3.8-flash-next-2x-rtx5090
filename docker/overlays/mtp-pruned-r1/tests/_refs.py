"""
Torch-only reference helpers shared by the GPU unit script and the CPU tests.

R522 lesson: two levels of output must never be compared with each other.
  - raw rows: PrunedEmbeddingMirror.lookup / the full-mirror gather / weight[ids], all in the TABLE dtype
    (the checkpoint's embed_tokens is loaded with allow_bf16=True, embedding.py:55, so it may be bf16)
  - Embedding.forward output: the same rows after to2(x, out_dtype) (embedding.py tail), i.e. fp16 when the
    MTP input layer asks for out_dtype=torch.half (arch_specific/qwen4_exp_mtp.py:154)
"""
import torch
import torch.nn.functional as F


def raw_rows(weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Exact table rows for ``ids`` (any device), in the table dtype, on the host."""
    return F.embedding(ids.cpu(), weight.cpu())


def forward_reference(weight: torch.Tensor, ids: torch.Tensor, out_dtype: torch.dtype = torch.half) -> torch.Tensor:
    """What Embedding.forward returns for plain token IDs with normalize=False, multiplier=1.0 (the Qwen4Exp trunk
    embedding, architecture/qwen4_exp.py:237-242): the rows cast to out_dtype (cast done on the host here)."""
    return raw_rows(weight, ids).to(out_dtype)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape or a.element_size() != 2:
        return False
    return torch.equal(a.cpu().view(torch.int16), b.cpu().view(torch.int16))


def describe(a: torch.Tensor, b: torch.Tensor) -> str:
    """Why two tensors are (not) bit-equal: dtype, shape, and the number of differing 16-bit words."""
    s = f"{a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}"
    if a.shape == b.shape and a.element_size() == b.element_size() == 2:
        diff = int((a.cpu().view(torch.int16) != b.cpu().view(torch.int16)).sum())
        s += f", {diff} differing words"
    return s
