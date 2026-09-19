from __future__ import annotations
import torch

"""
Pruned device mirror of a host embedding table (EXL3_EMBED_GPU_PRUNED=1).

The MTP device-resident draft chain (EXL3_MTP_DEVICE_DRAFT=1 + EXL3_EMBED_GPU=1) needs the token
embedding of device-resident IDs without a host round-trip. The full fp16 mirror is 1.27 GB, but with
a pruned draft head (EXL3_MTP_HEAD_N) every drafted ID is an argmax column of that head, so only the
head's vocabulary rows can ever be looked up from the device. This module keeps exact copies of just
those rows on the device, plus a vocab -> mirror-row remap (-1 = not mirrored).

Only the first chain position can carry an ID outside the set (the verified target token). Its IDs
originate on the host, so the caller hands the host copy along; the out-of-set decision is a host
operation (no device sync), and an out-of-set batch is gathered from the exact host table into
pinned memory and uploaded asynchronously. Every path returns exact copies of rows of the same table,
so the result is bit-identical to a gather from the full mirror.

Torch-only on purpose (no extension import) so the logic is testable on a CPU-only machine.
"""

stats = {
    "builds": 0,            # mirror (re)builds
    "device_calls": 0,      # lookups served from the device mirror
    "host_calls": 0,        # lookups served from the host table (an out-of-set ID in the batch)
    "host_rows": 0,         # rows uploaded by those host lookups
}


def build_remap(keep_ids: torch.Tensor, num_rows: int) -> torch.Tensor:
    """int32 (num_rows,) table: vocab ID -> row in the mirror, -1 when the ID is not mirrored."""
    keep = keep_ids.to("cpu", torch.long).view(-1)
    if keep.numel() == 0:
        raise ValueError("pruned embedding: empty keep set")
    if int(keep.min()) < 0 or int(keep.max()) >= num_rows:
        raise ValueError(f"pruned embedding: keep IDs outside [0, {num_rows})")
    if torch.unique(keep).numel() != keep.numel():
        raise ValueError("pruned embedding: duplicate keep IDs")
    remap = torch.full((num_rows,), -1, dtype = torch.int32)
    remap[keep] = torch.arange(keep.numel(), dtype = torch.int32)
    return remap


def stage_rows(weight: torch.Tensor, flat_ids: torch.Tensor, buf: torch.Tensor) -> torch.Tensor:
    """
    Exact rows ``weight[flat_ids]`` written into ``buf`` (the reused pinned staging buffer). The table is an
    nn.Parameter with requires_grad=True (embedding.py load: nn.Parameter(weight)); ops with out= refuse
    grad-requiring inputs outside no_grad/inference_mode, so the gather reads a detached view. detach() is a
    metadata-only alias (no copy, no sync), making this independent of the caller's grad mode.
    """
    return torch.index_select(weight.detach(), 0, flat_ids, out = buf)


def is_prefix(keep_ids: torch.Tensor) -> bool:
    keep = keep_ids.to("cpu", torch.long).view(-1)
    return torch.equal(keep, torch.arange(keep.numel(), dtype = torch.long))


class PrunedEmbeddingMirror:
    """
    Device copy of rows ``keep_ids`` of a host table ``weight`` (num_rows, dim).

    prefix keep set [0, n) (the served pruned head, qwen4_exp_mtp.py _pruned_head): the mirror is
        weight[:n] and a mirrored ID is its own row, so the device lookup is one gather on the IDs;
        no device remap table is allocated
    general keep set: mirror = weight[keep_ids], device lookup = gather(remap[ids])
    """

    def __init__(self, weight: torch.Tensor, keep_ids: torch.Tensor, device: torch.device | str):
        device = torch.device(device)
        self.device = device
        self.keep_ids = keep_ids
        self.num_rows, self.dim = weight.shape
        self.n = keep_ids.numel()
        self.prefix = is_prefix(keep_ids)
        self.remap_host = build_remap(keep_ids, self.num_rows)
        w = weight.detach()
        if self.prefix:
            self.mirror = w[:self.n].to(device).contiguous()
            self.remap = None
        else:
            keep = keep_ids.to("cpu", torch.long).view(-1)
            self.mirror = w.index_select(0, keep).to(device).contiguous()
            self.remap = self.remap_host.to(device)
        self._staging = {}
        self._staging_event = None
        stats["builds"] += 1

    def nbytes(self) -> int:
        n = self.mirror.numel() * self.mirror.element_size()
        if self.remap is not None:
            n += self.remap.numel() * self.remap.element_size()
        return n

    def matches(self, weight: torch.Tensor, keep_ids: torch.Tensor, device) -> bool:
        return (
            self.keep_ids is keep_ids and self.device == torch.device(device) and
            (self.num_rows, self.dim) == tuple(weight.shape) and self.mirror.dtype == weight.dtype
        )

    def host_ids_in_set(self, host_ids: torch.Tensor) -> bool:
        """Host-side membership test for a CPU ID tensor (no device work)."""
        h = host_ids.view(-1)
        if h.numel() == 0:
            return True
        if self.prefix:
            return int(h.min()) >= 0 and int(h.max()) < self.n
        if int(h.min()) < 0 or int(h.max()) >= self.num_rows:
            return False
        return bool((self.remap_host[h] >= 0).all())

    def gather_device(self, ids: torch.Tensor) -> torch.Tensor:
        """Rows for device IDs that are all in the keep set (guaranteed by the caller)."""
        stats["device_calls"] += 1
        rows = ids if self.remap is None else self.remap[ids]
        return torch.nn.functional.embedding(rows, self.mirror)

    def gather_host(self, weight: torch.Tensor, host_ids: torch.Tensor) -> torch.Tensor:
        """
        Exact rows of the host table for host IDs (any ID), placed on the mirror's device. CUDA: gathered
        into a reused pinned buffer and uploaded non-blocking; the buffer is only rewritten after the
        previous upload's event completed.
        """
        stats["host_calls"] += 1
        stats["host_rows"] += host_ids.numel()
        out_shape = (*host_ids.shape, self.dim)
        flat = host_ids.reshape(-1).to(torch.long)
        if self.device.type != "cuda":
            return torch.index_select(weight.detach(), 0, flat).view(out_shape)
        key = (flat.numel(), weight.dtype)
        buf = self._staging.get(key)
        if buf is None:
            if len(self._staging) > 8:
                if self._staging_event is not None:
                    self._staging_event.synchronize()
                self._staging.clear()
            buf = torch.empty((flat.numel(), self.dim), dtype = weight.dtype, pin_memory = True)
            self._staging[key] = buf
        if self._staging_event is not None:
            self._staging_event.synchronize()
        stage_rows(weight, flat, buf)
        dev = buf.to(self.device, non_blocking = True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(self.device))
        self._staging_event = ev
        return dev.view(out_shape)

    def lookup(self, weight: torch.Tensor, ids: torch.Tensor, host_ids: torch.Tensor | None) -> torch.Tensor:
        """
        ids: IDs on self.device. host_ids: None when every ID is in the keep set by provenance (argmax of the
        pruned head), else a host tensor with the same values as ``ids``.
        """
        if host_ids is not None and not self.host_ids_in_set(host_ids):
            return self.gather_host(weight, host_ids)
        return self.gather_device(ids)
