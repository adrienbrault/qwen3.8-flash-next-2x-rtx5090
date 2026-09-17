"""Opt-in prefill metadata changes; no floating-point kernel or tier-plan changes.

The name is the operator's feature flag, not a promise of zero waits: the exact
reconstruct planner still consumes one count readback per MoE layer.
"""
import os

import torch


def enabled(params):
    return (os.environ.get("EXL3_PREFILL_NOSYNC", "0") == "1" and
            params.get("_prefill_nosync", False))


def expert_histogram(flat_expert_local, num_experts):
    """Fixed-size int64 histogram, including the expert-slice sentinel bin.

    Routing supplies indices in [0, num_experts]. Unlike CUDA bincount, this
    does not read min/max back to the CPU to validate/size its output. Integer
    additions are exact; argsort and all floating-point work remain unchanged.
    """
    counts = torch.zeros((num_experts + 1,), dtype = torch.long,
                         device = flat_expert_local.device)
    counts.scatter_add_(0, flat_expert_local, torch.ones_like(flat_expert_local))
    return counts


def read_counts(counts):
    """One explicit pinned readback + event wait for the unchanged CPU planner.

    Fresh storage is owned until the event completes. Only then may tolist read
    the CPU values. The wait is local to this issue thread; the pipeline's other
    stage can progress. This is NOT elimination of the planner dependency.
    """
    host = torch.empty(counts.shape, dtype = counts.dtype, device = "cpu",
                       pin_memory = True)
    host.copy_(counts, non_blocking = True)
    done = torch.cuda.Event()
    done.record(torch.cuda.current_stream(counts.device))
    done.synchronize()
    return host.tolist()


def upload_metadata(host, device):
    """Own a fresh pinned copy; never asynchronously upload a mutable numpy view.

    PyTorch's pinned allocator tracks the async transfer before recycling the
    source allocation. There is no staging-buffer overwrite or global cache.
    """
    return host.pin_memory().to(device, non_blocking = True)


def join_stages(devices, streams):
    """One host event join covering both stage streams, before draft/commit.

    Both issue threads must have finished submitting before calling this.
    The consumer's join sits AFTER B_i and waits for A_(i+1); it cannot delay
    B_i's kernels. No new stream or scratch arena is introduced.
    """
    ready = torch.cuda.Event()
    done = torch.cuda.Event()
    with torch.cuda.device(devices[0]):
        ready.record(streams[0])
    with torch.cuda.device(devices[1]):
        streams[1].wait_event(ready)
        done.record(streams[1])
    done.synchronize()
