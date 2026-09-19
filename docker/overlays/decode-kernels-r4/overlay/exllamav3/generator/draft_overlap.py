"""Host-only guards for opt-in batched MTP verification readback.

This module deliberately has no torch dependency so the eligibility and RNG
ordering rules can be regression-tested on a CPU-only development machine.
"""

from __future__ import annotations

import os


DECODE_OVERLAP = os.environ.get("EXL3_DECODE_OVERLAP", "0") == "1"


def verification_batch_key(job):
    """Return the stateless-greedy implementation key, or None if serial sampling is required.

    The fast path may speculatively compute tokens past the first rejected draft position.
    It is therefore restricted to sampling with no mutable sampler/filter/forced-token state
    and no auxiliary probability result whose shape or timing would change.
    """

    key = getattr(job.sampler, "batch_verify_key", None)
    if key is None:
        return None
    if job.new_tokens < 0 or job.forced_ids is not None:
        return None
    if job.filters or job.device_logit_mask is not None:
        return None
    if job.return_probs or job.return_top_tokens > 0:
        return None
    if len(job.sequences) != 1:
        return None
    return key


def common_verification_batch_key(jobs):
    """Require every participating job to have the same stateless-greedy implementation."""

    keys = [verification_batch_key(job) for job in jobs]
    if not keys or any(key is None for key in keys):
        return None
    return keys[0] if all(key == keys[0] for key in keys[1:]) else None


def verification_batch_mode(job):
    """Return ``greedy``/``sampled`` for the round-4 verifier, else ``None``.

    Unlike the older all-jobs greedy path, round 4 launches one sampler call per
    eligible job and batches that job's verification positions. This admits the
    common stateless fused categorical sampler while still rejecting anything
    whose next distribution depends on a previously accepted token.
    """

    mode = getattr(job.sampler, "batch_verify_mode", None)
    if mode not in ("greedy", "sampled"):
        return None
    if job.new_tokens < 0 or job.forced_ids is not None:
        return None
    if job.filters or job.device_logit_mask is not None:
        return None
    if job.return_probs or job.return_top_tokens > 0:
        return None
    if len(job.sequences) != 1:
        return None
    if getattr(job.sampler, "reqs_past_ids", False):
        return None
    return mode


def draw_sampling_seed(job):
    """Advance the per-job RNG exactly where the serial receive_logits path advances it."""

    return job.rng.randint(0, (1 << 32) - 1)
