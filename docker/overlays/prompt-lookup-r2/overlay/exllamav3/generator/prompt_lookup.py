"""Pure prompt-lookup selection helpers.

The hot path uses the extension's incremental suffix automaton to locate a
previous suffix.  The functions here validate that source and construct the
bounded continuation.  Keeping that policy free of torch/extension imports
also makes its boundary cases testable on a CPU-only host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen = True)
class PromptLookupMatch:
    source_start: int
    source_end: int
    continuation: tuple[int, ...]


def _overlaps(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start < span_end and span_start < end for span_start, span_end in spans)


def build_lookup_match(
    history: Sequence[int],
    source_start: int,
    source_end: int,
    match_length: int,
    continuation_length: int,
    *,
    max_proposal_tokens: int | None = None,
    excluded_spans: Sequence[tuple[int, int]] = (),
) -> PromptLookupMatch | None:
    """Validate a suffix-automaton source and return its following tokens.

    ``source_end`` is exclusive.  Excluded spans are also half-open and are
    used for multimodal placeholder ranges: a lookup may still use ordinary
    text from a multimodal prompt, but never copies an image-token span into
    generated text.
    """

    if match_length <= 0 or continuation_length <= 0:
        raise ValueError("prompt lookup lengths must be positive")
    if max_proposal_tokens is not None and max_proposal_tokens < 0:
        raise ValueError("max_proposal_tokens must be nonnegative or None")

    history_length = len(history)
    if not (0 <= source_start <= source_end <= history_length):
        return None
    if source_end - source_start < match_length:
        return None

    proposal_length = continuation_length
    if max_proposal_tokens is not None:
        proposal_length = min(proposal_length, max_proposal_tokens)
    if proposal_length == 0 or source_end + proposal_length > history_length:
        return None

    suffix_start = history_length - match_length
    if suffix_start < 0:
        return None
    if _overlaps(suffix_start, history_length, excluded_spans):
        return None
    if _overlaps(source_start, source_end + proposal_length, excluded_spans):
        return None

    if tuple(int(x) for x in history[source_end-match_length:source_end]) != tuple(int(x) for x in history[-match_length:]):
        return None

    continuation = tuple(int(token) for token in history[source_end : source_end + proposal_length])
    return PromptLookupMatch(source_start, source_end, continuation)


def find_previous_suffix(
    history: Sequence[int],
    match_length: int,
    continuation_length: int,
    *,
    max_proposal_tokens: int | None = None,
    excluded_spans: Sequence[tuple[int, int]] = (),
) -> PromptLookupMatch | None:
    """Reference matcher for tests and small host-side uses.

    It finds the earliest occurrence of the longest previous suffix, matching
    the suffix automaton's ``min_end`` policy.  Production generation uses
    ``BC_SAM.accept_tensor`` incrementally and then calls
    :func:`build_lookup_match`, avoiding an O(context) scan per decode round.
    """

    if match_length <= 0 or continuation_length <= 0:
        raise ValueError("prompt lookup lengths must be positive")
    n = len(history)
    if n < match_length + 1:
        return None

    for length in range(n - 1, match_length - 1, -1):
        suffix_start = n - length
        suffix = tuple(int(token) for token in history[suffix_start:])
        for start in range(suffix_start):
            end = start + length
            # The source must predate the last token, as it does while the
            # online automaton matches a token immediately before extending.
            if end > n - 1:
                continue
            if tuple(int(token) for token in history[start:end]) != suffix:
                continue
            return build_lookup_match(
                history,
                start,
                end,
                match_length,
                continuation_length,
                max_proposal_tokens = max_proposal_tokens,
                excluded_spans = excluded_spans,
            )
    return None


def open_lookup_chain(
    mtp_first_token: int,
    match: PromptLookupMatch | None,
    draft_depth: int,
) -> tuple[int, ...] | None:
    """Return a full in-window chain only when the MTP head opens it."""

    if draft_depth <= 0:
        raise ValueError("draft_depth must be positive")
    if match is None or len(match.continuation) < draft_depth:
        return None
    chain = match.continuation[:draft_depth]
    return chain if chain[0] == int(mtp_first_token) else None


REVISION = "prompt-lookup-r2"
COUNTERS = ("decode_steps", "lookup_checks", "lookup_hits", "lookup_proposed", "lookup_accepted")

def lookup_mask(chain):
    """The MTP opener is not credited as a copied lookup token."""
    return [False] + [True] * (len(chain) - 1)
