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


REVISION = "prompt-lookup-r3"
COUNTERS = ("decode_steps", "lookup_checks", "lookup_hits", "lookup_proposed", "lookup_accepted")

def lookup_mask(chain):
    """The MTP opener is not credited as a copied lookup token."""
    return [False] + [True] * (len(chain) - 1)

# All fields are scalar so the existing Tabby finish merger carries them unchanged.
BASE_COUNTERS = COUNTERS
DETAIL_COUNTERS = (
    'lookup_match_3_7', 'lookup_match_8_15', 'lookup_match_16_31', 'lookup_match_32_plus',
    'lookup_opener_mismatches', 'lookup_all_hit_rounds', 'lookup_mixed_rounds',
    'lookup_skipped_forwards', 'lookup_probe_steps', 'lookup_on_steps', 'lookup_off_steps',
    'lookup_enable_transitions', 'lookup_disable_transitions', 'lookup_reprobe_transitions',
    'lookup_shadow_hits', 'lookup_shadow_proposed', 'lookup_shadow_accepted',
    'lookup_reprobe_checks', 'lookup_reprobe_hits', 'lookup_ineligible_steps',
) + tuple(f'lookup_accept_{source}_{pos}' for source in ('mtp', 'copy') for pos in range(3))
FLOAT_COUNTERS = tuple(f'lookup_{state}_seconds' for state in ('probe','on','off'))
COUNTERS = BASE_COUNTERS + DETAIL_COUNTERS + FLOAT_COUNTERS

@dataclass(frozen=True)
class AdaptiveConfig:
    min_hit: float = .5
    min_acc: float = .9
    window: int = 64
    min_proposals: int = 24  # hit rounds, not copied token count
    reprobe: int = 32
    hyst_hit: float = .1
    hyst_acc: float = .05

    def __post_init__(self):
        if not (0 < self.min_hit <= 1 and 0 < self.min_acc <= 1):
            raise ValueError('lookup fractions must be in (0,1]')
        if not (0 <= self.hyst_hit < self.min_hit and 0 <= self.hyst_acc < self.min_acc):
            raise ValueError('invalid lookup hysteresis')
        if min(self.window, self.min_proposals, self.reprobe) < 1 or self.min_proposals > self.window:
            raise ValueError('invalid lookup window/proposal/reprobe limits')

    @classmethod
    def from_env(cls, env):
        names = dict(min_hit='MIN_HIT', min_acc='MIN_ACC', window='WINDOW',
                     min_proposals='MIN_PROPOSALS', reprobe='REPROBE',
                     hyst_hit='HYST_HIT', hyst_acc='HYST_ACC')
        defaults = cls()
        return cls(**{key: (float if key in ('min_hit','min_acc','hyst_hit','hyst_acc') else int)(
            env.get('EXL3_PROMPT_LOOKUP_' + suffix, getattr(defaults,key))) for key,suffix in names.items()})


class AdaptiveLookup:
    """Decision at begin(), observation only at finish(); state retained by requeue.

    PROBE uses shadow proposals. Unobserved tails after served verification stops
    count as rejected: this is a conservative lower bound, not counterfactual logits.
    ON exit uses lower thresholds; OFF requires four sparse samples before probation.
    """
    def __init__(self, config=None):
        from collections import deque
        self.config = config or AdaptiveConfig()
        self.state = 'PROBE'
        self.history = deque(maxlen=self.config.window)
        self.reprobes = deque(maxlen=8)
        self.probe_age = 0
        self.off_age = 0
        self.metrics = dict.fromkeys(DETAIL_COUNTERS, 0)
        self.metrics.update(dict.fromkeys(FLOAT_COUNTERS, 0.0))
        self.pending = None

    def transition(self, state):
        if state == self.state: return
        if state == 'ON': self.metrics['lookup_enable_transitions'] += 1
        elif state == 'OFF': self.metrics['lookup_disable_transitions'] += 1
        else: self.metrics['lookup_reprobe_transitions'] += 1
        self.state = state
        self.history.clear()
        self.probe_age = self.off_age = 0
        if state == 'PROBE': self.reprobes.clear()

    def begin(self, eligible):
        assert self.pending is None, 'uncompleted adaptive round'
        # Only evidence from completed rounds may change this round's state.
        checks = len(self.history)
        hits = sum(h for h,p,a in self.history)
        proposed = sum(p for h,p,a in self.history)
        accepted = sum(a for h,p,a in self.history)
        hit_rate = hits/checks if checks else 0
        acc_rate = accepted/proposed if proposed else 0
        c = self.config
        if self.state == 'PROBE':
            if hits >= c.min_proposals and hit_rate >= c.min_hit and acc_rate >= c.min_acc:
                self.transition('ON')
            elif self.probe_age >= c.window:
                self.transition('OFF')
        elif self.state == 'ON' and checks:
            if (checks >= c.min_proposals and hit_rate < c.min_hit-c.hyst_hit) or (
                hits >= c.min_proposals and acc_rate < c.min_acc-c.hyst_acc):
                self.transition('OFF')
        elif self.state == 'OFF' and len(self.reprobes) >= 4:
            if sum(self.reprobes)/len(self.reprobes) >= min(1, c.min_hit+c.hyst_hit):
                self.transition('PROBE')
        self.metrics['lookup_' + self.state.lower() + '_steps'] += 1
        if not eligible: self.metrics['lookup_ineligible_steps'] += 1
        check = eligible and (self.state != 'OFF' or self.off_age % c.reprobe == 0)
        from time import perf_counter
        self.pending = dict(state=self.state, check=check, chain=None, accepted=0, live=True, started=perf_counter())
        if self.state == 'OFF': self.off_age += 1
        return self.state, check

    def match(self, match):
        if match is not None:
            length = match.source_end-match.source_start
            bucket = '3_7' if length < 8 else '8_15' if length < 16 else '16_31' if length < 32 else '32_plus'
            self.metrics['lookup_match_' + bucket] += 1

    def opened(self, opener, match, width):
        chain = open_lookup_chain(opener, match, width)
        self.pending['chain'] = chain
        if match is not None and len(match.continuation) >= width and int(opener) != match.continuation[0]:
            self.metrics['lookup_opener_mismatches'] += 1
        if chain is not None and self.pending['state'] == 'PROBE':
            self.metrics['lookup_shadow_hits'] += 1
            self.metrics['lookup_shadow_proposed'] += width-1
        return chain

    def observe(self, pos, token):
        p = self.pending
        if p and p['chain'] is not None and p['state'] != 'OFF' and p['live']:
            if pos < len(p['chain']) and int(token) == p['chain'][pos]:
                if pos: p['accepted'] += 1
            else: p['live'] = False

    def finish(self):
        p = self.pending
        if p is None: return
        from time import perf_counter
        self.metrics['lookup_' + p['state'].lower() + '_seconds'] += perf_counter()-p['started']
        if p['check']:
            hit = p['chain'] is not None
            if p['state'] == 'OFF':
                self.reprobes.append(int(hit))
                self.metrics['lookup_reprobe_checks'] += 1
                self.metrics['lookup_reprobe_hits'] += int(hit)
            else:
                proposed = len(p['chain'])-1 if hit else 0
                self.history.append((int(hit), proposed, min(p['accepted'], proposed)))
                if p['state'] == 'PROBE':
                    self.probe_age += 1
                    self.metrics['lookup_shadow_accepted'] += min(p['accepted'], proposed)
        self.pending = None
