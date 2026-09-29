"""prefill-merge r1: a replay of Job.prefill's chunking for a solo job (no exllamav3 import here).

Shared by tests/test_partition.py (CPU, with the harness stubs) and tests/gpu_stash_equiv.py (in the image, with the
real package): the caller passes the package's split_point (cache/prefill_merge.py) and window_ends
(generator/prefill_pipeline.py), so the rule under test is the installed one. Replays job.py:1231-1234 (chunk end),
:1316-1333 (served last-page cut or the merged split), :1491-1495 (stashes) and generator.recurrent_checkpoint()
after each round (job.py:1581-1639, default intervals 2,048 / 32,768), with or without the LS prefill pipeline.
"""
CHUNK = 2048
FINE = 2048
COARSE = 32768


def simulate(cached, n_new, merge, pipeline = False, *, split_point, window_ends, PAGE_SIZE = 256):
    """Serial forwards [(start, end, split_abs or None)], windows [[ends]], stash positions (absolute)."""
    prompt_end = cached + n_new - 1
    kv = cached
    last_ckpt = cached if cached else None
    forwards, windows, stashes = [], [], []

    def is_boundary(pos, interval = None):
        if interval:
            return pos % interval == 0
        if pos >= prompt_end - CHUNK * 2:
            return (pos - cached) % FINE == 0
        return (pos - cached) % COARSE == 0

    def maybe_stash(pos, interval = None):
        nonlocal last_ckpt
        if pos == 0:
            return
        if is_boundary(pos, interval) and last_ckpt != pos:
            assert pos % PAGE_SIZE == 0
            stashes.append(pos)
            last_ckpt = pos

    while kv < prompt_end:
        if pipeline and kv % PAGE_SIZE == 0:
            def checkpoint(end):
                iv = FINE if end >= prompt_end - 4096 else COARSE
                return (end - cached) % iv == 0
            ends = window_ends(kv, prompt_end, checkpoint, (prompt_end - kv) // CHUNK)
            if len(ends) >= 2:
                windows.append(ends)
                for e in ends:          # run_window: job.prefill then maybe_stash_recurrent per chunk
                    kv = e
                    maybe_stash(kv)
                continue
        prefill_start = kv
        prefill_end = min((kv + CHUNK) // PAGE_SIZE * PAGE_SIZE, prompt_end)
        last_page_b = prompt_end // PAGE_SIZE * PAGE_SIZE
        cut, split = False, None
        if prefill_start < last_page_b <= prefill_end:
            split = split_point(prefill_start, prefill_end, prompt_end) if merge else None
            if split is None:
                prefill_end = last_page_b
                cut = True
        forwards.append((prefill_start, prefill_end, split))
        kv = prefill_end
        if cut:
            maybe_stash(kv, PAGE_SIZE)
        if split is not None:                  # prefill_merge.stash_split: the guards of maybe_stash at last_page_b
            if last_ckpt != split:
                stashes.append(split)
                last_ckpt = split
        maybe_stash(kv)                        # generator.recurrent_checkpoint()
    return forwards, windows, stashes


def merged_forwards(cached, n_new, pipeline = False, **fns):
    """The number of merged forwards (0 or 1) the rule predicts for a solo prefill of n_new tokens (N_p = n_new - 1)
    from a page-aligned cached start, and the stash positions (same set with and without merge)."""
    fw, _, stashes = simulate(cached, n_new, True, pipeline, **fns)
    return sum(1 for _, _, sp in fw if sp is not None), stashes
