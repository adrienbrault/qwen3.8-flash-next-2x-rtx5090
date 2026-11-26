"""Read and independently check the published headline-r2 numeric observations."""
import json
import math
from pathlib import Path
import statistics as st

DATA = Path(__file__).resolve().parent / 'results/2026-11-26-headline-r2'


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7):
        raise ValueError(f'numeric mismatch: {a} != {b}')


def percentile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    i = int(x)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)


def indexed(rows, fields, expected):
    out = {tuple(r[k] for k in fields): r for r in rows}
    if len(out) != len(rows) or set(out) != expected:
        raise ValueError(f'missing or duplicate cells for {fields}: {set(out) ^ expected}')
    return out


def load(root=DATA):
    root = Path(root)
    real, code = read(root / 'real-summary.json'), read(root / 'code-summary.json')
    if real['status'] != 'MEASURED' or code['status'] != 'MEASURED':
        raise ValueError('unmeasured inputs')
    rr = indexed(real['table'], ('size', 'concurrency'), {(s, c) for s in ('subset', 'full') for c in (1, 4, 8)})
    cc = indexed(code['table'], ('prefix_target', 'append_target', 'concurrency'), {(p, a, c) for p in (1024, 32768, 65536) for a in (2048, 8192) for c in (1, 4, 8)})
    for (size, c), row in rr.items():
        reqs = lines(root / (row['tag'] + '-requests.jsonl'))
        server = read(root / (row['tag'] + '-server.json'))
        assert len(reqs) == row['requests'] == (120 if size == 'subset' else 480)
        assert len({r['question_id'] for r in reqs}) == len(reqs)
        assert all(r['completion_tokens'] == 256 for r in reqs)
        close(sum(r['completion_tokens'] for r in reqs), row['output_tokens'])
        close(row['output_tokens'] / row['duration_s'], row['output_tok_s'])
        close(len(reqs) / row['duration_s'], row['req_s'])
        tpot = [(r['latency_s'] - r['ttft_s']) / (r['completion_tokens'] - 1) * 1000 for r in reqs]
        close(percentile(tpot, .5), row['tpot_p50_ms'])
        close(percentile(tpot, .99), row['tpot_p99_ms'])
        close(1000 / row['tpot_p50_ms'], row['per_stream_tok_s'])
        close(st.median(r['latency_s'] for r in reqs), row['e2e_p50_s'])
        for q, key in ((.5, 'ttft_p50_ms'), (.99, 'ttft_p99_ms')):
            close(percentile([r['ttft_s'] * 1000 for r in reqs], q), row[key])
        if c == 1:
            close(st.median(r['prompt_tokens'] / r['ttft_s'] for r in reqs), row['prefill_tok_s_from_ttft'])
        assert server['cached_tokens'] == row['cached_prompt_tokens'] == 0
        check_counters(row, server)
    ratios = indexed(real['ratios'], ('concurrency',), {(c,) for c in (1, 4, 8)})
    for (c,), row in ratios.items():
        for key, ratio in row['subset_over_full'].items():
            close(rr['subset', c][key] / rr['full', c][key], ratio)
    observations = lines(root / 'code-requests.jsonl')
    for key, row in cc.items():
        reqs = sorted((r for r in observations if r['tag'] == row['tag']), key=lambda r: r['slot'])
        assert code['context_mode'] == row['context_mode'] == 'shared'
        assert len(reqs) == row['requests'] == key[2]
        assert [r['task_id'] for r in reqs] == row['task_ids']
        assert [r['task_sha256'] for r in reqs] == row['task_sha256']
        assert [r['prompt_tokens'] for r in reqs] == row['actual_prompt_tokens']
        assert [r['cached_tokens'] for r in reqs] == row['cached_tokens']
        assert all(r['context_sha256'] == row['context_sha256'] for r in reqs)
        assert all(r['completion_tokens'] == 400 and r['finish_reason'] == 'length' for r in reqs)
        for r in reqs:
            tl = r['timeline']
            assert tl[0][0] == r['first_s'] and tl[-1] == [r['last_s'], 400]
            assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(tl, tl[1:]))
        lo, hi = max(r['first_s'] for r in reqs), min(r['last_s'] for r in reqs)
        assert hi > lo
        def at(r, t):
            return max([n for when, n in r['timeline'] if when <= t] or [0])
        counts = [at(r, hi) - at(r, lo) for r in reqs]
        w = row['window']
        assert counts == w['output_tokens']
        close(hi - lo, w['duration_s'])
        close(lo, w['start_monotonic_s'])
        close(hi, w['end_monotonic_s'])
        for count, rate in zip(counts, w['per_stream_tok_s']):
            close(count / (hi - lo), rate)
        close(sum(counts) / (hi - lo), w['aggregate_tok_s'])
        close(st.mean(w['per_stream_tok_s']), w['mean_per_stream_tok_s'])
        for q, field in ((.5, 'append_ttft_p50_ms'), (.99, 'append_ttft_p99_ms')):
            close(percentile([r['ttft_s'] * 1000 for r in reqs], q), row[field])
        new = sum(r['prompt_tokens'] - r['cached_tokens'] for r in reqs)
        close(new, row['new_prompt_tokens'])
        close(new / sum(r['ttft_s'] for r in reqs), row['new_tok_s_from_ttft'])
        check_counters(row, read(root / (row['tag'] + '-server.json')))
    return real, code, rr, cc


def check_counters(row, server):
    reqs = server['requests']
    for field, key in (('generated_tokens', 'gen'), ('accepted', 'acc'), ('proposed', 'prop'), ('cached_tokens', 'cached'), ('new_tokens', 'new')):
        close(sum(r[key] for r in reqs), server[field])
    close(server['accepted'], row['draft_accepted'])
    close(server['proposed'], row['draft_proposed'])
    close(server['accepted'] / server['proposed'], row['draft_acceptance'])
    close(server['generated_tokens'] / (server['generated_tokens'] - server['accepted']), row['tau'])


if __name__ == '__main__':
    load()
    print('Validated 6 real-work cells and 18 code cells against numeric request observations')
