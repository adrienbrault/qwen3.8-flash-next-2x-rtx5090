#!/usr/bin/env python3
"""Export an explicit allowlist of numeric headline-r1 observations, IDs and hashes.

Usage: python3 -B bench/sanitize_headline.py SOURCE_CHAIN DESTINATION
No prompts, generated text, token IDs, logs or host paths are exported.
"""
import hashlib
import json
from pathlib import Path
import sys


def select(row, keys):
    return {key: row[key] for key in keys.split()}


def export(chain, dest):
    chain, dest = Path(chain), Path(dest)
    stem = chain.name.removesuffix('-chain')
    real, code = (chain.parent / (stem + '-' + unit) for unit in ('real', 'code'))
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(chain.parent))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def write(name, value):
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

    def lines(name, rows):
        (dest / name).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))

    receipt = read(chain / 'run-status.json')
    assert receipt['exit_code'] == 0
    assert receipt['restore_verified'] or (receipt['restore_deferred'] and (chain / 'restore-deferred.txt').read_text().strip())
    for unit in (real, code):
        assert (unit / 'STATUS').read_text().strip() == 'VALIDATED'
    a, b = read(real / 'summary.json'), read(code / 'summary.json')
    assert a['status'] == b['status'] == 'MEASURED'
    assert {(r['size'], r['concurrency']) for r in a['table']} == {(s, c) for s in ('subset', 'full') for c in (1, 4, 8)}
    assert len(a['table']) == 6 and len(b['table']) == 18
    assert {(r['prefix_target'], r['append_target'], r['concurrency']) for r in b['table']} == {(p, n, c) for p in (1024, 32768, 65536) for n in (2048, 8192) for c in (1, 4, 8)}
    assert b['context_mode'] == 'shared'
    write('run-status.json', receipt)
    write('real-summary.json', a)
    b['thinking_split'] = [select(r, 'kind utc reasoning_frames reasoning_chars content_frames content_chars finish_reason completion_tokens') for r in b['thinking_split']]
    write('code-summary.json', b)
    write('subset-manifest.json', read(chain / 'data/subset-manifest.json'))
    manifest = read(chain / 'data/specbench-manifest.json')
    write('specbench-manifest.json', manifest)  # ID/hash/token-length records only; contains no prompt text.
    template = read(chain / 'data/template-default.json')
    write('template-default.json', {
        'thinking_override': template['thinking_override'],
        'ends_with_think_marker': template['rendered_hi'].endswith('<think>\n'),
        'rendered_hi_sha256': hashlib.sha256(template['rendered_hi'].encode()).hexdigest(),
    })
    prov = read(real / 'boots/SINGLE/provenance.json')
    write('provenance.json', {
        **select(prov, 'image_id model pool selectors free_mib_launcher boot_to_serving_s definition kernel_cache'),
        'image_tag': 'tabbyapi:flashnext-names-fast-r1',
        'kernel_cache_seed': 'independent copy of the served tuning cache',
        'measurement_date': '2026-11-16',
        'source_run': stem,
    })
    write('graft.json', read(chain / 'graft-preflight.json'))
    write('packet-source-sha256.json', read(chain / 'packet/MANIFEST.json'))
    corpus = read(chain / 'packet/inputs/corpus-provenance.json')
    corpus.pop('snapshot_root', None)
    write('input-provenance.json', corpus)
    config = read(chain / 'packet/expected.json')
    write('served-config.json', select(config, 'launcher_sha256 image image_id model pool window split extra_env'))
    for cell in a['table']:
        tag = cell['tag']
        raw = read(real / (tag + '.json'))
        prompts = manifest['cells'][tag]['prompts']
        assert len(prompts) == raw['completed'] == cell['requests'] and raw['failed'] == 0
        rows = []
        for i, prompt in enumerate(prompts):
            assert raw['input_lens'][i] == prompt['served_prompt_tokens']
            rows.append({
                **prompt, 'request_index': i,
                'prompt_tokens': raw['input_lens'][i], 'completion_tokens': raw['output_lens'][i],
                'ttft_s': raw['ttfts'][i], 'latency_s': raw['latencies'][i],
                'start_s': raw['start_times'][i], 'queue_s': raw['queue_times'][i], 'itls_s': raw['itls'][i],
            })
        assert all(r['completion_tokens'] == 256 for r in rows)
        lines(tag + '-requests.jsonl', rows)
        server = read(real / (tag + '.server.json'))
        server['requests'] = [select(r, 'serial gen tps prompt cached new acc prop') for r in server['requests']]
        write(tag + '-server.json', server)
    code_rows = []
    for cell in b['table']:
        tag = cell['tag']
        for row in read(code / tag / 'measure-aligned.json'):
            assert row['completion_tokens'] == 400 and row['finish_reason'] == 'length'
            code_rows.append({
                'tag': tag, **select(cell, 'prefix_target append_target concurrency context_mode context_sha256'),
                **select(row, 'slot phase task_id task_sha256 prompt_tokens completion_tokens finish_reason first_s last_s ttft_s start_s appended_tokens timeline counting_method'),
                'cached_tokens': row['usage']['prompt_tokens_details']['cached_tokens'],
            })
        server = read(code / tag / 'measure.server.json')
        server['requests'] = [select(r, 'serial gen tps prompt cached new acc prop') for r in server['requests']]
        write(tag + '-server.json', server)
    lines('code-requests.jsonl', code_rows)
    write('source-sha256.json', hashes)
    print(f'Exported {len(a["table"])} real-work cells, {len(b["table"])} code cells, {len(code_rows)} code requests')


if __name__ == '__main__':
    export(*sys.argv[1:])
