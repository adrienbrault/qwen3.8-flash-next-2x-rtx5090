#!/usr/bin/env python3
"""Export numeric prefill-r1 records and allowlisted provenance; never copy logs.

Usage: python3 -B bench/sanitize_prefill.py SOURCE_CHAIN DESTINATION
"""
import hashlib
import json
from pathlib import Path
import sys

from sanitize_headline import select
from prefill import load


def export(chain, dest):
    chain, dest = Path(chain), Path(dest)
    unit = chain.parent / (chain.name.removesuffix('-chain') + '-b')
    hashes = {}

    def raw(path):
        value = path.read_bytes()
        hashes[str(path.relative_to(chain.parent))] = hashlib.sha256(value).hexdigest()
        return value

    def read(path):
        return json.loads(raw(path))

    def write(name, value):
        (dest / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

    manifest = read(chain / 'packet/MANIFEST.json')
    for name, sha in manifest.items():
        assert hashlib.sha256((chain / 'packet' / name).read_bytes()).hexdigest() == sha, name
    status = raw(unit / 'STATUS').decode().strip()
    exit_code = int(raw(chain / 'exit.txt'))
    assert status == 'VALIDATED' and exit_code == 0
    expected = read(chain / 'packet/expected.json')
    restored = raw(chain / 'restored-model.txt').decode().strip()
    assert restored == expected['model']
    dest.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in raw(unit / 'prefill.jsonl').splitlines()]
    keys = ('i ok max_tokens min_tokens temperature wall_s ttft_s decode_window_s '
            't_start_abs t_first_abs t_last_abs completion_tokens server_completion_tokens '
            'client_frames prompt_tokens finish_reason wall_tps decode_tps conc run round_wall_s tag ctx_requested')
    (dest / 'prefill.jsonl').write_text(''.join(json.dumps(select(r, keys), allow_nan=False) + '\n' for r in records))
    write('analysis.json', read(unit / 'analysis.txt'))
    (dest / 'prefill.tsv').write_bytes(raw(unit / 'prefill.tsv'))
    prov = read(unit / 'boots/S/provenance.json')
    write('provenance.json', {
        **select(prov, 'image_id model pool selectors free_mib_launcher boot_to_serving_s definition kernel_cache'),
        **select(expected, 'image launcher_sha256 window split'),
        'measurement_date': '2026-11-16',
        'source_run': chain.name.removesuffix('-chain'),
        'kernel_cache_seed': 'independent copy of the served tuning cache',
        'kind': 'prose', 'thinking': 'template default',
        'cold_cache_evidence': 'Upstream VALIDATED check requires zero cached tokens for all 22 server records; logs are excluded.',
    })
    write('validation.json', {
        'unit_status': status, 'exit_code': exit_code, 'restored_model': restored,
        'packet_check': read(chain / 'packet-check.json'),
        'verification_limit': 'Numeric request arithmetic is reproducible here; upstream server-log cold-cache checks cannot be repeated from this export.',
    })
    write('packet-source-sha256.json', manifest)
    write('source-sha256.json', hashes)
    load(dest)


if __name__ == '__main__':
    export(*sys.argv[1:])
