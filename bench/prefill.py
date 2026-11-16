"""Independently recompute the published cold-prefill curve from request records."""
import csv
import json
import statistics as st
from pathlib import Path

from headline import close, indexed

DATA = Path(__file__).resolve().parent / 'results/2026-11-16-prefill-r1'
TARGETS = (2000, 8000, 30000, 60000, 120000, 200000, 240000)


def load(root=DATA):
    root = Path(root)
    records = [json.loads(line) for line in (root / 'prefill.jsonl').read_text().splitlines()]
    summary = json.loads((root / 'analysis.json').read_text())
    receipt = json.loads((root / 'validation.json').read_text())
    provenance = json.loads((root / 'provenance.json').read_text())
    served = json.loads((DATA.parent / '2026-11-16-headline-r1/served-config.json').read_text())
    for key in ('image_id', 'model', 'pool'):
        assert provenance[key] == served[key], f'cold-prefill served {key} differs'
    assert receipt['unit_status'] == 'VALIDATED' and receipt['exit_code'] == 0
    assert receipt['restored_model'] == served['model']
    assert receipt['packet_check']['verified']
    assert [r['tag'] for r in records] == ['cal'] + [f'pf-{t}' for t in TARGETS for _ in range(3)]
    assert summary['n_ok'] == len(records) == 22
    for r in records:
        assert r['ok'] and r['finish_reason'] == 'length'
        assert r['conc'] == 1 and r['i'] == r['run'] == 0
        assert r['min_tokens'] == r['max_tokens'] == r['completion_tokens'] == r['server_completion_tokens'] == 64
        assert 0 < r['client_frames'] <= 64 and r['temperature'] == 0
        assert r['prompt_tokens'] > 0 and r['ttft_s'] > 0
    cells = indexed(summary['cells'], ('target',), {(t,) for t in TARGETS})
    with (root / 'prefill.tsv').open() as f:
        table = list(csv.DictReader(f, delimiter='\t'))
    assert [int(r['target']) for r in table] == list(TARGETS)
    calibration = records[0]['prompt_tokens']
    for target, tab in zip(TARGETS, table):
        row = cells[target,]
        reqs = [r for r in records if r['tag'] == f'pf-{target}']
        assert len(reqs) == row['n'] == 3
        assert all(abs(r['prompt_tokens'] / target - 1) <= .03 for r in reqs)
        assert row['budget'] == round((target - 86) * 60000 / (calibration - 86))
        assert all(r['ctx_requested'] == row['budget'] for r in reqs)
        close(st.mean(r['prompt_tokens'] for r in reqs), row['prompt_tokens'])
        close(st.mean(r['prompt_tokens'] / r['ttft_s'] for r in reqs), row['tps'])
        close(st.mean(r['ttft_s'] for r in reqs), row['ttft_s'])
        for key, value in row.items():
            close(float(tab[key]), value)
    return summary['cells']


if __name__ == '__main__':
    for row in load():
        print(f"{row['prompt_tokens']:.0f} prompt tokens: {row['tps']:.0f} tok/s, {row['ttft_s']:.3f} s TTFT")
